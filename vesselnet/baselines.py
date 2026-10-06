"""Iteration 0 baselines (PLAN.md §5) on windows of synthetic scenes.

    python vesselnet/baselines.py oracle    --split test --scenes 10 --out runs/baselines
    python vesselnet/baselines.py vesselmap --split test --scenes 10 --out runs/baselines
    python vesselnet/baselines.py detector  --split val  --scenes 5 --grid --out runs/detector_tune
    python vesselnet/baselines.py detector  --split test --scenes 10 --set z_junction=3 --set width=61

Each scene contributes --windows windows of --size px (the ones with the most visible crossings,
energy_check.windows), cut from the average still with the truth in window coordinates.

oracle     the truth with centrelines held fixed and profiles / halo / background fitted (optimize.oracle):
           E_truth, tau and the price per px every other result of the window is scored with; its own
           architecture metrics (an upper bound: the truth clipped to the window).
vesselmap  build_map -> refine_map -> consolidate_map -> search_map with default settings: E (at the
           oracle's scale) and dE to the oracle after each stage, the architecture metrics of the
           consolidated and searched maps, as they are and split at their branch nodes (to_segments: the
           strict convention ends a vessel at every fork), and the time of each stage.
detector   vesselmap.intersections.detect(trace=True) on the window: junction recall / precision against the
           observable junctions, and per-event recall (within 4 px + radius) at both granularities.  --set
           key=value overrides detect's arguments, plus width (darkness width, px); --grid tries a grid.

Rows are appended to <out>/<method>.jsonl with the seed, window, settings and git hashes; finished (seed,
window, settings) rows are skipped, so runs resume.  The oracle of a window is saved and reused by the others.
"""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from vesselmap.image import prepare  # noqa: E402
from vesselmap.network import VesselNetwork  # noqa: E402

from vesselnet import data as D  # noqa: E402
from vesselnet import metrics as M  # noqa: E402
from vesselnet import optimize as O  # noqa: E402
from vesselnet.energy_check import shift, windows  # noqa: E402


def _done(path, key_fn):
    if not os.path.exists(path):
        return set()
    return {key_fn(json.loads(line)) for line in open(path, encoding="utf-8")}


def _append(path, row):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(D._jsonable(row)) + "\n")


def _windows_of(a):
    lo = D.SPLITS[a.split][0]
    for seed in range(lo, lo + a.scenes):
        folder = D.scene_dir(a.data, seed)
        if not D.is_done(folder):
            print(f"{folder} is not generated yet: skipped", flush=True)
            continue
        S = D.load_compact(folder)
        H, W = S["img"].shape
        for y0, x0 in windows(S["junctions_average"], H, W, a.size, a.windows):
            yield seed, S, y0, x0


def _setup(S, y0, x0, size):
    I = O.to_unit(S["img"][y0:y0 + size, x0:x0 + size])
    P = prepare(I)
    truth = shift(VesselNetwork.load(os.path.join(S["folder"], "map.json")), x0, y0, (size, size))
    obs, js = M.window_truth(S["observable_average"], S["junctions_average"], y0, x0, size)
    return I, P, truth, obs, js


def get_oracle(a, seed, S, y0, x0, P, truth):
    """The window's oracle, fitted once and kept in <out>/oracle/."""
    d = os.path.join(a.out, "oracle")
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"s{seed}_{y0}_{x0}_{a.size}")
    if os.path.exists(stem + ".json"):
        info = json.load(open(stem + ".json", encoding="utf-8"))
        return VesselNetwork.load(stem + "_map.json"), info
    net, info = O.oracle(truth, P, iters=a.oracle_iters)
    info = D._jsonable({k: v for k, v in info.items() if k != "texture_null"} |
                       dict(texture_null={k: v for k, v in info["texture_null"].items()}))
    net.save(stem + "_map.json")
    json.dump(info, open(stem + ".json", "w", encoding="utf-8"), indent=1)
    return net, info


def cmd_oracle(a):
    path = os.path.join(a.out, "oracle.jsonl")
    done = _done(path, lambda r: (r["seed"], tuple(r["window"])))
    for seed, S, y0, x0 in _windows_of(a):
        if (seed, (y0, x0, a.size)) in done:
            continue
        I, P, truth, obs, js = _setup(S, y0, x0, a.size)
        t0 = time.perf_counter()
        net, info = get_oracle(a, seed, S, y0, x0, P, truth)
        arch = M.architecture(net, obs, js, truth)
        row = dict(seed=seed, preset=S["scene"]["vesselnet"]["preset"], window=[y0, x0, a.size], E=info["E"],
                   tau=info["tau"], price=info["price"], fit=info["fit"], arch=arch,
                   seconds=round(time.perf_counter() - t0, 1), provenance=a.prov)
        _append(path, row)
        print(f"oracle s{seed} ({y0},{x0}): E {info['E']['total']:.0f}, tau {info['tau']:.1f}, "
              f"price {info['price']:.0f}, {info['fit']['seconds']:.0f} s; recall "
              f"{arch['centreline_recall']:.3f} precision {arch['centreline_precision']:.3f}", flush=True)


def cmd_vesselmap(a):
    from vesselmap.consolidate import consolidate_map
    from vesselmap.fit import MapConfig, build_map
    from vesselmap.refine import refine_map
    from vesselmap.search import SearchConfig, search_map
    path = os.path.join(a.out, "vesselmap.jsonl")
    done = _done(path, lambda r: (r["seed"], tuple(r["window"])))
    for seed, S, y0, x0 in _windows_of(a):
        if (seed, (y0, x0, a.size)) in done:
            continue
        I, P, truth, obs, js = _setup(S, y0, x0, a.size)
        _, oinfo = get_oracle(a, seed, S, y0, x0, P, truth)
        tau, price, E0 = oinfo["tau"], oinfo["price"], oinfo["E"]["total"]
        mdir = os.path.join(a.out, "vesselmap_maps")
        os.makedirs(mdir, exist_ok=True)
        stem = os.path.join(mdir, f"s{seed}_{y0}_{x0}_{a.size}")
        stages, net, T = {}, None, {}
        steps = (("map", lambda n: build_map(I, MapConfig(verbose=False), prepared=P)),
                 ("refine", lambda n: refine_map(I, n, MapConfig(verbose=False), prepared=P)),
                 ("consolidate", lambda n: consolidate_map(I, n, MapConfig(verbose=False), prepared=P)),
                 ("search", lambda n: search_map(I, n, SearchConfig(verbose=False), prepared=P)))
        for name, step in steps:
            t0 = time.perf_counter()
            net = step(net)
            T[name] = round(time.perf_counter() - t0, 1)
            net.save(f"{stem}_{name}.json")
            E = O.energy(net, P, tau, price)
            stages[name] = dict(E=E["total"], dE=E["total"] - E0, nll=E["nll"], dnll=E["nll"] - oinfo["E"]["nll"],
                                n_vessels=E["n_vessels"], seconds=T[name])
            print(f"vesselmap s{seed} ({y0},{x0}) {name}: {T[name]:.0f} s, {E['n_vessels']} vessels, "
                  f"dE {stages[name]['dE']:+.0f}", flush=True)
        arch = {}
        for name in ("consolidate", "search"):
            n = VesselNetwork.load(f"{stem}_{name}.json")
            arch[name] = M.architecture(n, obs, js, truth)
            arch[name + "_segments"] = M.architecture(n.to_segments(), obs, js, truth)
        _append(path, dict(seed=seed, preset=S["scene"]["vesselnet"]["preset"], window=[y0, x0, a.size],
                           oracle_E=E0, tau=tau, price=price, stages=stages, arch=arch, seconds=T,
                           provenance=a.prov))


DETECT_KEYS = ("z_junction", "z_arm", "min_ratio", "reach", "pair_deg", "pair_len", "min_traced", "width")


def detect_window(P, params):
    from vesselmap import intersections as X
    p = dict(params)
    width = p.pop("width", 31)
    orig = X.darkness
    X.darkness = functools.partial(orig, width=int(width))
    try:
        t0 = time.perf_counter()
        dets = X.detect(P.logI, sigma=None, valid=P.valid, trace=True, **p)
        return dets, time.perf_counter() - t0
    finally:
        X.darkness = orig


def event_recall(dets, js, tol0=4.0):
    """Per-event recall (the zoo's rule: within 4 px + radius) of the visible junctions' events, and of their
    clustered centres."""
    pts = np.asarray([d["xy"] for d in dets]).reshape(-1, 2)
    hit = lambda p, tol: len(pts) > 0 and float(np.min(np.linalg.norm(pts - p, axis=1))) <= tol
    ev = M._events(js)
    n_ev = sum(hit(p, t) for _, p, t in ev)
    return dict(events=len(ev), events_found=int(n_ev), event_recall=(n_ev / len(ev) if ev else float("nan")))


def cmd_detector(a):
    grid = [dict()]
    if a.grid:
        grid = [dict(z_junction=zj, z_arm=za, width=w) for zj, za, w in
                itertools.product((2.0, 3.0, 4.0), (2.0, 3.0, 4.0), (31, 61))]
        # the thresholds alone change nothing (5 % recall throughout): the arm tests are tried too
        grid += [dict(z_junction=zj, z_arm=za, width=w, texture=False, min_traced=mt) for zj, za, w, mt in
                 itertools.product((2.0, 4.0), (2.0, 4.0), (31, 61), (0, 3))]

    def value(k, v):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return int(v) if k in ("width", "min_traced") else float(v)
    base = {k: value(k, v) for k, v in (s.split("=") for s in a.set)}
    path = os.path.join(a.out, "detector.jsonl")
    done = _done(path, lambda r: (r["seed"], tuple(r["window"]), json.dumps(r["params"], sort_keys=True)))
    for seed, S, y0, x0 in _windows_of(a):
        I = P = js = obs = None
        for g in grid:
            params = {**base, **g}
            if (seed, (y0, x0, a.size), json.dumps(params, sort_keys=True)) in done:
                continue
            if P is None:
                I, P, truth, obs, js = _setup(S, y0, x0, a.size)
            dets, secs = detect_window(P, params)
            from vesselscene.truth import score_tracing
            st = score_tracing(obs, [], [tuple(d["xy"]) for d in dets])
            row = dict(seed=seed, window=[y0, x0, a.size], params=params, n_dets=len(dets),
                       n_three_arms=sum(len(d["arms"]) >= 3 for d in dets), seconds=round(secs, 1),
                       **{k: v for k, v in st.items() if k.startswith(("junction", "n_"))},
                       **event_recall(dets, js), provenance=a.prov)
            _append(path, row)
            print(f"detector s{seed} ({y0},{x0}) {params}: {len(dets)} dets, recall "
                  f"{st['junction_recall']:.3f} precision {st['junction_precision']:.3f}, {secs:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("method", choices=("oracle", "vesselmap", "detector"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test", choices=tuple(D.SPLITS))
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--oracle-iters", type=int, default=300)
    ap.add_argument("--set", action="append", default=[], help="detector: key=value (detect's arguments, width)")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="runs/baselines")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    a.prov = D.provenance()
    dict(oracle=cmd_oracle, vesselmap=cmd_vesselmap, detector=cmd_detector)[a.method](a)


if __name__ == "__main__":
    main()
