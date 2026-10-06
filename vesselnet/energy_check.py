"""The energy check (PLAN.md §1): on synthetic scenes, is the true graph the minimum of E?

    python vesselnet/energy_check.py --data data --scenes 5 --out runs/energy_check

For each validation scene, windows of --size px (those with the most visible crossings) are cut from the
average still with the truth (map.json, shifted and clipped to the window).  In each window:

1. the oracle: the truth with centrelines held fixed, profiles / halo / background fitted (optimize.oracle);
   it fixes tau and the price per px for every energy of the window;
2. every perturbation and the truth are given the same local fit from the oracle's state: --iters steps on
   the parameters within R_EVENT px of the event (or R_VESSEL px of a deleted / added vessel), halo,
   background and the rest fixed, as the search fits a move (a global refit drifts elsewhere by far more
   than the change being scored); dE = E(perturbed) - E(truth).  dE > 0 means E prefers the truth.

Perturbations (vesselscene's strict convention: a vessel ends at every fork):
    swap_pairing      at a crossing, each vessel continues into the other's far arm
    crossing_to_node  both vessels of a crossing cut there (four vessels ending at one point)
    split             a vessel cut in two at its middle
    fork_to_through   at a fork, the parent joined to its straighter daughter (one vessel instead of two)
    texture_vessel    a vessel added along the darkest unexplained ridge of the residual, away from vessels
    delete_observable the faintest observable vessel deleted
    delete_hidden     a vessel no annotator could see deleted: in no observable run, not merged into one, and
                      under 5 % of its samples on the observable lumen or don't-care rasters (E should fall:
                      it explains almost nothing and pays its existence cost; not an error of E)

Results: <out>/energy_check.jsonl (one row per perturbation, with seeds, windows and git hashes) and a
summary table on stdout.  Resumable: windows already in the file are skipped.
"""
from __future__ import annotations

import argparse
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
from vesselnet import optimize as O  # noqa: E402

EXPECT_FALL = ("delete_hidden",)


# ----------------------------------------------------------------- windows
def shift(net: VesselNetwork, x0: float, y0: float, shape) -> VesselNetwork:
    out = net.copy()
    out.shape = tuple(int(v) for v in shape)
    for n in out.nodes.values():
        n.x, n.y = n.x - x0, n.y - y0
    for e in out.edges.values():
        e.ctrl = e.ctrl - np.array([x0, y0])
    out._touch()
    return O.clip_to_frame(out)


def windows(junctions, H, W, size, n, margin=48):
    """The n non-overlapping size x size windows (on a grid) with the most visible crossing events."""
    pts = np.asarray([m["xy"] for j in junctions if j.get("type_visible", "none") != "none"
                      for m in j["members"] if m["type"] == "crossing"]).reshape(-1, 2)
    cand = []
    for y0 in range(0, H - size + 1, size // 2):
        for x0 in range(0, W - size + 1, size // 2):
            inside = ((pts[:, 0] > x0 + margin) & (pts[:, 0] < x0 + size - margin) &
                      (pts[:, 1] > y0 + margin) & (pts[:, 1] < y0 + size - margin)).sum()
            cand.append((int(inside), y0, x0))
    out = []
    for c, y0, x0 in sorted(cand, reverse=True):
        if all(abs(y0 - a) >= size or abs(x0 - b) >= size for _, a, b in out):
            out.append((c, y0, x0))
        if len(out) == n:
            break
    return [(y0, x0) for _, y0, x0 in out]


# ----------------------------------------------------------------- perturbations
def _by_vessel(net):
    d = {}
    for k, e in net.edges.items():
        d.setdefault(e.info.get("true_vessel"), []).append(k)
    return d


def _nearest(net, k, p):
    s = net.sample(k, 0.5)
    i = int(np.argmin(np.linalg.norm(s["xy"] - p, axis=1)))
    return s, i, float(np.linalg.norm(s["xy"][i] - p))


def _cut(s, i):
    a = {k: s[k][:i + 1] for k in ("xy", "r", "s", "a")}
    b = {k: s[k][i:] for k in ("xy", "r", "s", "a")}
    return a, b


def _cat(p, q):
    return {k: np.concatenate([p[k], q[k][1:]]) for k in ("xy", "r", "s", "a")}


def _add(net, piece, info):
    if len(piece["xy"]) < 6:
        return None
    return net.add_edge_dense(piece["xy"], piece["r"], piece["s"], piece["a"], info=dict(info), faithful=True)


def _crossings(net, junctions, size, margin=40):
    byv = _by_vessel(net)
    for j in junctions:
        if j.get("type_visible", "none") == "none":
            continue
        for m in j["members"]:
            if m["type"] != "crossing" or len(m["vessels"]) != 2:
                continue
            p = np.asarray(m["xy"], float)
            if not (margin < p[0] < size - margin and margin < p[1] < size - margin):
                continue
            ks = []
            for v in m["vessels"]:
                best = min(((_nearest(net, k, p)[2], k) for k in byv.get(v, [])), default=(1e9, None))
                ks.append(best[1] if best[0] < 3.0 else None)
            if None not in ks and ks[0] != ks[1]:
                yield p, ks


def p_swap_pairing(net, p, ks):
    out = net.copy()
    (sa, ia, _), (sb, ib, _) = _nearest(out, ks[0], p), _nearest(out, ks[1], p)
    a1, a2 = _cut(sa, ia)
    b1, b2 = _cut(sb, ib)
    if min(len(x["xy"]) for x in (a1, a2, b1, b2)) < 6:
        return None
    ia_, ib_ = out.edges[ks[0]].info, out.edges[ks[1]].info
    for k in ks:
        out.remove_edge(k)
    _add(out, _cat(a1, b2), ia_)
    _add(out, _cat(b1, a2), ib_)
    return out


def p_crossing_to_node(net, p, ks):
    out = net.copy()
    for k in ks:
        s, i, _ = _nearest(out, k, p)
        info = out.edges[k].info
        a, b = _cut(s, i)
        if min(len(a["xy"]), len(b["xy"])) < 6:
            return None
        out.remove_edge(k)
        _add(out, a, info)
        _add(out, b, info)
    return out


def p_split(net, k):
    out = net.copy()
    s = out.sample(k, 0.5)
    a, b = _cut(s, len(s["xy"]) // 2)
    info = out.edges[k].info
    out.remove_edge(k)
    _add(out, a, info)
    _add(out, b, info)
    return out


def _forks(net, junctions, size, margin=40):
    """(p, parent edge, child edges) at visible forks / confluences inside the window."""
    byv = _by_vessel(net)
    for j in junctions:
        if j.get("type_visible", "none") == "none":
            continue
        for m in j["members"]:
            if m["type"] not in ("bifurcation", "confluence") or "parents" not in m:
                continue
            p = np.asarray(m["xy"], float)
            if not (margin < p[0] < size - margin and margin < p[1] < size - margin):
                continue
            trunk = m["parents"] if m["type"] == "bifurcation" else m["children"]
            branches = m["children"] if m["type"] == "bifurcation" else m["parents"]
            if len(trunk) != 1 or len(branches) != 2:
                continue
            get = lambda v: next((k for k in byv.get(v, []) if _nearest(net, k, p)[2] < 4.0), None)
            t, bs = get(trunk[0]), [get(v) for v in branches]
            if t is not None and None not in bs:
                yield p, t, bs


def _end_dir(s, at_start, look=12):
    xy = s["xy"]
    n = min(len(xy) - 1, int(look / 0.5))
    v = xy[n] - xy[0] if at_start else xy[-1 - n] - xy[-1]
    return v / (np.linalg.norm(v) + 1e-9)


def p_fork_to_through(net, p, t, bs):
    """Join the trunk to the branch it continues most straightly (one vessel instead of two)."""
    out = net.copy()
    st = out.sample(t, 0.5)
    t_end_at_start = np.linalg.norm(st["xy"][0] - p) < np.linalg.norm(st["xy"][-1] - p)
    dt = _end_dir(st, t_end_at_start)                         # pointing away from the fork, along the trunk
    best = None
    for b in bs:
        sb = out.sample(b, 0.5)
        b_at_start = np.linalg.norm(sb["xy"][0] - p) < np.linalg.norm(sb["xy"][-1] - p)
        db = _end_dir(sb, b_at_start)
        turn = math.degrees(math.acos(np.clip(-np.dot(dt, db), -1, 1)))
        if best is None or turn < best[0]:
            best = (turn, b, sb, b_at_start)
    turn, b, sb, b_at_start = best
    flip = lambda s: {k: s[k][::-1] for k in ("xy", "r", "s", "a")}
    T = flip(st) if t_end_at_start else {k: st[k] for k in ("xy", "r", "s", "a")}    # ends at the fork
    B = {k: sb[k] for k in ("xy", "r", "s", "a")} if b_at_start else flip(sb)           # starts at it
    info = dict(out.edges[t].info, joined_with=out.edges[b].info.get("true_vessel"))
    out.remove_edge(t)
    out.remove_edge(b)
    _add(out, _cat(T, B), info)
    return out, turn


def p_texture_vessel(net, P, S, min_len=30.0):
    """A vessel along the strongest dark ridge of the residual (texture the truth does not explain), at least
    8 px from every truth vessel."""
    import cv2
    from vesselmap.fit import band_scales, seeds_to_edges
    from vesselmap.ridges import detect
    R = P.logI - (S.B - S.optical_density())
    foot = np.zeros(P.shape, np.uint8)
    for k in net.edges:
        s = net.sample(k, 1.0)
        w = int(math.ceil(float(np.median(s["r"] + 2 * s["s"])) + 8))
        cv2.polylines(foot, [np.round(s["xy"] * 4).astype(np.int32)], False, 1, thickness=2 * w + 1, shift=2)
    best = None
    for band in ((1.2, 2.5), (2.5, 5.0)):
        for sd in detect(-R, P.sigma, band_scales(band, 3), min_len=min_len, valid=P.valid,
                         exclude=foot.astype(bool), z_hi=3.0, z_lo=1.5):
            L = float(np.linalg.norm(np.diff(sd.xy, axis=0), axis=1).sum())
            score = float(np.median(sd.z)) * L
            if L >= min_len and (best is None or score > best[0]):
                best = (score, sd, band, L)
    if best is None:
        return None, None
    out = net.copy()
    seeds_to_edges(out, [best[1]], best[2])
    k = max(out.edges)
    out.edges[k].info["true_vessel"] = -1
    return out, dict(length=round(best[3], 1), z=round(float(np.median(best[1].z)), 2),
                     xy=[round(float(v), 1) for v in best[1].xy[len(best[1].xy) // 2]])


def observable_vids(obs, x0, y0, size):
    vs = {}
    for r in obs["runs"]:
        xy = np.asarray(r["xy"], float).reshape(-1, 2) - [x0, y0]
        ins = (xy[:, 0] >= 0) & (xy[:, 0] < size) & (xy[:, 1] >= 0) & (xy[:, 1] < size)
        vs[r["vid"]] = vs.get(r["vid"], 0) + int(ins.sum())
    return {v: n for v, n in vs.items() if n > 0}


# ----------------------------------------------------------------- driver
def check_window(img, truth, obs, junctions, seen_px, y0, x0, size, iters, per_type, log, cfg=None):
    t0 = time.perf_counter()
    I = O.to_unit(img[y0:y0 + size, x0:x0 + size])
    P = prepare(I)
    tnet = shift(truth, x0, y0, (size, size))
    jw = []
    for j in junctions:                                     # junctions in window coordinates
        j2 = dict(j, members=[dict(m, xy=[m["xy"][0] - x0, m["xy"][1] - y0]) for m in j["members"]])
        jw.append(j2)
    onet, info = O.oracle(tnet, P, iters=300, cfg=cfg)
    tau, price = info["tau"], info["price"]
    log(f"  window ({y0}, {x0}): {len(onet.edges)} truth edges, oracle {info['fit']['seconds']:.0f} s, "
        f"tau {tau:.2f}, price {price:.1f}/px, E {info['E']['total']:.0f}")

    # the fits do not see Phi or the price, so every energy variant is scored on the same fitted networks
    from vesselmap.search import SearchConfig
    null = info.get("texture_null", {})
    variants = {"vesselmap": (False, price), "strict": (True, price)}
    if null.get("median_per_px"):
        variants["strict_median"] = (True, float(null["median_per_px"]))
    variants["strict_floor"] = (True, float(tau))                 # tau * lam_length: no texture null

    def score(n):
        return {v: O.energy(n, P, tau, pr, SearchConfig(strict_forks=st)) for v, (st, pr) in variants.items()}

    S = O._search(onet, P, tau, price, cfg)

    def local(net, pts, radius):
        """net fitted only near pts (the truth gets the same fit at the same focus, so both relax alike)."""
        n = net.copy()
        f = O.fit_local(n, P, pts, radius, iters, S)
        return n, f["seconds"]

    E_oracle = score(onet)
    rows = [dict(kind="truth", E={v: e["total"] for v, e in E_oracle.items()}, nll=E_oracle["vesselmap"]["nll"],
                 n_vessels=E_oracle["vesselmap"]["n_vessels"], prices={v: pr for v, (_, pr) in variants.items()})]
    cases = []
    cx = list(_crossings(onet, jw, size))
    for p, ks in cx[:per_type]:
        cases.append(("swap_pairing", dict(xy=p.round(1).tolist()), lambda p=p, ks=ks: p_swap_pairing(onet, p, ks),
                      [p], R_EVENT))
    for p, ks in cx[per_type:2 * per_type] or cx[:per_type]:
        cases.append(("crossing_to_node", dict(xy=p.round(1).tolist()),
                      lambda p=p, ks=ks: p_crossing_to_node(onet, p, ks), [p], R_EVENT))
    for p, t, bs in list(_forks(onet, jw, size))[:per_type]:
        def f(p=p, t=t, bs=bs):
            n, turn = p_fork_to_through(onet, p, t, bs)
            n.meta["turn_deg"] = round(turn, 1)
            return n
        cases.append(("fork_to_through", dict(xy=p.round(1).tolist()), f, [p], R_EVENT))
    seen = observable_vids(obs, x0, y0, size)
    byv = _by_vessel(onet)
    long_obs = [(float(np.median(onet.edges[k].a)), k) for v, ks in byv.items() if v in seen and seen[v] >= 40
                for k in ks if onet.length(k) >= 40]
    for _, k in sorted(long_obs)[:per_type]:
        cases.append(("delete_observable", dict(eid=k, a=round(float(np.median(onet.edges[k].a)), 4)),
                      lambda k=k: _without(onet, k), _along(onet, k), R_VESSEL))
    mids = sorted(long_obs, key=lambda t: -onet.length(t[1]))[:per_type]
    for _, k in mids:
        mid = onet.sample(k, 0.5)["xy"]
        cases.append(("split", dict(eid=k, L=round(onet.length(k), 1)), lambda k=k: p_split(onet, k),
                      [mid[len(mid) // 2]], R_EVENT))
    merged = {m["vid"] for m in obs.get("merged", [])}
    seen_mask = seen_px[y0:y0 + size, x0:x0 + size]

    def on_seen(k):
        xy = np.round(onet.sample(k, 1.0)["xy"]).astype(int).clip(0, size - 1)
        return float(seen_mask[xy[:, 1], xy[:, 0]].mean())
    hidden = [k for v, ks in byv.items() if v not in seen and v not in merged for k in ks
              if onet.length(k) >= 20 and on_seen(k) < 0.05]
    for k in hidden[:per_type]:
        cases.append(("delete_hidden", dict(eid=k, L=round(onet.length(k), 1)), lambda k=k: _without(onet, k),
                      _along(onet, k), R_VESSEL))
    tex, tinfo = p_texture_vessel(onet, P, S)
    if tex is not None:
        cases.append(("texture_vessel", tinfo, lambda: tex, _along(tex, max(tex.edges)), R_VESSEL))
    for kind, where, make, pts, radius in cases:
        try:
            n = make()
        except Exception as e:                                   # noqa: BLE001
            rows.append(dict(kind=kind, where=where, error=repr(e)))
            continue
        if n is None:
            continue
        tf, t0f = local(onet, pts, radius)
        nf, t = local(n, pts, radius)
        E, E0 = score(nf), score(tf)
        b, b0 = E["vesselmap"], E0["vesselmap"]
        row = dict(kind=kind, where=where, dnll=b["nll"] - b0["nll"], dprior=b["prior"] - b0["prior"],
                   n_vessels=b["n_vessels"], fit_seconds=t + t0f, extra=n.meta.get("turn_deg"),
                   dE={v: E[v]["total"] - E0[v]["total"] for v in variants},
                   dcost={v: E[v]["cost"] - E0[v]["cost"] for v in variants},
                   dphi={v: E[v]["frag"] - E0[v]["frag"] for v in variants})
        rows.append(row)
        log(f"    {kind:18s} dNLL {row['dnll']:+10.0f}  " +
            "  ".join(f"{v} {row['dE'][v]:+.0f}" for v in variants))
    return dict(window=[y0, x0, size], tau=tau, price=price, oracle=info, rows=rows,
                seconds=round(time.perf_counter() - t0, 1))


R_EVENT = 40.0        # px around an event (crossing, fork, cut) that the local fits may move
R_VESSEL = 20.0       # px around a deleted or added vessel


def _along(net, k, step=4.0):
    """Points every `step` px along edge k (the focus of a deleted or added vessel)."""
    xy = net.sample(k, 1.0)["xy"]
    return xy[::int(step)]


def _without(net, k):
    out = net.copy()
    out.remove_edge(k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--strict-forks", action="store_true", help="Phi does not charge three ends at one point")
    ap.add_argument("--out", default="runs/energy_check")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "energy_check.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["seed"], tuple(r["window"])))
    prov = D.provenance()
    from vesselmap.search import SearchConfig
    cfg = SearchConfig(strict_forks=a.strict_forks)
    log = lambda *m: print(*m, flush=True)
    lo = D.SPLITS["val"][0]
    for folder in [D.scene_dir(a.data, s) for s in range(lo, lo + a.scenes)]:
        if not D.is_done(folder):
            log(f"{folder} is not generated yet: skipped")
            continue
        S = D.load_compact(folder)
        seed = S["scene"]["vesselnet"]["seed"]
        truth = VesselNetwork.load(os.path.join(folder, "map.json"))
        H, W = S["img"].shape
        log(f"seed {seed} ({S['scene']['vesselnet']['preset']})")
        for y0, x0 in windows(S["junctions_average"], H, W, a.size, a.windows):
            if (seed, (y0, x0, a.size)) in done:
                continue
            seen_px = S["observable_lumen_average"] | S["observable_dont_care_average"]
            r = check_window(S["img"], truth, S["observable_average"], S["junctions_average"], seen_px, y0, x0, a.size,
                             a.iters, a.per_type, log, cfg)
            r.update(seed=seed, preset=S["scene"]["vesselnet"]["preset"], iters=a.iters, provenance=prov,
                     strict_forks=a.strict_forks)
            r["oracle"] = {k: v for k, v in r["oracle"].items() if k != "texture_null"}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(D._jsonable(r)) + "\n")
    summarize(path)


def summarize(path):
    """Per perturbation and energy variant: how often E prefers the truth (dE > 0; for delete_hidden, dE < 0),
    and the median dE."""
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows += [dict(x, seed=r["seed"]) for x in r["rows"] if x["kind"] != "truth" and isinstance(x.get("dE"), dict)]
    if not rows:
        return
    variants = list(rows[0]["dE"])
    kinds = sorted({x["kind"] for x in rows})
    print("\nE prefers the truth (count / n, median dE):\n")
    print("| perturbation | n | median dNLL | " + " | ".join(variants) + " |")
    print("|---|---|---|" + "---|" * len(variants))
    for k in kinds:
        xs = [x for x in rows if x["kind"] == k]
        cells = []
        for v in variants:
            d = np.array([x["dE"][v] for x in xs])
            ok = (d < 0) if k in EXPECT_FALL else (d > 0)
            cells.append(f"{int(ok.sum())}/{len(d)} ({np.median(d):+.3g})")
        print(f"| {k}{' (should fall)' if k in EXPECT_FALL else ''} | {len(xs)} | "
              f"{np.median([x['dnll'] for x in xs]):+.3g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
