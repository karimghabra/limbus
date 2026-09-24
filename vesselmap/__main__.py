"""Command line interface.

    python -m vesselmap map IMAGE -o OUT [--set key=value ...]
    python -m vesselmap refine MAP.json IMAGE -o OUT
    python -m vesselmap consolidate MAP.json IMAGE -o OUT
    python -m vesselmap fit-frames MAP.json FRAME [FRAME ...] -o OUT
    python -m vesselmap draw MAP.json [--image IMAGE] -o OUT
    python -m vesselmap synth-eval [--seeds 0 1 2] -o OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import fields

warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")


def _apply_sets(cfg, sets):
    types = {f.name: f.type for f in fields(cfg)}
    for kv in sets or []:
        k, v = kv.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown setting {k}; known: {sorted(types)}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, (int, float, str)):
            val = type(cur)(v)
        else:
            val = json.loads(v)
            if isinstance(cur, tuple):
                val = tuple(tuple(x) if isinstance(x, list) else x for x in val)
        setattr(cfg, k, val)
    return cfg


def write_outputs(net, intensity, prepared, out, stem="map"):
    import cv2
    from .draw import draw_digraph, export_html, model_panels, overlay
    os.makedirs(out, exist_ok=True)
    export_html(net, os.path.join(out, f"{stem}_digraph.html"), intensity)
    net.save(os.path.join(out, f"{stem}.json"))
    G = net.to_digraph()
    export_graphml(G, os.path.join(out, f"{stem}.graphml"))
    cv2.imwrite(os.path.join(out, f"{stem}_overlay.png"), overlay(net, intensity))
    cv2.imwrite(os.path.join(out, f"{stem}_overlay_blur.png"), overlay(net, intensity, color_by="blur"))
    if net.has_through() or "consolidation" in net.meta:
        cv2.imwrite(os.path.join(out, f"{stem}_overlay_vessels.png"),
                    overlay(net, intensity, color_by="vessel", arrows=False))
    draw_digraph(net, os.path.join(out, f"{stem}_digraph.png"))
    draw_digraph(net, os.path.join(out, f"{stem}_digraph_on_image.png"), intensity=intensity)
    if prepared is not None:
        model_panels(net, prepared, os.path.join(out, f"{stem}_model_residual.png"), scale=0.5)
    with open(os.path.join(out, f"{stem}_edges.csv"), "w", newline="") as f:
        w = None
        for u, v, d in G.edges(data=True):
            row = dict(u=u, v=v, **d)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
    with open(os.path.join(out, f"{stem}_summary.json"), "w") as f:
        json.dump(dict(summary=net.summary(), meta=net.meta,
                       crossings=len(G.graph.get("crossings", []))), f, indent=1)


def export_graphml(G, path):
    import networkx as nx
    H = G.copy()
    H.graph = {"image_height": G.graph["image_shape"][0], "image_width": G.graph["image_shape"][1],
               "n_crossings": len(G.graph.get("crossings", []))}
    nx.write_graphml(H, path)


def cmd_map(a):
    from .fit import MapConfig, build_map
    from .image import load_image, prepare
    I = load_image(a.image)
    P = prepare(I)
    cfg = _apply_sets(MapConfig(verbose=not a.quiet), a.set)
    t = time.time()
    net = build_map(I, cfg, prepared=P)
    net.meta["source_image"] = os.path.abspath(a.image)
    write_outputs(net, I, P, a.out)
    print(f"map written to {a.out} in {time.time() - t:.0f}s: {net.summary()}")


def cmd_refine(a):
    from .fit import MapConfig
    from .image import load_image, prepare
    from .network import VesselNetwork
    from .refine import RefineConfig, refine_map
    I = load_image(a.image)
    P = prepare(I)
    net = VesselNetwork.load(a.map)
    if tuple(net.shape) != P.shape:
        raise SystemExit(f"map shape {net.shape} does not match image {P.shape}")
    cfg = _apply_sets(MapConfig(verbose=not a.quiet), a.set)
    rc = _apply_sets(RefineConfig(verbose=not a.quiet), a.refine_set)
    t = time.time()
    L0 = net.summary()["total_length_px"]
    refine_map(I, net, cfg, rc, prepared=P)
    write_outputs(net, I, P, a.out)
    print(f"refined map written to {a.out} in {time.time() - t:.0f}s: "
          f"{L0:.0f} -> {net.summary()['total_length_px']:.0f} px of centreline; {net.summary()}")


def cmd_consolidate(a):
    from .consolidate import ConsolidateConfig, consolidate_map
    from .fit import MapConfig
    from .image import load_image, prepare
    from .network import VesselNetwork
    I = load_image(a.image)
    P = prepare(I)
    net = VesselNetwork.load(a.map)
    if tuple(net.shape) != P.shape:
        raise SystemExit(f"map shape {net.shape} does not match image {P.shape}")
    cfg = _apply_sets(MapConfig(verbose=not a.quiet), a.set)
    cc = _apply_sets(ConsolidateConfig(verbose=not a.quiet), a.consolidate_set)
    t = time.time()
    out = consolidate_map(I, net, cfg, cc, prepared=P)
    out.meta["source_image"] = os.path.abspath(a.image)
    write_outputs(out, I, P, a.out)
    c = out.meta["consolidation"]
    print(f"consolidated map written to {a.out} in {time.time() - t:.0f}s: "
          f"{c['edges_before']} segments -> {c['edges_after']} vessels; {out.summary()}")


def cmd_fit_frames(a):
    from .fit import FrameFitConfig, fit_frame
    from .image import load_image, prepare
    from .network import VesselNetwork
    import cv2
    from .draw import overlay
    ref = VesselNetwork.load(a.map)
    cfg = _apply_sets(FrameFitConfig(), a.set)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    prev = None
    for path in a.frames:
        I = load_image(path)
        net, rep = fit_frame(I, ref, cfg, prepared=prepare(I),
                             init=prev if a.chain else None)
        prev = net
        stem = os.path.splitext(os.path.basename(path))[0]
        net.meta["source_image"] = os.path.abspath(path)
        net.save(os.path.join(a.out, f"{stem}.json"))
        if a.overlays:
            cv2.imwrite(os.path.join(a.out, f"{stem}_overlay.png"), overlay(net, I, crossings=False))
        rows.append(dict(frame=stem, **{k: v for k, v in rep.items() if k != "affine"},
                         affine=json.dumps(rep["affine"])))
        print(f"{stem}: {rep}", flush=True)
    with open(os.path.join(a.out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_draw(a):
    from .image import load_image, prepare
    from .network import VesselNetwork
    net = VesselNetwork.load(a.map)
    I = load_image(a.image) if a.image else None
    if I is None:
        from .draw import draw_digraph
        os.makedirs(a.out, exist_ok=True)
        draw_digraph(net, os.path.join(a.out, "map_digraph.png"))
    else:
        write_outputs(net, I, prepare(I), a.out)


def cmd_synth_eval(a):
    from .fit import MapConfig, build_map
    from .image import prepare
    from .synthetic import centreline_metrics, make_scene
    os.makedirs(a.out, exist_ok=True)
    res = []
    for seed in a.seeds:
        I, vessels, _ = make_scene(seed)
        cfg = _apply_sets(MapConfig(verbose=False), a.set)
        P = prepare(I)
        t = time.time()
        net = build_map(I, cfg, prepared=P)
        m = centreline_metrics(net, vessels, I.shape)
        m.update(seed=seed, seconds=round(time.time() - t, 1), **net.summary())
        res.append(m)
        write_outputs(net, I, P, os.path.join(a.out, f"seed{seed}"))
        print(json.dumps(m), flush=True)
    with open(os.path.join(a.out, "synthetic_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vesselmap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="discover the vessel network of one image")
    m.add_argument("image")
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--set", nargs="*", help="MapConfig overrides key=value")
    m.add_argument("--quiet", action="store_true")
    m.set_defaults(func=cmd_map)
    r = sub.add_parser("refine", help="add fine detail (small and parallel vessels) to a map")
    r.add_argument("map")
    r.add_argument("image", help="the image the map was built from")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("--set", nargs="*", help="MapConfig overrides key=value")
    r.add_argument("--refine-set", nargs="*", help="RefineConfig overrides key=value")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_refine)
    c = sub.add_parser("consolidate", help="merge the segments of each vessel into one spline")
    c.add_argument("map")
    c.add_argument("image", help="the image the map was built from")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--set", nargs="*", help="MapConfig overrides key=value")
    c.add_argument("--consolidate-set", nargs="*", help="ConsolidateConfig overrides key=value")
    c.add_argument("--quiet", action="store_true")
    c.set_defaults(func=cmd_consolidate)
    f = sub.add_parser("fit-frames", help="adjust a map to each of several frames")
    f.add_argument("map")
    f.add_argument("frames", nargs="+")
    f.add_argument("-o", "--out", required=True)
    f.add_argument("--set", nargs="*", help="FrameFitConfig overrides key=value")
    f.add_argument("--overlays", action="store_true")
    f.add_argument("--chain", action="store_true",
                   help="start each frame from the previous frame's fit (prior stays on the map)")
    f.set_defaults(func=cmd_fit_frames)
    d = sub.add_parser("draw", help="draw a saved map")
    d.add_argument("map")
    d.add_argument("--image")
    d.add_argument("-o", "--out", required=True)
    d.set_defaults(func=cmd_draw)
    s = sub.add_parser("synth-eval", help="score the mapper on synthetic scenes")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--set", nargs="*")
    s.set_defaults(func=cmd_synth_eval)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
