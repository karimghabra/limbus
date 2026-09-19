"""Vessel network of a burst's averaged stabilized frame.

usage:
  python -m vessels <burst> [--method nonrigid] [--out DIR] [--res-base DIR]
                    [--no-join] [--no-fit] [--t-hi 3] [--t-lo 1.5] ...

Writes into <out> (default <res-base>/<method>/<burst>/vessels/):
  vessel_labels.tif  int32, each vessel's lumen carrying its id
  vessels.json       per vessel: id, length, radius, peak absorbance, the
                     centreline points, and every setting used
  overlay.png        the averaged frame with each vessel drawn and numbered
"""
import argparse
import json
import os
import sys
from dataclasses import asdict

import cv2
import numpy as np
import tifffile

if __package__ in (None, ""):                       # allow running the file directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "vessels"

from . import image as img          # noqa: E402
from . import network as net        # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(prog="vessels", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("burst", help="burst folder name, or a path to a TIFF of an averaged frame")
    p.add_argument("--method", default="nonrigid", help="stabilization method folder (default: nonrigid)")
    p.add_argument("--res-base", default=os.environ.get("RES_BASE"),
                   help="stabilization results root (default: $RES_BASE or <repo>/stabilization)")
    p.add_argument("--out", default=None, help="output folder (default: <result folder>/vessels)")
    d = net.NetConfig()
    p.add_argument("--t-hi", type=float, default=d.t_hi, help="strong ridge threshold, robust z")
    p.add_argument("--L-hi", type=int, default=d.L_hi, help="strong ridge minimum length, px")
    p.add_argument("--t-lo", type=float, default=d.t_lo, help="faint ridge threshold (kept when connected)")
    p.add_argument("--L-lo", type=int, default=d.L_lo)
    p.add_argument("--min-len", type=float, default=d.min_len)
    p.add_argument("--join-gap", type=float, default=d.join_gap)
    p.add_argument("--min-depth", type=float, default=d.min_depth, help="shallowest vessel accepted")
    p.add_argument("--psf", type=float, default=d.psf, help="optical blur sigma, px")
    p.add_argument("--no-join", action="store_true", help="do not join ridge pieces into one vessel")
    p.add_argument("--no-fit", action="store_true", help="skip the model fit (no radii, no echo trimming)")
    p.add_argument("--no-fixed-pattern", action="store_true",
                   help="keep the sensor's row and column offsets in the image")
    p.add_argument("--border-px", type=int, default=25,
                   help="margin kept away from the edge of the covered region")
    p.add_argument("--move-fit", type=float, default=d.fit_move,
                   help="let the fit move the centreline this far (px); 0 keeps the ridge")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = a.res_base or os.path.join(os.path.dirname(repo), "stabilization")
    if a.burst.lower().endswith((".tif", ".tiff")):
        src, out = a.burst, a.out or os.path.join(os.path.dirname(a.burst), "vessels")
    else:
        src = os.path.join(base, a.method, a.burst, "mean_stabilized.tif")
        out = a.out or os.path.join(base, a.method, a.burst, "vessels")
    if not os.path.exists(src):
        raise SystemExit(f"no averaged frame at {src} - stabilize the burst first")
    cfg = net.NetConfig(t_hi=a.t_hi, L_hi=a.L_hi, t_lo=a.t_lo, L_lo=a.L_lo, min_len=a.min_len,
                        join=not a.no_join, join_gap=a.join_gap, fit=not a.no_fit,
                        fit_move=a.move_fit, min_depth=a.min_depth, psf=a.psf)
    os.makedirs(out, exist_ok=True)
    mean = tifffile.imread(src).astype(np.float32)
    A, valid = img.prepare(mean, fixed_pattern=not a.no_fixed_pattern, border_px=a.border_px)
    print(f"{os.path.basename(src)}  {mean.shape[1]}x{mean.shape[0]}", flush=True)
    ves, _, _ = net.detect(A, valid, cfg, log=lambda m: print(m, flush=True))
    lab = net.labels(ves, A.shape, psf=cfg.psf) if cfg.fit else np.zeros(A.shape, np.int32)
    tifffile.imwrite(os.path.join(out, "vessel_labels.tif"), lab)
    recs = []
    for v in ves:
        pieces = []
        for (P, R, D, rec) in v["pieces"]:
            rr = None if R is None else np.asarray(R, float)
            pieces.append({
                "points": np.asarray(P, float).round(2).tolist(),
                "radius_px": (None if rr is None else rr.round(2).tolist()),
                "peak_absorbance": (None if D is None else round(float(D), 4)),
                "depth": rec.get("depth"), "length_px": rec.get("length"),
                # the fit's radius floor is 0.8 px. A vessel the optics can
                # resolve does not land there - of planted vessels of radius
                # 1-2.5 px only about one in ten does - so a piece sitting on
                # the floor is more likely a sensor line or a texture ridge
                # than a vessel. It is FLAGGED, not dropped: some real thin
                # vessels do land there too.
                "radius_at_limit": (None if rr is None else bool(np.median(rr) <= 0.85))})
        L = float(sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in v["centrelines"] if len(c) > 1))
        rr = [np.median(p["radius_px"]) for p in pieces if p["radius_px"]]
        recs.append({"id": v["id"], "anchor": list(v["anchor"]), "length_px": round(L, 1),
                     "radius_px": (round(float(np.median(rr)), 2) if rr else None),
                     "n_pieces": len(pieces), "pieces": pieces})
    json.dump({"source": src, "method": a.method, "config": asdict(cfg), "n_vessels": len(recs),
               "total_length_px": round(sum(r["length_px"] for r in recs), 1), "vessels": recs},
              open(os.path.join(out, "vessels.json"), "w"), indent=1)
    T = np.exp(-A)
    lo, hi = np.percentile(T[valid], [0.5, 99.8])
    g = np.clip((T - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for v in ves:
        hue = int((v["id"] * 137.508) % 180)
        col = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        for c in v["centrelines"]:
            cv2.polylines(ov, [np.rint(np.asarray(c, float)).astype(np.int32)], False, col, 1, cv2.LINE_AA)
        m = v["centrelines"][0][len(v["centrelines"][0]) // 2]
        for th, cc in ((2, (0, 0, 0)), (1, col)):
            cv2.putText(ov, str(v["id"]), (int(m[0]) + 3, int(m[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, cc, th, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out, "overlay.png"), ov)
    print(f"{len(recs)} vessels, {sum(r['length_px'] for r in recs):.0f} px of centreline -> {out}", flush=True)


if __name__ == "__main__":
    main()
