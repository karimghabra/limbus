"""Carry the vessel labels into every raw frame of a burst, and check them.

The labels live in the averaged stabilized frame. Stabilization maps
stabilized(x) = raw(x + d(x)), so raw pixel y shows stabilized position
y - d(y), and

    label_raw(y) = label_stab(y - d(y))

exactly for a translation and to first order for the smooth non-rigid fields.

The check is the point of this module: for every frame it measures the
vessel contrast the labels see,

    1 - mean(inside the labels) / mean(a ring just outside them),

twice - with the labels following the motion, and with the same labels held
still. Tracked labels should sit on dark vessels in every frame; static ones
drift off as the eye moves. If the two are close, the labels are not really
tracking and nothing downstream should be believed.

usage:
  python -m vessels.track <burst> [--method nonrigid] [--vessels DIR]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import tifffile

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "vessels"


def labels_in_frame(labels, fx, fy):
    """The stabilized-frame labels in a raw frame's own pixel coordinates."""
    H, W = labels.shape
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(labels, X - fx, Y - fy, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def contrast(img, lab, ids, ring_px=13, min_area=100):
    """Area-weighted vessel contrast of the labelled segments in one frame."""
    any_v = lab > 0
    k = np.ones((ring_px, ring_px), np.uint8)
    vals, weights = [], []
    for sid in ids:
        m = lab == sid
        n = int(m.sum())
        if n < min_area:
            continue
        ring = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~any_v
        if ring.sum() < min_area:
            continue
        vals.append(1 - img[m].mean() / img[ring].mean())
        weights.append(n)
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def check(burst_dir, res_dir, labels, method="nonrigid", min_area=300, log=print):
    from stabilize.bursts import load_burst, read_frame
    from stabilize.fields import FieldSet

    b = load_burst(burst_dir)
    with open(os.path.join(res_dir, "transforms.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fpath = os.path.join(res_dir, "fields.npz")
    fs = FieldSet(fpath) if os.path.exists(fpath) else None
    ids, area = np.unique(labels[labels > 0], return_counts=True)
    ids = ids[area >= min_area]
    tracked, static, frames, disp = [], [], [], []
    for i in range(len(b)):
        if fs is not None:
            if not (fs.has(i) and fs.is_used(i)):
                continue
            fx, fy = fs.field(i)
        else:
            if rows[i].get("registered") != "1":
                continue
            fx = np.full(labels.shape, float(rows[i]["dx_px"]), np.float32)
            fy = np.full(labels.shape, float(rows[i]["dy_px"]), np.float32)
        img = read_frame(b.files[i]).astype(np.float32)
        lab_i = labels_in_frame(labels, fx, fy)
        tracked.append(contrast(img, lab_i, ids))
        static.append(contrast(img, labels, ids))
        frames.append(i)
        disp.append(float(np.hypot(float(rows[i]["dx_px"]), float(rows[i]["dy_px"]))))
    tr, st, dp = np.array(tracked), np.array(static), np.array(disp)
    far = dp > np.percentile(dp, 75) if len(dp) > 4 else np.ones(len(dp), bool)
    out = {"burst": b.name, "method": method, "frames_checked": len(frames),
           "frames_total": len(b), "segments": int(len(ids)),
           "tracked_contrast_median": round(float(np.nanmedian(tr)), 4),
           "tracked_contrast_min": round(float(np.nanmin(tr)), 4),
           "static_contrast_median": round(float(np.nanmedian(st)), 4),
           "frames_below_half_median": int((tr < 0.5 * np.nanmedian(tr)).sum()),
           "most_displaced_quarter": {"tracked": round(float(np.nanmedian(tr[far])), 4),
                                      "static": round(float(np.nanmedian(st[far])), 4)},
           "per_frame": {"frames": frames, "tracked": tr.tolist(), "static": st.tolist(),
                         "displacement_px": dp.tolist()}}
    log(f"{b.name} [{method}]: {len(frames)} of {len(b)} frames, {len(ids)} segments")
    log(f"  contrast inside the labels: tracked {out['tracked_contrast_median']:.3f} "
        f"(min {out['tracked_contrast_min']:.3f}), held still {out['static_contrast_median']:.3f}")
    log(f"  in the 25% most displaced frames: tracked {out['most_displaced_quarter']['tracked']:.3f} "
        f"vs {out['most_displaced_quarter']['static']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vessels.track", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("burst")
    ap.add_argument("--method", default="nonrigid")
    ap.add_argument("--rec-dir", default=os.environ.get("REC_DIR"))
    ap.add_argument("--res-base", default=os.environ.get("RES_BASE"))
    ap.add_argument("--vessels", default=None, help="folder holding vessel_labels.tif")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    rec = a.rec_dir or os.path.join(os.path.dirname(repo), "recordings")
    base = a.res_base or os.path.join(os.path.dirname(repo), "stabilization")
    res_dir = os.path.join(base, a.method, a.burst)
    vdir = a.vessels or os.path.join(res_dir, "vessels")
    labels = tifffile.imread(os.path.join(vdir, "vessel_labels.tif"))
    out = check(os.path.join(rec, a.burst), res_dir, labels, a.method)
    with open(a.out or os.path.join(vdir, "tracking_check.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
