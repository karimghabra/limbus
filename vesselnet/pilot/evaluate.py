"""Evaluate a pilot model three ways.

    python vesselnet/pilot/evaluate.py val  --run runs/pilot1 --data data/pilot
    python vesselnet/pilot/evaluate.py zoo  --run runs/pilot1 --seeds 0 10 20
    python vesselnet/pilot/evaluate.py real --run runs/pilot1 --image mean_stabilized.tif --out real.png

val:  held-out vesselscene scenes (seed % 10 == 0), vesselscene's matching
      rule (a detection matches the nearest unmatched observable junction
      within max(radius, 11.9 px); unmatched ones near a don't-care junction
      are ignored), at a range of heat thresholds.
zoo:  vesselmap's structure zoo (a different generator, never trained on),
      counted like vesselmap.intersections.score_zoo: a marked intersection
      is found within 4 px + its widest radius; a detection near none is
      false.  Not a like-for-like test: the zoo counts each crossing of a
      repeated-crossing structure, vesselscene clusters events within 2
      lambda into one junction.
real: detections over a real image (PNG, circles coloured by type) and the
      heat map (.npy).  No truth: for looking at.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # the LIMBUS checkout, for vesselmap
import net  # noqa: E402


def load_model(run, device):
    m = net.UNet()
    m.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu"))
    return m.to(device)


def cmd_val(a, model):
    paths = sorted(p for p in glob.glob(os.path.join(a.data, "s*.npz")) if int(os.path.basename(p)[1:6]) % 10 == 0)
    scenes = [net.load_scene(p) for p in paths]
    thrs = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    tot = {t: dict(tp=0, fp=0, n=0, type_ok=0) for t in thrs}
    for S in scenes:
        d = net.peaks(net.predict(model, S["X"]), thr=min(thrs), valid=S["valid"])
        for t in thrs:
            r = net.score_scene(d, S, t)
            for k in r:
                tot[t][k] += r[k]
    print("held-out scenes:", [os.path.basename(p) for p in paths])
    for t, r in tot.items():
        rec, prec = r["tp"] / max(r["n"], 1), r["tp"] / max(r["tp"] + r["fp"], 1)
        print("thr %.2f  recall %.3f precision %.3f F1 %.3f  type acc %.2f  (%d found, %d false, of %d)" % (
            t, rec, prec, 2 * rec * prec / max(rec + prec, 1e-9), r["type_ok"] / max(r["tp"], 1),
            r["tp"], r["fp"], r["n"]))


def zoo_counts(dets, V, tiles, rows):
    """found / marks / false as in intersections.score_zoo (no arms)."""
    from vesselmap import intersections as X
    found = marks = false = 0
    for ri, (name, _, _) in enumerate(rows):
        is_x = any(w in name for w in X._INTERSECT) and "parallel pair" not in name
        for t in (t for t in tiles if t["row"] == ri):
            x0, y0, x1, y1 = t["box"]
            vs = [V[i] for i in t["vessels"]]
            mk = t["ambiguous"][:1] if "fork" in name else (t["ambiguous"] if is_x else [])
            mine = [d for d in dets if x0 <= d["xy"][0] <= x1 and y0 <= d["xy"][1] <= y1]
            used = set()
            for group in X._groups(mk, vs):
                near = [q for q, d in enumerate(mine)
                        if any(np.linalg.norm(d["xy"] - p) <= 4.0 + X._radius_at(vs, p) for p in group)]
                marks += 1
                if near:
                    found += 1
                    used.update(near)
            false += len(mine) - len(used)
    return found, marks, false


def cmd_zoo(a, model):
    from vesselmap import zoo
    thrs = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    tot = {t: [0, 0, 0] for t in thrs}
    for s in a.seeds:
        for k, rows in enumerate((zoo.ROWS, zoo.ROWS_CALIBRE, zoo.ROWS_CROSSINGS)):
            I, V, tiles = zoo.zoo_sheet(s + k, rows=rows)
            X_, v = net.preprocess(I)
            dets = net.peaks(net.predict(model, X_), thr=min(thrs), valid=v)
            for t in thrs:
                f, m, fa = zoo_counts([d for d in dets if d["score"] >= t], V, tiles, rows)
                tot[t][0] += f
                tot[t][1] += m
                tot[t][2] += fa
    for t in thrs:
        print("thr %.2f  found %d/%d  false %d" % (t, *tot[t]))


def cmd_real(a, model):
    import cv2
    import tifffile
    I = tifffile.imread(a.image).astype(np.float32)
    X_, v = net.preprocess(I)
    P = net.predict(model, X_)
    dets = net.peaks(P, thr=a.thr, valid=v)
    g = np.clip((X_ + 4) / 8 * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    col = dict(crossing=(0, 200, 255), branch=(80, 255, 80), compound=(255, 120, 255))
    for d in dets:
        cv2.circle(rgb, tuple(int(c) for c in d["xy"]), 9, col[d["type"]], 2)
    cv2.imwrite(a.out, rgb)
    np.save(os.path.splitext(a.out)[0] + "_heat.npy", P[0].astype(np.float16))
    print(len(dets), "junctions at threshold", a.thr, {t: sum(d["type"] == t for d in dets) for t in col})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("val", "zoo", "real"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 10, 20])
    ap.add_argument("--image")
    ap.add_argument("--out", default="real.png")
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = load_model(a.run, a.device)
    dict(val=cmd_val, zoo=cmd_zoo, real=cmd_real)[a.what](a, model)


if __name__ == "__main__":
    main()
