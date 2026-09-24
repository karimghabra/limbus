"""Ground-truth test of the lucky stage: does it give better annotations?

Synthetic bursts with a KNOWN vessel network, eye motion, red-cell flicker
and shot noise (as test_synthetic.py), plus the thing lucky fusion is for:
defocus that varies across the field and over time. The eye's surface is
curved and tilted to the shallow depth of field, and it moves axially, so a
band of sharp focus sweeps across the field; each region is sharp in some
frames and soft in others. A control burst with constant focus checks that
fusion does no harm when there is nothing to gain.

Each burst is stabilized by the real pipeline (translation), fused, and
annotated with identical constants for the plain mean and every fused
image. Their centrelines, and the skeleton of the stabilization's own
consensus vessel mask, are scored against the true ones (METHODS.md §L5).

usage: python analysis/tests/test_lucky_synthetic.py      (from the repository root)
"""
import csv
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import tifffile
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from lucky.annotate import annotate, calibrate, contrast_image  # noqa: E402
from lucky.config import AnnotateParams, LuckyParams  # noqa: E402
from lucky.evaluate import centreline_agreement  # noqa: E402
from lucky.pipeline import fuse_burst  # noqa: E402
from stabilize.config import Params  # noqa: E402
from stabilize.pipeline import process_burst  # noqa: E402

W, H, MARGIN = 1000, 600, 60
N = 80
TOL = 2.0
LEVELS = np.array([0.6, 1.0, 1.5, 2.0, 2.75, 3.5, 4.5], np.float32)
fail = []


def network(rng, h, w):
    """Vessel darkening depth, and the TRUE centrelines: the polylines the
    vessels are drawn along, 1 px wide. Sparser than bench_common's network,
    at the density of the reference bursts: a few wide venules (24-40 px, as
    the widest in the reference bursts), venules, small vessels, then
    capillaries down to 1-2 px wide."""
    m = np.zeros((h, w), np.float32)
    centre = np.zeros((h, w), np.uint8)
    # (count, width range px, steps, step length px)
    for count, (t0, t1), steps, seg in ((2, (24, 40), 60, 30),
                                         (6, (8, 16), 60, 25),
                                         (40, (3, 6), 40, 14),
                                         (160, (1, 2), 25, 7)):
        for _ in range(count):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            ang = rng.uniform(0, 2 * np.pi)
            pts = []
            for _ in range(steps):
                pts.append((x, y))
                ang += rng.normal(0, 0.2)
                x += seg * np.cos(ang)
                y += seg * np.sin(ang)
            depth = float(rng.uniform(0.25, 0.7))
            pts = np.array(pts, np.float32)
            # sub-pixel polylines (4 fractional bits) so centrelines are exact
            cv2.polylines(m, [np.round(pts * 16).astype(np.int32)], False, depth,
                          int(rng.integers(t0, t1 + 1)), cv2.LINE_AA, shift=4)
            cv2.polylines(centre, [np.round(pts * 16).astype(np.int32)], False, 1, 1,
                          cv2.LINE_8, shift=4)
    return cv2.GaussianBlur(m, (0, 0), 0.7), centre


def make_scene(seed):
    rng = np.random.default_rng(seed)
    hs, ws = H + 2 * MARGIN, W + 2 * MARGIN
    illum = cv2.resize(cv2.GaussianBlur(rng.random((hs // 16, ws // 16)).astype(np.float32),
                                        (0, 0), 3), (ws, hs), interpolation=cv2.INTER_CUBIC)
    illum = 0.7 + 0.6 * (illum - illum.min()) / (np.ptp(illum) + 1e-6)
    tex = cv2.GaussianBlur(rng.standard_normal((hs, ws)).astype(np.float32), (0, 0), 1.5)
    tex /= tex.std()
    depth, centre = network(rng, hs, ws)
    return illum, tex, depth, centre


def focus_sigma(t, sweep):
    """Defocus blur sigma (px) per frame pixel. The in-focus band follows
    z(t) across a field whose surface depth rises left to right."""
    X = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    Y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    surface = 0.8 * X + 0.2 * Y
    if not sweep:
        return np.full((H, W), 1.6, np.float32)
    z = 0.5 + 0.55 * np.sin(2 * np.pi * t / 37.0) + 0.1 * np.sin(2 * np.pi * t / 11.0)
    return np.clip(0.6 + 6.0 * np.abs(surface - z), 0.6, 4.5).astype(np.float32)


def varying_blur(img, sigma):
    stack = np.stack([cv2.GaussianBlur(img, (0, 0), float(s)) for s in LEVELS])
    pos = np.interp(sigma, LEVELS, np.arange(LEVELS.size, dtype=np.float32))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, LEVELS.size - 1)
    a = (pos - lo).astype(np.float32)
    rows, cols = np.indices(sigma.shape)
    return (1 - a) * stack[lo, rows, cols] + a * stack[hi, rows, cols]


def render(seed, scene, sweep):
    rng = np.random.default_rng(seed)
    illum, tex, depth, _ = scene
    hs, ws = depth.shape
    steps = rng.normal(0, 1.5, (N, 2))
    steps[0] = 0
    traj = np.cumsum(steps, axis=0)
    frames = []
    for t in range(N):
        cells = cv2.GaussianBlur(rng.random((hs, ws)).astype(np.float32), (0, 0), 2)
        cells = 0.55 + 0.45 * (cells - cells.min()) / (np.ptp(cells) + 1e-6)
        scene_t = 2200.0 * illum * (1 + 0.04 * tex) * (1 - depth * cells)
        m = np.float32([[1, 0, traj[t, 0]], [0, 1, traj[t, 1]]])
        moved = cv2.warpAffine(scene_t, m, (ws, hs), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        f = moved[MARGIN:MARGIN + H, MARGIN:MARGIN + W]
        f = varying_blur(f, focus_sigma(t, sweep))
        f = f + rng.normal(0, 1, f.shape).astype(np.float32) * np.sqrt(np.maximum(f, 0) / 4)
        frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16))
    return frames, traj


def write_burst(folder, frames):
    os.makedirs(folder)
    for i, f in enumerate(frames):
        tifffile.imwrite(os.path.join(folder, f"frame_{i:06d}.tif"), f)
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"acquisition_requested": {"pixel_format": "Mono12", "offset_y": 0},
                   "capture": {"effective_fps": 32.0},
                   "pixel_values": {"bit_depth": 12, "dtype": "uint16"}}, fh)


def truth_on_grid(centre, traj, est, used):
    """True centreline on the stabilized grid. Frame t shows the scene at
    x + MARGIN - traj_t; stabilization shows frame t at x + est_t, so the
    stabilized image shows the scene at x + MARGIN - mean(traj - est)."""
    off = np.mean(traj[used] - est[used], axis=0)
    m = np.float32([[1, 0, off[0] - MARGIN], [0, 1, off[1] - MARGIN]])
    c = cv2.warpAffine(centre.astype(np.float32), m, (W, H), flags=cv2.INTER_LINEAR)
    return skeletonize(c > 0.25)


def case(tmp, name, sweep, seed):
    scene = make_scene(1000 + seed)
    frames, traj = render(seed, scene, sweep)
    folder = os.path.join(tmp, "bursts", f"burst_{name}")
    write_burst(folder, frames)
    stab = os.path.join(tmp, "stab")
    process_burst(folder, os.path.join(stab, "translation"), Params(), log=lambda s: None,
                  method="translation")
    result = os.path.join(stab, "translation", f"burst_{name}")
    with open(os.path.join(result, "transforms.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    est = np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows])
    used = np.array([r["registered"] == "1" for r in rows])
    truth = truth_on_grid(scene[3], traj, est, used)

    ap = AnnotateParams()
    res = fuse_burst(folder, stab, LuckyParams(), method="translation", halves=False)
    valid = res["valid"]
    region = cv2.erode(valid.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    # the annotation the stabilization stage already gives: its consensus
    # vessel mask, skeletonized
    consensus = (tifffile.imread(os.path.join(result, "consensus_mask.tif")) > 0) & valid
    scores = {"consensus": centreline_agreement(skeletonize(consensus), truth, TOL, region)}
    # one Frangi constant per burst, from the plain mean, for every image
    c = calibrate(contrast_image(res["images"]["p0"], valid, ap), valid, ap)
    for label, img in res["images"].items():
        a = annotate(img, ap, c, valid)
        scores[label] = centreline_agreement(a["skeleton"], truth, TOL, region)
    return scores, res["n_eff"]


def main():
    tmp = tempfile.mkdtemp(prefix="lucky_synth_")
    out = LuckyParams().output
    lucky = f"p{out:g}"
    try:
        results = {}
        for name, sweep, seed in (("sweep_a", True, 1), ("sweep_b", True, 2),
                                  ("constant_focus", False, 3)):
            scores, n_eff = case(tmp, name, sweep, seed)
            results[name] = scores
            print(f"\n{name}   centreline vs truth, {TOL:g} px tolerance "
                  f"(n_eff: effective frames per band, finest first)")
            for label, s in scores.items():
                ne = ", ".join(f"{x:.0f}" for x in n_eff[label]) if label in n_eff else ""
                what = {"consensus": "consensus-mask skeleton", "p0": "annotated plain mean"}.get(
                    label, f"annotated lucky {label}")
                print(f"  {what:<26s} precision {s['precision']:.3f}  recall {s['recall']:.3f}  "
                      f"F1 {s['f1']:.3f}  {'[' + ne + ']' if ne else ''}")

        for name, s in results.items():
            # the new annotator must beat the skeleton it replaces by far
            if s["p0"]["f1"] < s["consensus"]["f1"] + 0.3:
                fail.append(f"{name}: annotator F1 {s['p0']['f1']:.3f} vs consensus "
                            f"skeleton {s['consensus']['f1']:.3f}")
        for name in ("sweep_a", "sweep_b"):
            s = results[name]
            gain = s[lucky]["f1"] - s["p0"]["f1"]
            print(f"{name}: lucky {lucky} F1 {s['p0']['f1']:.3f} -> {s[lucky]['f1']:.3f} ({gain:+.3f})")
            if gain < 0.015:
                fail.append(f"{name}: lucky fusion did not improve centreline F1 ({gain:+.3f})")
        s = results["constant_focus"]
        if s[lucky]["f1"] < s["p0"]["f1"] - 0.01:
            fail.append(f"constant focus: {lucky} lost F1 "
                        f"({s['p0']['f1']:.3f} -> {s[lucky]['f1']:.3f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nFAIL:\n  " + "\n  ".join(fail) if fail else "\nPASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
