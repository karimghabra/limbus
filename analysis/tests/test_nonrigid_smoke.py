"""Smoke test of the experimental non-rigid method on synthetic bursts.

Not the full ground-truth validation translation has (test_synthetic.py) —
that is still to do — but enough to catch the failures that matter most:

  still     a motionless burst must stay still: no deformation beyond
            measurement noise, and no worse alignment than translation
  rotation  two fixations 0.6 deg apart double the corners under translation;
            non-rigid must recover the rotation and remove the doubling
  strip     frames too short for a patch grid must fall back to translation

usage: python analysis/tests/test_nonrigid_smoke.py      (from the repository root)
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
from bench_common import _vessels  # noqa: E402
from stabilize.fields import FieldSet  # noqa: E402
from stabilize.methods import run  # noqa: E402

MARGIN = 120
fail = []


def scene(rng, w, h):
    hs, ws = h + 2 * MARGIN, w + 2 * MARGIN
    illum = cv2.resize(cv2.GaussianBlur(rng.random((hs // 16, ws // 16)).astype(np.float32), (0, 0), 3),
                       (ws, hs), interpolation=cv2.INTER_CUBIC)
    illum = 0.6 + 0.8 * (illum - illum.min()) / (np.ptp(illum) + 1e-6)
    depth = cv2.resize(_vessels(rng, hs // 2, ws // 2), (ws, hs), interpolation=cv2.INTER_LINEAR)
    return illum, depth


def render(rng, sc, w, h, poses):
    """poses: per frame (dx, dy, angle_deg) of the content about the frame centre."""
    illum, depth = sc
    hs, ws = depth.shape
    frames = []
    for dx, dy, ang in poses:
        cells = cv2.GaussianBlur(rng.random((hs, ws)).astype(np.float32), (0, 0), 3)
        cells = 0.55 + 0.45 * (cells - cells.min()) / (np.ptp(cells) + 1e-6)
        img = 2200.0 * illum * (1 - depth * cells)
        m = cv2.getRotationMatrix2D((ws / 2 - 0.5, hs / 2 - 0.5), ang, 1.0)
        m[:, 2] += (dx, dy)
        moved = cv2.warpAffine(img, m, (ws, hs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = moved[MARGIN:MARGIN + h, MARGIN:MARGIN + w]
        f = f + rng.normal(0, 1, f.shape).astype(np.float32) * np.sqrt(np.maximum(f, 0) / 4)
        frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16))
    return frames


def write_burst(folder, frames):
    os.makedirs(folder)
    for i, f in enumerate(frames):
        tifffile.imwrite(os.path.join(folder, f"frame_{i:06d}.tif"), f)
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"acquisition_requested": {"pixel_format": "Mono12", "offset_y": 0},
                   "capture": {"effective_fps": 32.0},
                   "pixel_values": {"bit_depth": 12, "dtype": "uint16"}}, fh)


def case(tmp, name, w, h, poses, seed):
    rng = np.random.default_rng(seed)
    folder = os.path.join(tmp, "recordings", f"burst_{name}")
    write_burst(folder, render(rng, scene(np.random.default_rng(1000), w, h), w, h, poses))
    out = os.path.join(tmp, "stabilization")
    rec = run("nonrigid", folder, out, log=lambda s: None)
    return rec, os.path.join(out, "nonrigid", f"burst_{name}")


def main():
    tmp = tempfile.mkdtemp(prefix="stabilize_nonrigid_")
    try:
        # ---- still -------------------------------------------------------------
        rec, res = case(tmp, "still", 1000, 720, [(0, 0, 0)] * 40, 1)
        fs = FieldSet(os.path.join(res, "fields.npz"))
        rms = []
        for i in fs.frame_index:
            fx, fy = fs.field(i)
            rms.append(np.sqrt(np.mean((fx - fx.mean()) ** 2 + (fy - fy.mean()) ** 2)))
        worst = float(np.max(rms))
        spread = rec["diagnostics"]["nonrigid"]["tile_residual_spread_max_px"]
        print(f"still     model {fs.model}, non-translational field RMS median {np.median(rms):.3f}, "
              f"max {worst:.3f} px (< 0.2); tile spread translation {spread['translation']:.3f} -> "
              f"non-rigid {spread['nonrigid']:.3f} px; used {int(fs.used.sum())}/40")
        # Patch shifts carry ~0.1 px of estimation noise (full resolution), and
        # the fitted field inherits it (measured: median 0.08, max 0.15 px) —
        # the same size as translation's own error, so its pass bar applies.
        # A bar below the noise floor (0.05 px, first planned) fails a correct
        # method. Structured, invented deformation would show up as larger
        # fields AND as worse tile alignment than translation.
        if worst >= 0.2:
            fail.append(f"still: invented deformation, field RMS {worst:.3f} px")
        if spread["nonrigid"] > spread["translation"] + 0.05:
            fail.append(f"still: tile spread worse than translation "
                        f"({spread['translation']:.3f} -> {spread['nonrigid']:.3f} px)")

        # ---- rotation: two fixations 0.6 deg apart, with jitter ----------------
        rng = np.random.default_rng(7)
        poses = [(rng.normal(0, 1.5), rng.normal(0, 1.5), 0.0 if t < 35 else 0.6) for t in range(60)]
        rec, res = case(tmp, "rotation", 1000, 720, poses, 2)
        nr = rec["diagnostics"]["nonrigid"]
        spread = nr["tile_residual_spread_max_px"]
        fs = FieldSet(os.path.join(res, "fields.npz"))
        rot = {int(i): (fs.affine[k, 1, 0] - fs.affine[k, 0, 1]) / 2
               for k, i in enumerate(fs.frame_index) if fs.used[k]}
        a = np.median([v for i, v in rot.items() if i < 35])
        b = np.median([v for i, v in rot.items() if i >= 35])
        got, want = abs(np.degrees(b - a)), 0.6
        print(f"rotation  model {nr['model']}, recovered pose difference {got:.3f} deg (true {want}); "
              f"tile spread max translation {spread['translation']:.2f} px -> non-rigid "
              f"{spread['nonrigid']:.2f} px; overlap translation "
              f"{rec['quality']['translation_after_same_frames']['overlap']:.3f} -> "
              f"{rec['quality']['after']['overlap']:.3f}")
        if abs(got - want) > 0.1 * want:
            fail.append(f"rotation: recovered {got:.3f} deg, true {want} deg")
        if not spread["nonrigid"] < 0.5 * spread["translation"]:
            fail.append(f"rotation: tile spread {spread['translation']:.2f} -> {spread['nonrigid']:.2f} px "
                        "not halved")
        if rec["quality"]["after"]["overlap"] < rec["quality"]["translation_after_same_frames"]["overlap"]:
            fail.append("rotation: non-rigid overlap below translation's")

        # ---- strip: falls back to translation ---------------------------------
        rec, res = case(tmp, "strip", 1000, 100, [(rng.normal(0, 1.5), 0, 0) for _ in range(40)], 3)
        model = rec.get("diagnostics", {}).get("nonrigid", {}).get("model")
        fs = FieldSet(os.path.join(res, "fields.npz"))
        print(f"strip     status {rec['status']}, model {model}, fields model {fs.model}")
        if rec["status"] != "ok" or model != "translation (fallback)" or fs.model != "translation":
            fail.append(f"strip: expected translation fallback, got {rec['status']} / {model}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
