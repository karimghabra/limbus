"""Ground-truth test of the stabilization pipeline.

Builds synthetic bursts where the true motion is known exactly: a vessel
network (the benchmark generator used for the camera tests), red-cell
flicker inside the vessels, shot noise, blinks, and a glare spot fixed to
the camera rather than the tissue. Then checks the pipeline recovers the
motion, catches the blinks, isn't pulled by the glare, and that the
stability metrics respond to motion the way they should.

usage: python analysis/tests/test_synthetic.py      (from the repository root)
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repository root
sys.path.insert(0, os.path.dirname(HERE))              # analysis/ (the stabilize package)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))   # the vessel-network generator
from bench_common import _vessels  # noqa: E402  existing vessel-network generator
from stabilize.config import Params  # noqa: E402
from stabilize.metrics import mask_consensus  # noqa: E402
from stabilize.pipeline import process_burst  # noqa: E402
from stabilize.vessels import (contrast_constant, envelope, glare_mask, to_working,  # noqa: E402
                               vesselness, working_sigmas)

W, H, MARGIN = 1000, 600, 120
N = 60
BLINKS = (15, 16, 40)
fail = []


def make_scene(rng):
    hs, ws = H + 2 * MARGIN, W + 2 * MARGIN
    illum = cv2.resize(cv2.GaussianBlur(rng.random((hs // 16, ws // 16)).astype(np.float32), (0, 0), 3),
                       (ws, hs), interpolation=cv2.INTER_CUBIC)
    illum = 0.6 + 0.8 * (illum - illum.min()) / (np.ptp(illum) + 1e-6)
    tex = cv2.GaussianBlur(rng.standard_normal((hs, ws)).astype(np.float32), (0, 0), 1.5)
    tex /= tex.std()
    # draw the network at half size and upscale: doubles vessel widths to
    # 2-28 px, matching the wide vessels measured in the real bursts
    depth = cv2.resize(_vessels(rng, hs // 2, ws // 2), (ws, hs), interpolation=cv2.INTER_LINEAR)
    return illum, tex, depth


def trajectory(rng, step_sd, saccade):
    steps = rng.normal(0, step_sd, size=(N, 2))
    steps[0] = 0
    if saccade:
        steps[30] += (25.0, -15.0)
    return np.cumsum(steps, axis=0)


def render(rng, scene, traj, glare, blinks):
    illum, tex, depth = scene
    hs, ws = depth.shape
    frames = []
    for t in range(N):
        if t in blinks:
            # eyelid across the field: bright, nearly featureless
            gy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
            f = 3000 + 200 * gy + rng.normal(0, 30, (H, W)).astype(np.float32)
        else:
            # red cells: a fresh random pattern of dense and sparse cells
            # inside the vessels each frame, while the vessel walls stay put
            cells = cv2.GaussianBlur(rng.random((hs, ws)).astype(np.float32), (0, 0), 3)
            cells = 0.55 + 0.45 * (cells - cells.min()) / (np.ptp(cells) + 1e-6)
            scene_t = 2200.0 * illum * (1 + 0.06 * tex) * (1 - depth * cells)
            m = np.float32([[1, 0, traj[t, 0]], [0, 1, traj[t, 1]]])   # content moves by +traj
            moved = cv2.warpAffine(scene_t, m, (ws, hs), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            f = moved[MARGIN:MARGIN + H, MARGIN:MARGIN + W]
            f = f + rng.normal(0, 1, f.shape).astype(np.float32) * np.sqrt(np.maximum(f, 0) / 4)
        if glare:
            cv2.circle(f, (760, 180), 28, 4095, -1)       # fixed to the camera
        frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16))
    return frames


def write_burst(folder, frames, fps=32.0):
    os.makedirs(folder)
    for i, f in enumerate(frames):
        tifffile.imwrite(os.path.join(folder, f"frame_{i:06d}.tif"), f)
    manifest = {"acquisition_requested": {"pixel_format": "Mono12", "offset_y": 0},
                "capture": {"effective_fps": fps},
                "pixel_values": {"bit_depth": 12, "dtype": "uint16"}}
    with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def perfect_registration_overlap(frames, truth, blinks):
    """Vessel overlap if registration were perfect: the same masks, aligned
    with the TRUE shifts. This is the right ceiling for a burst. Comparing
    against a motionless burst is not — the generator creates motion with
    bilinear warps, so moving frames are intrinsically ~12% blurrier and no
    registration could reach the still burst's score."""
    p = Params()
    keep = [t for t in range(N) if t not in blinks]
    scale = p.working_scale
    imgs = [to_working(frames[t], scale) for t in keep]
    sig = working_sigmas(p, scale, imgs[0].shape[0])
    c = contrast_constant(imgs[len(imgs) // 2], sig, p)
    glare = [glare_mask(im, 4095, p, scale) for im in imgs]
    V = np.stack([vesselness(im, sig, c, p) for im in imgs])
    for v, g in zip(V, glare):
        v[g] = 0
    M = np.stack([envelope(v, np.percentile(V, p.envelope_percentile)) for v in V])
    O = np.stack([~g for g in glare]).astype(np.uint8)
    rec, _ = mask_consensus(M, truth[keep] * scale, np.arange(len(keep)), p,
                            aligned=True, O=O)
    return rec["overlap"]


def run(tmp, name, seed, step_sd, saccade, glare, blinks):
    rng = np.random.default_rng(seed)
    scene = make_scene(np.random.default_rng(1000))      # same tissue every case
    truth = trajectory(rng, step_sd, saccade)
    folder = os.path.join(tmp, "bursts", f"burst_{name}")
    frames = render(rng, scene, truth, glare, blinks)
    write_burst(folder, frames)
    out = os.path.join(tmp, "out")
    rec = process_burst(folder, out, Params(), log=lambda s: None)
    rows = list(csv.DictReader(open(os.path.join(out, f"burst_{name}", "transforms.csv"),
                                    encoding="utf-8")))
    est = np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows])
    reg = np.array([r["registered"] == "1" for r in rows])
    gate = np.array([r["passed_gate"] == "1" for r in rows])
    use = reg & np.array([t not in blinks for t in range(N)])
    # both trajectories are only defined up to a constant offset
    err = (est[use] - est[use].mean(0)) - (truth[use] - truth[use].mean(0))
    rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    ceiling = perfect_registration_overlap(frames, truth, blinks)
    return rec, rms, gate, reg, truth, ceiling


def main():
    tmp = tempfile.mkdtemp(prefix="stabilize_synth_")
    try:
        cases = {}
        for name, seed, sd, sacc, glare, blinks in (
                ("still", 1, 0.0, False, False, ()),
                ("gentle", 2, 0.8, False, False, ()),
                ("rough", 3, 3.0, True, False, ()),
                ("rough_blinks_glare", 3, 3.0, True, True, BLINKS)):
            rec, rms, gate, reg, truth, ceiling = run(tmp, name, seed, sd, sacc, glare, blinks)
            q = rec["quality"]
            cases[name] = (rec, rms)
            print(f"{name:20s} true motion {np.ptp(truth[:, 0]):5.1f} x {np.ptp(truth[:, 1]):5.1f} px | "
                  f"RMS error {rms:.3f} px | overlap {q['before']['overlap']:.3f} -> "
                  f"{q['after']['overlap']:.3f} (perfect-registration ceiling {ceiling:.3f}) | "
                  f"Dice {q['before']['dice']:.3f} -> {q['after']['dice']:.3f} | "
                  f"registered {int(reg.sum())}/{N}, gate kept {int(gate.sum())}", flush=True)
            if rms > 0.2:
                fail.append(f"{name}: recovered motion RMS error {rms:.3f} px > 0.2 px")
            # stabilization must reach what perfect registration would achieve
            if q["after"]["overlap"] < ceiling - 0.01:
                fail.append(f"{name}: after-overlap {q['after']['overlap']:.3f} short of the "
                            f"perfect-registration ceiling {ceiling:.3f}")
            false_rej = [t for t in range(N) if not gate[t] and t not in blinks]
            if false_rej:
                fail.append(f"{name}: gate rejected good frames {false_rej}")
            if blinks:
                missed = [t for t in blinks if gate[t]]
                if missed:
                    fail.append(f"{name}: gate missed blinks at frames {missed}")

        still = cases["still"][0]["quality"]
        gentle = cases["gentle"][0]["quality"]
        rough = cases["rough"][0]["quality"]
        # more motion should make the UNstabilized masks agree less
        if not (still["before"]["overlap"] > gentle["before"]["overlap"]
                > rough["before"]["overlap"]):
            fail.append("raw vessel overlap does not fall as motion increases")
        # glare must not pull the registration: same motion, with vs without
        clean_rms, glare_rms = cases["rough"][1], cases["rough_blinks_glare"][1]
        print(f"glare/blink effect on accuracy: {clean_rms:.3f} -> {glare_rms:.3f} px RMS")
        if glare_rms > clean_rms + 0.1:
            fail.append(f"glare/blinks degraded accuracy {clean_rms:.3f} -> {glare_rms:.3f} px")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
