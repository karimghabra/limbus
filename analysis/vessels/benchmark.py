"""Benchmark for small-vessel detection on REAL averaged frames.

Moved into the package on 22 Sep 2026. It had been living in a scratch
directory while five files inside the repo imported it, so a fresh clone could
not run any of them - and every recall and false-alarm number quoted for the
vessel work comes from here, which made the repo unable to reproduce its own
figures.

Positives (injection-recovery): small vessels rendered with the physical
model - absorbance of a cylinder, D * sqrt(1 - (d/r)^2), blurred by the
optics (sigma ~ 1.8 px, measured bound 2.1 px) - are multiplied into the raw
stabilized mean (transmission = exp(-absorbance)) at known places, depths and
radii. A planted vessel counts as found if >= 50% of its centreline lies
within 2.5 px of a detected centreline.

Negatives (inverted control): the same pipeline run on the INVERTED mean
(2 x median - mean). Real vessels are dark, so the inverted frame holds no
dark vessels; anything detected there is a false positive from texture.
Reported as detected centreline length per megapixel.

Crops keep each run short. Every pipeline is a function
    detect(mean_crop) -> list of centrelines (N x 2 arrays, crop coords)
"""
import json
import os
import sys
import time

import cv2
import numpy as np

PSF = 1.8


def smooth_curve(rng, W, H, L, margin=20):
    for _ in range(200):
        x0, y0 = rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)
        ang = rng.uniform(0, 2 * np.pi)
        s = np.arange(0, L, 0.5)
        curv = rng.uniform(-0.012, 0.012) + 0.004 * np.sin(s / rng.uniform(30, 80) + rng.uniform(0, 6))
        th = ang + np.cumsum(curv) * 0.5
        xs = x0 + np.cumsum(np.cos(th)) * 0.5
        ys = y0 + np.cumsum(np.sin(th)) * 0.5
        if xs.min() > margin and ys.min() > margin and xs.max() < W - margin and ys.max() < H - margin:
            return np.stack([xs, ys], 1)
    return None


def render_tube(shape, C, r, D, psf=PSF):
    """Absorbance of a blurred cylinder along centreline C (dense samples)."""
    H, W = shape
    x0, y0 = int(max(0, C[:, 0].min() - r - 4 * psf - 2)), int(max(0, C[:, 1].min() - r - 4 * psf - 2))
    x1, y1 = int(min(W, C[:, 0].max() + r + 4 * psf + 3)), int(min(H, C[:, 1].max() + r + 4 * psf + 3))
    # supersampled distance to the centreline, via a fine raster + distance transform
    ss = 4
    h, w = (y1 - y0) * ss, (x1 - x0) * ss
    line = np.full((h, w), 255, np.uint8)
    pts = np.rint((C - [x0, y0]) * ss).astype(np.int32)
    cv2.polylines(line, [pts], False, 0, 1)
    d = cv2.distanceTransform(line, cv2.DIST_L2, 5) / ss
    a = D * np.sqrt(np.clip(1 - (d / r) ** 2, 0, None))
    a = cv2.resize(a, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), psf)
    out = np.zeros(shape, np.float32)
    out[y0:y1, x0:x1] = a
    return out


def plant(mean, avoid, rng, n, radius, depth, lengths=(80, 250), gap=10):
    """depth = PEAK absorbance after blur (what the image shows); the
    cylinder's D is scaled so the blurred peak equals it."""
    H, W = mean.shape
    free = ~avoid & np.isfinite(mean)
    free = cv2.erode(free.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    A = np.zeros((H, W), np.float32)
    lines = []
    for _ in range(n * 40):
        if len(lines) >= n:
            break
        C = smooth_curve(rng, W, H, rng.uniform(*lengths))
        if C is None:
            continue
        xi, yi = np.rint(C[:, 0]).astype(int), np.rint(C[:, 1]).astype(int)
        if not free[yi, xi].all():
            continue
        a = render_tube((H, W), C, radius, 1.0)
        a *= depth / max(a.max(), 1e-9)
        A += a
        lines.append(C)
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [np.rint(C).astype(np.int32)], False, 1, 1)
        free &= ~(cv2.dilate(m, np.ones((2 * gap + 1,) * 2, np.uint8)) > 0)
    return mean * np.exp(-A), lines, A


def found_fraction(lines, detected, shape, tol=2.5):
    m = np.zeros(shape, np.uint8)
    for C in detected:
        cv2.polylines(m, [np.rint(C).astype(np.int32)], False, 1, 1)
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(tol)) + 1,) * 2)) > 0
    out = []
    for C in lines:
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, shape[1] - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, shape[0] - 1)
        out.append(near[yi, xi].mean())
    return np.array(out)


def length_of(detected):
    return float(sum(np.hypot(*np.diff(C, axis=0).T).sum() for C in detected if len(C) > 1))


# (burst, method, x0, y0, x1, y1). NOTE that 1522L and 1522R are the LEFT and
# RIGHT halves of ONE frame, not two recordings: the sharpness difference
# between them is the focus gradient across a single field of view, which is
# large. Any comparison between these two crops is a comparison of focus.
#
# `published` says whether the burst's averaged frame is in the repo, on the
# `reference-frames` branch. The others need the local stabilization output and
# cannot be reproduced from a clone.
CROPS = {
    "1522L": ("burst_2026-09-16_15-22-26", "nonrigid", 20, 20, 980, 480),
    "1522R": ("burst_2026-09-16_15-22-26", "nonrigid", 960, 20, 1840, 480),
    "1531L": ("burst_2026-09-16_15-31-37", "nonrigid", 20, 20, 980, 480),
    "1531R": ("burst_2026-09-16_15-31-37", "nonrigid", 960, 20, 1840, 480),
    "1550C": ("burst_2026-09-16_15-50-52", "nonrigid", 400, 200, 1400, 900),
}
PUBLISHED = {"burst_2026-09-16_15-22-26", "burst_2026-09-16_15-31-37",
             "burst_2026-09-16_15-31-50"}
GRID = [(1.0, 0.03), (1.0, 0.05), (1.5, 0.03), (1.5, 0.05), (2.5, 0.03), (2.5, 0.05), (1.5, 0.08)]


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frame_path(burst, method="nonrigid"):
    """Where the averaged frame for a burst is, preferring the copy committed
    to the repo so a clone can run the benchmark without the raw recordings."""
    date = burst.replace("burst_", "")
    here = [os.path.join(ROOT, "reference_frames", f"mean_{date}.tif"),
            os.path.join(ROOT, "stabilization", method, burst, "mean_stabilized.tif"),
            os.path.join(os.environ.get("LIMBUS_STABILIZATION", ""), method, burst,
                         "mean_stabilized.tif")]
    for p in here:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no averaged frame for {burst}. Fetch the published frames with\n"
        f"    git checkout reference-frames -- reference_frames/\n"
        f"or point LIMBUS_STABILIZATION at a stabilization output directory.")


def load_crop(key):
    import tifffile
    b, m, x0, y0, x1, y1 = CROPS[key]
    mean = tifffile.imread(frame_path(b, m)).astype(np.float32)
    return mean[y0:y1, x0:x1]


def run_bench(detect, keys=("1522L", "1550C"), per=8, seeds=(1, 2), log=print, tag=""):
    """Returns a dict of results; prints a compact table."""
    res = {"tag": tag, "crops": {}}
    t0 = time.time()
    for key in keys:
        mean = load_crop(key)
        H, W = mean.shape
        base = detect(mean)
        # avoid planting on/near what's already detected in the unmodified crop
        avoid = np.zeros((H, W), np.uint8)
        for C in base:
            cv2.polylines(avoid, [np.rint(C).astype(np.int32)], False, 1, 1)
        avoid = cv2.dilate(avoid, np.ones((25, 25), np.uint8)) > 0
        inv = 2 * np.nanmedian(mean) - mean
        neg = detect(inv)
        mp = H * W / 1e6
        cell = {"base_len_per_mp": length_of(base) / mp, "neg_len_per_mp": length_of(neg) / mp,
                "neg_count": len(neg), "recall": {}}
        for (r, d) in GRID:
            fr = []
            for sd in seeds:
                rng = np.random.default_rng(1000 * sd + int(r * 10) + int(d * 1000))
                mp_, lines, _ = plant(mean, avoid, rng, per, r, d)
                det = detect(mp_)
                fr.extend(found_fraction(lines, det, (H, W)).tolist())
            fr = np.array(fr)
            cell["recall"][f"r{r}_d{d}"] = [float((fr >= 0.5).mean()), float(fr.mean()), int(len(fr))]
        res["crops"][key] = cell
        rec = "  ".join(f"r{r}/{int(d * 100)}%:{100 * cell['recall'][f'r{r}_d{d}'][0]:3.0f}" for r, d in GRID)
        log(f"[{tag}] {key}: detected {cell['base_len_per_mp']:6.0f} px/MP | NEGATIVE control {cell['neg_len_per_mp']:5.0f} px/MP "
            f"({cell['neg_count']} objects) | planted found: {rec}   ({time.time() - t0:.0f}s)")
    return res
