"""The large vessels, found the direct way.

A thick vessel is a dark region that is wide. That is the whole definition,
and it needs a threshold and a distance transform - not a filter bank. This
module exists because the Fourier matched-filter approach in `fourier.py`
spent a great deal of machinery answering "which straight vessel best explains
this pixel?" when the question for the trunks is only "is this dark, and is it
thick?", which the absorbance image answers directly.

    1. Threshold the absorbance with hysteresis. The level is taken from the
       frame's own background, and the spread is measured on the BRIGHT half
       only, so the vessels cannot inflate the threshold that is meant to find
       them.
    2. Measure the half-width everywhere with a distance transform. This IS
       the radius: the distance from a pixel to the nearest non-vessel pixel.
    3. Keep the regions that are thick enough, skeletonise them, and read the
       radius off the distance transform along each centreline.

Every vessel found this way carries its own measured radius, so the network
can be worked from the largest calibre down simply by sorting.
"""
import cv2
import numpy as np

from . import ridges
from . import skeleton as sk


def threshold(A, valid, k_hi=6.0, k_lo=2.5):
    """Hysteresis threshold on absorbance; returns (mask, level, spread).

    The spread comes from the half of the distribution BELOW the median.
    Vessels are dark, so they live in the upper tail: a two-sided spread is
    inflated by the very thing being detected, and a vessel-rich frame would
    raise its own threshold.
    """
    v = A[valid]
    med = float(np.median(v))
    lower = v[v < med]
    spread = 1.4826 * float(np.median(med - lower)) if lower.size else 1.0
    hi, lo = med + k_hi * spread, med + k_lo * spread
    strong = (A >= hi) & valid
    weak = (A >= lo) & valid
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    alive = np.zeros(n, bool)
    alive[lab[strong]] = True
    alive[0] = False
    return alive[lab], med, spread


def detect(A, valid, min_radius=4.0, min_len=40.0, min_aspect=4.0, k_hi=6.0,
           k_lo=2.5, smooth=1.0, log=None):
    """Large vessels as (centreline, radius-along-it) pairs, thickest first.

    `min_radius` is a real half-width in pixels, measured, not a filter scale.
    """
    Af = cv2.GaussianBlur(A, (0, 0), smooth) if smooth else A
    mask, med, spread = threshold(Af, valid, k_hi, k_lo)
    # half-width of the dark region at every pixel
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if log:
        log(f"[large] background {med:.4f} +- {spread:.4f}; "
            f"threshold {med + k_lo * spread:.3f}/{med + k_hi * spread:.3f}; "
            f"{100 * mask.mean():.1f}% of the frame is vessel, "
            f"max half-width {dist.max():.1f} px")

    # Thin the WHOLE dark region, then judge each centreline by the width
    # underneath it. Thinning first and measuring second keeps a vessel whole
    # where it narrows, instead of breaking it wherever it dips below the
    # width cut.
    spine = sk.thin(mask)
    spine = sk.prune(spine, np.maximum(dist, 1.0), spur=8)
    out = []
    drop = 0
    for C in ridges.trace(spine, min_len=min_len):
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, A.shape[1] - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, A.shape[0] - 1)
        r = dist[yi, xi]
        rad = float(np.median(r))
        if rad < min_radius:
            continue
        # A vessel is long relative to its width; a dark patch is not. Without
        # this the dark limbal corners of a frame come back as "vessels" of
        # 15-38 px radius, which is simply the half-width of a blob.
        length = float(np.hypot(*np.diff(C, axis=0).T).sum()) if len(C) > 1 else 0.0
        if length < min_aspect * rad:
            drop += 1
            continue
        out.append((C, r))
    out.sort(key=lambda cr: -float(np.median(cr[1])))
    if log:
        tot = sum(np.hypot(*np.diff(C, axis=0).T).sum() for C, _ in out if len(C) > 1)
        log(f"[large] {len(out)} vessels, {tot:.0f} px ({drop} dropped as blobs), "
            f"radii {', '.join(f'{np.median(r):.0f}' for _, r in out[:8])}...")
    return out


def lumen(vessels, shape, pad=0.0):
    """Label image: each vessel's own width filled in, largest drawn first so
    a thin vessel crossing a thick one does not erase it."""
    lab = np.zeros(shape, np.int32)
    for i, (C, r) in enumerate(vessels, 1):
        for (x, y), rr in zip(C, r):
            cv2.circle(lab, (int(round(x)), int(round(y))), int(round(rr + pad)), i, -1)
    return lab
