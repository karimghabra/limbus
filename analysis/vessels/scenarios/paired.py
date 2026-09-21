"""Comparing two detector variants without confounding them.

The obvious way to compare variant A with variant B is to run the sweep twice
and read off the two tables. That is wrong here, and it produced a result this
benchmark reported as a regression before a paired test found otherwise.

The planting locations depend on the detector. Scenarios are planted on ground
the detector finds empty, so the free mask is derived from the variant's own
baseline output - change the variant and the scenes move. The two tables then
differ for two reasons at once and there is no way to separate them.

Measured: on shallow bifurcations, two separate sweeps read 0.21 and 0.08 and
looked like a clear regression with a tidy mechanism behind it. Planting the
scenes ONCE and running both variants on the identical images gives 0.183 and
0.183, with the two disagreeing on 0 of 60 scenes. The whole difference was
where the scenarios happened to land.

So: build the scenes once, from one fixed free mask, and evaluate every
variant on the same images. The difference is then paired, and its interval
can be bootstrapped over scenes rather than over two independent samples.
"""
import cv2
import numpy as np

from .geometry import MAKERS, SCENARIOS, paint, place  # noqa: F401
from .score import score


def free_ground(A, valid, detect, busy_px=25, margin_px=15):
    """Where scenarios may be planted, from ONE reference detection.

    Taken once and reused for every variant under test, which is the whole
    point: the free mask must not depend on the variant.
    """
    m = np.zeros(A.shape, np.uint8)
    for C in detect(A, valid):
        cv2.polylines(m, [np.rint(np.asarray(C, float)).astype(np.int32)], False, 1, 1)
    free = valid & ~(cv2.dilate(m, np.ones((busy_px, busy_px), np.uint8)) > 0)
    return cv2.erode(free.astype(np.uint8), np.ones((margin_px, margin_px), np.uint8)) > 0


def build_scenes(A, free, family, params, seeds, per=6, depth=0.05, spacing=41):
    """[(image, copies)] - the scenes, made once."""
    H, W = A.shape
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        copies, avail = [], free.copy()
        for _ in range(per * 50):
            if len(copies) >= per:
                break
            lines = MAKERS[family](rng, W, H, **params)
            moved = place(avail, rng, lines)
            if moved is None:
                continue
            copies.append(moved)
            m = np.zeros((H, W), np.uint8)
            for ln in moved:
                cv2.polylines(m, [np.rint(ln["C"]).astype(np.int32)], False, 1, 1)
            avail &= ~(cv2.dilate(m, np.ones((spacing, spacing), np.uint8)) > 0)
        if not copies:
            continue
        Ap = np.zeros((H, W), np.float32)
        for c in copies:
            Ap += paint((H, W), c, depth)
        out.append((A + Ap, copies))
    return out


def resolved(scenes, valid, detect):
    """Per scene copy: did the whole scenario come back right?"""
    out = []
    for A, copies in scenes:
        det = detect(A, valid)
        for lines in copies:
            sc = score(lines, det, A.shape)
            sp = [r["span"] for r in sc["lines"].values()]
            out.append(float(all(s >= 0.8 for s in sp) and not sc["merges"]))
    return np.array(out)


def compare(scenes, valid, variants, n_boot=4000, seed=0):
    """Every variant on the SAME scenes; differences paired against the first.

    variants: {name: detect}. Returns {name: (mean, lo, hi, n_disagreed)}.
    """
    names = list(variants)
    got = {n: resolved(scenes, valid, variants[n]) for n in names}
    base = got[names[0]]
    rng = np.random.default_rng(seed)
    out = {names[0]: (float(base.mean()), 0.0, 0.0, 0)}
    for n in names[1:]:
        d = got[n] - base
        boot = [float(np.mean(rng.choice(d, len(d)))) for _ in range(n_boot)]
        out[n] = (float(got[n].mean()), float(np.percentile(boot, 2.5)),
                  float(np.percentile(boot, 97.5)), int((d != 0).sum()))
    return out
