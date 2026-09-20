"""Scoring that can tell a merge from a resolution, which the first attempt could not.

The first version assigned every detected point to the nearest truth line and
counted it only for that one. The intent was that a line drawn down the middle
of a close pair would be split between them and register as a merge. It does
the opposite: the ownership map is a Voronoi partition, the whole midline
falls on one side of it, and the merged detection is scored as a clean hit on
one vessel and a total miss on the other. Measured on planted pairs, the
detector emitted one midline detection at 3, 4 and 5 px separation - the same
behaviour every time - and the metric called it "both found", "one found one
lost" and "both missed" respectively, purely as a function of where the
tolerance fell relative to half the separation.

That is the same class of error as the coverage metric this was built to
replace: a number that cannot see the defect the eye sees first.

So: no ownership, and a tolerance that scales with how close the truth lines
actually are. For a truth point p, let d_nn(p) be the distance to the nearest
OTHER truth line. A detection counts as covering p if it passes within

    tol(p) = clip(0.6 * d_nn(p), 1.5, 8.0)

which is more than half the local separation and less than all of it. A single
midline detection therefore covers BOTH lines of a pair - and is reported as a
merge - while two correctly separated detections each cover their own and not
the other. Where no other truth line is near, tol falls back to a fixed 1.5-8 px.

Reported per truth line:
    span    largest fraction covered by ONE detection
    cover   fraction covered by all detections together
    parts   number of detections in a minimal cover (not a count of
            everything that touches it, which fell to zero as fragmentation
            became total in the first version)
and per scenario:
    merges  detections covering >= 0.35 of two different truth lines
    offset  transverse position of each detection relative to the truth it is
            nearest to - the direct measurement, free of any tolerance
"""
import cv2
import numpy as np


def _dist_to(shape, C, thick=1):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.rint(np.asarray(C, float)).astype(np.int32)], False, 1, thick)
    return cv2.distanceTransform(1 - m, cv2.DIST_L2, 5), m


def exempt_pairs(lines):
    """Truth pairs that ONE detection is allowed to cover.

    At a bifurcation the parent continues into one of its children: a single
    vessel spanning parent and child is the right answer, not a merge. The
    same holds for a trunk and each of its branches. Counting those as merges
    scored every planted bifurcation as a failure at every angle while span
    sat at 0.98 - a number uniform enough to be a metric error, and it was.

    Two CHILDREN covered by one detection is still a merge, and so is a trunk
    joined to another trunk.
    """
    ex = set()
    for i, a in enumerate(lines):
        for j, b in enumerate(lines):
            if i >= j:
                continue
            ra, rb = a.get("role", ""), b.get("role", "")
            if {ra, rb} == {"parent", "child"} or {ra, rb} == {"trunk", "branch"}:
                ex.add((i, j))
    return ex


def score(lines, detected, shape, tol_min=1.5, tol_max=8.0, frac=0.6, exempt=None):
    H, W = shape
    n = len(lines)
    if n == 0:
        return {}
    dts = [_dist_to(shape, ln["C"])[0] for ln in lines]
    pts, tols = [], []
    for t, ln in enumerate(lines):
        P = np.asarray(ln["C"], float)
        xi = np.clip(np.rint(P[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(P[:, 1]).astype(int), 0, H - 1)
        pts.append((xi, yi))
        if n > 1:
            dnn = np.min([dts[o][yi, xi] for o in range(n) if o != t], axis=0)
        else:
            dnn = np.full(len(xi), tol_max)
        tols.append(np.clip(frac * dnn, tol_min, tol_max))

    cov = np.zeros((len(detected), n))
    dets_d = []
    for di, C in enumerate(detected):
        C = np.asarray(C, float)
        if len(C) < 2:
            dets_d.append(None)
            continue
        d2, _ = _dist_to(shape, C)
        dets_d.append(d2)
        for t in range(n):
            xi, yi = pts[t]
            cov[di, t] = float((d2[yi, xi] <= tols[t]).mean())

    out = {}
    for t in range(n):
        xi, yi = pts[t]
        hit = np.zeros(len(xi), bool)
        order = np.argsort(-cov[:, t])
        parts = 0
        for di in order:
            if cov[di, t] < 0.05 or dets_d[di] is None:
                break
            new = (dets_d[di][yi, xi] <= tols[t]) & ~hit
            if new.mean() < 0.05:
                continue
            hit |= new
            parts += 1
        out[t] = {"role": lines[t]["role"], "span": float(cov[:, t].max()) if len(detected) else 0.0,
                  "cover": float(hit.mean()), "parts": int(parts)}
    ex = exempt_pairs(lines) if exempt is None else exempt
    merges = []
    for di in range(len(detected)):
        hit = [int(h) for h in np.nonzero(cov[di] >= 0.35)[0]]
        if len(hit) < 2:
            continue
        bad = [(a, b) for k, a in enumerate(hit) for b in hit[k + 1:] if (a, b) not in ex]
        if bad:
            merges.append(hit)
    return {"lines": out, "merges": merges, "cov": cov}


def transverse(lines, detected, shape, corridor=14.0):
    """Where the detections actually are, across the truth.

    For a parallel pair this is the measurement that settles what happened,
    with no tolerance in it: one detection near the midline is a merge, two
    near +-s/2 is a resolution."""
    mid = 0.5 * (np.asarray(lines[0]["C"], float) + np.asarray(lines[1]["C"], float))
    t = np.gradient(mid, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    out = []
    for C in detected:
        C = np.asarray(C, float)
        d = np.hypot(*(C[:, None, :] - mid[None, :, :]).transpose(2, 0, 1))
        j = d.min(1)
        sel = j <= corridor
        if sel.sum() < 10:
            continue
        k = d[sel].argmin(1)
        off = ((C[sel] - mid[k]) * nrm[k]).sum(1)
        out.append((float(np.median(off)), float(np.std(off)), int(sel.sum())))
    return sorted(out, key=lambda r: r[0])
