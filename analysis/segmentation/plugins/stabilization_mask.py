"""Quick look: the stabilization's own vessel mask, thinned to centrelines.

The stabilization already found a vessel mask to register on
(consensus_mask.tif: pixels that most frames called vessel, METHODS.md §6).
This thins it to one-pixel centrelines, splits them at junctions, and reads
each vessel's width from the distance to the mask's edge. No fitting, so it
takes seconds — a quick look, and a check of the overlay itself. The mask
comes from a vesselness filter tuned for registration, so it merges vessels
closer than its filter scale and misses the faintest ones.
"""
import numpy as np

LABEL = "Stabilization mask (quick)"
DESCRIPTION = ("The vessel mask the stabilization registered on, thinned to centrelines, "
               "with widths from the distance to its edge. Seconds; a quick look, not a "
               "measurement.")
VERSION = "1"

MIN_AREA_PX = 200       # mask pieces smaller than this are noise
MIN_SPUR_PX = 15        # side branches shorter than this are thinning artefacts
SMOOTH_PX = 7           # moving average along each centreline (odd)

_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def segment(inputs):
    import cv2
    from skimage.morphology import skeletonize

    mask = inputs.vessel_mask() & inputs.valid()
    # the mask was computed at working scale and enlarged: smooth its steps
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5) > 0.5
    n, lab, stats, _ = cv2.connectedComponentsWithStats(soft.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA_PX
    soft = keep[lab]
    inputs.log(f"mask: {int(keep.sum())} pieces, {100 * soft.mean():.1f}% of the frame")

    dist = cv2.distanceTransform(soft.astype(np.uint8), cv2.DIST_L2, 5)
    paths, centres = _trace(skeletonize(soft))

    # drop spurs: short pieces with a free end, then recount each junction
    kept = [(p, ends) for p, ends in paths
            if not (None in ends and _length(p) < MIN_SPUR_PX)]
    degree = {}
    for _, ends in kept:
        for j in ends:
            if j is not None:
                degree[j] = degree.get(j, 0) + 1

    ov = inputs.overlay()
    for p, ends in kept:
        xy = p[:, ::-1].astype(np.float64)
        r = np.maximum(dist[p[:, 0], p[:, 1]] - 0.5, 0.5)
        # run into the junction centres, so the pieces meet
        if ends[0] is not None:
            xy = np.vstack([centres[ends[0]], xy])
            r = np.r_[r[0], r]
        if ends[1] is not None:
            xy = np.vstack([xy, centres[ends[1]]])
            r = np.r_[r, r[-1]]
        if len(xy) < 2:
            continue
        ov.add_vessel(_smooth(xy), radius=r, length_px=round(_length(xy), 1),
                      diameter_px=round(float(2 * r.mean()), 2))
    for j, deg in degree.items():
        if deg >= 3:
            x, y = centres[j]
            ov.add_junction(x, y, "bifurcation" if deg == 3 else "junction")
    ov.add_mask("stabilization vessel mask", soft)
    return ov


def _length(p):
    p = np.asarray(p, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def _smooth(xy):
    """Moving average along the path, ends fixed: takes out the staircase of
    a pixel path without moving where it starts and stops."""
    k = SMOOTH_PX
    if len(xy) <= k:
        return xy
    pad = np.vstack([np.repeat(xy[:1], k // 2, 0), xy, np.repeat(xy[-1:], k // 2, 0)])
    out = np.stack([np.convolve(pad[:, c], np.ones(k) / k, "valid") for c in (0, 1)], 1)
    out[0], out[-1] = xy[0], xy[-1]
    return out


def _trace(skel):
    """Split a one-pixel skeleton at its junctions. Returns ([(path, (j0, j1))],
    centres): each path an (m, 2) array of (y, x) pixels, j0/j1 the junction
    at each end (None for a free end), and centres[j] = (x, y) of junction j."""
    import cv2
    skel = skel.astype(np.uint8)
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    nb = cv2.filter2D(skel.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    junc = (skel > 0) & (nb >= 3)
    nj, jl, _, jc = cv2.connectedComponentsWithStats(junc.astype(np.uint8), connectivity=8)
    centres = {j: (float(jc[j, 0]), float(jc[j, 1])) for j in range(1, nj)}
    body = (skel > 0) & ~junc
    nbody, bl = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(bl)
    order = np.argsort(bl[ys, xs], kind="stable")
    ys, xs = ys[order], xs[order]
    starts = np.searchsorted(bl[ys, xs], np.arange(1, nbody))     # label 0 is background
    stops = np.r_[starts[1:], len(ys)]
    h, w = skel.shape
    out = []
    for a, b in zip(starts, stops):
        pts = set(zip(ys[a:b].tolist(), xs[a:b].tolist()))
        adj = {p: [(p[0] + dy, p[1] + dx) for dy, dx in _NB8 if (p[0] + dy, p[1] + dx) in pts]
               for p in pts}
        ends = [p for p, q in adj.items() if len(q) <= 1]
        cur = min(ends) if ends else min(pts)
        path, seen = [cur], {cur}
        while True:
            nxt = [q for q in adj[cur] if q not in seen]
            if not nxt:
                break
            # 4-neighbours first, so a diagonal doesn't skip a corner pixel
            cur = min(nxt, key=lambda q: (abs(q[0] - cur[0]) + abs(q[1] - cur[1]), q))
            seen.add(cur)
            path.append(cur)
        path = np.array(path, int)

        def touching(y, x):
            win = jl[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            return sorted(set(win[win > 0].tolist()))

        if len(path) == 1:        # one pixel between two junctions: it joins both
            j = touching(*path[0]) + [None, None]
            js = (j[0], j[1])
        else:
            j0, j1 = touching(*path[0]), touching(*path[-1])
            js = (j0[0] if j0 else None, j1[0] if j1 else None)
        out.append((path, js))
    return out, centres
