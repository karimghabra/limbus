"""Path openings: keep what is LONG, not what is strong.

For a binary map B, the path opening measures, at every pixel, the length of
the longest path through B that passes through it and stays within a cone of
directions (Talbot & Appleton 2007; Hendriks 2010). Four cones (N-S, E-W and
the two diagonals) cover every orientation; a curving vessel stays within
one cone over long stretches. With GAP > 0 the path may cross up to GAP
pixels outside B ("incomplete" / robust path openings), so a faint vessel
that dips below threshold for a pixel or two isn't cut.

A faint vessel thresholded low is a long path; scleral texture thresholded
equally low is a scatter of short blobs. Length is the discriminating
feature, not contrast.

Dynamic programming: forward pass lambda_f(p) = [p in B] * (1 + max over the
3 predecessors in the cone), backward pass likewise, path length through p =
lambda_f + lambda_b - 1. With gaps, one array per gap count.
"""
import numpy as np

# each cone: successor offsets (dy, dx) - three neighbours spanning 90 degrees
CONES = {
    "S": [(1, -1), (1, 0), (1, 1)],
    "E": [(-1, 1), (0, 1), (1, 1)],
    "SE": [(1, 0), (1, 1), (0, 1)],
    "SW": [(1, 0), (1, -1), (0, -1)],
}


def _shift(a, dy, dx, fill=0):
    """out[y, x] = a[y - dy, x - dx] (value at the predecessor)."""
    H, W = a.shape
    out = np.full_like(a, fill)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]
    return out


def _pass(B, steps, gap):
    """One directional pass along a cone, processed in wavefront order."""
    H, W = B.shape
    # order pixels so every predecessor is processed first: for cones whose
    # steps all have dy >= 0 and include dy > 0, go row by row; the E cone
    # goes column by column; diagonal cones row by row with in-row carry.
    lam = [np.zeros((H, W), np.int32) for _ in range(gap + 1)]
    if all(dy >= 0 for dy, _ in steps) and not any(dy == 0 for dy, _ in steps):
        rows = range(H)
        for y in rows:
            for g in range(gap + 1):
                prev = np.zeros(W, np.int32)
                prev_g = np.zeros(W, np.int32)
                if y > 0:
                    for dy, dx in steps:
                        pr = np.roll(lam[g][y - dy], dx)
                        prg = np.roll(lam[g - 1][y - dy], dx) if g > 0 else None
                        if dx > 0:
                            pr[:dx] = 0
                            if prg is not None: prg[:dx] = 0
                        elif dx < 0:
                            pr[dx:] = 0
                            if prg is not None: prg[dx:] = 0
                        prev = np.maximum(prev, pr)
                        if prg is not None:
                            prev_g = np.maximum(prev_g, prg)
                on = B[y]
                lam[g][y] = np.where(on, 1 + prev, (1 + prev_g) * (prev_g > 0) if g > 0 else 0)
        return lam
    raise ValueError("use transpose/flip helpers")


def path_length(B, gap=0):
    """Max over the four cones of the longest cone-path through each pixel."""
    B = B.astype(bool)
    best = np.zeros(B.shape, np.int32)
    # S cone: rows top->bottom (forward) and bottom->top (backward)
    for Bt, back in ((B, lambda a: a), (B.T, lambda a: a.T)):          # S on B, E via transpose
        f = _pass(Bt, CONES["S"], gap)
        b = _pass(Bt[::-1], CONES["S"], gap)
        L = np.maximum.reduce([f[g] + b[h][::-1] - 1 for g in range(gap + 1) for h in range(gap + 1 - g)])
        best = np.maximum(best, back(np.where(Bt, L, 0)))
    # diagonal cones via a 45-degree shear: map each anti-diagonal/diagonal to rows
    for flip in (False, True):
        Bf = B[:, ::-1] if flip else B
        H, W = Bf.shape
        # SE cone successors (1,0),(1,1),(0,1): process by d = y + x
        S = np.zeros((H + W - 1, W), bool)
        ys, xs = np.nonzero(np.ones_like(Bf))
        S[ys + xs, xs] = Bf[ys, xs]
        # in sheared coords a step (1,0) -> (d+1, x), (0,1) -> (d+1, x+1), (1,1) -> (d+2, x+1)
        # approximate the cone with the two unit-d steps (d+1, x) and (d+1, x+1)
        f = _pass(S, [(1, 0), (1, 1)], gap)
        b = _pass(S[::-1], [(1, 0), (1, -1)], gap)
        L = np.maximum.reduce([f[g] + b[h][::-1] - 1 for g in range(gap + 1) for h in range(gap + 1 - g)])
        Ld = np.zeros((H, W), np.int32)
        Ld[ys, xs] = np.where(S[ys + xs, xs], L[ys + xs, xs], 0)
        best = np.maximum(best, Ld[:, ::-1] if flip else Ld)
    return best
