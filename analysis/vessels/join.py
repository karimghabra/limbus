"""Joining ridge pieces that belong to one vessel.

A detected ridge stops where another vessel crosses it or where its contrast
dips below the threshold, so one vessel arrives as several pieces. A join
links two pieces into one identity when their free ENDS point at each other
across a short gap and the cheapest path between them through the evidence
stays on vessel-like pixels. No centreline is invented across the gap: the
join carries identity only, so nothing is reported as detected where no
ridge was found.
"""
import heapq

import numpy as np

N8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


def dijkstra(cost, start, targets, reach):
    """Cheapest paths from start over a window; returns (target, path) for the
    cheapest reachable target pixel, or None. targets: bool array (window)."""
    H, W = cost.shape
    sy, sx = start
    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, np.int32)
    dist[sy, sx] = 0.0
    heap = [(0.0, sy, sx)]
    best = None
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        if targets[y, x] and (y, x) != (sy, sx):
            best = (y, x)
            break
        if abs(y - sy) > reach or abs(x - sx) > reach:
            continue
        for dy, dx, w in N8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nd = d + w * 0.5 * (cost[y, x] + cost[ny, nx])
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[ny, nx] = (y, x)
                    heapq.heappush(heap, (nd, ny, nx))
    if best is None:
        return None
    path = [best]
    while path[-1] != (sy, sx):
        py, px = prev[path[-1]]
        if py < 0:
            return None
        path.append((int(py), int(px)))
    return best, np.array(path[::-1])


def _tangent(C, at_start, n=8):
    seg = C[:n] if at_start else C[-n:]
    d = seg[0] - seg[-1] if at_start else seg[-1] - seg[0]
    return d / max(np.hypot(*d), 1e-9)


def join_ends(cl, z, max_gap=50.0, max_turn=np.radians(40), min_z=1.0, max_tort=1.6):
    """Join ridge pieces end to end across a short gap.

    A vessel's ridge dips below every threshold where another vessel crosses
    it or where it fades; lowering the threshold everywhere lets texture in,
    while a join only fires between two ENDS that point at each other. The
    pair must be within max_gap with both tangents inside max_turn of the
    line joining them, and the cheapest path between them through the
    evidence (Dijkstra on 1/z) must keep mean evidence >= min_z without
    being tortuous - so the join follows a vessel instead of cutting across
    the background.

    Returns (groups, joins): groups of piece indices that are ONE vessel, and
    a record of each join. The pieces themselves are not merged and no
    centreline is invented across the gap: the join carries identity only,
    so nothing is reported as detected where no ridge was found.
    Deterministic: pairs are tried in order of gap length, ties by position.
    """
    H, W = z.shape
    cost = 1.0 / (0.3 + np.clip(z, 0, 8)).astype(np.float64)
    ends = []
    for i, C in enumerate(cl):
        ends.append((i, True, np.asarray(C[0], float), _tangent(C, True)))
        ends.append((i, False, np.asarray(C[-1], float), _tangent(C, False)))
    cand = []
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            ia, _, pa, ta = ends[a]
            ib, _, pb, tb = ends[b]
            if ia == ib:
                continue
            v = pb - pa
            g = float(np.hypot(*v))
            if not 1e-6 < g <= max_gap:
                continue
            u = v / g
            if np.arccos(np.clip(np.dot(ta, u), -1, 1)) > max_turn or                np.arccos(np.clip(np.dot(tb, -u), -1, 1)) > max_turn:
                continue
            cand.append((g, tuple(np.round(pa, 1)), a, b))
    cand.sort()
    members = {i: [i] for i in range(len(cl))}
    owner = list(range(len(cl)))
    used = set()

    def root(i):
        while owner[i] != i:
            i = owner[i]
        return i

    joins = []
    for g, _, a, b in cand:
        if a in used or b in used:
            continue
        ia, sa, pa, _ = ends[a]
        ib, sb, pb, _ = ends[b]
        ra, rb = root(ia), root(ib)
        if ra == rb:
            continue
        m = int(max_gap)
        y0, y1 = max(0, int(min(pa[1], pb[1])) - m), min(H, int(max(pa[1], pb[1])) + m + 1)
        x0, x1 = max(0, int(min(pa[0], pb[0])) - m), min(W, int(max(pa[0], pb[0])) + m + 1)
        tgt = np.zeros((y1 - y0, x1 - x0), bool)
        ty, tx = int(round(pb[1])) - y0, int(round(pb[0])) - x0
        if not (0 <= ty < tgt.shape[0] and 0 <= tx < tgt.shape[1]):
            continue
        tgt[ty, tx] = True
        res = dijkstra(cost[y0:y1, x0:x1], (int(round(pa[1])) - y0, int(round(pa[0])) - x0), tgt, 2 * m)
        if res is None:
            continue
        path = res[1][:, ::-1].astype(float) + [x0, y0]                # (x, y)
        L = float(np.hypot(*np.diff(path, axis=0).T).sum()) if len(path) > 1 else 0.0
        if L > max_tort * g + 4:
            continue
        zs = z[np.rint(path[:, 1]).astype(int), np.rint(path[:, 0]).astype(int)]
        if zs.mean() < min_z:
            continue
        members[ra] = members.pop(ra) + members.pop(rb)
        owner[rb] = ra
        used.add(a)
        used.add(b)
        joins.append({"a": ia, "b": ib, "gap": round(g, 1), "mean_z": round(float(zs.mean()), 2),
                      "path": path.round(1).tolist()})
    groups = [sorted(m) for m in members.values()]
    groups.sort(key=lambda m: m[0])
    return groups, joins
