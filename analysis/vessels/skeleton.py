"""Skeleton and graph tracing: binary ridge mask -> centreline polylines.

Zhang-Suen thinning, spur pruning by the local radius, and a crossing-number
graph trace that splits the skeleton at junctions into edges.
Ported unchanged from the map builder used for the first vessel maps.
"""
import cv2
import numpy as np

N8 = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # orthogonal first


def thin(mask):
    """Zhang-Suen thinning, vectorised. Deterministic (fixed sub-iteration order)."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
            B = sum(p)
            seq = p + [p[0]]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            P2, P3, P4, P5, P6, P7, P8, P9 = p
            c1, c2 = (P2 * P4 * P6, P4 * P6 * P8) if step == 0 else (P2 * P4 * P8, P2 * P6 * P8)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def degree(sk):
    p = np.pad(sk.astype(np.uint8), 1)
    return sum(np.roll(np.roll(p, -dy, 0), -dx, 1) for dy, dx in N8)[1:-1, 1:-1] * sk


def branches(sk):
    """Crossing number: separate branches leaving each pixel. A staircase corner
    has 3 neighbours but 2 branches; counting neighbours would call it a junction."""
    p = np.pad(sk.astype(np.uint8), 1)
    ring = [np.roll(np.roll(p, -dy, 0), -dx, 1) for dy, dx in
            ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
    ring.append(ring[0])
    A = sum(((ring[k] == 0) & (ring[k + 1] == 1)).astype(np.uint8) for k in range(8))
    return A[1:-1, 1:-1] * sk


def prune(sk, rad_hint, spur):
    """Delete dead-end spurs shorter than spur + 2 x local radius (bumpy mask
    edges, not vessels): walk from each endpoint to the first junction."""
    H, W = sk.shape
    sk = sk.copy()
    for _ in range(10):
        A, deg = branches(sk), degree(sk)
        removed = 0
        for y, x in sorted(zip(*np.nonzero(sk & (deg == 1)))):
            path, cur, hit = [(y, x)], (y, x), False
            while len(path) < 200:
                nb = [(cur[0] + dy, cur[1] + dx) for dy, dx in N8
                      if 0 <= cur[0] + dy < H and 0 <= cur[1] + dx < W and sk[cur[0] + dy, cur[1] + dx]
                      and (cur[0] + dy, cur[1] + dx) not in path]
                if not nb:
                    break
                if A[nb[0]] >= 3:
                    hit = True
                    break
                path.append(nb[0])
                cur = nb[0]
            if hit and len(path) < spur + 2 * rad_hint[path[-1]]:
                for py, px in path:
                    sk[py, px] = False
                removed += 1
        if not removed:
            break
        sk = thin(sk)
    return sk


def trace_graph(sk):
    H, W = sk.shape
    deg = degree(sk)
    junc = sk & (branches(sk) >= 3)
    n, jl = cv2.connectedComponents(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8))
                                    & sk.astype(np.uint8), connectivity=8)
    node_of = np.full((H, W), -1, np.int32)
    nodes = []
    for k in range(1, n):
        ys, xs = np.nonzero(jl == k)
        node_of[ys, xs] = len(nodes)
        nodes.append({"type": "junction", "x": float(xs.mean()), "y": float(ys.mean()), "pixels": len(xs)})
    for y, x in zip(*np.nonzero(sk & (deg == 1))):
        if node_of[y, x] < 0:
            node_of[y, x] = len(nodes)
            nodes.append({"type": "end", "x": float(x), "y": float(y), "pixels": 1})
    visited = np.zeros((H, W), bool)
    edges = []

    def nbrs(y, x):
        for dy, dx in N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and sk[yy, xx]:
                yield yy, xx

    for (y, x) in sorted(zip(*np.nonzero(node_of >= 0))):
        a = node_of[y, x]
        for (yy, xx) in nbrs(y, x):
            if node_of[yy, xx] >= 0 or visited[yy, xx]:
                continue
            path = [(y, x), (yy, xx)]
            visited[yy, xx] = True
            cy, cx, end = yy, xx, -1
            while True:
                nxt = None
                for (ny, nx) in nbrs(cy, cx):
                    if (ny, nx) == path[-2]:
                        continue
                    if (node_of[ny, nx] >= 0 and node_of[ny, nx] != a) or (node_of[ny, nx] == a and len(path) > 3):
                        end = node_of[ny, nx]
                        path.append((ny, nx))
                        break
                    if not visited[ny, nx] and node_of[ny, nx] < 0:
                        nxt = (ny, nx)
                        break
                if end >= 0 or nxt is None:
                    break
                visited[nxt] = True
                path.append(nxt)
                cy, cx = nxt
            edges.append({"a": int(a), "b": int(end), "path": path})
    for y, x in sorted(zip(*np.nonzero(sk & ~visited & (node_of < 0)))):
        if visited[y, x]:
            continue
        path, cy, cx = [(y, x)], y, x
        visited[y, x] = True
        while True:
            nxt = next(((ny, nx) for ny, nx in nbrs(cy, cx) if not visited[ny, nx]), None)
            if nxt is None:
                break
            visited[nxt] = True
            path.append(nxt)
            cy, cx = nxt
        if len(path) > 10:
            k = len(nodes)
            nodes.append({"type": "loop", "x": float(x), "y": float(y), "pixels": 1})
            edges.append({"a": k, "b": k, "path": path})
    return nodes, edges


def smooth_path(path, k=5):
    p = np.array(path, float)[:, ::-1]
    if len(p) < 2 * k + 1:
        return p
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(np.pad(p[:, i], k, mode="edge"), ker, mode="valid") for i in (0, 1)], 1)
