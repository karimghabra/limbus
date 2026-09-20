"""Ridge pixels -> centreline polylines.

Two filters decide what counts as a vessel, and neither is a plain
threshold on brightness:

  length   a path opening keeps a pixel only if it lies on a path of at
           least L connected ridge pixels running in one direction (with a
           small gap tolerance). Texture blotches are round and short; a
           vessel is long and thin, so length separates them far better than
           contrast does.
  hysteresis  a faint ridge is kept when it connects to a strong long one.
           A vessel's evidence dips where another vessel crosses it or where
           it fades, and lowering the threshold everywhere would let texture
           in; requiring the connection keeps the dip without the texture.
"""
import cv2
import numpy as np

from . import evidence as ev
from . import pathopen
from . import skeleton as sk


def trace(keep, min_len=25.0, spur=8):
    """Binary mask -> smoothed centrelines, junctions split into edges."""
    thin = sk.prune(sk.thin(keep), np.full(keep.shape, 2.0, np.float32), spur)
    _, edges = sk.trace_graph(thin)
    out = []
    for e in edges:
        c = sk.smooth_path(e["path"], k=3)
        if len(c) > 1 and np.hypot(*np.diff(c, axis=0).T).sum() >= min_len:
            out.append(c)
    return out


def detect(z, ang, t_hi=3.0, L_hi=40, t_lo=1.5, L_lo=20, gap=2, min_len=25.0, reach=60,
           scale=None, A=None):
    """Centrelines of every ridge that is long and strong, plus the faint
    ridges connected to one.

    `reach` is how far (px, along the faint ridge) an extension may run from
    the strong ridge it hangs off. Without it, keeping a whole connected
    component lets texture chain: on a crop that is three-quarters bare
    sclera, whole-component hysteresis produced nearly as much centreline on
    a vessel-free texture surrogate as on the real image. A vessel's faint
    stretch is a gap of tens of pixels, not hundreds, so a bounded reach
    keeps the gap-filling and drops the chaining.
    """
    ridge = ev.nms(z, ang)
    if scale is not None:
        ridge = ev.drop_wall_echoes(ridge, scale, z > t_lo, A, ang)
    ridge = cv2.dilate(ridge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0   # close 1-px NMS breaks
    strong = pathopen.path_length(ridge & (z > t_hi), gap=gap) >= L_hi
    weak = (pathopen.path_length(ridge & (z > t_lo), gap=gap) >= L_lo) | strong
    weak_u = weak.astype(np.uint8)
    if reach:
        # geodesic growth: dilate from the strong ridges, but only through
        # weak-ridge pixels, `reach` steps at most
        grown = (strong & weak).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        for _ in range(int(reach)):
            nxt = cv2.dilate(grown, k) & weak_u
            if np.array_equal(nxt, grown):
                break
            grown = nxt
        keep = (grown > 0) | strong
    else:
        _, lab = cv2.connectedComponents(cv2.dilate(weak_u, np.ones((3, 3), np.uint8)), connectivity=8)
        hit = np.unique(lab[strong])
        keep = weak & np.isin(lab, hit[hit > 0])
    return trace(keep, min_len)
