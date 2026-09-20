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
from . import orientation as orient
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


def single_peaked(A, C, half=14.0, dip=0.12):
    """Is the absorbance across this line one vessel, or two side by side?

    A wide vessel absorbs most along its centre, so its cross-profile has a
    single maximum there. Two vessels running close together look like one
    wide vessel to a coarse scale, but the image between them is lighter than
    on either - the profile dips in the middle. The dip is measured relative
    to the profile's own height, so it does not depend on how dark the vessel
    is.
    """
    C = np.asarray(C, float)
    if len(C) < 5:
        return True
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-half, half + 0.5, 0.5, dtype=np.float32)
    sel = slice(None, None, max(1, len(C) // 40))
    px = (C[sel, None, 0] + n[sel, None, 0] * offs[None, :]).astype(np.float32)
    py = (C[sel, None, 1] + n[sel, None, 1] * offs[None, :]).astype(np.float32)
    prof = np.median(cv2.remap(A, px, py, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE), axis=0)
    c = len(offs) // 2
    base = min(float(np.median(prof[:6])), float(np.median(prof[-6:])))
    peak = float(prof.max()) - base
    if peak <= 0:
        return False
    centre = float(prof[max(c - 2, 0):c + 3].max()) - base
    return centre >= (1.0 - dip) * peak


def detect_staged(A, valid, cfg, log=None):
    """Large vessels first, then small ones, then reconcile.

    Each pass runs the same rules over its own band of scales, so a pass only
    ever sees structures its scales can resolve. The large pass is what keeps
    wide vessels whole and centred; the small pass is free to be as sensitive
    as it likes, because anything it finds lying inside a vessel the large
    pass already accounted for is removed later, when the model fit trims
    candidates inside an accepted lumen.

    Returns (candidates, z, scale): candidates are (large first) centrelines
    with the evidence maps of the small pass, which covers the wider range of
    thresholds used downstream.
    """
    out = []
    # Which filter provides the evidence. The Hessian has to describe two
    # crossing vessels with one number and loses the junction; the orientation
    # stack keeps a plane per direction and reads 1.00 where the Hessian reads
    # 0.12. It buys nothing for vessels running close together in PARALLEL -
    # measured, the same answer at every separation - so it is a crossing fix
    # and not a resolution fix.
    ridge_z = orient.ridge_z if getattr(cfg, "evidence", "hessian") == "orientation" else ev.ridge_z
    zl, angl, scl = ridge_z(A, valid, cfg.sigmas_large, with_angle=True, with_scale=True)
    large = detect(zl, angl, cfg.t_hi_large, cfg.L_hi_large, cfg.t_lo_large, cfg.L_lo_large,
                   cfg.gap, cfg.min_len_large, reach=cfg.reach)
    n_raw = len(large)
    large = [C for C in large if single_peaked(A, C)]
    zs, angs, scs = ridge_z(A, valid, cfg.sigmas_small, with_angle=True, with_scale=True)
    small = detect(zs, angs, cfg.t_hi, cfg.L_hi, cfg.t_lo, cfg.L_lo, cfg.gap, cfg.min_len,
                   reach=cfg.reach)
    if log:
        log(f"  large pass: {len(large)} centrelines ({n_raw - len(large)} dropped as "
            f"two vessels side by side); small pass: {len(small)}")
    return large, small, np.maximum(zl, zs), scl, scs


def detect(z, ang, t_hi=3.0, L_hi=40, t_lo=1.5, L_lo=20, gap=2, min_len=25.0, reach=60,
           scale=None, A=None, ridge=None):
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
    if ridge is None:
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
