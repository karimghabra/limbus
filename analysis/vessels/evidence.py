"""Ridge evidence: how vessel-like each pixel is, and which way the ridge runs.

The Hessian of the blurred transmission measures curvature; on a dark line
the curvature across the line is large and positive while along it is small,
so l2 - |l1| of the Hessian eigenvalues responds to lines and not to edges
or blobs. Each scale is put in units of ITS OWN robust spread (median and
MAD over the valid pixels) before the scales are combined by a maximum:
coarse scales respond strongly to the sclera's texture blotches, and without
this normalisation they would drown the thin vessels that the fine scales
see. The value is therefore a robust z score, comparable between images.

Small sigmas (1-3 px) are what small vessels need: their measured width is
5-8 px FWHM at this magnification. Large ones are just as necessary: a vessel
whose radius is much larger than the scale gives a ridge response at each of
its walls and nothing at its centre, so without coarse scales the widest
vessels are traced as two parallel edges - or, past about 12 px radius, not
traced down the middle at all.
"""
import cv2
import numpy as np

# Scales must cover the vessels present, not just the small ones: the Hessian
# of a vessel much wider than the scale responds at its WALLS, not its centre,
# so a 24 px-wide vessel detected only at sigma <= 3 comes back as two edge
# lines with nothing down the middle. These scales run from the thinnest
# vessel resolvable to the widest seen here (radius ~12 px).
SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.5, 6.5, 9.0, 12.0)

# Scale-normalisation exponent. The ridge response has to be normalised across
# scales before they can be compared, and the exponent decides which scale wins
# on a given vessel: with gamma = 1 the response keeps growing with sigma and
# the coarsest scale wins on everything, which puts a thin vessel's evidence at
# a scale that cannot see it. Lindeberg's gamma = 3/4 for ridges gives a real
# maximum at the scale matching the vessel's width.
GAMMA = 0.75


def ridge_z(A, valid, sigmas=SIGMAS, with_angle=False, with_scale=False):
    """Robust-z ridge strength of the absorbance A (vessels are positive).

    With with_angle, also returns the angle of the Hessian eigenvector ACROSS
    the ridge at the winning scale, for non-maximum suppression; with
    with_scale, the winning scale itself, which says how wide the thing that
    responded is."""
    T = np.exp(-A).astype(np.float32)          # ridges are dark in transmission
    best = np.full(A.shape, -np.inf, np.float32)      # robust z at the selected scale
    best_resp = np.full(A.shape, -np.inf, np.float32)  # what the selection is made on
    ang = np.zeros(A.shape, np.float32)
    scale = np.full(A.shape, float(min(sigmas)), np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(T, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) / 4
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) / 4
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4
        tr, dif = dxx + dyy, np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l2, l1 = 0.5 * (tr + dif), 0.5 * (tr - dif)
        v = s ** (2 * GAMMA) * np.where(l2 > 0, l2 - np.abs(l1), 0)
        m = np.median(v[valid])
        mad = 1.4826 * np.median(np.abs(v[valid] - m))
        z = ((v - m) / max(mad, 1e-12)).astype(np.float32)
        # The scale is chosen by the gamma-normalised response itself, which is
        # what says how wide the structure is. Choosing by the robust z instead
        # let a fine scale win on a wide vessel - its own spread is small, so
        # its z comes out large - and the ridge was then traced along the
        # vessel's walls. The z reported is the selected scale's, so texture is
        # still judged against the spread of the scale that saw it.
        better = v > best_resp
        best_resp = np.where(better, v, best_resp)
        best = np.where(better, z, best)
        if with_angle:
            ang = np.where(better, 0.5 * np.arctan2(2 * dxy, dxx - dyy), ang)
        scale = np.where(better, np.float32(s), scale)
    best[~valid] = 0
    out = (best,)
    if with_angle:
        out += (ang,)
    if with_scale:
        out += (scale,)
    return out if len(out) > 1 else best


def drop_wall_echoes(ridge, scale, significant, A=None, ang=None, factor=1.5, reach=1.3,
                     parallel_deg=25.0):
    """Remove the ridge lines a wide vessel throws along its own walls.

    A vessel much wider than the scale looking at it responds at each wall, so
    a wide vessel comes back as its centreline (found at a coarse scale) plus
    two fine-scale lines a radius either side. Those lie INSIDE the vessel the
    coarse scale found. A ridge pixel is therefore dropped when a much coarser
    ridge (at least `factor` times its scale) passes within that coarser
    ridge's own width: two real vessels side by side respond at similar
    scales, so neither suppresses the other.
    """
    import cv2 as _cv2
    keep = ridge.copy()
    base = ridge & significant
    if not base.any():
        return keep
    for s in np.unique(scale[base])[::-1]:
        coarse = base & (scale >= s)
        if not coarse.any():
            continue
        k = int(max(1, round(reach * s)))
        owned = _cv2.dilate(coarse.astype(np.uint8),
                            _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)) > 0
        drop = owned & base & (scale * factor <= s) & ~coarse
        if drop.any() and (A is not None or ang is not None):
            _, lab = _cv2.distanceTransformWithLabels(
                (~coarse).astype(np.uint8), _cv2.DIST_L2, 5,
                labelType=_cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(coarse)
            own = lab[ys, xs]                         # each coarse pixel's own label
            if A is not None:                         # the centre must be at least as dark
                table = np.zeros(int(lab.max()) + 1, np.float32)
                table[own] = A[ys, xs]
                drop &= table[lab] >= A
            if ang is not None:                       # and the two must run parallel
                ta = np.zeros(int(lab.max()) + 1, np.float32)
                ta[own] = ang[ys, xs]
                d = np.abs(np.angle(np.exp(1j * 2 * (ta[lab] - ang))) / 2)
                drop &= d <= np.radians(parallel_deg)
        keep &= ~drop
    return keep


def ridge_mask(A, valid, sigmas=SIGMAS, t_lo=1.5):
    """Ridge pixels, with the suppression done AT EACH SCALE.

    Non-maximum suppression has to use the scale that saw the structure. A
    vessel much wider than the scale gives a broad response covering its whole
    width, and suppressing across the ridge with that scale's orientation
    leaves a band rather than a line - whose skeleton is a loop around the
    vessel, not a centreline. Running the suppression per scale and taking the
    union gives one line per scale that responded: the centreline from the
    matching scale, and wall lines from finer ones, which the model fit then
    resolves by trimming whatever lies inside the accepted lumen.

    Returns (ridge, z, scale): the mask, the robust-z strength (max over
    scales) and the winning scale.
    """
    import cv2 as _cv2
    H, W = A.shape
    T = np.exp(-A).astype(np.float32)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    ridge = np.zeros(A.shape, bool)
    best = np.full(A.shape, -np.inf, np.float32)
    scale = np.full(A.shape, float(min(sigmas)), np.float32)
    for s in sigmas:
        g = _cv2.GaussianBlur(T, (0, 0), s)
        dxx = _cv2.Sobel(g, _cv2.CV_32F, 2, 0, ksize=3) / 4
        dyy = _cv2.Sobel(g, _cv2.CV_32F, 0, 2, ksize=3) / 4
        dxy = _cv2.Sobel(g, _cv2.CV_32F, 1, 1, ksize=3) / 4
        tr, dif = dxx + dyy, np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l2, l1 = 0.5 * (tr + dif), 0.5 * (tr - dif)
        v = s ** (2 * GAMMA) * np.where(l2 > 0, l2 - np.abs(l1), 0)
        m = np.median(v[valid])
        mad = 1.4826 * np.median(np.abs(v[valid] - m))
        z = ((v - m) / max(mad, 1e-12)).astype(np.float32)
        a = (0.5 * np.arctan2(2 * dxy, dxx - dyy)).astype(np.float32)
        nx, ny = np.cos(a), np.sin(a)
        z1 = _cv2.remap(z, X + nx, Y + ny, _cv2.INTER_LINEAR)
        z2 = _cv2.remap(z, X - nx, Y - ny, _cv2.INTER_LINEAR)
        peak = (z >= z1) & (z >= z2) & (z > t_lo) & valid
        ridge |= peak
        better = z > best
        scale = np.where(better, np.float32(s), scale)
        np.maximum(best, z, out=best)
    best[~valid] = 0
    return ridge, best, scale


def nms(z, ang):
    """Keep only pixels that are a maximum of z along the direction ACROSS the
    ridge: two vessels running side by side then keep separate centrelines
    instead of merging into one thresholded blob."""
    H, W = z.shape
    nx, ny = np.cos(ang), np.sin(ang)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z1 = cv2.remap(z, X + nx, Y + ny, cv2.INTER_LINEAR)
    z2 = cv2.remap(z, X - nx, Y - ny, cv2.INTER_LINEAR)
    return (z >= z1) & (z >= z2)
