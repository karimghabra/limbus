"""Vessel annotation of one stabilized image (METHODS.md §L3).

    image -> flat-field contrast -> multi-scale vesselness with orientation
          -> ridge (centreline) by non-maximum suppression across the vessel
          -> hysteresis along ridges -> thinning
          -> wide vessels re-traced by Steger line points
          -> spur pruning -> half-maximum vessel mask and widths

Centrelines are found directly as the ridge of the vesselness — the way
Canny finds edges — rather than by skeletonizing a thresholded mask: a mask
from a multi-scale filter is fattened by its large scales, merges vessels
that run close together, and its skeleton then cuts through the merged blob.

Run with the SAME constants on every image of a burst (the plain mean and
the lucky image alike), so any difference between their annotations comes
from the images, not from the annotator adapting to each one. The burst's
constants come from its mean image (`calibrate`).
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_holes, skeletonize

from .fuse import normalized_fill

_EIGHT = np.ones((3, 3), np.uint8)


def contrast_image(img, valid, params):
    """Relative darkening against the local background: 0 on bare tissue,
    negative on vessels. The background is a grey-level CLOSING (which
    removes dark structures narrower than the element) then a blur, so
    vessels don't pull their own background down and look fainter."""
    filled = normalized_fill(np.nan_to_num(img), valid, params.fill_sigma_px)
    f = params.background_downscale
    small = cv2.resize(filled, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    k = int(2 * round(params.background_radius_px / f) + 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (filled / np.maximum(bg, 1e-6) - 1.0).astype(np.float32)


def hessian(img, sigma, gamma, smoothed=None):
    """Hessian of the Gaussian-smoothed image, normalised by sigma^gamma.
    gamma = 2 makes a blob's response independent of its size (Frangi); a
    smaller gamma (Lindeberg's ridge normalisation) keeps wide scales from
    winning on the halo around thin vessels."""
    g = cv2.GaussianBlur(img, (0, 0), sigma) if smoothed is None else smoothed
    norm = sigma ** gamma
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * norm
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * norm
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * norm
    return dxx, dyy, dxy


def _eig(dxx, dyy, dxy):
    """Eigenvalues ordered |l1| <= |l2|, and the direction of the
    algebraically larger one — across a dark vessel, where curvature is
    strongly positive."""
    root = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
    a = 0.5 * (dxx + dyy - root)
    b = 0.5 * (dxx + dyy + root)
    swap = np.abs(a) > np.abs(b)
    theta = 0.5 * np.arctan2(2 * dxy, dxx - dyy)
    return np.where(swap, b, a), np.where(swap, a, b), theta


def vesselness(c_img, params, c):
    """Frangi vesselness of dark tubes, maximum over scales, with the scale
    that won at every pixel. Also the centreline strength: at each scale,
    only pixels that are line points at THAT scale (`_line_points`) keep
    their vesselness, and the maximum is taken over scales."""
    out = np.zeros(c_img.shape, np.float32)
    best = np.zeros(c_img.shape, np.float32)
    across = np.zeros(c_img.shape, np.float32)
    line = np.zeros(c_img.shape, np.float32)
    wide = np.zeros(c_img.shape, np.float32)
    for s in params.sigmas_px:
        g = cv2.GaussianBlur(c_img, (0, 0), s)
        l1, l2, th = _eig(*hessian(c_img, s, params.scale_gamma, g))
        rb = l1 / (l2 + np.where(l2 >= 0, 1e-9, -1e-9))
        s2 = l1 ** 2 + l2 ** 2
        v = np.exp(-(rb ** 2) / (2 * params.frangi_beta ** 2)) * (1.0 - np.exp(-s2 / (2 * c ** 2)))
        v[l2 <= 0] = 0.0
        better = v > out
        out[better] = v[better]
        best[better] = s
        across[better] = th[better]
        lp = np.where(_line_points(g, l2 / s ** params.scale_gamma, th), v, 0)
        np.maximum(line, lp, out=line)
        if s >= params.wide_sigma_px:
            np.maximum(wide, lp, out=wide)
    return out, best, across, line, wide


def _line_points(g, l2, theta):
    """Steger's line points (IEEE PAMI 1998): where the first derivative
    across the vessel crosses zero inside the pixel. Across a profile that is
    darkest in the middle the zero crossing is the centreline, at any width.
    Taking the maximum of vesselness over scales and then its ridge fails for
    wide, flat-bottomed vessels: small scales answer most strongly at the two
    inner corners, and the ridge of the maximum runs along the walls. At a
    corner the first derivative (the wall's slope) is not zero, so it is not
    a line point at any scale.

    g: the image smoothed at this scale; l2: the across-vessel second
    derivative (Sobel units, un-normalised); theta: the across direction."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0      # true first derivatives
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx, ny = np.cos(theta), np.sin(theta)
    d1 = gx * nx + gy * ny
    d2 = l2 / 4.0                                            # Sobel 2nd-derivative gain
    t = -d1 / np.where(d2 > 1e-12, d2, np.inf)
    return (d2 > 1e-12) & (np.abs(t * nx) <= 0.5) & (np.abs(t * ny) <= 0.5)


def ridges(v, across):
    """Non-maximum suppression across the vessel: keep a pixel only if its
    vesselness is at least that of both neighbours one pixel away along the
    across-vessel direction (sampled bilinearly)."""
    h, w = v.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cx, cy = np.cos(across).astype(np.float32), np.sin(across).astype(np.float32)
    plus = cv2.remap(v, X + cx, Y + cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    minus = cv2.remap(v, X - cx, Y - cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (v > 0) & (v >= plus) & (v >= minus)


def calibrate(c_img, valid, params):
    """Frangi's c for a burst: half the given percentile of the Hessian norm
    of the reference (mean) image at the middle scale — as the stabilization
    stage does (stabilize §5), but on the contrast image."""
    mid = params.sigmas_px[len(params.sigmas_px) // 2]
    l1, l2, _ = _eig(*hessian(c_img, mid, params.scale_gamma))
    s = np.sqrt(l1 ** 2 + l2 ** 2)[valid]
    return 0.5 * float(np.percentile(s, params.contrast_percentile)) + 1e-9


def _neighbours(skel):
    s = skel.astype(np.uint8)
    return cv2.filter2D(s, -1, _EIGHT, borderType=cv2.BORDER_CONSTANT) - s


def prune_spurs(skel, max_len):
    """Remove terminal branches shorter than max_len px: the short hairs a
    skeleton grows on every bump of a mask's outline. Walks from each
    endpoint to the first junction; repeated twice so hairs on hairs go too."""
    skel = skel.copy()
    h, w = skel.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for _ in range(2):
        nb = _neighbours(skel)
        ends = np.argwhere(skel & (nb == 1))
        removed = False
        for y, x in ends:
            path = [(y, x)]
            prev = None
            cy, cx = y, x
            hit_junction = False
            while len(path) <= max_len:
                nxt = None
                for dy, dx in offs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev \
                            and (ny, nx) not in path:
                        if nb[ny, nx] >= 3:
                            hit_junction = True
                            break
                        nxt = (ny, nx)
                        break
                if hit_junction or nxt is None:
                    break
                prev = (cy, cx)
                cy, cx = nxt
                path.append(nxt)
            # only hairs attached to a junction; an isolated short piece is
            # handled by the minimum-length rule instead
            if hit_junction and len(path) < max_len:
                for py, px in path:
                    skel[py, px] = False
                removed = True
        if not removed:
            break
    return skel


def annotate(img, params, c, valid=None):
    """Annotate one image with the burst's Frangi constant c (`calibrate`).
    Returns a dict of maps and graph statistics."""
    if valid is None:
        valid = np.isfinite(img)
    c_img = contrast_image(img, valid, params)
    v, scale, across, line, wide = vesselness(c_img, params, c)
    for m in (v, line, wide):
        m[~valid] = 0
    if params.centreline == "steger":
        skel = _trace(cv2.dilate(line, _EIGHT), params)
    else:
        skel = _trace(np.where(cv2.dilate(ridges(v, across).astype(np.uint8), _EIGHT) > 0,
                               v, 0), params)
        if params.centreline == "hybrid":
            skel = _replace_wide(skel, wide, c_img, scale, params)
    skel = prune_spurs(skel, params.spur_px)
    # drop centreline pieces too short to be a vessel segment
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    if n:
        size = ndi.sum(skel, lab, index=np.arange(1, n + 1))
        skel = np.concatenate([[False], size >= params.min_centreline_px])[lab]
    mask, width = half_max_mask(c_img, skel, scale)
    mask &= valid
    return {"contrast": c_img, "vesselness": v, "scale": scale, "mask": mask,
            "skeleton": skel, "width": width, "stats": graph_stats(skel, mask, valid)}


def _trace(strength, params):
    """Hysteresis on a centreline-strength map, then thinning: a faint
    stretch survives if it connects to a strong one, as in Canny."""
    cand = apply_hysteresis_threshold(strength, params.low_threshold, params.high_threshold)
    cand &= strength >= params.low_threshold
    # small holes would thin into little loops
    return skeletonize(remove_small_holes(cand, max_size=params.max_hole_px))


def _replace_wide(skel, wide, c_img, scale, params):
    """Wide vessels get their centreline from Steger's line points (§L3).

    Ridges of the scale maximum run along the WALLS of wide, flat-bottomed
    vessels (see `_line_points`). Wide vessels are found geometrically: the
    part of the half-maximum mask that survives an opening by a disk of
    radius wide_radius_px, i.e. at least twice that wide. Inside those
    bodies the ridge pixels are replaced by the line points of the wide
    scales, falling back to the body's medial axis where those are missing.
    """
    sm = cv2.GaussianBlur(c_img, (0, 0), 2.0)
    deep = float(np.percentile(sm, params.wide_depth_percentile))
    dark = sm <= params.wide_depth_fraction * min(deep, 0.0)
    r = int(params.wide_radius_px)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
    if not body.any():
        return skel
    centre = _trace(cv2.dilate(np.where(body, wide, 0), _EIGHT), params) & body
    if params.wide_centre == "medial" or not centre.any():
        centre = skeletonize(body)
    return (skel & ~body) | centre


def half_max_mask(c_img, skel, scale):
    """Vessel = pixels darker than half the depth of their nearest centreline
    point, within reach of it: the full width at half maximum, the usual
    definition of a vessel's diameter in an intensity profile."""
    if not skel.any():
        return np.zeros_like(skel), np.zeros(skel.shape, np.float32)
    dist, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    depth = np.minimum(c_img[iy, ix], 0)
    reach = 3.0 * scale[iy, ix] + 1.5
    mask = (c_img <= 0.5 * depth) & (dist <= reach) & (depth < 0)
    mask |= skel
    # keep only the part connected to a centreline
    lab, _ = ndi.label(mask, structure=np.ones((3, 3)))
    mask = np.isin(lab, np.unique(lab[skel]))
    width = np.where(skel, 2 * ndi.distance_transform_edt(mask) - 1, 0).astype(np.float32)
    return mask, width


def graph_stats(skel, mask, valid):
    nb = _neighbours(skel)
    ends = int(np.count_nonzero(skel & (nb == 1)))
    junction = skel & (nb >= 3)
    # adjacent junction pixels are one branch point
    _, n_branch = ndi.label(junction, structure=np.ones((3, 3)))
    _, n_comp = ndi.label(skel, structure=np.ones((3, 3)))
    h, v, d1, d2 = _pair_length(skel)
    length = int(h.sum() + v.sum()) + np.sqrt(2) * int(d1.sum() + d2.sum())
    area = max(int(valid.sum()), 1)
    mp = 1e6 / area                                  # per megapixel of valid field
    return {
        "centreline_px_per_mpx": float(length * mp),
        "endpoints_per_mpx": float(ends * mp),
        "branch_points_per_mpx": float(n_branch * mp),
        "components_per_mpx": float(n_comp * mp),
        "vessel_fraction": float(mask[valid].mean()) if valid.any() else 0.0,
        # length per piece: longer = less fragmented
        "mean_component_length_px": float(length / max(n_comp, 1)),
    }


def _pair_length(s):
    """Centreline length of a 1-px skeleton: orthogonal steps count 1,
    diagonal steps sqrt(2) (diagonals that shortcut an L are not counted)."""
    s = s.astype(bool)
    h = s[:, :-1] & s[:, 1:]
    v = s[:-1] & s[1:]
    d1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    d2 = s[:-1, 1:] & s[1:, :-1] & ~s[:-1, :-1] & ~s[1:, 1:]
    return h, v, d1, d2


def segment_table(skel, width):
    """One row per vessel segment: the centreline between junctions and/or
    ends. Length along the centreline, straight-line (chord) length between
    its ends, tortuosity = length / chord, and the half-maximum width
    (median and 10th-90th percentile along it). Pixels in full resolution."""
    nb = _neighbours(skel)
    junction = skel & (nb >= 3)
    pieces = skel & ~cv2.dilate(junction.astype(np.uint8), _EIGHT).astype(bool)
    lab, n = ndi.label(pieces, structure=np.ones((3, 3)))
    if not n:
        return []
    h, v, d1, d2 = _pair_length(pieces)
    length = np.zeros(n + 1)
    for pair, first, step in ((h, lab[:, :-1], 1.0), (v, lab[:-1], 1.0),
                              (d1, lab[:-1, :-1], np.sqrt(2)), (d2, lab[:-1, 1:], np.sqrt(2))):
        np.add.at(length, first[pair], step)
    ends_map = pieces & (_neighbours(pieces) <= 1)
    rows = []
    objs = ndi.find_objects(lab)
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = lab[sl] == k
        ys, xs = np.nonzero(sub)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        e = np.flatnonzero(ends_map[ys, xs])
        if e.size >= 2:
            a, b = e[0], e[-1]
            chord = float(np.hypot(ys[a] - ys[b], xs[a] - xs[b]))
        else:                                   # a closed loop
            a = b = 0
            chord = 0.0
        L = float(length[k]) + 1.0              # + the end pixel itself
        wv = width[ys, xs]
        wv = wv[wv > 0]
        rows.append({
            "segment": len(rows) + 1,
            "length_px": round(L, 2),
            "chord_px": round(chord, 2),
            "tortuosity": round(L / chord, 4) if chord > 0 else "",
            "width_median_px": round(float(np.median(wv)), 2) if wv.size else "",
            "width_p10_px": round(float(np.percentile(wv, 10)), 2) if wv.size else "",
            "width_p90_px": round(float(np.percentile(wv, 90)), 2) if wv.size else "",
            "x_start": int(xs[a]), "y_start": int(ys[a]),
            "x_end": int(xs[b]), "y_end": int(ys[b]),
            "x_centre": round(float(xs.mean()), 1), "y_centre": round(float(ys.mean()), 1),
        })
    return rows
