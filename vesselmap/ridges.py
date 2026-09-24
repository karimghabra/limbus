"""Multi-scale ridge detection used to *propose* vessel splines.

Proposals come from the Hessian of the (residual) log image at a range of
scales.  Nothing here decides what is a vessel: every proposal is turned
into a spline and must earn its place by improving the global score of the
whole network (see fit.py).  The detector therefore errs on the generous
side.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

# peak of sigma^2 * d2/dx2 [G_sigma * gaussian line of std w] at the best scale
# sigma = sqrt(2) w, relative to the line's peak amplitude (2 / 3^1.5)
GAUSS_LINE_PEAK = 2.0 / 3.0 ** 1.5


def hessian_eig(V: np.ndarray, sigma: float):
    """Eigen-decomposition of the Gaussian Hessian at scale sigma.

    Returns (l1, l2, nx, ny) with l1 <= l2 and (nx, ny) the unit eigenvector of
    l1, i.e. the direction ACROSS a bright ridge.
    """
    V = V.astype(np.float32, copy=False)
    kw = dict(sigma=sigma, mode="reflect", truncate=3.5)
    hxx = ndi.gaussian_filter(V, order=(0, 2), **kw)
    hyy = ndi.gaussian_filter(V, order=(2, 0), **kw)
    hxy = ndi.gaussian_filter(V, order=(1, 1), **kw)
    tr2 = 0.5 * (hxx + hyy)
    disc = np.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2)
    l1 = tr2 - disc
    l2 = tr2 + disc
    # eigenvector for l1: (hxy, l1 - hxx) or (l1 - hyy, hxy)
    ax = np.where(np.abs(hxx - l1) > np.abs(hyy - l1), hxy, l1 - hyy)
    ay = np.where(np.abs(hxx - l1) > np.abs(hyy - l1), l1 - hxx, hxy)
    nrm = np.sqrt(ax ** 2 + ay ** 2) + 1e-12
    return l1, l2, (ax / nrm).astype(np.float32), (ay / nrm).astype(np.float32)


@functools.lru_cache(maxsize=64)
def noise_l1_std(sigma: float) -> float:
    """Std of sigma^2 * max(-l1, 0)'s driving term for unit white noise.

    Measured once by simulation (cached), which is more faithful than the
    analytic value for the discrete, truncated filters actually used.
    """
    rng = np.random.default_rng(12345)
    n = int(max(256, 24 * sigma))
    z = rng.standard_normal((n, n)).astype(np.float32)
    l1, _, _, _ = hessian_eig(z, sigma)
    c = int(4 * sigma)
    return float(np.std(sigma ** 2 * l1[c:-c, c:-c]))


@dataclass
class Seed:
    """A proposed centreline with rough profile estimates along it."""
    xy: np.ndarray            # (m, 2) float, x then y, ordered
    scale: np.ndarray         # (m,) detection scale at each point
    amp: np.ndarray           # (m,) estimated peak depth (log units)
    z: np.ndarray             # (m,) detection z-score
    end_junction: list = field(default_factory=lambda: [None, None])

    @property
    def length(self) -> float:
        return float(np.sqrt((np.diff(self.xy, axis=0) ** 2).sum(1)).sum())


def _tile_rms(q: np.ndarray, tile: int) -> np.ndarray:
    """Smooth map of the robust RMS of q over tiles (q >= 0)."""
    import cv2
    h, w = q.shape
    ny, nx = max(1, h // tile), max(1, w // tile)
    est = np.zeros((ny, nx), np.float32)
    for iy in range(ny):
        for ix in range(nx):
            blk = q[iy * h // ny:(iy + 1) * h // ny, ix * w // nx:(ix + 1) * w // nx]
            # RMS of the valley statistic, robust to the odd strong valley
            v = np.sort(blk.ravel())
            v = v[: max(1, int(0.98 * len(v)))]
            est[iy, ix] = np.sqrt(np.mean(v.astype(np.float64) ** 2))
    est = ndi.median_filter(est, size=3, mode="nearest")
    return cv2.resize(est, (w, h), interpolation=cv2.INTER_CUBIC)


def ridge_maps(V: np.ndarray, noise: np.ndarray, scales, valid=None, aniso=1.0,
               texture_null=True):
    """Scale-selected ridge strength, its z-score, the scale and the normal.

    Ridge strength (bright ridge in V):   rho  = s^2 * max(0, -l1 - aniso*|l2|)
    Valley strength (opposite polarity):  rho' = s^2 * max(0,  l2 - aniso*|l1|)

    The anisotropy term suppresses blobs (background lumps, dust).  The null
    against which rho is tested is the larger of (a) white sensor noise at
    that scale and (b) the local RMS of the valley strength: background
    texture produces valleys as often as ridges, vessels only ridges.
    """
    best_z = np.zeros(V.shape, np.float32)
    best_rho = np.zeros(V.shape, np.float32)
    best_s = np.zeros(V.shape, np.float32)
    nx = np.zeros(V.shape, np.float32)
    ny = np.zeros(V.shape, np.float32)
    vv = V if valid is None else np.where(valid, V, ndi.median_filter(V, 5))
    for s in scales:
        l1, l2, ax, ay = hessian_eig(vv, s)
        rho = (s ** 2) * np.maximum(-l1 - aniso * np.abs(l2), 0.0)
        null = noise_l1_std(float(s)) * noise
        if texture_null:
            val = (s ** 2) * np.maximum(l2 - aniso * np.abs(l1), 0.0)
            tile = int(max(64, 10 * s))
            null = np.maximum(null, _tile_rms(val, tile))
        z = rho / null
        # the scale is selected by the scale-normalised strength (Lindeberg);
        # the z-score at that scale is what gets thresholded
        upd = rho > best_rho
        best_z[upd] = z[upd]
        best_rho[upd] = rho[upd]
        best_s[upd] = s
        nx[upd] = ax[upd]
        ny[upd] = ay[upd]
    if valid is not None:
        best_z[~valid] = 0
        best_rho[~valid] = 0
    return best_rho, best_z, best_s, nx, ny


@functools.lru_cache(maxsize=256)
def oriented_kernel(sigma: float, theta: float, elong: float):
    """Negative second derivative ACROSS direction theta of an anisotropic
    Gaussian (std sigma across, elong*sigma along), scale-normalised by
    sigma^2 and zero-mean.  theta is the direction of the line (along)."""
    sa = elong * sigma
    rad = int(math.ceil(3.0 * max(sigma, sa)))
    y, x = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float64)
    c, s_ = math.cos(theta), math.sin(theta)
    u = -s_ * x + c * y          # across
    v = c * x + s_ * y           # along
    g = np.exp(-0.5 * (u / sigma) ** 2 - 0.5 * (v / sa) ** 2)
    g /= g.sum()
    k = -(u ** 2 / sigma ** 4 - 1.0 / sigma ** 2) * g * sigma ** 2
    k -= k.mean()
    return k.astype(np.float32)


def oriented_ridge_maps(V: np.ndarray, noise: np.ndarray, scales, valid=None,
                        n_orient=12, elong=3.0, aniso=1.0, texture_null=True):
    """Like ridge_maps but with elongated oriented filters, which integrate
    along the vessel: a thin line gains ~sqrt(elong) in SNR over isotropic
    filters, while noise speckle and blobs do not.  Used for the fine bands
    where faint capillaries sit near the noise floor."""
    import cv2
    vv = V if valid is None else np.where(valid, V, ndi.median_filter(V, 5))
    vv = vv.astype(np.float32)
    best_z = np.zeros(V.shape, np.float32)
    best_rho = np.zeros(V.shape, np.float32)
    best_s = np.zeros(V.shape, np.float32)
    nx = np.zeros(V.shape, np.float32)
    ny = np.zeros(V.shape, np.float32)
    thetas = np.arange(n_orient) * math.pi / n_orient
    for s in scales:
        R = np.stack([cv2.filter2D(vv, cv2.CV_32F, oriented_kernel(float(s), float(t), elong),
                                   borderType=cv2.BORDER_REFLECT) for t in thetas])
        i = np.argmax(R, axis=0)
        Rmax = np.take_along_axis(R, i[None], 0)[0]
        Rperp = np.take_along_axis(R, ((i + n_orient // 2) % n_orient)[None], 0)[0]
        rho = np.maximum(Rmax - aniso * np.maximum(Rperp, 0.0), 0.0)
        knorm = float(np.sqrt((oriented_kernel(float(s), 0.0, elong) ** 2).sum()))
        null = knorm * noise
        if texture_null:
            j = np.argmin(R, axis=0)
            Vmin = -np.take_along_axis(R, j[None], 0)[0]
            Vperp = -np.take_along_axis(R, ((j + n_orient // 2) % n_orient)[None], 0)[0]
            val = np.maximum(Vmin - aniso * np.maximum(Vperp, 0.0), 0.0)
            tile = int(max(64, 10 * s))
            null = np.maximum(null, _tile_rms(val, tile))
        z = rho / null
        upd = rho > best_rho
        best_z[upd] = z[upd]
        best_rho[upd] = rho[upd]
        best_s[upd] = s
        th = thetas[i]
        # normal (across) direction of the best orientation
        nx[upd] = (-np.sin(th))[upd]
        ny[upd] = (np.cos(th))[upd]
    if valid is not None:
        best_z[~valid] = 0
        best_rho[~valid] = 0
    return best_rho, best_z, best_s, nx, ny


def nms(z: np.ndarray, nx: np.ndarray, ny: np.ndarray, step=None) -> np.ndarray:
    """Keep pixels whose z is a maximum across the ridge (along the normal)."""
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if step is None:
        step = np.ones_like(z)
    keep = np.ones(z.shape, bool)
    for sgn in (1.0, -1.0):
        cy = yy + sgn * ny * step
        cx = xx + sgn * nx * step
        nb = ndi.map_coordinates(z, [cy, cx], order=1, mode="nearest")
        keep &= z >= nb
    return keep


_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel: np.ndarray):
    """Split a 1-px skeleton into simple pixel paths at junctions.

    Returns (paths, junction_labels, n_junctions) where each path is an (m,2)
    int array of (y, x) and junction_labels labels the junction clusters.
    For each path we also return the junction id touching each end (or None).
    """
    skel = skel.astype(bool)
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    nb = ndi.convolve(skel.astype(int), k, mode="constant") * skel
    junc = skel & (nb >= 3)
    jl, nj = ndi.label(junc, structure=np.ones((3, 3)))
    body = skel & ~junc
    bl, nb_ = ndi.label(body, structure=np.ones((3, 3)))
    objs = ndi.find_objects(bl)
    h, w = skel.shape
    out = []
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(bl[sl] == lab)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        pts = set(zip(ys.tolist(), xs.tolist()))
        if len(pts) == 1:
            p = next(iter(pts))
            path = [p]
        else:
            adj = {}
            for p in pts:
                adj[p] = [(p[0] + dy, p[1] + dx) for dy, dx in _NB8
                          if (p[0] + dy, p[1] + dx) in pts]
            ends = [p for p, a in adj.items() if len(a) <= 1]
            start = ends[0] if ends else next(iter(pts))
            # walk, preferring 4-neighbours to avoid corner skipping
            path = [start]
            seen = {start}
            cur = start
            while True:
                cand = [q for q in adj[cur] if q not in seen]
                if not cand:
                    break
                cand.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
                cur = cand[0]
                seen.add(cur)
                path.append(cur)
        path = np.array(path, int)
        ends_j = []
        for p in (path[0], path[-1]):
            j = None
            y0, y1 = max(0, p[0] - 1), min(h, p[0] + 2)
            x0, x1 = max(0, p[1] - 1), min(w, p[1] + 2)
            win = jl[y0:y1, x0:x1]
            if win.max() > 0:
                j = int(win.max())
            ends_j.append(j)
        out.append((path, ends_j))
    centroids = ndi.center_of_mass(junc, jl, range(1, nj + 1)) if nj else []
    return out, np.array(centroids, float).reshape(-1, 2), jl


def detect(V: np.ndarray, noise: np.ndarray, scales, z_hi=5.0, z_lo=2.5,
           min_len=8.0, valid=None, exclude=None, long_len=40,
           long_frac=0.6, oriented=False, elong=3.0) -> list[Seed]:
    """Propose centrelines of bright ridges in V (vessels made bright).

    exclude: optional bool mask of pixels where proposals are not wanted.
    oriented: use elongated oriented filters (fine bands).
    """
    if oriented:
        rho, z, sc, nx, ny = oriented_ridge_maps(V, noise, scales, valid, elong=elong)
        rho = rho / GAUSS_LINE_PEAK * GAUSS_LINE_PEAK   # same scale convention
    else:
        rho, z, sc, nx, ny = ridge_maps(V, noise, scales, valid)
    keep = nms(z, nx, ny)
    cand = keep & (z > z_lo)
    if exclude is not None:
        cand &= ~exclude
    mask = apply_hysteresis_threshold(np.where(cand, z, 0.0), z_lo, z_hi)
    # a long, continuous ridge is significant even if no single pixel is:
    # keep components with enough length and a decent mean z (faint,
    # defocused vessels and wide vessels in dense networks)
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n:
        idx = np.arange(1, n + 1)
        size = ndi.sum(np.ones_like(z), lab, idx)
        mean = ndi.mean(z, lab, idx)
        good = idx[(size >= long_len) & (mean >= long_frac * z_hi)]
        if len(good):
            mask |= np.isin(lab, good)
    # close 1-px gaps left by NMS on diagonals before thinning
    mask = ndi.binary_closing(mask, structure=np.ones((2, 2))) | mask
    skel = skeletonize(mask)
    paths, jcent, _ = trace_skeleton(skel)
    seeds = []
    for path, ends_j in paths:
        xy = path[:, ::-1].astype(np.float64)
        # attach ends to junction centroids so pieces meet
        for side, j in ((0, ends_j[0]), (1, ends_j[1])):
            if j is not None:
                c = jcent[j - 1][::-1]
                if side == 0:
                    xy = np.vstack([c, xy])
                else:
                    xy = np.vstack([xy, c])
        if len(xy) < 2:
            continue
        L = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1)).sum()
        if L < min_len:
            continue
        iy, ix = path[:, 0], path[:, 1]
        s = sc[iy, ix]
        a = rho[iy, ix] / GAUSS_LINE_PEAK
        zz = z[iy, ix]
        # the ends attached to junctions borrow their neighbour's values
        if ends_j[0] is not None:
            s, a, zz = np.r_[s[:1], s], np.r_[a[:1], a], np.r_[zz[:1], zz]
        if ends_j[1] is not None:
            s, a, zz = np.r_[s, s[-1:]], np.r_[a, a[-1:]], np.r_[zz, zz[-1:]]
        seeds.append(Seed(xy, s.astype(float), a.astype(float), zz.astype(float),
                          list(ends_j)))
    return seeds
