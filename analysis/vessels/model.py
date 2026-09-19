"""The physical vessel model and its fit.

A vessel is a Catmull-Rom spline with a radius at every control point. Its
absorbance is that of a cylinder of blood, D * sqrt(1 - (d/r)^2), blurred by
the optics; the background under it is a plane. Fitting is Levenberg-
Marquardt on the weighted residual over a band fixed for the whole fit, so
the misfit is comparable with the no-vessel baseline on exactly the same
pixels.
"""
import cv2
import numpy as np

SPACING = 16.0      # control-point spacing along a candidate (px)
BAND = 6.0          # fit pixels within r + BAND + 3 sigma_psf of the centreline


def catmull(P, step=0.5):
    """Dense centreline through control points P (K,2); returns samples and
    each sample's fractional control-point index (for interpolating radius)."""
    K = len(P)
    if K == 2:
        L = np.hypot(*(P[1] - P[0]))
        n = max(2, int(L / step))
        t = np.linspace(0, 1, n)
        return P[0] + (P[1] - P[0]) * t[:, None], t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, u = [], []
    for i in range(K - 1):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        L = np.hypot(*(p2 - p1))
        n = max(2, int(np.ceil(L / step)))
        t = np.linspace(0, 1, n, endpoint=(i == K - 2))[:, None]
        pts = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(pts)
        u.append(i + t[:, 0])
    return np.vstack(out), np.concatenate(u)


class Vessel:
    def __init__(self, P0, r0, D0):
        self.P0 = np.asarray(P0, float)                     # initial control points
        t = np.gradient(self.P0, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
        self.N = np.stack([-t[:, 1], t[:, 0]], 1)            # normals (fixed during a fit)
        K = len(self.P0)
        # parameters: normal offsets (K), radii (K), peak absorbance D, offset b
        self.p = np.concatenate([np.zeros(K), np.full(K, r0, float), [D0, 0.0]])

    @property
    def K(self):
        return len(self.P0)

    def unpack(self, p=None):
        p = self.p if p is None else p
        K = self.K
        P = self.P0 + p[:K, None] * self.N
        return P, np.clip(p[K:2 * K], 0.5, 40), p[2 * K], p[2 * K + 1]

    def render(self, xs, ys, psf, p=None, with_offset=True, box=None):
        """Absorbance on the pixel grid (xs, ys: 1D coords of a box)."""
        P, R, D, b = self.unpack(p)
        C, u = catmull(P)
        r = np.interp(u, np.arange(self.K), R)
        X, Y = np.meshgrid(xs, ys)
        pts = np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32)
        best = np.full(len(pts), np.inf, np.float32)
        rb = np.zeros(len(pts), np.float32)
        for i0 in range(0, len(C), 256):                       # nearest centreline sample
            c = C[i0:i0 + 256].astype(np.float32)
            d2 = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
            k = d2.argmin(1)
            m = d2[np.arange(len(pts)), k]
            upd = m < best
            best[upd], rb[upd] = m[upd], r[i0:i0 + 256][k[upd]]
        d = np.sqrt(best)
        a = D * np.sqrt(np.clip(1 - (d / np.maximum(rb, 1e-3)) ** 2, 0, None))
        a = a.reshape(len(ys), len(xs)).astype(np.float32)
        if psf > 0:
            a = cv2.GaussianBlur(a, (0, 0), psf)
        return a + (b if with_offset else 0.0)

    def box(self, shape, psf, pad=None):
        P, R, _, _ = self.unpack()
        m = (pad if pad is not None else R.max() + BAND + 3 * psf)
        H, W = shape
        x0, x1 = int(max(0, P[:, 0].min() - m)), int(min(W, P[:, 0].max() + m + 1))
        y0, y1 = int(max(0, P[:, 1].min() - m)), int(min(H, P[:, 1].max() + m + 1))
        return x0, y0, x1, y1


def frame_coords(v, xs, ys, mask):
    """For every pixel in mask: fractional control index u of its nearest
    centreline sample and signed distance d from the CURRENT curve. Measured
    once per outer round; within a round a normal offset delta(u) of the
    centreline changes the distance to d - delta(u), so rendering is one
    vectorised formula (fast enough for finite-difference Jacobians)."""
    P, _, _, _ = v.unpack()
    C, u = catmull(P)
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    yy, xx = np.nonzero(mask)
    pts = np.stack([xs[xx], ys[yy]], 1).astype(np.float32)
    best = np.full(len(pts), np.inf, np.float32)
    idx = np.zeros(len(pts), np.int64)
    for i0 in range(0, len(C), 512):
        c = C[i0:i0 + 512].astype(np.float32)
        d2 = ((pts[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        k = d2.argmin(1)
        m = d2[np.arange(len(pts)), k]
        upd = m < best
        best[upd], idx[upd] = m[upd], i0 + k[upd]
    rel = pts - C[idx]
    d = (rel * nrm[idx]).sum(1)                       # signed distance along the normal
    # the offsets are defined along the CONTROL normals; project them onto the
    # sample normals (they agree closely for smooth curves)
    return yy, xx, u[idx], d.astype(np.float32)


def render_fast(p, K, frame, shape, psf, v):
    yy, xx, u, d = frame
    delta = np.interp(u, np.arange(K), p[:K])
    r = np.clip(np.interp(u, np.arange(K), p[K:2 * K]), 0.5, 40)
    D, b = p[2 * K], p[2 * K + 1]
    a = np.zeros(shape, np.float32)
    a[yy, xx] = D * np.sqrt(np.clip(1 - ((d - delta) / r) ** 2, 0, None))
    if psf > 0:
        a = cv2.GaussianBlur(a, (0, 0), psf)
    return a + b


SMOOTH_PX = 0.5      # a 0.5 px second difference in the centreline offsets costs 1 chi2 unit
RADIUS_PX = 0.5      # a 0.5 px jump in radius between control points costs 1 chi2 unit

_FIT_CACHE = {}


def fit_vessel_cached(v, target, weight, psf, exclude=None, **kw):
    """fit_vessel with a result cache keyed on everything the fit can see:
    the starting parameters and the exact pixels (target, weights, excluded
    junction disks) in the region it could possibly touch. Identical inputs
    give the identical, stored result - so re-running on an image changed
    only locally (e.g. vessels planted for a benchmark) refits only what the
    change can affect. Exact: no approximation."""
    import hashlib
    H, W = target.shape
    P, R, _, _ = v.unpack()
    rad = float(min(max(2 * R.max(), 6.0) + BAND + 3 * psf, 45.0))
    x0, y0, x1, y1 = v.box((H, W), psf, pad=rad + 3 * psf + 2 + 12)       # + room to move
    h = hashlib.sha1()
    for arr in (v.P0, v.N, v.p):
        h.update(np.ascontiguousarray(np.round(arr, 6)).tobytes())
    h.update(np.ascontiguousarray(target[y0:y1, x0:x1]).tobytes())
    h.update(np.ascontiguousarray(weight[y0:y1, x0:x1]).tobytes())
    if exclude is not None:
        h.update(np.ascontiguousarray(exclude[y0:y1, x0:x1]).tobytes())
    h.update(repr((psf, sorted(kw.items()), (x0, y0, x1, y1))).encode())
    key = h.hexdigest()
    if key in _FIT_CACHE:
        P0, N, pv, res = _FIT_CACHE[key]
        v.P0, v.N, v.p = P0.copy(), N.copy(), pv.copy()
        return res
    res = fit_vessel(v, target, weight, psf, exclude=exclude, **kw)
    _FIT_CACHE[key] = (v.P0.copy(), v.N.copy(), v.p.copy(), res)
    return res


def fit_vessel(v, target, weight, psf, iters=15, rounds=3, log=None, exclude=None, max_move=6.0):
    """Levenberg-Marquardt on the weighted residual.

    The fitting BAND (pixels near the initial centreline) is fixed for the
    whole fit, so the misfit is always summed over the same pixels as the
    no-vessel baseline it is compared with. The radius is capped to fit
    inside the band, and the background is a plane (offset + x, y slopes)
    started from the band's outer ring: otherwise a wide, flat "vessel" is
    the model's only way to absorb a background gradient.
    'rounds' re-measure pixel coordinates after the centreline has moved.
    max_move caps how far (px) a control point may leave its start; 0 freezes
    the centreline completely and fits only radius, depth and background -
    used when the centreline comes from a ridge detector that located it
    better than a local fit would."""
    H, W = target.shape
    K = v.K
    P, R, _, _ = v.unpack()
    rad = float(min(max(2 * R.max(), 6.0) + BAND + 3 * psf, 45.0))
    r_cap = max(1.0, rad - BAND - 2 * psf)
    x0, y0, x1, y1 = v.box((H, W), psf, pad=rad + 3 * psf + 2)
    xs, ys = np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)
    tgt = target[y0:y1, x0:x1]
    w = weight[y0:y1, x0:x1]
    C, _ = catmull(P)
    line = np.zeros_like(tgt, np.uint8)
    cv2.polylines(line, [np.rint(C - [x0, y0]).astype(np.int32)], False, 1, 1)
    band = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(rad) + 1,) * 2)) > 0
    inner = cv2.dilate(line, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(R.max() + 2 * psf) + 1,) * 2)) > 0
    band &= w > 0
    if exclude is not None:
        band &= ~exclude[y0:y1, x0:x1]
    n_data = int(band.sum())
    if n_data < 20:
        return 0.0, 0.0, (x0, y0, x1, y1), n_data
    sw = np.sqrt(w[band])
    X = (xs[None, :] - xs.mean()).repeat(len(ys), 0) / 100.0
    Y = (ys[:, None] - ys.mean()).repeat(len(xs), 1) / 100.0
    ring = band & ~inner
    b0 = float(np.median(tgt[ring])) if ring.sum() > 10 else float(np.median(tgt[band]))
    # baseline: the best plane alone (no vessel), on exactly these pixels
    G = np.stack([np.ones(n_data), X[band], Y[band]], 1) * sw[:, None]
    coef, *_ = np.linalg.lstsq(G, tgt[band] * sw, rcond=None)
    rb = tgt[band] * sw - G @ coef
    chi_null = float(rb @ rb)
    # parameters: offsets K, radii K, D, plane (b, bx, by)
    p = np.concatenate([v.p[:2 * K + 1], [b0, 0.0, 0.0]])
    p[K:2 * K] = np.clip(p[K:2 * K], 0.8, r_cap)
    for rnd in range(rounds):
        frame = frame_coords(v, xs, ys, cv2.dilate(band.astype(np.uint8), np.ones((int(6 * psf) + 3,) * 2, np.uint8)) > 0)
        p[:K] = 0.0

        def model(q):
            a = render_fast(np.concatenate([q[:2 * K + 1], [0.0]]), K, frame, tgt.shape, psf, v)
            return a + q[2 * K + 1] + q[2 * K + 2] * X + q[2 * K + 3] * Y

        def resid(q):
            data = (tgt - model(q))[band] * sw
            prior = [np.diff(q[K:2 * K]) / RADIUS_PX]
            if K >= 3:
                prior.append(np.diff(q[:K], 2) / SMOOTH_PX)
            return np.concatenate([data] + prior).astype(np.float32)

        r0 = resid(p)
        chi = float(r0 @ r0)
        active = np.arange(len(p)) if max_move > 0 else np.arange(K, len(p))
        lam = 1e-2
        steps = np.concatenate([np.full(K, 0.2), np.full(K, 0.2), [0.005, 0.001, 0.001, 0.001]])
        for it in range(iters):
            J = np.zeros((len(r0), len(active)), np.float32)
            for jj, j in enumerate(active):
                q = p.copy()
                q[j] += steps[j]
                J[:, jj] = (resid(q) - r0) / steps[j]
            JTJ = J.T @ J
            g = J.T @ r0
            improved = False
            for _ in range(8):
                A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-6)
                try:
                    da = -np.linalg.solve(A, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                dp = np.zeros(len(p))
                dp[active] = da
                q = p + dp
                q[:K] = np.clip(q[:K], -max_move, max_move)
                q[K:2 * K] = np.clip(q[K:2 * K], 0.8, r_cap)
                q[2 * K] = max(q[2 * K], 0.0)
                rq = resid(q)
                cq = float(rq @ rq)
                if cq < chi:
                    p, r0, chi = q, rq, cq
                    lam = max(lam / 3, 1e-6)
                    improved = True
                    break
                lam *= 5
            if not improved or np.abs(dp[:2 * K]).max() < 0.02:
                break
        # move the curve to its fitted position, re-measure next round
        moved = float(np.abs(p[:K]).max())
        v.p = np.concatenate([p[:2 * K + 1], [p[2 * K + 1]]])
        Pn = v.unpack()[0]
        v.P0 = Pn
        tg = np.gradient(Pn, axis=0)
        tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)[:, None]
        v.N = np.stack([-tg[:, 1], tg[:, 0]], 1)
        v.p[:K] = 0.0
        if moved < 0.5:
            break
    # data misfit only (priors excluded) on the fixed band, for the comparison
    data = (tgt - model(p))[band] * sw
    return chi_null, float(data @ data), (x0, y0, x1, y1), n_data
