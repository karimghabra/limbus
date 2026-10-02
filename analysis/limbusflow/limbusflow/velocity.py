"""Red-blood-cell velocity from the registered video.

Kymograph (space-time image)
----------------------------
For every registered frame t we read the intensity along the vessel
centreline (averaged across the central part of the lumen):

    K(t, s) = mean_{|w| < a d/2}  F_t( r(s) + w N(s) )

Samples are taken directly from the *raw* frame by pushing the reference
coordinates through the inverse registration transform, so each pixel is
interpolated only once.

Blood is not a uniform fluid at this resolution: red-cell aggregates and
plasma gaps make the intensity inside a vessel "flicker". These patterns are
carried along with the flow, so in K they draw oblique streaks

    K(t, s) ~ f(s - v t)     ->  slope ds/dt = v  (px / frame)

Pre-processing removes what does *not* move with the blood:
    K1 = K - <K>_t            static anatomy (vessel walls, crossings)
    K2 = K1 - <K1>_s          global brightness flicker (auto-gain, blinks)
then light Gaussian smoothing.

Three independent estimators
----------------------------
1. LSPIV (line-scan particle image velocimetry, Kim et al. 2012):
   cross-correlate each kymograph row with the row `lag` frames later,
       C_t(delta) = sum_s K(t, s) K(t + lag, s + delta),
   average C over many t and locate the peak delta* (sub-pixel, Gaussian
   3-point fit). v = delta* / lag.
2. Structure tensor (local orientation):
       J = G_rho * [[K_s^2, K_s K_t], [K_s K_t, K_t^2]]
   The eigenvector of the smallest eigenvalue points along the streaks,
   (e_s, e_t)  ->  v = e_s / e_t. Coherence ((l1 - l2)/(l1 + l2))^2 says how
   line-like the neighbourhood is. Gives a dense map v(t, s).
3. Time of flight ("flicker" correlation): the intensity trace at position
   s1 reappears at s2 = s1 + D after a delay tau:
       R_D(tau) = < K(t, s1) K(t + tau, s1 + D) >
   Repeating for several separations D and fitting D = v tau gives v.
Slow flow (multi-scale, see `estimate`): static anatomy is regressed out
row by row, the kymograph is time-binned by k in {1, 3, 9} so that slow
patterns move ~1 px/bin, the flow line is seeded from the antisymmetric part
of the correlation map (static structure is symmetric in delta, flow is not),
and binned results must persist across both halves of the recording.

Motion blur: the exposure (12.8 ms) is ~95 % of the frame period, so each
pattern is smeared by ~v px along s. The smear is identical in every frame,
so it lowers contrast but does not bias the displacement; it also acts as an
anti-aliasing low-pass filter along s.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .register import specular_mask
from .spline import VesselSpline


# ----------------------------------------------------------------------------- kymograph
def lumen_points(sp: VesselSpline, d: float | np.ndarray, frac=0.5, n_w=5):
    """(n_s, n_w, 2) sample grid across the central `frac` of the lumen."""
    d = np.broadcast_to(np.asarray(d, float), (len(sp.s),))
    w = np.linspace(-1, 1, n_w)[None, :] * (frac * d[:, None] / 2)
    return sp.xy[:, None, :] + w[..., None] * sp.N[:, None, :]


def _apply_affine(M, P):
    return (M[0, 0] * P[..., 0] + M[0, 1] * P[..., 1] + M[0, 2],
            M[1, 0] * P[..., 0] + M[1, 1] * P[..., 1] + M[1, 2])


def local_shifts(frames, reg, idx, bbox, ref, margin=20, max_shift=4.0):
    """Residual per-frame translation of a patch around one vessel.

    After the global similarity registration small local residuals remain
    (eye curvature, torsion). We sample the patch from every frame (through
    the inverse transform) and phase-correlate it with the reference patch.
    Returns (T, 2) shifts (dx, dy) in reference pixels to *add* to sample points.
    """
    x0, y0, x1, y1 = bbox
    H, W = reg.shape
    x0, y0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
    x1, y1 = min(W, int(x1 + margin)), min(H, int(y1 + margin))
    h, w = cv2.getOptimalDFTSize(y1 - y0), cv2.getOptimalDFTSize(x1 - x0)
    y1, x1 = min(H, y0 + h), min(W, x0 + w)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    P = np.stack([xx, yy], -1)

    def bp(a):
        a = a / (cv2.GaussianBlur(a, (0, 0), 15) + 1e-6)
        return cv2.GaussianBlur(a, (0, 0), 1.0) - cv2.GaussianBlur(a, (0, 0), 8.0)

    R = bp(ref[y0:y1, x0:x1].astype(np.float64))
    win = cv2.createHanningWindow(R.shape[::-1], cv2.CV_64F)
    out = np.zeros((len(idx), 2))
    for n, i in enumerate(idx):
        X, Y = _apply_affine(cv2.invertAffineTransform(reg.M[i]), P)
        f = np.asarray(frames[i], np.float64)
        p = ndi.map_coordinates(f, [Y, X], order=1, mode="nearest")
        (dx, dy), resp = cv2.phaseCorrelate(R, bp(p), win)
        if resp > 0.05 and np.hypot(dx, dy) < max_shift:
            out[n] = dx, dy
    # a shift of the patch content by +d means features sit at p + d
    return out - np.median(out, 0)


def kymograph(frames, reg, idx, pts, shifts=None, normalize=True, mask_specular=True):
    """K[t, s] averaged over the n_w lumen samples; NaN where masked."""
    K = np.empty((len(idx), pts.shape[0]), np.float64)
    for n, i in enumerate(idx):
        Q = pts if shifts is None else pts + shifts[n][None, None, :]
        X, Y = _apply_affine(cv2.invertAffineTransform(reg.M[i]), Q)
        f = np.asarray(frames[i], np.float32)
        v = ndi.map_coordinates(f, [Y, X], order=1, mode="nearest", cval=np.nan)
        if mask_specular:
            bad = ndi.map_coordinates(specular_mask(np.asarray(frames[i])).astype(np.float32), [Y, X], order=0)
            v = np.where(bad > 0, np.nan, v)
        outside = (X < 0) | (Y < 0) | (X > f.shape[1] - 1) | (Y > f.shape[0] - 1)
        v = np.where(outside, np.nan, v)
        with np.errstate(invalid="ignore"):
            K[n] = np.nanmean(v, 1)
        if normalize:
            K[n] /= np.median(f)
    return K


def kymographs(frames, reg, idx, pts_list, normalize=True, mask_specular=True, progress=True):
    """Kymographs for many vessels in ONE pass over the frames (much faster
    than calling `kymograph` per vessel). Returns a list of K arrays."""
    sizes = [p.shape[:2] for p in pts_list]
    allp = np.concatenate([p.reshape(-1, 2) for p in pts_list], 0)
    out = [np.empty((len(idx), s[0])) for s in sizes]
    cuts = np.cumsum([0] + [a * b for a, b in sizes])
    for n, i in enumerate(idx):
        X, Y = _apply_affine(cv2.invertAffineTransform(reg.M[i]), allp)
        raw = np.asarray(frames[i])
        f = raw.astype(np.float32)
        v = ndi.map_coordinates(f, [Y, X], order=1, mode="nearest")
        if mask_specular:
            bad = ndi.map_coordinates(specular_mask(raw).astype(np.float32), [Y, X], order=0) > 0
            v = np.where(bad, np.nan, v)
        v = np.where((X < 0) | (Y < 0) | (X > f.shape[1] - 1) | (Y > f.shape[0] - 1), np.nan, v)
        if normalize:
            v = v / np.median(f)
        for k, (a, b) in enumerate(sizes):
            with np.errstate(invalid="ignore"):
                seg = v[cuts[k]:cuts[k + 1]].reshape(a, b)
                out[k][n] = np.nanmean(seg, 1) if np.isnan(seg).any() else seg.mean(1)
        if progress and (n % 50 == 0 or n == len(idx) - 1):
            print(f"\rkymographs: frame {n + 1}/{len(idx)}", end="", flush=True)
    if progress:
        print()
    return out


def static_regressors(ref, pts):
    """Profiles along s that describe how *static* anatomy responds to a small
    misregistration: the reference image's x- and y-gradients averaged over the
    lumen sample points, (S, 2). A residual shift (dx, dy) of frame t adds
    ~ gx(s) dx + gy(s) dy to row t of the kymograph."""
    r = np.asarray(ref, np.float64)
    r = ndi.gaussian_filter(r / np.median(r), 1.0)
    gy, gx = np.gradient(r)
    X, Y = pts[..., 0], pts[..., 1]
    return np.stack([ndi.map_coordinates(gx, [Y, X], order=1, mode="nearest").mean(1),
                     ndi.map_coordinates(gy, [Y, X], order=1, mode="nearest").mean(1)], 1)


def remove_static(K, grads=None):
    """Per-row least squares: K(t, s) ~ c0 + c1 Kbar(s) + c2 gx(s) + c3 gy(s).

    Kbar (temporal mean profile) absorbs gain / illumination changes, gx, gy
    absorb residual translation of the static anatomy. What remains is what
    the static-anatomy model cannot explain - moving blood. Missing samples
    (NaN) are ignored and stay NaN.
    """
    Kbar = np.nanmean(K, 0)
    Kbar = np.where(np.isfinite(Kbar), Kbar, np.nanmean(Kbar))
    cols = [np.ones_like(Kbar), Kbar - Kbar.mean()]
    if grads is not None:
        cols += [grads[:, 0], grads[:, 1]]
    A = np.stack(cols, 1)
    out = np.full_like(K, np.nan)
    for t in range(len(K)):
        ok = np.isfinite(K[t])
        if ok.sum() < A.shape[1] + 3:
            continue
        c = np.linalg.lstsq(A[ok], K[t, ok], rcond=None)[0]
        out[t, ok] = K[t, ok] - A[ok] @ c
    return out


def bin_time(K, k):
    """Average k consecutive rows (NaN-aware). Slow flow: 0.1 px/frame becomes
    0.1 k px/bin, and noise drops by ~sqrt(k)."""
    if k <= 1:
        return K
    T = (len(K) // k) * k
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(K[:T].reshape(-1, k, K.shape[1]), 1)


def preprocess(K, t_window=10.0, highpass_s=10.0, smooth_s=1.0):
    """Keep only what moves with the blood.

    1. subtract a *running* temporal mean (Gaussian, sigma = t_window frames):
       removes static anatomy and anything that changes slowly without moving
       along the vessel (slow registration drift, illumination, vasomotion).
       A pattern moving >= 1 px/frame is displaced by many px within the
       window, so it survives the subtraction;
    2. subtract each row's mean: global brightness flicker (auto-gain, blinks);
    3. spatial high-pass along s (sigma = highpass_s px);
    4. light smoothing along s only - smoothing along t would correlate
       neighbouring rows and bias the displacement towards 0.
    Returns K2 (NaNs -> 0) and the weight map (0 where data were missing).
    """
    Wt = np.isfinite(K).astype(float)
    Kz = np.where(Wt > 0, K, 0.0)
    if t_window:   # normalised convolution: missing samples do not bias the running mean
        num = ndi.gaussian_filter1d(Kz, t_window, axis=0, mode="nearest")
        den = ndi.gaussian_filter1d(Wt, t_window, axis=0, mode="nearest")
        K1 = (Kz - num / np.maximum(den, 1e-6)) * Wt
    else:
        K1 = (Kz - Kz.sum(0) / np.maximum(Wt.sum(0), 1)) * Wt
    K2 = (K1 - (K1.sum(1) / np.maximum(Wt.sum(1), 1))[:, None]) * Wt
    if highpass_s:
        K2 = K2 - ndi.gaussian_filter1d(K2, highpass_s, axis=1, mode="nearest")
    if smooth_s:
        K2 = ndi.gaussian_filter1d(K2, smooth_s, axis=1)
    K2 *= Wt
    return K2, Wt


# ----------------------------------------------------------------------------- LSPIV
def _xcorr_rows(A, B, max_shift):
    """Normalised cross-correlation of each row of A with the same row of B,
    for integer shifts -max_shift..max_shift (B shifted to the right = +)."""
    n = A.shape[1]
    L = cv2.getOptimalDFTSize(2 * n)
    FA = np.fft.rfft(A, L, axis=1); FB = np.fft.rfft(B, L, axis=1)
    c = np.fft.irfft(np.conj(FA) * FB, L, axis=1)
    c = np.concatenate([c[:, -max_shift:], c[:, :max_shift + 1]], axis=1)
    shifts = np.arange(-max_shift, max_shift + 1)
    # unbiased normalisation by the overlap length
    c = c / (n - np.abs(shifts))[None, :]
    na = np.sqrt((A**2).mean(1)); nb = np.sqrt((B**2).mean(1))
    return c / (na * nb + 1e-12)[:, None], shifts


def _subpixel_peak(c, shifts):
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1 and np.all(c[k - 1:k + 2] > 0):
        l0, l1, l2 = np.log(c[k - 1:k + 2])      # Gaussian 3-point fit
        den = l0 - 2 * l1 + l2
        off = 0.5 * (l0 - l2) / den if den != 0 else 0.0
    elif 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]    # parabola fallback
        off = 0.5 * (c[k - 1] - c[k + 1]) / den if den != 0 else 0.0
    else:
        off = 0.0
    return shifts[k] + float(np.clip(off, -0.5, 0.5)), float(c[k])


def _peak_near(c, shifts, expect, frac=0.4, min_half=3.0):
    """Sub-pixel peak restricted to |delta - expect| <= max(min_half, frac |expect|)."""
    half = max(min_half, frac * abs(expect))
    win = np.abs(shifts - expect) <= half
    cc = np.where(win, c, -np.inf)
    k = int(np.argmax(cc))
    if not (0 < k < len(c) - 1) or not (win[k - 1] and win[k + 1]):
        return np.nan, np.nan
    return _subpixel_peak(c[k - 1:k + 2], shifts[k - 1:k + 2])


def lspiv(K2, lag=1, max_shift=None, window=None, step=None, expect_v=None):
    """Displacement per `lag` frames from row cross-correlation.

    window=None -> one global estimate from the mean correlation of all row pairs.
    Otherwise returns a time series using sliding windows of `window` row pairs.
    expect_v: if given (px per row), peaks are searched only near expect_v * lag,
    so the small-lag residue near delta = 0 cannot capture the estimate.
    Returns dict(v px/frame, peak, curve, shifts[, t_center, v_t, peak_t]).
    """
    T, S = K2.shape
    if max_shift is None:
        max_shift = int(min(40, S // 3))
        if expect_v is not None and np.isfinite(expect_v):
            max_shift = int(min(max(max_shift, abs(expect_v) * lag * 1.6 + 4), S // 2))
    C, shifts = _xcorr_rows(K2[:-lag], K2[lag:], max_shift)
    mean_c = C.mean(0)
    d, pk = _subpixel_peak(mean_c, shifts)
    # peak significance: height relative to the spread of the correlation curve
    snr = (pk - np.median(mean_c)) / (np.std(mean_c) + 1e-12)
    out = dict(v=d / lag, peak=pk, snr=snr, curve=mean_c, shifts=shifts, lag=lag, C=C)
    if window:
        step = step or max(1, window // 4)
        tc, vt, pt = [], [], []
        for a in range(0, len(C) - window + 1, step):
            cw = C[a:a + window].mean(0)
            if expect_v is not None and np.isfinite(expect_v):
                dd, pp = _peak_near(cw, shifts, expect_v * lag)
            else:
                dd, pp = _subpixel_peak(cw, shifts)
            tc.append(a + window / 2 + lag / 2); vt.append(dd / lag); pt.append(pp)
        out.update(t_center=np.array(tc), v_t=np.array(vt), peak_t=np.array(pt))
    return out


def correlation_map(K2, max_lag=15, max_shift=None):
    """C[lag-1, delta]: mean normalised correlation between rows `lag` frames apart
    as a function of spatial shift. A moving pattern draws a straight ridge
    delta = v * lag through the origin; static residue sits on delta = 0."""
    T, S = K2.shape
    if max_shift is None:
        max_shift = int(min(60, S // 3))
    rows = []
    for lag in range(1, max_lag + 1):
        C, shifts = _xcorr_rows(K2[:-lag], K2[lag:], max_shift)
        rows.append(C.mean(0))
    return np.array(rows), shifts


def antisymmetric_seed(Cmap, shifts, v_min=0.02, n_grid=240, min_disp=1.5, edge=3, min_lags=3):
    """Find the flow line in C(lag, delta) from its *asymmetry*.

    Static structure and registration jitter give correlations that are
    symmetric in delta (a stationary pattern looks the same shifted left or
    right); directed flow does not. Using the antisymmetric part

        A(lag, delta) = [C(lag, delta) - C(lag, -delta)] / 2

    we score every line through the origin, delta = v lag, by the mean of A
    along it (only where |v lag| >= min_disp, i.e. away from the static
    ridge) and return the best v - a Radon/Hough transform of A restricted to
    lines through the origin. Returns (v, score); v is NaN if nothing
    asymmetric is found.
    """
    lags = np.arange(1, len(Cmap) + 1, dtype=float)
    Amap = 0.5 * (Cmap - Cmap[:, ::-1])          # shifts are symmetric: -delta is the mirrored column
    g = np.geomspace(v_min, max(v_min * 1.01, (shifts[-1] - edge) / 1.0), n_grid)
    vgrid = np.concatenate([-g[::-1], g])
    best, best_v = -np.inf, np.nan
    scores = np.full(len(vgrid), np.nan)
    for i, v in enumerate(vgrid):
        dd = v * lags
        use = (np.abs(dd) >= min_disp) & (np.abs(dd) <= shifts[-1] - edge)
        if use.sum() < min_lags:
            continue
        vals = [np.interp(x, shifts, Amap[j]) for j, x in zip(np.flatnonzero(use), dd[use])]
        scores[i] = np.mean(vals)
        if scores[i] > best:
            best, best_v = scores[i], v
    return (best_v if best > 0 else np.nan), best


def ridge_velocity(Cmap, shifts, min_peak=0.08, min_disp=1.5, edge=3, n_seed=3):
    """Track the correlation ridge lag by lag and fit delta = v * lag.

    Seed: global maxima of the first `n_seed` lags give a first slope v0.
    Tracking: at every further lag the peak is searched only within
    +-max(3, 0.3 |v0 lag|) px of the prediction v0 * lag (so the search
    cannot jump to an unrelated peak), v0 is updated, and tracking stops
    once the prediction leaves the shift window or the peak fades.
    Kept lags must be displaced by >= min_disp px (the zero-shift residue
    cannot dominate). Weighted least squares through the origin; returns
    R^2 and the mean ridge height (quality).
    """
    lags = np.arange(1, len(Cmap) + 1)
    d = np.full(len(lags), np.nan); pk = np.full(len(lags), np.nan)
    v0, _ = antisymmetric_seed(Cmap, shifts, min_disp=min_disp, edge=edge)
    start = 0
    if not np.isfinite(v0):       # fall back to the first lags' global maxima
        for j in range(min(n_seed, len(lags))):
            d[j], pk[j] = _subpixel_peak(Cmap[j], shifts)
        seed = (pk[:n_seed] > min_peak) & np.isfinite(d[:n_seed])
        v0 = np.sum(d[:n_seed][seed] * lags[:n_seed][seed]) / np.sum(lags[:n_seed][seed] ** 2) if seed.any() else 0.0
        start = n_seed
    misses = 0
    for j in range(start, len(lags)):
        pred = v0 * lags[j]
        if abs(pred) > shifts[-1] - edge:
            break
        half = max(3.0, 0.3 * abs(pred))
        win = np.abs(shifts - pred) <= half
        c = np.where(win, Cmap[j], -np.inf)
        k = int(np.argmax(c))
        interior = 0 < k < len(shifts) - 1 and win[k - 1] and win[k + 1]
        if interior:
            d[j], pk[j] = _subpixel_peak(Cmap[j][k - 1:k + 2], shifts[k - 1:k + 2])
        if not interior or pk[j] < min_peak:
            # no clean local maximum at this lag (e.g. small-lag residue near delta = 0):
            # skip it, but give up after 3 consecutive misses once the ridge has been found
            d[j], pk[j] = np.nan, np.nan
            misses += 1
            if misses >= 3 and np.isfinite(d[:j]).any():
                break
            continue
        misses = 0
        m = np.isfinite(d[:j + 1]) & (pk[:j + 1] > min_peak)
        v0 = np.sum(pk[:j + 1][m] * d[:j + 1][m] * lags[:j + 1][m]) / np.sum(pk[:j + 1][m] * lags[:j + 1][m] ** 2)
    ok = np.isfinite(d) & (pk > min_peak) & (np.abs(d) >= min_disp) & (np.abs(d) < shifts[-1] - edge)
    # the ridge must be one-signed: keep the dominant sign
    if ok.sum():
        sgn = np.sign(np.sum(np.sign(d[ok]) * pk[ok]))
        ok &= np.sign(d) == sgn
    if ok.sum() >= 2:
        w = pk[ok]
        v = np.sum(w * lags[ok] * d[ok]) / np.sum(w * lags[ok] ** 2)
        pred = v * lags[ok]
        ss_res = np.sum(w * (d[ok] - pred) ** 2); ss_tot = np.sum(w * (d[ok] - np.average(d[ok], weights=w)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        quality = float(np.mean(pk[ok]))
    else:
        v, r2, quality = np.nan, np.nan, 0.0
    return dict(v=float(v), lags=lags, disp=d, peak=pk, used=ok, r2=float(r2), quality=quality)


def best_lag(v, target_disp=8.0, max_lag=8):
    if not np.isfinite(v) or v == 0:
        return 2
    return int(np.clip(round(target_disp / abs(v)), 2, max_lag))   # lag 1 is prone to small-lag residue


# ----------------------------------------------------------------------------- structure tensor
def structure_tensor_velocity(K2, v_hint=None, sigma_d=1.0, sigma_i=(4.0, 6.0)):
    """Dense velocity map from local streak orientation.

    The s-axis is first compressed by f = max(1, |v_hint|/1.5) so that streaks
    have a slope near 45 degrees (well-sampled derivatives); velocities are
    converted back at the end.
    """
    f = 1.0 if v_hint is None else max(1.0, abs(v_hint) / 1.5)
    Kc = ndi.zoom(K2, (1, 1 / f), order=1) if f > 1 else K2
    Ks = ndi.gaussian_filter(Kc, sigma_d, order=(0, 1))
    Kt = ndi.gaussian_filter(Kc, sigma_d, order=(1, 0))
    Jss = ndi.gaussian_filter(Ks * Ks, sigma_i); Jtt = ndi.gaussian_filter(Kt * Kt, sigma_i)
    Jst = ndi.gaussian_filter(Ks * Kt, sigma_i)
    tr = Jss + Jtt; disc = np.sqrt((Jss - Jtt) ** 2 + 4 * Jst**2)
    l1 = 0.5 * (tr + disc); l2 = 0.5 * (tr - disc)
    coh = (disc / (tr + 1e-12)) ** 2
    # eigenvector of the small eigenvalue l2: (Jst, l2 - Jss) in (s, t)
    es, et = Jst, l2 - Jss
    with np.errstate(divide="ignore", invalid="ignore"):
        v = es / et * f
    if f > 1:
        zoom_back = (1, K2.shape[1] / v.shape[1])
        v = ndi.zoom(v, zoom_back, order=1); coh = ndi.zoom(coh, zoom_back, order=1)
        l1 = ndi.zoom(l1, zoom_back, order=1)
    energy = l1
    w = coh * energy
    good = np.isfinite(v) & (np.abs(v) < 60)
    vg = np.average(v[good], weights=w[good]) if w[good].sum() > 0 else np.nan
    # weighted median is more robust than the weighted mean
    vs = v[good]; ws = w[good]
    o = np.argsort(vs); cw = np.cumsum(ws[o])
    vmed = vs[o][np.searchsorted(cw, cw[-1] / 2)] if len(vs) else np.nan
    return dict(v=float(vmed), v_mean=float(vg), v_map=v, coherence=coh, energy=energy, compress=f)


# ----------------------------------------------------------------------------- time of flight
def time_of_flight(K2, separations=None, max_lag=20, n_starts=60):
    """Delay of the flicker signal between points D px apart; v from D = v * tau.

    Blood-borne patterns decorrelate within a few tens of frames, so the
    separations are kept short (a few px to ~50 px)."""
    T, S = K2.shape
    if separations is None:
        separations = np.unique(np.linspace(4, min(48, S // 3), 8).astype(int))
    taus, peaks, curves = [], [], []
    lags = np.arange(-max_lag, max_lag + 1)
    for D in separations:
        s1 = np.linspace(0, S - D - 1, min(n_starts, S - D)).astype(int)
        a = K2[:, s1]; b = K2[:, s1 + D]
        a = a - a.mean(0); b = b - b.mean(0)
        L = cv2.getOptimalDFTSize(2 * T)
        c = np.fft.irfft(np.conj(np.fft.rfft(a, L, axis=0)) * np.fft.rfft(b, L, axis=0), L, axis=0)
        c = np.concatenate([c[-max_lag:], c[:max_lag + 1]], 0) / (T - np.abs(lags))[:, None]
        c = c / (a.std(0) * b.std(0) + 1e-12)[None, :]
        cm = c.mean(1)
        tau, pk = _subpixel_peak(cm, lags)
        taus.append(tau); peaks.append(pk); curves.append(cm)
    taus = np.array(taus, float); D = np.asarray(separations, float); peaks = np.array(peaks)
    ok = (np.abs(taus) > 0.3) & (peaks > 0.05)
    if ok.sum() >= 2:
        # least squares D = v * tau through the origin, weighted by peak height
        w = peaks[ok]
        v = np.sum(w * D[ok] * taus[ok]) / np.sum(w * taus[ok] ** 2)
    else:
        v = np.nan
    return dict(v=float(v), separations=D, tau=taus, peak=peaks, curves=np.array(curves), lags=lags, used=ok)


# ----------------------------------------------------------------------------- pulsatility
def pulsatility(t_s, v_t, fmin=0.6, fmax=3.0):
    """Dominant cardiac frequency and pulsatility index of a velocity trace."""
    v = np.asarray(v_t, float)
    ok = np.isfinite(v)
    if ok.sum() < 16:
        return dict(hr_bpm=np.nan, PI=np.nan, freqs=None, power=None)
    tt = np.asarray(t_s)[ok]; vv = v[ok] - v[ok].mean()
    dt = np.median(np.diff(tt))
    n = 4 * len(vv)
    F = np.fft.rfftfreq(n, dt); Pw = np.abs(np.fft.rfft(vv * np.hanning(len(vv)), n)) ** 2
    band = (F >= fmin) & (F <= fmax)
    f0 = F[band][np.argmax(Pw[band])] if band.any() else np.nan
    m = np.abs(np.mean(v[ok]))
    PI = (np.percentile(v[ok], 95) - np.percentile(v[ok], 5)) / m if m > 0 else np.nan
    return dict(hr_bpm=60 * f0, PI=float(abs(PI)), freqs=F, power=Pw)


# ----------------------------------------------------------------------------- multi-scale
def _single_scale(K2, p, k):
    """All three estimators on one (possibly time-binned) kymograph; velocities in px/bin."""
    Cm, shifts = correlation_map(K2, p["max_lag"])
    rid = ridge_velocity(Cm, shifts)
    st = structure_tensor_velocity(K2, rid["v"] if np.isfinite(rid["v"]) else None)
    tof = time_of_flight(K2, max_lag=min(p["max_lag"] + 5, len(K2) // 3))
    v = rid["v"]
    lag = best_lag(v)
    win = int(min(p["vt_window"], max(8, len(K2) // 3)))
    ts = lspiv(K2, lag, window=win, step=max(1, win // 12), expect_v=v if np.isfinite(v) else None)
    v_t = np.where(ts["peak_t"] > 0.05, ts["v_t"], np.nan)
    vt_ok = np.isfinite(v_t) & (np.sign(v_t) == np.sign(v)) if np.isfinite(v) else np.zeros(len(v_t), bool)
    v_time = float(np.median(v_t[vt_ok])) if vt_ok.sum() >= 5 else v
    agree = [m for m in (v, st["v"], tof["v"]) if np.isfinite(m) and np.isfinite(v_time)
             and np.sign(m) == np.sign(v_time) and abs(m - v_time) <= p["agree_tol"] * abs(v_time) + 0.5]
    reliable = bool(np.isfinite(v_time) and rid["quality"] >= p["min_quality"] and rid["r2"] >= p["min_r2"]
                    and len(agree) >= 2)
    return dict(k=k, v_bin=v_time, reliable=reliable, n_agree=len(agree), quality=rid["quality"], r2=rid["r2"],
                v_ridge=v, v_st=st["v"], v_tof=tof["v"], lag=lag, ts=ts, v_t=v_t, ridge=rid, Cmap=Cm,
                shifts=shifts, tof=tof, st_map=st["v_map"], coherence=st["coherence"], K2=K2)


def split_half(K2, v_bin, max_lag=15, tol=0.35, slack=0.3):
    """Persistence test: ridge velocity of the first and second half of the
    kymograph must share the sign of v_bin and agree within tol*|v| + slack.
    Real flow persists; slow registration drift or illumination changes that
    masquerade as slow motion generally do not."""
    h = len(K2) // 2
    vh = []
    for part in (K2[:h], K2[h:]):
        if len(part) < 32:
            return False, (np.nan, np.nan)
        Cm, sh = correlation_map(part, min(max_lag, len(part) // 3))
        vh.append(ridge_velocity(Cm, sh)["v"])
    ok = (np.isfinite(vh[0]) and np.isfinite(vh[1]) and np.sign(vh[0]) == np.sign(vh[1]) == np.sign(v_bin)
          and abs(vh[0] - vh[1]) <= tol * abs(v_bin) + slack)
    return bool(ok), tuple(vh)


def estimate(K, grads=None, fps=1.0, scales=(1, 3, 9), t_window=10.0, highpass_s=10.0, max_lag=15,
             vt_window=24, min_quality=0.10, min_r2=0.8, agree_tol=0.35, good_range=(0.7, 12.0),
             slow_min_quality=0.15):
    """Multi-scale velocity: remove static anatomy, then for each time-binning
    factor k run the three estimators on the binned kymograph (in px/bin).

    A scale is *usable* if it is reliable (clear ridge + 2 of 3 estimators
    agree). Binned scales (k > 1, slow flow) must in addition pass the
    split-half persistence test and reach `slow_min_quality`: on negative
    controls (lines through static tissue) these two conditions removed all
    slow false positives. Among usable scales we prefer those where the
    displacement per bin lies in `good_range` (where all estimators are
    accurate), then the highest ridge quality. Velocities are in px/frame.
    """
    p = dict(t_window=t_window, highpass_s=highpass_s, max_lag=max_lag, vt_window=vt_window,
             min_quality=min_quality, min_r2=min_r2, agree_tol=agree_tol)
    Kr = remove_static(K, grads)
    res = []
    for k in scales:
        Kb = bin_time(Kr, k)
        if len(Kb) < 3 * max_lag:
            continue
        K2, _ = preprocess(Kb, t_window, highpass_s)
        r = _single_scale(K2, p, k)
        r["split"], r["v_halves"] = (True, (np.nan, np.nan))
        if k > 1 and r["reliable"]:
            r["split"], r["v_halves"] = split_half(K2, r["v_bin"], max_lag)
            r["reliable"] = r["split"] and r["quality"] >= slow_min_quality
        res.append(r)
    usable = [r for r in res if r["reliable"]]
    in_range = [r for r in usable if good_range[0] <= abs(r["v_bin"]) <= good_range[1]]
    pool = in_range or usable
    best = max(pool, key=lambda r: r["quality"]) if pool else (res[0] if res else None)
    if best is None:
        return dict(v=np.nan, reliable=False, scale=1)
    k = best["k"]
    out = dict(v=best["v_bin"] / k if best["reliable"] else np.nan, v_time=best["v_bin"] / k,
               v_ridge=best["v_ridge"] / k, v_st=best["v_st"] / k, v_tof=best["v_tof"] / k,
               quality=best["quality"], r2=best["r2"], reliable=best["reliable"], n_agree=best["n_agree"],
               scale=k, lag=best["lag"], t_s=best["ts"]["t_center"] * k / fps, v_t=best["v_t"] / k,
               ridge=best["ridge"], Cmap=best["Cmap"], shifts=best["shifts"], tof=best["tof"],
               st_map=best["st_map"] / k, coherence=best["coherence"], K2=best["K2"],
               per_scale=[dict(k=r["k"], v=r["v_bin"] / r["k"], reliable=r["reliable"], quality=r["quality"],
                               n_agree=r["n_agree"], split=r["split"]) for r in res])
    pul = pulsatility(out["t_s"], out["v_t"])
    out.update(hr_bpm=pul["hr_bpm"], PI=pul["PI"])
    return out
