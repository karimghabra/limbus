"""Vessel diameter along the centreline, vessel masks, and tortuosity.

Diameter from cross-sectional intensity profiles
------------------------------------------------
At every arc-length sample s_i we read the image along the normal line

    q(t) = r(s_i) + t N(s_i),     t in [-h, h]

with bilinear interpolation, giving a profile P_i(t). Blood absorbs the
(green/NIR) light, so the vessel appears as a dip. Neighbouring profiles
(+-`avg` px along s) are averaged to beat down noise.

Two width estimators are provided:

1. FWHM (primary) - full width at half maximum depth. Baseline b = median of the two
   outer tails, depth A = b - min(P). Width = distance between the two points
   where P crosses b - A/2 (linear interpolation between samples).

2. Model fit ("erf" / blurred box) - a vessel of width w centred at c, seen
   through an optical point-spread function of width sigma, on a linear
   background:

       P(t) = b0 + b1 t - A/2 [ erf((t - c + w/2)/(sqrt2 sigma)) - erf((t - c - w/2)/(sqrt2 sigma)) ]

   fitted by non-linear least squares. c corrects the centreline position;
   sigma absorbs the edge softness. In these limbal images sigma is large
   (~0.3 d) and grows with d, i.e. it is dominated by the vessel's own
   rounded (cylindrical) absorption profile and by light scattering in the
   sclera, not by a fixed optical PSF. Width and softness then trade off, so
   w is model-dependent; a projected-cylinder model gives ~20 % larger
   widths with an identical goodness of fit. For narrow vessels (w < sigma)
   the fit collapses: their width is not resolved ("subres").

The FWHM is therefore the primary, model-free diameter. Caveat: for an
ideal cylinder with weak absorption the dip is a semi-ellipse whose FWHM is
sqrt(3)/2 ~ 0.87 of the true diameter; for a strongly absorbing (box-like)
vessel FWHM ~ d. Treat absolute diameters as +-15 %; relative changes along
a vessel and between vessels are much more reliable.

Tortuosity
----------
  DM    distance metric          L / C  (arc length / chord)          >= 1
  SOAM  sum of angles metric     (1/L) int |dtheta|  = (1/L) int |kappa| ds  (rad/px)
  TC    total curvature          int |kappa| ds  (rad)
  KSQ   mean squared curvature   (1/L) int kappa^2 ds  (1/px^2)
  ICM   inflection count metric  (n_inflections + 1) * DM
Inflection points are sign changes of kappa (ignoring |kappa| below a noise floor).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import erf

from .spline import VesselSpline


# ----------------------------------------------------------------------------- profiles
def cross_profiles(img, sp: VesselSpline, half_width: float, dt: float = 0.25):
    """Sample img along the normal at every centreline point -> (n_s, n_t) array and t axis."""
    t = np.arange(-half_width, half_width + 1e-9, dt)
    X = sp.xy[:, 0:1] + t[None, :] * sp.N[:, 0:1]
    Y = sp.xy[:, 1:2] + t[None, :] * sp.N[:, 1:2]
    P = ndi.map_coordinates(np.asarray(img, np.float64), [Y, X], order=1, mode="nearest")
    return P, t


def smooth_along(P, avg):
    """Average profiles over a window of +-avg samples along the vessel."""
    if avg <= 0:
        return P
    return ndi.uniform_filter1d(P, 2 * avg + 1, axis=0, mode="nearest")


def box_blur_model(t, b0, b1, A, c, w, sig):
    r2s = np.sqrt(2) * max(sig, 1e-3)
    return b0 + b1 * t - 0.5 * A * (erf((t - c + w / 2) / r2s) - erf((t - c - w / 2) / r2s))


def fwhm_width(p, t, tail_frac=0.2):
    n = len(p)
    k = max(2, int(tail_frac * n))
    base = np.median(np.r_[p[:k], p[-k:]])
    c0 = n // 2
    win = slice(max(0, c0 - n // 4), min(n, c0 + n // 4))
    imin = win.start + int(np.argmin(p[win]))
    depth = base - p[imin]
    if depth <= 0:
        return np.nan, np.nan, 0.0
    half = base - depth / 2
    # walk left / right from the minimum until the profile rises above half
    il = imin
    while il > 0 and p[il] < half:
        il -= 1
    ir = imin
    while ir < n - 1 and p[ir] < half:
        ir += 1
    if p[il] < half or p[ir] < half:
        return np.nan, t[imin], depth
    tl = np.interp(half, [p[il + 1], p[il]], [t[il + 1], t[il]])
    tr = np.interp(half, [p[ir - 1], p[ir]], [t[ir - 1], t[ir]])
    return tr - tl, 0.5 * (tl + tr), depth


def box_blur_jac(t, b0, b1, A, c, w, sig):
    """Analytic Jacobian of `box_blur_model` w.r.t. (b0, b1, A, c, w, sig)."""
    r = np.sqrt(2) * max(sig, 1e-3)
    a = (t - c + w / 2) / r; b = (t - c - w / 2) / r
    ea, eb = np.exp(-a**2), np.exp(-b**2)
    k = 2 / np.sqrt(np.pi)
    E = erf(a) - erf(b)
    dE_dc = k * (-1 / r) * (ea - eb)
    dE_dw = k * (0.5 / r) * (ea + eb)
    dE_ds = k * (-1 / sig) * (a * ea - b * eb)
    return np.stack([np.ones_like(t), t, -E / 2, -A / 2 * dE_dc, -A / 2 * dE_dw, -A / 2 * dE_ds], 1)


def fit_width(p, t, w0, c0=0.0, sig0=None, sig_weight=0.0):
    """Least-squares fit of the blurred-box model to one profile.

    If `sig0` is given, a prior residual  sig_weight * sqrt(n) * (sig - sig0)
    is appended. For vessels much narrower than the PSF the width and the
    blur are degenerate (only their product with the contrast is visible);
    the prior keeps sigma near the optical blur measured on large vessels.
    """
    k = max(2, len(p) // 5)
    b0 = np.median(np.r_[p[:k], p[-k:]])
    A0 = max(b0 - p.min(), 1e-6)
    x0 = [b0, 0.0, A0, c0, max(w0, 1.5), sig0 or 1.2]
    hw = t[-1]
    lo = [-np.inf, -np.inf, 0, -hw / 2, 0.5, 0.3]
    hi = [np.inf, np.inf, np.inf, hw / 2, 1.6 * hw, 6.0]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    use_prior = sig0 is not None and sig_weight > 0
    lam = sig_weight * np.sqrt(len(t)) * A0

    def fun(q):
        res = box_blur_model(t, *q) - p
        return np.r_[res, lam * (q[5] - sig0)] if use_prior else res

    def jac(q):
        J = box_blur_jac(t, *q)
        if use_prior:
            J = np.vstack([J, [0, 0, 0, 0, 0, lam]])
        return J
    try:
        r = least_squares(fun, x0, jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=100)
    except Exception:
        return dict(w=np.nan, c=np.nan, A=np.nan, sig=np.nan, r2=np.nan, params=None)
    res = r.fun[:len(t)]
    ss = np.sum((p - p.mean()) ** 2) + 1e-12
    return dict(w=r.x[4], c=r.x[3], A=r.x[2], sig=r.x[5], r2=1 - np.sum(res**2) / ss, params=r.x)


def measure_diameter(img, sp: VesselSpline, d_guess: float, avg: int = 3, dt: float = 0.5,
                     hw_factor: float = 1.6, min_hw: float = 8.0, exclude_pts=(), exclude_r=None,
                     stride: int = 2, sig0=None, sig_weight=0.0):
    """Diameter profile along a vessel.

    Returns dict of arrays over sp.s: d_fwhm, d_fit, center_offset, contrast, sigma_psf, r2, valid,
    subres. Profiles are fitted every `stride` samples (they are already
    averaged over +-avg samples) and linearly interpolated in between.
    `exclude_pts` (e.g. junction / crossing positions) invalidate samples within
    `exclude_r` px, because neighbouring vessels contaminate the profile there.
    """
    hw = max(min_hw, hw_factor * d_guess)
    P, t = cross_profiles(img, sp, hw, dt)
    Ps = smooth_along(P, avg)
    n = len(sp.s)
    keys = ("d_fwhm", "d_fit", "center_offset", "contrast", "sigma_psf", "r2")
    out = {k: np.full(n, np.nan) for k in keys}
    fit_at = np.unique(np.r_[np.arange(0, n, max(1, stride)), n - 1])
    for i in fit_at:
        wf, cf, dep = fwhm_width(Ps[i], t)
        out["d_fwhm"][i] = wf
        f = fit_width(Ps[i], t, wf if np.isfinite(wf) else d_guess, cf if np.isfinite(cf) else 0.0,
                      sig0=sig0, sig_weight=sig_weight)
        out["d_fit"][i] = f["w"]; out["center_offset"][i] = f["c"]; out["contrast"][i] = f["A"]
        out["sigma_psf"][i] = f["sig"]; out["r2"][i] = f["r2"]
    if len(fit_at) < n:
        s_all = np.arange(n)
        for k in keys:
            y = out[k][fit_at]; ok = np.isfinite(y)
            out[k] = np.interp(s_all, fit_at[ok], y[ok]) if ok.sum() >= 2 else out[k]
            if ok.sum() >= 2:   # do not interpolate across failed fits
                bad = ~np.isfinite(np.interp(s_all, fit_at, np.where(ok, 0.0, np.nan)))
                out[k][bad] = np.nan
    base = np.nanmedian(Ps[:, :max(2, len(t) // 5)])
    out["contrast"] = out["contrast"] / (abs(base) + 1e-12)         # fractional dip depth
    # primary diameter: FWHM (model-free). The model fit is used as a shape
    # check: a sample is valid only if the profile looks like a single dip.
    out["d"] = out["d_fwhm"].copy()
    valid = np.isfinite(out["d"]) & np.isfinite(out["d_fit"]) & (out["r2"] > 0.8)
    if exclude_r is None:
        exclude_r = 1.0 * max(d_guess, 5)
    for (x, y) in exclude_pts:
        valid &= np.hypot(sp.xy[:, 0] - x, sp.xy[:, 1] - y) > exclude_r
    # robust outlier rejection against a running median
    if valid.sum() > 5:
        med = ndi.median_filter(np.where(valid, out["d"], np.nanmedian(out["d"][valid])), 15, mode="nearest")
        valid &= np.abs(out["d"] - med) < 0.35 * med + 1.0
    out["valid"] = valid
    # the blurred-box fit collapses (w << sigma) when the vessel is narrower
    # than the edge blur: its width is then not resolved
    out["subres"] = out["d_fit"] < 1.0 * np.nan_to_num(out["sigma_psf"], nan=1.5)
    out["profiles"] = Ps; out["t"] = t
    return out


def smooth_diameter(d, valid, win=11):
    """Interpolate over invalid samples and median/mean filter -> smooth d(s) for masks."""
    s = np.arange(len(d))
    if valid.sum() < 2:
        fill = np.nanmedian(d) if np.isfinite(d).any() else 3.0
        return np.full(len(d), fill)
    dd = np.interp(s, s[valid], d[valid])
    dd = ndi.median_filter(dd, win, mode="nearest")
    return ndi.uniform_filter1d(dd, win, mode="nearest")


# ----------------------------------------------------------------------------- masks
def vessel_polygon(sp: VesselSpline, d_smooth, offset=None):
    """Closed outline: centreline +- d/2 along the normal (optionally re-centred)."""
    c = np.zeros(len(sp.s)) if offset is None else np.nan_to_num(offset)
    left = sp.xy + (c + d_smooth / 2)[:, None] * sp.N
    right = sp.xy + (c - d_smooth / 2)[:, None] * sp.N
    return np.vstack([left, right[::-1]])


def rasterize(poly, shape, cap_radius=None, sp=None, d_smooth=None):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly * 8).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=3)
    if sp is not None and d_smooth is not None:   # round caps at the ends
        for j in (0, -1):
            cv2.circle(m, tuple(int(v) for v in np.round(sp.xy[j])), int(round(d_smooth[j] / 2)), 1, -1)
    return m.astype(bool)


# ----------------------------------------------------------------------------- tortuosity
def tortuosity(sp: VesselSpline, kappa_floor=0.004, trim=0):
    s, k, xy = sp.s, sp.kappa, sp.xy
    if trim:
        s, k, xy = s[trim:-trim], k[trim:-trim], xy[trim:-trim]
    L = s[-1] - s[0]
    C = float(np.linalg.norm(xy[-1] - xy[0]))
    ds = np.gradient(s)
    TC = float(np.sum(np.abs(k) * ds))
    KSQ = float(np.sum(k**2 * ds) / L) if L > 0 else np.nan
    sig = np.sign(np.where(np.abs(k) < kappa_floor, 0, k))
    nz = sig[sig != 0]
    infl_idx = np.flatnonzero(np.diff(nz) != 0)
    n_infl = int(len(infl_idx))
    # positions of inflections along the curve (for plotting)
    nz_pos = np.flatnonzero(sig != 0)
    infl_s = s[nz_pos[infl_idx]] if n_infl else np.array([])
    # a closed loop (start ~ end) has no meaningful chord: DM / ICM undefined
    is_loop = C < max(2.0, 0.1 * L)
    DM = np.nan if is_loop else L / C
    return dict(length=L, chord=C, DM=DM, SOAM=TC / L if L > 0 else np.nan, TC=TC, KSQ=KSQ,
                n_inflections=n_infl, ICM=(n_infl + 1) * DM, max_abs_kappa=float(np.max(np.abs(k))),
                inflection_s=infl_s, loop=bool(is_loop))
