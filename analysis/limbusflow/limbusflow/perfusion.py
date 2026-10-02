"""Functional vessel detection: where does the image *flicker* over time?

Red-cell aggregates and plasma gaps moving through a vessel make its lumen
intensity fluctuate from frame to frame; static tissue does not. The temporal
standard deviation of the registered, temporally high-passed video is therefore
a map of *perfused* vessels (the same idea as motion-contrast angiography /
OCT-A). It reveals thin vessels that are barely visible in the mean image.

Model for one pixel x of the registered, brightness-normalised video F_t(x):

    F_t(x) = m(x) + b_t(x) + j_t(x) + n_t(x)

    m   static mean
    b   blood flicker            (what we want)
    j   residual-jitter leakage  ~ grad m(x) . delta_t   (delta_t = residual misregistration)
    n   sensor noise             variance ~ a + c m(x)   (shot + read noise)

After removing slow changes (running temporal mean over sigma_t frames), the
variance is

    var_t(x) ~ var(b) + sigma_delta^2 |grad m(x)|^2 + a + c m(x)

The jitter and noise terms are estimated by robust least squares over
*background* pixels (no vessel) and subtracted, leaving

    flicker(x) = sqrt(max(var_t - jitter - noise, 0)) / m(x)

Running means are computed by *normalised convolution*
(G * (F w)) / (G * w), with w = 0 where a pixel was outside the frame or
masked as a specular reflection, so partially covered pixels are unbiased.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import register as rg


def registered_stack(frames, reg, idx, mask_specular=True):
    """(T, H, W) float32 registered frames, each divided by its median; NaN outside."""
    idx = list(idx)
    out = np.empty((len(idx),) + reg.shape, np.float32)
    for n, i in enumerate(idx):
        raw = np.asarray(frames[i])
        f = raw.astype(np.float32)
        if mask_specular:
            f[rg.specular_mask(raw)] = np.nan
        w = rg.warp(f, reg.M[i], reg.shape)
        out[n] = w / np.nanmedian(w)
    return out


def flicker_map(stack, t_window=10.0, bg_mask=None, min_cover=0.9, return_parts=False):
    """Jitter- and noise-corrected temporal flicker (relative units).

    stack   : (T, H, W) registered video (NaN = missing)
    bg_mask : boolean background (non-vessel) pixels used to fit the jitter /
              noise model; if None, the pixels below the 60th flicker
              percentile are used.
    """
    w = np.isfinite(stack).astype(np.float32)
    x = np.nan_to_num(stack, nan=0.0)
    num = ndi.gaussian_filter1d(x * w, t_window, axis=0, mode="nearest")
    den = ndi.gaussian_filter1d(w, t_window, axis=0, mode="nearest")
    hp = (x - num / np.maximum(den, 1e-6)) * w
    del num, den
    cnt = w.sum(0)
    cover = cnt / len(stack)
    var = (hp**2).sum(0) / np.maximum(cnt - 1, 1)
    del hp
    m = (x * w).sum(0) / np.maximum(cnt, 1)
    gy, gx = np.gradient(ndi.gaussian_filter(m, 1.0))
    g2 = gx**2 + gy**2
    ok = cover >= min_cover
    if bg_mask is None:
        raw = np.sqrt(var) / np.maximum(m, 1e-6)
        bg_mask = ok & (raw < np.percentile(raw[ok], 60))
    else:
        bg_mask = bg_mask & ok
    # robust least squares  var = a + c m + s2 |grad m|^2  on background pixels
    A = np.stack([np.ones(bg_mask.sum()), m[bg_mask], g2[bg_mask]], 1)
    y = var[bg_mask]
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(3):                                  # iteratively reweighted (Huber-like)
        r = y - A @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        wt = 1 / np.maximum(1, np.abs(r) / (2 * s))
        coef = np.linalg.lstsq(A * wt[:, None], y * wt, rcond=None)[0]
    a, c, s2 = coef
    model = a + c * m + max(s2, 0) * g2
    fl = np.sqrt(np.maximum(var - model, 0)) / np.maximum(m, 1e-6)
    fl[~ok] = 0
    if return_parts:
        return fl, dict(mean=m, var=var, grad2=g2, cover=cover, model=model, coef=dict(a=a, c=c, sigma_delta2=s2),
                        bg_mask=bg_mask)
    return fl


def combined_vesselness(ff, fl, valid, sigmas_struct=(1, 1.5, 2, 3, 4, 6, 8), sigmas_func=(1, 1.5, 2, 3, 4, 6),
                        beta=0.5, thick_radius=4, thick_dilate=0, norm_pct=99.5):
    """Structural (dark ridges of the flat-fielded mean) + functional (bright
    ridges of the flicker map) vesselness, each normalised to [0, 1] by its
    `norm_pct` percentile inside `valid`, combined by the maximum.

    Inside *thick* structural vessels (mask opened by a disc of radius
    `thick_radius`) the functional term is suppressed: residual edge flicker
    there draws parallel ridges that are not separate vessels. The band is not
    dilated, so thin branches keep their connection to the parent vessel
    (whose own structural response fills the interior).
    Returns V, Vs, Vf, thick_mask.
    """
    from skimage import morphology
    from . import vesselness as vs
    Vs, _ = vs.frangi(ff, sigmas_struct, beta, None, dark=True)
    Vf, _ = vs.frangi(fl, sigmas_func, beta, None, dark=False)
    Vs = np.clip(Vs * valid / (np.percentile(Vs[valid], norm_pct) + 1e-12), 0, 1)
    Vf = np.clip(Vf * valid / (np.percentile(Vf[valid], norm_pct) + 1e-12), 0, 1)
    ms, _ = vs.segment(Vs, valid, 80, 94, min_size=300)
    thick = ndi.binary_opening(ms, morphology.disk(thick_radius)) if thick_radius else np.zeros_like(ms)
    if thick_dilate:
        thick = ndi.binary_dilation(thick, iterations=thick_dilate)
    V = np.maximum(Vs, np.where(thick, 0.0, Vf))
    return V, Vs, Vf, thick
