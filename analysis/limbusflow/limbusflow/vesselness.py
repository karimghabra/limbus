"""Multi-scale Hessian vessel enhancement (Frangi et al., MICCAI 1998) and segmentation.

Why the Hessian?
----------------
Around a point x0 the image intensity is approximated by its second-order
Taylor expansion

    I(x0 + d) ~ I(x0) + d^T grad I + 1/2 d^T H d,      H = [[Ixx, Ixy], [Ixy, Iyy]]

The eigen-decomposition H = Q diag(l1, l2) Q^T (|l1| <= |l2|) gives the
principal curvatures of the intensity landscape. A dark vessel on a bright
background is a *valley*:

    across the vessel : strong positive curvature   -> l2 >> 0
    along the vessel  : almost flat                  -> l1 ~ 0

Derivatives are taken at scale sigma by convolving with derivatives of a
Gaussian, and multiplied by sigma^2 ("scale normalisation") so responses at
different scales are comparable. The Frangi vesselness is

    V_sigma = 0                                  if l2 < 0   (bright ridge)
            = exp(-Rb^2 / 2 beta^2) * (1 - exp(-S^2 / 2 c^2))  otherwise
    Rb = l1 / l2          ("blobness": 0 for a line, 1 for a blob)
    S  = sqrt(l1^2 + l2^2)  ("structureness": 0 in flat background)

and V = max_sigma V_sigma. The arg-max scale is a rough indicator of the
vessel radius; the actual diameter is measured later from intensity profiles.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure


def flatfield(img, sigma=25.0):
    img = np.asarray(img, np.float32)
    return img / (ndi.gaussian_filter(img, sigma) + 1e-6)


def hessian(img, sigma):
    """Scale-normalised Hessian components (sigma^2 * second Gaussian derivatives)."""
    img = np.asarray(img, np.float64)
    Ixx = ndi.gaussian_filter(img, sigma, order=(0, 2)) * sigma**2
    Iyy = ndi.gaussian_filter(img, sigma, order=(2, 0)) * sigma**2
    Ixy = ndi.gaussian_filter(img, sigma, order=(1, 1)) * sigma**2
    return Ixx, Ixy, Iyy


def hessian_eigen(Ixx, Ixy, Iyy):
    """Closed-form eigenvalues of the symmetric 2x2 Hessian, sorted |l1| <= |l2|.

    Also returns the unit eigenvector (vx, vy) of l1, i.e. the local vessel direction.
    """
    tr = Ixx + Iyy
    disc = np.sqrt((Ixx - Iyy) ** 2 + 4 * Ixy**2)
    a = 0.5 * (tr + disc); b = 0.5 * (tr - disc)
    swap = np.abs(a) < np.abs(b)
    l1 = np.where(swap, a, b); l2 = np.where(swap, b, a)
    # eigenvector for l1: (Ixy, l1 - Ixx) (or (l1 - Iyy, Ixy) when degenerate)
    vx, vy = Ixy, l1 - Ixx
    deg = (np.abs(vx) + np.abs(vy)) < 1e-12
    vx = np.where(deg, 1.0, vx); vy = np.where(deg, 0.0, vy)
    n = np.hypot(vx, vy)
    return l1, l2, vx / n, vy / n


def frangi_single(l1, l2, beta=0.5, c=None, dark=True):
    Rb2 = (l1 / np.where(l2 == 0, 1e-12, l2)) ** 2
    S2 = l1**2 + l2**2
    if c is None:
        c = 0.5 * np.sqrt(S2.max())
    V = np.exp(-Rb2 / (2 * beta**2)) * (1 - np.exp(-S2 / (2 * c**2)))
    V[(l2 < 0) if dark else (l2 > 0)] = 0
    return V


def frangi(img, sigmas=(1.5, 2, 3, 4, 6, 8), beta=0.5, c=None, dark=True, return_all=False):
    """Multi-scale vesselness. Returns V, best_sigma (and per-scale stacks if asked)."""
    Vs, L1, L2 = [], [], []
    for s in sigmas:
        l1, l2, _, _ = hessian_eigen(*hessian(img, s))
        Vs.append(frangi_single(l1, l2, beta, c, dark)); L1.append(l1); L2.append(l2)
    Vs = np.stack(Vs)
    k = Vs.argmax(0)
    V = Vs.max(0)
    best = np.asarray(sigmas)[k]
    if return_all:
        return V, best, dict(V=Vs, l1=np.stack(L1), l2=np.stack(L2), sigmas=np.asarray(sigmas))
    return V, best


def segment(V, valid=None, low_pct=85, high_pct=95, min_size=300, fill_holes=64, closing=1):
    """Hysteresis threshold of the vesselness map -> binary vessel mask.

    Pixels above the high threshold seed vessels; connected pixels above the
    low threshold are added. Small islands are removed and small holes filled.
    """
    vv = V[valid] if valid is not None else V.ravel()
    lo, hi = np.percentile(vv, low_pct), np.percentile(vv, high_pct)
    m = filters.apply_hysteresis_threshold(V, lo, hi)
    if valid is not None:
        m &= valid
    if closing:
        m = ndi.binary_closing(m, morphology.disk(closing))
    m = morphology.remove_small_holes(m, max_size=fill_holes)
    m = morphology.remove_small_objects(m, max_size=min_size)
    return m, (lo, hi)


def skeletonize(mask):
    """1-px-wide medial axis (Zhang-Suen style thinning via skimage)."""
    return morphology.skeletonize(mask)
