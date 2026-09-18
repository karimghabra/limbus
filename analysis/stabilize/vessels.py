"""Vessel detection: glare exclusion, Hessian vesselness and the vessel
envelope mask (METHODS.md §4–§6)."""
import cv2
import numpy as np


def to_working(img, scale):
    """Downscale with area averaging, which low-pass filters before
    discarding pixels and so avoids aliasing (§2)."""
    img = img.astype(np.float32, copy=False)
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA)


def glare_mask(img_ws, full_scale, params, scale):
    """Clipped highlights, dilated to cover their bloom (§4). Specular
    reflection off the tear film is fixed by the light and camera geometry,
    so left in it would anchor registration to the camera, not the eye."""
    clipped = (img_ws >= params.glare_fraction_of_full_scale * full_scale)
    if not clipped.any():
        return clipped
    k = max(3, int(round(params.glare_dilate_px * scale)) | 1)
    # a round kernel: bloom spreads radially, and a square one leaves
    # square no-data holes that read as artefacts
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(clipped.astype(np.uint8), disk) > 0


def working_sigmas(params, scale, height_ws):
    """Filter scales in working-resolution pixels, dropping any too large
    for the image — a 100-row strip can't support a 32 px vessel filter."""
    sig = [s * scale for s in params.sigmas_px]
    usable = [s for s in sig if 4 * s <= height_ws]
    return usable or [min(sig)]


def _hessian_eigenvalues(img, sigma):
    """Eigenvalues of the scale-normalised Hessian, ordered |l1| <= |l2|.

    Derivatives are of the Gaussian-blurred image, so sigma acts as a ruler
    for vessel width; multiplying by sigma^2 (scale normalisation) lets
    different scales compete fairly, since blurring shrinks derivatives.
    """
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    norm = sigma * sigma
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) * norm
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) * norm
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) * norm
    root = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
    a = 0.5 * (dxx + dyy - root)
    b = 0.5 * (dxx + dyy + root)
    swap = np.abs(a) > np.abs(b)
    return np.where(swap, b, a), np.where(swap, a, b)


def contrast_constant(img_ws, sigmas_ws, params):
    """Frangi's c: the curvature that counts as 'strong'. Set ONCE per burst
    from a reference frame (§5) — normalising each frame separately rescales
    every map differently and breaks matching between distant frames."""
    mid = sigmas_ws[len(sigmas_ws) // 2]
    l1, l2 = _hessian_eigenvalues(img_ws, mid)
    s = np.sqrt(l1 ** 2 + l2 ** 2)
    return 0.5 * float(np.percentile(s, params.contrast_percentile)) + 1e-6


def _vesselness_one_scale(img, sigma, c, beta):
    l1, l2 = _hessian_eigenvalues(img, sigma)
    rb = l1 / (l2 + np.where(l2 >= 0, 1e-6, -1e-6))
    s2 = l1 ** 2 + l2 ** 2
    v = np.exp(-(rb ** 2) / (2 * beta ** 2)) * (1.0 - np.exp(-s2 / (2 * c ** 2)))
    # a DARK vessel is an intensity valley: positive curvature across it
    v[l2 <= 0] = 0.0
    return v


def vesselness(img_ws, sigmas_ws, c, params):
    """Frangi vesselness for dark tubular structures, max over scales (§5).

    Large scales are computed on a 2x-downsampled image and upsampled: the
    scale-normalised response is scale-invariant, so this gives the same
    answer far faster than blurring a full image with a huge kernel.
    """
    h, w = img_ws.shape
    out = np.zeros((h, w), np.float32)
    half = None
    for s in sigmas_ws:
        if s >= 8 and min(h, w) >= 64:
            if half is None:
                half = cv2.resize(img_ws, (w // 2, h // 2),
                                  interpolation=cv2.INTER_AREA)
            v = _vesselness_one_scale(half, s / 2.0, c, params.frangi_beta)
            v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            v = _vesselness_one_scale(img_ws, s, c, params.frangi_beta)
        np.maximum(out, v, out=out)
    return out


def envelope(v, threshold):
    """Binary vessel envelope: burst-wide threshold, then a small opening to
    drop isolated noise pixels (§6). Thresholding keeps geometry and discards
    amplitude, which is why it survives illumination changes."""
    m = (v > threshold).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
