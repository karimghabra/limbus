"""Loading a single still image and turning it into the fitting domain.

Everything here works on ONE image. No temporal information (frame
differences, kymographs, flow) is used anywhere in vesselmap.

Vessels absorb light, so in transmission/reflection the recorded intensity
is I = B * exp(-OD) with B the illumination/background and OD the optical
density of the blood column.  In the log domain the model becomes additive,

    log I = log B - sum_k OD_k,

so overlapping vessels (crossings at different depths) simply add up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

FULL_SCALE_12BIT = 4095.0


def load_image(path: str) -> np.ndarray:
    """Load a greyscale image as float32 in [0, 1].

    LIMBUS TIFFs hold right-aligned 12-bit data, so 16-bit files whose
    maximum fits in 12 bits are divided by 4095.  8-bit images by 255.
    """
    p = str(path).lower()
    if p.endswith((".tif", ".tiff")):
        import tifffile
        a = tifffile.imread(path)
    else:
        import cv2
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(path)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
        # a registered mean is NaN where no frame covered it
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
    if a.dtype == np.uint8:
        scale = 255.0
    elif a.dtype == np.uint16:
        scale = FULL_SCALE_12BIT if a.max() <= 4095 else 65535.0
    else:
        # float images in sensor DN (e.g. a registered 12-bit mean) keep the 12-bit scale
        scale = (FULL_SCALE_12BIT if a.max() <= FULL_SCALE_12BIT else float(a.max())) if a.max() > 1.0 else 1.0
    return (a.astype(np.float32) / scale).clip(0.0, 1.0)


@dataclass
class Prepared:
    """An image in the log domain plus its per-pixel weights."""
    intensity: np.ndarray   # [0,1] float32
    logI: np.ndarray        # log intensity, float32
    sigma: np.ndarray       # per-pixel noise std of logI (float32)
    valid: np.ndarray       # bool mask: False where saturated / near-black

    @property
    def shape(self):
        return self.logI.shape

    @property
    def weight(self) -> np.ndarray:
        """Inverse-variance weights, zero on invalid pixels."""
        w = 1.0 / np.square(self.sigma)
        return np.where(self.valid, w, 0.0).astype(np.float32)


def robust_noise_map(logI: np.ndarray, tile: int = 64) -> np.ndarray:
    """Per-pixel noise std of logI from the MAD of a discrete Laplacian.

    The Laplacian of white noise with std s has std s*sqrt(20).  Vessel edges
    are sparse, so the median absolute deviation ignores them.  The estimate
    is computed per tile and smoothly interpolated, which captures the
    brightness dependence of shot noise.
    """
    lap = ndi.convolve(logI.astype(np.float64),
                       np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
                       mode="reflect")
    h, w = logI.shape
    ny, nx = max(1, h // tile), max(1, w // tile)
    est = np.zeros((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            blk = lap[iy * h // ny:(iy + 1) * h // ny,
                      ix * w // nx:(ix + 1) * w // nx]
            est[iy, ix] = 1.4826 * np.median(np.abs(blk - np.median(blk)))
    est /= np.sqrt(20.0)
    est = ndi.median_filter(est, size=3, mode="nearest")
    import cv2
    full = cv2.resize(est.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    floor = max(1e-4, 0.25 * float(np.median(est)))
    return np.maximum(full, floor).astype(np.float32)


def prepare(intensity: np.ndarray, sat_level: float = 0.995,
            dark_level: float = 1e-3) -> Prepared:
    """Convert a [0,1] intensity image to the log domain with noise weights."""
    I = np.asarray(intensity, np.float32)
    valid = (I < sat_level) & (I > dark_level)
    # grow the saturated specular spots a little: their surroundings bloom
    valid = ~ndi.binary_dilation(~valid, iterations=2)
    logI = np.log(np.maximum(I, dark_level)).astype(np.float32)
    sigma = robust_noise_map(logI)
    return Prepared(I, logI, sigma, valid)
