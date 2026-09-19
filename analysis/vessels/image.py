"""Image preparation: the averaged stabilized frame -> absorbance.

Beer-Lambert: a vessel absorbs, so work with A = -log(T), where T is the
frame divided by its own illumination. The illumination comes from a
morphological closing wider than any vessel, which removes every dark line
and leaves vignetting and lamp shading. In A a vessel is a positive ridge
whose height is its peak absorbance, which is what the physical model fits.
"""
import cv2
import numpy as np


def valid_mask(mean, erode_px=15):
    """Pixels with data everywhere in their neighbourhood (stabilized frames
    have no-data borders where no frame covered them)."""
    return cv2.erode(np.isfinite(mean).astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)) > 0


def flat_absorbance(mean, valid, close_px=101):
    f = np.where(valid, mean, np.nan)
    f = np.where(np.isfinite(f), f, np.nanmedian(f)).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    illum = cv2.GaussianBlur(cv2.morphologyEx(f, cv2.MORPH_CLOSE, k), (0, 0), close_px / 3)
    T = f / np.maximum(illum, 1e-6)
    return -np.log(np.clip(T, 1e-3, None)).astype(np.float32)


def local_noise(img, valid, floor_pct=5.0):
    """Local standard deviation of the fine-scale content, with a floor so a
    quiet patch cannot give a vanishing noise estimate."""
    fine = img - cv2.GaussianBlur(img, (0, 0), 2.0)
    sd = np.sqrt(cv2.GaussianBlur(fine ** 2, (0, 0), 25.0))
    return np.maximum(sd, np.percentile(sd[valid], floor_pct))


def prepare(mean, close_px=101, erode_px=15):
    """(absorbance, valid) from a raw averaged frame."""
    valid = valid_mask(mean, erode_px)
    return flat_absorbance(mean, valid, close_px), valid
