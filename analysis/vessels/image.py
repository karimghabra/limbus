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


def remove_fixed_pattern(A, valid, width=5):
    """Subtract per-row and per-column offsets.

    A sensor's row and column offsets survive averaging and look exactly like
    long, perfectly straight vessels to any line detector - they were being
    detected as vessels with zero deviation from a single image row. The
    The offset of a row is the median of that row, which a vessel crossing it
    barely moves - but a vessel that runs ALONG a row would be removed with
    the artefact. So only the NARROW part of the offset profile is taken: the
    profile minus its own running median over `width` rows. With width 5 an
    offset one or two rows wide is removed completely, while a vessel running
    the full width of the frame along a row - the worst case, since a vessel
    crossing rows is untouched - keeps 80-97 % of its depth.
    """
    A = np.asarray(A, np.float32).copy()
    m = np.where(valid, A, np.nan)

    def narrow(prof):
        prof = np.where(np.isfinite(prof), prof, 0).astype(np.float64)
        pad = width // 2
        padded = np.pad(prof, pad, mode="edge")
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)
        return (prof - smooth).astype(np.float32)

    row = narrow(np.nanmedian(m, axis=1))
    col = narrow(np.nanmedian(m, axis=0))
    return A - row[:, None] - col[None, :]


def prepare(mean, close_px=101, erode_px=15, fixed_pattern=True, border_px=25):
    """(absorbance, valid) from a raw averaged frame. border_px keeps the
    detector away from the edge of the covered region, where a few frames
    contributed and the average is neither flat nor complete."""
    valid = valid_mask(mean, erode_px)
    A = flat_absorbance(mean, valid, close_px)
    if fixed_pattern:
        A = remove_fixed_pattern(A, valid)
    if border_px:
        valid = cv2.erode(valid.astype(np.uint8),
                          np.ones((2 * border_px + 1,) * 2, np.uint8)) > 0
    return np.where(valid, A, 0).astype(np.float32), valid
