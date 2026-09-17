"""Frame quality gate (METHODS.md §3): find blinks, blur and lighting jumps.

Frames are flagged, never deleted — the raw burst stays the record of what
happened, and the flags say which frames the analysis trusted.
"""
import cv2
import numpy as np


def frame_stats(img_ws, full_scale, params):
    """Sharpness, brightness and clipping for one working-scale frame.

    Sharpness is the variance of the Laplacian: high-frequency energy, which
    blur (a low-pass filter) and blinks (a featureless eyelid) both remove.
    It varies multiplicatively and is heavily skewed, hence the log.
    """
    smooth = cv2.GaussianBlur(img_ws, (0, 0), params.sharpness_blur_sigma)
    sharp = float(cv2.Laplacian(smooth, cv2.CV_32F).var())
    clipped = img_ws >= params.glare_fraction_of_full_scale * full_scale
    return np.log(sharp + 1.0), float(img_ws.mean()), float(clipped.mean())


def robust_z(x):
    """(x - median) / (1.4826 * MAD). The median and MAD ignore up to half
    the data being outliers, so the blinks we're hunting can't inflate the
    scale and hide themselves the way they would with mean and SD. 1.4826
    makes the MAD equal the SD for normally distributed data."""
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad <= 1e-12:
        return np.zeros_like(x)
    return (x - med) / mad


def gate(log_sharpness, means, params):
    """Boolean keep-mask plus a reason string for every rejected frame.

    A frame must be BOTH statistically unusual (robust z) AND physically
    different (a sharpness or brightness floor) to be rejected. In a very
    steady burst the MAD is tiny, so z alone would reject frames over
    trivial wobbles — the synthetic still burst lost a good frame that way.
    """
    z_sharp = robust_z(log_sharpness)
    z_mean = robust_z(means)
    sharp_ratio = np.exp(np.asarray(log_sharpness) - np.median(log_sharpness))
    rel_bright = np.abs(np.asarray(means) / max(np.median(means), 1e-9) - 1.0)
    blurred = (z_sharp < params.sharpness_min_z) & (sharp_ratio < params.sharpness_min_ratio)
    lighting = ((np.abs(z_mean) > params.brightness_max_abs_z)
                & (rel_bright > params.brightness_max_rel_change))
    reasons = []
    for b, l in zip(blurred, lighting):
        why = []
        if b:
            why.append("blink/blur")
        if l:
            why.append("brightness")
        reasons.append("+".join(why))
    return ~(blurred | lighting), reasons, z_sharp, z_mean
