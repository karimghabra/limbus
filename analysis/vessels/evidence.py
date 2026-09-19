"""Ridge evidence: how vessel-like each pixel is, and which way the ridge runs.

The Hessian of the blurred transmission measures curvature; on a dark line
the curvature across the line is large and positive while along it is small,
so l2 - |l1| of the Hessian eigenvalues responds to lines and not to edges
or blobs. Each scale is put in units of ITS OWN robust spread (median and
MAD over the valid pixels) before the scales are combined by a maximum:
coarse scales respond strongly to the sclera's texture blotches, and without
this normalisation they would drown the thin vessels that the fine scales
see. The value is therefore a robust z score, comparable between images.

Small sigmas (1-3 px) are what small vessels need: their measured width is
5-8 px FWHM at this magnification.
"""
import cv2
import numpy as np

SIGMAS = (1.0, 1.5, 2.0, 3.0)


def ridge_z(A, valid, sigmas=SIGMAS, with_angle=False):
    """Robust-z ridge strength of the absorbance A (vessels are positive).

    With with_angle, also returns the angle of the Hessian eigenvector ACROSS
    the ridge at the winning scale, for non-maximum suppression."""
    T = np.exp(-A).astype(np.float32)          # ridges are dark in transmission
    best = np.full(A.shape, -np.inf, np.float32)
    ang = np.zeros(A.shape, np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(T, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) / 4
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) / 4
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4
        tr, dif = dxx + dyy, np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l2, l1 = 0.5 * (tr + dif), 0.5 * (tr - dif)
        v = s ** 2 * np.where(l2 > 0, l2 - np.abs(l1), 0)
        m = np.median(v[valid])
        mad = 1.4826 * np.median(np.abs(v[valid] - m))
        z = (v - m) / max(mad, 1e-12)
        if with_angle:
            better = z > best
            ang = np.where(better, 0.5 * np.arctan2(2 * dxy, dxx - dyy), ang)
        np.maximum(best, z, out=best)
    best[~valid] = 0
    return (best, ang) if with_angle else best


def nms(z, ang):
    """Keep only pixels that are a maximum of z along the direction ACROSS the
    ridge: two vessels running side by side then keep separate centrelines
    instead of merging into one thresholded blob."""
    H, W = z.shape
    nx, ny = np.cos(ang), np.sin(ang)
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z1 = cv2.remap(z, X + nx, Y + ny, cv2.INTER_LINEAR)
    z2 = cv2.remap(z, X - nx, Y - ny, cv2.INTER_LINEAR)
    return (z >= z1) & (z >= z2)
