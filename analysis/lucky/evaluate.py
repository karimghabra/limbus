"""Scoring annotations (METHODS.md §L4).

Centreline agreement with a distance tolerance: a detected centreline pixel
is a true positive if a reference centreline lies within `tol` px, and a
reference pixel is recovered if a detected one does. Precision and recall are
the two fractions, F1 their harmonic mean. With ground truth (synthetic) the
reference is the truth; on real data it is the annotation of an independent
half of the frames (split-half reproducibility).
"""
import numpy as np
from scipy import ndimage as ndi


def centreline_agreement(detected, reference, tol, region=None):
    if region is not None:
        detected = detected & region
        reference = reference & region
    nd, nr = int(detected.sum()), int(reference.sum())
    if nd == 0 or nr == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    d_ref = ndi.distance_transform_edt(~reference)
    d_det = ndi.distance_transform_edt(~detected)
    precision = float(np.count_nonzero(detected & (d_ref <= tol)) / nd)
    recall = float(np.count_nonzero(reference & (d_det <= tol)) / nr)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def edge_sharpness(c_img, mask, skel, valid):
    """Median steepest contrast gradient across vessel edges, per unit of
    that vessel's contrast depth: 1/(edge width in px). Blur widens edges, so
    it falls; noise barely moves a median over thousands of edge pixels."""
    gy, gx = np.gradient(c_img)
    g = np.hypot(gx, gy)
    edge = mask ^ ndi.binary_erosion(mask)
    edge &= valid
    if not edge.any() or not skel.any():
        return 0.0
    # depth of the nearest centreline pixel = local vessel contrast
    _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    depth = np.abs(c_img[iy, ix])
    ok = edge & (depth > 0.02)
    if not ok.any():
        return 0.0
    return float(np.median(g[ok] / depth[ok]))


def background_noise(c_img, mask, valid):
    """Robust SD of the finest detail on bare tissue (vessels dilated out):
    the noise an annotator has to see capillaries through."""
    lap = c_img - ndi.uniform_filter(c_img, 3)
    region = valid & ~ndi.binary_dilation(mask, iterations=6)
    x = lap[region]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))
