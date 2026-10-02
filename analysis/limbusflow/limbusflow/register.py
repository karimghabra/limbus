"""Frame-to-reference registration.

The eye moves continuously (drift, micro-saccades, larger saccades, blinks),
so every frame of the burst has to be mapped into the coordinate frame of the
reference (mean) image before anything can be measured along a vessel.

Model
-----
Each frame F_t is related to the reference R by a 2-D similarity transform

    [x_R]   [ s cos(th)  -s sin(th)   tx ] [x_F]
    [y_R] = [ s sin(th)   s cos(th)   ty ] [y_F]
                                           [ 1 ]

(4 degrees of freedom: scale s, rotation th, translation tx, ty). Over the
small field of view the curvature of the globe is negligible and s ~ 1.

Algorithm
---------
1. "Enhance" both images: divide by a heavily blurred copy (flat-field) so
   the uneven illumination does not dominate, then stretch to 8 bit.
2. Detect SIFT keypoints / descriptors (scale-invariant blob features that
   land on vessel bends and branch points).
3. Match descriptors (nearest neighbour + Lowe ratio test).
4. Fit the similarity with RANSAC: repeatedly fit to 2 random matches, count
   matches that agree within 3 px ("inliers"), keep the best, refit to all
   inliers.
5. Refine to sub-pixel accuracy with ECC (enhanced correlation coefficient)
   maximisation, initialised at the RANSAC estimate.
6. Score the result by the normalised cross-correlation (NCC) between the
   warped frame and the reference.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


def flatfield(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Divide out slowly varying illumination: I / (G_sigma * I)."""
    a = img.astype(np.float32)
    return a / (cv2.GaussianBlur(a, (0, 0), sigma) + 1e-3)


def enhance(img: np.ndarray, sigma_bg: float = 25.0, sigma_smooth: float = 1.0) -> np.ndarray:
    """Flat-field + light denoise + robust 8-bit stretch (used for keypoints)."""
    a = cv2.GaussianBlur(flatfield(img, sigma_bg), (0, 0), sigma_smooth)
    lo, hi = np.percentile(a, [1, 99])
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def specular_mask(frame: np.ndarray, level: float = 0.93, dilate: int = 15) -> np.ndarray:
    """Saturated corneal/tear-film reflections (True = bad pixel)."""
    m = (frame >= level * 4095).astype(np.uint8)
    if dilate:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    return m.astype(bool)


@dataclass
class Registration:
    """Per-frame transforms (frame -> reference) and quality scores."""
    M: np.ndarray            # (T, 2, 3) affine matrices, NaN when failed
    table: pd.DataFrame      # i, n_match, n_inlier, ncc, tx, ty, scale, rot_deg, overlap, good
    shape: tuple

    def good_runs(self, min_len: int = 1) -> list[tuple[int, int]]:
        """Maximal runs [start, stop] (inclusive) of consecutive good frames, longest first."""
        g = self.table["good"].to_numpy()
        runs, s = [], None
        for i, v in enumerate(g):
            if v and s is None:
                s = i
            if not v and s is not None:
                runs.append((s, i - 1)); s = None
        if s is not None:
            runs.append((s, len(g) - 1))
        runs = [r for r in runs if r[1] - r[0] + 1 >= min_len]
        return sorted(runs, key=lambda r: r[0] - r[1])

    def save(self, path: str):
        np.savez_compressed(path, M=self.M, shape=np.array(self.shape),
                            **{f"t_{c}": self.table[c].to_numpy() for c in self.table.columns})

    @staticmethod
    def load(path: str) -> "Registration":
        z = np.load(path)
        tab = pd.DataFrame({k[2:]: z[k] for k in z.files if k.startswith("t_")})
        return Registration(M=z["M"], table=tab, shape=tuple(z["shape"]))


class Registrar:
    """Registers frames to a fixed reference image."""

    def __init__(self, ref: np.ndarray, ref_valid: np.ndarray | None = None,
                 downsample: float = 0.5, n_features: int = 3000, ratio: float = 0.75,
                 ransac_px: float = 3.0, ecc_iters: int = 60):
        self.ref = ref.astype(np.float32)
        self.shape = ref.shape
        self.valid = np.ones(ref.shape, bool) if ref_valid is None else ref_valid
        self.ds = downsample
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.ecc_iters = ecc_iters
        self.n_features = n_features
        R = enhance(self.ref)
        self.R8 = R
        self.Rs = self._small(R)
        vm = self._small((self.valid * 255).astype(np.uint8))
        self.kr, self.dr = cv2.SIFT_create(nfeatures=n_features).detectAndCompute(self.Rs, vm)
        # band-passed full-res reference for ECC / NCC
        self.Rbp = self._bandpass(self.ref)

    def _small(self, a):
        if self.ds == 1:
            return a
        return cv2.resize(a, None, fx=self.ds, fy=self.ds, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _bandpass(a, s1=1.0, s2=12.0):
        a = flatfield(a)
        return (cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)).astype(np.float32)

    # -- steps exposed individually for the notebook --------------------------
    def keypoints(self, frame):
        A = self._small(enhance(frame))
        mask = (~self._small(specular_mask(frame).astype(np.uint8)).astype(bool)).astype(np.uint8) * 255
        k, d = cv2.SIFT_create(nfeatures=self.n_features).detectAndCompute(A, mask)
        return A, k, d

    def match(self, d):
        if d is None or len(d) < 2:
            return []
        m = cv2.BFMatcher().knnMatch(d, self.dr, k=2)
        return [p[0] for p in m if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]

    def ransac(self, k, good):
        if len(good) < 6:
            return None, None
        src = np.float32([k[g.queryIdx].pt for g in good]) / self.ds
        dst = np.float32([self.kr[g.trainIdx].pt for g in good]) / self.ds
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_px / self.ds,
                                             maxIters=5000, confidence=0.999)
        return M, (None if inl is None else inl.ravel().astype(bool))

    def ecc_refine(self, frame, M):
        """Maximise the enhanced correlation coefficient over a full affine warp."""
        F = self._bandpass(frame)
        W = M.astype(np.float32).copy()
        # ECC works with the inverse convention (template = ref, input = frame)
        try:
            _, W = cv2.findTransformECC(self.Rbp, F, cv2.invertAffineTransform(W).astype(np.float32),
                                        cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.ecc_iters, 1e-5),
                                        inputMask=None, gaussFiltSize=3)
            return cv2.invertAffineTransform(W)
        except cv2.error:
            return M

    def ncc(self, frame, M):
        W = cv2.warpAffine(self._bandpass(frame), M, self.shape[::-1], flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        ok = np.isfinite(W) & self.valid
        ok &= ~cv2.warpAffine(specular_mask(frame).astype(np.uint8), M, self.shape[::-1]).astype(bool)
        if ok.sum() < 1000:
            return np.nan, ok.mean()
        return float(np.corrcoef(W[ok], self.Rbp[ok])[0, 1]), float(ok.mean())

    def register_one(self, frame, refine=True):
        A, k, d = self.keypoints(frame)
        good = self.match(d)
        M, inl = self.ransac(k, good)
        if M is None:
            return dict(n_match=len(good), n_inlier=0, M=np.full((2, 3), np.nan))
        n_in = int(inl.sum())
        if refine and n_in >= 15:
            M = self.ecc_refine(frame, M)
        return dict(n_match=len(good), n_inlier=n_in, M=M)

    # -- whole burst ----------------------------------------------------------
    def register_burst(self, frames, refine=True, workers=8, progress=True,
                       min_inliers=25, min_ncc=0.5, max_scale_dev=0.03, max_rot_deg=3.0):
        T = len(frames)
        out = [None] * T

        def job(i):
            r = self.register_one(np.asarray(frames[i]), refine=refine)
            if np.isfinite(r["M"]).all():
                r["ncc"], r["overlap"] = self.ncc(np.asarray(frames[i]), r["M"])
            else:
                r["ncc"], r["overlap"] = np.nan, 0.0
            return i, r

        with ThreadPoolExecutor(workers) as ex:
            for n, (i, r) in enumerate(ex.map(job, range(T))):
                out[i] = r
                if progress and (n % 100 == 0 or n == T - 1):
                    print(f"\rregistered {n + 1}/{T}", end="", flush=True)
        if progress:
            print()
        M = np.stack([o["M"] for o in out]).astype(np.float64)
        scale = np.hypot(M[:, 0, 0], M[:, 1, 0])
        rot = np.degrees(np.arctan2(M[:, 1, 0], M[:, 0, 0]))
        tab = pd.DataFrame(dict(
            i=np.arange(T), n_match=[o["n_match"] for o in out], n_inlier=[o["n_inlier"] for o in out],
            ncc=[o["ncc"] for o in out], tx=M[:, 0, 2], ty=M[:, 1, 2], scale=scale, rot_deg=rot,
            overlap=[o["overlap"] for o in out]))
        tab["good"] = ((tab.n_inlier >= min_inliers) & (tab.ncc >= min_ncc)
                       & ((tab.scale - 1).abs() <= max_scale_dev) & (tab.rot_deg.abs() <= max_rot_deg)
                       & (tab.overlap >= 0.5)).fillna(False)
        return Registration(M=M, table=tab, shape=self.shape)


def warp(frame, M, shape, order=cv2.INTER_LINEAR, border=np.nan):
    """Resample a frame into reference coordinates (outside -> NaN)."""
    return cv2.warpAffine(np.asarray(frame, np.float32), M, shape[::-1], flags=order,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def warp_stack(frames, reg: Registration, idx, mask_specular=True, normalize=True):
    """Registered float32 stack for frame indices `idx`.

    normalize: divide every frame by its median so that auto-exposure / auto-gain
    steps do not show up as brightness 'flicker' in the kymographs.
    """
    idx = list(idx)
    out = np.empty((len(idx),) + reg.shape, np.float32)
    for n, i in enumerate(idx):
        f = np.asarray(frames[i], np.float32)
        if mask_specular:
            f = np.where(specular_mask(np.asarray(frames[i])), np.nan, f)
        w = warp(f, reg.M[i], reg.shape)
        if normalize:
            w /= np.nanmedian(w)
        out[n] = w
    return out


def registered_mean(frames, reg: Registration, idx=None):
    """NaN-aware mean of the registered good frames (our own 'mean_stabilized')."""
    if idx is None:
        idx = np.flatnonzero(reg.table["good"].to_numpy())
    acc = np.zeros(reg.shape, np.float64); cnt = np.zeros(reg.shape, np.float64)
    for i in idx:
        f = np.asarray(frames[i], np.float32)
        f = np.where(specular_mask(np.asarray(frames[i])), np.nan, f)
        w = warp(f, reg.M[i], reg.shape)
        ok = np.isfinite(w)
        acc[ok] += w[ok]; cnt[ok] += 1
    with np.errstate(invalid="ignore"):
        return (acc / cnt).astype(np.float32), cnt


def register_cached(ref, ref_valid, frames, path, **kw) -> Registration:
    if os.path.exists(path):
        return Registration.load(path)
    reg = Registrar(ref, ref_valid).register_burst(frames, **kw)
    reg.save(path)
    return reg
