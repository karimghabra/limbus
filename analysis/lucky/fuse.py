"""Lucky fusion: local, band-by-band burst accumulation (METHODS.md §L2).

Every aligned frame is split into spatial-frequency bands by differences of
Gaussians, which add back up to the frame exactly:

    frame = sum_k band_k + lowpass,  band_k = G(s_k) * frame - G(s_k+1) * frame

Each band is then averaged over the frames with a weight that grows with how
much of that band's true detail the frame carries at that place, raised to
a power p:

    fused_k(x) = sum_i w_ik(x) band_ik(x) / sum_i w_ik(x),   w_ik = a_ik ^ p

a_ik(x), the frame's relative amplitude, is its band projected onto the same
band of a reference (the plain mean), over the reference's own energy, both
averaged over a window around x. Blur of any kind — defocus, motion during
the exposure, a local misregistration — lowers it. Noise does not raise it:
unlike the frame's own band energy, the projection averages the noise out,
so a noisier frame is not mistaken for a sharper one.

This is Fourier burst accumulation (Delbracio & Sapiro, CVPR 2015) made
local. Weighting each frame by its amplitude (p = 1) is the matched filter:
the linear combination with the best signal-to-noise ratio when frames
differ in blur. p = 0 is exactly the ordinary mean; larger p approaches
picking the sharpest frame per band and place, at the cost of averaging
fewer frames. The lowpass is always a plain mean.

Accumulation is streaming — every quantity is a sum over frames — so each
frame is read and warped once per pass, and several p values and the two
split-half subsets are all built in the same pass.
"""
import cv2
import numpy as np


def _blur(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def normalized_fill(img, valid, sigma):
    """Fill invalid pixels by normalised convolution, so band-pass filters
    don't ring at the edges of glare holes and the field border."""
    v = valid.astype(np.float32)
    num = _blur(np.where(valid, img, 0).astype(np.float32), sigma)
    den = _blur(v, sigma)
    fill = num / np.maximum(den, 1e-6)
    return np.where(valid, img, fill).astype(np.float32)


def bands(img, sigmas):
    blurred = [_blur(img, s) for s in sigmas]
    return [blurred[k] - blurred[k + 1] for k in range(len(sigmas) - 1)], blurred[-1]


class LuckyAccumulator:
    """Streaming accumulator for one set of frames and several powers p.

        acc = LuckyAccumulator(shape, params, template)
        for img, seen, gain in aligned frames:
            acc.add(img, seen, gain)
        images, n_eff, valid = acc.result(min_coverage)
    """

    def __init__(self, shape, params, template=None):
        self.shape = shape
        self.p = tuple(params.powers)
        self.sig = tuple(params.band_sigmas_px)        # s_0 < s_1 < ... ; s_0 may be 0
        self.nb = len(self.sig) - 1
        self.params = params
        h, w = shape
        # per p and band: weighted band sum, weight sum, squared-weight sum
        self.num = np.zeros((len(self.p), self.nb, h, w), np.float32)
        self.den = np.zeros_like(self.num)
        self.den2 = np.zeros_like(self.num)
        self.low = np.zeros((h, w), np.float32)
        self.low_n = np.zeros((h, w), np.float32)
        self.frames = 0
        self.tbands = self.tenergy = None
        if any(p > 0 for p in self.p):
            if template is None:
                raise ValueError("weights p > 0 need a template image")
            t = normalized_fill(np.nan_to_num(template), np.isfinite(template),
                                params.fill_sigma_px)
            self.tbands, _ = bands(t, self.sig)
            self.tenergy = [_blur(b * b, self._radius(k)) + 1e-12
                            for k, b in enumerate(self.tbands)]

    def _radius(self, k):
        par = self.params
        return max(par.energy_min_radius_px, par.energy_radius_factor * self.sig[k + 1])

    def add(self, img, seen, gain=1.0):
        """img: aligned frame (float32), seen: bool map of observed pixels.
        gain rescales the frame to the burst's common brightness."""
        par = self.params
        img = normalized_fill(img * gain, seen, par.fill_sigma_px)
        fb, low = bands(img, self.sig)
        seen_f = seen.astype(np.float32)
        self.low += low * seen_f
        self.low_n += seen_f
        for k in range(self.nb):
            band = fb[k]
            # a band value is trustworthy only where the filter footprint saw
            # real pixels: shrink the observed map by ~2 of the band's sigma
            support = (_blur(seen_f, 2 * self.sig[k + 1] + 1) > 0.98) & seen
            amp = None
            if self.tbands is not None:
                # relative amplitude; the floor keeps weights even where the
                # reference has no detail to compare against (§L2)
                proj = _blur(band * self.tbands[k], self._radius(k)) / self.tenergy[k]
                amp = (np.maximum(proj, 0) + par.template_floor).astype(np.float32)
            for j, p in enumerate(self.p):
                wgt = support.astype(np.float32) if p == 0 else \
                    np.power(amp, p, dtype=np.float32) * support
                self.num[j, k] += wgt * band
                self.den[j, k] += wgt
                self.den2[j, k] += wgt * wgt
        self.frames += 1

    def merge(self, other):
        """Sum of two disjoint accumulators (e.g. the split halves)."""
        out = LuckyAccumulator.__new__(LuckyAccumulator)
        out.__dict__.update(self.__dict__)
        for k in ("num", "den", "den2", "low", "low_n"):
            setattr(out, k, getattr(self, k) + getattr(other, k))
        out.frames = self.frames + other.frames
        return out

    def result(self, min_coverage):
        """Fused images keyed 'p<p>' ('p0' is the ordinary mean), NaN where
        too few frames observed a pixel; per-image effective frame counts per
        band; and the valid map."""
        valid = self.low_n >= max(1.0, min_coverage * self.frames)
        low = np.where(valid, self.low / np.maximum(self.low_n, 1), np.nan)
        images, n_eff = {}, {}
        for j, p in enumerate(self.p):
            img = low.copy()
            ne = []
            for k in range(self.nb):
                d = self.den[j, k]
                ok = d > 0
                img += np.where(ok, self.num[j, k] / np.where(ok, d, 1), 0)
                # effective number of frames: (sum w)^2 / sum w^2 — how many
                # equally weighted frames give the same noise averaging
                eff = np.where(ok, d * d / np.maximum(self.den2[j, k], 1e-30), 0)
                ne.append(float(np.median(eff[valid & ok])) if (valid & ok).any() else 0.0)
            images[f"p{p:g}"] = img.astype(np.float32)
            n_eff[f"p{p:g}"] = ne
        return images, n_eff, valid
