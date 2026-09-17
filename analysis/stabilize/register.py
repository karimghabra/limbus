"""Translation registration by phase correlation, groupwise against a
template (METHODS.md §7–§8).

Convention used throughout: a frame's displacement d = (dx, dy) is how far
its content has moved relative to the template, in working-scale pixels.
Stabilizing a frame means warping it by -d. (Verified empirically: OpenCV's
phaseCorrelate(ref, moved) returns +d for content moved by +d.)
"""
import cv2
import numpy as np


class PhaseCorrelator:
    """Phase correlation with a cached Hanning window (§7).

    The window tapers each image to zero at its borders: the Fourier
    transform treats the image as periodic, so without it the mismatched
    edges would correlate at zero shift regardless of the true motion.
    """

    def __init__(self, shape):
        h, w = shape
        self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def __call__(self, reference, moved):
        (dx, dy), response = cv2.phaseCorrelate(
            np.ascontiguousarray(reference, dtype=np.float32),
            np.ascontiguousarray(moved, dtype=np.float32),
            self.window)
        return np.array([dx, dy], dtype=np.float64), float(response)


def _upsampled_dft(data, size, factor, offsets):
    """Evaluate the inverse Fourier transform of `data` on a small grid,
    `factor` times finer than the pixel grid, around `offsets` — by matrix
    multiplication rather than zero-padding, so it costs almost nothing
    (Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 2008)."""
    im2pi = 1j * 2 * np.pi
    for n_items, off in list(zip(data.shape, offsets))[::-1]:
        kernel = (np.arange(size) - off)[:, None] * np.fft.fftfreq(n_items, factor)
        data = np.tensordot(np.exp(-im2pi * kernel), data, axes=(1, -1))
    return data


class SubpixelRefiner:
    """Sub-pixel refinement with no interpolation (§7).

    A sub-pixel shift is exact in the Fourier domain — just a phase ramp —
    so instead of warping the frame (which blurs by an amount that depends on
    the fractional shift and adds noise that grows with motion), the
    correlation peak is evaluated on a finer grid directly from the spectra.

    Partial whitening (alpha = 0.5) rather than full phase correlation: full
    whitening gives the noisiest high frequencies — red cells, sensor noise —
    the same vote as stable vessel geometry; half-whitening still suppresses
    illumination while letting geometry dominate. On synthetic ground truth
    it cut RMS error from 0.52 px (warp + cv2) to 0.08 px.

    The integer peak is searched only within `radius` px of the coarse
    estimate, so a spurious peak elsewhere can't override the robust
    mask-based capture.
    """

    def __init__(self, shape, factor=20, alpha=0.5):
        h, w = shape
        self.window = cv2.createHanningWindow((w, h), cv2.CV_64F)
        self.factor = factor
        self.alpha = alpha
        self.shape = np.array(shape)

    def set_template(self, template):
        self.F = np.fft.fft2(template * self.window)

    def __call__(self, frame, coarse_d, radius):
        G = np.fft.fft2(frame * self.window)
        prod = self.F * G.conj()
        mag = np.maximum(np.abs(prod), 1e-12)
        # confidence from the fully whitened surface, comparable to cv2's response
        phase_cc = np.abs(np.fft.ifft2(prod / mag))
        weighted = prod / mag ** self.alpha
        cc = np.abs(np.fft.ifft2(weighted))

        # integer peak near the coarse estimate; shifts are (row, col) needed
        # to register the frame onto the template, i.e. minus our +d
        centre = np.array([-coarse_d[1], -coarse_d[0]])
        r = int(np.ceil(radius))
        best, best_val = None, -1.0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cand = np.round(centre) + (dy, dx)
                iy, ix = (cand % self.shape).astype(int)
                if cc[iy, ix] > best_val:
                    best_val, best = cc[iy, ix], cand
        iy, ix = (best % self.shape).astype(int)
        confidence = float(phase_cc[iy, ix])

        size = int(np.ceil(self.factor * 1.5))
        mid = np.trunc(size / 2.0)
        up = _upsampled_dft(weighted.conj(), size, self.factor,
                            mid - best * self.factor).conj()
        peak = np.array(np.unravel_index(np.argmax(np.abs(up)), up.shape), float) - mid
        rowcol = best + peak / self.factor
        return np.array([-rowcol[1], -rowcol[0]]), confidence


def shift(img, d, interpolation=cv2.INTER_LINEAR, border=0.0):
    """Warp img by -d, undoing a content displacement of d."""
    m = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])
    return cv2.warpAffine(np.ascontiguousarray(img, dtype=np.float32), m,
                          (img.shape[1], img.shape[0]), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def observed(O, i, shape):
    """Pixels frame i actually observed: everything except glare (§4). A
    glare-hidden pixel is unknown, not 'not vessel'."""
    if O is None:
        return np.ones(shape, np.float32)
    return O[i].astype(np.float32)


def build_templates(V, M, traj, frames, O=None):
    """Consensus templates from the frames aligned by traj (§8).

    Returns the soft mask consensus p (fraction of frames calling each pixel
    vessel) and the mean vesselness, each normalised by coverage — the frames
    that actually observed each pixel. Pixels shifted in from outside the
    field, or hidden by glare, don't count as evidence either way.
    """
    h, w = V.shape[1:]
    msum = np.zeros((h, w), np.float32)
    vsum = np.zeros((h, w), np.float32)
    cover = np.zeros((h, w), np.float32)
    for i in frames:
        d = traj[i]
        msum += shift(M[i].astype(np.float32), d)
        vsum += shift(V[i], d)
        cover += shift(observed(O, i, (h, w)), d)
    valid = cover > 0
    p = np.zeros_like(msum)
    vmean = np.zeros_like(vsum)
    p[valid] = msum[valid] / cover[valid]
    vmean[valid] = vsum[valid] / cover[valid]
    return p, vmean, cover


def chained_initial(V, frames, correlator):
    """First guess: add up shifts between consecutive good frames. Cheap and
    locally precise, but errors accumulate as a random walk (~sigma*sqrt(N)),
    and one bad step offsets everything after it (§8) — so it only seeds
    the template, and is never the final answer."""
    n = V.shape[0]
    traj = np.zeros((n, 2), np.float64)
    for a, b in zip(frames[:-1], frames[1:]):
        step, _ = correlator(V[a], V[b])
        traj[b] = traj[a] + step
    traj[frames] -= np.median(traj[frames], axis=0)
    return traj


def register_groupwise(V, M, good, params, scale, O=None, log=None):
    """Coarse-to-fine groupwise registration (§8).

    Each iteration: build templates from the current alignment, then measure
    every frame directly against them — coarse on the binary envelope (robust
    to illumination, large capture range), then an interpolation-free
    sub-pixel refinement on the soft vesselness near the coarse estimate.
    Measuring against a template rather than chaining keeps errors from
    accumulating; alternating 'align given template' and 'template given
    alignment' is the same structure as expectation-maximisation, and
    converges in a few rounds.
    """
    n = V.shape[0]
    frames = np.flatnonzero(good)
    correlator = PhaseCorrelator(V.shape[1:])
    refiner = SubpixelRefiner(V.shape[1:])
    traj = chained_initial(V, frames, correlator)

    coarse_resp = np.zeros(n)
    fine_resp = np.zeros(n)
    registered = good.copy()
    converge = params.converge_rms_px * scale
    max_step = params.fine_max_step_px * scale
    history = []

    for iteration in range(1, params.max_iterations + 1):
        use = np.flatnonzero(registered) if registered.any() else frames
        p, vmean, _ = build_templates(V, M, traj, use, O)
        refiner.set_template(vmean)
        new = traj.copy()
        for i in frames:
            # Candidates, not a hand-off. The current estimate (seeded by the
            # locally precise chain) is always tried; the coarse template
            # estimate is tried too when it lands elsewhere. Whichever the
            # fine stage matches to the template with more confidence wins.
            # On real data the coarse response alone could not flag wrong
            # peaks (outliers off by 200 px scored like good frames), but
            # fine confidence separates them cleanly (~0.4 vs ~0.001) — so a
            # coarse outlier can't override a good prior, while a chain
            # broken by a saccade or blink can still be rescued.
            d_coarse, r_c = correlator(p, M[i].astype(np.float32))
            candidates = [traj[i]]
            if (r_c >= params.coarse_min_response
                    and np.hypot(*(d_coarse - traj[i])) > max_step):
                candidates.append(d_coarse)
            best_d, best_conf = None, -1.0
            for cand in candidates:
                d, conf = refiner(V[i], cand, radius=max_step)
                if conf > best_conf:
                    best_d, best_conf = d, conf
            coarse_resp[i], fine_resp[i] = r_c, best_conf
            registered[i] = best_conf >= params.fine_min_response
            # an unconfident frame keeps its previous estimate rather than
            # adopting a guess, and is left out of the next template
            new[i] = best_d if registered[i] else traj[i]
        keep = np.flatnonzero(registered)
        if keep.size:
            new[frames] -= np.median(new[keep], axis=0)   # keep template centred
        change = float(np.sqrt(np.mean(np.sum((new[frames] - traj[frames]) ** 2, axis=1))))
        traj = new
        history.append(change / scale)
        if log:
            log(f"    iteration {iteration}: RMS change {change / scale:.3f} px, "
                f"registered {keep.size}/{frames.size}")
        if change < converge:
            break

    return {
        "traj": traj,
        "coarse_response": coarse_resp,
        "fine_response": fine_resp,
        "registered": registered,
        "iterations": len(history),
        "convergence_px": history,
    }
