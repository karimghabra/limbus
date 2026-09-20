"""Growing a vessel outwards from what has already been measured.

Every other stage judges the image against one global rule - a threshold, a
length, a scale. That is why faint vessels are lost: the rule is set by what
the whole frame needs, not by what a particular vessel looks like. But once a
vessel has been found and fitted, its own diameter, depth and direction are
known, and they are a far more specific thing to look for than "a ridge".

So each free end is used as a nucleation point and the vessel is grown from
it. At each step the profile ACROSS the vessel that this vessel would produce
- a blurred cylinder of the current radius and depth - is compared with the
image over a fan of directions, and the best direction is taken if the match
is good enough. The state then adapts: radius, depth and direction are
updated towards what was just measured, slowly, so a vessel may taper or
darken gradually, as real vessels do, without a sudden jump in calibre being
accepted as the same vessel.

Growth stops when the match fails for several steps running, when the vessel
reaches another vessel (a junction, for the junction rules to interpret), or
at the edge of the data. A few weak steps in a row are tolerated and then
trimmed back if the vessel does not reappear, which is what lets a vessel
cross the shadow of another one without the growth being thrown away.

Nothing here uses velocity, and nothing here lowers a global threshold: a
vessel is only followed where the image matches THAT vessel's own profile.

MEASURED. On synthetic vessels it does what it should:
it follows a vessel that fades to a quarter of its depth, follows one that
tapers from 3 px to 1 px, and stops where a vessel genuinely ends. On real
crops it adds almost nothing - 0.2% of centreline on 1522L, 3% on 1522R -
because by this point the detector has already found the vessels and the
junction rules have joined them. On a phase-randomised texture surrogate, which contains no vessels at all,
the profile test alone let the same growth add 26-69% - it was extending the
detector's own false positives far more often than real vessels.

What fixed that is the direction of the LOCAL SPECTRUM. A vessel is coherently
oriented over a patch, so its power concentrates along one line in the Fourier
plane; a texture blotch also has a preferred direction in any one patch, but
it is unrelated to the way the vessel was heading. Requiring the patch ahead
to be both directional AND aligned with the vessel cut false growth by about
twenty times: the surrogate now gains 0-3% while the real crops gain 0.2-2.4%.

The honest summary is that this buys a little and costs a little. It is on by
default because what it adds is real vessel, but it is not what will find the
vessels that are still missing: by the time it runs, the detector has found
what it is going to find and the junction rules have joined it.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GrowConfig:
    step: float = 5.0             # how far each step advances (px)
    fan_deg: float = 35.0         # how far it may turn at one step
    fan_n: int = 9                # directions tried per step
    corr_min: float = 0.55        # profile match required
    amp_min: float = 0.15         # ... and at least this share of the expected depth
                                  # (a vessel may fade to a quarter of its depth and
                                  # still be the same vessel; the significance test
                                  # below is what keeps that honest)
    amp_max: float = 3.0          # ... and no more than this (that would be another vessel)
    adapt: float = 0.25           # how fast radius and depth follow what is measured
    max_weak: int = 4             # consecutive weak steps tolerated before stopping
    max_steps: int = 120          # a growth cannot run further than this
    turn_penalty: float = 0.0015  # per degree, to prefer going straight
    snr_min: float = 2.0          # the profile must also stand out from its own
                                  # flanks. Measured: growing through bare texture
                                  # never exceeds 0.5, while a real vessel faded to
                                  # a quarter of its depth still reaches 4.5
    use_spectrum: bool = True     # take the step direction from the local spectrum
    spec_size: int = 48           # patch side for it (px)
    spec_coherence: float = 0.70  # how directional the patch must be
    spec_align: float = 25.0      # ... and how close to the way the vessel is going
    stop_at_vessel: float = 1.0   # stop this many radii from another vessel
    min_grown: float = 12.0       # shorter growths are not worth keeping (px)
    keep_corr: float = 0.60       # a finished growth is judged as a whole: its
    keep_snr: float = 2.5         # profile must match and stand out along it, and
    keep_turn: float = 14.0       # it must not wander (mean turn per step, degrees)


def local_orientation(A, centre, size=48, kmin=2.0, kmax=None):
    """Direction and coherence of the texture in a patch, from its spectrum.

    A vessel is an elongated structure, so the power in its neighbourhood
    concentrates along a line in the Fourier plane PERPENDICULAR to the vessel
    - the longer and straighter the vessel, the tighter that concentration.
    Texture blotches have no preferred direction and their power spreads over
    all angles. Reading the direction this way uses the whole patch at once
    rather than one profile across one point, and it comes with its own
    measure of how directional the patch is at all.

    Returns (angle of the structure in image space, coherence in 0..1).
    """
    H, W = A.shape
    half = size // 2
    x, y = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(0, x - half), max(0, y - half)
    x1, y1 = min(W, x0 + size), min(H, y0 + size)
    x0, y0 = max(0, x1 - size), max(0, y1 - size)
    patch = A[y0:y1, x0:x1].astype(np.float64)
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        return None, 0.0
    patch = patch - patch.mean()
    win = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(patch * win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(patch.shape[0]))[:, None] * patch.shape[0]
    kx = np.fft.fftshift(np.fft.fftfreq(patch.shape[1]))[None, :] * patch.shape[1]
    k = np.hypot(kx, ky)
    kmax = kmax if kmax is not None else min(patch.shape) / 3.0
    band = (k >= kmin) & (k <= kmax)
    if not band.any():
        return None, 0.0
    ang = (np.degrees(np.arctan2(ky, kx)) % 180.0)[band]
    pw = P[band]
    bins = np.arange(0, 181, 5.0)
    hist = np.histogram(ang, bins=bins, weights=pw)[0]
    hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(3) / 3, mode="same")[2:-2]
    if hist.sum() <= 0:
        return None, 0.0
    j = int(np.argmax(hist))
    peak, floor = float(hist[j]), float(np.median(hist))
    coherence = (peak - floor) / (peak + floor) if peak + floor > 0 else 0.0
    spectral = bins[j] + 2.5                       # direction of the energy ridge
    return float((spectral + 90.0) % 180.0), float(coherence)   # image-space direction


def expected_profile(offs, r, D, psf):
    """The cross-profile a blurred cylinder of radius r and peak absorbance D
    would give at these offsets."""
    fine = np.arange(offs[0] - 3 * psf, offs[-1] + 3 * psf + 0.25, 0.25)
    a = D * np.sqrt(np.clip(1 - (fine / max(r, 0.5)) ** 2, 0, None))
    k = int(np.ceil(3 * psf / 0.25))
    g = np.exp(-0.5 * (np.arange(-k, k + 1) * 0.25 / psf) ** 2)
    g /= g.sum()
    a = np.convolve(a, g, mode="same")
    return np.interp(offs, fine, a)


def measure(A, p, u, r, psf, half=None):
    """Observed cross-profile at p, across direction u, background removed."""
    half = half if half is not None else max(3.0 * r + 4.0, 8.0)
    offs = np.arange(-half, half + 0.5, 0.5, dtype=np.float32)
    n = np.array([-u[1], u[0]], float)
    px = (p[0] + n[0] * offs).astype(np.float32)[None, :]
    py = (p[1] + n[1] * offs).astype(np.float32)[None, :]
    prof = cv2.remap(A, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    edge = max(4, len(offs) // 6)
    base = 0.5 * (np.median(prof[:edge]) + np.median(prof[-edge:]))
    return offs, prof.astype(float) - float(base)


def score_step(A, p, u, r, D, psf):
    """How well the image at p across u matches this vessel's own profile.
    Returns (correlation, amplitude, measured radius, snr)."""
    offs, obs = measure(A, p, u, r, psf)
    exp = expected_profile(offs, r, D, psf)
    ne, no = float(np.linalg.norm(exp)), float(np.linalg.norm(obs))
    if ne <= 0 or no <= 0:
        return -1.0, 0.0, r, 0.0
    corr = float(np.dot(exp, obs) / (ne * no))
    amp = float(np.dot(exp, obs) / np.dot(exp, exp))
    c = len(offs) // 2
    peak = float(obs[max(c - 4, 0):c + 5].max())
    if peak > 0:
        half = np.flatnonzero(obs >= peak / 2)
        hw = 0.25 * (half[-1] - half[0]) if half.size > 1 else r
    else:
        hw = r
    # Significance, judged against the profile's OWN flanks. A vessel stands
    # out from the sclera beside it; a texture blob does not, however well its
    # shape happens to match. Judging it against a global noise estimate does
    # not work here - the fine-grain noise is ten times smaller than the
    # blotchy texture that actually competes with a faint vessel, so every
    # blob looked significant.
    edge = max(4, len(offs) // 5)
    flank = np.concatenate([obs[:edge], obs[-edge:]])
    spread = 1.4826 * float(np.median(np.abs(flank - np.median(flank))))
    snr = peak / max(spread, 1e-9)
    return corr, amp, float(np.clip(hw, 0.5, 3 * r + 2)), float(snr)


def grow_end(A, valid, p0, u0, r0, D0, occupied, cfg=None, psf=1.8):
    """Follow one vessel outwards from (p0, u0). Returns the points added."""
    cfg = cfg or GrowConfig()
    H, W = A.shape
    p = np.asarray(p0, float).copy()
    u = np.asarray(u0, float) / max(np.hypot(*u0), 1e-9)
    r, D = float(r0), float(D0)
    angles = np.radians(np.linspace(-cfg.fan_deg, cfg.fan_deg, cfg.fan_n))
    step_ahead = cfg.step * 1.5
    pts, weak, last_good = [], 0, 0
    seen = []                                        # (corr, snr) as measured, step by step
    for _ in range(cfg.max_steps):
        # Where does the local texture say this vessel goes? The patch just
        # ahead is dominated by the vessel if it continues, and its spectrum
        # gives that direction with a coherence to judge it by. Texture also
        # has a preferred direction in any one patch, but it is unrelated to
        # the way this vessel was heading, so the ALIGNMENT is what separates
        # them - coherence alone does not.
        spec_dir, spec_ok = None, True
        if cfg.use_spectrum:
            ang_deg, coh = local_orientation(A, p + u * step_ahead, cfg.spec_size)
            if ang_deg is None or coh < cfg.spec_coherence:
                spec_ok = False
            else:
                v1 = np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])
                v1 = v1 if np.dot(v1, u) >= 0 else -v1        # resolve the 180 deg ambiguity
                off = np.degrees(np.arccos(np.clip(np.dot(v1, u), -1, 1)))
                if off > cfg.spec_align:
                    spec_ok = False
                else:
                    spec_dir = v1
        best = None
        for a in angles:
            ca, sa = np.cos(a), np.sin(a)
            ud = np.array([u[0] * ca - u[1] * sa, u[0] * sa + u[1] * ca])
            q = p + ud * cfg.step
            xi, yi = int(round(q[0])), int(round(q[1]))
            if not (0 <= xi < W and 0 <= yi < H) or not valid[yi, xi]:
                continue
            corr, amp, hw, snr = score_step(A, q, ud, r, D, psf)
            s = corr - cfg.turn_penalty * abs(np.degrees(a))
            if spec_dir is not None:                          # prefer where the texture points
                s += 0.3 * float(np.dot(ud, spec_dir))
            if best is None or s > best[0]:
                best = (s, corr, amp, hw, snr, q, ud)
        if best is None:
            break
        s, corr, amp, hw, snr, q, ud = best
        good = (corr >= cfg.corr_min and cfg.amp_min <= amp <= cfg.amp_max
                and snr >= cfg.snr_min and (spec_ok or not cfg.use_spectrum))
        pts.append(q)
        seen.append((corr, snr))
        p, u = q, ud
        if good:
            weak = 0
            last_good = len(pts)
            r += cfg.adapt * (hw - r)               # a vessel may taper, slowly
            D += cfg.adapt * (amp * D - D)
        else:
            weak += 1
            if weak > cfg.max_weak:
                break
        xi, yi = int(round(q[0])), int(round(q[1]))
        if occupied is not None and occupied[yi, xi] and len(pts) > 2:
            last_good = len(pts)                    # reached another vessel: stop here
            break
    pts = pts[:last_good]                            # trim the trailing weak steps
    seen = seen[:last_good]
    if not pts:
        return np.empty((0, 2))
    out = np.asarray(pts, float)
    L = float(np.hypot(*np.diff(np.vstack([p0, out]), axis=0).T).sum())
    if L < cfg.min_grown:
        return np.empty((0, 2))
    return out if accept_growth(np.vstack([p0, out]), seen, cfg) else np.empty((0, 2))


def accept_growth(path, seen, cfg=None):
    """Judge a completed growth as a whole, not step by step.

    Growth adapts as it goes, which is what lets it follow a vessel that
    tapers or fades - so the finished path must be judged on what was measured
    AS it grew, against the expectation in force at each step. Judging it
    against the seed's profile instead fails every vessel that changes along
    its length, which is most of them: good growths were being discarded after
    a first step scoring 0.96 correlation at 11 sigma.

    A vessel is coherent over distance: its profile matches and stands out
    along the whole path, and it does not wander. Texture does neither.
    """
    cfg = cfg or GrowConfig()
    path = np.asarray(path, float)
    if len(path) < 3 or not seen:
        return False
    d = np.diff(path, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    turn = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1)))
    corrs = [c for c, _ in seen]
    snrs = [s for _, s in seen]
    return (float(np.median(corrs)) >= cfg.keep_corr
            and float(np.median(snrs)) >= cfg.keep_snr
            and (float(np.mean(turn)) <= cfg.keep_turn if len(turn) else True))
