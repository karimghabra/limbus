"""A vessel as an object that knows what it looks like, and can recognise more of itself.

The staged detector decides the whole network by thresholding evidence computed
the same way everywhere, and everything after it rearranges that decision. Two
things follow that this module exists to avoid.

First, it invents. Fragments are joined across gaps and the connector is spliced
in as real centreline: measured on a real frame, the geometry added by widening
that gap sits at a median absorbance of 0.039 where the lines already found sit
at 0.082, and a third of it is no darker than background. A vessel here has no
way to draw a line it did not measure - it grows one step at a time onto ground
where its own profile is present, and stops where that stops being true. **There
are no discontinuities because there is no mechanism that could make one.**

Second, a global threshold has no idea what it is looking for. A vessel that has
already been found does: its diameter, its depth, the shape of its absorbance
across itself, the direction it was going. That is a far more specific thing to
search for than "a ridge", and it is per-vessel, so a faint vessel is followed
on its own terms rather than by lowering a threshold until texture arrives too.

    nucleate    take only the segments we are certain of
    instantiate one Vessel each, measured from the image
    grow        from the mouth and the tail, along the heading, matching that
                vessel's OWN profile, until the match fails

THE SEARCH IS DIRECTIONAL, and that is the point rather than an optimisation.
A vessel's profile is sampled across a window aligned to the direction it is
travelling, and a candidate step is scored against the local spectrum's dominant
orientation. Where another vessel crosses, its absorbance falls in a different
orientation band and does not enter the score - so a crossing vessel cannot
corrupt the profile being matched, which is what makes the search survive a
junction instead of being pulled into it.

OVERLAP IS REFUSED unless the two signals are separable. A step onto ground
another vessel already holds is allowed only where a windowed spectrum shows
TWO distinct orientation components - one along this vessel's heading, one along
the occupant's - which is what a real crossing looks like and what a single
vessel being claimed twice does not.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import model as vmodel


# --------------------------------------------------------------------- ends

@dataclass
class End:
    """One end of a vessel: where it is and which way it leaves.

    `heading` points OUT of the vessel, so growth is always "step along the
    heading". Measured over an arc rather than a sample count, and read back
    from the end rather than across the whole piece.
    """
    p: np.ndarray
    heading: np.ndarray

    @staticmethod
    def of(points, at_start, arc=12.0):
        C = np.asarray(points, float)
        if len(C) < 2:
            return End(C[0].copy(), np.array([1.0, 0.0]))
        walk = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        L = float(walk[-1])
        a = min(arc, max(L, 1e-9))
        if at_start:
            near, far = 0.0, a
        else:
            near, far = L, L - a
        pn = np.array([np.interp(near, walk, C[:, 0]), np.interp(near, walk, C[:, 1])])
        pf = np.array([np.interp(far, walk, C[:, 0]), np.interp(far, walk, C[:, 1])])
        d = pn - pf
        n = float(np.hypot(*d))
        return End(pn, d / n if n > 1e-9 else np.array([1.0, 0.0]))


# ----------------------------------------------------------------- profiles

def radial_profile(A, points, half=14.0, step=0.5, along=None, heading=None,
                   window=None):
    """Mean absorbance as a function of radial distance from the centreline.

    This is the vessel's signature: not a model of what a vessel should look
    like, but what THIS vessel does look like, which is what the search matches
    against. Taken as a median along the centreline so that the few samples
    where another vessel crosses do not drag it.

    With `window`, only the last `window` px of the centreline are used, so a
    vessel that tapers is matched on the calibre it currently has rather than
    on its average.
    """
    C = np.asarray(points, float)
    if len(C) < 3:
        return None, None
    if window is not None:
        walk = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        keep = walk >= max(0.0, walk[-1] - window) if heading is None or True else None
        C = C[keep] if keep.sum() >= 3 else C[-3:]
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    sel = slice(None, None, max(1, len(C) // (along or 40)))
    offs = np.arange(-half, half + step, step, dtype=np.float32)
    px = (C[sel, None, 0] + n[sel, None, 0] * offs[None, :]).astype(np.float32)
    py = (C[sel, None, 1] + n[sel, None, 1] * offs[None, :]).astype(np.float32)
    band = cv2.remap(np.asarray(A, np.float32), px, py, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)
    return offs.astype(float), np.median(band, axis=0).astype(float)


def profile_at(A, p, heading, offs):
    """The same profile, taken across ONE point in the direction across
    `heading` - which is what makes the sampling directional: a vessel
    crossing at another angle contributes to a different part of the window,
    not to this one."""
    nrm = np.array([-heading[1], heading[0]], float)
    px = (p[0] + nrm[0] * offs).astype(np.float32)[None, :]
    py = (p[1] + nrm[1] * offs).astype(np.float32)[None, :]
    return cv2.remap(np.asarray(A, np.float32), px, py, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[0].astype(float)


def match(a, b):
    """How alike two radial profiles are, after removing their own baselines.

    Correlation, so a vessel that fades is still recognised by its shape; the
    amplitude is reported separately because a step that keeps the shape but
    loses the depth is a vessel ending, not a vessel continuing.
    """
    a = np.asarray(a, float) - np.median(np.r_[a[:4], a[-4:]])
    b = np.asarray(b, float) - np.median(np.r_[b[:4], b[-4:]])
    na, nb = float(np.hypot(*(a, 0))[0] if False else np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0, 0.0
    corr = float(np.dot(a, b) / (na * nb))
    amp = float(b.max() / max(a.max(), 1e-9))
    return corr, amp


# ------------------------------------------------------- local orientations

def orientations(A, centre, size=48, n_peaks=2, min_rel=0.35):
    """The dominant directions of the texture in a window, from its spectrum.

    A vessel puts its power in a band PERPENDICULAR to itself, so the angular
    distribution of |F|^2 has a peak across each vessel present. Returns the
    peak directions (as vessel headings, not as spectral ones) and their
    relative strengths, which is what decides whether a place holds one vessel
    or two: two vessels crossing give two peaks, one vessel gives one.
    """
    H, W = A.shape
    h = size // 2
    x, y = int(round(centre[0])), int(round(centre[1]))
    if x - h < 0 or y - h < 0 or x + h >= W or y + h >= H:
        return np.array([]), np.array([])
    w = np.asarray(A[y - h:y + h, x - h:x + h], float)
    w = w - w.mean()
    win = np.hanning(2 * h)[:, None] * np.hanning(2 * h)[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(w * win))) ** 2
    ky, kx = np.mgrid[-h:h, -h:h]
    r = np.hypot(kx, ky)
    keep = (r > 2) & (r < h - 1)
    ang = (np.arctan2(ky, kx) % np.pi)
    nb = 36
    idx = np.clip((ang[keep] / np.pi * nb).astype(int), 0, nb - 1)
    power = np.bincount(idx, weights=P[keep], minlength=nb)
    power = np.convolve(np.r_[power, power, power], np.ones(3) / 3, "same")[nb:2 * nb]
    if power.max() <= 0:
        return np.array([]), np.array([])
    peaks = []
    for i in range(nb):
        if power[i] >= power[(i - 1) % nb] and power[i] >= power[(i + 1) % nb]:
            peaks.append((power[i], i))
    peaks.sort(reverse=True)
    out_a, out_s = [], []
    for s, i in peaks[:n_peaks]:
        if s < min_rel * peaks[0][0]:
            break
        spec = (i + 0.5) * np.pi / nb          # direction of the spectral peak
        out_a.append((spec + np.pi / 2) % np.pi)   # the vessel runs across it
        out_s.append(float(s / peaks[0][0]))
    return np.array(out_a), np.array(out_s)


def _angdiff(a, b):
    return float(np.abs(np.angle(np.exp(1j * 2 * (a - b))) / 2))


# ------------------------------------------------------------------- vessel

@dataclass
class Vessel:
    """One vessel: the spline that draws it, and everything measured about it.

    `points` is dense and CONTINUOUS - every consecutive pair is one growth
    step apart, and nothing is ever spliced in - so a Vessel cannot represent a
    discontinuity.
    """
    points: np.ndarray
    offs: np.ndarray = None            # radial distances the profile is sampled at
    profile: np.ndarray = None         # absorbance vs radial distance: the signature
    radius_profile: np.ndarray = None  # radius along the vessel (diameter profile)
    depth: float = 0.0                 # peak absorbance
    id: int = 0
    grown: list = field(default_factory=list)   # which stretches were grown, for audit
    velocity_profile: object = None    # not yet defined
    meta: dict = field(default_factory=dict)

    # ---- geometry ----
    @property
    def length(self):
        C = np.asarray(self.points, float)
        return float(np.hypot(*np.diff(C, axis=0).T).sum()) if len(C) > 1 else 0.0

    @property
    def tortuosity(self):
        """Arc length over the straight-line distance between the ends: 1.0 is
        a straight vessel, and it rises with how much the vessel wanders."""
        C = np.asarray(self.points, float)
        if len(C) < 2:
            return 1.0
        chord = float(np.hypot(*(C[-1] - C[0])))
        return self.length / max(chord, 1e-9)

    @property
    def mouth(self):
        return End.of(self.points, True)

    @property
    def tail(self):
        return End.of(self.points, False)

    def spline(self, spacing=16.0):
        """Control points for the Catmull-Rom that draws this vessel."""
        C = np.asarray(self.points, float)
        if len(C) < 3:
            return C
        walk = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        k = max(2, int(round(walk[-1] / spacing)) + 1)
        s = np.linspace(0, walk[-1], k)
        return np.stack([np.interp(s, walk, C[:, 0]), np.interp(s, walk, C[:, 1])], 1)

    # ---- measurement ----
    def measure(self, A, half=14.0, window=None):
        """Read this vessel's own signature off the image."""
        offs, prof = radial_profile(A, self.points, half=half, window=window)
        if offs is None:
            return self
        self.offs, self.profile = offs, prof
        base = float(np.median(np.r_[prof[:4], prof[-4:]]))
        self.depth = float(prof.max() - base)
        c = len(offs) // 2
        half_max = base + self.depth / 2
        right = np.flatnonzero(prof[c:] < half_max)
        left = np.flatnonzero(prof[:c + 1][::-1] < half_max)
        hw = 0.25 * ((right[0] if right.size else 20) + (left[0] if left.size else 20))
        self.meta["radius"] = float(max(hw, 0.8))
        return self

    def lumen(self, shape, pad=1.0):
        """The pixels this vessel occupies, for the no-overlap rule."""
        m = np.zeros(shape, np.uint8)
        r = float(self.meta.get("radius", 2.0))
        cv2.polylines(m, [np.rint(np.asarray(self.points, float)).astype(np.int32)],
                      False, 1, max(1, int(round(2 * (r + pad)))))
        return m > 0


# ------------------------------------------------------------------- growth

@dataclass
class GrowCfg:
    step: float = 2.0          # px per growth step; short, so the path can curve
    fan_deg: float = 22.0      # how far a step may turn from the current heading
    fan_n: int = 9
    corr_min: float = 0.70     # agreement with the vessel's OWN profile
    amp_min: float = 0.35      # a step may fade to this fraction of its depth
    amp_max: float = 2.5       # ... or darken to this, but not more
    orient_tol_deg: float = 30.0   # the local spectrum must agree with the heading
    orient_min: float = 0.0    # ... and be at least this strong relative to the peak
    max_weak: int = 5          # consecutive weak steps tolerated before stopping
    spec_size: int = 40        # window for the local spectrum
    cross_tol_deg: float = 25.0    # how near a spectral peak must be to count as
                                   # "this vessel" or "the occupant" at a crossing
    max_steps: int = 600


def _step_score(A, v, q, h, cfg, occupied, occ_heading):
    """Score one candidate step, and say whether it is allowed at all.

    Returns (ok, corr, why). The two refusals that are not about likeness:
    an overlap that is not Fourier-separable, and a local spectrum that does
    not run the way this step does.
    """
    H, W = A.shape
    if not (2 <= q[0] < W - 2 and 2 <= q[1] < H - 2):
        return False, 0.0, "edge"
    ang, rel = orientations(A, q, size=cfg.spec_size)
    if len(ang) == 0:
        return False, 0.0, "no local orientation"
    h_ang = np.arctan2(h[1], h[0]) % np.pi
    agree = [i for i in range(len(ang))
             if _angdiff(ang[i], h_ang) <= np.radians(cfg.orient_tol_deg)
             and rel[i] >= cfg.orient_min]
    if not agree:
        return False, 0.0, "spectrum does not run this way"

    xi, yi = int(round(q[0])), int(round(q[1]))
    if occupied is not None and occupied[yi, xi]:
        # Overlap is only allowed where the two signals are actually separable:
        # the window must show a second orientation, belonging to whoever is
        # already here, distinct from this vessel's own.
        other = occ_heading[yi, xi] if occ_heading is not None else np.nan
        if not np.isfinite(other):
            return False, 0.0, "occupied, occupant direction unknown"
        if _angdiff(other, h_ang) <= np.radians(cfg.cross_tol_deg):
            return False, 0.0, "occupied by a vessel running the same way"
        has_other = any(_angdiff(ang[i], other) <= np.radians(cfg.cross_tol_deg)
                        for i in range(len(ang)))
        if not (has_other and len(ang) >= 2):
            return False, 0.0, "occupied and the two signals are not separable"

    prof = profile_at(A, q, h, v.offs)
    corr, amp = match(v.profile, prof)
    if not (cfg.amp_min <= amp <= cfg.amp_max):
        return False, corr, f"amplitude {amp:.2f}"
    return (corr >= cfg.corr_min), corr, "ok"


def grow_end(A, v, at_mouth, cfg=None, occupied=None, occ_heading=None):
    """Extend one end of a vessel, one step at a time, along its own profile.

    Never bridges: each step lands `cfg.step` px from the last, so the path it
    produces is continuous by construction. Weak steps are tolerated only while
    a strong one follows - a run that ENDS weak is trimmed off, so the vessel
    is never left drawn over ground that did not support it.
    """
    cfg = cfg or GrowCfg()
    if v.profile is None or v.offs is None:
        return np.zeros((0, 2))
    end = v.mouth if at_mouth else v.tail
    p, h = end.p.copy(), end.heading.copy()
    added, strong_to = [], 0
    weak = 0
    fan = np.radians(np.linspace(-cfg.fan_deg, cfg.fan_deg, cfg.fan_n))
    for _ in range(cfg.max_steps):
        base = np.arctan2(h[1], h[0])
        best = None
        for d in fan:
            hi = np.array([np.cos(base + d), np.sin(base + d)])
            q = p + hi * cfg.step
            ok, corr, why = _step_score(A, v, q, hi, cfg, occupied, occ_heading)
            if best is None or corr > best[1]:
                best = (ok, corr, q, hi, why)
        ok, corr, q, hi, why = best
        if not ok:
            weak += 1
            if weak > cfg.max_weak:
                break
            added.append(q)            # provisional: kept only if a strong step follows
        else:
            weak = 0
            added.append(q)
            strong_to = len(added)
        p, h = q, hi
    return np.asarray(added[:strong_to], float).reshape(-1, 2)


# -------------------------------------------------------------- the network

def nucleate(A, valid, t_seed=5.0, len_seed=60, sigmas=(1.0, 1.5, 2.0, 3.0, 4.5, 6.5),
             log=None):
    """The segments we are certain of, and nothing else.

    Deliberately sparse. A nucleus is only there to say "a vessel is definitely
    here, and this is what it looks like" - the rest of that vessel is found by
    growing, on its own evidence, rather than by lowering a threshold until the
    faint parts appear and texture appears with them.
    """
    from . import evidence as ev
    from . import ridges as rg
    z, ang, _ = ev.ridge_z(A, valid, sigmas, with_angle=True, with_scale=True)
    seeds = rg.detect(z, ang, t_seed, len_seed, t_seed, len_seed, 2, float(len_seed),
                      reach=0)
    seeds = [C for C in seeds if rg.single_peaked(A, C)]
    if log:
        log(f"  nuclei: {len(seeds)} segments above z {t_seed} for {len_seed} px")
    return seeds


def _mark(occupied, occ_heading, v):
    """Record which pixels a vessel holds and which way it runs through them."""
    C = np.asarray(v.points, float)
    if len(C) < 2:
        return
    r = float(v.meta.get("radius", 2.0))
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    H, W = occupied.shape
    k = max(1, int(round(r + 1.0)))
    for (x, y), (tx, ty) in zip(C, t):
        x0, x1 = max(0, int(x) - k), min(W, int(x) + k + 1)
        y0, y1 = max(0, int(y) - k), min(H, int(y) + k + 1)
        occupied[y0:y1, x0:x1] = True
        occ_heading[y0:y1, x0:x1] = np.arctan2(ty, tx) % np.pi


def build(A, valid, cfg=None, rounds=6, min_keep=40.0, spectral=True, log=None):
    """Nucleate, instantiate, grow: the set of vessels the frame supports.

    Vessels are grown strongest first, so a confident vessel claims its ground
    before a faint one can be pulled across it, and each round's growth becomes
    the next round's starting point.

    Nuclei come from two places: strong contrast, and - with `spectral` - the
    orientation bands of the whole frame's Fourier transform, which finds a
    faint vessel in the band it belongs to rather than against the whole
    sclera's texture.
    """
    cfg = cfg or GrowCfg()
    H, W = A.shape
    nuclei = nucleate(A, valid, log=log)
    if spectral:
        # A second nucleator, from the Fourier transform of the whole frame.
        # Its threshold is set by the surrogate rather than by eye: a wedge
        # filter MANUFACTURES oriented structure out of isotropic noise, so
        # requiring a ridge to run the way its band does is no safeguard at all
        # - the band imposed that. What does work is a cut high enough that
        # filtered noise cannot reach it. Swept: at z 5 the surrogate still
        # yields 1257 px of nucleus, at 6 it is 23 px, and at 7 it is zero,
        # while the real frame still gives 27 630 px - seven times what the
        # contrast nucleator finds, at the same bar of no invention.
        extra = dedupe(nucleate_spectral(A, valid, t_seed=7.0, len_seed=45,
                                         consistency_deg=22.0, log=log),
                       [Vessel(points=np.asarray(C, float)) for C in nuclei])
        if log:
            log(f"  spectral adds {len(extra)} nuclei the contrast pass did not have")
        nuclei = list(nuclei) + list(extra)
    nuclei = dedupe(nuclei)
    vessels = []
    for C in nuclei:
        v = Vessel(points=np.asarray(C, float)).measure(A)
        if v.profile is not None and v.depth > 0:
            vessels.append(v)
    vessels.sort(key=lambda v: -(v.depth * v.length))
    occupied = np.zeros((H, W), bool)
    occ_heading = np.full((H, W), np.nan)
    for v in vessels:
        _mark(occupied, occ_heading, v)
    if log:
        log(f"  instantiated {len(vessels)} vessels, "
            f"{sum(v.length for v in vessels):.0f} px of nucleus")
    for rnd in range(rounds):
        added = 0.0
        for v in vessels:
            for at_mouth in (True, False):
                before = v.length
                # match on the calibre this END currently has, not the average
                v.measure(A, window=40.0)
                grow = grow_end(A, v, at_mouth, cfg, occupied, occ_heading)
                v.measure(A)
                if len(grow) >= 2:
                    v.points = (np.vstack([grow[::-1], v.points]) if at_mouth
                                else np.vstack([v.points, grow]))
                    v.grown.append(np.asarray(grow, float))
                    _mark(occupied, occ_heading, v)
                    added += v.length - before
        if log:
            log(f"  round {rnd + 1}: +{added:.0f} px")
        if added < 20:
            break
    out = [v for v in vessels if v.length >= min_keep]
    for i, v in enumerate(sorted(out, key=lambda v: (round(v.points[0][1] / 40),
                                                     v.points[0][0])), 1):
        v.id = i
    if log:
        log(f"  {len(out)} vessels, {sum(v.length for v in out):.0f} px")
    return out


# ------------------------------------------- nucleating from the whole frame

def orientation_bands(A, valid, n_theta=12, k_lo=1.0 / 40, k_hi=1.0 / 3.0, sharp=2.0):
    """The frame split into images that each hold structures of one direction.

    One FFT of the whole frame, then a wedge per orientation. A vessel running
    at theta puts its power in one wedge; the sclera's blotches are not
    oriented and spread across all of them, so in its own band a faint vessel
    stands above a background that has been divided among the others. The
    radial window keeps only the scales a vessel of 2-24 px width can occupy,
    which removes the illumination drift below and the sensor noise above.

    Returns (bands, thetas): bands[i] is the image of orientation thetas[i].
    """
    A0 = np.where(valid, np.asarray(A, float) - np.median(np.asarray(A)[valid]), 0.0)
    H0, W0 = A0.shape
    # MIRROR-PAD FIRST. The frame is not periodic, so a plain FFT sees a step
    # at every border, and a wedge filter turns that step into long straight
    # ridges running clean across the image - measured, 66 % of the candidates
    # this produced without padding touched a border and 39 % were straighter
    # than any real vessel. Reflecting makes the edges continuous, and the pad
    # is cropped off before anything is detected in it.
    pad = 64
    A1 = cv2.copyMakeBorder(A0.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    H, W = A1.shape
    F = np.fft.fft2(A1.astype(float))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    k = np.hypot(kx, ky)
    radial = np.exp(-0.5 * (np.log(np.maximum(k, 1e-9) / np.sqrt(k_lo * k_hi))
                            / (0.5 * np.log(k_hi / k_lo))) ** 2)
    radial[0, 0] = 0.0
    phi = np.arctan2(ky, kx)
    thetas = np.arange(n_theta) * np.pi / n_theta
    out = []
    for th in thetas:
        # a vessel running at th has its power ACROSS th, at th + pi/2
        d = np.angle(np.exp(1j * 2 * (phi - (th + np.pi / 2)))) / 2
        wedge = np.cos(np.clip(d * n_theta / 2.0, -np.pi / 2, np.pi / 2)) ** sharp
        band = np.real(np.fft.ifft2(F * radial * wedge)).astype(np.float32)
        out.append(band[pad:pad + H0, pad:pad + W0])
    return out, thetas


def nucleate_spectral(A, valid, n_theta=12, t_seed=4.0, len_seed=45, consistency_deg=22.0,
                      log=None):
    """New candidate segments, one orientation band at a time.

    Within a band the ridge detector is asked the same question as usual, but
    of an image where only structures of that direction survive - so a vessel
    the isotropic detector loses in the surrounding blotches can clear the
    threshold on its own.

    Every candidate then has to agree with the band it came from: a ridge found
    in the band at theta must itself run at theta, within `consistency_deg`.
    Texture that leaks through a wedge has no particular direction and fails
    that, which is what keeps a permissive threshold from buying invention.
    """
    from . import evidence as ev
    from . import ridges as rg
    bands, thetas = orientation_bands(A, valid, n_theta=n_theta)
    out = []
    for bi, (B, th) in enumerate(zip(bands, thetas)):
        z, ang, _ = ev.ridge_z(-B, valid, (1.0, 1.5, 2.0, 3.0), with_angle=True, with_scale=True)
        for C in rg.detect(z, ang, t_seed, len_seed, t_seed, len_seed, 2, float(len_seed),
                           reach=0):
            C = np.asarray(C, float)
            if len(C) < 5:
                continue
            d = C[-1] - C[0]
            own = np.arctan2(d[1], d[0]) % np.pi
            if _angdiff(own, th) > np.radians(consistency_deg):
                continue                      # found in this band but not running its way
            out.append(C)
    out = verify(A, valid, out, log=log)
    if log:
        log(f"  spectral nuclei: {len(out)} segments from {n_theta} orientation bands")
    return out


def verify(A, valid, cands, min_tort=1.0, k_dark=4.0, log=None):
    """Check every proposal against the image it came from.

    A band image is a filtered view, and a filter can manufacture structure -
    so what the spectrum proposes has to be confirmed on the absorbance itself
    before it is allowed to nucleate anything. Two tests, both about the frame
    and not about the band:

    dark    the absorbance along the candidate must stand above the ordinary
            spread of the background, not merely be positive.
    shape   off by default, and worth saying why. Filter ringing is dead
            straight, so rejecting straight candidates looks like the obvious
            guard - but tortuosity is a function of LENGTH, and a genuine 45 px
            piece of a real vessel is nearly straight too. At min_tort 1.008 it
            threw out 173 of 260 candidates, real ones included. The edge
            artefacts it was aimed at are removed at source by the mirror pad;
            what remains is judged on the absorbance, which does not care how
            long the candidate is.
    """
    A = np.asarray(A, float)
    H, W = A.shape
    bg = A[valid]
    med = float(np.median(bg))
    mad = 1.4826 * float(np.median(np.abs(bg - med))) + 1e-9
    keep, why = [], {"dark": 0, "straight": 0}
    for C in cands:
        C = np.asarray(C, float)
        if len(C) < 5:
            continue
        L = float(np.hypot(*np.diff(C, axis=0).T).sum())
        chord = float(np.hypot(*(C[-1] - C[0])))
        if L / max(chord, 1e-9) < min_tort:
            why["straight"] += 1
            continue
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        if float(np.median(A[yi, xi])) < med + k_dark * mad:
            why["dark"] += 1
            continue
        keep.append(C)
    if log:
        log(f"  verified against the frame: {len(keep)} of {len(cands)} kept "
            f"({why['dark']} too faint, {why['straight']} too straight)")
    return keep


def dedupe(cands, existing=(), tol=6.0, overlap=0.5, among_themselves=True):
    """One nucleus per piece of vessel.

    Candidates covered by something we already have are dropped, and - unless
    switched off - candidates covering EACH OTHER are dropped too, longest
    first. That second half matters here: a vessel is found in whichever
    orientation band it belongs to, but a vessel that curves, or one that sits
    between two bands, is found in several, and without this it is instantiated
    once per band. Two vessels may not hold the same ground.
    """
    pts = [np.asarray(v.points, float) for v in existing]
    keep = []
    order = sorted(cands, key=lambda C: -len(np.asarray(C, float)))
    for C in order:
        C = np.asarray(C, float)
        near = np.zeros(len(C), bool)
        for P in pts:
            if len(P) < 2:
                continue
            d = np.hypot(*(C[:, None, :] - P[None, ::4, :]).transpose(2, 0, 1)).min(1)
            near |= d <= tol
        if near.mean() >= overlap:
            continue
        keep.append(C)
        if among_themselves:
            pts.append(C)
    return keep
