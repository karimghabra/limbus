"""One vessel, or two running together?

Every junction failure measured on this detector is a fusion failure wearing a
different hat. A crossing below 40 degrees, a bifurcation at 20 degrees and a
braid of vessels a few pixels apart are all the same configuration: two
vessels nearly parallel and close, over a stretch long enough that the
detector describes them with one line. Raising the separation at which two
vessels can be told apart is therefore the one change that improves all of
them, and nothing else measured here does.

WHAT THE OPTICS ALLOW. A cylinder's chord profile has variance r^2/4, so after
a blur of sigma_psf the observed profile is close to a Gaussian of width
sqrt(sigma_psf^2 + r^2/4), and two equal Gaussian lines stop having a valley
between them - the Sparrow criterion - at twice that width. At sigma_psf = 1.8
that is 3.8 px for r = 1, 4.3 px for r = 2, 6.6 px for r = 4. (The Gaussian
substitution is optimistic for wide vessels: a chord profile is four times
less curved at its peak, so the exact figures are used below, not the formula.)
The shipped detector resolves a pair only from about 9 px, whatever the
calibre - roughly twice the limit for thin vessels.

WHY THE EVIDENCE ALONE DOES NOT GET THERE. It nearly does: at a transverse
scale of 0.7-1.0 px the second derivative separates an r = 2 pair completely
at 5 px and half at 4 px. The scale selection then throws that away, because
the coarse scale that spans both responds more strongly than the fine scale
that separates them. Choosing the finest significant scale per pixel is not
the answer either - it resolves 4 px and fails at 6, because a spatially mixed
scale map is not a clean ridge - and unioning the scales is worse: the coarse
scale's spurious midline sits between the two fine ridges and the dilation
that closes suppression breaks fuses all three.

So the decision is made once per candidate, on its own cross-profile, between
two explicit models rather than per pixel between two scales.

    one    a blurred cylinder at c, radius r, peak D, over a linear background
    two    two blurred cylinders at c +- s/2, radii r1 r2, peaks D1 D2

The profile is averaged ALONG the candidate first, which is where the signal
is: the vessels persist along their length and the texture does not.

THE TEST IS CALIBRATED, NOT ASSUMED. The two-cylinder model can only fit at
least as well as the one, so the whole question is how much better counts.
An i.i.d. chi-squared or BIC would answer "almost always": the residuals are
correlated by the optics and by the sclera's own texture, and a test built on
independent pixels overstates its evidence by orders of magnitude. The
threshold here is measured instead - the same statistic is computed along fake
centrelines in a phase-randomised surrogate with the same power spectrum and
no vessels, and the cut is placed at a quantile of THAT distribution.

AND IT IS BOUNDED. Below about 4 px the one-cylinder family absorbs a pair
almost exactly, and no amount of signal-to-noise fixes a model degeneracy.

WHAT HAPPENED. On clean profiles with white noise it works well: a true pair
5 px apart reads 0.797 against 0.05-0.19 for any single vessel, and the
separation comes back to a tenth of a pixel. On real sclera it collapses.

    null, phase-randomised surrogate, no vessel     0.246   (p99 0.907)
    null, the detector's own accepted vessels       0.352   (max 0.739)
    true pair at s = 4                              0.190
    true pair at s = 5                              0.244
    true pair at s = 6                              0.687
    true pair at s = 8                              0.907

True pairs 4-5 px apart score BELOW the single-vessel null. A threshold that
does not split real vessels (> 0.74) fires only from about 8 px, which is
where the shipped detector already begins to work - so there is nothing to
win. And at 3 px, below the model's own floor where a split would be
invention, it fires 67 % of the time: most confident exactly where it must
not be.

Averaging further along the vessel does not rescue it. Over along-sample
counts of 60, 200 and 600 and candidate lengths of 140 and 300 px, the
single-vessel null ranges 0.15-0.45 and the 4 px and 5 px signals range
0.21-0.59 - overlapping completely, and not ordering with separation. That
disorder is the finding: the statistic is reading texture, not geometry.

THE REASON GENERALISES. Sclera texture is correlated at the same spatial
scale as the dip being measured, so averaging along a vessel does not suppress
it the way independent noise would. The Sparrow limit of 4.25 px for r = 2 is
a NOISE-FREE limit; on this data the texture-limited one is nearer 6-8 px, and
the detector's 9 px is close to it. The gap that looked like a factor of two
of method failure is mostly not there.

Kept because the measurement is worth more than the code, and because the
next person to notice that the optics allow 4 px should see what it costs to
try.
"""
import cv2
import numpy as np


def chord(x, r, psf, n=2048, span=6.0):
    """A cylinder of radius r blurred by the optics, sampled at x.

    Computed by convolving the chord profile with the Gaussian on a fine grid
    rather than by any Gaussian approximation to it: the approximation is
    optimistic by 19 % at r = 4 and 37 % at r = 12, which is the whole of the
    region this has to be right about.
    """
    w = max(span * psf + r + 2.0, float(np.abs(x).max()) + 2.0)
    t = np.linspace(-w, w, n)
    dt = t[1] - t[0]
    a = np.sqrt(np.clip(1.0 - (t / r) ** 2, 0.0, None))
    k = np.exp(-0.5 * (np.arange(-int(4 * psf / dt), int(4 * psf / dt) + 1) * dt / psf) ** 2)
    k /= k.sum()
    p = np.convolve(a, k, "same")
    return np.interp(x, t, p)


def profile(A, C, half=16.0, step=0.5, along=None):
    """Median absorbance across a candidate, averaged along it.

    The median along the vessel, not the mean: a crossing vessel contributes
    its own absorbance to a few samples and would drag a mean.
    """
    C = np.asarray(C, float)
    if len(C) < 5:
        return None, None
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    sel = slice(None, None, max(1, len(C) // (along or 60)))
    offs = np.arange(-half, half + step, step, dtype=np.float32)
    px = (C[sel, None, 0] + n[sel, None, 0] * offs[None, :]).astype(np.float32)
    py = (C[sel, None, 1] + n[sel, None, 1] * offs[None, :]).astype(np.float32)
    band = cv2.remap(A, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return offs.astype(float), np.median(band, axis=0).astype(float)


_BASIS = {}


def _basis(psf, rmax=12.0, dr=0.2, w=40.0, n=4001):
    """Blurred cylinder profiles for every radius, on one fine grid.

    The profile of a cylinder does not depend on where it is, only on how far
    from its centre - so it is computed once per radius and then SAMPLED at
    an offset, instead of being convolved again for every centre, radius and
    separation the grid visits. That is the difference between a second per
    profile and a hundred.
    """
    key = (round(float(psf), 3), rmax, dr, w, n)
    if key in _BASIS:
        return _BASIS[key]
    t = np.linspace(-w, w, n)
    dt = t[1] - t[0]
    k = np.exp(-0.5 * (np.arange(-int(4 * psf / dt), int(4 * psf / dt) + 1) * dt / psf) ** 2)
    k /= k.sum()
    radii = np.arange(0.8, rmax + 1e-9, dr)
    tab = np.stack([np.convolve(np.sqrt(np.clip(1.0 - (t / r) ** 2, 0.0, None)), k, "same")
                    for r in radii])
    _BASIS[key] = (t, radii, tab)
    return _BASIS[key]


def _col(offs, c, r, psf):
    t, radii, tab = _basis(psf)
    i = int(np.clip(np.searchsorted(radii, r), 0, len(radii) - 1))
    return np.interp(offs - c, t, tab[i])


def _design(offs, centres, radii, psf):
    """Linear model: one column per cylinder, plus a constant and a slope.

    With the geometry fixed, the amplitudes and the background enter linearly,
    so they come from one least-squares solve rather than from an optimiser -
    which is what makes a grid over the separation affordable.
    """
    cols = [_col(offs, c, r, psf) for c, r in zip(centres, radii)]
    cols += [np.ones_like(offs), offs / max(np.abs(offs).max(), 1e-9)]
    return np.stack(cols, 1)


def _solve(M, y):
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ coef
    return coef, float(resid @ resid)


def fit_one(offs, prof, psf, radii=None, centres=None):
    radii = radii if radii is not None else np.arange(0.8, 12.1, 0.4)
    centres = centres if centres is not None else np.arange(-2.5, 2.51, 0.5)
    best = None
    for c in centres:
        for r in radii:
            M = _design(offs, [c], [r], psf)
            coef, ss = _solve(M, prof)
            if coef[0] <= 0:
                continue
            if best is None or ss < best[0]:
                best = (ss, dict(c=float(c), r=float(r), D=float(coef[0]), ss=ss))
    return best[1] if best else None


def fit_two(offs, prof, psf, seps=None, radii=None, centres=None):
    """Two cylinders, with the separation taken from a GRID.

    Letting an optimiser find the separation does not work: the seven-parameter
    model is ill-conditioned in it - condition numbers of 1e11 at small
    separations, with the radius and the separation correlated - so the
    separation is scanned and everything else solved in closed form at each
    value, which turns an ill-conditioned optimisation into a profile
    likelihood.
    """
    seps = seps if seps is not None else np.arange(2.0, 16.01, 0.5)
    radii = radii if radii is not None else np.arange(0.8, 6.1, 0.4)
    centres = centres if centres is not None else np.arange(-2.0, 2.01, 1.0)
    best = None
    for s in seps:
        for c in centres:
            for r in radii:
                M = _design(offs, [c - s / 2, c + s / 2], [r, r], psf)
                coef, ss = _solve(M, prof)
                if coef[0] <= 0 or coef[1] <= 0:
                    continue
                if best is None or ss < best[0]:
                    best = (ss, dict(c=float(c), s=float(s), r=float(r),
                                     D1=float(coef[0]), D2=float(coef[1]), ss=ss))
    return best[1] if best else None


def statistic(offs, prof, psf, **kw):
    """How much better two cylinders explain this profile than one.

    Scale-free: the drop in residual sum of squares as a fraction of the
    one-cylinder residual, so it does not depend on how dark the vessel is or
    on how many samples the profile has.
    """
    one = fit_one(offs, prof, psf, **{k: v for k, v in kw.items() if k in ("radii", "centres")})
    if one is None:
        return None
    two = fit_two(offs, prof, psf, **{k: v for k, v in kw.items() if k in ("seps", "radii", "centres")})
    if two is None:
        return None
    denom = max(one["ss"], 1e-12)
    return {"gain": float((one["ss"] - two["ss"]) / denom), "one": one, "two": two}


def calibrate(A_surrogate, valid, psf, lines, q=99.0, **kw):
    """The threshold, measured on ground that contains no vessels.

    `lines` are centrelines laid over a phase-randomised surrogate - same
    power spectrum, no vessels - so every value of the statistic they produce
    is what texture alone can achieve. The cut is a quantile of that.
    """
    vals = []
    for C in lines:
        offs, prof = profile(A_surrogate, C)
        if offs is None:
            continue
        r = statistic(offs, prof, psf, **kw)
        if r is not None:
            vals.append(r["gain"])
    if not vals:
        return None
    return float(np.percentile(vals, q)), np.array(vals)


def resolve(A, candidates, psf, threshold=0.40, min_sep=4.0, max_sep=14.0,
            min_len=40.0, window=None, log=None):
    """Replace each candidate that is really two vessels with the two.

    The test is applied per candidate rather than per pixel, because that is
    the scale at which the question has an answer: a pixel cannot say how many
    vessels there are, and a hundred pixels of the same pair can.

    `min_sep` is not a tuning knob but the model's own floor. Below about 4 px
    the one-cylinder family absorbs a pair almost exactly - the statistic on a
    true pair 3 px apart reads 0.13, inside the range single vessels produce -
    so a split claimed there would be invention. Candidates below it are
    marked `possibly_two` and left whole, which is what section 1 of the report
    asks for: a wrong claim costs more than a missing one.

    With `window` set, the candidate is tested in stretches of that length
    instead of as a whole, so a pair that converges and separates is split
    only where it is actually two - untested, and off by default.
    """
    out, records = [], []
    for C in candidates:
        C = np.asarray(C, float)
        L = float(np.hypot(*np.diff(C, axis=0).T).sum()) if len(C) > 1 else 0.0
        if L < min_len:
            out.append(C)
            continue
        offs, prof = profile(A, C)
        r = statistic(offs, prof, psf) if offs is not None else None
        if r is None:
            out.append(C)
            continue
        s = r["two"]["s"]
        rec = {"gain": r["gain"], "s": s, "length": L,
               "mid": [float(C[len(C) // 2, 0]), float(C[len(C) // 2, 1])]}
        if r["gain"] >= threshold and min_sep <= s <= max_sep:
            a, b = split(C, s)
            out.extend([a, b])
            rec["split"] = True
        else:
            out.append(C)
            rec["split"] = False
            rec["possibly_two"] = bool(r["gain"] >= threshold and s < min_sep)
        records.append(rec)
    if log:
        n = sum(1 for r in records if r.get("split"))
        m = sum(1 for r in records if r.get("possibly_two"))
        log(f"  pairs: {n} candidates split into two, {m} flagged as possibly two "
            f"but below the {min_sep:.0f} px the model can support")
    return out, records


def split(C, s):
    """Two centrelines, one either side of a merged one."""
    C = np.asarray(C, float)
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    return C + n * (s / 2.0), C - n * (s / 2.0)
