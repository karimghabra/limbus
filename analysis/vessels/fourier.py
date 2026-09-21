"""Large, dark, in-focus vessel segments, found by matched filtering in
Fourier space over the whole frame.

The staged detector asks "is this pixel on a ridge?" at a few blur scales and
then has to clean up what that lets in. This asks a narrower question with a
better-defined answer: **at this pixel, which straight vessel best explains
the absorbance?** The bank of candidate answers is every combination of

    orientation theta   (n_theta over 180 deg)
    radius r            (px)
    optical blur sigma  (px)

and each is a template rendered from the same physical model the fit uses -
a cylinder's chord absorbance `sqrt(1 - (d/r)^2)` blurred by sigma, tapered
along its axis. Correlating 256 such templates with the frame one at a time
would be hopeless in image space; as one forward FFT of the frame and one
multiply per template it costs a few seconds.

Each template is mean-removed, whitened by the frame's own background power
spectrum, and scaled to unit norm, and the statistic within one (radius, blur)
is the ORIENTATION CONTRAST - the best orientation's response minus the median
over all orientations. Getting the largest vessels found at all took four
corrections, each of which was needed and none of which was guessed:

  background model   The annulus median is inflated by the vessels' own power,
                     so whitening by it suppressed exactly the thick vessels
                     this is meant to find first (z fell from 4.9 at r=4 to
                     0.8 at r=13). A power-law fit that rejects positive
                     residuals gives a background they cannot inflate.
  per-radius scale   A whitened response is not comparable BETWEEN radii - a
                     wide vessel's power sits at low k where the sclera has
                     most of its own. Each radius is normalised by its own
                     spread before the radii compete, as `evidence.py` does.
  orientation        Whitening is isotropic and cannot separate a trunk's
                     low-frequency power from a sclera blotch's. A blotch
                     answers every orientation alike; a vessel answers one. The
                     median over theta subtracts the blotch away, and this took
                     the trunk response from z 8.5 to 18.8 against 0.7 on
                     background.
  aspect ratio       At the old 9 px axial length an r=13 template was 27 px
                     wide and 9 long: a blob, and the sclera is full of blobs.

    radius   argmax over r      how wide the vessel is
    blur     argmax over sigma  how sharply it is imaged (small = in focus)

Of these two, only the radius is calibrated. Against vessels planted at a
known radius the bank returns the right one 96-100 % of the time from 4 to
13 px. The blur is monotone in the true blur but reads LOW by roughly a factor
two - planted at 1.2 / 1.8 / 2.6 / 3.8 px it comes back as 0.8 / 1.2 / 1.6 /
2.0 - and the cause is not known, so it ranks how sharply two vessels in one
frame are imaged but its number is not a PSF.

Darkness is measured on the absorbance image where the bank placed the vessel,
not inferred from the response, which is an SNR.

This module deliberately finds the large vessels first. They are the ones that
are unambiguous, and the plan is to work down in calibre, removing each vessel
from the SPECTRUM rather than from the image, so that a thin vessel crossing a
thick one survives the removal of the thick one.

Where it stands, measured on a sharp crop against every centreline the image
plainly shows: 64 % of the trunk centreline (absorbance >= 0.45) is covered,
60 % at >= 0.18, and 100 % of what it does report lies on absorbance >= 0.04
against a background of 0.031. The missing third is real and visible - the
thickest, softest trunk in the crop is still not traced.
"""
import cv2
import numpy as np

from . import evidence as ev
from . import ridges

# The bank. Radii are half-widths in px; a "large" vessel here is r >= 4,
# which at the working magnification is roughly the calibre at which a vessel
# is unambiguous against sclera texture.
# A real trunk on a sharp crop measures r = 13.5 px with 4.8 px of blur, fitted
# to its own cross-profile with a residual of 7 % of peak - the cylinder model
# describes it well, but the bank used to stop at r = 13 and sigma = 3.8 and so
# could not represent it at all.
RADII = (3.0, 4.5, 6.5, 9.0, 13.0, 18.0)
SIGMAS = (1.2, 1.8, 2.6, 3.8, 5.2)
N_THETA = 16
ALONG = 20.0        # axial taper of the template (px, Gaussian sd). A template
                    # only 9 px long is 27 px WIDE at r=13: a blob, and the
                    # sclera is full of blobs. Lengthening it to 20 took the
                    # trunk response from z 8.5 to 18.8. Past ~28 it falls
                    # again, because real vessels curve away from a straight
                    # template.


def chord(d, r, sigma, step=0.05):
    """Blurred cross-profile of a cylinder of radius r: the chord length
    `sqrt(1-(d/r)^2)`, peak 1, convolved with a Gaussian of width sigma."""
    lo = -(r + 5 * sigma)
    u = np.arange(lo, -lo + step, step)
    p = np.sqrt(np.clip(1 - (u / r) ** 2, 0, None))
    k = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + step, step) / sigma) ** 2)
    k /= k.sum()
    p = np.convolve(p, k, mode="same")
    return np.interp(d, u, p, left=0.0, right=0.0) / max(p.max(), 1e-9)


def template(theta, r, sigma, along=ALONG):
    """The mean-removed, unit-norm filter for one vessel shape."""
    # 2.2 sigma of axial taper, not 3: the box is what sets `pad`, and `pad`
    # is subtracted from the usable frame at every edge.
    half = int(np.ceil(max(r + 4 * sigma, 2.2 * along)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    ct, st = np.cos(theta), np.sin(theta)
    d = -x * st + y * ct                 # across the vessel
    s = x * ct + y * st                  # along it
    cross = chord(d, r, sigma)
    t = cross * np.exp(-0.5 * (s / along) ** 2)
    z = t - t.mean()
    z /= max(np.linalg.norm(z), 1e-12)
    return z.astype(np.float32)


def noise_spectrum(F, shape, floor_pct=1.0, model="powerlaw", deg=1):
    """Isotropic power spectrum of the frame's BACKGROUND, from the frame.

    Taken as the median of |F|^2 over each radial annulus. The median over
    orientations is what makes this an estimate of the background rather than
    of everything: a vessel is anisotropic, putting its power in a few
    directions within an annulus, so it moves the annulus's median hardly at
    all while dominating its mean.

    Whitening by this is the difference between a matched filter and a
    CORRECT one. Sclera texture is strongly low-frequency, so without it the
    widest, blurriest template in the bank wins nearly everywhere - it is the
    one that looks most like a texture blotch - and the argmax that is
    supposed to measure a vessel's calibre measures nothing.
    """
    Ph, Pw = shape
    ky = np.fft.fftfreq(Ph)[:, None]
    kx = np.fft.rfftfreq(Pw)[None, :]
    kr = np.hypot(ky, kx)
    P = (F.real ** 2 + F.imag ** 2)
    nb = 256
    idx = np.minimum((kr / (kr.max() + 1e-12) * nb).astype(np.int32), nb - 1)
    prof = np.array([np.median(P[idx == b]) if np.any(idx == b) else np.nan
                     for b in range(nb)])
    ok = np.isfinite(prof)
    prof = np.interp(np.arange(nb), np.flatnonzero(ok), prof[ok])

    if model == "powerlaw":
        # The annulus median is the background PLUS whatever the vessels put
        # there, and thick vessels put a great deal at low k. Whitening by it
        # therefore suppresses the very vessels this is meant to find first:
        # measured against planted vessels, the response fell from z 4.9 at
        # radius 4 px to 0.8 at radius 13 px, the wrong way round.
        #
        # Sclera texture is close to a power law in |k|, and vessels show up
        # on it as positive bumps. Fitting a line through log P against log k
        # and rejecting positive residuals gives a background the vessels
        # cannot inflate, because the fit walks away from their bumps rather
        # than following them.
        # `deg` is 1 by default: a straight line in log-log, i.e. a power law.
        # Higher degrees were tried on planted vessels of known blur, on the
        # theory that the background flattens onto a sensor noise floor at
        # high k and a straight line would under-predict it. It does not help:
        # degree 3 collapsed every blur from 1.2 to 3.8 px onto one answer and
        # degree 4 was erratic, while degree 1 stayed monotone and tight. The
        # curvature is left available but off.
        kb = (np.arange(nb) + 0.5) / nb * kr.max()
        use = (kb > 0) & (prof > 0)
        lx, ly = np.log(kb[use]), np.log(prof[use])
        w = np.ones_like(ly)
        for _ in range(5):
            c = np.polyfit(lx, ly, deg, w=w)
            resid = ly - np.polyval(c, lx)
            s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            w = 1.0 / (1.0 + np.clip(resid / max(s, 1e-9), 0, None) ** 2)
        fit = np.full(nb, np.nan)
        fit[use] = np.exp(np.polyval(c, lx))
        okf = np.isfinite(fit)
        prof = np.interp(np.arange(nb), np.flatnonzero(okf), fit[okf])

    P0 = np.maximum(prof[idx], np.percentile(prof, floor_pct))

    # The two axes are not background-like and must not be whitened as though
    # they were. Row and column sensor offsets survive averaging; `image.
    # remove_fixed_pattern` takes out only the NARROW part of each profile, on
    # purpose, so as not to erase a vessel running along a row, and the broad
    # remainder lives exactly on kx=0 and ky=0. Measured on a sharp crop those
    # two lines are 0.36 % of the spectrum but sit ~6400x above the annulus
    # median, so dividing them by an isotropic estimate AMPLIFIES them: they
    # carried 32 % of the whitened power and painted the response map with
    # horizontal and vertical streaks in which no vessel was visible.
    #
    # So give each axis its own power, from itself, smoothed along its length.
    # Taking the max with the isotropic estimate means this can only ever
    # suppress the axes, never boost them. The price is real: a perfectly
    # horizontal or vertical vessel loses sensitivity, since its own power
    # lands there too. Vessels that straight are already the case the
    # fixed-pattern remover treats as suspect.
    for ax in (0, 1):
        line = (np.abs(kx) < 1e-12) if ax else (np.abs(ky) < 1e-12)
        sel = np.broadcast_to(line, P.shape)
        vals = P[sel]
        if vals.size < 5:
            continue
        w = min(9, vals.size // 2 * 2 + 1)
        sm = np.median(np.lib.stride_tricks.sliding_window_view(
            np.pad(vals, w // 2, mode="edge"), w), axis=1)
        P0[sel] = np.maximum(P0[sel], sm)
    return P0


def _bank(n_theta=N_THETA, radii=RADII, sigmas=SIGMAS, along=ALONG):
    out = []
    for i in range(n_theta):
        th = np.pi * i / n_theta
        for r in radii:
            for sg in sigmas:
                out.append((th, r, sg, template(th, r, sg, along)))
    return out


def respond(A, valid, n_theta=N_THETA, radii=RADII, sigmas=SIGMAS, along=ALONG,
            whiten=True, alpha=1.0, model="powerlaw", deg=1, contrast=True,
            roll=40.0, margin=0.0,
            log=None):
    """Correlate the whole frame with the bank; keep the best answer per pixel.

    Returns a dict of maps, all the shape of A:
        score   correlation with the winning template (absorbance * sqrt(px))
        z       the same in units of its own robust spread over valid pixels
        theta   winning orientation, radians, direction ALONG the vessel
        radius  winning radius, px
        blur    winning sigma, px  (small = in focus)
        inner   where the response means anything: `valid` shrunk by half a
                template, since nearer the edge the correlation is partly with
                the mirror fill

    Darkness is NOT taken from here. The whitened response is an SNR, not an
    absorbance, and converting one to the other depends on the noise estimate;
    how dark a vessel is gets measured straight off `A` where it was found.
    """
    H, W = A.shape
    bank = _bank(n_theta, radii, sigmas, along)
    pad = max(z.shape[0] for _, _, _, z in bank)
    # Mirror-pad: the FFT is periodic, and an averaged frame is not. Without
    # this, the left edge convolves with the right and the border grows
    # oriented structure that was never in the image.
    # Outside `valid` the frame has no data. Filling it with 0 puts a step of
    # the background absorbance (~0.03, as deep as a faint vessel) all the way
    # round the covered region, and the bank answers it with a ring of huge
    # responses that swamps every real vessel. Referencing A to its own mean
    # first makes the fill continuous with the data instead.
    #
    # Mirror padding is not enough either. A whitened filter is not compact -
    # dividing by the background spectrum gives it long tails - so the fold at
    # a mirrored edge acts like a strong symmetric feature many templates deep,
    # and the strongest responses in the frame came back as horizontal and
    # vertical bars lying along the borders, with only 15 % of them on a dark
    # pixel at all. Tapering the data smoothly to zero at the edge of `valid`
    # removes the discontinuity instead of reflecting it.
    base = float(np.mean(A[valid])) if valid.any() else 0.0
    W0 = np.where(valid, A - base, 0).astype(np.float32)
    if roll > 0:
        d = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
        # the frame's own edge counts as an edge of the data, not just the
        # boundary of the covered region
        ey, ex = np.ogrid[:H, :W]
        d = np.minimum(d, np.minimum(np.minimum(ey, H - 1 - ey),
                                     np.minimum(ex, W - 1 - ex)).astype(np.float32))
        W0 *= (0.5 * (1 - np.cos(np.pi * np.clip(d / roll, 0, 1)))).astype(np.float32)
    Ap = cv2.copyMakeBorder(W0, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    Ph, Pw = cv2.getOptimalDFTSize(Ap.shape[0]), cv2.getOptimalDFTSize(Ap.shape[1])
    Ap = cv2.copyMakeBorder(Ap, 0, Ph - Ap.shape[0], 0, Pw - Ap.shape[1],
                            cv2.BORDER_CONSTANT, value=0)
    F = np.fft.rfft2(Ap)
    P = noise_spectrum(F, (Ph, Pw), model=model, deg=deg) if whiten else None
    # alpha scales how hard the whitening bites: 1 is the full prewhitened
    # matched filter, 0 is none.
    Pa = P ** alpha if whiten else None

    # Kept PER RADIUS, not pooled. A whitened matched filter is the right
    # detector for a given shape but it is not comparable BETWEEN shapes here:
    # a wide vessel's power sits at low k, which is exactly where the sclera
    # has most of its own, so the same vessel scores ~3x lower at r=13 than at
    # r=4 (measured on planted vessels: z 43.6 -> 15.0). Pooling the bank and
    # taking one argmax therefore lets a narrow template sitting on a trunk's
    # steep flank outbid the correct wide one, and the trunks - the vessels
    # this is supposed to find FIRST - come out weakest of all.
    #
    # So each radius is put in units of its own spread before the radii are
    # compared, which is the same thing `evidence.py` does across its scales
    # and for the same reason.
    #
    # Within one (radius, blur) the statistic is the ORIENTATION CONTRAST: the
    # best orientation's response minus the median over all orientations, not
    # the response itself. Whitening is isotropic, so on its own it cannot
    # tell a trunk's low-frequency power from a sclera blotch's - and the
    # sclera is full of blotches at exactly the scale of a thick vessel, which
    # is why the trunks scored no better than z 4 while a crisp thin vessel
    # scored 20. What actually separates them is that a vessel is oriented and
    # a blotch is not: a blotch answers every orientation alike, so the median
    # over theta subtracts it away, while a vessel answers one.
    thetas = [np.pi * i / n_theta for i in range(n_theta)]
    per = {r: [np.full((H, W), -np.inf, np.float32),
               np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)]
           for r in radii}
    byshape = {}
    for th, r, sg, z in bank:
        k = np.zeros((Ph, Pw), np.float32)
        n = z.shape[0]
        k[:n, :n] = z
        k = np.roll(k, (-(n // 2), -(n // 2)), axis=(0, 1))   # centre on the origin
        Z = np.fft.rfft2(k)
        if whiten:
            # <A, z>_whitened = IFFT(F conj(Z) / P); the filter's own whitened
            # norm is what makes responses comparable across the bank.
            num = F * np.conj(Z) / Pa
            zt = np.fft.irfft2(Z / np.sqrt(Pa), s=(Ph, Pw))
            nrm = max(float(np.linalg.norm(zt)), 1e-12)
        else:
            num, nrm = F * np.conj(Z), 1.0
        # correlation, not convolution: conjugate the filter's transform
        c = np.fft.irfft2(num, s=(Ph, Pw))[pad:pad + H, pad:pad + W] / nrm
        byshape.setdefault((r, sg), []).append(c.astype(np.float32))
        if len(byshape[(r, sg)]) < n_theta:
            continue
        stack = np.stack(byshape.pop((r, sg)))
        if contrast:
            ang = stack.max(0) - np.median(stack, 0)
        else:
            ang = stack.max(0)
        who = stack.argmax(0)
        b = per[r]
        hit = ang > b[0]
        b[0] = np.where(hit, ang, b[0])
        b[1] = np.where(hit, np.asarray(thetas, np.float32)[who], b[1])
        b[2] = np.where(hit, np.float32(sg), b[2])
    # A matched filter has no answer within half a template of the edge of the
    # data: there the correlation is partly with the mirror fill. Say so by
    # shrinking the mask rather than reporting responses that mean nothing.
    # borderValue=0: cv2.erode treats outside-the-image as set by default, so
    # without it the frame's own edge is never trimmed and `inner` comes back
    # essentially equal to `valid` - which is how a band of border responses
    # 76 px wide survived into the detections.
    # The margin is `roll`, not the whole template. The old code eroded by a
    # full template half-width, which on a 460-row crop threw away 58 % of the
    # trunk centreline - and it was there to fight border responses that the
    # taper has since removed at source: beyond the taper the data rolls to
    # zero, so an edge response is attenuated, not invented.
    k = int(max(roll, pad * margin)) | 1
    inner = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0

    best = np.full((H, W), -np.inf, np.float32)
    z_m = np.full((H, W), -np.inf, np.float32)
    th_m = np.zeros((H, W), np.float32)
    r_m = np.zeros((H, W), np.float32)
    sg_m = np.zeros((H, W), np.float32)
    for r in radii:
        c, th_r, sg_r = per[r]
        v = c[inner]
        med = float(np.median(v))
        # Spread from the NEGATIVE half only. Vessels make the response
        # positive, so a two-sided MAD is inflated by the very things being
        # detected, and a vessel-rich frame raises its own threshold.
        spread = 1.4826 * float(np.median(med - v[v < med])) if (v < med).any() else 1.0
        zr = (c - med) / max(spread, 1e-9)
        hit = zr > z_m
        z_m = np.where(hit, zr, z_m)
        best = np.where(hit, c, best)
        th_m = np.where(hit, th_r, th_m)
        sg_m = np.where(hit, sg_r, sg_m)
        r_m = np.where(hit, np.float32(r), r_m)
        if log:
            log(f"[fourier]   r={r:<5g} median {med:+.4f} spread {spread:.4f}  "
                f"z>=6 on {int(((zr >= 6) & inner).sum()):6d} px")
    best = np.where(inner, best, 0).astype(np.float32)
    z_m = np.where(inner, z_m, 0).astype(np.float32)
    if log:
        log(f"[fourier] bank {len(bank)} templates, frame {W}x{H} padded to {Pw}x{Ph}")
    return {"score": best, "z": z_m, "theta": th_m, "radius": r_m,
            "blur": sg_m, "inner": inner, "whiten": whiten}


def segments(A, valid, resp=None, t_hi=8.0, t_lo=3.5, min_radius=0.0,
             max_blur=99.0, min_depth=0.05, min_len=40.0, log=None, **kw):
    """Ridge points that are large and dark, linked into polylines.

    Each word is one test on what the bank measured, so a rejection can always
    be named:
        large     radius >= min_radius   (0 keeps every calibre)
        in focus  blur   <= max_blur     (on the uncalibrated blur scale)
        dark      absorbance A at the ridge >= min_depth

    Darkness is read off the absorbance image rather than derived from the
    response: a whitened correlation is an SNR and would call a shallow vessel
    in a quiet patch "dark".

    The response is taken with hysteresis, not a single threshold. A trunk's
    response dips where another vessel crosses it and where it curves away
    from a straight template, and a single cut at t_hi chopped the trunks into
    dashes; a run of weaker ridge that is CONNECTED to a confident one is part
    of the same vessel.
    """
    R = respond(A, valid, log=log, **kw) if resp is None else resp
    dark = cv2.GaussianBlur(A, (0, 0), 1.0)
    # across-ridge direction for the non-maximum suppression
    ok = (ev.nms(R["score"], R["theta"] + np.pi / 2) & R["inner"]
          & (R["radius"] >= min_radius) & (R["blur"] <= max_blur)
          & (dark >= min_depth))
    weak, strong = ok & (R["z"] >= t_lo), ok & (R["z"] >= t_hi)
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    alive = np.zeros(n, bool)
    alive[lab[strong]] = True
    alive[0] = False
    m = alive[lab]
    if log:
        log(f"[fourier] ridge px: {int(ok.sum())} pass the tests, "
            f"{int(strong.sum())} reach z {t_hi}, {int(m.sum())} kept after hysteresis")
    segs = ridges.trace(m, min_len=min_len)
    if log:
        tot = sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in segs if len(c) > 1)
        log(f"[fourier] {len(segs)} segments, {tot:.0f} px total")
    return segs, R
