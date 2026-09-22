"""The large vessels, found the direct way.

A thick vessel is a dark region that is wide. That is the whole definition,
and it needs a threshold and a distance transform - not a filter bank. This
module exists because the Fourier matched-filter approach in `fourier.py`
spent a great deal of machinery answering "which straight vessel best explains
this pixel?" when the question for the trunks is only "is this dark, and is it
thick?", which the absorbance image answers directly.

    1. Threshold the absorbance with hysteresis, at the Isodata level - the
       level that equals the mean of the two classes it creates. Nothing is
       tuned.
    2. Skeletonise, and use the distance transform only as a GATE on how thick
       a region is - never as the reported calibre.
    3. Measure radius, depth and wall sharpness off the ABSORBANCE along each
       centreline (`measure`), not off the mask.

The split in 2-3 matters. The mask is a good way to say WHERE a vessel is and
a bad way to say how wide it is: a distance-transform radius moves 41 % as the
threshold is swept over its usable range, because it measures where the
threshold fell rather than where the vessel ends. The intensity half-width is
referenced to each vessel's own peak, so it cannot do that, and against
planted vessels of known radius it is accurate to about 0.2 px from 4 to 13 px
and unaffected by focus.

Every vessel therefore carries its own measured radius, peak absorbance and
wall sharpness, so the network can be worked from the largest calibre down by
sorting - and "in focus" is a measured per-vessel property, which matters
because the wall rise varies 6.8-fold WITHIN a single frame.
"""
import cv2
import numpy as np

from . import ridges
from . import skeleton as sk


def isodata(v, iters=100, tol=1e-7):
    """Ridler-Calvard threshold: the level that equals the mean of the two
    classes it creates, found by iterating from the midpoint."""
    T = 0.5 * (float(v.min()) + float(v.max()))
    for _ in range(iters):
        a, b = v[v < T], v[v >= T]
        if not a.size or not b.size:
            break
        Tn = 0.5 * (float(a.mean()) + float(b.mean()))
        if abs(Tn - T) < tol:
            break
        T = Tn
    return float(T)


def threshold(A, valid, weak=0.7):
    """Hysteresis threshold on absorbance; returns (mask, level, background).

    The strong level is Isodata (Ridler-Calvard), which has no tuned constant
    in it: it is the level that equals the mean of the two classes it creates.
    It was taken from Saleh et al., J Digital Imaging 24(4), and on our frames
    it beat the hand-tuned `median + k * spread` it replaces - 98 % of the
    centreline at absorbance >= 0.30 against 91 %, and from FEWER vessel pixels
    (16.5 % of the frame against 21.6 %).

    The weak level sits `weak` of the way from the background to the strong
    one, and a weak region is kept only where it touches a strong one. Straight
    Isodata alone finds the dark vessels well but drops to 80 % on the
    0.10-absorbance ones; at 0.7 that becomes 95 % with no loss of purity.
    """
    v = A[valid]
    med = float(np.median(v))
    hi = isodata(v)
    lo = med + (hi - med) * weak
    strong = (A >= hi) & valid
    weak = (A >= lo) & valid
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    alive = np.zeros(n, bool)
    alive[lab[strong]] = True
    alive[0] = False
    return alive[lab], hi, med


def detect(A, valid, min_radius=4.0, min_len=40.0, min_aspect=4.0, weak=0.7,
           smooth=1.0, log=None):
    """Large vessels, thickest first, as (centreline, radius, measurements).

    `radius` is measured from the absorbance profile by `measure`, not from the
    mask, and is a real half-width in pixels. `measurements` also carries the
    peak absorbance and the wall sharpness at every point along the vessel.
    """
    Af = cv2.GaussianBlur(A, (0, 0), smooth) if smooth else A
    mask, hi, med = threshold(Af, valid, weak)
    # half-width of the dark region at every pixel
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if log:
        log(f"[large] background {med:.4f}; "
            f"threshold {med + (hi - med) * weak:.3f}/{hi:.3f} (isodata); "
            f"{100 * mask.mean():.1f}% of the frame is vessel, "
            f"max half-width {dist.max():.1f} px")

    # Thin the WHOLE dark region, then judge each centreline by the width
    # underneath it. Thinning first and measuring second keeps a vessel whole
    # where it narrows, instead of breaking it wherever it dips below the
    # width cut.
    spine = sk.thin(mask)
    spine = sk.prune(spine, np.maximum(dist, 1.0), spur=8)
    out = []
    drop = 0
    for C in ridges.trace(spine, min_len=min_len):
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, A.shape[1] - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, A.shape[0] - 1)
        # The mask says WHERE a vessel is; the intensity says how wide it is.
        # The distance transform is used only as a gate, never as the reported
        # calibre, because it tracks the threshold rather than the vessel.
        if float(np.median(dist[yi, xi])) < min_radius * 0.6:
            continue
        m = measure(A, C)
        r = m["radius"]
        if not np.isfinite(r).any():
            continue
        r = np.where(np.isfinite(r), r, np.nanmedian(r))
        rad = float(np.median(r))
        if rad < min_radius:
            continue
        # A vessel is long relative to its width; a dark patch is not. Without
        # this the dark limbal corners of a frame come back as "vessels" of
        # 15-38 px radius, which is simply the half-width of a blob.
        length = float(np.hypot(*np.diff(C, axis=0).T).sum()) if len(C) > 1 else 0.0
        if length < min_aspect * rad:
            drop += 1
            continue
        out.append((C, r, m))
    out.sort(key=lambda v: -float(np.median(v[1])))
    if log:
        tot = sum(np.hypot(*np.diff(C, axis=0).T).sum() for C, *_ in out if len(C) > 1)
        log(f"[large] {len(out)} vessels, {tot:.0f} px ({drop} dropped as blobs), "
            f"radii {', '.join(f'{np.median(r):.0f}' for _, r, _ in out[:8])}...")
    return out


HWHM_TO_R = 0.83   # see measure()

_BLUR_R = np.array([1.5, 2., 3., 4., 6., 9., 13., 18., 25.])
_BLUR_S = np.array([0.6, 0.8, 1.2, 1.8, 2.6, 3.8, 5.2, 7.0, 9.5])
_BLUR_TAB = None


def _blur_table():
    """Model 10-90 % wall rise for every (radius, blur) on the grid."""
    global _BLUR_TAB
    if _BLUR_TAB is None:
        from . import fourier as Fo
        o = np.arange(-120, 120.01, 0.05)
        T = np.empty((len(_BLUR_R), len(_BLUR_S)))
        for i, r in enumerate(_BLUR_R):
            for j, sg in enumerate(_BLUR_S):
                p = Fo.chord(o, r, sg)
                pk = p.max()
                k = int(np.argmax(p))
                ends = []
                for side in (p[k::-1], p[k:]):
                    a = np.flatnonzero(side <= 0.9 * pk)
                    b = np.flatnonzero(side <= 0.1 * pk)
                    if len(a) and len(b) and b[0] > a[0]:
                        ends.append((b[0] - a[0]) * 0.05)
                T[i, j] = np.mean(ends) if ends else np.nan
        _BLUR_TAB = T
    return _BLUR_TAB


def blur(radius, rise):
    """Optical blur sigma in px, from the measured radius and wall rise.

    The raw 10-90 % rise is NOT a measure of focus: it depends on the vessel's
    calibre at least as strongly as on the blur. At sigma 0.8 px the model
    gives a rise of 2.05 px for a 2 px vessel and 10.30 px for an 18 px one, so
    on real vessels the raw rise came out correlated +0.60 with radius and was
    mostly restating calibre.

    For a FIXED radius the rise is monotone in sigma, so inverting the model at
    the measured radius separates the two. What comes back is a property of the
    optics at that point rather than of the vessel.

    DO NOT use this to decide identity, and do not use the raw rise either.
    Both were tested as a fourth agreement test across a gap, beside the
    diameter, darkness and direction that `graph.py` already checks, and both
    lose. Across 15219 true continuations and 74537 false pairs on a sharp
    crop:

        radius          AUC 0.969
        depth           AUC 0.902
        raw wall rise   AUC 0.859
        blur sigma      AUC 0.778   (inverting the model made it WORSE)

    and more to the point, at every matched true-kept rate simply tightening
    the existing radius and depth tests let through about half as many false
    pairs as adding a wall-rise test did - 17.6 % against 22.0 % at 80 % kept,
    7.8 % against 13.8 % at 60 %. Sharpness correlates +0.60 with radius and
    +0.50 with depth, so as a linking cue it is mostly restating them.

    It remains worth measuring and reporting: it says how well a given vessel
    is imaged, and it varies enormously within one frame. The untested use,
    and the interesting one, is separating two vessels AT a crossing, where
    they may be at different depths and so differently blurred - that is a
    different question from linking across a gap and is not answered here.
    """
    T = _blur_table()
    r = np.atleast_1d(np.asarray(radius, float))
    v = np.atleast_1d(np.asarray(rise, float))
    i = np.clip(np.searchsorted(_BLUR_R, r) - 1, 0, len(_BLUR_R) - 2)
    f = np.clip((r - _BLUR_R[i]) / (_BLUR_R[i + 1] - _BLUR_R[i]), 0, 1)
    out = np.full(len(r), np.nan)
    for k in range(len(r)):
        if not (np.isfinite(r[k]) and np.isfinite(v[k])):
            continue
        row = T[i[k]] * (1 - f[k]) + T[i[k] + 1] * f[k]
        out[k] = np.interp(v[k], row, _BLUR_S)
    return out if np.ndim(radius) else float(out[0])



def measure(A, C, half=45.0, step=0.25, span=25):
    """Radius, peak absorbance and wall sharpness along a centreline, read off
    the INTENSITY rather than off a binary mask.

    A radius from the distance transform of a thresholded mask is a property of
    the threshold, not of the vessel: measured on real vessels it moves 41 % as
    the threshold is swept over its usable range, and the level this module
    ships was reading 31 % high. The half-width at half the vessel's OWN peak
    cannot do that, because it is referenced to the vessel.

    Against the model, hwhm/r sits at 0.82-0.86 for every radius >= 6 px and
    every blur from 1.2 to 5.2 px - it does not depend on focus, which is what
    makes it usable as a calibre. Below about 4 px the blur dominates the
    width and the number becomes an upper bound; `radius_at_limit` says so.

    Returns dict of arrays along the centreline: radius, depth (peak
    absorbance over the local background), rise (10-90 % distance across the
    wall, small = sharply imaged), and the flag.
    """
    C = np.asarray(C, float)
    offs = np.arange(-half, half + step, step)
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    nx, ny = -t[:, 1], t[:, 0]
    px = (C[:, 0:1] + nx[:, None] * offs).astype(np.float32)
    py = (C[:, 1:2] + ny[:, None] * offs).astype(np.float32)
    P = cv2.remap(A, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    edge = max(int(round(4.0 / step)), 4)
    base = np.median(np.concatenate([P[:, :edge], P[:, -edge:]], 1), axis=1)
    P = P - base[:, None]
    mid = int(np.argmin(np.abs(offs)))
    cross = np.zeros(len(C), bool)
    rad = np.full(len(C), np.nan)
    rise = np.full(len(C), np.nan)
    peak = np.full(len(C), np.nan)
    for i, p in enumerate(P):
        # Everything is anchored on the CONTIGUOUS run through the centreline,
        # never on the whole window. A neighbouring vessel also standing above
        # half maximum would otherwise be swallowed along with the gap between
        # them, which read a 12 px vessel as 26 px where vessels run close.
        j = mid + int(np.argmax(p[max(mid - 8, 0):mid + 9])) - min(mid, 8)
        pk = float(p[j])
        if pk <= 0:
            continue
        peak[i] = pk
        # Walk out from the peak and stop at the vessel's OWN edge, which is
        # whichever comes first: the half-maximum, or the point where the
        # profile stops falling and turns back up. A turn upward is a
        # neighbouring vessel, and taking the half-maximum crossing regardless
        # walks straight through it - on a real crossing the profile fell from
        # 0.089 to 0.052, rose again to 0.066, fell to 0.046 and rose to 0.054,
        # and the half-max run spanned all of it, calling a 1.1 px vessel
        # 16.6 px wide. That is the measurement aggregating several vessels,
        # and smoothing it afterwards hides it rather than fixing it.
        ps = np.convolve(p, _gauss(1.0 / step), mode="same")
        half = []
        clean = []
        for s in (ps[j::-1], ps[j:]):
            hit = np.flatnonzero(s < pk / 2)
            turn = _turn(s, pk)
            if len(hit) and (turn is None or hit[0] <= turn):
                half.append(hit[0] * step)
                clean.append(True)
            else:
                half.append((turn if turn is not None else len(s) - 1) * step)
                clean.append(False)
        cross[i] = not all(clean)
        good = [h for h, c in zip(half, clean) if c]
        # A turn immediately beside the centreline means the profile is not
        # peaked on this vessel at all - the centreline is on the flank of
        # something else. That is unmeasurable, not a zero-radius vessel.
        if max(half) < step * 2:
            cross[i] = True
            continue
        # A contaminated side gives a distance to the neighbour, not to the
        # edge, so use the clean side and assume symmetry rather than average
        # a number that is not a half-width at all.
        rad[i] = (float(np.mean(good)) if good else float(min(half))) / HWHM_TO_R
        ends = []
        for s in (p[j::-1], p[j:]):
            a = np.flatnonzero(s <= 0.9 * pk)
            b = np.flatnonzero(s <= 0.1 * pk)
            if len(a) and len(b) and b[0] > a[0]:
                ends.append((b[0] - a[0]) * step)
        if ends:
            rise[i] = float(np.mean(ends))
    # A vessel's calibre varies smoothly along its length, so a spike over a
    # few px is another vessel crossing, not a change in calibre: where one
    # crosses, the run above half maximum spans both of them. Measured on the
    # staged network, 80 % of vessels had some point reading more than twice
    # their own median, the median worst case being 5.4x, and painting the
    # lumen from those raw values covered the junctions in blobs. A running
    # median along the centreline removes the crossing without touching a real
    # taper, which no crossing is short enough to imitate.
    # Where a vessel crosses another, its width is NOT MEASURABLE from an
    # averaged frame, and the honest thing is to say so rather than report a
    # number. Two cases, and the second is the one that matters:
    #
    #   comparable calibres - the profile dips between them and `_turn` finds
    #       the edge;
    #   a thin vessel over a thick one - there is no dip at all. The profile
    #       taken across the thin vessel runs ALONG the thick one, so it
    #       measures the thick vessel. This is how a 1.1 px vessel came back
    #       as 16.6 px, and no amount of edge-finding can fix it, because the
    #       thin vessel is not visible in that direction.
    #
    # So a point whose width is far from the vessel's own robust calibre is
    # marked unmeasurable and its radius interpolated from the neighbours that
    # were measurable. The vessel keeps a continuous width, the crossings are
    # named, and nothing is invented quietly.
    rad = _runmed(rad, span)
    rise = _runmed(rise, span)
    good = np.isfinite(rad)
    if good.any():
        base_r = float(np.median(rad[good]))
        for _ in range(3):
            keep = good & (rad < 1.8 * base_r) & (rad > base_r / 1.8)
            if keep.sum() < 3:
                break
            base_r = float(np.median(rad[keep]))
        bad = ~good | (rad > 1.8 * base_r) | cross
        if bad.any() and (~bad).sum() >= 2:
            idx = np.arange(len(rad))
            rad = np.interp(idx, idx[~bad], rad[~bad])
        cross = cross | bad
    return {"radius": rad, "depth": peak, "rise": rise, "crossing": cross,
            "blur": blur(rad, rise), "radius_at_limit": rad < 4.0}


def _gauss(sd):
    n = max(int(round(3 * sd)), 1)
    k = np.exp(-0.5 * (np.arange(-n, n + 1) / sd) ** 2)
    return k / k.sum()


def _turn(s, pk, eps=0.04):
    """Index where a profile walking away from the peak stops falling.

    Returns None if it falls all the way. `eps` is in units of the peak: a
    rise smaller than that is noise on the flank, not another vessel.
    """
    run = s[0]
    for i in range(1, len(s)):
        if s[i] < run:
            run = s[i]
        elif s[i] > run + eps * pk:
            return int(np.argmin(s[:i + 1]))
    return None


def _runmed(v, span):
    """Running median that leaves NaNs alone and copes with short vessels."""
    n = len(v)
    if span < 3 or n < 3:
        return v
    k = min(span | 1, (n // 2) * 2 - 1)
    if k < 3:
        return v
    pad = k // 2
    w = np.lib.stride_tricks.sliding_window_view(
        np.pad(v, pad, mode="edge"), k)
    with np.errstate(all="ignore"):
        out = np.nanmedian(w, axis=1)
    return np.where(np.isfinite(out), out, v)


def lumen(vessels, shape, pad=0.0, cap=0.0):
    """Label image: each vessel's own width filled in, largest drawn first so
    a thin vessel crossing a thick one does not erase it.

    Each point's radius is capped at `cap` times the vessel's own median. The
    running median in `measure` reduces the crossing spikes but does not remove
    them - after it, the worst point on a vessel still reads 3.9x its median,
    against 5.4x before - and painting a lumen from the raw values buries every
    junction under a disc the size of the crossing vessel.
    """
    lab = np.zeros(shape, np.int32)
    for i, (C, r, *_) in enumerate(vessels, 1):
        if cap:
            r = np.minimum(r, cap * float(np.nanmedian(r)))
        for (x, y), rr in zip(C, r):
            cv2.circle(lab, (int(round(x)), int(round(y))), int(round(rr + pad)), i, -1)
    return lab
