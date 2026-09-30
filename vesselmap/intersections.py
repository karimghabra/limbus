"""Where vessels meet: crossings, bifurcations and nodes where several do
(prototype).

A dark vessel is a valley of log intensity: across it the curvature is large
and positive, along it near zero. Where vessels cross or branch, the image
curves up in every direction: both Hessian eigenvalues are positive. The
smaller eigenvalue, scale-normalised (s^2 * lambda_2) and divided by its
robust spread over the image at that scale (sensor noise and texture alike
set that spread), peaks at such places (and at bumps of contrast along one
vessel, and at vessel ends).  Each candidate is kept only if vessels leave
it in at least three directions, its arms: the peaks of darkness averaged
along rays from it (ray_arms), against the noise that texture as well as
the sensor puts into such averages (texture_noise), with the vessel's other
half across a crossing (continuations), no gap along the arm (continuous),
and, where two arms are close in angle, each one's vessel traced back to
the candidate (approach: a neighbour running past alongside is not an arm).
Vessels crossing at a small angle lie on top of each other for a stretch
whose two ends each look like a fork; such pairs become one crossing
(pair_ys).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

SCALES = (1.0, 1.5, 2.2, 3.2, 4.5, 6.5)


def hessian_eig(L, s, orient=False):
    """Scale-normalised Hessian eigenvalues (l1 >= l2) of L at scale s (and
    with orient the angle of l2's eigenvector: along a vessel)."""
    Lyy = ndi.gaussian_filter(L, s, order=(2, 0))
    Lxx = ndi.gaussian_filter(L, s, order=(0, 2))
    Lxy = ndi.gaussian_filter(L, s, order=(1, 1))
    half = 0.5 * (Lxx + Lyy)
    disc = np.sqrt(0.25 * (Lxx - Lyy) ** 2 + Lxy ** 2)
    if orient:
        along = 0.5 * np.arctan2(2 * Lxy, Lxx - Lyy) + 0.5 * math.pi
        return s * s * (half + disc), s * s * (half - disc), along
    return s * s * (half + disc), s * s * (half - disc)


def _robust_scale(x, valid):
    v = x[valid]
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-12


def maps(logI, valid=None, scales=SCALES):
    """(junction z, its scale, ridge z, vessel direction, tube scale,
    isotropy): the best over scales of lambda_2 and of the tube-like part of
    lambda_1, each in units of its robust spread; the along-vessel
    eigenvector's angle and the scale where the image looks most like a
    tube; and lambda_2 / lambda_1 at the junction scale (near 1 where
    vessels meet, small on a bump of contrast along one vessel)."""
    L = np.asarray(logI, np.float64)
    valid = np.ones(L.shape, bool) if valid is None else valid
    J = np.full(L.shape, -np.inf)
    Js = np.zeros(L.shape)
    R = np.full(L.shape, -np.inf)
    Th = np.zeros(L.shape)
    Ts = np.zeros(L.shape)
    Q = np.zeros(L.shape)
    for s in scales:
        l1, l2, along = hessian_eig(L, s, orient=True)
        zj = l2 / _robust_scale(l2, valid)
        tube = np.maximum(l1 - 2.0 * np.abs(l2), 0.0)       # one-sided: vessels only
        zr = tube / _robust_scale(l1, valid)
        better = zj > J
        J[better], Js[better] = zj[better], s
        Q[better] = (l2 / np.maximum(l1, 1e-12))[better]
        tb = zr > R
        R[tb], Th[tb], Ts[tb] = zr[tb], along[tb], s
    J[~valid] = 0.0
    R[~valid] = 0.0
    return J, Js, R, Th, Ts, Q


def _angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def classify(arms, straight_deg=25.0):
    """bifurcation (3 arms), crossing (4 arms forming two straight pairs),
    node (anything else with 3 or more)."""
    n = len(arms)
    if n == 3:
        return "bifurcation"
    if n == 4:
        a = np.array(arms)
        opp = lambda i, j: abs(abs((a[i] - a[j] + math.pi) % (2 * math.pi) - math.pi) - math.pi)
        tol = math.radians(straight_deg)
        if (opp(0, 2) < tol and opp(1, 3) < tol):
            return "crossing"
    return "node"


def darkness(logI, width=31, smooth=4.0):
    """How much darker than its surroundings each pixel is: the local upper
    envelope of log intensity (a grey closing wider than any vessel,
    smoothed) less the image.  It peaks at a vessel's centre however wide
    the vessel is, unlike curvature, which peaks near the edges of a wide
    one."""
    L = np.asarray(logI, np.float64)
    B = ndi.gaussian_filter(ndi.grey_closing(L, size=(width, width)), smooth)
    return B - L


def ray_profile(D, p, r0, r1, step_deg=1.0):
    """Darkness along rays from p, r0 to r1 px out, per direction: (angles,
    mean over each ray, mean over its near half).  A vessel leaving p is a
    peak in its direction (it darkens the whole ray); a vessel merely
    passing near p is not."""
    th = np.radians(np.arange(0.0, 360.0, step_deg))
    rr = np.arange(r0, r1 + 1e-9, 0.5)
    H, W = D.shape
    x = np.clip(p[0] + np.outer(np.cos(th), rr), 0, W - 1)
    y = np.clip(p[1] + np.outer(np.sin(th), rr), 0, H - 1)
    v = ndi.map_coordinates(D, [y.ravel(), x.ravel()], order=1).reshape(len(th), -1)
    return th, v.mean(1), v[:, :v.shape[1] // 2].mean(1)


def _prominence(v, i):
    """How far v[i] stands above the lower of the highest dips on its way to
    a higher value either side (circular)."""
    n = len(v)
    lo = []
    for step in (1, -1):
        m, j = v[i], i
        for _ in range(n - 1):
            j = (j + step) % n
            m = min(m, v[j])
            if v[j] > v[i]:
                break
        lo.append(m)
    return v[i] - max(lo)


def _refine(v, i, th):
    n = len(v)
    a, b, c = v[i - 1], v[i], v[(i + 1) % n]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if den < 0 else 0.0
    return float((th[i] + off * (th[1] - th[0])) % (2 * math.pi))


def _tail_spread(e):
    """Spread of a ray profile from its lower tail (30th less 5th
    percentile, as a Gaussian's standard deviation): the directions
    without a vessel."""
    q5, q30 = np.percentile(e, (5.0, 30.0))
    return float(q30 - q5) / 1.12


def texture_noise(D, r0=3.0, reach=20.0, step=16, win=9):
    """How much the darkness along a ray varies from direction to direction
    where no vessel is (texture as well as sensor noise): the lower-tail
    spread of ray profiles at points every step px, its median over win x
    win of them (most are away from vessels), per pixel."""
    H, W = D.shape
    ys = np.arange(step // 2, H, step)
    xs = np.arange(step // 2, W, step)
    g = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            th, v, _ = ray_profile(D, (float(x), float(y)), r0, r0 + reach, step_deg=3.0)
            g[i, j] = _tail_spread(v - np.percentile(v, 20))
    g = ndi.median_filter(g, size=win, mode="nearest")
    return ndi.zoom(g, (H / len(ys), W / len(xs)), order=1, mode="nearest")[:H, :W]


def _profile_noise(sig, p, length, tex=None):
    """The noise level of a ray profile at p: sensor sigma over the root of
    the ray length, or the texture's (texture_noise) where larger."""
    H, W = sig.shape
    iy, ix = int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))
    sensor = float(sig[iy, ix]) / math.sqrt(max(length, 1.0))
    return sensor if tex is None else max(sensor, float(tex[iy, ix]))


def ray_arms(D, sig, p, r0, r1, z_arm=4.0, rel=0.3, sep_deg=12.0, near=0.35, tex=None):
    """The arms at p from the ray profile (darkness above the background's
    along each direction): its peaks that stand above the dips either side
    by z_arm noise levels (_profile_noise: the sensor's, or the texture's
    where larger) and by rel of their height (not the shoulder of a wide
    arm), with the ray's
    near half dark too (at least near of it: not another junction met far
    out), at least sep_deg apart, strongest first; angles refined by a
    parabola."""
    th, v, vn = ray_profile(D, p, r0, r1)
    v = ndi.gaussian_filter1d(v, 1.0, mode="wrap")
    vn = ndi.gaussian_filter1d(vn, 1.0, mode="wrap")
    n = len(v)
    e = v - float(np.percentile(v, 20))                # darkness above the background's
    en = vn - float(np.percentile(vn, 20))
    noise = _profile_noise(sig, p, r1 - r0, tex)
    sep = math.radians(sep_deg)
    cand = []
    for i in range(n):
        if e[i] > e[i - 1] and e[i] >= e[(i + 1) % n]:
            prom = _prominence(e, i)
            if prom > max(z_arm * noise, rel * e[i]) and en[i] >= near * e[i]:
                cand.append((e[i], i))
    arms = []
    for h, i in sorted(cand, reverse=True):
        a = _refine(e, i, th)
        if all(_angdiff(a, b) >= sep for b in arms):
            arms.append(a)
    return sorted(arms)


def _ridge_point(D, p, a, r, w):
    """The darkest point across the ray at angle a, r px out, within w px
    of it."""
    H, W = D.shape
    c, sn = math.cos(a), math.sin(a)
    u = np.arange(-w, w + 1e-9, 0.5)
    xs, ys = p[0] + r * c - u * sn, p[1] + r * sn + u * c
    v = ndi.map_coordinates(D, [np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)], order=1)
    k = int(np.argmax(v))
    return np.array([xs[k], ys[k]])


def approach(D, p, a, r1, others=(), Th=None, R=None, w_max=4.0, lost=0.25, blend=0.5,
             r_tube=2.0, max_dev=45.0, path=None):
    """How close to p the vessel along arm a comes: from the darkest point
    across the ray r1 px out (within half the angle to the nearest other
    arm) its ridge is followed inward a px at a time, steered by the
    vessel's own direction (Th, where the image looks like a tube: R above
    r_tube, and within max_dev of the way it was going) and re-centred
    across it within a px either side, until it has passed p (its distance
    grown 2 px past the least), fades below lost of where it started, or
    has gone twice the way; the least distance.  A vessel leaving p comes
    to it however it curves; a neighbour seen across the gap next to the
    vessel through p runs past at their separation.  The points followed
    are appended to path, if given."""
    H, W = D.shape

    def at(x, y):
        return ndi.map_coordinates(D, [np.clip(np.atleast_1d(y), 0, H - 1),
                                       np.clip(np.atleast_1d(x), 0, W - 1)], order=1)

    p = np.asarray(p, float)
    half = min([_angdiff(a, b) / 2 for b in others] + [math.radians(40.0)])
    q = _ridge_point(D, p, a, r1, float(np.clip(r1 * math.tan(half), 1.0, w_max)))
    v0 = float(at(*q)[0])
    if v0 <= 0:
        return np.inf
    d = (p - q) / (np.linalg.norm(p - q) + 1e-9)
    if path is not None:
        path.append(q)
    u = np.arange(-1.0, 1.0 + 1e-9, 0.25)
    cos_dev = math.cos(math.radians(max_dev))
    best = np.linalg.norm(q - p)
    for _ in range(int(2 * r1) + 4):
        if Th is not None:
            iy, ix = int(np.clip(round(q[1]), 0, H - 1)), int(np.clip(round(q[0]), 0, W - 1))
            if R is None or R[iy, ix] > r_tube:
                t = np.array([math.cos(Th[iy, ix]), math.sin(Th[iy, ix])])
                c = float(t @ d)
                t = t if c >= 0 else -t
                if abs(c) >= cos_dev:
                    d = (1 - blend) * d + blend * t
                    d /= np.linalg.norm(d) + 1e-9
        n = np.array([-d[1], d[0]])
        c = q + d
        xs, ys = c[0] + u * n[0], c[1] + u * n[1]
        vals = at(xs, ys)
        k = int(np.argmax(vals))
        if vals[k] < lost * v0:
            break
        q = np.array([xs[k], ys[k]])
        if path is not None:
            path.append(q)
        dist = np.linalg.norm(q - p)
        best = min(best, dist)
        if dist > best + 2.0:
            break
    return float(best)


def arms_multi(D, sig, p, r0, reach, z_arm=4.0, sep_deg=12.0, tex=None, Ds=None, Th=None, R=None,
               pair_deg=50.0, meet_px=3.0):
    """ray_arms along the reach and the continuations through p of arms
    found (see continuations), each kept if dark along the ray without a
    gap (see continuous) and, where another arm is within pair_deg of it,
    if its vessel traced inward comes within meet_px of p (see approach:
    two vessels side by side past p look like arms parting from it)."""
    arms = ray_arms(D, sig, p, r0, r0 + reach, z_arm, sep_deg=sep_deg, tex=tex)
    arms = arms + continuations(D, sig, p, r0, r0 + reach, arms, sep_deg=sep_deg, tex=tex)
    arms = sorted(a for a in arms if continuous(D, p, a, r0, r0 + reach))
    if pair_deg and Ds is not None and len(arms) > 1:
        keep = []
        for a in arms:
            others = [b for b in arms if b != a]
            if min(_angdiff(a, b) for b in others) > math.radians(pair_deg) or \
                    approach(Ds, p, a, r0 + reach, others, Th, R) <= meet_px:
                keep.append(a)
        arms = keep
    return arms


def continuous(D, p, a, r0, r1, low=0.25, high=0.6, slack_px=2.0, win_px=3.0):
    """Whether direction a is dark without a gap: along the ray (the darkest
    within slack_px of arc at each radius: an arm may curve), darkness above
    the ray profile's background never dips (over win_px) below low of its
    median and then recovers above high of it.  An arm may fade or curve
    away; a ray from one crossing of a mesh to the next dips between them."""
    th, v, _ = ray_profile(D, p, r0, r1)
    base = float(np.percentile(v, 20))
    H, W = D.shape
    rr = np.arange(r0, r1 + 1e-9, 0.5)
    best = np.full(len(rr), -np.inf)
    for off in (-1.0, -0.5, 0.0, 0.5, 1.0):
        da = off * slack_px / np.maximum(rr, 1.0)
        x = np.clip(p[0] + rr * np.cos(a + da), 0, W - 1)
        y = np.clip(p[1] + rr * np.sin(a + da), 0, H - 1)
        best = np.maximum(best, ndi.map_coordinates(D, [y, x], order=1))
    g = best - base
    med = float(np.median(g))
    if med <= 0:
        return False
    k = max(1, int(round(win_px / 0.5)))
    wmin = ndi.minimum_filter1d(g, k, mode="nearest")
    dip = np.flatnonzero(wmin < low * med)
    if not len(dip):
        return True
    return not np.any(wmin[dip[0]:] > high * med)          # a dip, but no recovery after it


def continuations(D, sig, p, r0, r1, arms, window_deg=12.0, frac=0.4, z=3.0, near=0.35,
                  sep_deg=12.0, tex=None, skip_deg=12.0):
    """Arms opposite found ones that the ray profile shows too weakly on
    its own: a vessel crossing another goes on through it, so where nothing
    was found opposite an arm, the darkest direction within window_deg of
    the opposite is an arm if its darkness above what a wide arm next to it
    gives there (that arm's darkness mirrored about it) is at least frac of
    the arm's own, z noise levels, and dark near p too."""
    th, v, vn = ray_profile(D, p, r0, r1)
    v = ndi.gaussian_filter1d(v, 1.0, mode="wrap")
    vn = ndi.gaussian_filter1d(vn, 1.0, mode="wrap")
    n = len(v)
    step = th[1] - th[0]
    e = v - float(np.percentile(v, 20))
    en = vn - float(np.percentile(vn, 20))
    noise = _profile_noise(sig, p, r1 - r0, tex)
    idx = lambda a: int(round((a % (2 * math.pi)) / step)) % n
    out = []
    for a in arms:
        o = a + math.pi
        if any(_angdiff(o, b) <= math.radians(skip_deg) for b in arms + out):
            continue
        w = int(round(math.radians(window_deg) / step))
        cand = [(idx(o) + k) % n for k in range(-w, w + 1)]
        i = max(cand, key=lambda i: e[i])
        b = float(th[i])
        x = e[i]
        c = [q for q in arms if _angdiff(b, q) <= math.radians(45.0)]
        if c:                                   # on the shoulder of a wide arm: mirror it
            q = min(c, key=lambda q: _angdiff(b, q))
            x = e[i] - e[idx(2 * q - b)]
        if x >= max(frac * e[idx(a)], z * noise) and en[i] >= near * e[i] and \
                all(_angdiff(b, q) >= math.radians(sep_deg) for q in arms + out):
            out.append(b % (2 * math.pi))
    return out


def detect(logI, sigma=None, valid=None, z_junction=4.0, z_arm=4.0, scales=SCALES,
           min_ratio=0.0, reach=20.0, texture=True, pair_deg=50.0, pair_len=60.0):
    """Intersections: list of dict(xy, scale, z, arms (angles, rad), kind),
    with parts and span where it is a shallow crossing (pair_ys).
    Candidates are local maxima of the junction z above z_junction where
    lambda_2 >= min_ratio lambda_1 (0: any where both are positive); each
    is kept if at least three arms leave it (arms_multi: from just outside
    its core out to reach px beyond; close pairs of arms checked with
    approach unless pair_deg is 0), then Y-shaped pairs up to pair_len px
    apart joined (none if 0).  texture: arms must stand out of the texture
    as well as the sensor noise."""
    J, Js, R, Th, Ts, Q = maps(logI, valid, scales)
    if sigma is None:
        sigma = np.full(np.shape(logI), 1.4826 * float(np.median(np.abs(np.diff(logI, axis=1)))) / math.sqrt(2))
    D = darkness(logI)
    tex = texture_noise(D, reach=reach) if texture else None
    Ds = ndi.gaussian_filter(D, 1.0)
    peak = (J == ndi.maximum_filter(J, size=7)) & (J > z_junction) & (Q >= min_ratio)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-J[ys, xs])
    out = []
    for i in order:
        p = np.array([xs[i], ys[i]], float)
        s = float(Js[ys[i], xs[i]])
        if any(np.linalg.norm(p - q["xy"]) < 2 * max(s, q["scale"]) + 3 for q in out):
            continue
        r0 = max(3.0, 1.0 * s + 2.0)                   # outside the junction's own core
        best = arms_multi(D, sigma, p, r0, reach, z_arm, tex=tex, Ds=Ds, Th=Th, R=R, pair_deg=pair_deg)
        if len(best) < 3:
            continue
        out.append(dict(xy=p, scale=s, z=float(J[ys[i], xs[i]]), arms=best, kind=classify(best)))
    if pair_len:
        out = pair_ys(out, D, reach, max_len=pair_len)
    return out


def _mean_along(D, a, b, lo=0.0, hi=1.0, step=0.5):
    """Mean of D along the segment a-b, from lo to hi of the way."""
    H, W = D.shape
    L = float(np.linalg.norm(b - a))
    t = np.arange(lo * L, hi * L + 1e-9, step) / max(L, 1e-9)
    x = np.clip(a[0] + t * (b[0] - a[0]), 0, W - 1)
    y = np.clip(a[1] + t * (b[1] - a[1]), 0, H - 1)
    return float(ndi.map_coordinates(D, [y, x], order=1).mean())


def pair_ys(dets, D, reach=20.0, max_len=60.0, trunk_deg=20.0, v_deg=50.0, darker=1.25):
    """Shallow crossings.  Two vessels crossing at a small angle lie on top
    of each other over a stretch, and each end of it looks like a
    bifurcation whose trunk points at the other end: a Y and a mirrored Y.
    Pairs of three-armed junctions up to max_len apart, each with an arm
    within trunk_deg of the way to the other, its other two arms within
    v_deg of straight on beyond it (a narrow V), and the stretch between
    them darker than their outer arms by darker (two vessels, not one),
    become one crossing at the middle of the stretch with the four outer
    arms (parts: the two ends; any other junction on the stretch is part of
    it too).  Closest pairs first."""
    Y = [i for i, d in enumerate(dets) if len(d["arms"]) == 3]
    cand = []
    for ii, i in enumerate(Y):
        for j in Y[ii + 1:]:
            a, b = dets[i], dets[j]
            v = b["xy"] - a["xy"]
            L = float(np.linalg.norm(v))
            if L > max_len or L < 1.0:
                continue
            ang = math.atan2(v[1], v[0])
            outer = []
            for d, t in ((a, ang), (b, ang + math.pi)):
                k = min(range(3), key=lambda k: _angdiff(d["arms"][k], t))
                if _angdiff(d["arms"][k], t) > math.radians(trunk_deg):
                    break
                o = [x for n, x in enumerate(d["arms"]) if n != k]
                if any(_angdiff(x, t + math.pi) > math.radians(v_deg) for x in o):
                    break
                outer.append(o)
            if len(outer) < 2:
                continue
            seg = _mean_along(D, a["xy"], b["xy"], 0.2, 0.8)
            arm = np.mean([_mean_along(D, d["xy"] + (d["scale"] + 2) * np.array([math.cos(x), math.sin(x)]),
                                       d["xy"] + (d["scale"] + 2 + reach) * np.array([math.cos(x), math.sin(x)]))
                           for d, o in ((a, outer[0]), (b, outer[1])) for x in o])
            if seg < darker * arm:
                continue
            cand.append((L, i, j, outer[0] + outer[1]))
    used, new = set(), []
    for L, i, j, arms in sorted(cand, key=lambda c: c[0]):
        if i in used or j in used:
            continue
        a, b = dets[i]["xy"], dets[j]["xy"]
        u = (b - a) / L
        on = [k for k, d in enumerate(dets) if k not in used and k not in (i, j) and
              -5.0 <= float((d["xy"] - a) @ u) <= L + 5.0 and
              abs(float((d["xy"] - a) @ np.array([-u[1], u[0]]))) <= 2 * dets[k]["scale"] + 3]
        used.update([i, j] + on)
        arms = sorted(x % (2 * math.pi) for x in arms)
        new.append(dict(xy=(a + b) / 2, scale=max(dets[i]["scale"], dets[j]["scale"]),
                        z=max(dets[i]["z"], dets[j]["z"]), arms=arms, kind=classify(arms),
                        parts=[a, b] + [dets[k]["xy"] for k in on], span=L))
    return [d for k, d in enumerate(dets) if k not in used] + new


# ---------------------------------------------------------------- scoring
_INTERSECT = ("bifurcation", "crossing", "over thick", "ending on", "complex node", "twisted",
              "weaving", "ladder", "mesh", "faint capillary", "fork")


def _radius_at(vessels, p, within=8.0):
    """The widest radius of the vessels passing within `within` px of p."""
    rr = [float(v["r"][np.argmin(np.linalg.norm(v["xy"] - p, axis=1))]) for v in vessels
          if np.linalg.norm(v["xy"] - p, axis=1).min() < within]
    return max(rr) if rr else 1.0


def _groups(marks, vessels, gap=2.0):
    """Marked intersections too close to tell apart (nearer each other than
    the widest radii there plus gap px: their lumens overlap, or nearly)
    grouped (single linkage); each group is one place where vessels meet."""
    n = len(marks)
    lab = list(range(n))
    r = [_radius_at(vessels, p) for p in marks]
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(marks[i] - marks[j]) < r[i] + r[j] + gap:
                a, b = lab[i], lab[j]
                lab = [a if x == b else x for x in lab]
    return [[marks[i] for i in range(n) if lab[i] == g] for g in sorted(set(lab))]


def score_zoo(seed=0, **detect_kw):
    """Detection and arms on the zoo (zoo.ROWS, ROWS_CALIBRE, ROWS_CROSSINGS).

    Marked intersections too close to tell apart are one (_groups).  An
    intersection is found when a detection lies within 4 px + the widest
    radius there of a mark of it; a detection near none is false (in rows
    of parallel pairs, kissing pairs and hairpins every detection is). At a
    found intersection its arms are matched to the true arms (zoo.true_arms:
    one per direction a vessel leaves it) within 20 degrees: arm recall and
    precision, whether all arms are right, whether the kind is (from the
    true arms), and the angular error.  Returns dict(total=..., rows=[...])."""
    from .image import prepare
    from . import zoo
    rows_out = []
    T = dict(marks=0, found=0, false=0, false_other=0, arms_true=0, arms_det=0, arms_ok=0,
             all_right=0, kind_right=0, n=0, err=[])
    for k, rows in enumerate((zoo.ROWS, zoo.ROWS_CALIBRE, zoo.ROWS_CROSSINGS)):
        I, V, tiles = zoo.zoo_sheet(seed + k, rows=rows)
        P = prepare(I)
        dets = detect(P.logI, P.sigma, P.valid, **detect_kw)
        for ri, (name, _, _) in enumerate(rows):
            is_x = any(w in name for w in _INTERSECT) and "parallel pair" not in name
            R = dict(name=name, marks=0, found=0, false=0, arms_true=0, arms_det=0, arms_ok=0,
                     all_right=0, kind_right=0, n=0, err=[])
            for t in (t for t in tiles if t["row"] == ri):
                x0, y0, x1, y1 = t["box"]
                vs = [V[i] for i in t["vessels"]]
                marks = t["ambiguous"][:1] if "fork" in name else (t["ambiguous"] if is_x else [])
                mine = [d for d in dets if x0 <= d["xy"][0] <= x1 and y0 <= d["xy"][1] <= y1]
                used = set()
                for group in _groups(marks, vs):
                    c = np.mean(group, 0)
                    near = [q for q, d in enumerate(mine)
                            if any(np.linalg.norm(d["xy"] - p) <= 4.0 + _radius_at(vs, p) for p in group)]
                    R["marks"] += 1
                    if not near:
                        continue
                    R["found"] += 1
                    used.update(near)
                    d = mine[min(near, key=lambda q: np.linalg.norm(mine[q]["xy"] - c))]
                    ta = zoo.true_arms(vs, c, near=4.0 + max(np.linalg.norm(p - c) for p in group))
                    m, err = zoo.match_arms(d["arms"], ta)
                    R["arms_true"] += len(ta)
                    R["arms_det"] += len(d["arms"])
                    R["arms_ok"] += m
                    R["err"] += err
                    R["n"] += 1
                    R["all_right"] += m == len(ta) == len(d["arms"])
                    R["kind_right"] += d["kind"] == classify(sorted(ta))
                R["false"] += len(mine) - len(used)
            rows_out.append(R)
            for key in ("marks", "found", "arms_true", "arms_det", "arms_ok", "all_right",
                        "kind_right", "n"):
                T[key] += R[key]
            T["err"] += R["err"]
            T["false" if is_x else "false_other"] += R["false"]
    tot = dict(recall=T["found"] / max(T["marks"], 1), found=T["found"], marks=T["marks"],
               false=T["false"] + T["false_other"],
               arm_recall=T["arms_ok"] / max(T["arms_true"], 1),
               arm_precision=T["arms_ok"] / max(T["arms_det"], 1),
               all_arms_right=T["all_right"] / max(T["n"], 1),
               kind_right=T["kind_right"] / max(T["n"], 1),
               angle_error_deg=float(np.mean(T["err"])) if T["err"] else 0.0)
    return dict(total=tot, rows=rows_out)
