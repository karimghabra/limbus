"""Where vessels meet: crossings, bifurcations and nodes where several do
(prototype).

A dark vessel is a valley of log intensity: across it the curvature is large
and positive, along it near zero. Where vessels cross or branch, the image
curves up in every direction: both Hessian eigenvalues are positive. The
smaller eigenvalue, scale-normalised (s^2 * lambda_2) and divided by its
robust spread over the image at that scale (sensor noise and texture alike
set that spread), peaks at such places. A vessel end is blob-like too, so
each candidate is kept only if vessels leave it in at least three
directions: ridge strength is sampled on circles around it, and its
separate angular peaks are the arms, the directions in which to look for
the vessels that start there.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

SCALES = (1.0, 1.5, 2.2, 3.2, 4.5, 6.5)


def hessian_eig(L, s):
    """Scale-normalised Hessian eigenvalues (l1 >= l2) of L at scale s."""
    Lyy = ndi.gaussian_filter(L, s, order=(2, 0))
    Lxx = ndi.gaussian_filter(L, s, order=(0, 2))
    Lxy = ndi.gaussian_filter(L, s, order=(1, 1))
    half = 0.5 * (Lxx + Lyy)
    disc = np.sqrt(0.25 * (Lxx - Lyy) ** 2 + Lxy ** 2)
    return s * s * (half + disc), s * s * (half - disc)


def _robust_scale(x, valid):
    v = x[valid]
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-12


def maps(logI, valid=None, scales=SCALES):
    """(junction z, its scale, ridge z): the best over scales of lambda_2 and
    of the tube-like part of lambda_1, each in units of its robust spread."""
    L = np.asarray(logI, np.float64)
    valid = np.ones(L.shape, bool) if valid is None else valid
    J = np.full(L.shape, -np.inf)
    Js = np.zeros(L.shape)
    R = np.full(L.shape, -np.inf)
    for s in scales:
        l1, l2 = hessian_eig(L, s)
        zj = l2 / _robust_scale(l2, valid)
        tube = np.maximum(l1 - 2.0 * np.abs(l2), 0.0)       # one-sided: vessels only
        zr = tube / _robust_scale(l1, valid)
        better = zj > J
        J[better], Js[better] = zj[better], s
        R = np.maximum(R, zr)
    J[~valid] = 0.0
    R[~valid] = 0.0
    return J, Js, R


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


def _dark_arms(D, sig, p, radius, z_arm, rel=0.0):
    """Angles (rad) of the separate peaks of darkness on a circle round p
    whose prominence over the dips either side exceeds z_arm noise levels
    and rel of the peak's own darkness (a gap between two vessels is nearly
    as bright as the background; a bump of contrast along one wide vessel
    is not)."""
    H, W = D.shape
    n = int(max(90, round(2 * math.pi * radius * 2)))           # ~0.5 px apart
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = np.clip(p[0] + radius * np.cos(th), 0, W - 1)
    y = np.clip(p[1] + radius * np.sin(th), 0, H - 1)
    v = ndi.map_coordinates(D, [y, x], order=1)
    v = ndi.gaussian_filter1d(v, 1.0 * n / (2 * math.pi * radius), mode="wrap")   # ~1 px
    noise = float(np.median(ndi.map_coordinates(sig, [y, x], order=1))) / math.sqrt(2.0)
    peaks = [i for i in range(n) if v[i] > v[i - 1] and v[i] >= v[(i + 1) % n]]
    out = []
    for i in peaks:
        # prominence: the higher of the lowest points on the way to a higher peak either side
        lo = []
        for step in (1, -1):
            m, j = v[i], i
            for _ in range(n - 1):
                j = (j + step) % n
                m = min(m, v[j])
                if v[j] > v[i]:
                    break
            lo.append(m)
        if v[i] - max(lo) > max(z_arm * noise, rel * v[i]):
            out.append(float(th[i]))
    return out


def _connected(D, sig, p, a, radius, z_gap=3.0, frac=0.3):
    """Whether the straight path from p out to the circle at angle a stays
    dark: no gap brighter than the darker end by more than z_gap noise
    levels and frac of its darkness (a neighbouring vessel is reached across
    one)."""
    t = np.arange(0.0, radius + 1e-9, 0.5)
    x = p[0] + t * math.cos(a)
    y = p[1] + t * math.sin(a)
    H, W = D.shape
    if x.min() < 0 or y.min() < 0 or x.max() > W - 1 or y.max() > H - 1:
        return True
    v = ndi.map_coordinates(D, [y, x], order=1)
    end = min(v[0], v[-1])
    noise = float(np.median(ndi.map_coordinates(sig, [y, x], order=1)))
    return end - v.min() < max(z_gap * noise, frac * end)


def detect(logI, sigma=None, valid=None, z_junction=4.0, z_arm=3.0, scales=SCALES, rings=2):
    """Intersections: list of dict(xy, scale, z, arms (angles, rad), kind).
    Candidates are local maxima of the junction z above z_junction; each is
    kept if at least three arms leave it: darkness peaks on circles round
    it, chained from the innermost circle out at like angles, the innermost
    reached from the centre without a bright gap.  rings = 2 is precise;
    more rings (4) also catch shallow crossings and slow forks, whose arms
    part only farther out, at many more false detections next to wide
    vessels."""
    J, Js, R = maps(logI, valid, scales)
    if sigma is None:
        sigma = np.full(np.shape(logI), 1.4826 * float(np.median(np.abs(np.diff(logI, axis=1)))) / math.sqrt(2))
    D = darkness(logI)
    peak = (J == ndi.maximum_filter(J, size=7)) & (J > z_junction)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-J[ys, xs])
    out = []
    for i in order:
        p = np.array([xs[i], ys[i]], float)
        s = float(Js[ys[i], xs[i]])
        if any(np.linalg.norm(p - q["xy"]) < 2 * max(s, q["scale"]) + 3 for q in out):
            continue
        r_in = max(4.0, 1.5 * s + 3.0)
        radii = [r_in + k * (6.0 + s) for k in range(rings)]    # arms of a shallow crossing
        found = [_dark_arms(D, sigma, p, rr, z_arm) for rr in radii]    # or slow fork part late
        # an arm is a chain of darkness peaks from the innermost circle out,
        # a like angle from each circle to the next: it may split (a shallow
        # crossing parting) but not appear from nowhere (a vessel passing by)
        chain = [a for a in found[0] if _connected(D, sigma, p, a, radii[0])]
        best = []
        for ring in found[1:]:
            nxt = []
            for a in ring:
                if any(_angdiff(a, b) <= math.radians(30.0) for b in chain) and \
                        all(_angdiff(a, b) >= math.radians(10.0) for b in nxt):
                    nxt.append(a)
            chain = nxt
            if len(chain) > len(best):
                best = sorted(chain)
        if len(best) < 3:
            continue
        out.append(dict(xy=p, scale=s, z=float(J[ys[i], xs[i]]), arms=best, kind=classify(best)))
    return out
