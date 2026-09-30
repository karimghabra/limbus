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


def _aligned(Th, R, p, a, radius, max_deg):
    """Whether the vessel at the circle, angle a, runs along the ray from p
    (its along-vessel eigenvector within max_deg of radial, where the image
    there looks most like a tube): an arm leaves p, a vessel seen across
    the ray does not."""
    H, W = Th.shape
    best, th = -np.inf, None
    for da in np.radians((-4.0, 0.0, 4.0)):             # the tube's centre, nearby on the circle
        x = int(round(p[0] + radius * math.cos(a + da)))
        y = int(round(p[1] + radius * math.sin(a + da)))
        if 0 <= x < W and 0 <= y < H and R[y, x] > best:
            best, th = R[y, x], Th[y, x]
    if th is None:
        return True
    d = abs((th - a + math.pi / 2) % math.pi - math.pi / 2)     # orientation: mod pi
    return d <= math.radians(max_deg)


def detect(logI, sigma=None, valid=None, z_junction=4.0, z_arm=3.0, scales=SCALES, rings=3,
           sep_deg=20.0, align_deg=35.0, min_ratio=0.25):
    """Intersections: list of dict(xy, scale, z, arms (angles, rad), kind).
    Candidates are local maxima of the junction z above z_junction; each is
    kept if at least three arms leave it: darkness peaks on circles round
    it (rings of them, 4 px + 1.5 scales out and every 6 px + a scale
    beyond) that the centre reaches without a bright gap, at least sep_deg
    apart."""
    J, Js, R, Th, Ts, Q = maps(logI, valid, scales)
    if sigma is None:
        sigma = np.full(np.shape(logI), 1.4826 * float(np.median(np.abs(np.diff(logI, axis=1)))) / math.sqrt(2))
    D = darkness(logI)
    peak = (J == ndi.maximum_filter(J, size=7)) & (J > z_junction) & (Q >= min_ratio)
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
        # an arm is a darkness peak on any of the circles that the centre
        # reaches without a bright gap (a vessel passing by is reached across
        # one); read from the outermost circle in, where arms have parted
        # most, an inner peak adding an arm only in a new direction
        best = []
        for rr, ring in sorted(zip(radii, found), key=lambda x: -x[0]):
            for a in ring:
                if all(_angdiff(a, b) >= math.radians(sep_deg) for b in best) and \
                        _aligned(Th, R, p, a, rr, align_deg) and _connected(D, sigma, p, a, rr):
                    best.append(a)
        best.sort()
        if len(best) < 3:
            continue
        out.append(dict(xy=p, scale=s, z=float(J[ys[i], xs[i]]), arms=best, kind=classify(best)))
    return out


# ---------------------------------------------------------------- scoring
_INTERSECT = ("bifurcation", "crossing", "over thick", "ending on", "complex node", "twisted",
              "weaving", "ladder", "mesh", "faint capillary", "fork")


def score_zoo(seed=0, **detect_kw):
    """Detection and arms on the zoo (zoo.ROWS, ROWS_CALIBRE, ROWS_CROSSINGS).

    An intersection is found when a detection lies within 4 px + the widest
    radius there of a marked one; a detection near none is false (in rows
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
                for p in marks:
                    rr = [float(v["r"][np.argmin(np.linalg.norm(v["xy"] - p, axis=1))]) for v in vs
                          if np.linalg.norm(v["xy"] - p, axis=1).min() < 8]
                    tol = 4.0 + (max(rr) if rr else 1.0)
                    near = [q for q, d in enumerate(mine) if np.linalg.norm(d["xy"] - p) <= tol]
                    R["marks"] += 1
                    if not near:
                        continue
                    R["found"] += 1
                    used.update(near)
                    d = mine[min(near, key=lambda q: np.linalg.norm(mine[q]["xy"] - p))]
                    ta = zoo.true_arms(vs, p)
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
