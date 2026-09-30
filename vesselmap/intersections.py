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


def ray_arms(D, sig, p, r0, r1, z_arm=4.0, rel=0.3, sep_deg=12.0, near=0.35):
    """The arms at p from the ray profile (darkness above the background's
    along each direction): its peaks that stand above the dips either side
    by z_arm noise levels (sensor sigma over the root of the ray length) and
    by rel of their height (not the shoulder of a wide arm), with the ray's
    near half dark too (at least near of it: not another junction met far
    out), at least sep_deg apart, strongest first; angles refined by a
    parabola."""
    th, v, vn = ray_profile(D, p, r0, r1)
    v = ndi.gaussian_filter1d(v, 1.0, mode="wrap")
    vn = ndi.gaussian_filter1d(vn, 1.0, mode="wrap")
    n = len(v)
    H, W = sig.shape
    noise = float(sig[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]) / math.sqrt(max(r1 - r0, 1.0))
    e = v - float(np.percentile(v, 20))                # darkness above the background's
    en = vn - float(np.percentile(vn, 20))
    sep = math.radians(sep_deg)
    cand = []
    for i in range(n):
        if e[i] > e[i - 1] and e[i] >= e[(i + 1) % n]:
            prom = _prominence(e, i)
            if prom > max(z_arm * noise, rel * e[i]) and en[i] >= near * e[i]:
                cand.append((e[i], i))
    arms, idx = [], []
    for h, i in sorted(cand, reverse=True):
        a = _refine(e, i, th)
        if all(_angdiff(a, b) >= sep for b in arms):
            arms.append(a)
            idx.append(i)
    return sorted(arms)


def arms_multi(D, sig, p, r0, reach, z_arm=4.0, sep_deg=12.0):
    """ray_arms along the reach and the continuations through p of arms
    found (see continuations), each kept if dark along the ray without a
    gap (see continuous)."""
    arms = ray_arms(D, sig, p, r0, r0 + reach, z_arm, sep_deg=sep_deg)
    arms = arms + continuations(D, sig, p, r0, r0 + reach, arms, sep_deg=sep_deg)
    return sorted(a for a in arms if continuous(D, p, a, r0, r0 + reach))


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
                  sep_deg=12.0):
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
    H, W = sig.shape
    noise = float(sig[int(np.clip(p[1], 0, H - 1)), int(np.clip(p[0], 0, W - 1))]) / math.sqrt(max(r1 - r0, 1.0))
    e = v - float(np.percentile(v, 20))
    en = vn - float(np.percentile(vn, 20))
    idx = lambda a: int(round((a % (2 * math.pi)) / step)) % n
    out = []
    for a in arms:
        o = a + math.pi
        if any(_angdiff(o, b) <= math.radians(25.0) for b in arms + out):
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
           min_ratio=0.25, reach=20.0):
    """Intersections: list of dict(xy, scale, z, arms (angles, rad), kind).
    Candidates are local maxima of the junction z above z_junction where
    lambda_2 >= min_ratio lambda_1; each is kept if at least three arms
    leave it: peaks of darkness averaged along rays from just outside its
    core out to reach px beyond (ray_arms)."""
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
        r0 = max(3.0, 1.0 * s + 2.0)                   # outside the junction's own core
        best = arms_multi(D, sigma, p, r0, reach, z_arm)
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
