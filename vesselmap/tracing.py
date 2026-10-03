"""Vessels traced in position x orientation, and where their traces meet
(prototype).

An orientation score lifts the image to (orientation, y, x): at each angle,
how much a dark line runs through each pixel along it (a second derivative
across it times a Gaussian along it, the best of three widths). Two vessels
crossing are apart in this space however they overlap in the image, so a
tracer that follows the score at its own heading, preferring to go on
turning as it has been turning, goes straight through a crossing instead of
turning onto the other vessel.

Every line in the image is traced (the strongest point not yet traced at its
own orientation, again and again), pieces of one vessel are joined end to
end, and junctions are read off where traces meet:

* two traces crossing: one leaves the other on the side opposite to the one
  it came from, and the meeting point is as dark as both lines together
  (two vessels on top of each other; where they only touch, it is about as
  dark as one);
* a trace ending on another: a T or a fork;
* two traces ending on the same one from opposite sides, in line: one
  vessel passing through another at a small angle, or a thin one through a
  thick or blurred one, which no trace follows across;
* touching and parting on the same side: a kiss, not a junction.

Any junction, found here or by intersections.detect, is kept only where at
least three of its arms are traced vessels that reach it (arm_support): a
streak of texture is not traced, and a neighbour running alongside passes
by.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

NK = 36                                   # orientations over pi (5 degrees)
DK = math.pi / NK
NH = 2 * NK                               # headings over 2 pi
SCALES = ((1.0, 6.0), (2.0, 7.0), (3.5, 9.0))   # (across, along) sigmas, px


def _kernel(theta, s_c, s_a):
    half = int(math.ceil(3 * max(s_a, s_c)))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    u = x * math.cos(theta) + y * math.sin(theta)            # along
    v = -x * math.sin(theta) + y * math.cos(theta)           # across
    g = np.exp(-0.5 * (u / s_a) ** 2 - 0.5 * (v / s_c) ** 2)
    k = (v ** 2 / s_c ** 4 - 1 / s_c ** 2) * g               # d2/dv2: a dark line is > 0
    k -= k.mean()
    return (k * s_c ** 2 / g.sum()).astype(np.float32)


def orientation_score(logI, valid=None, scales=SCALES):
    """U[k, y, x]: how much a dark line runs through (x, y) at angle k * DK,
    the best over (across, along) scales, each in units of its robust
    spread over the image; S[k, y, x] the index of that scale (a width)."""
    L = np.asarray(logI, np.float32)
    H, W = L.shape
    valid = np.ones((H, W), bool) if valid is None else valid
    U = np.full((NK, H, W), -np.inf, np.float32)
    S = np.zeros((NK, H, W), np.uint8)
    for j, (s_c, s_a) in enumerate(scales):
        r = np.array([cv2.filter2D(L, -1, _kernel(k * DK, s_c, s_a), borderType=cv2.BORDER_REFLECT)
                      for k in range(NK)])
        v = r[:, valid][:, ::7]
        r /= 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-9
        better = r > U
        U[better] = r[better]
        S[better] = j
    U[:, ~valid] = 0.0
    return U, S


# ---------------------------------------------------------------- tracing
def _sample(U, k, x, y):
    H, W = U.shape[1:]
    return ndi.map_coordinates(U, [np.asarray(k, float), np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)],
                               order=1, mode="nearest")


def trace_one(U, q, h, valid, z_stop=2.5, turn=3, max_steps=400, miss=3, lam=0.25, lat=1.0,
              visited=None, join_len=10):
    """Follow a dark line from q along heading bin h (NH bins over 2 pi) a
    px at a time: the next point is the best of turning up to `turn` bins
    and moving up to `lat` px sideways, scored by the orientation score at
    the new heading less lam per bin away from the turn made last (a curve
    goes on curving; a line crossing it does not pull it off).  Stops after
    `miss` steps below z_stop, off the valid region, or after join_len
    steps on a line already traced at this heading (visited), cut back to
    where that began.  Returns (points, headings, values)."""
    H, W = U.shape[1:]
    pts, hs = [np.asarray(q, float)], [int(h)]
    vals = [float(_sample(U, [h % NK], [q[0]], [q[1]])[0])]
    dh = np.arange(-turn, turn + 1)
    us = np.arange(-lat, lat + 1e-9, 0.5)
    last_turn, misses, on_old, joined = 0, 0, 0, None
    for _ in range(max_steps):
        hc = h + dh
        ang = hc * DK
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        P = pts[-1][None, None, :] + d[:, None, :] + us[None, :, None] * n[:, None, :]
        K = np.repeat((hc % NK)[:, None], len(us), 1)
        v = _sample(U, K.ravel(), P[..., 0].ravel(), P[..., 1].ravel()).reshape(K.shape)
        score = v - lam * np.abs(dh - last_turn)[:, None] - 0.2 * np.abs(us)[None, :]
        i, j = np.unravel_index(np.argmax(score), score.shape)
        qn = P[i, j]
        x, y = int(round(qn[0])), int(round(qn[1]))
        if not (0 <= x < W and 0 <= y < H) or not valid[y, x]:
            break
        misses = misses + 1 if v[i, j] < z_stop else 0
        last_turn = int(dh[i])
        h = int(hc[i])
        pts.append(qn)
        hs.append(h)
        vals.append(float(v[i, j]))
        if misses >= miss:
            break
        if visited is not None:
            on_old = on_old + 1 if visited[h % NK, y, x] else 0
            if on_old >= join_len:
                joined = len(pts) - join_len
                break
    k = len(vals)
    while k > 1 and vals[k - 1] < z_stop:              # not the trailing steps below threshold
        k -= 1
    if joined is not None:
        k = min(k, joined + 1)
    return np.array(pts[:k]), np.array(hs[:k]), np.array(vals[:k])


def _mark(visited, pts, hs, r):
    """Mark a trace as traced r px (one per point) either side, at its
    orientation and the ones next to it."""
    for q, h, ri in zip(pts, hs, r):
        x, y = int(round(q[0])), int(round(q[1]))
        ri = int(math.ceil(ri))
        for kk in ((h - 1) % NK, h % NK, (h + 1) % NK):
            visited[kk, max(y - ri, 0):y + ri + 1, max(x - ri, 0):x + ri + 1] = True


def trace_all(U, valid, S, scales=SCALES, z_seed=8.0, min_len=20.0, min_mean=10.0, max_seeds=20000, **kw):
    """Traces of every line in the image: again and again, the strongest
    point not yet traced at its own orientation (a line crossing a traced
    one is still untraced at its orientation) is traced both ways and
    marked across its width, until none is left above z_seed.  Traces
    shorter than min_len px or weaker than min_mean on average (texture)
    are dropped; pieces of one vessel are joined (join_ends).  Each trace
    is dict(xy, h (heading bins), v (score), w (width: the across sigma
    that answered))."""
    NK_, H, W = U.shape
    widths = np.array([s[0] for s in scales])
    visited = np.zeros(U.shape, bool)
    Rv = np.where(valid, U.max(0), -np.inf)
    traces = []
    for _ in range(max_seeds):
        y, x = divmod(int(np.argmax(Rv)), W)
        if Rv[y, x] < z_seed:
            break
        k = int(np.argmax(np.where(visited[:, y, x], -np.inf, U[:, y, x])))
        (pa, ha, va), (pb, hb, vb) = (trace_one(U, (float(x), float(y)), h, valid, visited=visited, **kw)
                                      for h in (k, k + NK))
        pts = np.vstack([pb[::-1], pa[1:]])
        hs = np.r_[(hb[::-1] + NK) % NH, ha[1:]]
        vals = np.r_[vb[::-1], va[1:]]
        iy = np.clip(np.rint(pts[:, 1]).astype(int), 0, H - 1)
        ix = np.clip(np.rint(pts[:, 0]).astype(int), 0, W - 1)
        w = widths[S[hs % NK, iy, ix]]
        _mark(visited, pts, hs % NK, 1.0 + 1.2 * w)
        w0 = int(math.ceil(1.0 + 1.2 * widths[S[k, y, x]]))
        for kk in range(k - 2, k + 3):                 # the seed itself, even if nothing came of it
            visited[kk % NK, max(y - w0, 0):y + w0 + 1, max(x - w0, 0):x + w0 + 1] = True
        x0, y0 = int(max(min(pts[:, 0].min(), x) - 4, 0)), int(max(min(pts[:, 1].min(), y) - 4, 0))
        x1, y1 = int(min(max(pts[:, 0].max(), x) + 5, W)), int(min(max(pts[:, 1].max(), y) + 5, H))
        sub = np.where(visited[:, y0:y1, x0:x1], -np.inf, U[:, y0:y1, x0:x1]).max(0)
        Rv[y0:y1, x0:x1] = np.where(valid[y0:y1, x0:x1], sub, -np.inf)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
        if L >= min_len and float(np.mean(vals)) >= min_mean:
            traces.append(dict(xy=pts, h=hs, v=vals, w=w))
    return join_ends(traces)


def _arclen(xy):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def _end(tr, which, back=6.0):
    """A trace's end (0: first, 1: last): its point and the unit direction
    out of it (from the point back px inside)."""
    xy = tr["xy"]
    s = _arclen(xy)
    if which == 0:
        p, q = xy[0], xy[int(np.argmin(np.abs(s - min(back, s[-1]))))]
    else:
        p, q = xy[-1], xy[int(np.argmin(np.abs(s - max(s[-1] - back, 0.0))))]
    u = p - q
    return p, u / (np.linalg.norm(u) + 1e-9)


def join_ends(traces, gap=8.0, max_deg=35.0):
    """One vessel traced in pieces (from two seeds, round a tight turn,
    across a faint stretch): two ends within gap px, each pointing at the
    other within max_deg, become one trace.  Closest pairs first."""
    traces = list(traces)
    c = math.cos(math.radians(max_deg))
    while True:
        best = None
        for a in range(len(traces)):
            for ea in (0, 1):
                pa, ua = _end(traces[a], ea)
                for b in range(a + 1, len(traces)):
                    for eb in (0, 1):
                        pb, ub = _end(traces[b], eb)
                        d = float(np.linalg.norm(pb - pa))
                        if d > gap or (best is not None and d >= best[0]) or float(ua @ -ub) < c:
                            continue
                        if d > 1.0 and (float(ua @ (pb - pa)) / d < c or float(ub @ (pa - pb)) / d < c):
                            continue
                        best = (d, a, ea, b, eb)
        if best is None:
            return traces
        _, a, ea, b, eb = best
        A, B = traces[a], traces[b]
        fa = (lambda key: A[key][::-1]) if ea == 0 else (lambda key: A[key])     # A ending at the join
        fb = (lambda key: B[key]) if eb == 0 else (lambda key: B[key][::-1])     # B starting at it
        ha = (fa("h") + NK) % NH if ea == 0 else fa("h")
        hb = fb("h") if eb == 0 else (fb("h") + NK) % NH
        new = dict(xy=np.vstack([fa("xy"), fb("xy")]), h=np.r_[ha, hb], v=np.r_[fa("v"), fb("v")],
                   w=np.r_[fa("w"), fb("w")])
        traces = [t for k, t in enumerate(traces) if k not in (a, b)] + [new]


# ---------------------------------------------------------------- events
def _side(A, B, tb):
    """For each point of A: its signed distance from polyline B (+ on B's
    left), its distance from B (past an end, the distance beyond it too),
    the nearest B index, and whether that point is inside B's extent."""
    j = np.clip(tb.query(A)[1], 0, len(B) - 1)
    t = B[np.clip(j + 1, 0, len(B) - 1)] - B[np.clip(j - 1, 0, len(B) - 1)]
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    r = A - B[j]
    s = (r * n).sum(1)
    along = (r * t).sum(1)
    inside = ~(((j == 0) & (along < -0.5)) | ((j == len(B) - 1) & (along > 0.5)))
    return s, np.abs(s) + np.where(inside, 0.0, np.abs(along)), j, inside


def trace_events(traces, d_c=2.5, d_end=5.0, min_arm=8.0):
    """Where traces meet.  For each pair of traces A, B and each stretch
    where A runs within d_c px of B: 'cross' if A leaves it on the other
    side of B from where it came (both going on at least min_arm px either
    way), 'kiss' if on the same side, 'end' if A ends in it (or its end lies
    within d_end px of B's body without touching it).  Returns a list of
    dict(kind, a, b, ia, jb, xy): the traces and their indices there."""
    ev = []
    trees = [cKDTree(t["xy"]) for t in traces]
    lens = [_arclen(t["xy"]) for t in traces]
    boxes = [np.r_[t["xy"].min(0) - 8, t["xy"].max(0) + 8] for t in traces]
    for a, A in enumerate(traces):
        for b, B in enumerate(traces):
            ba, bb = boxes[a], boxes[b]
            if a == b or ba[0] > bb[2] or bb[0] > ba[2] or ba[1] > bb[3] or bb[1] > ba[3]:
                continue
            s, dist, j, inside = _side(A["xy"], B["xy"], trees[b])
            sa, sb = lens[a], lens[b]
            through = lambda jj: min(sb[jj], sb[-1] - sb[jj]) >= min_arm
            near = dist <= d_c
            for idx in (0, len(s) - 1):                # A stopped just short of B
                if not near[idx] and dist[idx] <= d_end and inside[idx] and through(j[idx]):
                    ev.append(dict(kind="end", a=a, b=b, ia=idx, jb=int(j[idx]), xy=B["xy"][j[idx]].copy()))
            if not near.any():
                continue
            idx = np.flatnonzero(near)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                i0, i1 = run[0], run[-1]
                im = run[np.argmin(dist[run])]
                jm = int(j[im])
                if not through(jm):
                    continue
                if i0 == 0 or i1 == len(s) - 1:        # A ends here
                    ia = 0 if i0 == 0 else len(s) - 1
                    ev.append(dict(kind="end", a=a, b=b, ia=ia, jb=int(j[ia]), xy=B["xy"][int(j[ia])].copy()))
                    continue
                if a > b:                              # each pair once from here on
                    continue
                pre = [k for k in range(i0 - 1, -1, -1) if dist[k] > d_c and inside[k]][:1]
                post = [k for k in range(i1 + 1, len(s)) if dist[k] > d_c and inside[k]][:1]
                if not (pre and post):
                    continue
                if np.sign(s[pre[0]]) == np.sign(s[post[0]]):
                    ev.append(dict(kind="kiss", a=a, b=b, ia=int(im), jb=jm, xy=A["xy"][im].copy()))
                elif sa[im] >= min_arm and sa[-1] - sa[im] >= min_arm:
                    seg = np.arange(pre[0], post[0] + 1)            # where A goes over
                    ch = seg[:-1][np.sign(s[seg[:-1]]) != np.sign(s[seg[1:]])]
                    ic = int(ch[np.argmin(np.abs(ch - (i0 + i1) / 2))]) if len(ch) else int(im)
                    ev.append(dict(kind="cross", a=a, b=b, ia=ic, jb=int(j[ic]), xy=A["xy"][ic].copy()))
    return ev


def pair_ends(traces, events, max_sep=60.0, max_deg=30.0):
    """A vessel passing through another at a small angle, or through a
    thick or blurred one, is traced as two pieces that end on it from
    opposite sides: two 'end' events on the same trace whose ending pieces
    point at each other (within max_deg), up to max_sep px apart, become one
    'pass', where the line between the two ends meets the trace they end
    on (a2, ia2: the second piece)."""
    ends = [k for k, e in enumerate(events) if e["kind"] == "end"]
    c = math.cos(math.radians(max_deg))
    cand = []
    for n, i in enumerate(ends):
        for j in ends[n + 1:]:
            e1, e2 = events[i], events[j]
            if e1["b"] != e2["b"] or e1["a"] == e2["a"]:
                continue
            p1, u1 = _end(traces[e1["a"]], 0 if e1["ia"] == 0 else 1)
            p2, u2 = _end(traces[e2["a"]], 0 if e2["ia"] == 0 else 1)
            d = float(np.linalg.norm(p2 - p1))
            if not 1.0 <= d <= max_sep:
                continue
            v = (p2 - p1) / d
            if float(u1 @ v) < c or float(u2 @ -v) < c:
                continue
            B = traces[e1["b"]]
            tb = cKDTree(B["xy"])
            s, _, _, inside = _side(np.array([p1, p2]), B["xy"], tb)
            if not inside.all() or np.sign(s[0]) == np.sign(s[1]):
                continue
            q = p1 + s[0] / (s[0] - s[1]) * (p2 - p1)
            cand.append((d, i, j, q, int(tb.query(q)[1])))
    used, out = set(), []
    for d, i, j, q, jq in sorted(cand, key=lambda t: t[0]):
        if i in used or j in used:
            continue
        used.update([i, j])
        e1, e2 = events[i], events[j]
        out.append(dict(kind="pass", a=e1["a"], b=e1["b"], ia=e1["ia"], jb=jq, xy=q,
                        a2=e2["a"], ia2=e2["ia"], span=d))
    return [e for k, e in enumerate(events) if k not in used] + out


# ---------------------------------------------------------------- junctions
def _width_at(tr, i):
    return float(tr["w"][int(np.clip(i, 0, len(tr["w"]) - 1))])


def darkness_along(D, tr, i, near, far):
    """Median darkness on trace tr between near and far px along it from
    its point i, both ways (where it goes on that far), or None."""
    s = _arclen(tr["xy"])
    sel = (np.abs(s - s[i]) >= near) & (np.abs(s - s[i]) <= far)
    if not sel.any():
        return None
    p = tr["xy"][sel]
    H, W = D.shape
    return float(np.median(ndi.map_coordinates(D, [np.clip(p[:, 1], 0, H - 1), np.clip(p[:, 0], 0, W - 1)],
                                               order=1)))


def overlap_ratio(D, traces, e, pad=3.0, span=7.0):
    """How dark the point where traces a and b meet is against the two
    lines just outside it (from the other line's half-width + pad px out,
    span px on): about 1 where two vessels lie on top of each other (their
    darkness adds up), about half where they only touch."""
    A, B = traces[e["a"]], traces[e["b"]]
    na = 1.2 * _width_at(B, e["jb"]) + pad
    nb = 1.2 * _width_at(A, e["ia"]) + pad
    da = darkness_along(D, A, e["ia"], na, na + span)
    db = darkness_along(D, B, e["jb"], nb, nb + span)
    if da is None or db is None:
        return None
    H, W = D.shape
    x, y = int(round(e["xy"][0])), int(round(e["xy"][1]))
    c = float(D[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].max())
    return c / max(da + db, 1e-9)


def _arm_dirs(tr, i, reach=12.0, min_arm=8.0):
    """Directions (rad) in which trace tr leaves its point i: towards the
    point reach px along it each way where it goes on at least min_arm."""
    xy = tr["xy"]
    s = _arclen(xy)
    out = []
    for sign in (1, -1):
        room = (s[-1] - s[i]) if sign > 0 else s[i]
        if room < min_arm:
            continue
        v = xy[int(np.argmin(np.abs(s - (s[i] + sign * min(reach, room)))))] - xy[i]
        out.append(math.atan2(v[1], v[0]) % (2 * math.pi))
    return out


def junctions(traces, events, D, merge=6.0, sep_deg=15.0, min_overlap=0.72, host_arm=12.0):
    """Events (paired: pair_ends) grouped into junctions, within merge px
    plus twice the widest line there, with the arms of the traces meeting
    there: dict(xy, arms, kind, width, events).  Crossings of traces whose
    meeting point is less than min_overlap as dark as both lines (touching,
    not lying on top of each other), ends on a trace going on less than
    host_arm px either side (a stub past a turn), and kisses are left out."""
    from .intersections import _angdiff, classify
    keep = []
    for e in pair_ends(traces, events):
        if e["kind"] == "kiss":
            continue
        if e["kind"] == "cross":
            r = overlap_ratio(D, traces, e)
            if r is not None and r < min_overlap:
                continue
        if e["kind"] == "end":
            sb = _arclen(traces[e["b"]]["xy"])
            if min(sb[e["jb"]], sb[-1] - sb[e["jb"]]) < host_arm:
                continue
        keep.append((e, max(_width_at(traces[e["a"]], e["ia"]), _width_at(traces[e["b"]], e["jb"]))))
    groups = []
    for e, w in keep:
        for g in groups:
            c = np.mean([x["xy"] for x, _ in g], 0)
            if np.linalg.norm(c - e["xy"]) < merge + 2 * max(w, max(ww for _, ww in g)):
                g.append((e, w))
                break
        else:
            groups.append([(e, w)])
    out = []
    for g in groups:
        c = np.mean([e["xy"] for e, _ in g], 0)
        arms, seen = [], set()
        for e, _ in g:
            members = [(e["a"], e["ia"], e["kind"] == "pass"), (e["b"], e["jb"], False)]
            if e["kind"] == "pass":
                members.append((e["a2"], e["ia2"], True))
            for t, i, own_end in members:
                if t in seen:
                    continue
                seen.add(t)
                # a piece ending on the host leaves from its own end; others from where they pass
                k = i if own_end else int(np.argmin(np.linalg.norm(traces[t]["xy"] - c, axis=1)))
                for d in _arm_dirs(traces[t], k):
                    if all(_angdiff(d, x) >= math.radians(sep_deg) for x in arms):
                        arms.append(d)
        arms = sorted(arms)
        if len(arms) >= 3:
            out.append(dict(xy=c, arms=arms, kind=classify(arms), scale=2.0, z=0.0,
                            width=max(w for _, w in g), events=sorted({e["kind"] for e, _ in g})))
    return out


def arm_support(p, arms, traces, near=3.0, per_w=1.2, end_slack=4.0, reach=15.0, out_r=12.0,
                tol_deg=25.0, aim_gap=20.0, aim_lat=2.0):
    """How many of the arms (rad) at p are traced vessels, matched one to
    one within tol_deg.  A trace reaches p when it comes within near +
    per_w x its width of it (end_slack px more where it ends there: a
    branch's trace stops a few px short of its host), or when it stops
    short within aim_gap px and its line carried on passes within aim_lat
    + per_w x its width (at a shallow crossing the second vessel's trace is
    cut back where it merges into the first's; a capillary's pieces stop at
    a blurred vessel's edge).  It leaves p each way it goes on to at least
    reach px from p, in the direction of its first point out_r px out.  A
    streak of texture is not traced, and a neighbour running alongside
    passes by instead of reaching p."""
    from .intersections import _angdiff
    p = np.asarray(p, float)
    dirs = []

    def leave(seq):
        r = np.linalg.norm(seq - p, axis=1)
        far = np.flatnonzero(r >= out_r)
        if len(far) and r.max() >= reach:
            q = seq[far[0]] - p
            dirs.append(math.atan2(q[1], q[0]))

    for t in traces:
        xy = t["xy"]
        if (xy.min(0) > p + aim_gap + 1.0).any() or (xy.max(0) < p - aim_gap - 1.0).any():
            continue
        d = np.linalg.norm(xy - p, axis=1)
        i = int(np.argmin(d))
        if d[i] <= near + per_w * t["w"][i] + (end_slack if i in (0, len(xy) - 1) else 0.0):
            leave(xy[i:])
            leave(xy[i::-1])
            continue
        for which in (0, 1):
            e, u = _end(t, which, back=8.0)
            v = p - e
            w = t["w"][0 if which == 0 else -1]
            if float(v @ u) > 0 and np.linalg.norm(v) <= aim_gap and \
                    abs(float(v[0] * u[1] - v[1] * u[0])) <= aim_lat + per_w * w:
                leave(xy if which == 0 else xy[::-1])
    pairs = sorted((_angdiff(a, b), i, j) for i, a in enumerate(arms) for j, b in enumerate(dirs))
    used_a, used_d = set(), set()
    for dd, i, j in pairs:
        if dd <= math.radians(tol_deg) and i not in used_a and j not in used_d:
            used_a.add(i)
            used_d.add(j)
    return len(used_a)


def combine(dets, traces, events, D, merge=8.0, cross_clear=20.0, **jkw):
    """intersections.detect's detections, and the junctions found from
    traces that it lacks: where one vessel passes through another ('pass':
    shallow crossings, a thin vessel through a thick or blurred one) more
    than merge px + twice the width from any detection, and where two
    traces cross in an X (two straight pairs of arms) more than cross_clear
    px from any (nearer, it is the same crossing seen off centre).  Ends of
    traces on others (Ts, forks) are left to the detector."""
    out = list(dets)
    for j in junctions(traces, events, D, **jkw):
        far = lambda r: all(np.linalg.norm(j["xy"] - d["xy"]) > r for d in dets)
        if "pass" in j["events"] and far(merge + 2 * j["width"]):
            out.append(j)
        elif "cross" in j["events"] and j["kind"] == "crossing" and far(cross_clear):
            out.append(j)
    return out
