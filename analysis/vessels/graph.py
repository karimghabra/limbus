"""Junctions: which pieces are one vessel, and which vessel branches off which.

A detected piece stops wherever the ridge stopped - at a crossing, at a
bifurcation, or where contrast dipped - so one real vessel arrives as several
pieces. This module decides what happens at each meeting point and threads the
pieces back into vessels that are as long and as continuous as the evidence
allows.

At a meeting point the incident piece-ends are PAIRED: two ends that are
paired belong to the same vessel and it runs through. The pairing rule depends
on how many pieces meet.

  two ends      a continuation - the ridge simply stopped. Paired if the turn
                between them is small.
  four ends     a crossing: two vessels passing over and under each other.
                Each end is paired with the one most nearly opposite it, and
                the pairing is only accepted if both pairs run nearly straight
                through AND the two radii in each pair agree. A vessel does
                not change calibre because another vessel passes it.
  three ends    a bifurcation: a parent divides into two children. The parent
                is identified by calibre, and the vessel continues into
                whichever child best matches it in direction and calibre; the
                other child starts a new vessel whose parent is this one.
                Murray's law - r_parent^3 = r_child1^3 + r_child2^3, which
                follows from minimising the work of driving flow through a
                branch - is recorded as a residual, so a junction that does
                not behave like a bifurcation can be told apart afterwards.

Anything with more ends, or where the geometry does not support a pairing, is
left unpaired and reported as unresolved rather than guessed at. That costs
real continuity where junctions crowd - a trunk with three branches inside
30 px comes back in four pieces, and resolves 0.23 of the time against 0.73 in
a 60 px window - and the obvious repair is worse. Pairing the ends of a
degree->=5 node greedily by turn and calibre (cfg.high_degree, off) takes that
same case to 0.14 and raises merges per copy from 0.02 to 0.59, because at a
crowded node a trunk and its own branch look exactly like a continuation.

And calibre cannot rescue it, which is worth stating because it is the obvious
next idea. By Murray's law a trunk that sheds a symmetric branch continues at
exactly the branch's own radius - r=2.5 sheds r=1.98 and carries on at 1.98,
a ratio of 1.00 - so the two are indistinguishable by width by construction.
Only an asymmetric split separates them (a 30 % branch leaves a ratio of 1.33).
What does separate them is DIRECTION: the trunk runs on, the branch turns
away. That is already the test; it fails at a crowded node because the pieces
there are short and an end's tangent, measured over 10 px of a 15 px piece,
is mostly noise. The repair to try is a tangent measured over an arc that
provably excludes the junction, and a refusal to pair when no such arc
exists.

Vessels are then maximal chains under the pairing, and are numbered from the
largest down: the thickest trunk is vessel 1, and a branch off vessel 3
carrying the second-largest calibre is labelled 3.2. That is the order blood
takes through the network, from large vessels into small ones.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GraphConfig:
    node_tol: float = 10.0        # ends this close are the same meeting point (px)
    touch_tol: float = 8.0        # an end this close to another piece splits it (px)
    straight_deg: float = 50.0    # how far from straight a crossing may run
    branch_deg: float = 75.0      # how far from straight a bifurcation may continue
    radius_ratio: float = 1.8     # calibre agreement across a crossing
    min_radius: float = 0.5
    # What makes two fragments one vessel. A vessel does not change calibre or
    # darkness abruptly, and it does not turn sharply, so fragments are only
    # consolidated when all three agree. Each is a ratio, so none of them
    # depends on how thick or how dark the vessel happens to be.
    join_radius_ratio: float = 2.0    # diameters may differ by this much (the width of a
                                      # thin vessel is itself only good to a factor ~1.5)
    join_dark_ratio: float = 2.0      # absorbance along them may differ by this much
    join_turn_deg: float = 45.0       # how far the two ends may be from continuing
    dup_overlap: float = 0.5          # share of the shorter piece lying on the longer
    dup_dist: float = 1.2             # ... within this many radii of it
    tangent_standoff: float = 0.0 # skip this many px back from a piece's end before
                                  # reading its direction. The last pixels of a piece are
                                  # bent by whatever it ran into, and at a crowded node a
                                  # piece may be 15 px long, so a tangent from the tip
                                  # describes the junction and not the vessel.
    high_degree: bool = False     # read meeting points of degree >= 5 rather than
                                  # abandoning them. OFF: measured, greedy pairing at a
                                  # crowded node fuses a trunk with its own branches -
                                  # three branches in 30 px go from 0.23 resolved to 0.14
                                  # and from 0.02 merges per copy to 0.59. The mechanism
                                  # it targets is real; this pairing rule is not the fix.
    cluster: str = "single"       # how piece ends are grouped into meeting points.
                                  # "single" is transitive and lets crowded junctions
                                  # chain into one node of degree 7-8 that classify()
                                  # cannot read; "complete" bounds a node's diameter and
                                  # fixes the tightest crowding - 0.24 -> 0.64 on three
                                  # branches in 30 px - but LOSES as much elsewhere
                                  # (0.87 -> 0.56 on four branches in 40 px) and merges
                                  # more, so it is offered rather than assumed.
    trim_split_fits: bool = True      # a piece split at a touch gets its OWN fit, covering
                                      # only the part it kept. Without this the two halves
                                      # share one fit object, and if they end up in
                                      # different vessels both export the whole original -
                                      # 26 % of the exported pieces on one crop were the
                                      # same geometry claimed by two vessels, which then
                                      # read the same kymograph and the same velocity.


def _tangent(C, at_start, arc=10.0, standoff=0.0):
    """Unit vector pointing OUT of the piece at one of its ends.

    Measured over a fixed LENGTH of the centreline, not a fixed number of
    samples: centrelines are sampled every half pixel, so ten samples span
    five pixels and the direction they give is mostly noise - which quietly
    cost branches their attachment to the vessel they leave.

    `standoff` skips that many pixels back from the end before measuring. The
    last few pixels of a piece are the ones bent by whatever it ran into, and
    at a crowded junction a piece may be only 15 px long, so a tangent taken
    from its very tip describes the junction rather than the vessel. With a
    standoff the direction is read from the part of the piece that is still
    the vessel, and the chord is taken back to the tip so it still points out.
    """
    C = np.asarray(C, float)
    if len(C) < 2:
        return np.array([1.0, 0.0])
    walk = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    L = float(walk[-1])
    if L <= 1e-9:
        return np.array([1.0, 0.0])
    if standoff + 2.0 > L:               # too short to stand off from
        standoff = 0.0
    arc = min(arc, max(L - standoff, 1e-9))
    if at_start:
        s_near, s_far = standoff, standoff + arc
    else:
        s_near, s_far = L - standoff, L - standoff - arc
    near = np.array([np.interp(s_near, walk, C[:, 0]), np.interp(s_near, walk, C[:, 1])])
    far = np.array([np.interp(s_far, walk, C[:, 0]), np.interp(s_far, walk, C[:, 1])])
    d = near - far
    n = float(np.hypot(*d))
    if n < 1e-9:
        d = C[0] - C[-1] if at_start else C[-1] - C[0]
        n = max(float(np.hypot(*d)), 1e-9)
    return d / n


def _turn(t1, t2):
    """Angle (deg) between the direction one end points and the reverse of the
    other: 0 means the two pieces continue each other exactly."""
    return float(np.degrees(np.arccos(np.clip(np.dot(t1, -t2), -1, 1))))


def _ratio(a, b, floor=1e-6):
    a, b = abs(float(a)), abs(float(b))
    return max(a, b) / max(min(a, b), floor)


def end_stats(points, at_start, A, span=24.0):
    """Diameter and darkness measured NEAR one end of a piece.

    Two fragments of one vessel match where they meet, not on average: a
    vessel tapers, and a piece that runs through a crossing carries the other
    vessel's absorbance in its middle. The width comes from the half-maximum
    of the absorbance profile across the vessel, the darkness from the
    absorbance along it, both over the last `span` pixels.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        return 2.0, 0.0
    step = np.hypot(*np.diff(P, axis=0).T)
    walk = np.concatenate([[0.0], np.cumsum(step)])
    sel = walk <= span if at_start else walk >= walk[-1] - span
    seg = P[sel]
    if len(seg) < 3:
        seg = P[:3] if at_start else P[-3:]
    H, W = A.shape
    xi = np.clip(np.rint(seg[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.rint(seg[:, 1]).astype(int), 0, H - 1)
    dark = float(np.median(A[yi, xi]))
    t = np.gradient(seg, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = np.arange(-12, 12.5, 0.5, dtype=np.float32)
    px = (seg[:, None, 0] + n[:, None, 0] * offs[None, :]).astype(np.float32)
    py = (seg[:, None, 1] + n[:, None, 1] * offs[None, :]).astype(np.float32)
    prof = np.median(cv2.remap(A, px, py, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE), axis=0)
    c = len(offs) // 2
    base = min(float(np.median(prof[:6])), float(np.median(prof[-6:])))
    peak = float(prof[max(c - 3, 0):c + 4].max()) - base
    if peak <= 0:
        return 2.0, dark
    half = base + peak / 2
    right = np.flatnonzero(prof[c:] < half)
    left = np.flatnonzero(prof[:c + 1][::-1] < half)
    hw = 0.25 * ((right[0] if right.size else 24) + (left[0] if left.size else 24))
    return float(max(hw, 0.8)), dark


def same_vessel(sa, sb, ta, tb, cfg):
    """Do two fragments look like one vessel? Diameter, darkness, direction.

    Returns (verdict, why) so a refusal can be read back. None of the three
    tests is about how strong the evidence is - a faint vessel and a dark one
    are both allowed, as long as each fragment matches the other.
    """
    dr = _ratio(sa.get("radius", 2.0), sb.get("radius", 2.0), 1.0)
    dk = _ratio(sa.get("darkness", 1.0), sb.get("darkness", 1.0), 1e-3)
    turn = _turn(ta, tb)
    why = []
    if dr > cfg.join_radius_ratio:
        why.append(f"diameters differ x{dr:.1f}")
    if dk > cfg.join_dark_ratio:
        why.append(f"darkness differs x{dk:.1f}")
    if turn > cfg.join_turn_deg:
        why.append(f"turns {turn:.0f} deg")
    return (not why), why


def drop_duplicates(segments, cfg=None):
    """Consolidate fragments that trace the SAME vessel twice.

    The large and small passes overlap by design, so a vessel both can see
    comes back as two nearly coincident lines. Where a piece runs within about
    a radius of a longer piece over most of its length, and the two agree in
    diameter and in darkness, it is the same vessel found twice and only the
    longer line is kept. Direction needs no separate test here: lines that lie
    on top of each other already run the same way.
    """
    cfg = cfg or GraphConfig()

    def length(i):
        P = np.asarray(segments[i]["points"], float)
        return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0

    order = sorted(range(len(segments)), key=lambda i: -length(i))
    keep, kept_pts = [], []
    for i in order:
        C = np.asarray(segments[i]["points"], float)
        r = float(segments[i].get("radius", 2.0))
        dup = False
        for j, K in zip(keep, kept_pts):
            if _ratio(segments[i].get("radius", 2.0), segments[j].get("radius", 2.0),
                      cfg.min_radius) > cfg.join_radius_ratio:
                continue
            if _ratio(segments[i].get("darkness", 1.0),
                      segments[j].get("darkness", 1.0)) > cfg.join_dark_ratio:
                continue
            step = max(1, len(C) // 200)                      # sampled, for speed
            d = np.min(np.hypot(*(C[::step, None, :] - K[None, ::max(1, len(K) // 200), :]
                                  ).transpose(2, 0, 1)), axis=1)
            if float((d <= cfg.dup_dist * max(r, 1.0)).mean()) >= cfg.dup_overlap:
                dup = True
                break
        if not dup:
            keep.append(i)
            kept_pts.append(C)
    return [segments[i] for i in sorted(keep)]


def merge_gaps(segments, z, join_ends_fn, max_gap=50.0, turn=np.radians(40), min_z=1.0,
               cfg=None, A=None):
    """Repair a piece broken by a gap BEFORE any junction is interpreted.

    A vessel often breaks where a branch leaves it or where contrast dips, and
    the two halves then end 20 px apart with the branch's end between them.
    Clustering sees two separate meetings instead of one three-way junction,
    and the branch never finds its parent. Joining the halves first - only
    when their ends point at each other and the cheapest path between them
    keeps to vessel-like pixels, which is what join_ends checks - leaves one
    continuous piece for the branch to attach to.

    The connector is inserted as centreline, unlike the identity-only joins:
    the evidence test has already established that a vessel runs there.
    """
    cfg = cfg or GraphConfig()
    pts = [np.asarray(s["points"], float) for s in segments]
    _, joins = join_ends_fn(pts, z, max_gap, turn, min_z)
    # a gap is only closed between fragments that look like the same vessel
    ok = []
    for j in joins:
        ta = _tangent(pts[j["a"]], j["a_start"])
        tb = _tangent(pts[j["b"]], j["b_start"])
        if A is None:
            good = True
        else:
            ra, da = end_stats(pts[j["a"]], j["a_start"], A)
            rb, db = end_stats(pts[j["b"]], j["b_start"], A)
            good, _ = same_vessel({"radius": ra, "darkness": da},
                                  {"radius": rb, "darkness": db}, ta, tb, cfg)
        if good:
            ok.append(j)
    joins = ok
    owner = list(range(len(segments)))

    def root(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    parts = {i: [pts[i]] for i in range(len(segments))}
    heads = {i: (pts[i][0], pts[i][-1]) for i in range(len(segments))}
    merged_into = {}
    for j in sorted(joins, key=lambda j: j["gap"]):
        ra, rb = root(j["a"]), root(j["b"])
        if ra == rb:
            continue
        A_, B_ = parts.pop(ra), parts.pop(rb)
        path = np.asarray(j["path"], float)
        pa = np.asarray(pts[j["a"]][0] if j["a_start"] else pts[j["a"]][-1], float)
        if np.hypot(*(A_[0][0] - pa)) < np.hypot(*(A_[-1][-1] - pa)):
            A_ = [q[::-1] for q in A_][::-1]
        pb = np.asarray(pts[j["b"]][0] if j["b_start"] else pts[j["b"]][-1], float)
        if np.hypot(*(B_[-1][-1] - pb)) < np.hypot(*(B_[0][0] - pb)):
            B_ = [q[::-1] for q in B_][::-1]
        if len(path) > 2 and np.hypot(*(path[0] - A_[-1][-1])) > np.hypot(*(path[-1] - A_[-1][-1])):
            path = path[::-1]
        parts[ra] = A_ + ([path[1:-1]] if len(path) > 2 else []) + B_
        owner[rb] = ra
        merged_into[rb] = ra
    out = []
    for i, ps in parts.items():
        seg = dict(segments[i])
        seg["points"] = np.concatenate([q for q in ps if len(q)], 0)
        members = [k for k in range(len(segments)) if root(k) == i]
        seg["radius"] = float(np.median([segments[k]["radius"] for k in members]))
        # Keep EVERY member's fit, not just the one whose dict was copied. The
        # merged piece spans all of them, and dropping the rest is why a vessel
        # could report 681 px of centreline and carry a single 120 px fit - and
        # why the velocity stage, which samples the longest fitted piece, saw
        # only 58 % of what the threading produced.
        fits = []
        for k in members:
            for f in (segments[k].get("fits") or ([segments[k]["fit"]]
                                                  if segments[k].get("fit") is not None else [])):
                if all(f is not g for g in fits):
                    fits.append(f)
        if fits:
            seg["fits"] = fits
        out.append(seg)
    return out


def attach_to_body(segments, A, cfg=None, max_gap=30.0, point_deg=60.0, dark_frac=0.5):
    """Extend a piece that stops just short of a wider vessel it joins.

    A small vessel leaving a large one is lost in the large one's shadow for
    its first stretch: the wide vessel's own blur swamps it, and the fit cuts
    whatever lies inside an accepted lumen. The branch is therefore detected
    starting a dozen pixels away from its parent, with nothing in between.

    The end is extended to the parent's centreline when it points at it, the
    gap is short, and the image stays DARK along the way - at least half as
    dark as the branch itself. That is the test that matters: the gap has to
    be explained by absorption (the parent's penumbra, or the branch fading
    into it), not by bright sclera, which would mean there is nothing there.
    An earlier version asked instead that the gap lie inside an accepted
    lumen, which a branch stub can never satisfy - it stops OUTSIDE its
    parent's wall, so at most half the path is ever inside.
    """
    cfg = cfg or GraphConfig()
    A = np.asarray(A, np.float32)
    H, W = A.shape
    out = [dict(s) for s in segments]
    for si, s in enumerate(out):
        C = np.asarray(s["points"], float)
        for at_start in (True, False):
            p = C[0] if at_start else C[-1]
            t = _tangent(C, at_start)
            best = None
            for sj, other in enumerate(out):
                if sj == si:
                    continue
                O = np.asarray(other["points"], float)
                if len(O) < 4 or float(other["radius"]) <= float(s["radius"]):
                    continue                       # only attach to something wider
                d = np.hypot(*(O - p).T)
                k = int(np.argmin(d))
                if not (1.0 < d[k] <= max_gap):      # 'already touching' is handled by the split
                    continue
                if k < 5 or k > len(O) - 6:
                    # the nearest point is the other piece's own END, so these
                    # are two ends meeting - a crossing or a continuation, for
                    # the junction rules to decide. A branch joins a body.
                    continue
                v = O[k] - p
                if np.degrees(np.arccos(np.clip(np.dot(t, v / max(np.hypot(*v), 1e-9)), -1, 1))) > point_deg:
                    continue
                n = max(3, int(d[k]))
                xs = np.clip(np.rint(np.linspace(p[0], O[k][0], n)).astype(int), 0, W - 1)
                ys = np.clip(np.rint(np.linspace(p[1], O[k][1], n)).astype(int), 0, H - 1)
                tail = C[:20] if at_start else C[-20:]
                ti = np.clip(np.rint(tail).astype(int), [0, 0], [W - 1, H - 1])
                own = float(np.median(A[ti[:, 1], ti[:, 0]]))
                if float(np.mean(A[ys, xs])) < dark_frac * own:
                    continue
                if best is None or d[k] < best[0]:
                    best = (d[k], O[k])
            if best is not None:
                link = np.linspace(p, best[1], max(3, int(best[0] / 2)))
                C = np.vstack([link[::-1], C]) if at_start else np.vstack([C, link])
        out[si]["points"] = C
    return out


def split_at_touches(segments, cfg=None):
    """Split a piece where another piece's end lands on its middle.

    A bifurcation is often detected as one piece running through and a second
    arriving at its side; without splitting the through piece there is no
    meeting point to classify, and the side branch looks like a free end.
    Returns new segments; each keeps the original's radius and a 'source'.
    """
    cfg = cfg or GraphConfig()
    tips = []
    for si, s in enumerate(segments):
        C = np.asarray(s["points"], float)
        tips.append((si, C[0]))
        tips.append((si, C[-1]))
    out = []
    for si, s in enumerate(segments):
        C = np.asarray(s["points"], float)
        cuts = []
        for tj, p in tips:
            if tj == si or len(C) < 6:
                continue
            d = np.hypot(*(C - p).T)
            k = int(np.argmin(d))
            if d[k] <= cfg.touch_tol and 3 <= k <= len(C) - 4:
                cuts.append(k)
        cuts = sorted(set(cuts))
        merged = []
        for k in cuts:                      # ignore cuts that sit on top of each other
            if not merged or k - merged[-1] > 3:
                merged.append(k)
        bounds = [0] + merged + [len(C) - 1]
        split = len(bounds) > 2
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 3:
                continue
            piece = dict(s)
            piece["points"] = C[a:b + 1]
            piece["source"] = s.get("source", si)
            if split and cfg.trim_split_fits:
                if s.get("fits"):
                    piece["fits"] = _fits_within(s["fits"], C[a:b + 1])
                elif s.get("fit") is not None:
                    piece["fit"] = _trim_fit(s["fit"], a, b)
            out.append(piece)
    return out


def _fits_within(fits, span, tol=4.0):
    """The fits that belong to this part of a split piece.

    A piece merged from several fragments carries all their fits; splitting it
    at a touch must hand each fit to the part that actually contains it, or
    every part claims all of them and two vessels export the same geometry
    again - the defect this is here to remove, reappearing one stage later.
    """
    from . import model
    out = []
    for f in fits:
        P = np.asarray(f[0], float)
        if len(P) < 2:
            continue
        C = model.catmull(P)[0] if len(P) > 2 else P
        d = np.hypot(*(C[:, None, :] - span[None, :, :]).transpose(2, 0, 1)).min(1)
        if float((d <= tol).mean()) >= 0.5:
            out.append(f)
    return out


def _trim_fit(fit, a, b, spacing=16.0):
    """The part of a fit that belongs to one half of a split piece.

    `dict(s)` copies the mapping but not the fit inside it, so both halves of
    a split would otherwise carry - and export - the whole original vessel.
    The control points are resampled along the span that was kept, and the
    radii interpolated at the same places, so each half describes itself and
    nothing else. Depth and the fit record are unchanged: they are properties
    of the vessel, not of the span.
    """
    from . import model
    P, R, D, rec = fit
    P = np.asarray(P, float)
    if R is None or len(P) < 2:
        return fit
    C, u = model.catmull(P)
    b = min(int(b), len(C) - 1)
    a = max(0, int(a))
    if b - a < 3:
        return fit
    seg, useg = C[a:b + 1], u[a:b + 1]
    L = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(seg, axis=0).T))])
    if L[-1] <= 1e-6:
        return fit
    k = max(2, int(round(L[-1] / spacing)) + 1)
    s = np.linspace(0.0, L[-1], k)
    Pn = np.stack([np.interp(s, L, seg[:, 0]), np.interp(s, L, seg[:, 1])], 1)
    Rn = np.interp(np.interp(s, L, useg), np.arange(len(R)), np.asarray(R, float))
    return (Pn, Rn, D, rec)


def build(segments, cfg=None):
    """segments: [{'points': (N,2) centreline, 'radius': px, ...}].

    Returns (nodes, ends, pairs) where ends[i] = (segment, at_start) in the
    order they were collected, nodes maps a node index to its end indices, and
    pairs maps an end index to the end it continues into (or -1).
    """
    cfg = cfg or GraphConfig()
    ends = []
    for si, s in enumerate(segments):
        C = np.asarray(s["points"], float)
        so = float(getattr(cfg, "tangent_standoff", 0.0))
        ends.append({"seg": si, "start": True, "p": C[0], "t": _tangent(C, True, standoff=so),
                     "r": float(s["radius"])})
        ends.append({"seg": si, "start": False, "p": C[-1], "t": _tangent(C, False, standoff=so),
                     "r": float(s["radius"])})
    # cluster ends into meeting points (union-find on distance)
    parent = list(range(len(ends)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Ends meet at one point if they are close COMPARED WITH THE VESSELS
    # involved. Where a vessel crosses another, both are cut back by the
    # width of what occludes them, so the four ends of a crossing sit a
    # radius or two apart; a fixed tolerance clusters only some of them and
    # the junction is then misread as a continuation between two different
    # vessels.
    #
    # Single linkage is transitive, and that is a problem where junctions
    # crowd: three branches leaving a trunk within 30 px chain into ONE node
    # whose ends are 35 px apart, classify() handles degree <= 4 and returns
    # "unresolved" above it, and every end in the cluster is left unpaired.
    # Measured, the same three branches resolve 0.64 of the time in a 60 px
    # window and 0.25 in a 30 px one - crowding, not the number of branches,
    # is what breaks it. Complete linkage keeps a node's own diameter bounded:
    # two ends join a node only if the end is close to EVERY member, so a
    # chain of near-misses cannot become one junction.
    if getattr(cfg, "cluster", "single") == "single":
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                if ends[i]["seg"] == ends[j]["seg"]:
                    continue
                tol = cfg.node_tol + ends[i]["r"] + ends[j]["r"]
                if np.hypot(*(ends[i]["p"] - ends[j]["p"])) <= tol:
                    parent[root(i)] = root(j)
        nodes = {}
        for i in range(len(ends)):
            nodes.setdefault(root(i), []).append(i)
    else:
        groups = [[i] for i in range(len(ends))]
        merged = True
        while merged:
            merged = False
            best = None
            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    ok, worst = True, 0.0
                    for i in groups[a]:
                        for j in groups[b]:
                            if ends[i]["seg"] == ends[j]["seg"]:
                                ok = False
                                break
                            d = float(np.hypot(*(ends[i]["p"] - ends[j]["p"])))
                            if d > cfg.node_tol + ends[i]["r"] + ends[j]["r"]:
                                ok = False
                                break
                            worst = max(worst, d)
                        if not ok:
                            break
                    if ok and (best is None or worst < best[0]):
                        best = (worst, a, b)
            if best is not None:
                _, a, b = best
                groups[a] = groups[a] + groups[b]
                groups.pop(b)
                merged = True
        nodes = {min(g): sorted(g) for g in groups}
    pairs = {i: -1 for i in range(len(ends))}
    records = []
    for r, members in sorted(nodes.items()):
        rec = classify(ends, members, cfg)
        records.append(rec)
        for a, b in rec["pairs"]:
            pairs[a], pairs[b] = b, a
    return ends, nodes, pairs, records


def classify(ends, members, cfg):
    """Decide what a meeting point is and which ends continue each other."""
    k = len(members)
    pos = np.mean([ends[i]["p"] for i in members], 0)
    rec = {"x": float(pos[0]), "y": float(pos[1]), "degree": k, "ends": list(members),
           "kind": "unresolved", "pairs": []}
    if k == 1:
        rec["kind"] = "free end"
        return rec
    if k == 2:
        a, b = members
        turn = _turn(ends[a]["t"], ends[b]["t"])
        rec["turn_deg"] = round(turn, 1)
        rr = _ratio(ends[a]["r"], ends[b]["r"], cfg.min_radius)
        rec["radius_ratio"] = round(rr, 2)
        if turn <= cfg.branch_deg and rr <= cfg.join_radius_ratio:
            rec["kind"] = "continuation"
            rec["pairs"] = [(a, b)]
        return rec
    if k == 3:
        order = sorted(members, key=lambda i: -ends[i]["r"])
        p, c1, c2 = order
        rp, r1, r2 = (ends[i]["r"] for i in order)
        murray = (r1 ** 3 + r2 ** 3) ** (1 / 3)
        rec.update({"kind": "bifurcation", "parent_end": p,
                    "radius_parent": round(rp, 2), "radius_children": [round(r1, 2), round(r2, 2)],
                    "murray_radius": round(float(murray), 2),
                    "murray_residual": round(float((murray - rp) / max(rp, 1e-9)), 3)})
        # the vessel continues into whichever child best matches the parent
        best, best_score = None, None
        for c in (c1, c2):
            turn = _turn(ends[p]["t"], ends[c]["t"])
            if turn > cfg.branch_deg:
                continue
            score = turn + 40 * abs(np.log(max(ends[c]["r"], cfg.min_radius)
                                            / max(rp, cfg.min_radius)))
            if best_score is None or score < best_score:
                best, best_score = c, score
        if best is not None:
            rec["pairs"] = [(p, best)]
            rec["continues_into"] = int(best)
            rec["branch_end"] = int(c1 if best == c2 else c2)
        return rec
    if k == 4:
        # pair each end with the one most nearly opposite it
        best = None
        for first in members[1:]:
            rest = [m for m in members if m not in (members[0], first)]
            cand = [(members[0], first), (rest[0], rest[1])]
            turns = [_turn(ends[a]["t"], ends[b]["t"]) for a, b in cand]
            ratios = [max(ends[a]["r"], ends[b]["r"]) / max(min(ends[a]["r"], ends[b]["r"]), cfg.min_radius)
                      for a, b in cand]
            score = max(turns)
            if best is None or score < best[0]:
                best = (score, cand, turns, ratios)
        score, cand, turns, ratios = best
        rec.update({"turns_deg": [round(t, 1) for t in turns],
                    "radius_ratios": [round(r, 2) for r in ratios]})
        if max(turns) <= cfg.straight_deg and max(ratios) <= cfg.radius_ratio:
            rec["kind"] = "crossing"
            rec["pairs"] = cand
        return rec
    if k >= 5 and getattr(cfg, "high_degree", True):
        # Several junctions inside one cluster. Ends are grouped by distance,
        # and where branches crowd onto a trunk the groups run together: three
        # bifurcations within 30 px arrive as one meeting point of degree 7 or
        # 8. Returning "unresolved" here leaves EVERY end in the cluster
        # unpaired, so the trunk is not threaded through its own junction and
        # comes back in as many pieces as it has branches - which is why the
        # same three branches resolve 0.61 of the time in a 60 px window and
        # 0.24 in a 30 px one.
        #
        # There is no single junction type to name, so none is claimed. What
        # can still be said is which ends continue each other, and that is
        # decided the same way as everywhere else: small turn, matching
        # calibre, cheapest first, each end used once. Ends left over are
        # reported as free rather than forced into a pairing.
        cand = []
        for ii, a in enumerate(members):
            for b in members[ii + 1:]:
                turn = _turn(ends[a]["t"], ends[b]["t"])
                ratio = _ratio(ends[a]["r"], ends[b]["r"], cfg.min_radius)
                if turn <= cfg.straight_deg and ratio <= cfg.radius_ratio:
                    cand.append((turn + 40 * abs(np.log(max(ratio, 1e-6))), turn, ratio, a, b))
        cand.sort()
        used, pairs, why = set(), [], []
        for cost, turn, ratio, a, b in cand:
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            pairs.append((a, b))
            why.append({"turn_deg": round(turn, 1), "radius_ratio": round(ratio, 2)})
        if pairs:
            rec["kind"] = "cluster"
            rec["pairs"] = pairs
            rec["continuations"] = why
            rec["unpaired"] = [int(m) for m in members if m not in used]
        return rec
    return rec


def crossing_under(segments, ends, pairs, occupied, cfg=None, max_gap=45.0,
                   turn_deg=35.0, inside_frac=0.6):
    """Pair the two stubs a vessel leaves where it passes under a wider one.

    Where a thin vessel crosses a thick one it disappears into the thick
    vessel's shadow, and the fit trims whatever lies inside an accepted
    vessel's lumen, so the thin vessel arrives as two stubs on opposite sides
    with nothing between them. There is no four-way junction to classify.

    Two unpaired ends are joined as one vessel passing under another when they
    point at each other, their calibres agree, and the straight line between
    them runs mostly THROUGH another vessel's lumen - which is what makes this
    different from a gap in open sclera, where there is no reason for the
    evidence to be missing.
    """
    cfg = cfg or GraphConfig()
    H, W = occupied.shape
    out, cand = [], []
    for a in range(len(ends)):
        if pairs.get(a, -1) >= 0:
            continue
        for b in range(a + 1, len(ends)):
            if pairs.get(b, -1) >= 0 or ends[a]["seg"] == ends[b]["seg"]:
                continue
            v = ends[b]["p"] - ends[a]["p"]
            g = float(np.hypot(*v))
            if not 1e-6 < g <= max_gap:
                continue
            u = v / g
            if np.degrees(np.arccos(np.clip(np.dot(ends[a]["t"], u), -1, 1))) > turn_deg:
                continue
            if np.degrees(np.arccos(np.clip(np.dot(ends[b]["t"], -u), -1, 1))) > turn_deg:
                continue
            ra, rb = max(ends[a]["r"], cfg.min_radius), max(ends[b]["r"], cfg.min_radius)
            if max(ra, rb) / min(ra, rb) > cfg.radius_ratio:
                continue
            n = max(3, int(g))
            xs = np.clip(np.rint(np.linspace(ends[a]["p"][0], ends[b]["p"][0], n)).astype(int), 0, W - 1)
            ys = np.clip(np.rint(np.linspace(ends[a]["p"][1], ends[b]["p"][1], n)).astype(int), 0, H - 1)
            inside = float(occupied[ys, xs].mean())
            if inside < inside_frac:
                continue
            cand.append((g, a, b, inside))
    cand.sort()
    for g, a, b, inside in cand:
        if pairs.get(a, -1) >= 0 or pairs.get(b, -1) >= 0:
            continue
        pairs[a], pairs[b] = b, a
        out.append({"kind": "crossing under", "degree": 2,
                    "x": float((ends[a]["p"][0] + ends[b]["p"][0]) / 2),
                    "y": float((ends[a]["p"][1] + ends[b]["p"][1]) / 2),
                    "gap": round(g, 1), "inside_fraction": round(inside, 2),
                    "ends": [a, b], "pairs": [(a, b)]})
    return out


def hierarchy(segments, chains_, records, ends):
    """Parent, children, depth and Strahler order for each chain.

    Bifurcations make the vessels into a graph; which vessel is the parent is
    decided globally, not junction by junction. Each connected group is rooted
    at its thickest vessel and explored thickest-first, so every edge is
    oriented from the larger vessel to the smaller - the direction blood takes
    - and the numbering that follows is the order it reaches them. Rooting the
    whole group at once also means there are no cycles to break.
    """
    import heapq

    seg_chain = {}
    for ci, ch in enumerate(chains_):
        for seg, _ in ch:
            seg_chain[seg] = ci

    def calibre(ci):
        """Length-weighted median radius: what the vessel's calibre actually is,
        not the calibre of whichever end happened to meet a junction."""
        rs, ws = [], []
        for seg, _ in chains_[ci]:
            P = np.asarray(segments[seg]["points"], float)
            rs.append(float(segments[seg]["radius"]))
            ws.append(float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 1.0)
        order = np.argsort(rs)
        rs, ws = np.array(rs)[order], np.array(ws)[order]
        c = np.cumsum(ws)
        return float(rs[np.searchsorted(c, c[-1] / 2)]) if c[-1] > 0 else float(rs[0])

    def length(ci):
        return sum(float(np.hypot(*np.diff(np.asarray(segments[s]["points"], float), axis=0).T).sum())
                   for s, _ in chains_[ci])

    adj = {i: set() for i in range(len(chains_))}
    node_of = {}
    for rec in records:
        if rec["kind"] != "bifurcation" or "branch_end" not in rec:
            continue
        a_ = seg_chain.get(ends[rec["parent_end"]]["seg"])
        b_ = seg_chain.get(ends[rec["branch_end"]]["seg"])
        if a_ is None or b_ is None or a_ == b_:
            continue
        adj[a_].add(b_)
        adj[b_].add(a_)
        node_of[(min(a_, b_), max(a_, b_))] = (rec["x"], rec["y"])

    parent = [None] * len(chains_)
    branch_node = [None] * len(chains_)
    seen = set()
    for root in sorted(range(len(chains_)), key=lambda c: (-calibre(c), -length(c), c)):
        if root in seen:
            continue
        seen.add(root)
        heap = [(-calibre(root), root)]
        while heap:                       # thickest frontier vessel first
            _, cur = heapq.heappop(heap)
            for nb in sorted(adj[cur]):
                if nb in seen:
                    continue
                seen.add(nb)
                parent[nb] = cur
                branch_node[nb] = node_of.get((min(cur, nb), max(cur, nb)))
                heapq.heappush(heap, (-calibre(nb), nb))

    children = {i: [] for i in range(len(chains_))}
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)
    for p in children:
        children[p].sort(key=lambda c: (-calibre(c), -length(c)))
    labels = {}
    roots = sorted([i for i in range(len(chains_)) if parent[i] is None],
                   key=lambda c: (-calibre(c), -length(c)))
    order = []

    def walk(ci, label, depth):
        labels[ci] = label
        order.append((ci, depth))
        for k, c in enumerate(children[ci], start=1):
            walk(c, f"{label}.{k}", depth + 1)

    for n, r in enumerate(roots, start=1):
        walk(r, str(n), 0)

    strahler = {}

    def order_of(ci):
        if ci in strahler:
            return strahler[ci]
        kids = [order_of(c) for c in children[ci]]
        if not kids:
            strahler[ci] = 1
        else:
            m = max(kids)
            strahler[ci] = m + 1 if kids.count(m) > 1 else m
        return strahler[ci]

    for ci in range(len(chains_)):
        order_of(ci)
    return {"parent": parent, "children": children, "label": labels,
            "strahler": strahler, "order": order, "branch_node": branch_node,
            "calibre": {i: calibre(i) for i in range(len(chains_))}}


def _entry(seg, fwd):
    """Index of the end a segment is entered through when run this way."""
    return 2 * seg + (0 if fwd else 1)


def _exit(seg, fwd):
    return 2 * seg + (1 if fwd else 0)


def chains(segments, pairs):
    """Maximal chains of segments under the pairing: each chain is one vessel.

    Returns [[(segment index, runs forwards), ...], ...]; consecutive entries
    are connected, so concatenating their points (reversing where the flag is
    False) gives one continuous centreline.
    """
    used = [False] * len(segments)
    out = []
    order = sorted(range(len(segments)), key=lambda s: (-float(segments[s]["radius"]),
                                                        -len(segments[s]["points"]), s))
    for s0 in order:
        if used[s0]:
            continue
        used[s0] = True
        chain = [(s0, True)]
        # forwards from the tail
        while True:
            seg, fwd = chain[-1]
            nxt = pairs.get(_exit(seg, fwd), -1)
            if nxt < 0 or used[nxt // 2]:
                break
            ns = nxt // 2
            used[ns] = True
            chain.append((ns, nxt % 2 == 0))      # entered through its start -> forwards
        # backwards from the head
        while True:
            seg, fwd = chain[0]
            prv = pairs.get(_entry(seg, fwd), -1)
            if prv < 0 or used[prv // 2]:
                break
            ps = prv // 2
            used[ps] = True
            chain.insert(0, (ps, prv % 2 == 1))   # left through its end -> forwards
        out.append(chain)
    return out


def centreline(segments, chain):
    """One continuous centreline for a chain."""
    parts = []
    for seg, fwd in chain:
        P = np.asarray(segments[seg]["points"], float)
        parts.append(P if fwd else P[::-1])
    return np.concatenate(parts, 0)
