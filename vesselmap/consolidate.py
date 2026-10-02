"""Consolidation: one spline per vessel instead of one per segment.

A map from ``build_map`` / ``refine_map`` has one edge per *segment*: every
edge ends where it meets another, so a vessel that gives off five branches
is six edges, and a vessel the detector lost for a stretch (it dips out of
focus, runs under a wide vessel, or was found at two scales) is two or more
edges with free ends facing each other.  ``consolidate_map`` finds the
edges that are, in reality, one vessel and replaces each such group by a
single spline fitted to the whole vessel.

1. **Candidate continuations** between edge ends (every end is described by
   its position, outward direction, calibre, blur and contrast, measured
   over the last few tens of px):

   * *node*: two edges meeting at a node continue straight through it, with
     similar calibre and blur (blur is a depth cue: a vessel does not
     change depth abruptly).  At a bifurcation only an unambiguous pair
     qualifies: the through vessel must fit clearly better than any other
     pairing, so a symmetric fork (no through vessel) is left alone.
   * *gap*: a free end points at another end across a gap, within a cone,
     and the two ends continue each other.  The gap is bridged by a cubic
     Hermite curve that keeps both directions.
   * *overlap*: two free ends run past each other along the same course
     (a vessel found twice, shingled); the overlap is blended into one path.

2. **Matching.**  Every end may continue into at most one other end.  The
   pairing is a maximum-weight matching over all candidates, weighted by
   how well each pair fits, so a vessel is traced as a whole rather than
   greedily.  The matched pairs form chains of edges (cycles are cut at
   their weakest link).

3. **One spline per chain.**  The pieces, bridges and blends are
   concatenated and refitted as one edge.  Nodes where branches leave the
   vessel stay in the graph as nodes the vessel passes *through*
   (``info["through"]``): the branch still starts there, and the renderer
   keeps the node on the vessel's centreline.  Nodes left with nothing
   attached (sharp-free joints, the free ends of a bridged gap) disappear.

4. **Verification against the image.**  Segments and consolidated vessels
   are fitted jointly under the same priors, and every link is tested in
   its own neighbourhood.  A link is kept only if the single spline
   explains the image there no worse than the pieces did, allowing for the
   parameters the merge saves (the MDL cost of the removed ends).  A bridge
   across a gap must also carry its own evidence: the drop in NLL due to
   the bridged stretch alone must pay for its length.  Rejected links are
   removed and the matching is re-run (an end freed by a rejected link may
   continue elsewhere).

The result is a network with one edge per vessel.  ``to_digraph`` and
``to_segments`` still give the segment graph, with every segment carrying
the id of its vessel.  Like the rest of vesselmap, only the intensities of
one image are used.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .fit import MapConfig, _say, optimize, residual_image
from .image import Prepared, prepare
from .network import VesselNetwork, despike
from .render import NetworkModel


@dataclass
class ConsolidateConfig:
    max_turn_deg: float = 40.0      # largest change of direction at a continuation
    max_width_ratio: float = 1.8    # of the Gaussian-equivalent widths of the two ends
    max_blur_ratio: float = 2.2
    margin: float = 0.4             # cost margin of the through pair at a branch point
    max_gap: float = 60.0           # px, longest gap bridged
    min_gap_reach: float = 30.0     # px, a free end looks at least this far
    gap_factor: float = 8.0         # ... and up to this many calibres (r + s)
    cone_deg: float = 30.0          # the gap must lie ahead of both ends
    near_gap: float = 8.0           # px, gap between ends at two different nodes
    max_overlap: float = 60.0       # px, longest shingled overlap blended
    end_len: float = 30.0           # px of an edge describing its end
    iters: int = 150                # joint fit of segments / vessels
    rounds: int = 3                 # verify -> reject -> re-match passes
    calibre_window: int = 4         # profile knots (x 30 px) the calibre prior averages over
    link_pad: float = 12.0          # px of vessel on either side of a link it is tested on
    keep_frac: float = 0.05         # a link may lose this fraction of the zone's evidence
    switch_ratio: float = 1.6       # blur / width step along an edge that splits it
    switch_min_piece: float = 15.0  # px, shortest piece a split may leave
    verbose: bool = True


# ---------------------------------------------------------------- ends
def _oriented(smp, end):
    """Samples ordered from the given end inwards."""
    if end == 0:
        return smp
    return {k: v[::-1].copy() for k, v in smp.items()}


def _end_info(net: VesselNetwork, eid, end, smp, cc: ConsolidateConfig):
    """Position, direction, calibre, blur and contrast of an edge end.  The
    direction is that of the edge's body: the last few px next to a node
    are skipped, because a fitted edge often hooks into a node that sits a
    little off its course (the node is a compromise between the edges that
    meet there)."""
    o = _oriented(smp, end)
    xy, arc = o["xy"], smp["s_arc"]
    L = float(arc[-1])
    k = max(2, int(np.searchsorted(arc, min(cc.end_len, max(L, 1.0)))))
    r, s, a = (float(np.median(o[q][:k])) for q in ("r", "s", "a"))
    w = math.sqrt(s * s + 0.25 * r * r)
    skip = min(float(np.clip(1.5 * (r + s), 3.0, 10.0)), 0.25 * L)
    look = min(float(np.clip(4.0 * (r + s), 10.0, 25.0)), 0.5 * L)
    j0 = min(len(xy) - 2, int(np.searchsorted(arc, skip)))
    j1 = min(len(xy) - 1, max(j0 + 1, int(np.searchsorted(arc, skip + look))))
    t = xy[j0] - xy[j1]
    t = t / (np.linalg.norm(t) + 1e-9)
    e = net.edges[eid]
    return dict(key=(eid, end), eid=eid, end=end, node=e.u if end == 0 else e.v,
                far=e.v if end == 0 else e.u, xy=xy[0].copy(), t=t, r=r, s=s, a=a, w=w, L=L,
                skip=float(arc[j0]), body=xy[j0].copy())


def _cost(A, B, cc: ConsolidateConfig):
    """(cost, feasible) of A continuing into B: both outward tangents must
    be anti-parallel, and calibre and blur must match."""
    c = float(np.clip(-np.dot(A["t"], B["t"]), -1.0, 1.0))
    turn = math.degrees(math.acos(c))
    wr = max(A["w"], B["w"]) / max(min(A["w"], B["w"]), 1e-6)
    br = max(A["s"], B["s"]) / max(min(A["s"], B["s"]), 1e-6)
    cost = (turn / cc.max_turn_deg) ** 2 + (math.log(wr) / math.log(cc.max_width_ratio)) ** 2 + \
        0.5 * (math.log(br) / math.log(cc.max_blur_ratio)) ** 2
    ok = turn <= cc.max_turn_deg and wr <= cc.max_width_ratio and br <= cc.max_blur_ratio
    return cost, ok


def _within(pts, smp, slack=1.5, frac=0.8):
    """Do most of pts lie inside the footprint of the sampled edge?"""
    if len(pts) == 0:
        return False
    d, j = cKDTree(smp["xy"]).query(pts)
    return float(np.mean(d <= smp["r"][j] + 0.7 * smp["s"][j] + slack)) >= frac


def end_infos(net: VesselNetwork, cc: ConsolidateConfig, samples):
    return {(eid, end): _end_info(net, eid, end, samples[eid], cc)
            for eid in net.edges for end in (0, 1)}


def candidates(net: VesselNetwork, cc: ConsolidateConfig, samples=None, ends=None):
    """All candidate continuations: list of dict(a, b, kind, cost) with a, b
    edge-end keys (eid, end)."""
    samples = samples or {eid: net.sample(eid, 0.7) for eid in net.edges}
    ends = ends or end_infos(net, cc, samples)
    deg = net.degrees()
    out = {}

    def add(A, B, kind, cost):
        key = tuple(sorted((A["key"], B["key"])))
        if key not in out or cost < out[key]["cost"]:
            out[key] = dict(a=A["key"], b=B["key"], kind=kind, cost=float(cost))

    # (1) straight through a node
    for nid in net.nodes:
        if net.nodes[nid].fixed_kind:
            continue
        inc = [ends[k] for k in net.incident(nid)]
        if len(inc) < 2:
            continue
        raw = {}
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                raw[(i, j)] = _cost(inc[i], inc[j], cc)
        for (i, j), (c, ok) in raw.items():
            A, B = inc[i], inc[j]
            if not ok or A["eid"] == B["eid"] or A["far"] == B["far"]:
                continue
            # the through pair must beat every other pairing at the node
            alt = [raw[tuple(sorted((x, y)))][0] for x in (i, j) for y in range(len(inc))
                   if y not in (i, j)]
            if alt and min(alt) - c < cc.margin:
                continue
            add(A, B, "node", c)
    # (2) across a gap, (3) shingled overlap: from free ends, and between
    # ends at two different nodes a few px apart (a junction found twice)
    keys = list(ends)
    P = np.array([ends[k]["xy"] for k in keys])
    tree = cKDTree(P)
    cone = math.cos(math.radians(cc.cone_deg))
    for ka in keys:
        A = ends[ka]
        free = deg[A["node"]] == 1 and net.node_kind(A["node"], 1) == "endpoint"
        if free:
            R = min(cc.max_gap, max(cc.min_gap_reach, cc.gap_factor * (A["r"] + A["s"])))
        elif not net.nodes[A["node"]].fixed_kind and net.node_kind(A["node"]) != "border":
            R = max(cc.near_gap, 2.0 * (A["r"] + A["s"]))
        else:
            continue
        for j in tree.query_ball_point(A["xy"], max(R, cc.max_overlap)):
            B = ends[keys[j]]
            if B["eid"] == A["eid"] or B["node"] == A["node"] or B["node"] == A["far"]:
                continue
            if net.nodes[B["node"]].fixed_kind or net.node_kind(B["node"]) == "border":
                continue
            c, ok = _cost(A, B, cc)
            if not ok:
                continue
            d = B["xy"] - A["xy"]
            dist = float(np.linalg.norm(d))
            along = float(np.dot(d, A["t"]))
            b_free = deg[B["node"]] == 1
            if not free and dist > max(cc.near_gap, 2.0 * (B["r"] + B["s"])):
                continue
            if dist < 1.0 or along > 0:
                if dist > R:
                    continue
                if dist >= 1.0:
                    dh = d / dist
                    if np.dot(dh, A["t"]) < cone or np.dot(-dh, B["t"]) < cone:
                        continue
                add(A, B, "gap", c + 0.3 + (dist / R) ** 2)
            elif free and b_free and -along <= cc.max_overlap:
                lateral = abs(float(d[0] * A["t"][1] - d[1] * A["t"][0]))
                if lateral > max(2.0, A["r"] + B["r"]):
                    continue
                ov = -along + 2.0
                sa, sb = samples[A["eid"]], samples[B["eid"]]
                oa = _oriented(sa, A["end"])
                ob = _oriented(sb, B["end"])
                na = int(np.searchsorted(sa["s_arc"], ov))
                nb = int(np.searchsorted(sb["s_arc"], ov))
                if not (_within(oa["xy"][:max(na, 1)], sb) and _within(ob["xy"][:max(nb, 1)], sa)):
                    continue
                add(A, B, "overlap", c + 0.3 + (-along / cc.max_overlap) ** 2)
    return list(out.values())


# ---------------------------------------------------------------- switches
def _change_point(y, m):
    """Index splitting y (n x d) into two runs of at least m samples with
    the largest drop of the squared error around the run means, and that
    squared error."""
    n = len(y)
    if n < 2 * m + 1:
        return None, None
    c1 = np.cumsum(y, 0)
    c2 = np.cumsum(y * y, 0)
    i = np.arange(m, n - m)
    n1, n2 = i[:, None].astype(float), (n - i)[:, None].astype(float)
    s1, q1 = c1[i - 1], c2[i - 1]
    s2, q2 = c1[-1] - s1, c2[-1] - q1
    sse = (q1 - s1 ** 2 / n1 + q2 - s2 ** 2 / n2).sum(1)
    k = int(np.argmin(sse))
    return int(i[k]), float(sse[k])


def _is_step(y, i, m, ratio, sharp=0.5):
    """Is there an abrupt step at i: the medians of the m samples on either
    side (a few samples around i left out) differ by more than `ratio`, and
    a step explains y much better than a linear trend does?  A vessel whose
    blur drifts as it changes depth is a ramp, not a step."""
    g = 3
    a = np.median(y[max(0, i - g - m):max(1, i - g)], 0)
    b = np.median(y[i + g:i + g + m], 0)
    if np.abs(b - a).max() < math.log(ratio):
        return False
    x = np.arange(len(y), dtype=float)
    X = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse_lin = float(((y - X @ coef) ** 2).sum())
    step = np.where(x[:, None] < i, y[:i].mean(0), y[i:].mean(0))
    sse_step = float(((y - step) ** 2).sum())
    return sse_step < sharp * sse_lin


def split_switches(net: VesselNetwork, cc: ConsolidateConfig, max_splits=3):
    """Cut edges whose blur or calibre steps along them.  Such an edge
    usually runs along one vessel and then switches to another at a
    different depth (where two vessels cross or touch).  Consolidating it
    whole would carry the switch into a longer spline; cut at the step,
    each piece can continue into its own vessel.  In place; returns the
    number of cuts."""
    cuts = 0
    todo = list(net.edges)
    while todo:
        eid = todo.pop()
        if eid not in net.edges:
            continue
        smp = net.sample(eid, 1.0)
        m = int(cc.switch_min_piece)
        y = np.stack([np.log(smp["s"]), 0.5 * np.log(smp["s"] ** 2 + 0.25 * smp["r"] ** 2)], 1)
        i, _ = _change_point(y, m)
        if i is None or not _is_step(y, i, m, cc.switch_ratio):
            continue
        n_before = len(net.edges)
        nid = net.split_edge(eid, smp["xy"][i])
        if len(net.edges) == n_before:
            continue
        net.nodes[nid].fixed_kind = None
        cuts += 1
        if cuts >= max_splits * max(1, n_before):
            break
        todo += [k for k, _ in net.incident(nid)]
    return cuts


# ---------------------------------------------------------------- chains
def match(cands, w0=4.0):
    """Maximum-weight matching of edge ends: each end continues into at
    most one other end."""
    import networkx as nx
    G = nx.Graph()
    for c in cands:
        G.add_edge(c["a"], c["b"], weight=w0 - c["cost"], c=c)
    return [G.edges[a, b]["c"] for a, b in nx.max_weight_matching(G)]


def chains(net: VesselNetwork, links):
    """Group edges joined by links into ordered chains.  Returns a list of
    (steps, links) with steps [(eid, reversed)] and links[i] joining steps
    i and i + 1.  Cycles, and chains that would start and end at the same
    node, are cut at their most expensive link."""
    links = list(links)
    while True:
        at = {}
        for L in links:
            at[L["a"]] = L
            at[L["b"]] = L
        seen, out, cut = set(), [], None

        def walk(eid, entry):
            steps, ls = [], []
            while True:
                seen.add(eid)
                steps.append((eid, entry == 1))
                L = at.get((eid, 1 - entry))
                if L is None:
                    return steps, ls
                nxt = L["b"] if L["a"] == (eid, 1 - entry) else L["a"]
                if nxt[0] in seen:
                    ls.append(L)
                    return steps, ls           # closed a cycle
                ls.append(L)
                eid, entry = nxt

        starts = [(eid, end) for eid in net.edges for end in (0, 1) if (eid, end) not in at]
        for eid, end in starts:
            if eid not in seen:
                out.append(walk(eid, end))
        for eid in net.edges:
            if eid not in seen:                 # a cycle
                steps, ls = walk(eid, 0)
                cut = max(ls, key=lambda L: L["cost"])
                break
        if cut is None:
            for steps, ls in out:
                if len(steps) < 2:
                    continue
                e0, e1 = net.edges[steps[0][0]], net.edges[steps[-1][0]]
                u = e0.v if steps[0][1] else e0.u
                v = e1.u if steps[-1][1] else e1.v
                if u == v:
                    cut = max(ls, key=lambda L: L["cost"])
                    break
        if cut is None:
            return out
        links = [L for L in links if L is not cut]


def _hermite(p0, t0, p1, t1, step=0.7):
    """Interior points of a cubic Hermite curve from p0 (direction t0) to p1
    (direction t1)."""
    D = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(D / step)) + 1)
    s = np.linspace(0.0, 1.0, n)[1:-1, None]
    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * p0 + h10 * D * t0 + h01 * p1 + h11 * D * t1


def _resample(o, n):
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(o["xy"], axis=0), axis=1))]
    q = np.linspace(0.0, arc[-1], n)
    out = {"xy": np.stack([np.interp(q, arc, o["xy"][:, i]) for i in range(2)], 1)}
    for k in ("r", "s", "a"):
        out[k] = np.interp(q, arc, o[k])
    return out


def _trim(o, length, at_end):
    """Drop `length` px from the end (at_end) or the start of a path."""
    if length <= 0 or len(o["xy"]) < 3:
        return o
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(o["xy"], axis=0), axis=1))]
    if at_end:
        n = max(2, int(np.searchsorted(arc, arc[-1] - length, side="right")))
        return {k: v[:n] for k, v in o.items()}
    n = min(len(arc) - 2, int(np.searchsorted(arc, length)))
    return {k: v[n:] for k, v in o.items()}


def _bridge(parts, o, A, B, cc):
    """parts (ending at end A) and o (starting at end B) joined by a cubic
    Hermite curve between the bodies of the two edges, along their body
    directions; profiles interpolated between the two ends' medians."""
    parts = _trim(parts, A["skip"], at_end=True)
    o = _trim(o, B["skip"], at_end=False)
    pa, pb = parts["xy"][-1], o["xy"][0]
    br = _hermite(pa, A["t"], pb, -B["t"])
    t = np.linspace(0, 1, len(br) + 2)[1:-1]
    bridge = {"xy": br, **{k: A[k] * (1 - t) + B[k] * t for k in ("r", "s", "a")}}
    zone = (pa.copy(), pb.copy())
    return {k: np.concatenate([parts[k], bridge[k], o[k]]) for k in parts}, zone


def build(net: VesselNetwork, groups, cc: ConsolidateConfig, samples=None, ends=None):
    """A copy of net with every chain replaced by one edge.  Returns
    (network, records) with one record per link: the vessel it is in, its
    kind, and the points where its zone starts and ends."""
    samples = samples or {eid: net.sample(eid, 0.7) for eid in net.edges}
    ends = ends or end_infos(net, cc, samples)
    out = net.copy()
    records, junction = [], {}
    for steps, ls in groups:
        if len(steps) < 2:
            continue
        parts = None
        zones, jn = [], set()
        for i, (eid, rev) in enumerate(steps):
            o = _oriented(samples[eid], 1 if rev else 0)
            o = {k: o[k] for k in ("xy", "r", "s", "a")}
            if parts is None:
                parts = o
                continue
            L = ls[i - 1]
            prev = steps[i - 1]
            e_prev, e_cur = net.edges[prev[0]], net.edges[eid]
            n_prev = e_prev.u if prev[1] else e_prev.v          # node where prev ends
            n_cur = e_cur.v if rev else e_cur.u                  # node where cur starts
            A = ends[(prev[0], 0 if prev[1] else 1)]
            B = ends[(eid, 1 if rev else 0)]
            if L["kind"] in ("node", "gap"):
                parts, zone = _bridge(parts, o, A, B, cc)
                zones.append(zone)
                jn |= {n_prev, n_cur}
            else:                                                # overlap: blend
                ia = len(parts["xy"]) - 1 - int(np.argmin(np.linalg.norm(
                    parts["xy"][::-1][:int(cc.max_overlap / 0.5)] - o["xy"][0], axis=1)))
                ib = int(np.argmin(np.linalg.norm(
                    o["xy"][:int(cc.max_overlap / 0.5)] - parts["xy"][-1], axis=1)))
                if ia >= len(parts["xy"]) - 1 or ib == 0:
                    ia, ib = len(parts["xy"]) - 1, 0
                A = {k: parts[k][ia:] for k in parts}
                B = {k: o[k][:ib + 1] for k in o}
                la = float(np.linalg.norm(np.diff(A["xy"], axis=0), axis=1).sum()) if len(A["xy"]) > 1 else 0.0
                lb = float(np.linalg.norm(np.diff(B["xy"], axis=0), axis=1).sum()) if len(B["xy"]) > 1 else 0.0
                n = max(2, int(math.ceil(0.5 * (la + lb) / 0.7)) + 1)
                A = _resample(A, n) if len(A["xy"]) > 1 else {k: np.repeat(A[k][:1], n, 0) for k in A}
                B = _resample(B, n) if len(B["xy"]) > 1 else {k: np.repeat(B[k][:1], n, 0) for k in B}
                w = np.linspace(0.0, 1.0, n)
                w = w * w * (3 - 2 * w)
                blend = {k: (A[k] * (1 - w[:, None]) + B[k] * w[:, None]) if A[k].ndim == 2
                         else A[k] * (1 - w) + B[k] * w for k in A}
                zones.append((blend["xy"][0].copy(), blend["xy"][-1].copy()))
                parts = {k: np.concatenate([parts[k][:ia], blend[k], o[k][ib + 1:]]) for k in parts}
                jn |= {n_prev, n_cur}
        e0, e1 = net.edges[steps[0][0]], net.edges[steps[-1][0]]
        u = e0.v if steps[0][1] else e0.u
        v = e1.u if steps[-1][1] else e1.v
        info = _merged_info(net, [eid for eid, _ in steps])
        spacing = min(net.edges[eid].info.get("spacing", 12.0) for eid, _ in steps)
        for eid, _ in steps:
            out.remove_edge(eid, drop_orphans=False)
        # the pieces are fitted splines and the bridges smooth: reproduce them
        # closely (explicit despike, then a faithful fit), as join_edges does
        xy, r, s, a = despike(parts["xy"], parts["r"], parts["s"], parts["a"])
        new = out.add_edge_dense(xy, r, s, a, u=u, v=v, spacing=spacing, info=info,
                                 faithful=True)
        for n in jn - {u, v}:
            junction.setdefault(n, []).append(new)
        for L, (z0, z1) in zip(ls, zones):
            records.append(dict(vessel=new, kind=L["kind"], cost=L["cost"], a=L["a"], b=L["b"],
                                xy0=z0, xy1=z1))
    # nodes on the joined vessels: keep those that still join something
    thru = {}
    for n, vs in junction.items():
        vs = [x for x in vs if x in out.edges]
        if n not in out.nodes or not vs:
            continue
        if out.incident(n) or len(vs) >= 2:
            for x in vs:
                thru.setdefault(x, []).append(n)
        else:
            del out.nodes[n]
            out._touch()
    for x, ns in thru.items():
        out.edges[x].info["through"] = ns
    out._touch()
    for n in {n for ns in thru.values() for n in ns}:
        for k, _ in out.incident(n):
            out.sync_ends(k)
    out.snap_through_nodes()
    return out, records


def _merged_info(net, eids):
    info = {}
    bands = [net.edges[e].info.get("band") for e in eids]
    bands = [b for b in bands if b]
    if bands:
        info["band"] = [min(b[0] for b in bands), max(b[1] for b in bands)]
    through = []
    for e in eids:
        through += net.edges[e].through
    if through:
        info["through"] = list(dict.fromkeys(through))
    src = []
    for e in eids:
        src += net.edges[e].info.get("consolidated_from", [int(e)])
    info["consolidated_from"] = src
    return info


# ---------------------------------------------------------------- testing
def _fit(net, P, cfg: MapConfig, cc: ConsolidateConfig, iters):
    priors = dict(cfg.priors(), calibre_window=cc.calibre_window)
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=cfg.bg_spacing, ref=net)
    optimize(model, iters, cfg.lr_pos * 0.5, cfg.lr_prof, cfg.lr_bg, cfg.rebuild_every,
             priors=priors, track=cfg.anchor())
    model.write_back()
    return model


def _zone_mask(net, eid, i0, i1, smp, shape, pad):
    import cv2
    arc = smp["s_arc"]
    j0 = int(np.searchsorted(arc, arc[i0] - pad))
    j1 = int(np.searchsorted(arc, arc[i1] + pad))
    xy = smp["xy"][j0:max(j1, j0 + 2)]
    R = float((smp["r"][j0:j1 + 1] + 2.5 * smp["s"][j0:j1 + 1]).max() + 3.0)
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(xy * 4).astype(np.int32)], False, 1,
                  thickness=int(2 * math.ceil(R) + 1), shift=2)
    return m.astype(bool)


def verify(trial, records, model, R0, P: Prepared, cfg: MapConfig, cc: ConsolidateConfig):
    """Test each link in its own neighbourhood; returns (passed, failed)
    lists of records, each annotated with its dnll, tolerance and, for
    gaps, the bridge's own gain."""
    R1 = residual_image(model)
    W = P.weight
    gs, arc_m = model.sample_gains()
    with_edges = {eid: k for k, eid in enumerate(model.eids)}
    C = model.samples()[0].detach().numpy()
    passed, failed = [], []
    smp_cache = {}
    for rec in records:
        eid = rec["vessel"]
        if eid not in trial.edges:
            continue
        smp = smp_cache.setdefault(eid, trial.sample(eid, 1.0))
        i0 = int(np.argmin(np.linalg.norm(smp["xy"] - rec["xy0"], axis=1)))
        i1 = int(np.argmin(np.linalg.norm(smp["xy"] - rec["xy1"], axis=1)))
        i0, i1 = min(i0, i1), max(i0, i1)
        w = float(np.median(smp["r"][i0:i1 + 1] + smp["s"][i0:i1 + 1]))
        pad = max(cc.link_pad, 3.0 * w)
        mask = _zone_mask(trial, eid, i0, i1, smp, P.shape, pad)
        npix = max(int(mask.sum()), 2)
        dnll = 0.5 * float((W[mask] * (R1[mask] ** 2 - R0[mask] ** 2)).sum())
        # the evidence the vessel carries in the zone (its NLL drop there)
        k = with_edges[eid]
        a0, a1 = model.samp_first[k], model.samp_last[k] + 1
        m0 = a0 + int(np.argmin(np.linalg.norm(C[a0:a1] - rec["xy0"], axis=1)))
        m1 = a0 + int(np.argmin(np.linalg.norm(C[a0:a1] - rec["xy1"], axis=1)))
        m0, m1 = min(m0, m1), max(m0, m1)
        z = (arc_m[a0:a1] >= arc_m[m0] - pad) & (arc_m[a0:a1] <= arc_m[m1] + pad)
        zone_gain = float(gs[a0:a1][z].sum())
        # the merge saves one pair of ends (4 dof, see fit.n_params).  At a
        # node (or an overlap) both pieces already explain the image, so the
        # NLL barely tells one vessel from two; there the test only has to
        # catch a merge that clearly hurts the fit: the single spline must
        # keep all but keep_frac of the evidence in the zone.  Across a gap
        # nothing explained the image before, so the data decide strictly.
        tol = cfg.penalty_scale * 0.5 * 4.0 * math.log(npix)
        if rec["kind"] != "gap":
            tol = max(tol, cc.keep_frac * zone_gain)
        ok = dnll <= tol
        rec.update(dnll=dnll, tol=tol, zone_gain=zone_gain)
        if rec["kind"] == "gap":
            Lb = max(float(arc_m[m1] - arc_m[m0]), 1.0)
            gain = float(gs[m0:m1 + 1].sum())
            pen = cfg.penalty_scale * 0.5 * (2.0 * Lb / 15.0) * math.log(npix)
            rec.update(bridge_len=Lb, bridge_gain=gain, bridge_pen=pen)
            ok = ok and gain >= max(pen, cfg.min_gain_per_px * Lb)
        (passed if ok else failed).append(rec)
    return passed, failed


# ---------------------------------------------------------------- driver
def consolidate_map(intensity: np.ndarray, net: VesselNetwork, cfg: MapConfig | None = None,
                    cc: ConsolidateConfig | None = None,
                    prepared: Prepared | None = None) -> VesselNetwork:
    """One spline per vessel (see the module docstring).  `net` is a map of
    the same image (from build_map / refine_map); it is not modified.
    Returns the consolidated network."""
    cfg = cfg or MapConfig()
    cc = cc or ConsolidateConfig()
    P = prepared or prepare(intensity)
    t0 = time.time()
    log = (lambda *a: _say(cfg, f"[{time.time() - t0:6.0f}s]", *a)) if cc.verbose else (lambda *a: None)
    base = net.to_segments() if net.has_through() else net.copy()
    n0 = len(base.edges)
    n_cut = split_switches(base, cc)
    if n_cut:
        log(f"split {n_cut} edges where their blur or calibre steps (switches between vessels)")
    m0 = _fit(base, P, cfg, cc, cc.iters)
    R0 = residual_image(m0)
    nll0 = float(m0.data_nll(m0.predict()).detach())
    samples = {eid: base.sample(eid, 0.7) for eid in base.edges}
    ends = end_infos(base, cc, samples)
    cands = candidates(base, cc, samples, ends)
    kinds = lambda L: {k: sum(1 for x in L if x["kind"] == k) for k in ("node", "gap", "overlap")}
    log(f"consolidate: {n0} edges, {len(cands)} candidate links {kinds(cands)}")
    rejected = set()
    trial, model, passed, history = base, m0, [], []
    for rnd in range(cc.rounds):
        pool = [c for c in cands if (c["a"], c["b"]) not in rejected]
        links = match(pool)
        groups = chains(base, links)
        trial, records = build(base, groups, cc, samples, ends)
        model = _fit(trial, P, cfg, cc, cc.iters)
        passed, failed = verify(trial, records, model, R0, P, cfg, cc)
        history.append(dict(links=len(records), passed=kinds(passed), failed=kinds(failed)))
        log(f"round {rnd}: {len(records)} links matched; passed {kinds(passed)}, "
            f"failed {kinds(failed)}; {len(trial.edges)} edges")
        if cc.verbose and failed:
            log("   failed (kind, dnll/tol, bridge gain/pen): " + ", ".join(
                f"({r['kind']}, {r['dnll'] / r['tol']:.1f}"
                + (f", {r['bridge_gain']:.0f}/{r['bridge_pen']:.0f})" if "bridge_gain" in r else ")")
                for r in failed))
        if not failed:
            break
        rejected |= {(r["a"], r["b"]) for r in failed} | {(r["b"], r["a"]) for r in failed}
    else:
        keep = {(r["a"], r["b"]) for r in passed}
        links = [L for L in links if (L["a"], L["b"]) in keep]
        trial, records = build(base, chains(base, links), cc, samples, ends)
        model = _fit(trial, P, cfg, cc, cc.iters)
        passed = records
    gains, _ = model.edge_gains()
    for k, eid in enumerate(model.eids):
        L = max(trial.length(eid), 1.0)
        trial.edges[eid].info.update(gain=float(gains[k]), gain_per_px=float(gains[k] / L))
    trial.orient_structural()
    nll1 = float(model.data_nll(model.predict()).detach())
    multi = [e for e in trial.edges.values() if len(e.info.get("consolidated_from", [])) > 1]
    trial.meta["consolidation"] = dict(
        seconds=round(time.time() - t0, 1), edges_before=n0, switch_cuts=n_cut,
        edges_after=len(trial.edges),
        vessels_merged=len(multi), candidates=kinds(cands), links=kinds(passed),
        rounds=history, nll_segments=nll0, nll_vessels=nll1)
    trial.meta["final_nll"] = nll1
    log(f"consolidated {n0} -> {len(trial.edges)} edges ({len(multi)} vessels from several "
        f"pieces); links {kinds(passed)}; NLL {nll0:.0f} -> {nll1:.0f}")
    return trial
