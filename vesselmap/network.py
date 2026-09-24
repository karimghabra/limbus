"""The vessel network: a graph whose edges are parametric vessel splines.

Nodes are the places where vessel segments meet or stop:

* ``endpoint``    - a vessel fades out / leaves the focal volume (degree 1)
* ``border``      - a vessel leaves the image (degree 1, at the image edge)
* ``bifurcation`` - three segments meet (degree 3)
* ``junction``    - four or more segments meet (degree >= 4)

Crossings of vessels lying at different depths are NOT nodes: the two
vessels are separate edges that overlap, and their optical densities add.
They are listed in ``crossings()`` for display.

Each edge carries

* ``ctrl`` - control points (n, 2) of a clamped cubic B-spline centreline
  (x, y in pixels).  ``ctrl[0]`` / ``ctrl[-1]`` coincide with the edge's
  end nodes.
* ``r`` / ``s`` / ``a`` - control values of splines along the edge for the
  lumen half-width (px), the blur (Gaussian std, px; focus) and the peak
  optical density (contrast), so all three may vary along the vessel.

The graph is exported as a ``networkx.DiGraph``.  Without flow data the
direction of flow is not observable from a still image; edges are oriented
by a purely structural convention (from wider to narrower vessels, away
from the widest vessel of each connected component) and every edge records
``orientation = "structural"`` so that a later, independent measurement can
flip it with ``set_direction``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import spline as sp

POS_SPACING_MIN = 5.0
POS_SPACING_MAX = 20.0
PROFILE_SPACING = 30.0
R_MIN, S_MIN, A_MIN = 0.3, 0.6, 1e-3


@dataclass
class Node:
    x: float
    y: float
    fixed_kind: str | None = None   # e.g. "border" pinned by the user

    @property
    def xy(self):
        return np.array([self.x, self.y], float)


@dataclass
class Edge:
    u: int
    v: int
    ctrl: np.ndarray
    r: np.ndarray
    s: np.ndarray
    a: np.ndarray
    info: dict = field(default_factory=dict)

    def copy(self):
        return Edge(self.u, self.v, self.ctrl.copy(), self.r.copy(), self.s.copy(),
                    self.a.copy(), dict(self.info))


def pos_spacing_for(width: float) -> float:
    """Upper bound on the control-point spacing of a centreline from its
    calibre (r + s): thin vessels may be tortuous on a scale of a few px."""
    return float(np.clip(1.5 * width, POS_SPACING_MIN, POS_SPACING_MAX))


def spacing_for_path(xy, max_spacing, tol=0.6):
    """Largest control-point spacing (<= max_spacing) whose spline follows
    the (lightly smoothed) path within tol px."""
    from scipy.ndimage import gaussian_filter1d
    xy = np.asarray(xy, float)
    arc = sp.arclength(xy)
    L = float(arc[-1])
    if L < 3 * POS_SPACING_MIN or len(xy) < 5:
        return float(max_spacing)
    m = int(np.ceil(L / 0.5)) + 1
    q = np.linspace(0, L, m)
    u = np.stack([np.interp(q, arc, xy[:, 0]), np.interp(q, arc, xy[:, 1])], 1)
    u = gaussian_filter1d(u, 3.0, axis=0, mode="nearest")
    cands = [c for c in (24.0, 18.0, 14.0, 11.0, 8.0, 6.0, POS_SPACING_MIN) if c <= max_spacing]
    for c in cands or [max_spacing]:
        n = sp.n_ctrl_for_length(L, c, minimum=4)
        ctrl = sp.fit_ctrl(u, n, smooth=2e-4, pin_ends=True)
        err = np.linalg.norm(sp.design(n, m) @ ctrl - u, axis=1).max()
        if err <= tol:
            return float(c)
    return float(POS_SPACING_MIN)


class VesselNetwork:
    def __init__(self, shape):
        self.shape = tuple(int(v) for v in shape[:2])
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self._nid = 0
        self._eid = 0
        self._adj = None
        self.background = None          # coarse log-background grid
        self.bg_spacing = None
        self.meta: dict = {}

    # ------------------------------------------------------------------ basic
    def copy(self) -> "VesselNetwork":
        n = VesselNetwork(self.shape)
        n.nodes = {k: Node(v.x, v.y, v.fixed_kind) for k, v in self.nodes.items()}
        n.edges = {k: e.copy() for k, e in self.edges.items()}
        n._nid, n._eid = self._nid, self._eid
        n._adj = None
        n.background = None if self.background is None else self.background.copy()
        n.bg_spacing = self.bg_spacing
        n.meta = json.loads(json.dumps(self.meta))
        return n

    def add_node(self, x, y) -> int:
        nid = self._nid
        self._nid += 1
        self.nodes[nid] = Node(float(x), float(y))
        self._touch()
        return nid

    def _touch(self):
        self._adj = None

    def _adjacency(self):
        adj = getattr(self, "_adj", None)
        if adj is None:
            adj = {k: [] for k in self.nodes}
            for eid, e in self.edges.items():
                adj.setdefault(e.u, []).append((eid, 0))
                adj.setdefault(e.v, []).append((eid, 1))
            self._adj = adj
        return adj

    def incident(self, nid):
        """[(eid, end)] with end 0 if the edge starts at nid, 1 if it ends."""
        return list(self._adjacency().get(nid, []))

    def degrees(self) -> dict:
        adj = self._adjacency()
        return {k: len(adj.get(k, [])) for k in self.nodes}

    def node_kind(self, nid, deg=None, margin=3.0) -> str:
        n = self.nodes[nid]
        if n.fixed_kind:
            return n.fixed_kind
        if deg is None:
            deg = len(self.incident(nid))
        h, w = self.shape
        if deg <= 1:
            if n.x < margin or n.y < margin or n.x > w - 1 - margin or n.y > h - 1 - margin:
                return "border"
            return "endpoint"
        if deg == 2:
            return "joint"
        if deg == 3:
            return "bifurcation"
        return "junction"

    def sync_ends(self, eid):
        e = self.edges[eid]
        e.ctrl[0] = self.nodes[e.u].xy
        e.ctrl[-1] = self.nodes[e.v].xy

    # --------------------------------------------------------------- sampling
    def length(self, eid) -> float:
        e = self.edges[eid]
        xy = sp.design(len(e.ctrl), 64) @ e.ctrl
        L0 = sp.arclength(xy)[-1]
        n = sp.n_samples_for_length(L0, 1.0)
        xy = sp.design(len(e.ctrl), n) @ e.ctrl
        return float(sp.arclength(xy)[-1])

    def sample(self, eid, spacing: float = 0.7) -> dict:
        """Dense samples of an edge: xy, unit tangent, arclength, r, s, a."""
        self.sync_ends(eid)
        e = self.edges[eid]
        n = sp.n_samples_for_length(self.length(eid), spacing)
        xy = sp.design(len(e.ctrl), n) @ e.ctrl
        d1 = sp.design(len(e.ctrl), n, 1) @ e.ctrl
        tan = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-9)
        Bp = sp.design(len(e.r), n)
        return dict(xy=xy.astype(float), tan=tan, s_arc=sp.arclength(xy),
                    r=np.maximum(Bp @ e.r, R_MIN), s=np.maximum(Bp @ e.s, S_MIN),
                    a=np.maximum(Bp @ e.a, A_MIN))

    def edge_stats(self, eid) -> dict:
        smp = self.sample(eid, 1.0)
        xy = smp["xy"]
        L = float(smp["s_arc"][-1])
        chord = float(np.linalg.norm(xy[-1] - xy[0]))
        # curvature from the tangent angle
        ang = np.unwrap(np.arctan2(smp["tan"][:, 1], smp["tan"][:, 0]))
        ds = np.maximum(np.diff(smp["s_arc"]), 1e-6)
        kappa = np.abs(np.diff(ang)) / ds
        e = self.edges[eid]
        return dict(length=L, chord=chord, tortuosity=L / max(chord, 1e-6),
                    diameter=float(2 * smp["r"].mean()),
                    diameter_min=float(2 * smp["r"].min()),
                    diameter_max=float(2 * smp["r"].max()),
                    blur=float(smp["s"].mean()), contrast=float(smp["a"].mean()),
                    mean_curvature=float(kappa.mean()) if len(kappa) else 0.0,
                    max_curvature=float(kappa.max()) if len(kappa) else 0.0,
                    gain=float(e.info.get("gain", np.nan)))

    # ------------------------------------------------------------ construction
    def add_edge_dense(self, xy, r, s, a, u=None, v=None, spacing=None, info=None) -> int:
        """Add an edge from a dense ordered polyline with per-point profile
        values.  New end nodes are created unless u / v are given (in which
        case the polyline ends are moved onto those nodes)."""
        xy = np.asarray(xy, float)
        if u is None:
            u = self.add_node(*xy[0])
        if v is None:
            v = self.add_node(*xy[-1])
        xy = xy.copy()
        xy[0] = self.nodes[u].xy
        xy[-1] = self.nodes[v].xy
        width = float(np.median(np.asarray(r)) + np.median(np.asarray(s)))
        spacing = spacing or spacing_for_path(xy, pos_spacing_for(width))
        ctrl, pr, ps, pa = _fit_edge_params(xy, r, s, a, spacing)
        eid = self._eid
        self._eid += 1
        info = dict(info or {})
        info["spacing"] = float(spacing)
        self.edges[eid] = Edge(u, v, ctrl, pr, ps, pa, info)
        self._touch()
        return eid

    def refit_edge(self, eid, xy, r, s, a, spacing=None):
        e = self.edges[eid]
        spacing = spacing or e.info.get("spacing") or pos_spacing_for(float(np.median(r) + np.median(s)))
        xy = np.asarray(xy, float).copy()
        xy[0] = self.nodes[e.u].xy
        xy[-1] = self.nodes[e.v].xy
        e.ctrl, e.r, e.s, e.a = _fit_edge_params(xy, r, s, a, spacing)
        e.info["spacing"] = float(spacing)

    def remove_edge(self, eid, drop_orphans=True):
        e = self.edges.pop(eid)
        self._touch()
        if drop_orphans:
            for n in (e.u, e.v):
                if n in self.nodes and not self.incident(n):
                    del self.nodes[n]
                    self._touch()

    def reverse_edge(self, eid):
        e = self.edges[eid]
        self._touch()
        e.u, e.v = e.v, e.u
        e.ctrl = e.ctrl[::-1].copy()
        e.r, e.s, e.a = e.r[::-1].copy(), e.s[::-1].copy(), e.a[::-1].copy()

    def set_direction(self, eid, source="user"):
        """Mark an edge's current u->v direction as coming from `source`."""
        self.edges[eid].info["orientation"] = source

    def split_edge(self, eid, xy_point, min_piece=2.0):
        """Split an edge at the point closest to xy_point.  Returns the new
        node id (or an existing end node if the split point is at an end)."""
        e = self.edges[eid]
        smp = self.sample(eid, 0.5)
        d = np.linalg.norm(smp["xy"] - np.asarray(xy_point, float), axis=1)
        i = int(np.argmin(d))
        L = smp["s_arc"][-1]
        if smp["s_arc"][i] < min_piece:
            return e.u
        if L - smp["s_arc"][i] < min_piece:
            return e.v
        nid = self.add_node(*smp["xy"][i])
        info = dict(e.info)
        sl1, sl2 = slice(0, i + 1), slice(i, None)
        u, v = e.u, e.v
        spacing = e.info.get("spacing")
        self.remove_edge(eid, drop_orphans=False)
        self.add_edge_dense(smp["xy"][sl1], smp["r"][sl1], smp["s"][sl1], smp["a"][sl1],
                            u=u, v=nid, spacing=spacing, info=info)
        self.add_edge_dense(smp["xy"][sl2], smp["r"][sl2], smp["s"][sl2], smp["a"][sl2],
                            u=nid, v=v, spacing=spacing, info=info)
        return nid

    def merge_nodes(self, keep, drop):
        """Re-attach every edge of `drop` to `keep` and delete `drop`."""
        if keep == drop:
            return
        for eid, end in self.incident(drop):
            e = self.edges[eid]
            if end == 0:
                e.u = keep
            else:
                e.v = keep
            self.sync_ends(eid)
        del self.nodes[drop]
        self._touch()
        # an edge whose both ends are now the same node and which is short
        # is a degenerate loop: remove it
        for eid in [k for k, e in self.edges.items() if e.u == e.v]:
            if self.length(eid) < 6.0:
                self.remove_edge(eid, drop_orphans=False)

    def join_edges(self, e1, end1, e2, end2):
        """Join edge e1 (at its end `end1`) to edge e2 (at `end2`) into one
        edge; the two joined ends are bridged by a straight segment.  Returns
        the new edge id."""
        s1 = self.sample(e1, 0.7)
        s2 = self.sample(e2, 0.7)
        E1, E2 = self.edges[e1], self.edges[e2]
        if end1 == 0:          # we need e1 ending at the join
            s1 = {k: v[::-1] for k, v in s1.items()}
            n_far1 = E1.v
        else:
            n_far1 = E1.u
        if end2 == 1:          # we need e2 starting at the join
            s2 = {k: v[::-1] for k, v in s2.items()}
            n_far2 = E2.u
        else:
            n_far2 = E2.v
        gap = np.linalg.norm(s2["xy"][0] - s1["xy"][-1])
        nb = int(np.ceil(gap / 0.7))
        parts = {}
        for k in ("xy", "r", "s", "a"):
            a1, a2 = s1[k], s2[k]
            if nb > 1:
                t = np.linspace(0, 1, nb + 1)[1:-1]
                if a1.ndim == 2:
                    br = a1[-1][None] * (1 - t[:, None]) + a2[0][None] * t[:, None]
                else:
                    br = a1[-1] * (1 - t) + a2[0] * t
                parts[k] = np.concatenate([a1, br, a2], 0)
            else:
                parts[k] = np.concatenate([a1, a2], 0)
        info = dict(self.edges[e1].info)
        for k in ("gain",):
            info.pop(k, None)
        b1, b2 = E1.info.get("band"), E2.info.get("band")
        if b1 and b2:
            info["band"] = [min(b1[0], b2[0]), max(b1[1], b2[1])]
        spacing = min(E1.info.get("spacing", 12.0), E2.info.get("spacing", 12.0))
        join_nodes = {E1.u if end1 == 0 else E1.v, E2.u if end2 == 0 else E2.v}
        self.remove_edge(e1, drop_orphans=False)
        self.remove_edge(e2, drop_orphans=False)
        if n_far1 == n_far2 and e1 != e2:
            # joining would create a loop from a node to itself; keep it
            pass
        eid = self.add_edge_dense(parts["xy"], parts["r"], parts["s"], parts["a"],
                                  u=n_far1, v=n_far2, spacing=spacing, info=info)
        for n in join_nodes:
            if n in self.nodes and not self.incident(n):
                del self.nodes[n]
                self._touch()
        return eid

    def end_tangent(self, eid, end, look=6.0):
        """Unit vector pointing OUT of the edge at the given end."""
        smp = self.sample(eid, 0.7)
        xy, s = smp["xy"], smp["s_arc"]
        L = s[-1]
        look = min(look, 0.5 * L) if L > 0 else look
        if end == 0:
            j = int(np.searchsorted(s, look))
            v = xy[0] - xy[min(j, len(xy) - 1)]
        else:
            j = int(np.searchsorted(s, L - look))
            v = xy[-1] - xy[max(0, min(j, len(xy) - 1))]
        return v / (np.linalg.norm(v) + 1e-9)

    def end_width(self, eid, end):
        e = self.edges[eid]
        i = 0 if end == 0 else -1
        return float(max(e.r[i], R_MIN) + max(e.s[i], S_MIN))

    # ----------------------------------------------------------- topology ops
    def merge_close_nodes(self, tol=1.5):
        changed = True
        while changed:
            changed = False
            ids = list(self.nodes)
            if len(ids) < 2:
                return
            P = np.array([self.nodes[i].xy for i in ids])
            from scipy.spatial import cKDTree
            pairs = cKDTree(P).query_pairs(tol)
            for a, b in sorted(pairs):
                ia, ib = ids[a], ids[b]
                if ia in self.nodes and ib in self.nodes:
                    self.merge_nodes(ia, ib)
                    changed = True
            if changed:
                continue

    def merge_joints(self):
        """Fuse the two edges at every degree-2 node into a single edge."""
        changed = True
        while changed:
            changed = False
            for nid in list(self.nodes):
                if nid not in self.nodes or self.nodes[nid].fixed_kind:
                    continue
                inc = self.incident(nid)
                if len(inc) == 2 and inc[0][0] != inc[1][0]:
                    (e1, end1), (e2, end2) = inc
                    self.join_edges(e1, end1, e2, end2)
                    changed = True

    def resolve_crossings(self, cos_thr=0.80, width_ratio=2.5):
        """At nodes of degree >= 3, pair up edges that continue straight
        through (anti-parallel outward tangents, similar width) and join them,
        turning a skeleton junction into a crossing or a bifurcation whose
        parent vessel is a single continuous edge where appropriate."""
        for nid in list(self.nodes):
            if nid not in self.nodes:
                continue
            inc = self.incident(nid)
            if len(inc) < 3:
                continue
            tans = {k: self.end_tangent(*k) for k in inc}
            wid = {k: self.end_width(*k) for k in inc}
            cand = []
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    if inc[i][0] == inc[j][0]:
                        continue
                    c = float(np.dot(tans[inc[i]], tans[inc[j]]))
                    ratio = max(wid[inc[i]], wid[inc[j]]) / min(wid[inc[i]], wid[inc[j]])
                    if c < -cos_thr and ratio < width_ratio:
                        cand.append((c + 0.05 * np.log(ratio), inc[i], inc[j]))
            cand.sort()
            used = set()
            joins = []
            for _, a, b in cand:
                if a in used or b in used:
                    continue
                used.add(a)
                used.add(b)
                joins.append((a, b))
            if len(inc) == 3:
                # bifurcation: keep the node, the through-pair remains two
                # edges meeting there (C0) -- nothing to do
                continue
            for a, b in joins:
                if a[0] in self.edges and b[0] in self.edges:
                    # detach both from the node, join them through it
                    self.join_edges(a[0], a[1], b[0], b[1])
        self.merge_joints()

    def resolve_near_crossings(self, cos_thr=0.8, width_ratio=2.5, slack=4.0):
        """Two branch points close together on the same vessel whose side
        branches continue straight through each other are one crossing
        vessel, not two bifurcations: join the side branches into one edge
        that passes over (the densities add, see render.py)."""
        from scipy.spatial import cKDTree
        changed = True
        while changed:
            changed = False
            deg = self.degrees()
            hubs = [n for n in self.nodes if deg[n] >= 3]
            if len(hubs) < 2:
                return
            reach = {}
            for n in hubs:
                reach[n] = max(self.end_width(eid, end) for eid, end in self.incident(n))
            P = np.array([self.nodes[n].xy for n in hubs])
            R = max(reach.values())
            tree = cKDTree(P)
            best = None
            for i, j in tree.query_pairs(2 * R + slack):
                u, v = hubs[i], hubs[j]
                duv = P[j] - P[i]
                dist = np.linalg.norm(duv)
                if dist > reach[u] + reach[v] + slack or dist < 1e-6:
                    continue
                dirv = duv / dist
                for eu, endu in self.incident(u):
                    e = self.edges[eu]
                    if {e.u, e.v} == {u, v}:
                        continue
                    tu = self.end_tangent(eu, endu)
                    # the side branch must not run towards the other hub
                    if np.dot(tu, dirv) > 0.3:
                        continue
                    for ev, endv in self.incident(v):
                        if ev == eu:
                            continue
                        f = self.edges[ev]
                        if {f.u, f.v} == {u, v}:
                            continue
                        tv = self.end_tangent(ev, endv)
                        if np.dot(tv, -dirv) > 0.3 or np.dot(tu, tv) > -cos_thr:
                            continue
                        wu, wv = self.end_width(eu, endu), self.end_width(ev, endv)
                        if max(wu, wv) / min(wu, wv) > width_ratio:
                            continue
                        score = float(np.dot(tu, tv)) + 0.01 * dist
                        if best is None or score < best[0]:
                            best = (score, eu, endu, ev, endv)
            if best is not None:
                _, eu, endu, ev, endv = best
                self.join_edges(eu, endu, ev, endv)
                self.merge_joints()
                changed = True

    def crossings(self, tol=1.0):
        """Points where two edges overlap without sharing a node."""
        import cv2
        h, w = self.shape
        pix, lab = [], []
        for eid in self.edges:
            xy = np.round(self.sample(eid, 1.0)["xy"]).astype(np.int32)
            canvas = np.zeros((h, w), np.uint8)
            cv2.polylines(canvas, [xy], False, 1, 1)
            idx = np.flatnonzero(canvas)
            pix.append(idx)
            lab.append(np.full(len(idx), eid))
        if not pix:
            return []
        pix = np.concatenate(pix)
        lab = np.concatenate(lab)
        o = np.argsort(pix, kind="stable")
        pix, lab = pix[o], lab[o]
        dup = np.flatnonzero(pix[1:] == pix[:-1])
        pairs = {}
        for k in dup:
            a, b = int(lab[k]), int(lab[k + 1])
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            pairs.setdefault(key, []).append(pix[k])
        out = []
        for (a, b), ps in pairs.items():
            ea, eb = self.edges[a], self.edges[b]
            shared = {ea.u, ea.v} & {eb.u, eb.v}
            ys, xs = np.divmod(np.array(ps), w)
            pts = np.stack([xs, ys], 1).astype(float)
            # cluster the overlapping pixels of this pair
            used = np.zeros(len(pts), bool)
            for i in range(len(pts)):
                if used[i]:
                    continue
                m = np.linalg.norm(pts - pts[i], axis=1) < 6
                used |= m
                c = pts[m].mean(0)
                if any(np.linalg.norm(c - self.nodes[n].xy) < 8 for n in shared):
                    continue
                out.append(dict(edges=(a, b), x=float(c[0]), y=float(c[1])))
        return out

    # ------------------------------------------------------------- direction
    def orient_structural(self):
        """Orient edges from wide to narrow: in each connected component a BFS
        starts at the degree-1 node of the widest vessel and edges point away
        from it.  This is a convention, NOT a flow measurement."""
        import networkx as nx
        G = nx.MultiGraph()
        G.add_nodes_from(self.nodes)
        width = {}
        for eid, e in self.edges.items():
            width[eid] = float(np.mean(np.maximum(e.r, R_MIN)))
            G.add_edge(e.u, e.v, key=eid)
        deg = self.degrees()
        for comp in nx.connected_components(G):
            comp_edges = [eid for eid, e in self.edges.items() if e.u in comp]
            if not comp_edges:
                continue
            widest = max(comp_edges, key=lambda k: width[k])
            e = self.edges[widest]
            # start from the end of the widest edge that is a leaf / border,
            # else the end with the wider local calibre
            cands = [n for n in (e.u, e.v) if deg[n] == 1] or [e.u if e.r[0] >= e.r[-1] else e.v]
            root = cands[0]
            depth = nx.single_source_shortest_path_length(G, root)
            for eid in comp_edges:
                ed = self.edges[eid]
                du, dv = depth.get(ed.u, 0), depth.get(ed.v, 0)
                flip = dv < du or (dv == du and ed.r[-1] > ed.r[0])
                if ed.info.get("orientation") in (None, "structural"):
                    if flip:
                        self.reverse_edge(eid)
                    ed.info["orientation"] = "structural"

    # ------------------------------------------------------------- exports
    def to_digraph(self):
        """networkx.DiGraph (MultiDiGraph if parallel edges exist) with
        node positions/kinds and per-edge geometry + profile statistics."""
        import networkx as nx
        multi = len({(e.u, e.v) for e in self.edges.values()}) < len(self.edges)
        G = nx.MultiDiGraph() if multi else nx.DiGraph()
        deg = self.degrees()
        for nid, n in self.nodes.items():
            G.add_node(nid, x=n.x, y=n.y, kind=self.node_kind(nid, deg[nid]))
        for eid, e in self.edges.items():
            st = self.edge_stats(eid)
            attrs = dict(eid=eid, orientation=e.info.get("orientation", "structural"), **st)
            if multi:
                G.add_edge(e.u, e.v, key=eid, **attrs)
            else:
                G.add_edge(e.u, e.v, **attrs)
        G.graph["crossings"] = self.crossings()
        G.graph["image_shape"] = self.shape
        return G

    def to_dict(self) -> dict:
        deg = self.degrees()
        return dict(
            format="vesselmap-network-v1", shape=list(self.shape),
            nodes=[dict(id=k, x=n.x, y=n.y, kind=self.node_kind(k, deg[k]),
                        fixed_kind=n.fixed_kind) for k, n in self.nodes.items()],
            edges=[dict(id=k, u=e.u, v=e.v, ctrl=e.ctrl.round(4).tolist(),
                        r=e.r.round(5).tolist(), s=e.s.round(5).tolist(),
                        a=e.a.round(6).tolist(),
                        info={kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                              for kk, vv in e.info.items()})
                   for k, e in self.edges.items()],
            background=None if self.background is None else self.background.round(6).tolist(),
            bg_spacing=self.bg_spacing, meta=self.meta)

    @classmethod
    def from_dict(cls, d) -> "VesselNetwork":
        n = cls(d["shape"])
        for nd in d["nodes"]:
            n.nodes[nd["id"]] = Node(nd["x"], nd["y"], nd.get("fixed_kind"))
        for ed in d["edges"]:
            n.edges[ed["id"]] = Edge(ed["u"], ed["v"], np.array(ed["ctrl"], float),
                                     np.array(ed["r"], float), np.array(ed["s"], float),
                                     np.array(ed["a"], float), dict(ed.get("info", {})))
        n._nid = max(n.nodes, default=-1) + 1
        n._eid = max(n.edges, default=-1) + 1
        if d.get("background") is not None:
            n.background = np.array(d["background"], np.float32)
        n.bg_spacing = d.get("bg_spacing")
        n.meta = d.get("meta", {})
        return n

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path) -> "VesselNetwork":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> dict:
        deg = self.degrees()
        kinds = {}
        for k in self.nodes:
            kd = self.node_kind(k, deg[k])
            kinds[kd] = kinds.get(kd, 0) + 1
        L = [self.length(e) for e in self.edges]
        return dict(n_edges=len(self.edges), n_nodes=len(self.nodes), node_kinds=kinds,
                    total_length_px=float(np.sum(L)) if L else 0.0)


def _fit_edge_params(xy, r, s, a, spacing):
    xy = np.asarray(xy, float)
    r, s, a = (np.asarray(v, float).reshape(-1) for v in (r, s, a))
    arc = sp.arclength(xy)
    L = float(arc[-1])
    n_ctrl = sp.n_ctrl_for_length(L, spacing, minimum=4)
    m = max(4 * n_ctrl, int(np.ceil(L / 0.5)) + 1)
    q = np.linspace(0, L, m) if L > 0 else np.zeros(m)
    if L > 0:
        xyu = np.stack([np.interp(q, arc, xy[:, 0]), np.interp(q, arc, xy[:, 1])], 1)
    else:
        xyu = np.repeat(xy[:1], m, 0)
    ctrl = sp.fit_ctrl(xyu, n_ctrl, smooth=2e-4, pin_ends=True)
    n_p = sp.n_ctrl_for_length(L, PROFILE_SPACING, minimum=2)
    out = []
    for vals, lo in ((r, R_MIN), (s, S_MIN), (a, A_MIN)):
        if len(vals) != len(arc):
            vals = np.interp(np.linspace(0, 1, len(arc)), np.linspace(0, 1, len(vals)), vals)
        vu = np.interp(q, arc, vals) if L > 0 else np.full(m, vals.mean())
        c = sp.fit_ctrl(vu, n_p, smooth=1e-2, pin_ends=False)[:, 0]
        out.append(np.maximum(c, lo))
    return ctrl, out[0], out[1], out[2]
