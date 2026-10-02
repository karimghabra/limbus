"""Skeleton -> vessel graph.

A 1-px skeleton is converted into a graph whose *nodes* are branch points,
end points and crossings, and whose *edges* ("segments") are the ordered pixel
chains between them.

Pixel classification by 8-neighbour count n(p):
    n = 1 -> end point      n = 2 -> interior of a segment      n >= 3 -> junction

Clean-up rules applied iteratively:
  * spur pruning   - short dangling segments (length < spur_len) are thinning
                     artefacts where a vessel's edge was bumpy; delete them;
  * degree-2 merge - after pruning, a node with exactly two segments is no
                     longer a junction: concatenate the two segments;
  * junction merge - two junctions joined by a very short segment are really
                     one branch point smeared by thinning; contract them;
  * crossings      - in the limbus, conjunctival vessels pass *over* deeper
                     episcleral ones. In 2-D this looks like a 4-way junction
                     but no blood is exchanged. A degree-4 node whose
                     segments form two nearly straight pairs is split into two
                     independent pass-through vessels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy import ndimage as ndi

K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class VesselGraph:
    nodes: dict = field(default_factory=dict)   # id -> {pos: (x, y), kind: str}
    edges: dict = field(default_factory=dict)   # id -> {u, v, path: (N, 2) float [x, y] ordered u -> v}
    _nid: int = 0
    _eid: int = 0

    def add_node(self, pos, kind="junction"):
        self.nodes[self._nid] = dict(pos=np.asarray(pos, float), kind=kind)
        self._nid += 1
        return self._nid - 1

    def add_edge(self, u, v, path):
        self.edges[self._eid] = dict(u=u, v=v, path=np.asarray(path, float))
        self._eid += 1
        return self._eid - 1

    def incident(self, n):
        return [e for e, d in self.edges.items() if d["u"] == n or d["v"] == n]

    def degree(self, n):
        return sum((d["u"] == n) + (d["v"] == n) for d in self.edges.values())

    @staticmethod
    def length(path):
        return float(np.sum(np.hypot(*np.diff(path, axis=0).T))) if len(path) > 1 else 0.0

    def path_from(self, e, n):
        """Path of edge e oriented so that it starts at node n."""
        d = self.edges[e]
        return d["path"] if d["u"] == n else d["path"][::-1]

    def to_networkx(self):
        G = nx.MultiGraph()
        for n, d in self.nodes.items():
            G.add_node(n, **d)
        for e, d in self.edges.items():
            G.add_edge(d["u"], d["v"], key=e, **d)
        return G


# ----------------------------------------------------------------------------
def classify_pixels(skel):
    n = ndi.convolve(skel.astype(np.uint8), K8, mode="constant") * skel
    return n  # 0 background, 1 end, 2 interior, >=3 junction


def _trace(start, seg_mask, visited):
    """Walk along a chain of pixels (each with <= 2 neighbours in seg_mask)."""
    path = [start]; visited[start] = True
    H, W = seg_mask.shape
    while True:
        y, x = path[-1]
        nxt = None
        for dy, dx in OFFS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and seg_mask[yy, xx] and not visited[yy, xx]:
                nxt = (yy, xx); break
        if nxt is None:
            return path
        visited[nxt] = True
        path.append(nxt)


def skeleton_to_graph(skel) -> VesselGraph:
    skel = skel.astype(bool)
    H, W = skel.shape
    nb = classify_pixels(skel)
    junc = nb >= 3
    jlab, nj = ndi.label(junc, structure=np.ones((3, 3)))
    G = VesselGraph()
    jnode = {}
    for lab, sl in enumerate(ndi.find_objects(jlab), start=1):
        ys, xs = np.nonzero(jlab[sl] == lab)
        jnode[lab] = G.add_node((xs.mean() + sl[1].start, ys.mean() + sl[0].start), "junction")

    seg = skel & ~junc
    visited = np.zeros_like(seg)
    nseg = ndi.convolve(seg.astype(np.uint8), K8, mode="constant") * seg

    def touching_junction(p):
        y, x = p
        for dy, dx in OFFS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and jlab[yy, xx]:
                return jlab[yy, xx]
        return 0

    # chains start at pixels with <=1 neighbour inside `seg` (ends or next to junctions)
    starts = list(zip(*np.nonzero(seg & (nseg <= 1))))
    for s in starts:
        if visited[s]:
            continue
        path = _trace(s, seg, visited)
        if len(path) == 1:  # a single pixel bridging two junction clusters
            y, x = path[0]
            labs = {jlab[y + dy, x + dx] for dy, dx in OFFS
                    if 0 <= y + dy < H and 0 <= x + dx < W and jlab[y + dy, x + dx]}
            if len(labs) == 2:
                a, b = (jnode[l] for l in labs)
                G.add_edge(a, b, [tuple(G.nodes[a]["pos"]), (x, y), tuple(G.nodes[b]["pos"])])
            continue
        ends = []
        for p in (path[0], path[-1]):
            j = touching_junction(p)
            if j:
                ends.append(jnode[j])
            else:
                ends.append(G.add_node((p[1], p[0]), "end"))
        xy = [(x, y) for y, x in path]
        xy = [tuple(G.nodes[ends[0]]["pos"])] + xy + [tuple(G.nodes[ends[1]]["pos"])]
        if len(path) == 1 and ends[0] == ends[1]:
            continue
        G.add_edge(ends[0], ends[1], xy)
    # closed loops without any end/junction
    for s in zip(*np.nonzero(seg & ~visited)):
        if visited[s]:
            continue
        path = _trace(s, seg, visited)
        n0 = G.add_node((path[0][1], path[0][0]), "loop")
        G.add_edge(n0, n0, [(x, y) for y, x in path] + [(path[0][1], path[0][0])])
    # junction clusters that touch each other directly (no pixels between) are
    # already merged by the 8-connected labelling above
    return G


# ----------------------------------------------------------------------------
def _remove_node_if_isolated(G, n):
    if G.degree(n) == 0:
        G.nodes.pop(n, None)


def merge_degree2(G: VesselGraph):
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n not in G.nodes or G.nodes[n]["kind"] in ("crossing",):
                continue
            inc = G.incident(n)
            if len(inc) == 2 and inc[0] != inc[1]:
                e1, e2 = inc
                p1 = G.path_from(e1, n)[::-1]          # ... -> n
                p2 = G.path_from(e2, n)                # n -> ...
                a = G.edges[e1]["v"] if G.edges[e1]["u"] == n else G.edges[e1]["u"]
                b = G.edges[e2]["v"] if G.edges[e2]["u"] == n else G.edges[e2]["u"]
                if a == n or b == n:
                    continue
                G.edges.pop(e1); G.edges.pop(e2)
                G.add_edge(a, b, np.vstack([p1, p2[1:]]))
                G.nodes.pop(n)
                changed = True


def _radius_at(G, n, radius):
    if radius is None:
        return 0.0
    x, y = np.round(G.nodes[n]["pos"]).astype(int)
    return float(radius[np.clip(y, 0, radius.shape[0] - 1), np.clip(x, 0, radius.shape[1] - 1)])


def prune_spurs(G: VesselGraph, spur_len=15.0, max_iter=10, radius=None, radius_factor=2.0):
    """Remove dangling segments shorter than max(spur_len, radius_factor * r_parent),
    r_parent = local radius at the junction: a 'branch' that does not even reach
    far beyond the wall of the vessel it hangs from is a thinning artefact."""
    for _ in range(max_iter):
        removed = False
        for e in list(G.edges):
            d = G.edges.get(e)
            if d is None:
                continue
            du, dv = G.degree(d["u"]), G.degree(d["v"])
            j = d["u"] if du > 1 else d["v"]
            thr = max(spur_len, radius_factor * _radius_at(G, j, radius))
            if (du == 1) != (dv == 1) and G.length(d["path"]) < thr:
                G.edges.pop(e)
                for n in (d["u"], d["v"]):
                    _remove_node_if_isolated(G, n)
                removed = True
            elif du == 1 and dv == 1 and G.length(d["path"]) < spur_len:  # tiny isolated piece
                G.edges.pop(e)
                for n in (d["u"], d["v"]):
                    _remove_node_if_isolated(G, n)
                removed = True
        merge_degree2(G)
        if not removed:
            break


def merge_close_junctions(G: VesselGraph, max_len=10.0, radius=None):
    """Contract short segments between two junctions.

    With a `radius` map (distance transform of the vessel mask = local vessel
    radius) the threshold grows to r_u + r_v: when the two junction 'discs'
    overlap, thinning has split one branch point - or one crossing of two
    thick vessels - into two T-junctions.
    """
    def rad(n):
        if radius is None:
            return 0.0
        x, y = np.round(G.nodes[n]["pos"]).astype(int)
        return float(radius[np.clip(y, 0, radius.shape[0] - 1), np.clip(x, 0, radius.shape[1] - 1)])
    changed = True
    while changed:
        changed = False
        for e in list(G.edges):
            d = G.edges.get(e)
            if d is None or d["u"] == d["v"]:
                continue
            thr = max(max_len, rad(d["u"]) + rad(d["v"]))
            if G.degree(d["u"]) >= 3 and G.degree(d["v"]) >= 3 and G.length(d["path"]) < thr:
                u, v = d["u"], d["v"]
                c = d["path"][len(d["path"]) // 2]
                G.edges.pop(e)
                G.nodes[u]["pos"] = np.asarray(c, float)
                for e2 in G.incident(v):
                    d2 = G.edges[e2]
                    if d2["u"] == v:
                        d2["u"] = u; d2["path"] = np.vstack([c, d2["path"]])
                    if d2["v"] == v:
                        d2["v"] = u; d2["path"] = np.vstack([d2["path"], c])
                for e2 in G.incident(u):
                    d2 = G.edges[e2]
                    if d2["u"] == u:
                        d2["path"][0] = c
                    if d2["v"] == u:
                        d2["path"][-1] = c
                G.nodes.pop(v)
                changed = True
                break


def _out_tangent(G, e, n, reach=12.0):
    """Unit vector pointing away from node n along edge e (averaged over `reach` px)."""
    p = G.path_from(e, n)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    k = np.searchsorted(s, min(reach, s[-1]))
    v = p[max(k, 1)] - p[0]
    return v / (np.linalg.norm(v) + 1e-9)


def split_crossings(G: VesselGraph, max_dev_deg=35.0, reach=15.0):
    """Split 4-way nodes made of two straight-through pairs into two degree-2 nodes."""
    for n in list(G.nodes):
        inc = G.incident(n)
        if len(inc) != 4 or len(set(inc)) != 4:
            continue
        t = {e: _out_tangent(G, e, n, reach) for e in inc}
        best = None
        for pairing in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            devs = []
            for a, b in pairing:
                cosang = np.dot(t[inc[a]], t[inc[b]])
                devs.append(np.degrees(np.arccos(np.clip(-cosang, -1, 1))))  # 0 = perfectly straight
            if best is None or max(devs) < best[0]:
                best = (max(devs), pairing)
        if best[0] > max_dev_deg:
            continue
        pos = G.nodes[n]["pos"]
        for a, b in best[1]:
            m = G.add_node(pos, "crossing")
            for e in (inc[a], inc[b]):
                d = G.edges[e]
                if d["u"] == n:
                    d["u"] = m
                if d["v"] == n:
                    d["v"] = m
        G.nodes.pop(n)


def merge_through_crossings(G: VesselGraph):
    """Concatenate the two halves of each crossing so one vessel passes straight through.

    The crossing location is remembered on the merged edge (``crossings``)."""
    for n in [k for k, d in G.nodes.items() if d["kind"] == "crossing"]:
        inc = G.incident(n)
        if len(inc) != 2 or inc[0] == inc[1]:
            continue
        e1, e2 = inc
        p1 = G.path_from(e1, n)[::-1]; p2 = G.path_from(e2, n)
        a = G.edges[e1]["v"] if G.edges[e1]["u"] == n else G.edges[e1]["u"]
        b = G.edges[e2]["v"] if G.edges[e2]["u"] == n else G.edges[e2]["u"]
        cr = G.edges[e1].get("crossings", []) + G.edges[e2].get("crossings", []) + [tuple(G.nodes[n]["pos"])]
        G.edges.pop(e1); G.edges.pop(e2)
        e = G.add_edge(a, b, np.vstack([p1, p2[1:]]))
        G.edges[e]["crossings"] = cr
        G.nodes.pop(n)


def label_border_nodes(G: VesselGraph, valid, margin=20):
    """End points close to the edge of the field of view are 'border' nodes:
    the vessel continues outside the image (blood enters/leaves the FOV there)."""
    dist = ndi.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]  # image edge counts as border
    for n, d in G.nodes.items():
        if G.degree(n) == 1:
            x, y = np.round(d["pos"]).astype(int)
            y = np.clip(y, 0, valid.shape[0] - 1); x = np.clip(x, 0, valid.shape[1] - 1)
            d["kind"] = "border" if dist[y, x] < margin else "end"
        elif G.degree(n) >= 3:
            d["kind"] = "junction"


def build_graph(skel, valid, spur_len=15.0, junction_merge=10.0, crossing_dev=35.0,
                min_vessel_len=20.0, border_margin=25, radius=None, spur_radius_factor=2.0):
    """Full clean-up pipeline; returns a VesselGraph. `radius`: local vessel radius map."""
    G = skeleton_to_graph(skel)
    prune_spurs(G, spur_len, radius=radius, radius_factor=spur_radius_factor)
    merge_close_junctions(G, junction_merge, radius)
    prune_spurs(G, spur_len, radius=radius, radius_factor=spur_radius_factor)
    split_crossings(G, crossing_dev)
    merge_through_crossings(G)
    merge_degree2(G)
    # drop tiny isolated fragments
    for e in list(G.edges):
        d = G.edges[e]
        if G.degree(d["u"]) == 1 and G.degree(d["v"]) == 1 and G.length(d["path"]) < min_vessel_len:
            G.edges.pop(e)
            for n in (d["u"], d["v"]):
                _remove_node_if_isolated(G, n)
    label_border_nodes(G, valid, border_margin)
    return G
