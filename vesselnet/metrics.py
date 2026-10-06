"""PLAN.md §6 architecture metrics of a VesselNetwork against vesselscene's truth (one image).

The convention is vesselscene's, strictly (decided at iteration 0): a vessel ends at every fork, so a fork
is three edges meeting at a branch node, and an edge running on through a fork (vesselmap's `through`)
joins two different vessels.  Crossings are not nodes.

    m = architecture(net, obs, junctions, truth_net)   # obs: observable_average.json, junctions: junctions_average.json

Groups of numbers returned:
- centreline / junction (vesselscene.truth.score_tracing against the observable truth; the network's
  junctions are its nodes where 3 or more edge ends or through-passes meet);
- vessels (vesselmap.synthetic.vessel_metrics against the truth's vessels: fragments, best cover, purity,
  mixed edges, excess, spurious);
- topology, against the visible junctions' events (junctions_average.json):
  * crossings_as_nodes: crossing events with a network node of degree >= 3 within tol and no true node
    event within tol (should be 0);
  * forks_found: node events (bifurcation, confluence, anastomosis) with a network node of degree >= 3
    within tol;
  * pairing: per visible junction with at least 3 visible in-frame arms, each arm is assigned the network
    edge nearest its point (within max(3, width / 2) px).  Two arms are joined in the network when they have
    the same edge, and in the truth when they share a `partition` group.  A junction is paired correctly
    when every arm has an edge and every pair of arms agrees.  false_joins (joined, truth apart: a through-
    edge at a fork, a crossing that turns) and false_splits (apart, truth joined: a vessel broken at a
    crossing) count arm pairs.  Reported over all such junctions and the unambiguous ones.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial import cKDTree

NODE_EVENTS = ("bifurcation", "confluence", "anastomosis")


def polylines(net, spacing=1.0) -> list[np.ndarray]:
    return [net.sample(k, spacing)["xy"] for k in net.edges]


def branch_nodes(net) -> np.ndarray:
    """(n, 2) positions of the nodes where at least 3 edge ends or through-passes meet."""
    out = []
    for nid, n in net.nodes.items():
        if len(net.incident(nid)) + len(net.passing(nid)) >= 3:
            out.append(n.xy)
    return np.asarray(out, float).reshape(-1, 2)


def _events(junctions):
    """Visible junctions' events: (kind, xy, tol) with kind 'crossing' or 'node'."""
    ev = []
    for j in junctions:
        if j.get("type_visible", "none") == "none":
            continue
        w = max([a.get("width_px") or 0.0 for a in j["arms"]] + [0.0])
        tol = 4.0 + 0.5 * w
        for m in j["members"]:
            kind = "crossing" if m["type"] == "crossing" else ("node" if m["type"] in NODE_EVENTS else None)
            if kind:
                ev.append((kind, np.asarray(m["xy"], float), tol))
    return ev


def topology(net, junctions) -> dict:
    nodes = branch_nodes(net)
    nt = cKDTree(nodes) if len(nodes) else None
    ev = _events(junctions)
    node_ev = np.asarray([p for k, p, _ in ev if k == "node"]).reshape(-1, 2)
    nev = cKDTree(node_ev) if len(node_ev) else None
    near = lambda tree, p, tol: tree is not None and bool(tree.query_ball_point(p, tol))
    cx = [(p, t) for k, p, t in ev if k == "crossing"]
    cx_nodes = sum(near(nt, p, t) and not near(nev, p, t) for p, t in cx)
    nd = [(p, t) for k, p, t in ev if k == "node"]
    found = sum(near(nt, p, t) for p, t in nd)
    out = dict(crossing_events=len(cx), crossings_as_nodes=int(cx_nodes),
               crossings_as_nodes_rate=(cx_nodes / len(cx) if cx else float("nan")),
               fork_events=len(nd), forks_found=int(found), forks_found_rate=(found / len(nd) if nd else float("nan")),
               branch_nodes=int(len(nodes)))
    out.update(pairing(net, junctions))
    return out


def pairing(net, junctions, spacing=1.0) -> dict:
    pts, eid = [], []
    for k in net.edges:
        xy = net.sample(k, spacing)["xy"]
        pts.append(xy)
        eid.append(np.full(len(xy), k))
    tree = cKDTree(np.concatenate(pts)) if pts else None
    eid = np.concatenate(eid) if eid else np.zeros(0, int)
    res = {"all": [0, 0], "unambiguous": [0, 0]}
    fj = fs = pairs = unmatched = 0
    for j in junctions:
        arms = [(i, a) for i, a in enumerate(j["arms"]) if a.get("visibility") == "visible" and a.get("in_frame", True)]
        if j.get("type_visible", "none") == "none" or len(arms) < 3:
            continue
        group = {}
        for g, idx in enumerate(j.get("partition") or []):
            for i in idx:
                group[i] = g
        lab = {}
        for i, a in arms:
            if tree is None:
                lab[i] = None
                continue
            d, q = tree.query(np.asarray(a["xy"], float))
            lab[i] = int(eid[q]) if d <= max(3.0, 0.5 * (a.get("width_px") or 0.0)) else None
        ok = all(v is not None for v in lab.values())
        unmatched += sum(v is None for v in lab.values())
        for (i, _), (k, _) in itertools.combinations(arms, 2):
            if lab[i] is None or lab[k] is None:
                continue
            pairs += 1
            joined, same = lab[i] == lab[k], group.get(i, -1 - i) == group.get(k, -1 - k)
            fj += joined and not same
            fs += same and not joined
            ok &= joined == same
        for key in ("all",) + (("unambiguous",) if not j.get("ambiguous") else ()):
            res[key][0] += 1
            res[key][1] += int(ok)
    return dict(pairing_junctions=res["all"][0],
                pairing_accuracy=(res["all"][1] / res["all"][0] if res["all"][0] else float("nan")),
                pairing_junctions_unambiguous=res["unambiguous"][0],
                pairing_accuracy_unambiguous=(res["unambiguous"][1] / res["unambiguous"][0]
                                              if res["unambiguous"][0] else float("nan")),
                arm_pairs=pairs, false_joins=int(fj), false_splits=int(fs), arms_unmatched=int(unmatched))


def vessels(net, truth_net) -> dict:
    """vesselmap.synthetic.vessel_metrics with the truth's edges (vesselscene vessels) as the true vessels."""
    from vesselmap.synthetic import vessel_metrics
    tv = []
    for k in truth_net.edges:
        s = truth_net.sample(k, 1.0)
        if len(s["xy"]) >= 2:
            tv.append(dict(xy=s["xy"], r=s["r"]))
    return vessel_metrics(net, tv, net.shape)


def _runs_of(mask):
    """[a, b) index ranges where mask is True."""
    d = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(d[::2], d[1::2]))


def window_truth(obs: dict, junctions: list, y0: int, x0: int, size: int) -> tuple[dict, list]:
    """The observable truth and the junctions of a size x size window at (y0, x0), in window coordinates:
    runs cut to their stretches inside (their 'unobserved' ranges re-indexed), junctions (with their events
    and arms) whose centre lies inside."""
    off = np.array([x0, y0], float)
    inside = lambda P: (P[:, 0] >= 0) & (P[:, 0] <= size - 1) & (P[:, 1] >= 0) & (P[:, 1] <= size - 1)
    runs = []
    for r in obs["runs"]:
        P = np.asarray(r["xy"], float).reshape(-1, 2) - off
        unobs = np.zeros(len(P), bool)
        for p_, q_ in r.get("unobserved") or ():
            unobs[p_:q_ + 1] = True
        for a, b in _runs_of(inside(P)):
            if b - a < 2:
                continue
            u = [[int(p), int(q - 1)] for p, q in _runs_of(unobs[a:b])]
            runs.append(dict(r, xy=P[a:b].tolist(), unobserved=u))
    keep = lambda d: 0 <= d["x"] - x0 <= size - 1 and 0 <= d["y"] - y0 <= size - 1
    sh = lambda d: dict(d, x=d["x"] - x0, y=d["y"] - y0)
    o = dict(obs, runs=runs, junctions=[sh(j) for j in obs["junctions"] if keep(j)],
             dont_care_junctions=[sh(j) for j in obs.get("dont_care_junctions", []) if "x" in j and keep(j)])
    js = []
    for j in junctions:
        if keep(j):
            js.append(dict(sh(j), members=[dict(m, xy=[m["xy"][0] - x0, m["xy"][1] - y0]) for m in j["members"]],
                           arms=[dict(a, xy=[a["xy"][0] - x0, a["xy"][1] - y0],
                                      in_frame=bool(a.get("in_frame", True)) and
                                      0 <= a["xy"][0] - x0 <= size - 1 and 0 <= a["xy"][1] - y0 <= size - 1)
                                 for a in j["arms"]]))
    return o, js


def architecture(net, obs: dict, junctions: list, truth_net=None, prof=None) -> dict:
    from vesselscene.truth import score_tracing
    nodes = branch_nodes(net)
    st = score_tracing(obs, polylines(net), [tuple(p) for p in nodes], prof=prof)
    out = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in st.items()}
    out.update(topology(net, junctions))
    if truth_net is not None:
        out.update({"vessels_" + k: v for k, v in vessels(net, truth_net).items()})
    out["n_edges"] = len(net.edges)
    out["length_px"] = float(sum(net.length(k) for k in net.edges))
    return out
