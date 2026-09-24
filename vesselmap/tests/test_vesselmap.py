"""Tests for vesselmap.  Run:  python -m pytest vesselmap/tests -q"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from vesselmap.network import VesselNetwork
from vesselmap.synthetic import centreline_metrics, make_scene


def _line(net, p, q, r=2.0, s=1.0, a=0.3, n=50, **kw):
    xy = np.linspace(p, q, n)
    return net.add_edge_dense(xy, np.full(n, r), np.full(n, s), np.full(n, a), **kw)


def test_split_join_roundtrip():
    net = VesselNetwork((200, 200))
    e = _line(net, (10, 100), (190, 100))
    L = net.length(e)
    nid = net.split_edge(e, (100, 101))
    assert len(net.edges) == 2 and net.degrees()[nid] == 2
    assert abs(sum(net.length(k) for k in net.edges) - L) < 1.0
    net.merge_joints()
    assert len(net.edges) == 1
    (k,) = net.edges
    assert abs(net.length(k) - L) < 1.0


def test_bifurcation_and_crossing_classification():
    net = VesselNetwork((200, 200))
    parent = _line(net, (10, 100), (190, 100), r=4)
    nid = net.split_edge(parent, (100, 100))
    _line(net, (100, 100), (160, 30), r=1.5, u=nid)
    # an unrelated vessel crossing the parent at x=50
    _line(net, (50, 20), (50, 180), r=1.0, s=3.0)
    deg = net.degrees()
    kinds = sorted(net.node_kind(n, deg[n]) for n in net.nodes)
    assert kinds.count("bifurcation") == 1
    cr = net.crossings()
    assert len(cr) == 1 and abs(cr[0]["x"] - 50) < 3 and abs(cr[0]["y"] - 100) < 3


def test_resolve_crossings_joins_straight_through():
    net = VesselNetwork((200, 200))
    c = net.add_node(100, 100)
    for p in ((20, 100), (180, 100), (100, 20), (100, 180)):
        _line(net, p, (100, 100), v=c)
    net.resolve_crossings()
    assert len(net.edges) == 2               # two vessels crossing
    assert c not in net.nodes                # no node at a crossing


def test_near_crossing_becomes_one_vessel():
    net = VesselNetwork((200, 200))
    parent = _line(net, (10, 100), (190, 100), r=5)
    a = net.split_edge(parent, (100, 95))
    # a thin vessel found as two pieces ending on opposite sides of the parent
    _line(net, (100, 20), (100, 95), r=1, v=a)
    eid = [k for k, e in net.edges.items()
           if a in (e.u, e.v) and net.length(k) > 80 and e.ctrl[:, 0].max() > 150][0]
    b = net.split_edge(eid, (104, 100))
    _line(net, (104, 100), (104, 180), r=1, u=b)
    net.resolve_near_crossings()
    deg = net.degrees()
    assert sum(1 for n in net.nodes if deg[n] >= 3) == 0
    assert len(net.edges) == 2


def test_json_and_digraph(tmp_path):
    net = VesselNetwork((200, 200))
    parent = _line(net, (10, 100), (190, 100), r=4)
    nid = net.split_edge(parent, (100, 100))
    _line(net, (100, 100), (160, 30), r=1.5, u=nid)
    net.orient_structural()
    p = tmp_path / "m.json"
    net.save(p)
    back = VesselNetwork.load(p)
    a, b = back.summary(), net.summary()
    assert a["node_kinds"] == b["node_kinds"] and a["n_edges"] == b["n_edges"]
    assert abs(a["total_length_px"] - b["total_length_px"]) < 1e-3
    G = back.to_digraph()
    assert G.number_of_edges() == 3 and G.number_of_nodes() == 4
    # structural orientation: the thin branch points away from the junction
    thin = [(u, v) for u, v, d in G.edges(data=True) if d["diameter"] < 5][0]
    assert thin[0] == nid
    assert all(d["orientation"] == "structural" for _, _, d in G.edges(data=True))


@pytest.mark.slow
def test_synthetic_scene_recall_precision():
    from vesselmap.fit import MapConfig, build_map
    I, vessels, _ = make_scene(3, shape=(320, 480), n_trees=2, n_cross=2, n_capillary=8)
    net = build_map(I, MapConfig(verbose=False, iters_band=60, iters_final=100))
    m = centreline_metrics(net, vessels, I.shape)
    assert m["recall"] > 0.7, m
    assert m["precision"] > 0.7, m


@pytest.mark.slow
def test_fit_frame_recovers_shift():
    from scipy import ndimage as ndi
    from vesselmap.fit import FrameFitConfig, MapConfig, build_map, fit_frame
    I, vessels, _ = make_scene(4, shape=(320, 480), n_trees=2, n_cross=1, n_capillary=4)
    ref = build_map(I, MapConfig(verbose=False, iters_band=60, iters_final=80))
    shifted = ndi.shift(I, (-2.0, 3.0), order=3, mode="nearest")   # dy, dx
    net, rep = fit_frame(shifted, ref, FrameFitConfig(iters=30))
    t = np.array(rep["affine"]["t"])
    assert abs(t[0] - 3.0) < 0.6 and abs(t[1] + 2.0) < 0.6, rep
    assert set(net.edges) == set(ref.edges)


def test_chained_start_carries_local_deformation_only():
    from vesselmap.fit import _carry_deformation, apply_affine
    ref = VesselNetwork((200, 300))
    _line(ref, (20, 100), (280, 100))
    A_prev, t_prev = [[1.0, 0.0], [0.0, 1.0]], [10.0, -5.0]
    prev = ref.copy()
    apply_affine(prev, A_prev, t_prev)
    init = prev.copy()                        # previous frame's fit ...
    (k,) = init.edges
    init.edges[k].ctrl[1:-1, 1] += 2.0        # ... with a local 2 px bulge
    init.meta["frame_fit"] = {"affine": {"A": A_prev, "t": t_prev}}
    base = ref.copy()
    apply_affine(base, [[1.0, 0.0], [0.0, 1.0]], [-20.0, 7.0])   # this frame moved
    net = _carry_deformation(ref, base, init)
    d = net.edges[k].ctrl - base.edges[k].ctrl
    assert np.allclose(d[1:-1, 1], 2.0) and np.allclose(d[:, 0], 0.0)
