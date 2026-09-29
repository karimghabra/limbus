"""Tests for the energy search (search.py) and the vessel-level metrics.

    python -m pytest vesselmap/tests/test_search.py -q
"""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from vesselmap.search import SearchConfig, VesselSearch, edge_samples, join_path
from vesselmap.image import prepare
from vesselmap.network import VesselNetwork
from vesselmap.synthetic import fragment_network, make_scene, render, vessel_metrics


def _vessel(xy, r=2.0, blur=1.0, amp=0.35):
    xy = np.asarray(xy, float)
    return dict(xy=xy, r=np.full(len(xy), r), blur=blur, amp=amp)


def _line(p, q, n=300):
    return np.linspace(p, q, n)


def _edge(net, xy, r=2.0, s=1.0, a=0.35, **kw):
    n = len(xy)
    return net.add_edge_dense(np.asarray(xy, float), np.full(n, r), np.full(n, s),
                              np.full(n, a), **kw)


def _scene(vessels, shape, seed=0):
    I, _ = render(vessels, shape, np.random.default_rng(seed))
    return I, prepare(I)


def _cfg(**kw):
    kw.setdefault("init_iters", 60)
    kw.setdefault("global_iters", 40)
    kw.setdefault("verbose", False)
    return SearchConfig(**kw)


def test_metrics_see_fragmentation():
    I, vessels, _ = make_scene(0, shape=(256, 384), n_trees=2, n_cross=2, n_capillary=6)
    ideal = fragment_network(vessels, I.shape, np.random.default_rng(0), piece_len=(1e9, 1e9),
                             dup_frac=0, n_spurious=0, jitter=0)
    m = vessel_metrics(ideal, vessels, I.shape)
    assert m["excess"] == 0 and m["fragments_weighted"] == 1.0 and m["best_cover"] > 0.99, m
    frag = fragment_network(vessels, I.shape, np.random.default_rng(1))
    f = vessel_metrics(frag, vessels, I.shape)
    assert f["fragments_weighted"] > 2 and f["best_cover"] < 0.7 and f["spurious"] >= 5, f


def test_join_path_trims_overlap_and_bridges_gap():
    A = edge_samples(_edge_obj(_line((0, 50), (60, 50), 60)), 0.7)
    B = edge_samples(_edge_obj(_line((50, 50), (120, 50), 70)), 0.7)   # overlaps A by 10 px
    xy, r, s, a = join_path(A, B, trim_max=25, max_turn_deg=120)
    L = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    assert abs(L - 120) < 2 and np.all(np.diff(xy[:, 0]) > 0), L
    C = edge_samples(_edge_obj(_line((75, 50), (120, 50), 45)), 0.7)   # 15 px gap
    xy, *_ = join_path(A, C, trim_max=25, max_turn_deg=120)
    assert abs(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() - 120) < 2
    assert np.abs(xy[:, 1] - 50).max() < 0.5


def _edge_obj(xy):
    net = VesselNetwork((200, 200))
    k = _edge(net, xy)
    return net.edges[k]


def test_fragments_become_one_vessel_and_spurious_goes():
    H, W = 120, 240
    v = _vessel(_line((-10, 60), (250, 64)), r=2.0, blur=1.0)
    I, P = _scene([v], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 60.0), (70, 61.1)))
    _edge(net, _line((76, 61.2), (150, 62.3)))           # 6 px gap
    _edge(net, _line((150, 62.3), (239, 63.7)))          # touching (a former node)
    _edge(net, _line((100, 20), (130, 30)), r=1.0, a=0.02)   # nothing there
    C = VesselSearch(net, P, _cfg())
    nll0 = C.energy_total()["nll"]
    out = C.run()
    assert len(out.edges) == 1, out.summary()
    (k,) = out.edges
    L = out.length(k)
    assert L > 230, L
    assert out.meta["search"]["energy_after"]["nll"] <= nll0 * 1.01


def test_crossing_stays_two_straight_vessels():
    H, W = 200, 200
    v1 = _vessel(_line((-10, 100), (210, 100)), r=2.0)
    v2 = _vessel(_line((100, -10), (100, 210)), r=1.5)
    I, P = _scene([v1, v2], (H, W))
    net = VesselNetwork((H, W))
    c = net.add_node(100, 100)
    for p, r in (((0, 100), 2.0), ((199, 100), 2.0), ((100, 0), 1.5), ((100, 199), 1.5)):
        _edge(net, _line(p, (100, 100), 100), r=r, v=c)
    out = VesselSearch(net, P, _cfg()).run()
    assert len(out.edges) == 2, out.summary()
    for k in out.edges:
        xy = out.sample(k, 1.0)["xy"]
        straight = np.ptp(xy[:, 0]) < 6 or np.ptp(xy[:, 1]) < 6
        assert straight and out.length(k) > 190, (k, np.ptp(xy, 0), out.length(k))


def test_branch_leaves_an_unsplit_parent():
    H, W = 160, 240
    parent = _vessel(_line((-10, 100), (250, 100)), r=3.0)
    branch = _vessel(_line((120, 100), (200, 10)), r=1.5)
    I, P = _scene([parent, branch], (H, W))
    net = VesselNetwork((H, W))
    j = net.add_node(120, 100)
    _edge(net, _line((0, 100), (120, 100), 120), r=3.0, v=j)
    _edge(net, _line((120, 100), (239, 100), 120), r=3.0, u=j)
    _edge(net, _line((120, 100), (200, 10), 120), r=1.5, u=j)
    out = VesselSearch(net, P, _cfg()).run()
    assert len(out.edges) == 2, out.summary()
    long_ = max(out.edges, key=out.length)
    short = min(out.edges, key=out.length)
    assert out.length(long_) > 230
    # the branch ends at a node the parent passes through: a bifurcation
    (nid,) = out.edges[long_].through
    assert nid in (out.edges[short].u, out.edges[short].v)
    assert out.node_kind(nid, out.degrees()[nid]) == "bifurcation"
    assert abs(out.nodes[nid].x - 120) < 8 and abs(out.nodes[nid].y - 100) < 3
    # the segment graph cuts the parent there, and remembers the vessels
    back = VesselNetwork.from_dict(out.to_dict())
    G = back.to_digraph()
    assert G.number_of_edges() == 3 and G.degree(nid) == 3
    assert sorted(d["vessel"] for _, _, d in G.edges(data=True)).count(long_) == 2
    assert not G.graph["crossings"]


@pytest.mark.slow
def test_search_reduces_fragmentation_on_a_scene():
    I, vessels, _ = make_scene(2, shape=(256, 384), n_trees=2, n_cross=2, n_capillary=6)
    P = prepare(I)
    frag = fragment_network(vessels, I.shape, np.random.default_rng(3))
    C = VesselSearch(frag, P, _cfg(init_iters=150, global_iters=80))
    before = vessel_metrics(C.net, vessels, I.shape)
    nll0 = C.energy_total()["nll"]
    out = C.run()
    after = vessel_metrics(out, vessels, I.shape)
    assert after["excess"] <= 0.4 * before["excess"], (before, after)
    assert after["best_cover"] > before["best_cover"] + 0.15, (before, after)
    assert after["purity"] > 0.93, after
    assert out.meta["search"]["energy_after"]["nll"] <= nll0 * 1.02
