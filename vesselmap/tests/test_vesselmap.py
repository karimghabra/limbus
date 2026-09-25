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


def test_parallel_split_detects_two_close_vessels():
    """A single edge fitted over two parallel vessels 7 px apart is split."""
    from vesselmap.image import prepare
    from vesselmap.refine import RefineConfig, apply_split, parallel_candidates
    from vesselmap.synthetic import render
    rng = np.random.default_rng(0)
    H, W = 120, 200
    ys = np.full(200, 60.0)
    xs = np.linspace(-10, 210, 200)
    vessels = [dict(xy=np.stack([xs, ys - 3.5], 1), r=np.full(200, 1.3), blur=0.9, amp=0.35),
               dict(xy=np.stack([xs, ys + 3.5], 1), r=np.full(200, 1.3), blur=0.9, amp=0.35)]
    I, _ = render(vessels, (H, W), rng)
    P = prepare(I)
    net = VesselNetwork((H, W))
    _line(net, (0, 60), (199, 60), r=4.0, s=2.0, a=0.3, n=100)
    # residual of a model that explains nothing but a flat background
    R = P.logI - np.median(P.logI) - np.zeros_like(P.logI)
    cands = parallel_candidates(net, P, R, RefineConfig())
    assert cands, "no parallel stretch found"
    c = max(cands, key=lambda c: c["s1"] - c["s0"])
    sep = np.median(np.abs(c["u2"] - c["u1"]))
    assert 5.5 < sep < 8.5, sep
    e1, e2 = apply_split(net, c)
    y1 = np.median(net.sample(e1, 1.0)["xy"][:, 1])
    y2 = np.median(net.sample(e2, 1.0)["xy"][:, 1])
    assert abs(abs(y1 - y2) - 7.0) < 1.5


# ------------------------------------------------------------ consolidation
def _link_all(net):
    from vesselmap.consolidate import ConsolidateConfig, build, candidates, chains, match
    cc = ConsolidateConfig()
    c = candidates(net, cc)
    out, rec = build(net, chains(net, match(c)), cc)
    return c, out, rec


def test_consolidate_vessel_through_two_branch_points():
    net = VesselNetwork((300, 400))
    e = _line(net, (10, 150), (390, 160), r=4)
    n1 = net.split_edge(e, (130, 153))
    e2 = [k for k in net.edges if net.edges[k].u == n1][0]
    n2 = net.split_edge(e2, (260, 157))
    _line(net, (130, 153), (180, 60), r=2, u=n1)
    _line(net, (260, 157), (300, 260), r=2, u=n2)
    L = sum(net.length(k) for k in net.edges)
    cands, out, _ = _link_all(net)
    assert sorted(c["kind"] for c in cands) == ["node", "node"]
    assert len(out.edges) == 3                       # the trunk and two branches
    trunk = max(out.edges, key=out.length)
    assert sorted(out.edges[trunk].through) == sorted([n1, n2])
    assert out.summary()["node_kinds"] == {"endpoint": 4, "bifurcation": 2}
    assert abs(sum(out.length(k) for k in out.edges) - L) < 2.0
    assert out.crossings() == []                     # branches leave at shared nodes
    # the segment view and the graph are unchanged, labelled by vessel
    seg = out.to_segments()
    assert len(seg.edges) == 5 and seg.summary()["node_kinds"] == out.summary()["node_kinds"]
    G = out.to_digraph()
    assert G.number_of_edges() == 5 and G.number_of_nodes() == 6
    assert sorted(d["vessel"] for *_, d in G.edges(data=True)).count(trunk) == 3


def test_consolidate_leaves_symmetric_fork_alone():
    net = VesselNetwork((300, 300))
    p = _line(net, (150, 290), (150, 150), r=3)
    nid = net.edges[p].v
    _line(net, (150, 150), (80, 30), r=2.5, u=nid)
    _line(net, (150, 150), (220, 30), r=2.5, u=nid)
    cands, out, _ = _link_all(net)
    assert cands == [] and len(out.edges) == 3


def test_consolidate_bridges_gap_and_blends_overlap():
    net = VesselNetwork((200, 300))
    _line(net, (10, 100), (120, 102))
    _line(net, (145, 103), (290, 108))
    cands, out, rec = _link_all(net)
    assert [c["kind"] for c in cands] == ["gap"] and len(out.edges) == 1
    assert len(out.nodes) == 2 and abs(out.length(next(iter(out.edges))) - 280.2) < 2.0
    net = VesselNetwork((200, 300))
    _line(net, (10, 100), (150, 101))
    _line(net, (120, 101.5), (290, 104))
    cands, out, _ = _link_all(net)
    assert [c["kind"] for c in cands] == ["overlap"] and len(out.edges) == 1
    xy = out.sample(next(iter(out.edges)), 1.0)["xy"]
    assert abs(out.length(next(iter(out.edges))) - 280.0) < 2.0
    assert np.abs(np.diff(xy[:, 0])).min() > 0.5          # no back-tracking in the blend


def test_consolidate_rejects_bridge_without_evidence():
    """Two collinear pieces: bridged where the vessel is visible in the gap,
    left apart where the gap is empty (two different vessels)."""
    from vesselmap.consolidate import ConsolidateConfig, consolidate_map
    from vesselmap.fit import MapConfig
    from vesselmap.synthetic import render
    H, W = 100, 260
    xs = np.linspace(-10, 270, 300)
    out = {}
    for name, keep in (("visible", lambda x: np.ones_like(x, bool)),
                       ("empty", lambda x: (x < 112) | (x > 138))):
        m = keep(xs)
        pieces = [(xs[m & (xs < 130)]), (xs[m & (xs >= 130)])] if name == "empty" else [xs]
        vessels = [dict(xy=np.stack([p, np.full(len(p), 50.0)], 1), r=np.full(len(p), 2.0),
                        blur=1.0, amp=0.35) for p in pieces]
        I, _ = render(vessels, (H, W), np.random.default_rng(1))
        net = VesselNetwork((H, W))
        _line(net, (0, 50), (110, 50), r=2.0, s=1.0, a=0.35)
        _line(net, (140, 50), (259, 50), r=2.0, s=1.0, a=0.35)
        res = consolidate_map(I, net, MapConfig(verbose=False),
                              ConsolidateConfig(iters=60, verbose=False))
        out[name] = res
    assert len(out["visible"].edges) == 1
    assert len(out["empty"].edges) == 2


def test_consolidated_map_roundtrip_and_frame_fit(tmp_path):
    """A vessel with a branch leaving mid-way survives save/load, and a
    per-frame fit keeps the branch point on the vessel."""
    from scipy import ndimage as ndi
    from vesselmap.fit import FrameFitConfig, fit_frame
    from vesselmap.synthetic import render
    H, W = 160, 240
    xs = np.linspace(-10, 250, 300)
    trunk = np.stack([xs, 80 + 8 * np.sin(xs / 50)], 1)
    b0 = trunk[140]
    branch = b0 + np.stack([np.linspace(0, 60, 80), np.linspace(0, -70, 80)], 1)
    I, _ = render([dict(xy=trunk, r=np.full(300, 3.0), blur=1.0, amp=0.4),
                   dict(xy=branch, r=np.full(80, 1.8), blur=1.0, amp=0.4)], (H, W),
                  np.random.default_rng(0))
    net = VesselNetwork((H, W))
    e = net.add_edge_dense(trunk[5:-5], np.full(290, 3.0), np.full(290, 1.0), np.full(290, 0.4))
    n = net.split_edge(e, b0)
    net.add_edge_dense(branch, np.full(80, 1.8), np.full(80, 1.0), np.full(80, 0.4), u=n)
    _, out, _ = _link_all(net)
    assert len(out.edges) == 2 and any(out.edges[k].through == [n] for k in out.edges)
    p = tmp_path / "v.json"
    out.save(p)
    back = VesselNetwork.load(p)
    assert back.summary()["node_kinds"] == out.summary()["node_kinds"]
    assert [e.through for e in back.edges.values()] == [e.through for e in out.edges.values()]
    shifted = ndi.shift(I, (1.0, -2.0), order=3, mode="nearest")
    fitted, rep = fit_frame(shifted, back, FrameFitConfig(iters=20, iters_align=15))
    trunk_id = next(k for k in fitted.edges if fitted.edges[k].through)
    xy = fitted.sample(trunk_id, 0.25)["xy"]
    assert np.linalg.norm(xy - fitted.nodes[n].xy, axis=1).min() < 0.2
    assert fitted.summary()["node_kinds"] == out.summary()["node_kinds"]


def test_faint_tracking_finds_sub_threshold_line_and_joins_map():
    """A line whose per-pixel contrast is below the noise is found by
    integrating along it, joined to the mapped vessel it meets, and the
    opposite polarity (the null) yields nothing comparable."""
    from vesselmap.faint import FaintConfig, add_faint, map_zone, search_mask, trace_faint
    rng = np.random.default_rng(3)
    H, W = 240, 320
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    # faint vessel: y = 60 + 0.5 x for x in [40, 260], Gaussian profile sigma 1.5
    d = np.abs(yy - 60 - 0.5 * xx) / np.sqrt(1.25)
    on = (xx >= 40) & (xx <= 260)
    noise = rng.normal(0, 1.0, (H, W))
    res = noise - 0.8 * np.exp(-0.5 * (d / 1.5) ** 2) * on
    net = VesselNetwork((H, W))
    _line(net, (262, 20), (262, 230), r=2.0)          # mapped vessel the line runs into
    zone = map_zone(net, (H, W))
    cfg = FaintConfig(fit_profiles=False)
    tracks = trace_faint(res, zone, cfg)
    # the null: the same noise without the line, both polarities
    null = trace_faint(noise, zone, cfg) + trace_faint(noise, zone, cfg, polarity=-1)
    assert sum(t["L"] for t in null) < 40
    assert tracks, "no track found"
    xy = np.concatenate([t["xy"] for t in tracks])
    err = np.abs(xy[:, 1] - 60 - 0.5 * xy[:, 0]) / np.sqrt(1.25)
    assert np.median(err) < 2.0
    covered = np.ptp(xy[err < 3, 0])
    assert covered > 150
    assert xy[:, 0].min() > 40 - 25                  # little overshoot
    new = add_faint(net, tracks, cfg)
    assert new and all(net.edges[e].info["tier"] == "faint" for e in new)
    # the track that reached the mapped vessel is connected to it
    assert any(net.degrees()[n] >= 3 for n in net.nodes)
    m = search_mask(net, (H, W))
    assert (m == 1).any() and (m == 2).any()
