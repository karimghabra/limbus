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


def test_truth_panels_draw(tmp_path):
    from vesselmap.draw import truth_panels
    I, vessels, _ = make_scene(0, shape=(128, 192), n_trees=1, n_cross=1, n_capillary=2)
    net = fragment_network(vessels, I.shape, np.random.default_rng(0))
    p = tmp_path / "panels.png"
    truth_panels(I, vessels, [net, VesselNetwork(I.shape)], p)
    import cv2
    img = cv2.imread(str(p))
    assert img.shape[0] == 3 * 2 * 128 and img.shape[1] == 2 * 192


def test_join_dE_is_the_energy_change_apply_makes():
    """The score of a join covers the whole footprint: after apply() the
    global energy changes by exactly the scored dE (up to the halo's
    truncation at the window edge), even where the joined vessel differs
    from its pieces far from the junction."""
    H, W = 120, 320
    x = np.linspace(-10, 330, 400)
    v = dict(xy=np.stack([x, 60 + 3 * np.sin(x / 40)], 1),
             r=1.8 + 0.4 * np.sin(x / 25), blur=1.0, amp=0.3)
    I, P = _scene([v], (H, W))
    net = VesselNetwork((H, W))
    xs = np.linspace(0, 319, 400)
    ys = 60 + 3 * np.sin(xs / 40)
    left, right = xs < 158, xs > 162
    _edge(net, np.stack([xs[left], ys[left]], 1), r=2.0)
    _edge(net, np.stack([xs[right], ys[right]], 1), r=2.0)
    C = VesselSearch(net, P, _cfg(init_iters=80, workers=1))

    def check(move):
        dE, prop = C.evaluate(move)
        dE += C._phi_delta(prop)
        E0 = C.energy_total()["total"]
        C.apply(prop)
        E1 = C.energy_total()["total"]
        assert abs((E1 - E0) - dE) < 0.02 * abs(dE) + 25.0, (move["kind"], E1 - E0, dE)
        return prop

    (move,) = [m for m in C.join_moves()]
    check(move)
    assert C.energy_total()["frag"] == 0.0         # one vessel, its ends at the border
    (k,) = C.net.edges                              # split it again where it was joined
    s = C.samples(k)
    i = int(np.argmin(np.abs(s["xy"][:, 0] - 160)))
    check(dict(kind="split", anchor=[k], key=("split", k, i), focus=s["xy"][i],
               build=C._split_builder(k, s["s_arc"][i])))
    assert len(C.net.edges) == 2 and C.energy_total()["frag"] > 0
    check(C.delete_moves()[0])


def _straight(net, p, q, r=2.5, s=1.0, n=60):
    xy = np.linspace(p, q, n)
    return net.add_edge_dense(xy, np.full(n, r), np.full(n, s), np.full(n, 0.3))


def _valid_through(net):
    """The representation's invariants after to_through."""
    for k, e in net.edges.items():
        assert e.u != e.v, ("loop", k)
        assert not set(e.through) & {e.u, e.v}, ("ends at its own through node", k)
        assert len(set(e.through)) == len(e.through)
        assert all(n in net.nodes for n in e.through)
    for k in net.edges:                            # no end folded back onto its own vessel
        smp = net.sample(k, 1.0)
        for end_xy, far in ((smp["xy"][0], smp["s_arc"] > 5.0),
                            (smp["xy"][-1], smp["s_arc"] < smp["s_arc"][-1] - 5.0)):
            d = np.linalg.norm(smp["xy"][far] - end_xy, axis=1)
            assert not len(d) or d.min() > 1.0, ("fold-back", k, float(d.min()))
    for k in net.edges:                            # no end hooks back past itself
        smp = net.sample(k, 0.5)
        xy, s = smp["xy"], smp["s_arc"]
        L = float(s[-1])
        for end, near in ((xy[0], s <= min(10.0, L)), (xy[-1], s >= max(L - 10.0, 0.0))):
            pts = xy[near]
            back = pts[-1] if end is xy[0] else pts[0]
            t = end - back
            if np.linalg.norm(t) < 1e-6:
                continue
            over = float(((pts - end) @ (t / np.linalg.norm(t))).max())
            assert over < 0.5, ("hook", k, over)
    for k, e in net.edges.items():                 # a through node lies on its edge
        xy = net.sample(k, 0.5)["xy"]
        for n in e.through:
            d = float(np.linalg.norm(xy - net.nodes[n].xy, axis=1).min())
            assert d < 1.0, ("through node off its edge", k, n, d)
    segs = net.to_segments()
    assert all(e.u != e.v for e in segs.edges.values())
    pairs = [frozenset((e.u, e.v)) for e in segs.edges.values()]
    assert len(pairs) == len(set(pairs)), "two segments between one pair of nodes"
    return segs


def test_to_through_never_ends_an_edge_at_its_own_through_node():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (60, 100), (120, 100))
    _straight(net, (64, 140), (64, 100.5))          # ends just inside k, near k's end
    to_through(net, 1.5)
    _valid_through(net)
    assert "junction" not in net.summary()["node_kinds"]


@pytest.mark.parametrize("overlap, rk, rj", [(6, 2.5, 2.5), (7, 4.0, 1.5), (7, 1.5, 4.0),
                                             (15, 2.5, 2.5), (25, 2.5, 2.5)])
def test_to_through_overlapping_ends_share_one_joint(overlap, rk, rj):
    """Two vessel ends lying on each other become one shared joint, never
    mutual through nodes, whatever the widths.  A long overlap (the ends
    would move more than about half a calibre) is left alone: the edit
    would change what the map renders, and joining a vessel found twice is
    the search's move."""
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    k = _straight(net, (60, 100), (100 if overlap > 20 else 120, 100), r=rk)
    j = _straight(net, (10, 100.3), (60 + overlap, 100.3), r=rj)
    to_through(net, 1.5)
    segs = _valid_through(net)
    assert not net.edges[k].through and not net.edges[j].through
    shared = {net.edges[k].u, net.edges[k].v} & {net.edges[j].u, net.edges[j].v}
    assert len(shared) == (1 if overlap < 10 else 0)
    assert len(segs.edges) == 2


def test_to_through_short_fragment_in_a_vessel_end_is_no_loop():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (60, 100), (120, 100), r=4.0, s=1.5)
    _straight(net, (61, 102), (69, 102), r=3.0, s=1.0, n=12)
    to_through(net, 1.5)
    _valid_through(net)


def test_to_through_three_edges_no_fold_back():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (60, 100), (100, 100))
    _straight(net, (91, 100), (119.3, 71.7))
    _straight(net, (95, 95.5), (125, 125))
    to_through(net, 1.5)
    _valid_through(net)


def test_to_through_keeps_border_ends():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    j = _straight(net, (100, 100), (100, 0))        # leaves the image at the top
    _straight(net, (60, 30), (99.5, 3.0))           # a branch landing next to the border end
    to_through(net, 1.5)
    _valid_through(net)
    deg = net.degrees()
    top = net.edges[j].v
    assert net.node_kind(top, deg[top]) == "border"


def test_split_and_rejoin_keeps_link_records_once():
    from vesselmap.search import _merge_info, _split_links, _with_links
    xy = np.stack([np.linspace(0, 100, 101), np.zeros(101)], 1)
    info = dict(links=[dict(kind="join", evidence="energy", xy=[20.0, 0.0]),
                       dict(kind="join", evidence="energy", xy=[80.0, 0.0])],
                consolidated_from=[1, 2], spacing=6.0)

    def split(info, at):
        pieces = [xy[:at + 1], xy[at:]]
        return [_with_links(info, lk) for lk in _split_links(info.get("links", []), pieces, xy[at])]

    a, b = split(info, 50)
    assert len(a["links"]) == 1 and len(b["links"]) == 1
    merged = info
    for _ in range(5):              # repeated split / join cycles
        merged = _merge_info(split(merged, 50))
    assert len(merged["links"]) == 2
    a, b = split(info, 21)          # a cut at a recorded link undoes it: the record goes
    assert "links" not in a and len(b["links"]) == 1


def test_cached_join_score_follows_changes_around_it():
    """A join is scored, then a vessel next to the junction, whose end
    faces it, is deleted.  The cached data score, corrected through its
    sensitivity, plus a fresh change of Phi must be the energy change
    applying the join then makes, although the stale score is far off."""
    H, W = 140, 360
    x = np.linspace(-10, 370, 400)
    v = dict(xy=np.stack([x, 70 + 0 * x], 1), r=np.full(400, 2.0), blur=1.0, amp=0.3)
    cross = dict(xy=np.stack([190 + 0 * x[:200], np.linspace(-10, 150, 200)], 1),
                 r=np.full(200, 1.5), blur=1.0, amp=0.25)
    side = dict(xy=np.stack([178 + 0 * x[:100], np.linspace(-10, 62, 100)], 1),
                r=np.full(100, 2.0), blur=1.0, amp=0.35)
    I, P = _scene([v, cross, side], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 70), (168, 70), 200))
    _edge(net, _line((188, 70), (359, 70), 200))
    xk = _edge(net, _line((190, 0), (190, 139), 150), r=1.5, a=0.25)
    xs = _edge(net, _line((178, 0), (178, 62), 70), r=2.0, a=0.35)  # its end faces the junction
    C = VesselSearch(net, P, _cfg(init_iters=60, workers=1, refit_tol=1e9))
    (join,) = [m for m in C.join_moves() if xk not in m["anchor"] and xs not in m["anchor"]]
    dE0, prop0 = C.evaluate(join)
    phi0 = C._phi_delta(prop0)
    stale = dE0 + phi0
    (dele,) = [m for m in C.delete_moves() if m["anchor"] == [xs]]
    C.apply(C.evaluate(dele)[1])                   # the side vessel goes
    hit, (dE, prop) = C._lookup(join)
    assert hit
    phi1 = C._phi_delta(prop)
    assert abs(phi1 - phi0) > 100, (phi0, phi1)    # the side end mattered to Phi too
    dE += phi1
    E0 = C.energy_total()["total"]
    C.apply(prop)
    change = C.energy_total()["total"] - E0
    assert abs(change - stale) > 30, (change, stale)          # the context mattered
    assert abs(change - dE) < 0.005 * abs(dE) + 3.0, (change, dE)


def test_to_through_branch_attaches_to_the_geometry_after_earlier_merges():
    """k and j overlap and are joined first (their ends move); m's end, on
    the old overlap, must then land on the vessels as they are now."""
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (60, 100), (160, 100))
    _straight(net, (10, 100.3), (90, 100.3))
    m = _straight(net, (64, 60), (64, 96))
    to_through(net, 1.5)
    _valid_through(net)
    end = net.edges[m].v
    assert any(end in e.through or end in (e.u, e.v) for k, e in net.edges.items() if k != m)


def test_to_through_side_by_side_vessels_stay_apart():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    k = _straight(net, (60, 100), (160, 100))
    j = _straight(net, (60, 104.5), (160, 104.5))
    to_through(net, 1.5)
    _valid_through(net)
    assert not {net.edges[k].u, net.edges[k].v} & {net.edges[j].u, net.edges[j].v}
    assert not net.edges[k].through and not net.edges[j].through


def test_to_through_drops_through_nodes_a_cut_removes():
    """m first becomes a branch of k near k's end; k's end is then cut back
    onto j, taking m's branch point with it.  m must be settled again, not
    left passing through a node off k."""
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (40, 100), (160, 100), r=5.0, s=2.0)
    _straight(net, (100, 40), (100, 108), r=2.0, s=1.0)
    _straight(net, (130, 140), (101.5, 104), r=1.0, s=0.8)
    to_through(net, 1.5)
    _valid_through(net)


def test_to_through_moving_a_shared_node_moves_no_end_by_its_node_alone():
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (40, 100), (160, 100), r=2.5, s=1.0)
    _straight(net, (100, 40), (100, 95.5), r=4.0, s=1.5)
    _straight(net, (97, 130), (97, 93.5), r=0.5, s=0.6)
    to_through(net, 1.5)
    _valid_through(net)


def test_cached_score_is_refitted_when_the_sparse_correction_would_mislead():
    """A faint vessel next to a strong one: deleting the strong one changes
    the faint one's cached delete score by more than the refit tolerance,
    in pixels far from the faint vessel's peak.  The lookup must either
    re-fit or return the energy change applying the move then makes."""
    H, W = 120, 360
    vA = _vessel(_line((-10, 50), (370, 50)), r=2.0, amp=0.12)
    vB = _vessel(_line((-10, 68), (370, 68)), r=3.5, amp=0.6)
    I, P = _scene([vA, vB], (H, W))
    net = VesselNetwork((H, W))
    kA = _edge(net, _line((0, 50), (359, 50)), r=2.0, a=0.12)
    kB = _edge(net, _line((0, 68), (359, 68)), r=3.5, a=0.6)
    C = VesselSearch(net, P, _cfg(workers=1))
    dels = {m["anchor"][0]: m for m in C.delete_moves()}
    C.evaluate(dels[kA])
    C.apply(C.evaluate(dels[kB])[1])
    hit, res = C._lookup(dels[kA])
    dE, prop = res if hit else C.evaluate(dels[kA])
    E0 = C.energy_total()["total"]
    C.apply(prop)
    change = C.energy_total()["total"] - E0
    tol = C.cfg.refit_tol * C.tau * C.cfg.lam_vessel
    assert abs(change - dE) < tol, (hit, change, dE, tol)


def _rand_rows(rng, vids, box=90.0):
    rows = []
    for v in vids:
        for end in (0, 1):
            a = rng.uniform(0, 2 * np.pi)
            rows.append((v, rng.uniform(0, box, 2), np.array([np.cos(a), np.sin(a)]),
                         float(rng.uniform(1, 4)), float(rng.choice([0.0, rng.uniform(0, 500)],
                                                                    p=[0.15, 0.85])), end))
    return rows


def test_mwm_matches_networkx():
    import networkx as nx
    from vesselmap.search import _mwm
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(2, 16))
        edges = [(i, j, float(rng.uniform(0.1, 10))) for i in range(n) for j in range(i + 1, n)
                 if rng.random() < 0.35]
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        ref = sum(G[i][j]["weight"] for i, j in nx.max_weight_matching(G)) if edges else 0.0
        assert abs(_mwm(edges) - ref) < 1e-9, (trial, n)


def test_phi_delta_is_exact_and_bounded():
    """The change of Phi a move makes, computed on the components it
    touches, is the change of Phi over the whole map; removing a vessel
    releases at most its ends' om, adding ends never lowers Phi, and
    removing then re-adding ends restores it (split and join are exact
    inverses)."""
    from vesselmap.search import EndGraph, SearchConfig
    cfg = SearchConfig()
    rng = np.random.default_rng(1)
    for trial in range(300):
        nv = int(rng.integers(10, 21))
        rows = _rand_rows(rng, range(nv))
        G = EndGraph(rows, cfg)
        gone_v = set(rng.choice(nv, int(rng.integers(0, 4)), replace=False).tolist())
        nn = int(rng.integers(0, 4))
        new = _rand_rows(rng, range(-1, -1 - nn, -1))
        gone = [i for i, r in enumerate(rows) if r[0] in gone_v]
        d = G.delta(gone, new)
        after = [r for r in rows if r[0] not in gone_v] + \
            [(1000 - r[0],) + r[1:] for r in new]
        assert abs(d - (EndGraph(after, cfg).total - G.total)) < 1e-6, trial
        assert G.delta([], new) >= -1e-9                           # monotone
        v = int(rng.integers(nv))
        gv = G.by_vid[v]
        assert -G.delta(gv, []) <= G.bound([v]) + 1e-9             # existence bound
        G1 = EndGraph([r for r in rows if r[0] != v], cfg)          # remove, then re-add
        back = [(-1,) + rows[i][1:] for i in gv]
        assert abs(G1.delta([], back) + G.delta(gv, [])) < 1e-6, trial


def test_phi_off_is_zero():
    H, W = 120, 240
    I, P = _scene([_vessel(_line((-10, 60), (250, 60)))], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 60), (100, 60)))
    _edge(net, _line((108, 60), (239, 60)))
    C = VesselSearch(net, P, _cfg(workers=1, lam_frag=0.0, lam_pair=0.0))
    assert C.energy_total()["frag"] == 0.0
    for m in C.all_moves():
        r = C.evaluate(m)
        if r is not None:
            assert C._phi_delta(r[1]) == 0.0
    C2 = VesselSearch(net, P, _cfg(workers=1))
    E = C2.energy_total()
    assert E["frag"] > 0 and E["matched_pairs"] == 1 and E["facing_pairs"] == 1


def _faint_scene(pieces):
    """A faint straight vessel (r 1.5, amplitude 0.15) across a 320 px wide
    image, mapped as the given pieces (x ranges)."""
    H, W = 100, 320
    I, P = _scene([_vessel(_line((-10, 50), (330, 50)), r=1.5, amp=0.15)], (H, W))
    net = VesselNetwork((H, W))
    for a, b in pieces:
        _edge(net, _line((a, 50), (b, 50), int(b - a)), r=1.5, a=0.15)
    return P, net


def test_faint_fragmented_vessel_joins_not_deleted():
    """What no per-vessel cost can do: a vessel cost high enough to join
    the pieces of a faint vessel deletes them first; Phi joins them and
    leaves existence to the price."""
    from vesselmap.render import NetworkModel
    P, net = _faint_scene([(5, 100), (108, 203), (211, 306)])
    kw = dict(workers=1, texture_null=False, lam_length=1.0)
    C = VesselSearch(net, P, _cfg(**kw))
    g, _ = C._global_model().edge_gains()
    ratio = g / np.array([C.samples(k)["L"] for k in C.net.edges]) / C.price
    assert np.all((ratio > 2) & (ratio < 6)), ratio          # faint, but above the price
    old = VesselSearch(net, P, _cfg(lam_vessel=1500.0, lam_frag=0.0, **kw)).run()
    assert len(old.edges) == 0, old.summary()
    out = VesselSearch(net, P, _cfg(**kw)).run()
    assert len(out.edges) == 1, out.summary()
    (k,) = out.edges
    assert out.length(k) > 0.9 * 300, out.length(k)


def test_isolated_faint_vessel_survives():
    P, net = _faint_scene([(5, 306)])
    out = VesselSearch(net, P, _cfg(workers=1, texture_null=False, lam_length=1.0,
                                    lam_frag=0.6)).run()
    assert len(out.edges) == 1 and out.length(list(out.edges)[0]) > 290


def test_no_junk_tail_on_a_strong_end():
    """A strong vessel ending where a texture-grade piece continues it: the
    junk end's om is 0, so joining it earns nothing (the min) and it goes."""
    H, W = 100, 320
    I, P = _scene([_vessel(_line((-10, 50), (200, 50)), r=2.0, amp=0.35)], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 50), (200, 50), 200), r=2.0, a=0.35)
    _edge(net, _line((206, 50), (236, 50), 30), r=1.5, a=0.02)
    out = VesselSearch(net, P, _cfg(workers=1)).run()
    assert len(out.edges) == 1, out.summary()
    xy = out.sample(list(out.edges)[0], 1.0)["xy"]
    assert abs(xy[:, 0].max() - 200) < 4, xy[:, 0].max()


def test_face_weight_gap_taper_and_mutual_overlap():
    """Every pair here passes the join test (it stays a candidate), but the
    fragmentation weight counts a gap only up to frag_gap (full) .. twice
    it (none), and an overlap only when each end lies on the other's line."""
    from vesselmap.search import SearchConfig, _faces, face_weight
    cfg = SearchConfig()
    E = lambda p, t, w=3.0: (np.array(p, float), np.array(t, float) / np.linalg.norm(t), w)
    faces = lambda a, b: _faces(*a, *b, cfg) or _faces(*b, *a, cfg)
    for gap, want in ((0.0, 1.0), (4.0, 1.0), (9.0, 0.5), (12.0, 0.0), (16.0, 0.0)):
        a, b = E((100, 50), (1, 0)), E((100 + gap, 50), (-1, 0))
        assert faces(a, b)
        assert abs(face_weight(a, b, cfg) - want) < 1e-6, (gap, face_weight(a, b, cfg))
    a, b = E((100, 50), (1, 0)), E((88, 50.5), (-1, 0))             # a vessel found twice
    assert faces(a, b) and face_weight(a, b, cfg) == 1.0
    # a branch leaving a parent 14 px before the parent's end: from the
    # parent's end the branch start lies behind it on its line, but not
    # the other way round
    parent_end = E((297.8, 240.5), (-0.41, 0.91), 3.7)
    branch_start = E((302.8, 227.3), (0.70, -0.71), 3.7)
    assert faces(parent_end, branch_start)
    assert face_weight(parent_end, branch_start, cfg) < 0.1


def test_end_evidence_is_the_same_in_any_model():
    """The same vessel gets the same end evidence whether it is rendered
    alone on a window or among other vessels, and a short vessel's halves
    never both count its middle sample."""
    from vesselmap.render import NetworkModel
    from vesselmap.search import end_evidence
    rng = np.random.default_rng(0)
    for trial in range(12):
        L = rng.uniform(8, 55)
        ang, c = rng.uniform(0, np.pi), rng.uniform(40, 150, 2)
        t = np.linspace(-L / 2, L / 2, 40)
        bend = rng.uniform(-0.02, 0.02) * t ** 2
        arc = c + np.stack([t * np.cos(ang) - bend * np.sin(ang), t * np.sin(ang) + bend * np.cos(ang)], 1)
        shape = (200, 300)
        big = VesselNetwork(shape)
        for i in range(6):
            _edge(big, np.linspace((5, 10 + 30 * i), (295, 12 + 30 * i), 200))
        _edge(big, arc)
        off = np.floor(arc.min(0)) - 20
        alone = VesselNetwork(shape)
        _edge(alone, arc - off)
        res = []
        for net in (big, alone):
            net.background = np.zeros((5, 6), np.float32)
            net.bg_spacing = 64.0
            m = NetworkModel(net, np.zeros(shape, np.float32), np.ones(shape, np.float32), stride=1)
            S0, S1, l = end_evidence(m, 0.3, 30.0)[-1]
            ent = m.vessel_entries().detach()
            tot = float((0.5 * (0.7 * ent[m.e_edge == len(m.eids) - 1]) ** 2).sum())
            res.append((S0, S1))
        assert np.allclose(res[0], res[1], rtol=1e-4), (trial, res)
        assert abs(res[1][0] + res[1][1] - tot) < 1e-3 * tot, (trial, res[1], tot)


def test_face_weight_has_no_jump_where_ends_pass_each_other():
    from vesselmap.search import SearchConfig, face_weight
    cfg = SearchConfig()
    E = lambda p, t, w: (np.array(p, float), np.array(t, float), w)
    for w1, w2, y in ((1.0, 11.0, 9.75), (1.5, 10.0, 8.0), (1.77, 13.92, 11.0), (3.34, 8.61, 7.0)):
        g = [face_weight(E((0, 0), (1, 0), w1), E((x, y), (-1, 0), w2), cfg) for x in (-0.01, 0.01)]
        if None not in g:
            assert abs(g[0] - g[1]) < 0.05, (w1, w2, y, g)


def test_move_log_parts_add_up_and_temperature_is_the_draws():
    H, W = 120, 240
    I, P = _scene([_vessel(_line((-10, 60), (250, 64)), r=2.0)], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 60.0), (70, 61.1)))
    _edge(net, _line((76, 61.2), (150, 62.3)))
    _edge(net, _line((150, 62.3), (239, 63.7)))
    _edge(net, _line((100, 20), (130, 30)), r=1.0, a=0.02)
    C = VesselSearch(net, P, _cfg(workers=1))
    C.run()
    log = C.move_log
    assert log
    for r in log:
        assert abs(r["nll"] + r["prior"] + r["cost"] + r["frag"] - r["dE"]) < 0.3, r
    T0 = C.cfg.t_start * C.tau * C.cfg.t_scale
    assert abs(log[0]["T"] - round(T0, 2)) < 0.02 or log[0]["step"] > 0


def test_search_report_attributes_deletions():
    from vesselmap.synthetic import search_report
    vessels = [_vessel(_line((10, 50), (190, 50)))]
    xy = np.stack([np.linspace(20, 180, 81), np.full(81, 50.0)], 1).round(1).tolist()
    base = dict(kind="delete", L=160.0, frag=0.0, xy=xy, nll=0.0, prior=0.0)
    recs = [dict(base, dE=20.0, loss=1400.0, cost=-1380.0, grave=0),     # uphill: hot
            dict(base, dE=-10.0, loss=500.0, cost=-1380.0, grave=1),     # below the price
            dict(base, dE=-10.0, loss=1300.0, cost=-1380.0, grave=2),    # existence cost
            dict(base, dE=-10.0, loss=1500.0, cost=-1380.0, frag=-200.0, grave=3),   # Phi
            dict(base, dE=-10.0, loss=1500.0, cost=-1380.0, frag=-200.0, grave=4)]   # revived
    net = VesselNetwork((100, 200))
    net.meta["search"] = dict(price_per_px=6.0, tau=5.0, config=dict(lam_vessel=50.0),
                              deleted=recs, moves=recs + [dict(kind="revive", grave=4)])
    rep = search_report(net, vessels, (100, 200))
    assert rep["deleted"] == 4 and rep["deleted_true"] == 4
    assert rep["deleted_true_by"] == dict(hot=1, price=1, lam=1, phi=1, other=0), rep


def test_to_through_refused_edits_are_undone():
    """An edit the accept hook refuses leaves the network exactly as it was
    (geometry, nodes and through lists), and every edit is shown to it."""
    from vesselmap.search import to_through
    net = VesselNetwork((200, 200))
    _straight(net, (40, 100), (160, 100), r=2.5)
    _straight(net, (100, 40), (100, 95.5), r=2.0)          # a branch
    _straight(net, (10, 100.3), (46, 100.3), r=2.5)        # an overlapping piece
    before = net.copy()
    seen = []
    to_through(net, 1.5, accept=lambda edges, ids, focus: seen.append(ids) or False)
    assert len(seen) >= 2
    assert set(net.nodes) == set(before.nodes) and set(net.edges) == set(before.edges)
    for k, e in net.edges.items():
        assert np.array_equal(e.ctrl, before.edges[k].ctrl) and not e.through
        assert (e.u, e.v) == (before.edges[k].u, before.edges[k].v)
    kept = net.copy()
    to_through(kept, 1.5, accept=lambda edges, ids, focus: True)
    plain = before.copy()
    to_through(plain, 1.5)
    assert sum(len(e.through) for e in kept.edges.values()) == \
        sum(len(e.through) for e in plain.edges.values()) == 1


@pytest.mark.slow
def test_parallel_scoring_with_a_cold_compile_cache(tmp_path):
    """Worker processes are forked from the search; torch.compile may have
    to compile again in them, and must not wait for the parent's compile
    workers (it hung with an empty inductor cache)."""
    import os
    import subprocess
    import sys
    code = """
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, %r); sys.path.insert(0, %r)
from test_search import _scene, _edge, _line, _cfg, _vessel
from vesselmap.search import VesselSearch
from vesselmap.network import VesselNetwork
I, P = _scene([_vessel(_line((-10, 60), (250, 64)), r=2.0)], (120, 240))
net = VesselNetwork((120, 240))
for a, b in (((0, 60.0), (70, 61.1)), ((76, 61.2), (150, 62.3)), ((150, 62.3), (239, 63.7))):
    _edge(net, _line(a, b))
C = VesselSearch(net, P, _cfg(workers=2))
moves = C.all_moves()
C._evaluate_all(moves * 3)
print("scored", len(moves))
""" % (os.path.dirname(os.path.dirname(os.path.dirname(__file__))), os.path.dirname(__file__))
    env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"))
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and "scored" in out.stdout, out.stderr[-2000:]


def test_relaxed_move_is_conservative_and_renders_neighbours_whole():
    """A move scored with the vessels near it re-fitted too (relax_radius)
    lowers E by at least its score once applied (the vessels before it are
    re-fitted for the score as well), and the neighbours it re-fitted, which
    reach beyond the move's window, are rendered whole."""
    H, W = 160, 240
    parent = _vessel(_line((-10, 100), (250, 100)), r=3.0)
    branch = _vessel(_line((120, 100), (200, 10)), r=1.5)
    I, P = _scene([parent, branch], (H, W))
    net = VesselNetwork((H, W))
    _edge(net, _line((0, 100), (117, 100), 118), r=3.0)
    _edge(net, _line((123, 100), (239, 100), 117), r=3.0)
    _edge(net, _line((121, 97), (200, 10), 118), r=1.5)
    C = VesselSearch(net, P, _cfg(workers=1, relax_radius=12.0))
    (move,) = [m for m in C.join_moves()
               if all(C.samples(k)["xy"][:, 1].std() < 1.0 for k in m["anchor"])]
    C.cfg.relax_radius = 0.0
    _, plain = C._compute(move)
    C.cfg.relax_radius = 12.0
    rel = C._relaxed(plain[0], plain[1])
    assert rel is not None and len(rel[0]) == 3            # the branch is re-fitted too
    dE, prop = C._proposal(rel[0], rel[1], rel[2], rel[3][0], rel[3][1], rel[4], False, rel[5])
    E0 = C.energy_total()["total"]
    C.apply(prop)
    E1 = C.energy_total()["total"]
    assert E1 - E0 <= dE + 0.02 * abs(dE) + 5.0, (E1 - E0, dE)
    assert len(C.net.edges) == 2
    fresh = VesselSearch(C.net, P, _cfg(init_iters=0, workers=1))
    assert np.abs(fresh.V - C.V).max() < 1e-3


def test_short_end_is_extended_along_its_vessel():
    """A trace that stops short of its vessel's end is lengthened (extend
    moves follow the dark ridge the model leaves beyond the end) to within
    a couple of px of the end, and no further."""
    H, W = 120, 200
    x = np.linspace(20, 170, 300)
    v = dict(xy=np.stack([x, 60 + 8 * np.sin(x / 30)], 1), r=np.full(300, 1.2), blur=1.0, amp=0.3)
    I, P = _scene([v], (H, W))
    net = VesselNetwork((H, W))
    keep = x < 150                                  # the trace stops 20 px short
    _edge(net, v["xy"][keep], r=1.2, s=1.0, a=0.3)
    C = VesselSearch(net, P, _cfg(workers=1))
    assert any(m["kind"] == "extend" for m in C.all_moves(("extend",)))
    C.anneal(0.0, 0, kinds=("extend", "trim"))
    (k,) = C.net.edges
    xy = C.samples(k)["xy"]
    far = xy[np.argmax(xy[:, 0])]
    assert np.linalg.norm(far - v["xy"][-1]) < 3.0, far
    near = xy[np.argmin(xy[:, 0])]
    assert np.linalg.norm(near - v["xy"][0]) < 3.0, near


def test_vessels_traced_across_a_shallow_crossing_are_swapped_back():
    """Two vessels crossing at a shallow angle, each traced into the
    other's far half (two V-shaped traces meeting at the crossing), become
    the two straight vessels again (swap moves)."""
    H, W = 140, 260
    ang = np.radians(25.0)
    c = np.array([130.0, 70.0])
    u1 = np.array([np.cos(ang / 2), np.sin(ang / 2)])
    u2 = np.array([np.cos(ang / 2), -np.sin(ang / 2)])
    t = np.linspace(-150, 150, 301)[:, None]
    a, b = c + t * u1, c + t * u2
    I, P = _scene([_vessel(a, r=1.3, amp=0.35), _vessel(b, r=1.3, amp=0.35)], (H, W))
    net = VesselNetwork((H, W))
    inside = lambda xy: xy[(xy[:, 0] >= 0) & (xy[:, 0] <= W - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= H - 1)]
    _edge(net, inside(np.vstack([a[:151], b[151:]])), r=1.3, s=1.0, a=0.35)
    _edge(net, inside(np.vstack([b[:151], a[151:]])), r=1.3, s=1.0, a=0.35)
    C = VesselSearch(net, P, _cfg(workers=1))
    assert any(m["kind"] == "swap" for m in C.all_moves(("swap",)))
    C.anneal(0.0, 0, kinds=("swap",))
    assert len(C.net.edges) == 2
    for k in C.net.edges:
        xy = C.samples(k)["xy"]
        d = np.abs((xy - xy.mean(0)) @ np.array([-(xy[-1] - xy[0])[1], (xy[-1] - xy[0])[0]])
                   / np.linalg.norm(xy[-1] - xy[0]))
        assert d.max() < 2.0, d.max()                   # straight again
