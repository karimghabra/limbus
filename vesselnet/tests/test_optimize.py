"""Fast tests of vesselnet.optimize on a small scene rendered by vesselmap's own synthetic renderer."""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from vesselmap.image import prepare  # noqa: E402
from vesselmap.network import VesselNetwork  # noqa: E402
from vesselmap.synthetic import render  # noqa: E402

from vesselnet import optimize as O  # noqa: E402


def _scene():
    xy = np.linspace((-10, 50), (170, 56), 300)
    v = dict(xy=xy, r=np.full(len(xy), 2.0), blur=1.0, amp=0.35)
    I, _ = render([v], (100, 160), np.random.default_rng(0))
    net = VesselNetwork((100, 160))
    n = len(xy)
    net.add_edge_dense(xy, np.full(n, 2.0), np.full(n, 1.0), np.full(n, 0.35))
    return I, net


def test_clip_to_frame_keeps_the_inside_and_the_fork_node():
    net = VesselNetwork((100, 160))
    line = lambda p, q: np.linspace(p, q, 200)
    n = 200
    one = lambda xy, **kw: net.add_edge_dense(xy, np.full(n, 2.0), np.full(n, 1.0), np.full(n, 0.3), **kw)
    k0 = one(line((-30, 50), (80, 50)))                          # leaves the frame on the left
    k1 = one(line((80, 50), (150, 20)), u=net.edges[k0].v)       # a fork at (80, 50), wholly inside
    k2 = one(line((80, 50), (200, 90)), u=net.edges[k0].v)       # leaves on the right
    c = O.clip_to_frame(net)
    assert len(c.edges) == 3 and k1 in c.edges and k0 not in c.edges and k2 not in c.edges
    fork = net.edges[k0].v
    assert len(c.incident(fork)) == 3                            # the fork stays connected
    for k, e in c.edges.items():
        xy = c.sample(k, 0.5)["xy"]
        assert xy[:, 0].min() >= -0.5 and xy[:, 0].max() <= 159.5
        assert e.info.get("clip_err_px", 0.0) < 0.5


def test_fit_local_moves_only_what_is_near_the_focus():
    I, truth = _scene()
    P = prepare(I)
    net, info = O.oracle(truth, P, iters=30)
    k = next(iter(net.edges))
    net.edges[k].ctrl = net.edges[k].ctrl + np.array([0.0, 1.5])     # off the vessel: the fit will move it back
    for n in (net.edges[k].u, net.edges[k].v):                        # (edge ends follow their nodes)
        net.nodes[n].y += 1.5
    before = net.copy()
    S = O._search(net, P, info["tau"], info["price"])
    O.fit_local(net, P, [(30.0, 51.5)], 15.0, 20, S)                 # near the left end only
    c0, c1 = before.edges[k].ctrl, net.edges[k].ctrl
    near = np.linalg.norm(c0 - [30.0, 51.5], axis=1) <= 15.0
    assert near.any() and not near.all()
    assert np.allclose(c0[~near], c1[~near])                         # far control points held
    assert np.abs(c1[near] - c0[near]).max() > 0.05                  # near ones fitted
    assert np.allclose(before.background, net.background)            # background held


def test_to_unit_keeps_nan_and_scales_12_bit():
    a = np.array([[0.0, 4095.0], [np.nan, 2047.5]], np.float32)
    u = O.to_unit(a)
    assert np.isnan(u[1, 0]) and u[0, 1] == 1.0 and abs(u[1, 1] - 0.5) < 1e-6


def test_oracle_holds_centrelines_and_deleting_the_vessel_raises_E():
    I, truth = _scene()
    P = prepare(I)
    net, info = O.oracle(truth, P, iters=60)
    clipped = O.clip_to_frame(truth)
    k = next(iter(net.edges))
    assert np.allclose(net.edges[k].ctrl, clipped.edges[k].ctrl, atol=1e-3)   # centrelines held fixed
    assert info["tau"] >= 1.0 and info["price"] > 0
    empty = net.copy()
    empty.remove_edge(k)
    E0 = O.energy(net, P, info["tau"], info["price"])
    E1 = O.energy(empty, P, info["tau"], info["price"])
    assert E0["n_vessels"] == 1 and E1["n_vessels"] == 0
    assert E1["total"] > E0["total"] + 100, (E0, E1)                      # the vessel pays for itself
    assert E1["tau"] == E0["tau"] and E1["price"] == E0["price"]          # one scale for both
