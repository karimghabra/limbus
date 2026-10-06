"""Fast tests of vesselnet.metrics' topology on hand-made graphs and junction truth."""
import numpy as np

from vesselmap.network import VesselNetwork

from vesselnet import metrics as M


def _edge(net, p, q, u=None, v=None):
    xy = np.linspace(p, q, 100)
    n = len(xy)
    return net.add_edge_dense(xy, np.full(n, 2.0), np.full(n, 1.0), np.full(n, 0.3), u=u, v=v)


def _arm(xy, d=(1, 0)):
    return dict(xy=list(map(float, xy)), visibility="visible", in_frame=True, width_px=4.0, dir=list(d))


# a crossing at (100, 100) of a horizontal and a vertical vessel; a fork at (100, 200): parent from the
# left, daughters up-right and down-right (strict convention: three vessels, three groups)
CROSS = dict(type_visible="crossing", ambiguous=False, partition=[[0, 1], [2, 3]],
             arms=[_arm((80, 100)), _arm((120, 100)), _arm((100, 80)), _arm((100, 120))],
             members=[dict(type="crossing", xy=[100.0, 100.0], vessels=[1, 2])])
FORK = dict(type_visible="bifurcation", ambiguous=False, partition=[[0], [1], [2]],
            arms=[_arm((80, 200)), _arm((115, 185)), _arm((115, 215))],
            members=[dict(type="bifurcation", xy=[100.0, 200.0], vessels=[3, 4, 5])])


def _strict():
    net = VesselNetwork((300, 300))
    _edge(net, (50, 100), (150, 100))
    _edge(net, (100, 50), (100, 150))
    p = _edge(net, (50, 200), (100, 200))
    f = net.edges[p].v
    _edge(net, (100, 200), (140, 160), u=f)
    _edge(net, (100, 200), (140, 240), u=f)
    return net


def test_the_true_architecture_scores_perfectly():
    t = M.topology(_strict(), [CROSS, FORK])
    assert t["crossings_as_nodes"] == 0 and t["forks_found"] == 1
    assert t["pairing_accuracy"] == 1.0 and t["false_joins"] == 0 and t["false_splits"] == 0, t


def test_window_truth_cuts_runs_and_reindexes_unobserved():
    obs = dict(runs=[dict(vid=1, xy=[[x, 50.0] for x in range(0, 300)], unobserved=[[150, 159]])],
               junctions=[dict(x=120.0, y=50.0, radius=5.0, id=0), dict(x=20.0, y=50.0, radius=5.0, id=1)],
               dont_care_junctions=[])
    o, js = M.window_truth(obs, [dict(CROSS, x=120.0, y=60.0)], 0, 100, 100)
    assert len(o["runs"]) == 1
    r = o["runs"][0]
    assert r["xy"][0] == [0.0, 50.0] and r["xy"][-1] == [99.0, 50.0]
    assert r["unobserved"] == [[50, 59]]
    assert [j["id"] for j in o["junctions"]] == [0] and o["junctions"][0]["x"] == 20.0
    assert len(js) == 1 and js[0]["members"][0]["xy"] == [0.0, 100.0]


def test_a_crossing_made_a_node_and_a_through_edge_are_errors():
    net = VesselNetwork((300, 300))
    c = net.add_node(100, 100)
    for q in ((50, 100), (150, 100), (100, 50), (100, 150)):     # four pieces meeting at the crossing
        _edge(net, q, (100, 100), v=c)
    xy = np.r_[np.linspace((50, 200), (100, 200), 50), np.linspace((100, 200), (140, 160), 50)[1:]]
    n = len(xy)
    net.add_edge_dense(xy, np.full(n, 2.0), np.full(n, 1.0), np.full(n, 0.3))   # parent runs on into a daughter
    _edge(net, (100, 200), (140, 240))                           # ... the other is a separate piece
    t = M.topology(net, [CROSS, FORK])
    assert t["crossings_as_nodes"] == 1
    assert t["pairing_accuracy"] == 0.0
    assert t["false_splits"] == 2                                # both crossing vessels broken
    assert t["false_joins"] == 1                                 # parent joined to a daughter
