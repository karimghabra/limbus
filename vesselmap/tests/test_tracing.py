"""Tests for tracing vessels in position x orientation (tracing.py) on zoo rows."""
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from vesselmap import intersections as X
from vesselmap import tracing as T
from vesselmap.image import prepare
from vesselmap.zoo import ROWS, ROWS_CROSSINGS, zoo_sheet


def _traced(rows, seed=0):
    I, V, tiles = zoo_sheet(seed, rows=rows)
    P = prepare(I)
    inside = np.zeros(P.logI.shape, bool)              # no scene between the tiles
    for t in tiles:
        x0, y0, x1, y1 = t["box"]
        inside[y0:y1 + 1, x0:x1 + 1] = True
    U, S = T.orientation_score(P.logI, P.valid)
    traces = T.trace_all(U, P.valid & inside, S)
    return V, tiles, traces, T.trace_events(traces), X.darkness(P.logI)


def test_traces_go_straight_through_every_crossing_of_a_mesh():
    """Grids of straight vessels of r 0.8-4: at every crossing (marks too
    close to tell apart counted once) two traces cross each other."""
    V, tiles, traces, events, D = _traced([ROWS_CROSSINGS[4]])
    cross = [e["xy"] for e in events if e["kind"] == "cross"]
    for t in tiles:
        for g in X._groups(t["ambiguous"], [V[i] for i in t["vessels"]]):
            c = np.mean(g, 0)
            assert any(np.linalg.norm(q - c) <= 5.0 for q in cross), (t["value"], c)


def test_a_kiss_is_not_taken_for_a_crossing():
    """Two vessels that touch and part: their traces meet but part on the
    side they came from, or cross where the point is only as dark as one
    vessel, or one is cut into pieces ending on the other from one side;
    nothing is added there."""
    V, tiles, traces, events, D = _traced([r for r in ROWS if r[0].startswith("kissing")])
    added = T.combine([], traces, events, D)
    for t in tiles:
        near = [j for j in added if np.linalg.norm(j["xy"] - t["ambiguous"][0]) <= 10.0]
        assert not near, (t["value"], [(j["kind"], j["events"]) for j in near])


def test_a_capillary_through_a_blurred_vessel_is_one_crossing():
    """A sharp capillary crossing a wide blurred vessel at 20 degrees is
    lost inside it: traced as two pieces ending on it from opposite sides,
    in line, which make one crossing ('pass') where the detector finds
    none."""
    V, tiles, traces, events, D = _traced([r for r in ROWS if r[0].startswith("crossing, capillary")])
    added = T.combine([], traces, events, D)
    t = [t for t in tiles if t["value"] == 20][0]
    p = t["ambiguous"][0]
    near = [j for j in added if np.linalg.norm(j["xy"] - p) <= 4.0 + 4.0]
    assert any("pass" in j["events"] and j["kind"] == "crossing" for j in near), [(j["xy"] - p, j["events"]) for j in added]


def _line(p0, p1, w=1.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = int(np.linalg.norm(p1 - p0)) + 1
    return dict(xy=p0 + np.linspace(0.0, 1.0, n)[:, None] * (p1 - p0), w=np.full(n, w))


def test_an_arm_counts_only_where_a_traced_vessel_reaches_the_point():
    """On a lone traced vessel, a third arm with no trace along it (a streak
    of texture) is not a vessel, nor is a neighbour whose trace runs past 8
    px away; a branch whose trace stops 5 px short is, and so is one cut
    back 15 px short whose line runs on through the point, but not one
    whose line misses it by 6 px.  Each traced direction matches one arm."""
    import math
    p = np.array([50.0, 50.0])
    host = _line([10, 50], [90, 50])
    up = -math.pi / 2
    assert T.arm_support(p, [0.0, math.pi, up], [host]) == 2
    assert T.arm_support(p, [0.0, math.pi, -0.38], [host, _line([10, 42], [90, 42])]) == 2
    assert T.arm_support(p, [0.0, math.pi, up], [host, _line([50, 5], [50, 45])]) == 3
    assert T.arm_support(p, [0.0, math.pi, up], [host, _line([50, 0], [50, 35])]) == 3
    assert T.arm_support(p, [0.0, math.pi, up], [host, _line([56, 0], [56, 35])]) == 2
    assert T.arm_support(p, [0.0, 0.2, math.pi], [host]) == 2


def _det(xy, arms, **kw):
    return dict(xy=np.asarray(xy, float), arms=list(arms), z=1.0, **kw)


def test_the_ends_of_a_shallow_overlap_are_one_junction_where_the_centrelines_cross():
    """Two traced vessels crossing at 10 degrees lie on top of each other
    for a stretch; detections at its two ends (each looks like a fork) are
    one junction, placed where the centrelines cross.  A lone detection at
    one end is moved there too, unless the traces placed it themselves."""
    import math
    a = _line([0, 50], [100, 50])
    b = _line([0, 50 - 50 * math.tan(math.radians(10))], [100, 50 + 50 * math.tan(math.radians(10))])
    y_end = _det([40, 49.1], [0.1, math.pi, math.pi + 0.1])
    other = _det([60, 50.9], [0.0, 0.1, math.pi + 0.1])
    out = T.one_per_region([y_end, other], [a, b], [])
    assert len(out) == 1 and np.linalg.norm(out[0]["xy"] - [50, 50]) < 0.5, [d["xy"] for d in out]
    out = T.one_per_region([y_end], [a, b], [])
    assert np.linalg.norm(out[0]["xy"] - [50, 50]) < 0.5
    out = T.one_per_region([dict(y_end, events=["pass"])], [a, b], [])
    assert np.allclose(out[0]["xy"], y_end["xy"])


def test_two_crossings_of_the_same_vessels_with_a_gap_between_stay_two():
    """A vessel winding across another crosses it twice, 40 px apart, and
    runs 9 px clear of it between: two junctions, each where it was."""
    import math
    a = _line([0, 50], [100, 50])
    pts = [(10, 60), (30, 50), (50, 40), (70, 50), (90, 60)]
    xy = np.vstack([_line(p, q)["xy"][:-1] for p, q in zip(pts[:-1], pts[1:])] + [np.array([pts[-1]], float)])
    b = dict(xy=xy, w=np.ones(len(xy)))
    dets = [_det([30, 50], [0.0, math.pi, 2.68, 5.82]), _det([70, 50], [0.0, math.pi, 0.46, 3.6])]
    out = T.one_per_region(dets, [a, b], [])
    assert len(out) == 2
    assert sorted(round(float(d["xy"][0])) for d in out) == [30, 70]
