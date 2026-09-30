"""Tests for the structure zoo and the completeness score (zoo.py)."""
import warnings

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

from vesselmap.image import prepare
from vesselmap.zoo import ROWS, completeness, zoo_sheet


def test_zoo_sheet_lays_out_every_case():
    I, V, tiles = zoo_sheet(0, rows=ROWS[:2])
    assert len(tiles) == 10 and all(len(t["vessels"]) == 3 for t in tiles)
    for t in tiles:                                  # every vessel inside its own tile
        x0, y0, x1, y1 = t["box"]
        for i in t["vessels"]:
            xy = V[i]["xy"]
            assert xy[:, 0].min() >= x0 and xy[:, 0].max() <= x1
            assert xy[:, 1].min() >= y0 and xy[:, 1].max() <= y1
    assert np.isfinite(I).all() and 0.2 < float(I.mean()) < 1.0


def test_completeness_flags_a_parallel_pair_annotated_as_one_vessel():
    """Two parallel vessels annotated each on its own are complete; the
    same pair annotated as one thick vessel down the middle is not, once a
    bright gap separates them (and cannot be told when they touch)."""
    row = [r for r in ROWS if r[0].startswith("parallel")]
    I, V, tiles = zoo_sheet(0, rows=row)
    P = prepare(I)
    merged = {}
    for t in tiles:
        a, b = (V[i] for i in t["vessels"])
        for v in (a, b):
            assert completeness(P.logI, P.sigma, v["xy"], v["r"])["score"] == 1.0
        d, j = cKDTree(b["xy"]).query(a["xy"])
        keep = d < 12
        mid = 0.5 * (a["xy"][keep] + b["xy"][j[keep]])
        rm = d[keep] / 2 + np.maximum(a["r"][keep], b["r"][j[keep]])
        merged[t["value"]] = completeness(P.logI, P.sigma, mid, rm)["score"]
    assert merged[2.4] == 1.0                        # lumens touch: no gap to see
    assert merged[4.2] < 0.3 and merged[5.5] < 0.3 and merged[7.5] < 0.3


def test_crossing_rows_mark_every_crossing():
    """Twisted pairs, weaves and ladders cross as often as their parameter
    says, and every crossing is marked as an ambiguous spot; the calibre
    rows span thin to thick vessels."""
    from vesselmap.zoo import ROWS_CALIBRE, ROWS_CROSSINGS
    I, V, tiles = zoo_sheet(0, rows=ROWS_CROSSINGS[:4])
    for t in tiles:
        assert len(t["ambiguous"]) == int(t["value"]), (t["name"], t["value"], len(t["ambiguous"]))
    I, V, tiles = zoo_sheet(0, rows=ROWS_CALIBRE)
    radii = np.concatenate([v["r"] for v in V])
    assert radii.min() < 0.8 and radii.max() > 5.0


def test_complex_nodes_mark_every_junction():
    """Two forks 4 px apart are two marks; a vessel crossing both branches of
    a fork is a fork and two crossings; three vessels through one point, a
    fork with a crossing through it and a trifurcation are one each."""
    from vesselmap.zoo import complex_node
    rng = np.random.default_rng(0)
    counts = [len(complex_node(128, case, rng)[1]) for case in range(5)]
    assert counts == [2, 1, 1, 1, 3], counts
