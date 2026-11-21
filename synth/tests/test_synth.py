"""Tests of the synthetic conjunctiva: the physical laws, conservation, the network's anatomy, rendering and export.

usage: python -m pytest synth/tests            (from the repository root)
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synth import Scene, simulate, write_burst  # noqa: E402
from synth import cells as C  # noqa: E402
from synth import check  # noqa: E402
from synth import haemodynamics as H  # noqa: E402
from synth import network as N  # noqa: E402
from synth import targets as T  # noqa: E402


def small_scene(seed=0):
    return Scene(width_px=256, height_px=192, margin_um=250.0, seed=seed)


@pytest.fixture(scope="module")
def solved():
    scene = small_scene(1)
    net = N.generate(scene)
    flow = H.solve(net, scene.physiology)
    return scene, net, flow


# ------------------------------------------------------------------------------------------- rheology laws

def test_viscosity_laws():
    # plasma alone has relative viscosity 1; whole blood is more viscous in large tubes than in ~7 µm tubes
    assert np.isclose(H.viscosity_in_vitro(50.0, 0.0), 1.0)
    assert H.viscosity_in_vitro(300.0, 0.45) > H.viscosity_in_vitro(7.0, 0.45)       # Fahraeus-Lindqvist
    # in vitro at H 0.45 tends to 3.2 in wide tubes [Pries1992]
    assert abs(H.viscosity_in_vitro(5000.0, 0.45) - 3.2) < 0.05
    # in vivo exceeds in vitro in small vessels (the endothelial surface layer) [Pries1994]
    assert H.viscosity_in_vivo(8.0, 0.45) > H.viscosity_in_vitro(8.0, 0.45)
    assert np.all(np.diff(H.viscosity_in_vivo(20.0, np.linspace(0, 0.6, 7))) > 0)


def test_fahraeus_and_phase_separation():
    d = np.array([5.0, 10.0, 30.0, 100.0])
    ht = H.tube_haematocrit(d, 0.45)
    assert np.all(ht < 0.45) and np.all(ht > 0.2)
    # symmetric bifurcation, equal flows: equal cell shares
    assert np.isclose(H.phase_separation(0.5, 20.0, 10.0, 10.0, 0.45), 0.5)
    # the daughter with more flow takes a disproportionately larger share of the cells
    assert H.phase_separation(0.8, 20.0, 10.0, 10.0, 0.45) > 0.8
    # plasma skimming: below X0 a small branch gets no cells
    assert H.phase_separation(0.01, 20.0, 5.0, 19.0, 0.45) == 0.0


# ------------------------------------------------------------------------------------------- conservation

def test_conservation_at_every_junction(solved):
    _, net, flow = solved
    flow_error, cell_error = flow.conservation(net)
    assert flow_error < 1e-8
    assert cell_error < 1e-8


def test_pulse_keeps_conservation(solved):
    _, net, flow = solved
    q = flow.flow_at(0.3)
    n = len(net.nodes)
    balance = np.zeros(n)
    scale = np.zeros(n)
    for i, e in enumerate(net.edges):
        balance[e.u] -= q[i]
        balance[e.v] += q[i]
        scale[e.u] += abs(q[i])
        scale[e.v] += abs(q[i])
    interior = np.array([k not in net.boundary for k in range(n)])
    assert np.max(np.abs(balance[interior]) / np.maximum(scale[interior], 1e-6 * scale.max())) < 1e-8


def test_flow_runs_downhill(solved):
    _, net, flow = solved
    for i, e in enumerate(net.edges):
        drop = flow.pressure[e.u] - flow.pressure[e.v]
        assert flow.flow[i] * drop >= -1e-9


def test_cells_follow_haematocrit(solved):
    scene, net, flow = solved
    cells = C.Cells(net, flow, scene.physiology, np.random.default_rng(0))
    for _ in range(20):
        cells.step(1.0 / scene.optics.fps / 3)
    counts = np.bincount(cells.edge, minlength=len(net.edges))
    busy = cells.expected > 30
    ratio = counts[busy].sum() / cells.expected[busy].sum()
    assert 0.8 < ratio < 1.25
    # every cell stays inside its vessel's lumen
    xyz = cells.positions()
    g = cells.geo
    index = g.start[cells.edge] + np.clip(np.round(cells.s).astype(int), 0, g.count[cells.edge] - 1)
    radial = np.linalg.norm(xyz - g.centre[index], axis=1)
    assert np.all(radial <= cells.radius[cells.edge] * g.width[index] + 1e-6)


# ------------------------------------------------------------------------------------------- anatomy

def test_network_is_3d_and_layered(solved):
    scene, net, _ = solved
    a = scene.anatomy
    kinds = {e.kind for e in net.edges}
    for needed in ("arteriole", "venule", "capillary", "episcleral artery", "episcleral vein"):
        assert needed in kinds
    for e in net.edges:
        z = e.points[:, 2]
        if e.layer == "conjunctival":
            assert a.epithelium <= z.min() and z.max() <= a.conjunctival_plexus[1] + 40
        if e.layer == "episcleral":
            assert a.episcleral_plexus[0] - 40 <= z.min() and z.max() <= a.episcleral_plexus[1] + 40
    # communicating vessels run between layers
    rising = [e for e in net.edges if e.layer == "communicating"]
    for e in rising:
        assert np.ptp(e.points[:, 2]) > 50


def test_anatomy_check_passes():
    report = check.anatomy_report(*_solve(Scene(seed=2)))   # the default field: medians are unstable in tiny fields
    failing = [row for row in report if row["status"] == "outside"]
    assert not failing, failing


def _solve(scene):
    net = N.generate(scene)
    return scene, net, H.solve(net, scene.physiology)


def test_generation_is_deterministic():
    a = N.generate(small_scene(5))
    b = N.generate(small_scene(5))
    assert len(a.edges) == len(b.edges)
    assert np.allclose(a.nodes, b.nodes)


# ------------------------------------------------------------------------------------------- rendering and export

@pytest.fixture(scope="module")
def burst(tmp_path_factory):
    scene = replace(small_scene(3))
    result = simulate(scene, n_frames=6, warm_up_s=0.1)
    folder = write_burst(result, tmp_path_factory.mktemp("synthetic") / "burst_synthetic")
    return result, folder


def test_frames_look_like_the_recorder(burst):
    result, folder = burst
    frames = result.frames
    assert frames.dtype == np.uint16 and frames.max() <= result.scene.optics.ceiling_dn
    loaded = T.load_burst(folder)
    assert loaded.frames.shape == frames.shape
    assert np.isclose(loaded.fps, result.scene.optics.fps, rtol=1e-3)
    # vessels absorb: the darkest percentile is well below the background
    assert np.percentile(frames, 1) < 0.85 * np.median(frames)


def test_truth_export(burst):
    result, folder = burst
    graph = json.loads((folder / "truth" / "graph.json").read_text())
    assert len(graph["edges"]) == len(result.network.edges)
    for edge in graph["edges"][:50]:
        assert edge["flow_pl_s"] >= 0 and len(edge["centreline_um"][0]) == 3
    cells = np.load(folder / "truth" / "cells.npz")
    assert len(cells["frame_offsets"]) == len(result.frames) + 1
    xyz = cells["origin_um"] + cells["xyz_decimicron"][: cells["frame_offsets"][1]] / 10.0
    assert np.allclose(xyz, result.cell_frames[0]["xyz"], atol=0.06)
    masks = np.load(folder / "truth" / "masks.npz")
    assert masks["edge"].shape == result.frames.shape
    assert masks["velocity"].shape == (len(result.frames), 2) + result.frames.shape[1:]
    frames = np.load(folder / "truth" / "frames.npz")
    assert frames["edge_rbc_velocity_mm_s"].shape == (len(result.frames), len(result.network.edges))


def test_random_scenes_stay_anatomical():
    from synth.randomize import random_scene
    for seed in (11, 12):
        scene = random_scene(seed, anatomical=True)   # redrawn until the check passes
        report = check.anatomy_report(*_solve(scene))
        failing = [row for row in report if row["status"] == "outside"]
        assert not failing, (seed, failing)
    assert random_scene(3).optics.psf_sigma_px == random_scene(3).optics.psf_sigma_px
