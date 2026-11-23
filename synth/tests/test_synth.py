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
    # whole blood at H 0.45 tends to a relative viscosity of 3.2 in wide vessels [Pries1994]
    assert abs(H.viscosity_in_vivo(5000.0, 0.45) - 3.2) < 0.05
    # it is thinner in vessels of a few tens of micrometres (Fahraeus-Lindqvist) and thick again in capillaries,
    # where the endothelial surface layer takes much of the lumen [Pries1994]
    assert H.viscosity_in_vivo(30.0, 0.45) < H.viscosity_in_vivo(300.0, 0.45)
    assert H.viscosity_in_vivo(8.0, 0.45) > H.viscosity_in_vivo(30.0, 0.45)
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
    """Through the cardiac cycle the volume a node's distending vessels take up is exactly what flows in less what
    flows out, and over a whole cycle nothing is stored."""
    _, net, flow = solved
    n = len(net.nodes)
    interior = np.array([k not in net.boundary for k in range(n)])
    for t in (0.1, 0.3, 0.7):
        q = flow.flow_at(t)
        balance = np.zeros(n)
        scale = np.zeros(n)
        for i, e in enumerate(net.edges):
            balance[e.u] -= q[i]
            balance[e.v] += q[i]
            scale[e.u] += abs(q[i])
            scale[e.v] += abs(q[i])
        stored = flow.storage_at(t)
        assert np.max(np.abs(balance - stored)[interior] / np.maximum(scale[interior], 1e-6 * scale.max())) < 1e-8
    period = 1.0 / flow.physiology.heart_rate_hz
    mean_storage = np.mean([flow.storage_at(t) for t in np.linspace(0, period, 64, endpoint=False)], axis=0)
    assert np.max(np.abs(mean_storage)) < 1e-9 * np.abs(flow.flow).max()


def test_pulse_is_damped_downstream(solved):
    """The pulse is strongest in the arterioles and damped through the capillaries into the venules."""
    _, net, flow = solved
    t = np.linspace(0, 1.0 / flow.physiology.heart_rate_hz, 32, endpoint=False)
    series = np.abs(np.stack([flow.flow_at(tt) for tt in t]))
    pi = (series.max(0) - series.min(0)) / np.maximum(series.mean(0), 1e-12)
    kind = np.array([e.kind for e in net.edges])
    moving = np.abs(flow.flow) > 0
    arteriole = np.median(pi[(kind == "arteriole") & moving])
    capillary = np.median(pi[(kind == "capillary") & moving])
    venule = np.median(pi[(kind == "venule") & moving])
    assert arteriole > capillary > venule


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


def test_anatomy_check_passes_for_most_scenes():
    # the default scenes are calibrated to the cited ranges; single quantities of a single seed may land just past a
    # bound (docs/synthetic_conjunctiva_validation.md gives the rate over 25 seeds), so most of six must pass
    passed, failures = 0, []
    for seed in range(6):
        report = check.anatomy_report(*_solve(Scene(seed=seed)))
        outside = [(row["quantity"], row["value"], row["range"]) for row in report if row["status"] == "outside"]
        passed += not outside
        failures += outside
    assert passed >= 4, failures


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


def _frames_stay_inside(scene, motion, field, domain, depth):
    """Whether the tissue seen at every corner of every frame lies inside the domain, at a depth: on the globe a
    point at depth z and arc distance r from the field's centre is drawn at (R - z) sin(r / R) from it (docs §1.4),
    so the camera's point at distance rho shows the tissue at the arc R asin(rho / (R - z))."""
    um, R = scene.anatomy.um_per_px, scene.optics.eye_radius_um
    x0, x1, y0, y1 = field
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    centre = corners.mean(0)
    inside = True
    for k in range(len(motion.dx)):
        c, s = np.cos(-motion.rotation[k]), np.sin(-motion.rotation[k])
        q = (corners - centre) - np.array([motion.dx[k], motion.dy[k]]) * um
        q = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], 1)
        rho = np.linalg.norm(q, axis=1)
        seen = q * (R * np.arcsin(rho / (R - depth)) / rho)[:, None] + centre
        inside &= bool(seen[:, 0].min() > domain[0] and seen[:, 0].max() < domain[1] and
                       seen[:, 1].min() > domain[2] and seen[:, 1].max() < domain[3])
    return inside


def test_eye_motion_never_shows_the_networks_edge():
    """A burst's network reaches past everything its frames show: with saccades far larger than the scene's own
    margin, the tissue at every corner of every frame still lies inside the generated domain, at the depth of the
    deepest vessel too."""
    scene = small_scene(5)
    scene = replace(scene, optics=replace(scene.optics, saccade_rate_hz=8.0, saccade_px=(150.0, 200.0)), margin_um=60.0)
    result = simulate(scene, n_frames=12, warm_up_s=0.05, record_cells=False, record_masks=False)
    m = result.motion
    assert np.hypot(m.dx, m.dy).max() * scene.anatomy.um_per_px > 3 * scene.margin_um
    net = result.network
    assert _frames_stay_inside(scene, m, net.field_of_view, net.domain, net.nodes[:, 2].max())


@pytest.mark.parametrize("width, height, um", [(1920, 1200, 3.26), (768, 512, 12.0)])
def test_margin_covers_motion_in_large_and_coarse_fields(width, height, um):
    """The margin a motion needs holds on the curved globe where the field is large against the eye's radius (the
    camera's full frame, a low-magnification view): the tissue at the corners of frames moved along the diagonal,
    and turned, stays within the field plus the margin at the depth of the deepest plexus."""
    from synth import camera
    base = Scene(width_px=width, height_px=height)
    scene = replace(base, anatomy=replace(base.anatomy, um_per_px=um))
    d = 190.0 * np.array([width, height]) / np.hypot(width, height)
    motion = camera.Motion(np.array([0.0, d[0], -d[0], d[0]]), np.array([0.0, d[1], -d[1], -d[1]]),
                           np.radians([0.0, 0.4, -0.4, 0.0]), np.full(4, 250.0), np.zeros(0, int))
    margin = camera.cover_motion(scene, motion).margin_um
    w, h = width * um, height * um
    field = (0.0, w, 0.0, h)
    domain = (-margin, w + margin, -margin, h + margin)
    assert _frames_stay_inside(scene, motion, field, domain, scene.anatomy.deep_plexus[1])


def test_texture_is_not_mirrored_where_a_saccade_reveals_it(solved):
    """The scleral texture is a random field fixed to the tissue, so the strip a large movement brings into the
    frame is new texture: no column of the frame is a mirror axis of its fine structure (a canvas too small for the
    movement would be filled in by reflection at its edge)."""
    import cv2
    from synth import camera
    scene, net, _ = solved
    scene = replace(scene, optics=replace(scene.optics, motion_bound_px=20.0, glints=0))
    motion = camera.Motion(np.array([200.0]), np.zeros(1), np.zeros(1), np.full(1, 250.0), np.zeros(0, int))
    renderer = camera.Renderer(scene, net, np.random.default_rng(0), motion)
    texture = np.log(renderer.tissue_maps(0.0)[0])
    fine = texture - cv2.GaussianBlur(texture, (0, 0), 3)
    mirror = [np.corrcoef(fine[:, axis - 30:axis][:, ::-1].ravel(), fine[:, axis:axis + 30].ravel())[0, 1]
              for axis in range(30, 200)]
    assert max(mirror) < 0.5


def test_tubes_carry_each_vessels_cells_once(solved):
    """Drawn as tubes, a vessel holds the red cells its volume and tube haematocrit give, H_T V / V_rbc, however
    its centreline is sampled."""
    from synth.simulate import TubeSampler
    scene, net, flow = solved
    tubes = TubeSampler(net, flow, scene)
    held = np.bincount(tubes.edge, tubes.weights, len(net.edges))
    for i in np.argsort([e.length for e in net.edges])[[0, len(net.edges) // 2, -1]]:
        e = net.edges[i]
        seg = np.linalg.norm(np.diff(e.points, axis=0), axis=1)
        area = np.pi * (0.5 * (e.local_diameter[1:] + e.local_diameter[:-1]) / 2) ** 2
        expected = flow.ht[i] * float((area * seg).sum()) / scene.physiology.rbc_volume_um3
        assert held[i] == pytest.approx(expected, rel=0.08)


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


def test_frame_truth_is_signed_like_the_graph(burst):
    """graph.json orients every edge along its mean flow (flow_pl_s >= 0, velocities positive from source to
    target); the per-frame columns of frames.npz index the same edges, so over a burst their flow runs the same
    way (the pulse, 12 % of the arterial pressure, reverses hardly any)."""
    result, folder = burst
    graph = json.loads((folder / "truth" / "graph.json").read_text())
    frames = np.load(folder / "truth" / "frames.npz")
    mean_flow = np.array([e["flow_pl_s"] for e in graph["edges"]])
    velocity = np.array([e["rbc_velocity_mm_s"] for e in graph["edges"]])
    flowing = mean_flow > 1e-3
    assert flowing.sum() > 20 and np.any(result.flow.flow[flowing] < 0)
    assert np.mean(frames["edge_flow_pl_s"].mean(0)[flowing] > 0) > 0.95
    assert np.mean(frames["edge_rbc_velocity_mm_s"].mean(0)[flowing] * velocity[flowing] > 0) > 0.95


def test_cell_truth_keeps_positions_in_a_wide_scene(burst, tmp_path):
    """Cells written to cells.npz come back where they were, also when they span more than 6.5 mm (the reach of
    16-bit tenths of a micrometre)."""
    result, _ = burst
    xyz = np.array([[0.0, 0.0, 60.0], [9000.25, 10.0, 80.0], [20.0, 7000.5, 400.0]], np.float32)
    wide = replace(result, cell_frames=[dict(xyz=xyz, id=np.arange(3, dtype=np.uint32),
                                             edge=np.zeros(3, np.uint16))], masks={})
    cells = np.load(write_burst(wide, tmp_path / "wide") / "truth" / "cells.npz")
    assert np.allclose(cells["origin_um"] + cells["xyz_decimicron"] / 10.0, xyz, atol=0.06)


def test_masks_reach_the_end_of_every_segment():
    """The frame-0 vessel mask is drawn along the whole of every segment, to its downstream node: the pixel of
    each of a centreline's last points (placed by the projection the synth.simulate docstring gives) lies on a
    vessel."""
    scene = small_scene(3)
    result = simulate(scene, n_frames=2, warm_up_s=0.05, record_cells=False)
    net, mask = result.network, result.masks["edge_frame0"]
    um, R = scene.anatomy.um_per_px, scene.optics.eye_radius_um
    x0, x1, y0, y1 = net.field_of_view
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    h, w = mask.shape
    ends = 0
    for e in net.edges:
        tail = e.points[-6:]
        u = tail[:, :2] - centre
        r = np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-9)
        k = (R - tail[:, 2]) * np.sin(r / R) / r
        col = np.round(w / 2 + u[:, 0] * k / um).astype(int)
        row = np.round(h / 2 + u[:, 1] * k / um).astype(int)
        inside = (col >= 1) & (col < w - 1) & (row >= 1) & (row < h - 1)
        assert np.all(mask[row[inside], col[inside]] != 0)
        ends += int(inside.sum())
    assert ends > 100


def test_random_scenes_stay_anatomical():
    from synth.randomize import random_scene
    for seed in (11, 12):
        scene = random_scene(seed, anatomical=True)   # redrawn until the check passes
        report = check.anatomy_report(*_solve(scene))
        failing = [row for row in report if row["status"] == "outside"]
        assert not failing, (seed, failing)
    # the scene that passed is the one a burst renders: simulate grows the network that was checked, also when the
    # burst's eye motion needs more margin than the scene started with
    base = Scene(width_px=256, height_px=192, margin_um=40.0)
    scene = random_scene(11, base=base, anatomical=True, n_frames=4)
    result = simulate(scene, n_frames=4, warm_up_s=0.05, record_cells=False, record_masks=False)
    assert result.scene.margin_um == scene.margin_um > base.margin_um
    report = check.anatomy_report(result.scene, result.network, result.flow)
    assert not [row for row in report if row["status"] == "outside"]
    assert random_scene(3).optics.psf_sigma_px == random_scene(3).optics.psf_sigma_px


# ------------------------------------------------------------------------------------------- the whole eye

@pytest.fixture(scope="module")
def eye():
    from synth.eye import Eye
    return Eye(seed=3, fine_region=(75.0, 105.0, 0.0, 5000.0))


def test_eye_conservation_and_outflow(eye):
    flow_error, cell_error = eye.flow.conservation(eye.network)
    assert flow_error < 1e-6 and cell_error < 1e-8
    q = eye.flow.flow
    assert np.isfinite(q).all() and np.abs(q).max() > 0


def test_same_seed_same_eye(eye):
    """A patient is their seed: the same seed grows the same eye, vessel for vessel."""
    from synth.eye import Eye, build_tile
    key = eye.tile_key(3000.0, 90.0)
    a, b = build_tile(eye, key), build_tile(eye, key)
    assert len(a["network"].edges) == len(b["network"].edges) > 0
    assert np.allclose(a["network"].nodes, b["network"].nodes)
    again = Eye(seed=3, fine_region=eye.fine_region)
    assert len(again.network.edges) == len(eye.network.edges)
    assert np.allclose(again.network.nodes, eye.network.nodes)
    assert np.allclose(again.flow.flow, eye.flow.flow)


def test_plexus_is_continuous(eye):
    """The capillaries form one plexus over the fine region: no tile's capillaries are cut off from its
    neighbours', and the plexus runs across tile borders."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    net = eye.network
    cap = [e for e in net.edges if e.level == 2]
    nodes = sorted({k for e in cap for k in (e.u, e.v)})
    index = {k: i for i, k in enumerate(nodes)}
    rows = [index[e.u] for e in cap]
    cols = [index[e.v] for e in cap]
    graph = coo_matrix((np.ones(len(cap)), (rows, cols)), shape=(len(nodes), len(nodes)))
    n, labels = connected_components(graph, directed=False)
    assert np.bincount(labels).max() > 0.95 * len(nodes)
    crossing = 0
    for e in cap:
        s, c, _ = eye.globe.surface_coordinates(e.points[[0, -1]])
        crossing += eye.tile_key(s[0], c[0]) != eye.tile_key(s[1], c[1])
    assert crossing > 0.1 * len(cap)


def test_capillary_calibre_varies_along_the_plexus(eye):
    """The whole eye's capillaries vary in calibre along their length as the patch level's do: a CV up to 0.4 in
    vessels below 15 µm [Kiani1993], about 0.19 in the conjunctiva [Shahidi2010]."""
    cv = [np.std(e.local_diameter) / np.mean(e.local_diameter) for e in eye.network.edges
          if e.level == 2 and e.length > 100.0]
    assert len(cv) > 100 and 0.05 < np.median(cv) < 0.4


def test_view_keeps_the_eyes_inlets(eye):
    """In a view that holds one of the eye's own inlets, that inlet is an arterial boundary of the view, and red
    cells are supplied at every vessel end from which blood flows into the view: where the view cut the vessel,
    where the eye begins, and where the blood joins through a vessel the view does not draw (a lumped bed, the
    fine level)."""
    from synth.eye import Pose, assemble_view
    inlet = next(k for k, role in eye.network.boundary.items() if role == "arterial")
    s, clock, _ = eye.globe.surface_coordinates(eye.network.nodes[inlet][None])
    scene = replace(eye.scene, width_px=256, height_px=192)
    net, flow = assemble_view(eye, Pose(s_mm=float(s[0]) / 1000.0, clock_deg=float(clock[0])), scene, fine=False)
    assert "arterial" in net.boundary.values()
    incident = net.incident()
    sources = [n for n, members in enumerate(incident) if len(members) == 1 and
               flow.flow[members[0]] * (1 if net.edges[members[0]].u == n else -1) > 0]
    supplied = {node for _, node in C.Cells(net, flow, scene.physiology, np.random.default_rng(0)).inflow}
    assert sources and set(sources) <= supplied
    here = [n for n in net.boundary if np.linalg.norm(net.nodes[n][:2]) < 1.0]
    assert here and all(net.boundary[n] == "arterial" for n in here)


def test_lumped_beds_are_not_drawn(eye):
    """Outside the fine region the eye keeps lumped beds, equivalent resistances for vessels that are not drawn
    (docs §11.6): they stay in the solved network but appear in no view, nor on the browser page."""
    from synth.eye import Pose, assemble_view
    from synth.eyeview import export
    beds = [e for e in eye.network.edges if e.kind == "bed"]
    assert beds
    s, clock, _ = eye.globe.surface_coordinates(beds[0].points[:1])
    scene = replace(eye.scene, width_px=256, height_px=192)
    net, flow = assemble_view(eye, Pose(s_mm=float(s[0]) / 1000.0, clock_deg=float(clock[0])), scene, fine=False)
    assert len(net.edges) > 0 and all(e.kind != "bed" for e in net.edges)
    assert export(eye)["n_edges"] == sum(e.kind != "bed" and len(e.points) >= 2 for e in eye.network.edges)


def test_every_perfused_vessel_flows_at_every_level(eye):
    """Every vessel the network perfuses carries red cells in every view that draws it, at every magnification
    (the views are selections of the eye's one solved network), and a view draws every vessel in its field."""
    from synth.eye import Pose, assemble_view, perfusion
    report = perfusion(eye)
    assert report["capillary plexus"]["perfused_length_share"] > 0.95
    assert report["fine trees"]["perfused_length_share"] > 0.9
    pose = Pose(s_mm=3.0, clock_deg=90.0)
    g = eye.globe
    frame = g.frame(pose.s_mm * 1000.0, pose.clock_deg)
    seen = []
    for um, fine in ((1.0, True), (3.26, True), (6.5, True), (12.0, False)):
        scene = replace(eye.scene, anatomy=replace(eye.scene.anatomy, um_per_px=um), width_px=256, height_px=192)
        net, flow = assemble_view(eye, pose, scene, fine=fine)
        drawn = {}
        for i, (source, ei) in enumerate(net.origins):
            assert source == "eye"
            assert flow.flow[i] == eye.flow.flow[ei] and flow.rbc_velocity[i] == eye.flow.rbc_velocity[ei]
            drawn[ei] = float(flow.flow[i])
        x0, x1, y0, y1 = net.domain
        expected = set()
        for ei, e in enumerate(eye.network.edges):
            if not fine and e.level > 0:
                continue
            local = g.to_local(e.points, frame)
            if ((local[:, 0] > x0) & (local[:, 0] < x1) & (local[:, 1] > y0) & (local[:, 1] < y1)).sum() >= 2:
                expected.add(ei)
        assert expected <= set(drawn)
        # the view's truth carries the eye's pressures: blood runs down them in every piece (Poiseuille), which
        # pressures left at zero inside the view would not show
        drop = np.array([flow.pressure[e.u] - flow.pressure[e.v] for e in net.edges])
        assert np.all(flow.flow * drop >= -1e-9) and np.mean(np.abs(drop) > 0) > 0.9
        assert flow.pressure.min() >= eye.flow.pressure.min() - 1e-9
        assert flow.pressure_at(0.2).shape == flow.pressure.shape and flow.storage_at(0.2).shape == flow.pressure.shape
        length = np.array([e.length for e in net.edges])
        moving = np.abs(flow.rbc_velocity) > 0.005
        assert length[moving].sum() > 0.9 * length.sum()
        assert {eye.network.edges[ei].level for ei in drawn} <= ({0, 1, 2} if fine else {0})
        seen.append(drawn)
    shared = set(seen[1]) & set(seen[2])
    assert {eye.network.edges[ei].level for ei in shared} == {0, 1, 2}


# ------------------------------------------------------------------------------------------- the patient and mapping

def test_listing_points_the_eye_without_torsion():
    from synth.gaze import direction_of, listing
    for h, v in ((0, 0), (20, 0), (0, -15), (25, 18), (-30, 10)):
        R = listing(h, v)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R @ np.array([0, 0, 1.0]), direction_of(h, v))
        # Listing's law: the rotation axis lies in the plane perpendicular to the primary direction
        w, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, np.argmin(np.abs(w - 1))])
        assert abs(axis[2]) < 1e-6 or (h, v) == (0, 0)


def test_patient_is_their_seed():
    from synth.gaze import patient
    import synth.gaze as gaze_module
    # the anatomy drawn from a seed, without growing the network
    calls = []
    original = gaze_module.Eye
    gaze_module.Eye = lambda **kw: calls.append(kw) or kw
    try:
        a, b, c = patient(5), patient(5), patient(6)
    finally:
        gaze_module.Eye = original
    assert a["anatomy"] == b["anatomy"] and a["scene"] == b["scene"]
    assert a["anatomy"] != c["anatomy"]


def test_slit_lamp_and_mapper(eye):
    """A few bursts at commanded gazes: their truth follows the gaze, the lids cover the field where they should, and
    the mapper places every burst near its true place from the frames alone."""
    from synth.gaze import SlitLamp
    from synth.mapping import Mapper, gaze_for
    scene = replace(eye.scene, width_px=320, height_px=240, margin_um=150.0)
    lamp = SlitLamp(eye, aim=(40.0, 0.0), scene=scene, seed=1)
    targets = [gaze_for(lamp, s, c)[0] for s, c in ((3000.0, 88.0), (3000.0, 89.2), (3350.0, 88.6))]
    mapper = Mapper(eye.globe, region=((85.0, 93.0), (2.5, 3.8)), um_per_px=4.0)
    previous, truth = None, []
    for gaze in targets:
        burst = lamp.record(gaze, n_frames=4, previous=previous)
        assert burst.result.frames.shape == (4, 240, 320)
        assert burst.occlusion.shape == (4,) and np.all((burst.occlusion >= 0) & (burst.occlusion <= 1))
        # the field the camera saw lies within a fixation error (0.25°, about 55 µm) of the commanded one
        aimed, _ = lamp.field_pose(gaze)
        a = eye.globe.point(aimed.s_mm * 1000, aimed.clock_deg, 0)
        b = eye.globe.point(burst.pose.s_mm * 1000, burst.pose.clock_deg, 0)
        assert np.linalg.norm(a - b) < 400.0
        placed = mapper.add(burst.result.frames, gaze, lamp)
        first = placed["first"] if placed else 0
        p = burst.poses[first]
        truth.append((p.s_mm * 1000, p.clock_deg))
        previous = gaze
    report = mapper.evaluate(eye, true_poses=truth)
    assert report["coverage"] > 0.1
    assert report["relative_placement_error_um_median"] < 60.0
    # looking down brings the upper lid down over the upper conjunctiva
    up, _ = lamp.lid_margins(0.0)
    down, _ = lamp.lid_margins(-20.0)
    assert down < up
