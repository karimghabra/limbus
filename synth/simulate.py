"""One synthetic burst from a Scene: network, flow, cells and camera together, and its exact ground truth.

    from synth import Scene, simulate, write_burst
    result = simulate(Scene(seed=3), n_frames=120)
    write_burst(result, "burst_synthetic_seed3")      # the recorder's format, plus truth/

write_burst writes the frames exactly as the LIMBUS recorder does (frame_NNNNNN.tif as 16-bit right-aligned 12-bit
DN, manifest.json, frames.csv), so notebooks/foundation.py and the flow analysis run on a synthetic burst unchanged,
and a truth/ folder beside them:

| file                 | content                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| truth/scene.json     | every parameter of the scene (anatomy, physiology, optics, size, seed)                    |
| truth/graph.json     | the directed vessel graph: nodes with 3D position (µm) and pressure (mmHg); edges with     |
|                      | 3D centreline with the local diameter at each point, nominal diameter, length, kind,       |
|                      | layer, vessel id, the arteriole a companion venule accompanies, mean flow (pl/s), blood and |
|                      | red-cell velocity (mm/s), discharge and tube haematocrit, wall shear rate, and their        |
|                      | direction (edges are stored oriented along the mean flow)                                 |
| truth/frames.npz     | per frame: the tissue motion (dx, dy px, rotation rad), the focal depth (µm), the time (s, |
|                      | the start of the exposure; cells and masks are half an exposure later)                     |
|                      | and every edge's flow and red-cell velocity at that instant (the cardiac pulse), signed    |
|                      | like the graph: positive from the edge's source to its target, so along the mean flow, and  |
|                      | negative only while the pulse reverses it                                                   |
| truth/cells.npz      | per frame, at the middle of its exposure: the 3D position of every red cell the camera     |
|                      | draws in or near the field (µm = origin_um + xyz_decimicron / 10, unsigned integers of      |
|                      | 0.1 µm), its id (stable while it stays in the domain), its segment (edge index); frame k's  |
|                      | cells are rows frame_offsets[k] to frame_offsets[k + 1]. Cells passing through the stretch  |
|                      | of a narrower vessel that lies inside a wider one at a junction are not drawn (so that the  |
|                      | junction is not darkened twice) and are left out until they come out of it                  |
| truth/masks.npz      | per frame, at the middle of its exposure (unsigned integer/float16, compressed): the id of  |
|                      | the shallowest segment at every                                                             |
|                      | pixel (0 is none, else edge index + 1), its depth (µm), and its projected red-cell velocity |
|                      | (px per frame, x and y); and, with every vessel at its frame-0 position, the id and depth   |
|                      | maps and the projected velocity of the mean (cycle-averaged) flow. Each segment is drawn at |
|                      | its local diameter, but never narrower than about 1.5 px, so that no vessel has holes: the  |
|                      | maps say where vessels are, not how wide the thin ones are (take widths from graph.json)    |

The integers are uint16, or uint32 where a scene is too large for them (more than 65 535 segments, or cells spread
over more than 6.5 mm).

Coordinates: graph and cells in tissue µm (x along the limbus, y away from it, z depth below the surface), x and y
being arc lengths along the curved surface of the globe. The camera looks along the globe's normal at the centre
(cx, cy) of field_of_view_um, so a tissue point (x, y, z) at arc distance r = hypot(x - cx, y - cy) from it appears
in frame 0 at pixel

    (W / 2 + (x - cx) k / um_per_px, H / 2 + (y - cy) k / um_per_px),   k = (R - z) sin(r / R) / r

with R = optics.eye_radius_um and W, H the frame's size (integers at the pixels' centres); k is 1 at the centre of
a flat field and falls with the distance from the centre and with depth (0.97 for an episcleral vessel 1.25 mm
out). In frame t that pixel is turned about the image centre (W / 2, H / 2) by rotation_rad and moved by (dx_px,
dy_px), as recorded in frames.npz. optics.camera_tilt_deg changes the focus across the field, not the positions.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import camera, cells, haemodynamics, network
from .anatomy import Scene


@dataclass
class Result:
    scene: Scene
    network: network.Network
    flow: haemodynamics.Flow
    motion: camera.Motion
    frames: np.ndarray                       # (T, H, W) uint16 DN
    times: np.ndarray                        # (T,) s, the start of each exposure
    cell_frames: list = field(default_factory=list)     # per frame: dict(xyz float32 (n, 3), id, edge)
    masks: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)


class TubeSampler:
    """Vessels as tubes carrying their red-cell density: sample points across every segment's cross-section,
    weighted so that each sample carries its share of the expected cells (H_T A ds / V_rbc), spread across the lumen
    by the chord length a ray crosses (the projection of a cylinder). The density is modulated along each vessel by
    the axial structure of the red cells (aggregates and plasma gaps: a log-normal field with the clump length and
    amplitude of synth.cells), carried downstream at the vessel's red-cell speed through the cardiac cycle, so that
    blood is seen to flow at magnifications too low to draw single cells (weights_at)."""

    PERIOD_UM = 4096.0

    def __init__(self, net, flow, scene):
        step = max(scene.anatomy.um_per_px / 2.0, 1.0)
        points, weights, edge, arc = [], [], [], []
        for i, e in enumerate(net.edges):
            local = np.column_stack([e.points, e.local_diameter])
            sampled = network.resample(local, step)
            pts, diam = sampled[:, :3], sampled[:, 3]
            if len(pts) < 2:
                continue
            tangent = np.gradient(pts[:, :2], axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.r_[0.0, np.cumsum(seg)]
            span = 0.5 * (np.r_[seg, 0.0] + np.r_[0.0, seg])                # the length of vessel each sample stands for
            radius = diam / 2
            k = max(1, int(np.ceil(float(diam.max()) / step)))
            offsets = (np.arange(k) + 0.5) / k * 2 - 1                      # across the diameter, -1..1
            chord = np.sqrt(np.clip(1 - offsets ** 2, 0, None))
            chord /= chord.sum()
            cells_per_sample = flow.ht[i] * np.pi * radius ** 2 * span / scene.physiology.rbc_volume_um3
            for o, w in zip(offsets, chord):
                p = pts.copy()
                p[:, :2] += (o * radius)[:, None] * normal
                points.append(p)
                weights.append(cells_per_sample * w)
                edge.append(np.full(len(p), i))
                arc.append(s)
        self.points = np.vstack(points) if points else np.zeros((0, 3))
        self.weights = np.concatenate(weights) if weights else np.zeros(0)
        self.edge = np.concatenate(edge) if edge else np.zeros(0, int)
        self.arc = np.concatenate(arc) if arc else np.zeros(0)
        # the axial red-cell structure: one periodic log-normal field shared by all vessels, entered at a random
        # place per vessel
        p = scene.physiology
        rng = np.random.default_rng([scene.seed, 7])
        n = int(self.PERIOD_UM)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freq = np.fft.rfftfreq(n)
        g = np.fft.irfft(spectrum * np.exp(-0.5 * (2 * np.pi * freq * p.clump_length_um) ** 2), n)
        g /= max(g.std(), 1e-12)
        # stronger than the per-cell value: here it stands for the whole texture of the cells, which single cells
        # (not drawn at this magnification) would add
        sigma = min(2.0 * p.clump_amplitude, 0.8)
        self.field = np.exp(sigma * g - 0.5 * sigma ** 2).astype(np.float32)
        self.entry = rng.uniform(0, self.PERIOD_UM, max(len(net.edges), 1))
        self.flow = flow

    def weights_at(self, t):
        """The samples' weights at time t (s): the expected cells, modulated by the axial structure carried along
        each vessel by its red-cell speed (pulsatile when the flow has harmonics)."""
        f = self.flow
        v = f.rbc_velocity * 1e3                                          # µm/s, signed along u -> v
        travel = v * t
        if len(f.pulse):
            w = 2 * np.pi * f.physiology.heart_rate_hz
            k = np.arange(1, len(f.pulse) + 1)[:, None]
            ratio = np.where(np.abs(f.flow) > 1e-12, f.pulse / np.where(np.abs(f.flow) > 1e-12, f.flow, 1.0), 0.0)
            phase = np.exp(1j * k * w * t)
            travel = v * (t + np.real(ratio * phase / (1j * k * w)).sum(0))
        position = (self.arc - travel[self.edge] + self.entry[self.edge]) % self.PERIOD_UM
        return self.weights * self.field[position.astype(np.int64) % len(self.field)]


def simulate(scene: Scene, n_frames=120, warm_up_s=0.6, record_cells=True, record_masks=True, progress=None) -> Result:
    """Generate the network, solve the flow, run the cells and render n_frames. The burst's eye motion is drawn
    first and the scene's margin widened to hold it (Result.scene has the margin used), so no frame shows the edge
    of the network."""
    motion = camera.burst_motion(n_frames, scene)
    scene = camera.cover_motion(scene, motion)
    timings = {}
    start = time.perf_counter()
    net = network.generate(scene)
    timings["network"] = time.perf_counter() - start
    start = time.perf_counter()
    flow = haemodynamics.solve(net, scene.physiology)
    timings["flow"] = time.perf_counter() - start
    return render_network(scene, net, flow, motion, n_frames, warm_up_s, record_cells, record_masks, progress, timings)


def render_network(scene: Scene, net, flow, motion, n_frames=120, warm_up_s=0.6, record_cells=True, record_masks=True,
                   progress=None, timings=None, tubes=None, occluder=None, time_offset=0.0) -> Result:
    """Run the red cells of a given network and flow and render n_frames of it.

    motion: the camera's motion over the burst (camera.Motion); the network must reach as far as it shows
    (camera.cover_motion).

    tubes: draw every vessel as a tube carrying its mean red-cell density instead of discrete cells (no flicker,
    no cell transport, far cheaper); by default when the magnification is low (Optics.tube_um_per_px or coarser).
    occluder: callable(frame index, renderer) -> (mask, irradiance) that covers part of each frame (the lids).
    time_offset: the time (s) of the first frame, for the phase of the cardiac cycle."""
    timings = dict(timings or {})
    if tubes is None:
        tubes = scene.anatomy.um_per_px >= scene.optics.tube_um_per_px
    rng = np.random.default_rng([scene.seed, 1])
    start = time.perf_counter()
    dt = 1.0 / scene.optics.fps / 3.0
    if tubes:
        red_cells = None
        record_cells = False
        tube = TubeSampler(net, flow, scene)
    else:
        red_cells = cells.Cells(net, flow, scene.physiology, rng)
        red_cells.t = time_offset - warm_up_s
        red_cells.warm_up(warm_up_s, dt)
        red_cells.t = time_offset
    timings["cells warm-up"] = time.perf_counter() - start
    renderer = camera.Renderer(scene, net, rng, motion)
    scale = camera.background_scale(scene)
    period = 1.0 / scene.optics.fps
    frames = np.empty((n_frames, scene.height_px, scene.width_px), np.uint16)
    times = time_offset + np.arange(n_frames) * period
    cell_frames = []
    masks = {"edge": [], "depth": [], "velocity": []}
    x0, x1, y0, y1 = net.field_of_view
    # cells are recorded where the camera can see them: the field plus what the frames show beyond it (the motion,
    # and the globe's curvature down to the deepest plexus)
    pad = camera.shown_beyond_field_um(scene, motion) + 8.0 * scene.anatomy.um_per_px
    edge_dtype = _unsigned(len(net.edges))

    def cells_at(t):
        t = t + time_offset
        if tubes:
            return tube.points, tube.weights_at(t)
        while red_cells.t < t - 1e-9:
            red_cells.step(min(dt, t - red_cells.t))
        xyz, seen = red_cells.positions(), red_cells.visible()
        if record_cells:
            keep = seen & (xyz[:, 0] > x0 - pad) & (xyz[:, 0] < x1 + pad) & (xyz[:, 1] > y0 - pad) & (xyz[:, 1] < y1 + pad)
            exposed.append(dict(xyz=xyz[keep].astype(np.float32), id=red_cells.ids[keep].astype(np.uint32),
                                edge=red_cells.edge[keep].astype(edge_dtype)))
        return xyz[seen]

    start = time.perf_counter()
    for k in range(n_frames):
        exposed = []                                   # the cells at each instant the exposure is sampled at
        irradiance, glint = renderer.expose(k, cells_at)
        if occluder is not None:
            mask, cover = occluder(k, renderer)
            irradiance = irradiance * (1 - mask) + cover * mask * renderer.light
            glint = glint * (1 - mask)
        frames[k] = renderer.sensor(irradiance, glint, scale)
        mid = k + 0.5 * scene.optics.exposure_s / period
        if record_cells:
            cell_frames.append(exposed[len(exposed) // 2])           # the middle one: mid-exposure, as the masks
        if record_masks:
            edge_map, depth_map, velocity_map = truth_maps(net, flow, renderer, mid, time_offset + mid * period)
            masks["edge"].append(edge_map)
            masks["depth"].append(depth_map)
            masks["velocity"].append(velocity_map)
        if progress:
            progress(k + 1, n_frames)
    timings["render"] = time.perf_counter() - start
    if record_masks:
        masks = {key: np.stack(value) for key, value in masks.items()}
        renderer.motion = camera.Motion(np.zeros(1), np.zeros(1), np.zeros(1), motion.focus[:1], np.zeros(0, int))
        edge0, depth0, velocity0 = truth_maps(net, flow, renderer, 0.0, None)
        masks["edge_frame0"], masks["depth_frame0"], masks["mean_velocity_frame0"] = edge0, depth0, velocity0
        renderer.motion = motion
    else:
        masks = {}
    return Result(scene, net, flow, motion, frames, times, cell_frames, masks, timings)


def truth_maps(net, flow, renderer, frame_time, t_seconds=None):
    """Per-pixel truth for one frame: the shallowest segment's id (edge index + 1), its depth (µm) and its
    projected red-cell velocity (px/frame, x and y), drawn with each segment's local diameter, deepest first. No
    segment is drawn narrower than about 1.5 px (3 px of the 2x grid the maps are drawn on), so that thin vessels
    keep an unbroken trace when the grid is reduced; below that calibre the maps overstate the width.
    t_seconds sets the instant of the cardiac cycle; None gives the mean flow."""
    scene = renderer.scene
    h, w = renderer.shape
    ss = 2
    edge_map = np.zeros((h * ss, w * ss), np.int32)
    depth_map = np.zeros((h * ss, w * ss), np.float32)
    vx = np.zeros((h * ss, w * ss), np.float32)
    vy = np.zeros((h * ss, w * ss), np.float32)
    if t_seconds is None:
        v_now = flow.rbc_velocity
    else:
        v_now = flow.rbc_velocity * (flow.flow_at(t_seconds) / np.where(np.abs(flow.flow) > 1e-12, flow.flow, 1.0))
    period = 1.0 / scene.optics.fps
    um = scene.anatomy.um_per_px
    order = np.argsort([-float(np.mean(e.points[:, 2])) for e in net.edges])      # deepest first
    for index in order:
        e = net.edges[index]
        n = len(e.points)
        step = max(1, int(12 / max(np.median(np.linalg.norm(np.diff(e.points, axis=0), axis=1)), 1e-6)))
        pick = np.unique(np.r_[0:n:step, n - 1])                          # every step-th point, and the last one
        pts = e.points[pick]
        q = renderer.project(pts, frame_time) / renderer.ss * ss        # px at the mask's own supersampling
        if (q[:, 0].max() < -10 or q[:, 0].min() > w * ss + 10 or q[:, 1].max() < -10 or q[:, 1].min() > h * ss + 10):
            continue
        local = e.local_diameter[pick]
        tangent = np.diff(q, axis=0)
        seg_len = np.linalg.norm(tangent, axis=1, keepdims=True)
        direction = tangent / np.maximum(seg_len, 1e-9)
        # projected speed: the 3D speed times the share of each piece's length that shows in the image
        length3 = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        share = (seg_len[:, 0] / ss * um) / np.maximum(length3, 1e-9)
        speed = v_now[index] * 1e3 * period / um * share                     # px per frame, signed along u -> v
        for j in range(len(q) - 1):
            a = tuple(int(round(c)) for c in q[j])
            b = tuple(int(round(c)) for c in q[j + 1])
            thickness = max(ss + 1, int(round(0.5 * (local[j] + local[j + 1]) / um * ss)))    # survives the reduction
            cv2.line(edge_map, a, b, int(index + 1), thickness)
            cv2.line(depth_map, a, b, float(np.mean(pts[j:j + 2, 2])), thickness)
            cv2.line(vx, a, b, float(speed[j] * direction[j, 0]), thickness)
            cv2.line(vy, a, b, float(speed[j] * direction[j, 1]), thickness)
    pick = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    return (edge_map[pick].astype(_unsigned(len(net.edges))), depth_map[pick].astype(np.float16),
            np.stack([vx[pick], vy[pick]]).astype(np.float16))


def _unsigned(largest):
    """The unsigned integer type that holds values up to largest: uint16 where it does, else wider."""
    return np.promote_types(np.uint16, np.min_scalar_type(int(largest)))


def graph_dict(net: network.Network, flow: haemodynamics.Flow):
    """The directed graph, every edge oriented along its mean flow."""
    nodes = [dict(id=i, xyz_um=[round(float(c), 3) for c in net.nodes[i]], pressure_mmhg=round(float(flow.pressure[i]), 4),
                  boundary=net.boundary.get(i)) for i in range(len(net.nodes))]
    edges = []
    for i, e in enumerate(net.edges):
        forward = flow.flow[i] >= 0
        points = e.points if forward else e.points[::-1]
        sign = 1.0 if forward else -1.0
        edges.append(dict(
            id=i, source=int(e.u if forward else e.v), target=int(e.v if forward else e.u),
            kind=e.kind, layer=e.layer, vessel=int(e.vessel), diameter_um=round(e.diameter, 4),
            length_um=round(float(flow.length[i]), 3), depth_mean_um=round(float(points[:, 2].mean()), 2),
            flow_pl_s=round(float(abs(flow.flow[i])), 6), velocity_mm_s=round(float(sign * flow.velocity[i]), 6),
            rbc_velocity_mm_s=round(float(sign * flow.rbc_velocity[i]), 6),
            discharge_haematocrit=round(float(flow.hd[i]), 5), tube_haematocrit=round(float(flow.ht[i]), 5),
            relative_viscosity=round(float(flow.viscosity[i]), 5), wall_shear_rate_s=round(float(flow.shear_rate[i]), 3),
            reversed_from_generation=not forward, paired_with_vessel=int(e.paired_with),
            centreline_um=np.round(points, 2).tolist(),
            diameter_profile_um=np.round(e.local_diameter if forward else e.local_diameter[::-1], 2).tolist()))
    return dict(nodes=nodes, edges=edges, domain_um=list(net.domain), field_of_view_um=list(net.field_of_view),
                vessel_kinds={int(k): v for k, v in net.vessel_names.items()},
                conventions="x along the limbus, y away from it, z depth below the epithelial surface, all µm; "
                            "edges oriented along the mean flow, velocities positive from source to target")


def write_burst(result: Result, folder):
    """Write the frames in the recorder's format and the ground truth under truth/. Returns the folder."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("frame_*.tif"):            # frames of an earlier, longer burst written here
        stale.unlink()
    scene = result.scene
    o = scene.optics
    period = 1.0 / o.fps
    for k, frame in enumerate(result.frames):
        tifffile.imwrite(folder / f"frame_{k:06d}.tif", frame, description=json.dumps({"index": k, "synthetic": True}))
    with (folder / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera_timestamp_ns", "exposure_us", "gain", "camera_counter", "index", "host_time_utc",
                         "host_monotonic_s", "filename"])
        for k in range(len(result.frames)):
            writer.writerow([int(round(k * period * 1e9)), o.exposure_s * 1e6, 20.4, k, k, "", f"{k * period:.6f}",
                             f"frame_{k:06d}.tif"])
    height, width = result.frames.shape[1:]
    manifest = {
        "camera": {"model": "synthetic (synth package)", "serial": f"seed-{scene.seed}", "vendor": "synthetic"},
        "acquisition_requested": {"pixel_format": "Mono12", "bit_depth": 12, "width": width, "height": height,
                                  "requested_fps": o.fps},
        "camera_state_at_start": {"exposure_us": o.exposure_s * 1e6, "gain": 20.4, "gamma": o.gamma,
                                  "resulting_fps": o.fps},
        "capture": {"frames_offered": len(result.frames), "frames_written": len(result.frames), "frames_dropped": 0,
                    "span_s": len(result.frames) * period, "effective_fps": o.fps},
        "pixel_values": {"dtype": "uint16", "bit_depth": 12, "alignment": "right", "max_value": 4095,
                         "note": "synthetic sensor DN"},
        "synthetic": {"um_per_px": scene.anatomy.um_per_px, "truth": "truth/"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    truth = folder / "truth"
    truth.mkdir(exist_ok=True)
    for name, present in (("cells.npz", result.cell_frames), ("masks.npz", result.masks)):
        if not present:
            (truth / name).unlink(missing_ok=True)
    (truth / "scene.json").write_text(json.dumps(_jsonable(asdict(scene)), indent=1), encoding="utf-8")
    (truth / "graph.json").write_text(json.dumps(graph_dict(result.network, result.flow)), encoding="utf-8")
    along = np.where(result.flow.flow >= 0, 1.0, -1.0)       # the graph's edges run along the mean flow
    flows = np.stack([result.flow.flow_at(t) for t in result.times])
    scale = flows / np.where(np.abs(result.flow.flow) > 1e-12, result.flow.flow, 1.0)
    m = result.motion
    np.savez_compressed(truth / "frames.npz", time_s=result.times, dx_px=m.dx, dy_px=m.dy, rotation_rad=m.rotation,
                        focus_um=m.focus, saccade_frames=m.saccades, edge_flow_pl_s=(along * flows).astype(np.float32),
                        edge_rbc_velocity_mm_s=(along * result.flow.rbc_velocity * scale).astype(np.float32))
    if result.cell_frames:
        counts = np.array([len(c["id"]) for c in result.cell_frames])
        xyz = np.concatenate([c["xyz"] for c in result.cell_frames])
        origin = np.floor(xyz.min(axis=0)) - 1.0 if len(xyz) else np.zeros(3)
        decimicron = np.round((xyz - origin) * 10.0)
        np.savez_compressed(truth / "cells.npz", frame_offsets=np.r_[0, np.cumsum(counts)], origin_um=origin,
                            xyz_decimicron=decimicron.astype(_unsigned(decimicron.max(initial=0.0))),
                            id=np.concatenate([c["id"] for c in result.cell_frames]),
                            edge=np.concatenate([c["edge"] for c in result.cell_frames]))
    if result.masks:
        np.savez_compressed(truth / "masks.npz", **result.masks)
    return folder


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
