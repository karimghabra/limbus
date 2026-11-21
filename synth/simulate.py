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
| truth/frames.npz     | per frame: the tissue motion (dx, dy px, rotation rad), the focal depth (µm), the time (s) |
|                      | and every edge's flow and red-cell velocity at that instant (the cardiac pulse)             |
| truth/cells.npz      | per frame: the 3D position of every red cell the camera can see (µm = origin_um +          |
|                      | xyz_decimicron / 10, uint16 in 0.1 µm), its id (stable while it stays in the domain), its   |
|                      | segment                                                                                     |
|                      | (edge index); frame k's cells are rows frame_offsets[k] to frame_offsets[k + 1]          |
| truth/masks.npz      | per frame (uint16/float16, compressed): the id of the shallowest segment at every pixel (0 |
|                      | is none, else edge index + 1), its depth (µm), and its projected red-cell velocity (px per |
|                      | frame, x and y); and, with every vessel at its frame-0 position, the id and depth maps and   |
|                      | the projected velocity of the mean (cycle-averaged) flow                                    |

Coordinates: graph and cells in tissue µm (x along the limbus, y away from it, z depth); a tissue point (x, y)
appears in frame 0 at pixel ((x - field_x0) / um_per_px, (y - field_y0) / um_per_px), and in frame t after the
rotation about the image centre and the translation recorded in frames.npz.
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


def simulate(scene: Scene, n_frames=120, warm_up_s=0.6, record_cells=True, record_masks=True, progress=None) -> Result:
    """Generate the network, solve the flow, run the cells and render n_frames."""
    timings = {}
    start = time.perf_counter()
    net = network.generate(scene)
    timings["network"] = time.perf_counter() - start
    start = time.perf_counter()
    flow = haemodynamics.solve(net, scene.physiology)
    timings["flow"] = time.perf_counter() - start
    rng = np.random.default_rng([scene.seed, 1])
    start = time.perf_counter()
    red_cells = cells.Cells(net, flow, scene.physiology, rng)
    dt = 1.0 / scene.optics.fps / 3.0
    red_cells.warm_up(warm_up_s, dt)
    red_cells.t = 0.0
    timings["cells warm-up"] = time.perf_counter() - start
    motion = camera.eye_motion(n_frames, scene, rng)
    renderer = camera.Renderer(scene, net, rng, motion)
    scale = camera.background_scale(scene)
    period = 1.0 / scene.optics.fps
    frames = np.empty((n_frames, scene.height_px, scene.width_px), np.uint16)
    times = np.arange(n_frames) * period
    cell_frames = []
    masks = {"edge": [], "depth": [], "velocity": []}
    x0, x1, y0, y1 = net.field_of_view
    # cells are recorded where the camera can see them: the field plus the furthest the tissue moves, plus the
    # rotation's reach at the corners
    reach = np.hypot(motion.dx, motion.dy).max() + np.abs(motion.rotation).max() * np.hypot(scene.width_px, scene.height_px) / 2
    pad = (reach + 8.0) * scene.anatomy.um_per_px

    def cells_at(t):
        while red_cells.t < t - 1e-9:
            red_cells.step(min(dt, t - red_cells.t))
        return red_cells.positions()

    start = time.perf_counter()
    for k in range(n_frames):
        irradiance, glint = renderer.expose(k, cells_at)
        frames[k] = renderer.sensor(irradiance, glint, scale)
        mid = k + 0.5 * scene.optics.exposure_s / period
        if record_cells:
            xyz = red_cells.positions()
            inside = (xyz[:, 0] > x0 - pad) & (xyz[:, 0] < x1 + pad) & (xyz[:, 1] > y0 - pad) & (xyz[:, 1] < y1 + pad)
            cell_frames.append(dict(xyz=xyz[inside].astype(np.float32), id=red_cells.ids[inside].astype(np.uint32),
                                    edge=red_cells.edge[inside].astype(np.uint16)))
        if record_masks:
            edge_map, depth_map, velocity_map = truth_maps(net, flow, renderer, mid, k * period)
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
    projected red-cell velocity (px/frame, x and y), drawn with each segment's diameter, deepest first.
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
        pts = e.points[:: max(1, int(12 / max(np.median(np.linalg.norm(np.diff(e.points, axis=0), axis=1)), 1e-6)))]
        if len(pts) < 2:
            pts = e.points[[0, -1]]
        q = renderer.project(pts, frame_time) / renderer.ss * ss        # px at the mask's own supersampling
        if (q[:, 0].max() < -10 or q[:, 0].min() > w * ss + 10 or q[:, 1].max() < -10 or q[:, 1].min() > h * ss + 10):
            continue
        local = e.local_diameter[:: max(1, int(12 / max(np.median(np.linalg.norm(np.diff(e.points, axis=0), axis=1)), 1e-6)))]
        if len(local) != len(pts):
            local = np.full(len(pts), e.diameter)
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
            thickness = max(1, int(round(0.5 * (local[j] + local[j + 1]) / um * ss)))
            cv2.line(edge_map, a, b, int(index + 1), thickness)
            cv2.line(depth_map, a, b, float(np.mean(pts[j:j + 2, 2])), thickness)
            cv2.line(vx, a, b, float(speed[j] * direction[j, 0]), thickness)
            cv2.line(vy, a, b, float(speed[j] * direction[j, 1]), thickness)
    pick = (slice(ss // 2, None, ss), slice(ss // 2, None, ss))
    return (edge_map[pick].astype(np.uint16), depth_map[pick].astype(np.float16),
            np.stack([vx[pick], vy[pick]]).astype(np.float16))


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


def write_burst(result: Result, folder, name=None):
    """Write the frames in the recorder's format and the ground truth under truth/. Returns the folder."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
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
    (truth / "scene.json").write_text(json.dumps(_jsonable(asdict(scene)), indent=1), encoding="utf-8")
    (truth / "graph.json").write_text(json.dumps(graph_dict(result.network, result.flow)), encoding="utf-8")
    flows = np.stack([result.flow.flow_at(t) for t in result.times])
    scale = flows / np.where(np.abs(result.flow.flow) > 1e-12, result.flow.flow, 1.0)
    m = result.motion
    np.savez_compressed(truth / "frames.npz", time_s=result.times, dx_px=m.dx, dy_px=m.dy, rotation_rad=m.rotation,
                        focus_um=m.focus, saccade_frames=m.saccades, edge_flow_pl_s=flows.astype(np.float32),
                        edge_rbc_velocity_mm_s=(result.flow.rbc_velocity[None] * scale).astype(np.float32))
    if result.cell_frames:
        counts = np.array([len(c["id"]) for c in result.cell_frames])
        xyz = np.concatenate([c["xyz"] for c in result.cell_frames])
        origin = np.floor(xyz.min(axis=0)) - 1.0 if len(xyz) else np.zeros(3)
        np.savez_compressed(truth / "cells.npz", frame_offsets=np.r_[0, np.cumsum(counts)], origin_um=origin,
                            xyz_decimicron=np.clip(np.round((xyz - origin) * 10.0), 0, 65535).astype(np.uint16),
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


def load_graph(folder):
    """The truth graph of a written burst as a dict (see graph_dict)."""
    return json.loads((Path(folder) / "truth" / "graph.json").read_text(encoding="utf-8"))
