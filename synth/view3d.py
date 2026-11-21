"""3D views of a network: an interactive plotly figure and a rotating render.

Both draw every segment's centreline in 3D on the curved globe (the same mapping the camera uses: arc lengths along
the surface, depth along the radius), coloured by kind and with line width following the diameter. Depth is drawn
exaggerated (z_scale) so the layers, a few hundred micrometres apart over millimetres of field, stay visible.
"""
from __future__ import annotations

import math

import numpy as np

KIND_COLORS = {"arteriole": "#e34948", "venule": "#2a78d6", "capillary": "#e87ba4", "episcleral artery": "#9b1c1c",
               "episcleral vein": "#123f7a", "deep artery": "#eda100", "deep vein": "#1baf7a",
               "communicating artery": "#008300", "communicating vein": "#6bd16b", "perforating artery": "#000000",
               "perforating vein": "#6250d6", "arteriovenous anastomosis": "#c97d00"}


def globe_coordinates(network, eye_radius_um, z_scale=1.0):
    """Every edge's centreline on the globe: x, y in mm across the field's centre, z in mm (negative downwards,
    the surface's sagitta included, depth below it multiplied by z_scale)."""
    x0, x1, y0, y1 = network.field_of_view
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    R = eye_radius_um
    out = []
    for e in network.edges:
        u, v, depth = e.points[:, 0] - cx, e.points[:, 1] - cy, e.points[:, 2]
        r = np.hypot(u, v)
        theta = r / R
        scale = np.where(r > 1e-9, np.sin(theta) * R / np.maximum(r, 1e-9), 1.0)
        sag = R * (1 - np.cos(theta))
        out.append(np.column_stack([u * scale * 1e-3, v * scale * 1e-3, -(sag + z_scale * depth) * 1e-3]))
    return out


def plotly_figure(network, eye_radius_um=12500.0, z_scale=3.0, kinds=None):
    """An interactive 3D figure of the network (plotly), one trace per vessel kind."""
    import plotly.graph_objects as go

    paths = globe_coordinates(network, eye_radius_um, z_scale)
    figure = go.Figure()
    for kind, color in KIND_COLORS.items():
        if kinds is not None and kind not in kinds:
            continue
        chosen = [(p, e) for p, e in zip(paths, network.edges) if e.kind == kind]
        if not chosen:
            continue
        xs, ys, zs = [], [], []
        for p, _ in chosen:
            step = max(1, len(p) // 40)
            q = p[::step] if len(p) > 2 else p
            if not np.array_equal(q[-1], p[-1]):
                q = np.vstack([q, p[-1]])
            xs += list(q[:, 0]) + [None]
            ys += list(q[:, 1]) + [None]
            zs += list(q[:, 2]) + [None]
        width = float(np.clip(np.median([e.diameter for _, e in chosen]) / 12.0, 1.0, 8.0))
        figure.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=kind,
                                      line=dict(color=color, width=width), hoverinfo="name"))
    figure.update_layout(scene=dict(xaxis_title="x along the limbus (mm)", yaxis_title="y away from the limbus (mm)",
                                    zaxis_title=f"depth (mm, x{z_scale:g})", aspectmode="data"),
                         margin=dict(l=0, r=0, t=30, b=0), legend=dict(itemsizing="constant"),
                         title="the 3D network on the globe (depth exaggerated)")
    return figure


def rotating_frames(network, eye_radius_um=12500.0, z_scale=3.0, n_frames=48, size=(640, 480), elevation=28):
    """A turntable of the network: (n_frames, h, w, 3) 8-bit colour frames rendered with matplotlib."""
    import io

    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PIL import Image

    paths = globe_coordinates(network, eye_radius_um, z_scale)
    segments, colors, widths = [], [], []
    for p, e in zip(paths, network.edges):
        step = max(1, len(p) // 30)
        q = p[::step]
        if len(q) < 2:
            q = p[[0, -1]]
        segments.extend(np.stack([q[:-1], q[1:]], 1))
        colors.extend([KIND_COLORS.get(e.kind, "#888888")] * (len(q) - 1))
        widths.extend([float(np.clip(e.diameter / 25.0, 0.3, 3.0))] * (len(q) - 1))
    allpts = np.vstack(paths)
    frames = []
    figure = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    axis = figure.add_subplot(projection="3d")
    collection = Line3DCollection(segments, colors=colors, linewidths=widths)
    axis.add_collection3d(collection)
    for k, lim in enumerate(("x", "y", "z")):
        getattr(axis, f"set_{lim}lim")(allpts[:, k].min(), allpts[:, k].max())
    axis.set_box_aspect(np.ptp(allpts, axis=0) + 1e-6)
    axis.set_axis_off()
    for k in range(n_frames):
        axis.view_init(elev=elevation, azim=360.0 * k / n_frames)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        frames.append(np.asarray(Image.open(buffer).convert("RGB")))
    plt.close(figure)
    return np.stack(frames)


def eye_figure(eye, kinds=None, step=4):
    """An interactive 3D figure of a whole eye's coarse network on the globe (eye coordinates, mm), with the globe
    as a faint sphere and the limbus as a ring."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for kind, color in list(KIND_COLORS.items()) + [("ciliary artery", "#7a0000"), ("conjunctival artery", "#e34948"),
                                                   ("conjunctival vein", "#2a78d6"), ("limbal vein", "#5f9fe0")]:
        if kinds is not None and kind not in kinds:
            continue
        chosen = [e for e in eye.network.edges if e.kind == kind]
        if not chosen:
            continue
        xs, ys, zs = [], [], []
        for e in chosen:
            q = e.points[::step] if len(e.points) > 2 * step else e.points
            q = np.vstack([q, e.points[-1]]) * 1e-3
            xs += list(q[:, 0]) + [None]
            ys += list(q[:, 1]) + [None]
            zs += list(q[:, 2]) + [None]
        width = float(np.clip(np.median([e.diameter for e in chosen]) / 20.0, 1.0, 6.0))
        figure.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=kind, line=dict(color=color, width=width)))
    R = eye.anatomy.radius_um * 1e-3 * 0.995
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi / 1.6:20j]
    figure.add_trace(go.Surface(x=R * np.cos(u) * np.sin(v), y=R * np.sin(u) * np.sin(v), z=R * np.cos(v),
                                opacity=0.15, showscale=False, colorscale=[[0, "#dddddd"], [1, "#dddddd"]],
                                hoverinfo="skip", name="globe"))
    t = np.linspace(0, 2 * np.pi, 200)
    rl = eye.anatomy.limbus_radius_um * 1e-3
    zl = math.sqrt(max(R ** 2 - rl ** 2, 0))
    figure.add_trace(go.Scatter3d(x=rl * np.cos(t), y=rl * np.sin(t), z=np.full(200, zl), mode="lines",
                                  name="limbus", line=dict(color="black", width=3)))
    figure.update_layout(scene=dict(aspectmode="data", xaxis_title="temporal (mm)", yaxis_title="superior (mm)",
                                    zaxis_title="anterior (mm)"), margin=dict(l=0, r=0, t=30, b=0),
                         title="the coarse network of the whole anterior eye")
    return figure
