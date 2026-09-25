"""Plotting helpers and the interactive vessel explorer."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon

from . import morphometry as mo

NODE_STYLE = {
    "divergent": ("#d62728", "o", "divergent (1 in → 2+ out)"),
    "convergent": ("#1f77b4", "o", "convergent (2+ in → 1 out)"),
    "inlet": ("#2ca02c", ">", "inlet (enters FOV)"),
    "outlet": ("#9467bd", "s", "outlet (leaves FOV)"),
    "source": ("#ff7f0e", "^", "source (inconsistent)"),
    "sink": ("#8c564b", "v", "sink (inconsistent)"),
    "through": ("#7f7f7f", "o", "through"),
    "partially resolved": ("#bcbd22", "D", "partially resolved"),
}


def stretch(img, lo=1, hi=99.5):
    a = np.asarray(img, float)
    l, h = np.nanpercentile(a, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-12), 0, 1)


def show(ax, img, title=None, cmap="gray", **kw):
    ax.imshow(stretch(img) if cmap == "gray" else img, cmap=cmap, **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def units(net):
    """(length factor, length unit, velocity factor from px/frame, velocity unit)."""
    if net.um_per_px:
        return net.um_per_px, "µm", net.um_per_px * net.fps / 1000.0, "mm/s"
    return 1.0, "px", net.fps, "px/s"


def _node_label(net, n):
    return net.dg.nodes[n].get("flow_type", "") if net.dg is not None else ""


def flow_arrows(ax, xy, n=2, color="k", size=10, lw=1.2, zorder=6):
    """Arrowheads along a polyline in its point order (= flow direction once oriented)."""
    if len(xy) < 6:
        return
    for f in np.linspace(0, 1, n + 2)[1:-1]:
        i = int(f * (len(xy) - 3))
        ax.add_patch(FancyArrowPatch(xy[i], xy[i + 3], arrowstyle="-|>", mutation_scale=size, color=color,
                                     lw=lw, zorder=zorder))


def oriented_xy(ves):
    return ves.spline.xy if ves.flow_sign >= 0 else ves.spline.xy[::-1]


def draw_network(ax, net, color_by="speed", arrows=True, labels=False, lw=2.0, alpha=0.95, nodes=True,
                 cmap="turbo", vmax=None, cbar=True, highlight=None):
    """Centre-lines coloured by speed / diameter / tortuosity; arrows show flow direction."""
    lf, lu, vf, vu = units(net)
    vals = {}
    for vid, v in net.vessels.items():
        if color_by == "speed":
            vals[vid] = v.speed * vf
        elif color_by == "diameter":
            vals[vid] = v.d_median * lf
        elif color_by == "tortuosity":
            vals[vid] = v.tort["DM"]
        else:
            vals[vid] = np.nan
    arr = np.array([x for x in vals.values() if np.isfinite(x)])
    norm = colors.Normalize(0 if color_by != "tortuosity" else 1,
                            vmax or (np.percentile(arr, 98) if len(arr) else 1))
    cmap_ = plt.get_cmap(cmap)
    for vid, v in net.vessels.items():
        xy = oriented_xy(v)
        val = vals[vid]
        c = cmap_(norm(val)) if np.isfinite(val) else (0.6, 0.6, 0.6, 0.8)
        ls = "-" if (v.flow_sign != 0 or color_by != "speed") else (0, (2, 2))
        w = lw * (2 if highlight == vid else 1)
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=w, alpha=alpha, ls=ls, solid_capstyle="round", zorder=4)
        if arrows and v.flow_sign != 0:
            flow_arrows(ax, xy, n=max(1, int(v.length // 150)), color=c if np.isfinite(val) else "0.3", size=9)
        if labels:
            m = xy[len(xy) // 2]
            ax.text(m[0], m[1], vid, fontsize=6, color="w", ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="k", ec="none", alpha=0.6))
    if nodes and net.dg is not None:
        seen = set()
        for n, d in net.dg.nodes(data=True):
            ft = d.get("flow_type", "").split(" (")[0]
            if ft not in NODE_STYLE:
                continue
            c, mk, lab = NODE_STYLE[ft]
            ax.scatter(d["x"], d["y"], s=28, c=c, marker=mk, edgecolors="w", linewidths=0.6, zorder=7,
                       label=lab if ft not in seen else None)
            seen.add(ft)
    if cbar and color_by in ("speed", "diameter", "tortuosity") and len(arr):
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_)
        lab = {"speed": f"RBC speed ({vu})", "diameter": f"diameter ({lu})", "tortuosity": "DM = L / chord"}[color_by]
        plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, label=lab)
    return ax


def draw_vessel(ax, net, vid, color="#00e5ff", fill_alpha=0.35, arrows=True, label=True):
    ves = net.vessels[vid]
    ax.add_patch(Polygon(ves.polygon, closed=True, fc=color, ec=color, alpha=fill_alpha, lw=1.0, zorder=5))
    xy = oriented_xy(ves)
    ax.plot(xy[:, 0], xy[:, 1], color="w", lw=1.0, zorder=6)
    if arrows and ves.flow_sign != 0:
        flow_arrows(ax, xy, n=max(2, int(ves.length // 80)), color="yellow", size=12)
    for (cx, cy) in ves.crossings:
        ax.scatter(cx, cy, marker="x", c="m", s=40, zorder=7)
    if label:
        ax.text(*xy[0], " start", color="w", fontsize=7, zorder=8)
        ax.text(*xy[-1], " end", color="w", fontsize=7, zorder=8)


def zoom_to(ax, net, vid, pad=40):
    xy = net.vessels[vid].spline.xy
    x0, y0 = xy.min(0) - pad; x1, y1 = xy.max(0) + pad
    ax.set_xlim(max(0, x0), min(net.ref.shape[1], x1)); ax.set_ylim(min(net.ref.shape[0], y1), max(0, y0))


# ----------------------------------------------------------------------------- dashboard
def vessel_dashboard(net, vid, figsize=(15, 11)):
    """Everything measured for one vessel on one figure."""
    ves = net.vessels[vid]
    lf, lu, vf, vu = units(net)
    sp = ves.spline if ves.flow_sign >= 0 else ves.spline.reversed()
    flip = ves.flow_sign < 0
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    show(ax, net.ff)
    draw_network(ax, net, color_by="none", arrows=False, lw=0.8, alpha=0.35, nodes=False, cbar=False)
    draw_vessel(ax, net, vid)
    ax.set_title(f"{vid}: vessel mask over the reference image (yellow arrows = flow direction, "
                 f"{ves.direction_source})", fontsize=10)

    s = sp.s * lf
    d = ves.diam
    order = slice(None, None, -1) if flip else slice(None)
    dd, dfit, val = d["d"][order] * lf, d["d_fit"][order] * lf, d["valid"][order]
    ds = ves.d_smooth[order] * lf

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s, np.where(val, dd, np.nan), ".", ms=3, color="C0", label="FWHM (valid)")
    ax.plot(s, np.where(~val, dd, np.nan), ".", ms=2, color="0.7", label="excluded")
    ax.plot(s, np.where(val & ~d["subres"][order], dfit, np.nan), ".", ms=2, color="C1", alpha=0.4,
            label="box-model fit (resolved)")
    ax.plot(s, ds, "-", color="k", lw=1.5, label="smoothed d(s)")
    ax.set_xlabel(f"arc length s ({lu}), along flow"); ax.set_ylabel(f"diameter ({lu})")
    ax.set_title("diameter along the vessel", fontsize=10); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    k = sp.kappa / lf
    ax.plot(s, k, color="C2")
    ax.axhline(0, color="k", lw=0.5)
    infl = mo.tortuosity(sp)["inflection_s"] * lf
    for x in infl:
        ax.axvline(x, color="C3", lw=0.6, alpha=0.7)
    ax.set_xlabel(f"s ({lu})"); ax.set_ylabel(f"curvature κ (1/{lu})")
    ax.set_title(f"signed curvature; red = inflections ({len(infl)})", fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    t = ves.tort
    rows = [("length", f"{ves.length * lf:.1f} {lu}"),
            ("median diameter (FWHM)", f"{ves.d_median * lf:.1f} {lu}"),
            ("diameter 5–95 %", f"{np.nanpercentile(dd[val], 5):.1f} – {np.nanpercentile(dd[val], 95):.1f} {lu}"
             if val.any() else "n/a"),
            ("tortuosity DM (L/C)", f"{t['DM']:.3f}"),
            ("SOAM", f"{t['SOAM'] / lf:.4f} rad/{lu}"),
            ("mean κ²", f"{t['KSQ'] / lf ** 2:.2e} 1/{lu}²"),
            ("inflections / ICM", f"{t['n_inflections']} / {t['ICM']:.2f}"),
            ("RBC speed (median of v(t))", f"{ves.speed * vf:.3g} {vu}" if np.isfinite(ves.speed) else "not reliable"),
            ("  ridge / ST / TOF", " / ".join(f"{abs(x) * vf:.3g}" if np.isfinite(x) else "–"
                                              for x in (ves.vel['v_ridge'], ves.vel['v_st'], ves.vel['v_tof']))),
            ("direction", ves.direction_source),
            ("velocity quality", f"{ves.vel['quality']:.2f} (R² {ves.vel['r2']:.2f}), time bin ×{ves.vel.get('scale', 1)}"),
            ("flow Q", f"{ves.flow_Q(net.params.rbc_to_mean) * net.fps * lf ** 3:.3g} {lu}³/s"
             if np.isfinite(ves.speed) else "–"),
            ("dominant freq. of v(t)", f"{ves.vel['hr_bpm'] / 60:.2f} Hz" if np.isfinite(ves.vel['hr_bpm']) else "–"),
            ("crossings passed", f"{len(ves.crossings)}")]
    tb = ax.table(cellText=rows, colWidths=[0.55, 0.45], loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.2)

    # kymographs
    K, K2 = ves.K, ves.vel["K2"]
    if flip:
        K, K2 = K[:, ::-1], K2[:, ::-1]
    tmax = K.shape[0] / net.fps
    ext = [0, sp.length * lf, tmax, 0]
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(stretch(K), cmap="gray", aspect="auto", extent=ext)
    ax.set_xlabel(f"s ({lu})"); ax.set_ylabel("time (s)"); ax.set_title("raw kymograph K(t, s)", fontsize=10)
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(stretch(K2, 1, 99), cmap="gray", aspect="auto", extent=ext)
    v = ves.vel["v_ridge"] * (-1 if flip else 1)
    vbest = ves.vel["v_time"] * (-1 if flip else 1)
    if np.isfinite(vbest) and vbest != 0:
        # lines of slope ds/dt = v, spread over the whole panel
        dur = sp.length / abs(vbest) / net.fps            # time to traverse the vessel
        for t0 in np.arange(-dur, tmax, max(dur / 2, tmax / 10)):
            tt = np.array([t0, t0 + dur]); ss = (0 if vbest > 0 else sp.length) + vbest * (tt - t0) * net.fps
            ax.plot(ss * lf, tt, color="yellow", lw=0.8, alpha=0.8)
        ax.set_xlim(0, sp.length * lf); ax.set_ylim(tmax, 0)
    ax.set_xlabel(f"s ({lu})"); ax.set_title("moving component K₂ + measured slope (yellow)", fontsize=10)

    ax = fig.add_subplot(gs[2, 2])
    C, sh = ves.vel["Cmap"], ves.vel["shifts"]
    if flip:
        C, sh = C[:, ::-1], sh
    ax.imshow(C, aspect="auto", cmap="magma", extent=[sh[0] * lf, sh[-1] * lf, len(C) + 0.5, 0.5])
    r = ves.vel["ridge"]
    dsp = r["disp"] * (-1 if flip else 1)
    kb = ves.vel.get("scale", 1)
    ax.plot(np.where(r["used"], dsp, np.nan) * lf, r["lags"], "c.", ms=6, label="ridge peaks")
    if np.isfinite(v):
        ax.plot(v * kb * r["lags"] * lf, r["lags"], "c-", lw=1, label=f"fit v = {abs(v) * vf:.3g} {vu}")
    ax.set_xlabel(f"shift δ ({lu})"); ax.set_ylabel("lag (frames)" if kb == 1 else f"lag (bins of {kb} frames)")
    ax.set_title("spatio-temporal correlation C(lag, δ)" + ("" if kb == 1 else f"  [time-binned ×{kb}: slow flow]"),
                 fontsize=10); ax.legend(fontsize=7, loc="lower right")

    ax = fig.add_subplot(gs[3, 0:2])
    vt = ves.vel["v_t"] * (-1 if flip else 1) * vf
    ax.plot(ves.vel["t_s"], vt, color="C3")
    if np.isfinite(vbest):
        ax.axhline(vbest * vf, color="k", ls="--", lw=0.8, label="median = reported velocity")
    if np.isfinite(v):
        ax.axhline(v * vf, color="0.5", ls=":", lw=0.8, label="whole-window ridge fit")
    ax.set_xlabel("time (s)"); ax.set_ylabel(f"RBC velocity ({vu})")
    ax.set_title("velocity vs time (sliding-window LSPIV) – cardiac pulsatility", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[3, 2])
    tof = ves.vel["tof"]
    for j, (D, c) in enumerate(zip(tof["separations"], tof["curves"])):
        ax.plot(tof["lags"] * ves.vel.get("scale", 1) / net.fps * 1000, c, color=plt.cm.viridis(j / max(1, len(tof["curves"]) - 1)),
                lw=1, label=f"D={D * lf:.0f}{lu}" if j % 2 == 0 else None)
    ax.set_xlabel("delay τ (ms)"); ax.set_ylabel("correlation")
    ax.set_title("time-of-flight ('flicker') correlation", fontsize=10); ax.legend(fontsize=6)
    fig.suptitle(f"Vessel {vid}", fontsize=13)
    return fig


def explorer(net, default=None):
    """ipywidgets UI: pick a vessel from the list -> mask highlight + full dashboard."""
    import ipywidgets as W
    from IPython.display import display

    lf, lu, vf, vu = units(net)

    def label(vid):
        v = net.vessels[vid]
        sp = f"{v.speed * vf:.3g} {vu}" if np.isfinite(v.speed) else "–"
        arrow = {"measured": "→", "unknown": "?"}.get(v.direction_source, "⇢")
        return f"{vid}  L={v.length * lf:.0f}{lu}  d={v.d_median * lf:.1f}{lu}  v={sp} {arrow}"

    opts = [(label(k), k) for k in net.vessels]
    sort = W.Dropdown(options=["id", "length", "diameter", "speed", "tortuosity"], value="id",
                      description="sort by", layout=W.Layout(width="220px"))
    only = W.Checkbox(value=False, description="only reliable velocity")
    sel = W.Select(options=opts, value=default or opts[0][1], rows=22, layout=W.Layout(width="360px"))
    out = W.Output()

    def resort(*_):
        keyf = {"id": lambda k: k, "length": lambda k: -net.vessels[k].length,
                "diameter": lambda k: -np.nan_to_num(net.vessels[k].d_median),
                "speed": lambda k: -np.nan_to_num(net.vessels[k].speed),
                "tortuosity": lambda k: -net.vessels[k].tort["DM"]}[sort.value]
        ks = [k for k in net.vessels if not only.value or net.vessels[k].vel.get("reliable")]
        cur = sel.value
        sel.options = [(label(k), k) for k in sorted(ks, key=keyf)]
        if cur in ks:
            sel.value = cur

    def render(*_):
        with out:
            out.clear_output(wait=True)
            fig = vessel_dashboard(net, sel.value)
            plt.show(fig)

    sort.observe(resort, "value"); only.observe(resort, "value"); sel.observe(render, "value")
    ui = W.HBox([W.VBox([W.HTML("<b>Vessels</b>"), sort, only, sel]), out])
    display(ui)
    render()
    return ui


# ----------------------------------------------------------------------------- plotly network
def plotly_network(net, color_by="speed"):
    """Hover-able directed network over the image (renders in HTML exports too)."""
    import plotly.graph_objects as go
    lf, lu, vf, vu = units(net)
    img = (stretch(net.ff) * 255).astype(np.uint8)
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False, hoverinfo="skip"))
    vals = [v.speed * vf for v in net.vessels.values()]
    vmax = np.nanpercentile([x for x in vals if np.isfinite(x)], 98) if np.isfinite(vals).any() else 1
    cmap = plt.get_cmap("turbo")
    for vid, v in net.vessels.items():
        xy = oriented_xy(v)[::3]
        val = v.speed * vf
        c = colors.to_hex(cmap(min(val / vmax, 1))) if np.isfinite(val) else "#999999"
        txt = (f"<b>{vid}</b><br>L = {v.length * lf:.0f} {lu}<br>d = {v.d_median * lf:.1f} {lu}"
               f"<br>DM = {v.tort['DM']:.3f}<br>speed = {val:.3g} {vu}<br>direction: {v.direction_source}")
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color=c, width=3,
                      dash="solid" if v.flow_sign else "dot"), name=vid, text=txt, hoverinfo="text",
                      showlegend=False))
        if v.flow_sign != 0 and len(xy) > 4:
            i = len(xy) // 2
            fig.add_annotation(x=xy[i + 1, 0], y=xy[i + 1, 1], ax=xy[i - 1, 0], ay=xy[i - 1, 1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2,
                               arrowcolor=c)
    if net.dg is not None:
        xs, ys, tx, cs = [], [], [], []
        for n, d in net.dg.nodes(data=True):
            ft = d.get("flow_type", "")
            base = ft.split(" (")[0]
            if base not in NODE_STYLE:
                continue
            xs.append(d["x"]); ys.append(d["y"]); cs.append(NODE_STYLE[base][0])
            res = d.get("conservation_residual")
            tx.append(f"node {n}: {ft}" + (f"<br>mass-balance residual {res:+.2f}" if res is not None and np.isfinite(res) else ""))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=cs, size=8, line=dict(color="white", width=1)),
                                 text=tx, hoverinfo="text", showlegend=False))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      title=f"Directed vessel network – colour = RBC speed ({vu}), dotted = direction unknown")
    return fig
