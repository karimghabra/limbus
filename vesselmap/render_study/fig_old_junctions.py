"""Figure: what today's renderer draws at junctions, before blur.

Today's renderer (render.py) gives every edge its own tube, cut flat where
it meets the node (the end-cap taper with the blur taken to zero), and adds
the tubes.  The true lumen is the union of the tubes (round where they meet).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import junction_check as jc
import render_figures as rf
from render_figures import AQUA, BLUE, DIVERGE, GRID, INK, INK2, ORANGE, SURF
from vesselmap import spline as sp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
EDGE_COL = [BLUE, ORANGE, AQUA]
HALF = 12          # zoom half-size around the node, px
SS = 10            # fine grid per px


def fine_grid(c):
    # offset by an irrational fraction so no grid point sits exactly on a box edge
    xs = np.arange((c[0] - HALF) * SS, (c[0] + HALF) * SS + 1) / SS + 0.0137
    ys = np.arange((c[1] - HALF) * SS, (c[1] + HALF) * SS + 1) / SS + 0.0291
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, np.stack([gx.ravel(), gy.ravel()], 1)


def today_sharp(sc, b, P):
    """Sum over edges of the renderer's own tube with the blur taken to zero:
    nearest sample (0.75 px apart in arclength), its tangent frame, the box
    cross-section, and the end caps as hard steps at arclength 0 and L."""
    tot = np.zeros(len(P))
    for p in b["prm"]:
        u, L = sp.arclength_params(p["ctrl"], 0.75)
        n = len(p["ctrl"])
        C = sp.design_at(n, u) @ p["ctrl"]
        D = sp.design_at(n, u, 1) @ p["ctrl"]
        T = D / np.linalg.norm(D, axis=1, keepdims=True)
        arc = sp.arclength(C)
        _, j = cKDTree(C).query(P)
        v = P - C[j]
        along = arc[j] + (v * T[j]).sum(1)
        d = np.abs(v[:, 0] * T[j, 1] - v[:, 1] * T[j, 0])
        inside = (along >= 0) & (along <= arc[-1])
        tot += p["a"] * rf.box_sharp(d, p["r"]) * inside
    return tot


def union_sharp(sc, b, P):
    """The true lumen: the largest chord of any tube (they share one depth),
    tubes rounded where they meet the node."""
    out = np.zeros(len(P))
    for e, p in zip(sc["edges"], b["prm"]):
        cl = sp.design(len(p["ctrl"]), int(p["L"] / 0.02) + 2) @ p["ctrl"]
        dd, j = cKDTree(cl).query(P)
        ok = np.ones(len(P), bool)
        if e["u"] is None:
            ok &= j != 0
        if e["v"] is None:
            ok &= j != len(cl) - 1
        out = np.maximum(out, np.where(ok, p["a"] * rf.box_sharp(dd, p["r"]), 0.0))
    return out


def tube_outlines(ax, sc, b, jn, node):
    """Each edge's tube as today's renderer cuts it: two walls and, at the
    node, a flat end perpendicular to its own axis."""
    for k, (ei, kind) in enumerate(jn["inc"]):
        p = b["prm"][ei]
        at, L = jc.edge_frame(p["ctrl"])
        if kind == "through":
            s0 = float(sp.arclength(sp.design(len(p["ctrl"]), 20000) @ p["ctrl"])[
                np.argmin(np.linalg.norm(sp.design(len(p["ctrl"]), 20000) @ p["ctrl"] - node, axis=1))])
            s = np.linspace(max(0, s0 - 2 * HALF), min(L, s0 + 2 * HALF), 400)
        else:
            s = np.linspace(0, min(L, 2 * HALF), 400)
        C, N, _ = at(s)
        col = EDGE_COL[k]
        for sgn in (1, -1):
            ax.plot(*(C + sgn * p["r"] * N).T, color=col, lw=1.6, zorder=5)
        if kind != "through":
            ax.plot(*np.stack([C[0] + p["r"] * N[0], C[0] - p["r"] * N[0]]).T, color=col, lw=1.6, zorder=5)
        ax.plot(*C.T, color=col, lw=0.8, zorder=5, alpha=0.8)


def union_outline(ax, sc, b, jn, node):
    polys = []
    for ei, kind in jn["inc"]:
        p = b["prm"][ei]
        at, L = jc.edge_frame(p["ctrl"])
        s = np.linspace(0, L, 3000)
        C, N, _ = at(s)
        polys.append(Polygon(np.vstack([C + p["r"] * N, (C - p["r"] * N)[::-1]])).buffer(0))
        if kind != "through":
            polys.append(Point(node).buffer(p["r"], quad_segs=32))
    U = unary_union(polys)
    for g in getattr(U, "geoms", [U]):
        ax.plot(*np.asarray(g.exterior.coords).T, color=INK, lw=1.4, zorder=5)


def style(ax, c):
    ax.set_xlim(c[0] - HALF, c[0] + HALF)
    ax.set_ylim(c[1] + HALF, c[1] - HALF)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def main():
    rows = [r for r in jc.run() if r["sc"]["mode"] == "union"]
    I0 = 0.85
    n = len(rows)
    fig, axs = plt.subplots(n, 4, figsize=(15, 3.9 * n), gridspec_kw=dict(wspace=0.12, hspace=0.32))
    fig.subplots_adjust(top=0.85, left=0.09)
    heads = ["Today: each edge's own tube, added\n(before blur)", "True lumen: the union of the tubes\n(before blur)",
             "Today − true, before blur", "Today − exact, after blur"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, pad=10)
    for i, row in enumerate(rows):
        sc, b = row["sc"], row["b"]
        jn = sc["junctions"][0]
        node = b["node_xy"][jn["node"]]
        gx, gy, P = fine_grid(node)
        old = today_sharp(sc, b, P).reshape(gx.shape)
        tru = union_sharp(sc, b, P).reshape(gx.shape)
        ext = [gx[0, 0] - 0.5 / SS, gx[0, -1] + 0.5 / SS, gy[-1, 0] + 0.5 / SS, gy[0, 0] - 0.5 / SS]
        for j, od in enumerate([old, tru]):
            axs[i, j].imshow(I0 * np.exp(-od), cmap="gray", vmin=I0 * np.exp(-0.62), vmax=I0,
                             extent=ext, interpolation="nearest")
        amax = max(p["a"] for p in b["prm"])
        diff = 100 * (old - tru) / amax
        im1 = axs[i, 2].imshow(diff, cmap=DIVERGE, vmin=-100, vmax=100, extent=ext, interpolation="nearest")
        twice = (old - tru > 1e-3 * amax).sum() / SS ** 2
        notch = (tru - old > 1e-3 * amax).sum() / SS ** 2
        axs[i, 2].text(0.02, -0.03, f"counted twice: {twice:.0f} px²   ·   missing: {notch:.1f} px²",
                       transform=axs[i, 2].transAxes, va="top", color=INK2, fontsize=8.5)
        # blurred error from the earlier comparison (crop coordinates -> canvas coordinates)
        M = b["M"]
        e = row["err"]["current"]
        im2 = axs[i, 3].imshow(e, cmap=DIVERGE, vmin=-30, vmax=30, interpolation="nearest",
                               extent=[M - 0.5, M + 63.5, M + 63.5, M - 0.5])
        worst = e.flat[np.abs(e).argmax()]
        axs[i, 3].text(0.02, -0.03, f"largest error {worst:+.1f}% of peak".replace("-", "−"),
                       transform=axs[i, 3].transAxes, va="top", color=INK2, fontsize=8.5)
        tube_outlines(axs[i, 0], sc, b, jn, node)
        union_outline(axs[i, 1], sc, b, jn, node)
        for ax in axs[i]:
            ax.scatter(*node, s=28, color=INK, zorder=7, edgecolor=SURF, lw=1)
            style(ax, node)
        axs[i, 0].text(-0.06, 0.5, sc["name"], transform=axs[i, 0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    # legends and colour bars
    h = [plt.Line2D([], [], color=c, lw=1.6) for c in EDGE_COL]
    fig.legend(h, ["edge 1", "edge 2", "edge 3"], loc="upper left", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(axs[-1, 0].get_position().x0, 0.075))
    fig.text(axs[-1, 0].get_position().x0, 0.047,
             "Coloured lines: each edge's tube walls and its flat end at the node.\n"
             "Black line (2nd column): the outline of the true lumen.", fontsize=8.5, color=INK2, va="top")
    for j, (im, lim, lab) in enumerate([(im1, 100, "blue: counted twice  ·  red: missing\n(% of the densest edge's centre)"),
                                         (im2, 30, "blue: too dark  ·  red: too light\n(% of vessel peak)")]):
        pos = axs[-1, 2 + j].get_position()
        cax = fig.add_axes([pos.x0, 0.06, pos.width, 0.007])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.set_ticks([-lim, 0, lim])
        cb.ax.tick_params(labelsize=8, length=0)
        cb.set_label(lab, fontsize=8.3, color=INK2)
    fig.suptitle("What today's renderer draws at a junction", x=0.07, ha="left", va="bottom",
                 fontsize=13, fontweight="bold", y=0.945)
    fig.text(0.07, 0.938,
             "render.py gives each edge its own tube, cut flat where it meets the node, and adds the tubes. "
             "Where they overlap the blood is counted twice;\nwhere the flat ends leave a corner uncovered it is "
             "missing. Views are 24 × 24 px around the node, before and after the blur.",
             color=INK2, fontsize=9.5, va="top", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "vesselmap_junctions_today.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
