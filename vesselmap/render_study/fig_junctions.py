"""Figure: splines that meet - current renderer and the closed form with a
union at nodes, against the exact image."""
import os

import numpy as np
import matplotlib.pyplot as plt

import junction_check as jc
from render_figures import AQUA, BLUE, DIVERGE, GRID, INK, INK2, ORANGE, SURF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
TAG = {"add": "separate vessels: densities add", "union": "joined lumen: union of the tubes"}


def main():
    rows = jc.run()
    I0 = 0.85
    n = len(rows)
    fig, axs = plt.subplots(n, 5, figsize=(16.2, 3.2 * n),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.3], wspace=0.25, hspace=0.45))
    fig.subplots_adjust(top=0.9, left=0.1)
    heads = ["Exact", "Current renderer", "Current − exact",
             "Closed form + union at nodes − exact", "Profile along the green line"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, pad=10)
    lim = 30.0
    for i, row in enumerate(rows):
        sc = row["sc"]
        x0, y0, y1 = sc["prof"]
        for j, od in enumerate([row["exact"], row["current"]]):
            axs[i, j].imshow(I0 * np.exp(-od), cmap="gray", vmin=I0 * np.exp(-0.26), vmax=I0,
                             interpolation="nearest")
        for j, key in [(2, "current"), (3, "nodes")]:
            e = row["err"][key]
            im = axs[i, j].imshow(e, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
            worst = e.flat[np.abs(e).argmax()]
            axs[i, j].text(0.02, -0.04, f"largest error {worst:+.1f}% of peak".replace("-", "−"),
                           transform=axs[i, j].transAxes, va="top", color=INK2, fontsize=8.5)
        for j in range(4):
            ax = axs[i, j]
            ax.plot([x0, x0], [y0, y1], color=AQUA, lw=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
        axs[i, 0].text(-0.2, 0.5, sc["name"], transform=axs[i, 0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=10.5, color=INK, fontweight="bold")
        axs[i, 0].text(-0.08, 0.5, TAG[sc["mode"]], transform=axs[i, 0].transAxes, rotation=90,
                       va="center", ha="center", fontsize=8.5, color=INK2)
        ax = axs[i, 4]
        yf = np.arange(y0 * 8, y1 * 8 + 1)
        yp = np.arange(y0, y1 + 1)
        ex = row["exact_fine"][yf, x0 * 8]
        ax.plot(yf / 8 - y0, ex, color=BLUE, lw=2, label="exact", zorder=2)
        ax.plot(yp - y0, row["current"][yp, x0], color=ORANGE, lw=2, marker="o", ms=4, mec=SURF,
                mew=1, label="current renderer", zorder=3)
        ax.plot(yp - y0, row["nodes"][yp, x0], color=AQUA, lw=0, marker="o", ms=5.5, mec=SURF,
                mew=1.2, label="closed form + union at nodes", zorder=4)
        top = max(ex.max(), row["current"][yp, x0].max())
        ax.set_ylim(0, 1.15 * top)
        ax.set_xlim(0, y1 - y0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(length=0)
        ax.set_ylabel("optical density")
        ax.set_xlabel("px along the line (top → bottom)")
    x2, x3 = axs[-1, 2].get_position().x0, axs[-1, 3].get_position().x1
    cax = fig.add_axes([x2, 0.045, x3 - x2, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-lim, -15, 0, 15, lim])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("red: too light  ·  blue: too dark   (% of the darker vessel's peak)",
                 fontsize=8.5, color=INK2)
    h, l = axs[0, 4].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(axs[-1, 4].get_position().x0, 0.06),
               fontsize=8.5, handlelength=1.6, frameon=False)
    fig.suptitle("Splines that meet: touching, crossing, branching", x=0.07, ha="left", va="bottom",
                 fontsize=13, fontweight="bold", y=0.968)
    fig.text(0.07, 0.962,
             "vesselmap adds every edge's density. That is exact for separate vessels, but where lumens join "
             "it counts the overlap twice and leaves notches at the flat end caps.\nThe fix: within a few px of "
             "each node, draw the union of the incident tubes as polygons and blur them in closed form "
             "(Gaussian mass of a polygon via Owen's T).",
             color=INK2, fontsize=9.5, va="top", linespacing=1.5)
    fig.savefig(os.path.join(OUT, "vesselmap_junctions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
