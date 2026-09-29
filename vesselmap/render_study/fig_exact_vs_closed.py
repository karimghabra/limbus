"""Figure: the exact image of each tube next to the closed-form render."""
import os

import numpy as np
import matplotlib.pyplot as plt

import closed_form_check as cf
from render_figures import AQUA, BLUE, DIVERGE, GRID, INK, INK2, SURF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    rows = [r for r in cf.run() if not r["title"].startswith("Straight")]
    I0 = 0.85
    fig, axs = plt.subplots(len(rows), 4, figsize=(13.2, 3.25 * len(rows)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25], wspace=0.28, hspace=0.42))
    fig.subplots_adjust(top=0.93)
    heads = ["Exact: the tube blurred in 2-D", "Closed form (line integral)",
             "Difference, closed form − exact", "Profile along the green line"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, pad=10)
    lim = 2.0
    for i, row in enumerate(rows):
        res = row["res"]
        _, x0, y0, y1 = row["prof"]
        for j, od in enumerate([res["od_e"], row["adaptive"]]):
            axs[i, j].imshow(I0 * np.exp(-od), cmap="gray", vmin=I0 * np.exp(-0.3), vmax=I0,
                             interpolation="nearest")
        diff = 100 * (row["adaptive"] - res["od_e"]) / res["peak"]
        im = axs[i, 2].imshow(diff, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
        worst = diff.flat[np.abs(diff).argmax()]
        axs[i, 2].text(0.02, -0.04, f"largest error {worst:+.1f}% of vessel peak".replace("-", "−"),
                       transform=axs[i, 2].transAxes, va="top", color=INK2, fontsize=8.5)
        for j in range(3):
            ax = axs[i, j]
            ax.plot([x0, x0], [y0, y1], color=AQUA, lw=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
        axs[i, 0].set_ylabel(row["title"], fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
        ax = axs[i, 3]
        ss = res["ss"]
        yf = np.arange(y0 * ss, y1 * ss + 1)
        yp = np.arange(y0, y1 + 1)
        ax.plot(yf / ss - y0, res["od_fine"][yf, x0 * ss], color=BLUE, lw=2, label="exact", zorder=2)
        ax.plot(yp - y0, row["adaptive"][yp, x0], color=AQUA, lw=0, marker="o", ms=5.5, mec=SURF,
                mew=1.2, label="closed form (pixel centres)", zorder=3)
        ax.set_xlim(0, y1 - y0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.set_ylim(0, 0.34)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(length=0)
        ax.set_ylabel("optical density")
        ax.set_xlabel("px along the line (top → bottom)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8.5, handlelength=1.6)
    cax = fig.add_axes([axs[-1, 2].get_position().x0, 0.045, axs[-1, 2].get_position().width, 0.008])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-lim, -1, 0, 1, lim])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("red: too light  ·  blue: too dark   (% of vessel peak;\nnote the scale is 20× finer than before)",
                 fontsize=8.5, color=INK2)
    fig.suptitle("The exact image against the closed-form render", x=0.07, ha="left", va="bottom",
                 fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.07, 0.968, "Same spline, width, blur and halo in both. The closed form sums a separable "
             "erf term per centreline piece, with the tube's curvature Jacobian.",
             color=INK2, fontsize=9.5, va="top")
    fig.savefig(os.path.join(OUT, "vesselmap_exact_vs_closed_form.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
