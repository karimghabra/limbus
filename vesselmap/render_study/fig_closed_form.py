"""Figure: current renderer vs the closed-form line integral, both against
the exact image of the same tube."""
import os

import numpy as np
import matplotlib.pyplot as plt

import closed_form_check as cf
import render_figures as rf
from render_figures import AQUA, BLUE, DIVERGE, GRID, INK, INK2, ORANGE, SURF

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    rows = [r for r in cf.run() if not r["title"].startswith("Straight")]
    fig, axs = plt.subplots(len(rows), 3, figsize=(11.6, 3.2 * len(rows)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.45], wspace=0.3, hspace=0.45))
    fig.subplots_adjust(top=0.815)
    heads = ["Current renderer − exact", "Closed-form line integral − exact",
             "Profile along the green line"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, pad=10)
    lim = 40.0
    for i, row in enumerate(rows):
        res = row["res"]
        _, x0, y0, y1 = row["prof"]
        for j, key in enumerate(["current", "adaptive"]):
            ax = axs[i, j]
            diff = 100 * (row[key] - res["od_e"]) / res["peak"]
            im = ax.imshow(diff, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
            worst = diff.flat[np.abs(diff).argmax()]
            ax.text(0.02, -0.04, f"largest error {worst:+.1f}% of vessel peak".replace("-", "−"),
                    transform=ax.transAxes, va="top", color=INK2, fontsize=8.5)
            ax.plot([x0, x0], [y0, y1], color=AQUA, lw=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
        axs[i, 0].set_ylabel(row["title"], fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
        ax = axs[i, 2]
        ss = res["ss"]
        yf = np.arange(y0 * ss, y1 * ss + 1)
        yp = np.arange(y0, y1 + 1)
        ax.plot(yf / ss - y0, res["od_fine"][yf, x0 * ss], color=BLUE, lw=2, label="exact", zorder=2)
        ax.plot(yp - y0, row["current"][yp, x0], color=ORANGE, lw=2, marker="o", ms=4, mec=SURF,
                mew=1, label="current renderer", zorder=3)
        ax.plot(yp - y0, row["adaptive"][yp, x0], color=AQUA, lw=0, marker="o", ms=5.5, mec=SURF,
                mew=1.2, label="closed form", zorder=4)
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
    cax = fig.add_axes([axs[-1, 0].get_position().x0, 0.035,
                        axs[-1, 1].get_position().x1 - axs[-1, 0].get_position().x0, 0.007])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-lim, -20, 0, 20, lim])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label("red: too light  ·  blue: too dark   (% of vessel peak)", fontsize=8.5, color=INK2)

    x = 0.07
    fig.suptitle("A closed form for the blurred tube: sum the 2-D Gaussian along the centreline",
                 x=x, ha="left", va="bottom", fontsize=13, fontweight="bold", y=0.975)
    fig.text(x, 0.962, r"$\mathrm{OD}(x)\ =\ a\ \sum_j\ A_j(u_j)\ \sum_k\ W_k\ [\ B_k(d_j)\ -\ \kappa_j\,D_k(d_j)\ ]$",
             fontsize=13, va="top", color=INK)
    fig.text(x, 0.922,
             r"$A_j(u)=\Phi\left(\frac{u+\ell_j/2}{s}\right)-\Phi\left(\frac{u-\ell_j/2}{s}\right)$"
             r"$\qquad B_k(d)=\Phi\left(\frac{w_k-d}{s}\right)+\Phi\left(\frac{w_k+d}{s}\right)-1$"
             r"$\qquad D_k(d)=d\,B_k(d)+s^2\,[\varphi_s(d+w_k)-\varphi_s(d-w_k)]$",
             fontsize=11, va="top", color=INK)
    fig.text(x, 0.888,
             "j: centreline pieces of length ℓ_j with midpoint C_j, tangent T_j, signed curvature κ_j;  "
             "u_j, d_j: the pixel's offset along T_j and across it.\n"
             "k: the 6 nested boxes of half-width w_k = r·c_k and weight W_k (B_k is today's term).  "
             "−κ_j D_k is the tube's area Jacobian (1 − κ d′) in closed form.",
             fontsize=9, va="top", color=INK2, linespacing=1.5)
    fig.savefig(os.path.join(OUT, "vesselmap_closed_form.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
