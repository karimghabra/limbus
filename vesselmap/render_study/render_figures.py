"""Figures: how vesselmap's renderer (render.py, the real NetworkModel) draws
tortuous tubes, against the exact image of the same tube (same spline
centreline, same 6-box cross-section, blurred by a true 2-D Gaussian and the
same two-component PSF)."""
import os
import sys

os.environ.setdefault("VESSELMAP_COMPILE", "0")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
from scipy.special import erf
from scipy.spatial import cKDTree

from vesselmap import spline as sp
from vesselmap.network import VesselNetwork
from vesselmap.render import NetworkModel, _C, _W

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (dataviz reference instance, light mode) -------------------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIVERGE = LinearSegmentedColormap.from_list(
    "div", ["#8f2424", "#e34948", "#f0efec", "#2a78d6", "#104281"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


# ---- geometry helpers ---------------------------------------------------
def turtle(parts, step=0.05):
    """Path from ('line', L) and ('arc', radius, degrees) parts."""
    p, h, out = np.zeros(2), 0.0, [np.zeros(2)]
    for part in parts:
        if part[0] == "line":
            n = int(part[1] / step)
            for _ in range(n):
                p = p + step * np.array([np.cos(h), np.sin(h)])
                out.append(p)
        else:
            rad, deg = part[1], np.deg2rad(part[2])
            n = int(abs(deg) * rad / step)
            dh = deg / n
            for _ in range(n):
                h += dh
                p = p + rad * abs(dh) * np.array([np.cos(h), np.sin(h)])
                out.append(p)
    return np.array(out)


def rotate(xy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return xy @ np.array([[c, -s], [s, c]]).T


def box_sharp(d, r):
    return ((np.abs(d)[..., None] < r * _C) * _W).sum(-1)


def box_blur(d, r, s):
    rr, ss = r * _C, np.sqrt(2) * s
    return (0.5 * (erf((rr - d[..., None]) / ss) + erf((rr + d[..., None]) / ss)) * _W).sum(-1)


def sharp_strands(cl, P, r):
    """Sharp optical density of the tube at points P: every strand of the
    centreline passing within r adds its own chord, so where the tube
    crosses itself (at another depth) the two strands add."""
    tree = cKDTree(cl)
    d0, _ = tree.query(P, distance_upper_bound=r)
    near = np.flatnonzero(np.isfinite(d0))
    out = np.zeros(len(P))
    last = len(cl) - 1
    for i, idx in zip(near, tree.query_ball_point(P[near], r)):
        idx = np.sort(idx)
        dist = np.linalg.norm(cl[idx] - P[i], axis=1)
        cut = np.flatnonzero(np.diff(idx) > 1) + 1
        tot = 0.0
        for di, ii in zip(np.split(dist, cut), np.split(idx, cut)):
            k = ii[di.argmin()]
            if k == 0 or k == last:          # beyond a free end: the tube is cut flat there
                continue
            tot += box_sharp(np.array([di.min()]), r)[0]
        out[i] = tot
    return out


# ---- one scene ----------------------------------------------------------
def scene(xy, r, s, a, H, W, M=None, ss=8):
    """Render one tube through vesselmap and exactly.  Returns a dict."""
    hw, hs = 0.25, 5.0                                   # render.py's default optics
    M = M or int(np.ceil(4 * np.hypot(s, hs))) + 2        # margin so the halo sees the whole tube
    Hc, Wc = H + 2 * M, W + 2 * M
    net = VesselNetwork((Hc, Wc))
    n = len(xy)
    eid = net.add_edge_dense(xy + M, np.full(n, r), np.full(n, s), np.full(n, a))
    model = NetworkModel(net, np.zeros((Hc, Wc), np.float32), np.ones((Hc, Wc), np.float32))
    with torch.no_grad():
        od_m = model.optical_density().numpy()
        C, T, R, S, A, arc, L = (t.numpy() for t in model.samples())
        hw, hs = (float(v) for v in model.halo())
    e = net.edges[eid]
    # the exact image of the SAME spline centreline and profile values
    Lc = net.length(eid)
    cl = sp.design(len(e.ctrl), int(Lc / 0.02) + 2) @ e.ctrl
    rr, sss, aa = float(np.median(R)), float(np.median(S)), float(np.median(A))
    xs = np.arange(0, Wc * ss) / ss
    ys = np.arange(0, Hc * ss) / ss
    gx, gy = np.meshgrid(xs, ys)
    sharp = sharp_strands(cl, np.stack([gx.ravel(), gy.ravel()], 1), rr).reshape(gx.shape)
    core = gaussian_filter(sharp, sss * ss, mode="constant", truncate=4)
    halo = gaussian_filter(sharp, np.hypot(sss, hs) * ss, mode="constant", truncate=4)
    od_fine = aa * ((1 - hw) * core + hw * halo)
    od_e = od_fine[::ss, ::ss]
    peak = aa * ((1 - hw) * box_blur(np.zeros(1), rr, sss)[0] + hw * box_blur(np.zeros(1), rr, np.hypot(sss, hs))[0])
    crop = (slice(M, M + H), slice(M, M + W))
    return dict(model=model, net=net, eid=eid, M=M, ss=ss, od_m=od_m[crop], od_e=od_e[crop],
                od_fine=od_fine[M * ss:(M + H) * ss, M * ss:(M + W) * ss], peak=peak,
                C=C - M, T=T, arc=arc, L=float(L[0]), ctrl=e.ctrl - M,
                e_xy=model.e_xy.numpy() - M, e_samp=model.e_samp.numpy(), r=rr, s=sss, a=aa)


# ---- the four tubes -----------------------------------------------------
def gentle():
    x = np.arange(-30, 94, 0.05)
    xy = np.stack([x, 32 + 4 * np.sin(2 * np.pi * (x - 17) / 60)], 1)
    return xy, dict(r=1.5, s=1.5, a=0.35), ("col", 32, 22, 42), "Gentle wiggle  (ρ ≈ 23 px)"


def bend():
    rho, turn = 4.0, 135.0
    xy = turtle([("line", 60), ("arc", rho, turn), ("line", 60)])
    i_apex = np.argmin(np.abs(sp.arclength(xy) - (60 + rho * np.deg2rad(turn) / 2)))
    t = xy[i_apex + 1] - xy[i_apex - 1]
    xy = rotate(xy - xy[i_apex], -np.arctan2(t[1], t[0]))
    # put the centre of curvature below the apex (image y grows downwards)
    if xy[i_apex + 40, 1] < 0:
        xy[:, 1] *= -1
    xy = xy + [32, 22]
    return xy, dict(r=1.5, s=2.0, a=0.35), ("col", 32, 12, 42), "Tight bend  (ρ = 4 px, 135°)"


def hairpin():
    gap = 5.0
    xy = turtle([("line", 70), ("arc", gap / 2, 180), ("line", 70)])
    xy = xy - [70, 0] + [46, 32 - gap / 2]
    return xy, dict(r=1.5, s=2.0, a=0.35), ("col", 26, 18, 46), "Hairpin  (legs 5 px apart)"


def coil():
    b, c = 5.5, 2.3
    t = np.arange(-4 * np.pi, 6 * np.pi, 0.002)
    xy = np.stack([c * t + b * np.sin(t), -b * np.cos(t)], 1)
    dl = brentq(lambda d: b * np.sin(d) - c * d, 0.5, 3.0)       # self-crossing near t = pi
    xc = c * (np.pi - dl) + b * np.sin(np.pi - dl)
    shift = np.array([32 - (xc + c * 2 * np.pi), 32])
    xy = xy + shift
    return xy, dict(r=1.0, s=1.2, a=0.35), ("col", 32, 18, 46), "Coil that crosses itself"


SCENES = [gentle, bend, hairpin, coil]


# ---- figure 1: the comparison -------------------------------------------
def fig_comparison(results):
    I0 = 0.85
    fig, axs = plt.subplots(len(results), 4, figsize=(13.2, 3.25 * len(results)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25], wspace=0.28, hspace=0.42))
    heads = ["Exact: the tube blurred in 2-D", "vesselmap render (render.py)",
             "Difference, vesselmap − exact", "Profile along the green line"]
    for j, t in enumerate(heads):
        axs[0, j].set_title(t, pad=10)
    lim = 40.0
    for i, (res, title, prof) in enumerate(results):
        H, W = res["od_m"].shape
        for j, od in enumerate([res["od_e"], res["od_m"]]):
            ax = axs[i, j]
            ax.imshow(I0 * np.exp(-od), cmap="gray", vmin=I0 * np.exp(-0.3), vmax=I0,
                      interpolation="nearest")
        diff = 100 * (res["od_m"] - res["od_e"]) / res["peak"]
        im = axs[i, 2].imshow(diff, cmap=DIVERGE, vmin=-lim, vmax=lim, interpolation="nearest")
        worst = diff[np.unravel_index(np.abs(diff).argmax(), diff.shape)]
        axs[i, 2].text(0.02, -0.04, f"largest error {worst:+.0f}% of vessel peak".replace("-", "−"),
                       transform=axs[i, 2].transAxes, va="top", color=INK2, fontsize=8.5)
        _, x0, y0, y1 = prof
        for j in range(3):
            ax = axs[i, j]
            ax.plot([x0, x0], [y0, y1], color=AQUA, lw=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
        axs[i, 0].set_ylabel(title, fontsize=10.5, color=INK, fontweight="bold", labelpad=10)
        # profile
        ax = axs[i, 3]
        ss = res["ss"]
        yf = np.arange(y0 * ss, y1 * ss + 1)
        ax.plot(yf / ss - y0, res["od_fine"][yf, x0 * ss], color=BLUE, lw=2, label="exact")
        yp = np.arange(y0, y1 + 1)
        ax.plot(yp - y0, res["od_m"][yp, x0], color=ORANGE, lw=2, marker="o", ms=4,
                mec=SURF, mew=1, label="vesselmap")
        ax.set_xlim(0, y1 - y0)
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
    cb.set_ticks([-lim, -20, 0, 20, lim])
    cb.ax.tick_params(labelsize=8, color=MUTED, length=0)
    cb.set_label("red: vesselmap too light  ·  blue: too dark   (% of vessel peak)", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.93)
    fig.suptitle("How vesselmap renders tortuous tubes, against the exact image of the same spline",
                 x=0.07, ha="left", va="bottom", fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.07, 0.968, "Same centreline, width, blur and halo in both. Only the way the blur is applied differs: "
             "vesselmap uses a 1-D erf across the local tangent of each pixel's nearest sample.",
             color=INK2, fontsize=9.5, va="top")
    fig.savefig(os.path.join(OUT, "vesselmap_tortuous_render.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- figure 2: the mechanism --------------------------------------------
def spokes(ax, res, window, colour_fn, alpha=0.55):
    (x0, x1), (y0, y1) = window
    P, j = res["e_xy"], res["e_samp"]
    C = res["C"]
    keep = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    for p, q, col in zip(P[keep], C[j[keep]], colour_fn(j[keep])):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.6, alpha=alpha, solid_capstyle="round")
    ax.scatter(P[keep, 0], P[keep, 1], s=5, color=MUTED, zorder=3, lw=0)


def frame(ax, window):
    (x0, x1), (y0, y1) = window
    for x in np.arange(np.ceil(x0) - 0.5, x1 + 1, 1):
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)
    for y in np.arange(np.ceil(y0) - 0.5, y1 + 1, 1):
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)
    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(y1 + 0.5, y0 - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)


def centreline(ax, res, window, samples=True, ctrl=True):
    C = res["C"]
    ax.plot(C[:, 0], C[:, 1], color=INK, lw=2, zorder=4, solid_capstyle="round")
    if samples:
        ax.scatter(C[:, 0], C[:, 1], s=6, color=INK, zorder=5, lw=0)
    if ctrl:
        c = res["ctrl"]
        ax.plot(c[:, 0], c[:, 1], color=INK2, lw=0.8, zorder=4)
        ax.scatter(c[:, 0], c[:, 1], s=46, marker="s", facecolor=SURF, edgecolor=INK2, lw=1.4, zorder=6)


def fig_mechanism(res_bend, res_hp):
    fig = plt.figure(figsize=(15, 6.4))
    y0, h = 0.30, 0.56                                   # common panel band
    hin = h * 6.4                                        # panel height in inches
    wa = 0.25
    wb = hin * 20 / 20 / 15                              # square window, 20 x 20 px
    wc = hin * 32 / 20 / 15                              # 32 x 20 px window
    xa, xb = 0.045, 0.045 + wa + 0.06
    xc = xb + wb + 0.035
    ax_a = fig.add_axes([xa, y0, wa, h])
    ax_b = fig.add_axes([xb, y0, wb, h])
    ax_c = fig.add_axes([xc, y0, wc, h])
    cap = dict(va="top", fontsize=8.6, color=INK2, linespacing=1.45)

    # (a) the cross-section is a blurred cylinder, not a Gaussian
    ax = ax_a
    r = 1.0
    d = np.linspace(-4, 4, 1601)
    chord = np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
    ax.plot(d, box_sharp(d, r), color=MUTED, lw=1.2, drawstyle="steps-mid", label="6 nested boxes, sharp")
    ax.plot(d, chord, color=INK, lw=1, label="cylinder chord √(1 − d²/r²)")
    blues = ["#86b6ef", "#2a78d6", "#104281"]
    for sgm, col in zip([0.25, 0.6, 1.2], blues):
        ax.plot(d, box_blur(d, r, sgm), color=col, lw=2, label=f"blurred, s = {sgm:g} r")
    p = box_blur(d, r, 0.25)
    area = np.trapezoid(p, d)
    sg = area / (p.max() * np.sqrt(2 * np.pi))
    ax.plot(d, p.max() * np.exp(-0.5 * (d / sg) ** 2), color=AQUA, lw=2,
            label="a Gaussian with the height and\narea of the s = 0.25 r curve")
    ax.set_xlabel("distance from the centreline d, in units of the radius r")
    ax.set_ylabel("optical density / a")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-3.2, 3.2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=2, fontsize=8.3,
              handlelength=1.6, columnspacing=1.2)
    ax.set_title("a · Cross-section: a blurred cylinder, not a Gaussian", loc="left")

    # (b) tight bend: each pixel is drawn from its nearest sample, in that sample's tangent frame
    ax = ax_b
    win = ((22, 42), (14, 34))
    frame(ax, win)
    spokes(ax, res_bend, win, lambda j: [MUTED] * len(j), alpha=0.7)
    centreline(ax, res_bend, win)
    P, J = res_bend["e_xy"], res_bend["e_samp"]
    k = np.argmin(np.linalg.norm(P - [40, 25], axis=1))
    p, q, t = P[k], res_bend["C"][J[k]], res_bend["T"][J[k]]
    u = float(np.dot(p - q, t))
    foot = q + u * t
    ax.plot(*np.stack([q - 3.2 * t, q + 3.2 * t]).T, color=BLUE, lw=2.2, zorder=7, solid_capstyle="round")
    ax.annotate("", xy=p, xytext=foot, arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2,
                                                       shrinkA=0, shrinkB=4), zorder=8)
    ax.scatter(*p, s=80, color=ORANGE, zorder=9, edgecolor=SURF, lw=1.5)
    ax.scatter(*q, s=60, color=BLUE, zorder=9, edgecolor=SURF, lw=1.5)
    ax.text(p[0] + 0.7, p[1] + 0.2, "pixel", color=INK, fontsize=9, zorder=10, va="center",
            bbox=dict(fc=SURF, ec="none", pad=1))
    mid = 0.5 * (p + foot)
    ax.text(mid[0] + 0.1, mid[1] - 0.9, "d", color=INK, fontsize=10, fontweight="bold", zorder=10,
            bbox=dict(fc=SURF, ec="none", pad=1))
    tp = q + 3.2 * t
    ax.text(tp[0] + 0.4, tp[1] - 0.4, "tangent at its\nnearest sample", color=INK, fontsize=8.5,
            zorder=10, va="bottom", bbox=dict(fc=SURF, ec="none", pad=1))
    ax.set_title("b · Tight bend: one sample per pixel", loc="left")
    fig.text(xb, y0 - 0.04,
             "Grey spokes join each pixel centre to the one centreline\n"
             "sample it is drawn from; inside the bend they converge\n"
             "on the centre of curvature. The pixel gets a·P(d)·taper,\n"
             "with d (orange) measured across that sample's tangent\n"
             "(blue). Black dots: samples every 0.75 px. Squares:\n"
             "B-spline control points (5 px apart here).", **cap)

    # (c) hairpin: pixels between the legs are drawn from one leg only
    ax = ax_c
    win = ((18, 50), (22, 42))
    frame(ax, win)
    half = res_hp["L"] / 2
    arc = res_hp["arc"]
    spokes(ax, res_hp, win, lambda j: [BLUE if arc[i] < half else ORANGE for i in j], alpha=0.8)
    centreline(ax, res_hp, win, ctrl=True)
    ax.plot([], [], color=BLUE, lw=2, label="pixel drawn from leg 1")
    ax.plot([], [], color=ORANGE, lw=2, label="pixel drawn from leg 2")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8.5, borderaxespad=0.2)
    ax.set_title("c · Hairpin: a seam down the middle", loc="left", pad=22)
    fig.text(xc, y0 - 0.04,
             "One edge gives each pixel exactly one sample, so a pixel between the legs takes only the\n"
             "nearer leg's blur tail. The other leg's tail, equally large on the midline, is never added,\n"
             "so the gap renders about half as dark as it should. Two separate edges would add; one\n"
             "edge folding back on itself does not. The same happens where a coil crosses itself.", **cap)

    fig.suptitle("What the renderer does per pixel", x=xa, ha="left", va="bottom", fontsize=13,
                 fontweight="bold", y=0.95)
    fig.savefig(os.path.join(OUT, "vesselmap_render_mechanism.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    H = W = 64
    results, by_name = [], {}
    for f in SCENES:
        xy, prm, prof, title = f()
        res = scene(xy, prm["r"], prm["s"], prm["a"], H, W)
        results.append((res, title, prof))
        by_name[f.__name__] = res
        e = res["net"].edges[res["eid"]]
        diff = 100 * (res["od_m"] - res["od_e"]) / res["peak"]
        print(f"{f.__name__:8s} r={res['r']:.2f} s={res['s']:.2f} a={res['a']:.3f} "
              f"ctrl spacing={e.info['spacing']:.0f}px n_ctrl={len(e.ctrl)} "
              f"err min/max={diff.min():+.1f}/{diff.max():+.1f}%")
    fig_comparison(results)
    fig_mechanism(by_name["bend"], by_name["hairpin"])
    print("written to", OUT)
