"""Validation: synthetic ground truth + validation videos rendered from the real burst.

Ground-truth phantoms
---------------------
* diameter phantom  - straight vessels of known width (box or cylinder
  profile), blurred and noisy -> how well do FWHM / model fit recover d?
* curvature phantom - a rasterised sine wave y = A sin(2 pi x / lambda), whose
  curvature is known analytically, kappa = y'' / (1 + y'^2)^(3/2).
* flow phantom      - the *real* reference image and *real* vessel centre-lines,
  with a random red-cell pattern that moves along each vessel at a known
  velocity. Frames get motion blur (exposure integration), random eye jitter
  and sensor noise, then go through the same registration -> kymograph ->
  velocity chain as the real data.

Videos
------
* overlay video - registered frames with vessel outlines and *tracer
  particles* advected at the measured velocity v(t). If the measurement is
  right, the tracers ride along with the visible red-cell aggregates.
* kymograph video - one vessel: the registered crop on the left and its
  kymograph being written line by line on the right.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import morphometry as mo
from . import spline as spl


# ============================================================================ diameter
def vessel_profile(t, d, model="box", blur=1.5, depth=0.35):
    """Transmission profile of a vessel of diameter d (1 = background)."""
    tf = np.arange(t[0] - 10 * blur - d, t[-1] + 10 * blur + d, 0.05)
    if model == "box":
        a = (np.abs(tf) <= d / 2).astype(float)
    else:  # cylinder: absorption ~ chord length 2 sqrt(r^2 - t^2)
        a = np.sqrt(np.clip(1 - (2 * tf / d) ** 2, 0, None))
    a = ndi.gaussian_filter1d(a, blur / 0.05) if blur > 0 else a
    return np.interp(t, tf, 1 - depth * a)


def diameter_phantom_test(widths=(3, 4, 6, 8, 10, 14, 18, 24, 32), model="box", blur=1.5, noise=0.01, n_rep=20,
                          seed=0):
    """Straight synthetic vessel: returns rows (true d, FWHM mean/sd, fit mean/sd)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in widths:
        L = 80
        path = np.stack([np.linspace(10, 10 + L, L + 1), np.full(L + 1, 60.0)], 1)
        sp = spl.fit_spline(path)
        hw = max(8.0, 1.6 * d)
        yy = np.arange(120)[:, None] - 60.0
        fw, ft = [], []
        for _ in range(n_rep):
            img = np.repeat(vessel_profile(yy[:, 0], d, model, blur)[:, None], 120, 1)
            img = img + rng.normal(0, noise, img.shape)
            r = mo.measure_diameter(img, sp, d_guess=d, avg=3, stride=4)
            m = slice(10, -10)
            fw.append(np.nanmedian(r["d_fwhm"][m])); ft.append(np.nanmedian(r["d_fit"][m]))
        rows.append(dict(true_d=d, fwhm=np.mean(fw), fwhm_sd=np.std(fw), fit=np.mean(ft), fit_sd=np.std(ft)))
    return rows


# ============================================================================ curvature
def sine_curve_test(A=20.0, lam=160.0, n_periods=2, sigma=0.8):
    x = np.arange(0, n_periods * lam + 1)
    y = 60 + A * np.sin(2 * np.pi * x / lam)
    # rasterise (what a skeleton gives): integer pixels, 8-connected
    pix = np.stack([np.round(x), np.round(y)], 1)
    fine = []
    for a, b in zip(pix[:-1], pix[1:]):
        n = int(max(abs(b - a))) + 1
        fine.append(np.round(np.linspace(a, b, n + 1))[:-1])
    pix = np.vstack(fine + [pix[-1:]])
    sp = spl.fit_spline(pix, sigma=sigma)
    # analytic curvature at the spline's x positions
    xs = sp.xy[:, 0]
    k = 2 * np.pi / lam
    y1 = A * k * np.cos(k * xs); y2 = -A * k**2 * np.sin(k * xs)
    kappa_true = y2 / (1 + y1**2) ** 1.5
    # analytic length via dense integration
    xd = np.linspace(0, n_periods * lam, 20001)
    L_true = np.trapezoid(np.sqrt(1 + (A * k * np.cos(k * xd)) ** 2), xd)
    return dict(spline=sp, pixels=pix, kappa_true=kappa_true, L_true=L_true, DM_true=L_true / (n_periods * lam))


# ============================================================================ flow phantom
def _vessel_coords(net, vids, shape):
    """For every pixel inside the chosen vessels: (vessel index, s, |w|/(d/2))."""
    H, W = shape
    lab = np.full(shape, -1, np.int32)
    S = np.zeros(shape, np.float32); R = np.full(shape, 9.0, np.float32)
    for k, vid in enumerate(vids):
        ves = net.vessels[vid]
        m = mo.rasterize(ves.polygon, shape)
        ys, xs = np.nonzero(m & (lab < 0))
        tree = cKDTree(ves.spline.xy)
        dist, j = tree.query(np.stack([xs, ys], 1))
        lab[ys, xs] = k
        S[ys, xs] = ves.spline.s[j]
        R[ys, xs] = dist / (ves.d_smooth[j] / 2 + 1e-6)
    return lab, S, R


def make_flow_phantom(net, vids, v_true, n_frames=150, exposure_frac=0.95, n_sub=5, contrast=0.12,
                      grain=6.0, jitter_px=6.0, jitter_rot_deg=0.3, noise_rel=0.01, seed=1):
    """Synthetic burst with known velocities (px/frame, sign relative to spline direction).

    Returns frames (T, H, W) uint16-like floats, the true frame->reference
    transforms, and the per-vessel ground truth.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(net.ref, np.float64)
    H, W = base.shape
    lab, S, R = _vessel_coords(net, vids, base.shape)
    inside = lab >= 0
    ys, xs = np.nonzero(inside)
    ks, ss, rr = lab[inside], S[inside], R[inside]
    prof = np.clip(1 - rr**2, 0, 1)                 # stronger modulation near the axis
    # one random 1-D "aggregate" pattern per vessel, long enough for the whole run
    pats = []
    for k, vid in enumerate(vids):
        Lp = int(net.vessels[vid].length + abs(v_true[k]) * n_frames + 200)
        p = ndi.gaussian_filter1d(rng.normal(size=Lp), grain)
        pats.append((p - p.mean()) / (p.std() + 1e-9))
    frames = np.empty((n_frames, H, W), np.float32)
    M_true = np.zeros((n_frames, 2, 3))
    sels = [np.flatnonzero(ks == k) for k in range(len(vids))]
    for t in range(n_frames):
        img = base.copy()
        mod = np.zeros(len(ks))
        for j in range(n_sub):            # integrate over the exposure -> motion blur
            tt = t + exposure_frac * (j / max(1, n_sub - 1) - 0.5)
            for k, sel in enumerate(sels):
                pos = ss[sel] - v_true[k] * tt + abs(v_true[k]) * n_frames + 100
                mod[sel] += np.interp(pos, np.arange(len(pats[k])), pats[k])
        mod /= n_sub
        img[ys, xs] *= 1 - contrast * prof * mod
        # eye jitter: the camera sees the reference moved by a random similarity
        dx, dy = rng.normal(0, jitter_px, 2); th = np.radians(rng.normal(0, jitter_rot_deg))
        c, s_ = np.cos(th), np.sin(th)
        cx, cy = W / 2, H / 2
        # frame -> reference transform (what registration must recover)
        Mfr = np.array([[c, -s_, cx - c * cx + s_ * cy + dx], [s_, c, cy - s_ * cx - c * cy + dy]])
        M_true[t] = Mfr
        Mrf = cv2.invertAffineTransform(Mfr)
        f = cv2.warpAffine(img.astype(np.float32), Mrf, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        f = f + rng.normal(0, noise_rel * np.median(f), f.shape)
        frames[t] = np.clip(f, 0, 4095)
    return frames, M_true


# ============================================================================ videos
def _to8(a, lo, hi):
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def _writer(path, fps):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8,
                              ffmpeg_log_level="error", pixelformat="yuv420p")


def tracer_positions(net, vid, n_frames, spacing=45.0, phase=0.0):
    """Arc-length positions (px) over time of tracers moving at the measured v(t)."""
    ves = net.vessels[vid]
    v_med = ves.vel["v_time"]
    t_c = ves.vel["t_s"] * net.fps
    vt = np.interp(np.arange(n_frames), t_c, np.where(np.isfinite(ves.vel["v_t"]), ves.vel["v_t"], v_med))
    disp = np.concatenate([[0], np.cumsum(vt[:-1])])
    s0 = np.arange(phase, ves.length, spacing)
    return (s0[None, :] + disp[:, None]) % ves.length


def render_overlay_video(frames, reg, idx, net, path, fps_out=24, scale=0.6, vids=None, show_raw=True,
                         max_frames=None, tracer_spacing=45.0):
    """Raw frame (top) and registered frame with outlines + tracers (bottom)."""
    idx = list(idx)[:max_frames] if max_frames else list(idx)
    H, W = reg.shape
    vids = vids or [k for k, v in net.vessels.items() if v.vel.get("reliable")]
    lo, hi = np.percentile(np.asarray(frames[idx[len(idx) // 2]]), [0.5, 99.5])
    tr = {vid: tracer_positions(net, vid, len(idx), tracer_spacing, phase=(int(vid[1:]) * 7) % 30) for vid in vids}
    wr = _writer(path, fps_out)
    sz = (int(W * scale) // 8 * 8, int(H * scale) // 8 * 8)
    for n, i in enumerate(idx):
        f = np.asarray(frames[i], np.float32)
        reg_f = cv2.warpAffine(f, reg.M[i], (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        g = cv2.cvtColor(_to8(reg_f, lo, hi), cv2.COLOR_GRAY2BGR)
        for vid, ves in net.vessels.items():
            pts = np.round(ves.polygon).astype(np.int32)
            col = (0, 200, 255) if vid in vids else (120, 120, 120)
            cv2.polylines(g, [pts], True, col, 1, cv2.LINE_AA)
        for vid in vids:
            ves = net.vessels[vid]
            for s in tr[vid][n]:
                x, y = ves.spline.at(s)
                cv2.circle(g, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(g, f"registered + tracers @ measured v(t)   frame {i}  t={n / net.fps:5.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        bottom = cv2.resize(g, sz, interpolation=cv2.INTER_AREA)
        if show_raw:
            r = cv2.cvtColor(_to8(f, lo, hi), cv2.COLOR_GRAY2BGR)
            cv2.putText(r, f"raw camera frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
            top = cv2.resize(r, sz, interpolation=cv2.INTER_AREA)
            out = np.vstack([top, bottom])
        else:
            out = bottom
        wr.append_data(out[..., ::-1])
    wr.close()
    return path


def render_kymograph_video(frames, reg, idx, net, vid, path, fps_out=20, pad=30, crop_scale=2.0,
                           max_frames=None):
    """Left: registered crop around one vessel with tracers; right: its (pre-processed) kymograph growing row by row."""
    idx = list(idx)[:max_frames] if max_frames else list(idx)
    ves = net.vessels[vid]
    H, W = reg.shape
    x0, y0 = np.maximum(ves.spline.xy.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(ves.spline.xy.max(0) + pad, [W - 1, H - 1]).astype(int)
    K = ves.vel["K2"]                    # moving component: static anatomy removed (rows = bins of k frames)
    kb = ves.vel.get("scale", 1)
    kl, kh = np.nanpercentile(K, [1, 99])
    K8 = _to8(K, kl, kh)
    lo, hi = np.percentile(np.asarray(frames[idx[0]]), [0.5, 99.5])
    tr = tracer_positions(net, vid, len(idx), 40.0)
    ch = int((y1 - y0) * crop_scale); cw = int((x1 - x0) * crop_scale)
    kh_px = max(ch, 300)
    kw = int(np.clip(K.shape[1] * 1.5, 300, 700))
    Hc = max(ch, kh_px) // 8 * 8 + 8
    wr = _writer(path, fps_out)
    for n, i in enumerate(idx):
        f = np.asarray(frames[i], np.float32)
        reg_f = cv2.warpAffine(f, reg.M[i], (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        g = cv2.cvtColor(_to8(reg_f, lo, hi), cv2.COLOR_GRAY2BGR)
        cv2.polylines(g, [np.round(ves.polygon).astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
        for s in tr[n]:
            x, y = ves.spline.at(s)
            cv2.circle(g, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, cv2.LINE_AA)
        x, y = ves.spline.xy[0]
        cv2.putText(g, "s=0", (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        crop = cv2.resize(g[y0:y1, x0:x1], (cw, ch), interpolation=cv2.INTER_CUBIC)
        kym = np.full((len(idx), K.shape[1]), 0, np.uint8)
        kym = np.full((len(K), K.shape[1]), 0, np.uint8)
        kym[:n // kb + 1] = K8[:n // kb + 1]
        kym = cv2.cvtColor(cv2.resize(kym, (kw, kh_px), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        yline = int((n + 1) / len(idx) * kh_px)
        cv2.line(kym, (0, yline), (kw - 1, yline), (0, 0, 255), 1)
        cv2.putText(kym, "moving component K2(t,s): s ->, t down", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        canvas = np.zeros((Hc, cw + kw + 10, 3), np.uint8)
        canvas[:ch, :cw] = crop
        canvas[:kh_px, cw + 10:cw + 10 + kw] = kym
        cv2.putText(canvas, f"{vid}  t={n / net.fps:5.2f}s", (5, Hc - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        canvas = canvas[:, :canvas.shape[1] // 8 * 8]
        wr.append_data(canvas[..., ::-1])
    wr.close()
    return path


# ============================================================================ negative control
def negative_control(net, frames, reg, idx, offset=8.0, min_len=40, perfused_pct=75):
    """Kymographs along lines parallel to each vessel, `offset` px beyond its
    wall, keeping only stretches that avoid every detected vessel (dilated)
    and every perfused pixel of the flicker map. True velocity there is zero,
    so any 'reliable' velocity is a false positive.
    Returns a DataFrame (vessel, side, length, reliable, v, scale, quality)
    and the control centre-lines."""
    import pandas as pd
    from . import velocity as ve
    bad = ndi.binary_dilation(net.mask, iterations=3)
    if getattr(net, "fl", None) is not None:
        fl = net.fl
        bad |= ndi.binary_dilation(fl > np.percentile(fl[fl > 0], perfused_pct), iterations=2)
    H, W = bad.shape
    lines, meta = [], []
    for vid, v in net.vessels.items():
        d = v.d_median if np.isfinite(v.d_median) else v.d_guess
        for side in (1, -1):
            xy = v.spline.xy + side * (d / 2 + offset) * v.spline.N
            xi = np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1)
            clean = ~bad[yi, xi] & net.valid_e[yi, xi]
            best, s0 = (0, 0), None
            for i, c in enumerate(np.r_[clean, False]):
                if c and s0 is None:
                    s0 = i
                if not c and s0 is not None:
                    if i - s0 > best[1] - best[0]:
                        best = (s0, i)
                    s0 = None
            if best[1] - best[0] < min_len:
                continue
            sp2 = spl.fit_spline(xy[best[0]:best[1]], sigma=0.3)
            lines.append(sp2); meta.append((vid, side, sp2.length, max(d, 4)))
    pts = [ve.lumen_points(sp2, m[3], net.params.lumen_frac) for sp2, m in zip(lines, meta)]
    Ks = ve.kymographs(frames, reg, idx, pts, progress=False)
    rows = []
    for K, (vid, side, L, d) in zip(Ks, meta):
        r = net.velocity_for(K)
        rows.append(dict(vessel=vid, side=side, length_px=L, reliable=r["reliable"], v_px_f=r.get("v_time", np.nan),
                         scale=r.get("scale", 1), quality=r["quality"]))
    return pd.DataFrame(rows), lines
