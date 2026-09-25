"""A video of a registered burst with the vessel map and its measured flow.

Top panel: the registered frames with the vessel centrelines, coloured by
measured red-cell speed (grey where flow is unknown).  Bottom panel: the
same frames minus their running temporal mean ("flicker"), which shows what
moves: red-cell aggregates and plasma gaps travelling along the vessels.

On both panels, tracer dots move along every vessel with a measured flow,
at the measured velocity (in lanes across the lumen at the measured
velocity profile where there is one).  The dots are not detected cells:
they show the measurement.  Where it is right they travel with the flicker,
where it is wrong they visibly drift against it.
"""
from __future__ import annotations

import math
import os
import subprocess
from collections import deque

import cv2
import numpy as np

from .draw import _colormap
from .network import VesselNetwork


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class _Tracers:
    """Dots advected along each vessel at its measured velocity."""

    def __init__(self, net: VesselNetwork, spacing=45.0):
        self.items = []
        rng = np.random.default_rng(0)
        for eid, e in net.edges.items():
            fl = e.info.get("flow", {})
            if not fl.get("reliable"):
                continue
            v = float(fl["v_px_per_frame"])         # edges are oriented along the flow
            smp = net.sample(eid, 1.0)
            xy, arc = smp["xy"], smp["s_arc"]
            if arc[-1] < 12:
                continue
            t = smp["tan"] / np.maximum(np.linalg.norm(smp["tan"], axis=1, keepdims=True), 1e-9)
            N = np.stack([-t[:, 1], t[:, 0]], 1)
            prof = [p for p in fl.get("profile", []) if p.get("v") is not None]
            if len(prof) >= 3:
                # centre lane and the two outermost measured lanes: shows the profile
                pr = sorted(prof, key=lambda p: p["offset_px"])
                c = min(pr, key=lambda p: abs(p["offset_px"]))
                lanes = [(p["offset_px"], float(p["v"])) for p in {id(q): q for q in (pr[0], c, pr[-1])}.values()]
            else:
                lanes = [(0.0, v)]
            for off, vl in lanes:
                n = max(1, int(arc[-1] // spacing))
                phase = rng.uniform(0, spacing)
                self.items.append(dict(xy=xy, N=N, arc=arc, off=off, v=vl,
                                       s0=phase + spacing * np.arange(n), L=float(arc[-1])))

    def positions(self, k):
        """Dot positions at frame k (from the start of the video)."""
        out = []
        for it in self.items:
            s = np.mod(it["s0"] + it["v"] * k, it["L"])
            x = np.interp(s, it["arc"], it["xy"][:, 0]) + it["off"] * np.interp(s, it["arc"], it["N"][:, 0])
            y = np.interp(s, it["arc"], it["xy"][:, 1]) + it["off"] * np.interp(s, it["arc"], it["N"][:, 1])
            out.append(np.stack([x, y], 1))
        return np.concatenate(out) if out else np.zeros((0, 2))


def _centrelines(net: VesselNetwork, shape, scale, fps):
    """BGR layer + alpha of the centrelines coloured by speed."""
    H, W = int(shape[0] * scale), int(shape[1] * scale)
    lay = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.uint8)
    sp = [e.info.get("flow", {}).get("speed_px_per_s") for e in net.edges.values()]
    sp = [x for x in sp if x]
    lo, hi = (np.percentile(sp, 5), np.percentile(sp, 95)) if sp else (50.0, 500.0)
    for eid, e in net.edges.items():
        xy = np.round(net.sample(eid, 1.0)["xy"] * scale * 8).astype(np.int32)
        spd = e.info.get("flow", {}).get("speed_px_per_s")
        if spd:
            c = _colormap([math.log(max(spd, 1e-3))], math.log(lo), math.log(max(hi, lo * 1.01)))[0]
            col, th = tuple(int(x) for x in c), 1
        else:
            col, th = (120, 120, 120), 1
        cv2.polylines(lay, [xy], False, col, th, cv2.LINE_AA, shift=3)
        cv2.polylines(alpha, [xy], False, 255, th, cv2.LINE_AA, shift=3)
    return lay, alpha, (lo, hi)


def _vessel_mask(net: VesselNetwork, shape, pad=2.0, rmax=7.0):
    m = np.zeros(shape, np.uint8)
    for eid in net.edges:
        smp = net.sample(eid, 1.0)
        rad = min(float(np.median(smp["r"])), rmax) + pad
        cv2.polylines(m, [np.round(smp["xy"] * 8).astype(np.int32)], False, 255,
                      max(1, int(round(2 * rad)) + 1), cv2.LINE_AA, shift=3)
    return cv2.GaussianBlur(m, (0, 0), 1.5).astype(np.float32) / 255.0


def _colorbar(lo, hi, w=220, h=10):
    t = np.linspace(math.log(lo), math.log(hi), w)
    bar = _colormap(t, t[0], t[-1])[None].repeat(h, 0)
    return bar.astype(np.uint8)


def render(net: VesselNetwork, frames, reg, idx, fps, out_path, ref=None, scale=0.75,
           out_fps=30, window=15, crop=None, crf=26, flicker_gain=0.28, label=""):
    """Write an H.264 MP4.  frames: (T, H, W) raw burst; reg: limbusflow
    Registration (frame -> reference); idx: consecutive frame indices; ref:
    reference image (for the display stretch); crop: (x0, y0, x1, y1) in
    reference px to render a zoomed region instead of the whole frame."""
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "analysis", "limbusflow"))
    from limbusflow import register as rg

    shape = tuple(reg.shape)
    x0, y0, x1, y1 = crop or (0, 0, shape[1], shape[0])
    sub = (slice(y0, y1), slice(x0, x1))
    Wp, Hp = int((x1 - x0) * scale) // 2 * 2, int((y1 - y0) * scale) // 2 * 2

    def to_panel(a):
        return cv2.resize(a[sub], (Wp, Hp), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    full_lay, full_alpha, (lo, hi) = _centrelines(net, shape, 1.0, fps)
    lay = to_panel(full_lay)
    alpha = to_panel(full_alpha).astype(np.float32)[..., None] / 255.0 * 0.9
    vmask = to_panel(_vessel_mask(net, shape))[..., None]
    tr = _Tracers(net)

    def warped(i):
        f = np.asarray(frames[i], np.float32)
        w = rg.warp(f, reg.M[i], shape)
        return w / np.nanmedian(w)

    # display stretch from a few frames
    probe = np.nanmean([warped(i) for i in idx[:: max(1, len(idx) // 12)]], 0)
    a, b = np.nanpercentile(probe, [0.5, 99.5])

    head = 44
    Hout = 2 * Hp + head + 6
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{Wp}x{Hout}", "-r", str(out_fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    idx = list(idx)
    n = len(idx)
    half = window // 2
    buf, start = deque(), 0
    run_sum = np.zeros(shape, np.float32)
    run_cnt = np.zeros(shape, np.float32)
    run_sq = np.zeros(shape, np.float32)

    def push():
        w = warped(idx[start + len(buf)])
        buf.append(w)
        run_sum.__iadd__(np.nan_to_num(w))
        run_sq.__iadd__(np.nan_to_num(w) ** 2)
        run_cnt.__iadd__(np.isfinite(w))

    cb = _colorbar(lo, hi)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, i in enumerate(idx):
        # the running mean covers frames k - half .. k + half
        while start + len(buf) - 1 < min(n - 1, k + half):
            push()
        while start < k - half:
            old = buf.popleft()
            run_sum -= np.nan_to_num(old)
            run_sq -= np.nan_to_num(old) ** 2
            run_cnt -= np.isfinite(old)
            start += 1
        cur = buf[k - start]
        mean = run_sum / np.maximum(run_cnt, 1)
        sd = np.sqrt(np.maximum(run_sq / np.maximum(run_cnt, 1) - mean ** 2, 1e-8))
        sd = cv2.GaussianBlur(sd, (0, 0), 3.0)
        # top: registered frame
        g = np.clip((np.nan_to_num(cur, nan=a) - a) / (b - a), 0, 1)
        top = cv2.cvtColor((to_panel(g) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
        top = top * (1 - alpha) + lay.astype(np.float32) * alpha
        # bottom: flicker, vessels bright, tissue dimmed
        # flicker as a local z-score: each pixel's change relative to its own
        # running variability, lightly smoothed (aggregates span several px)
        fl = cv2.GaussianBlur(np.nan_to_num((cur - mean) / sd), (0, 0), 1.2)
        fl = to_panel(fl)
        fl = np.clip(0.5 + flicker_gain * fl, 0, 1)
        bot = (fl * 255)[..., None].repeat(3, 2).astype(np.float32)
        bot = bot * (0.25 + 0.75 * vmask)
        # tracers
        P = (tr.positions(k) - [x0, y0]) * scale
        inside = (P[:, 0] >= 0) & (P[:, 1] >= 0) & (P[:, 0] < Wp) & (P[:, 1] < Hp)
        top8, bot8 = top.clip(0, 255).astype(np.uint8), bot.clip(0, 255).astype(np.uint8)
        rr = 1.6 * max(scale, 1.0)
        for (px, py) in P[inside]:
            c = (int(round(px * 8)), int(round(py * 8)))
            for img in (top8, bot8):
                cv2.circle(img, c, int(rr * 8 + 8), (20, 20, 20), -1, cv2.LINE_AA, shift=3)
                cv2.circle(img, c, int(rr * 8), (80, 230, 255), -1, cv2.LINE_AA, shift=3)
        # header
        hdr = np.full((head, Wp, 3), 24, np.uint8)
        t_s = (i - idx[0]) / fps
        txt = (f"{label}frame {i}  t = {t_s:5.2f} s   played {fps / out_fps:.1f}x slower   "
               f"dots move at the measured velocity")
        cv2.putText(hdr, txt, (10, 18), font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(hdr, "top: registered video, centrelines by speed   bottom: flicker (change from the running mean, z-score)",
                    (10, 37), font, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        if Wp > cb.shape[1] + 200:
            xb = Wp - cb.shape[1] - 12
            hdr[8:8 + cb.shape[0], xb:xb + cb.shape[1]] = cb
            cv2.putText(hdr, f"{lo:.0f}", (xb - 4, 36), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(hdr, f"{hi:.0f} px/s", (xb + cb.shape[1] - 60, 36), font, 0.4, (200, 200, 200), 1,
                        cv2.LINE_AA)
        gap = np.full((6, Wp, 3), 24, np.uint8)
        frame = np.vstack([hdr, top8, gap, bot8])
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_path
