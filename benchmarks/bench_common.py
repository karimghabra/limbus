"""Shared helpers for the LIMBUS recording benchmarks.

Synthetic frames imitate a conjunctival microcirculation image from the
Basler acA1920-40um: an uneven illuminated sclera with fine texture, a
network of vessels from large venules down to 1-px capillaries, frame-to-
frame eye motion, and photon shot noise at a realistic gain. Frames are
12-bit values MSB-aligned into uint16, exactly what the app's pylon
converter produces (OutputBitAlignment_MsbAligned).
"""
import os
import subprocess
import threading
import time

import cv2
import numpy as np

W, H = 1928, 1208
FRAME_MB = W * H * 2 / 1e6
# the repository root: benchmarks/ lives inside it, next to camera_recorder.py
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vessels(rng, h, w):
    m = np.zeros((h, w), np.float32)
    # (count, thickness range px, steps, segment length px): venules -> capillaries
    for count, (t0, t1), steps, seg in ((25, (6, 14), 40, 25),
                                         (150, (2, 5), 30, 14),
                                         (900, (1, 2), 20, 7)):
        for _ in range(count):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            ang = rng.uniform(0, 2 * np.pi)
            pts = []
            for _ in range(steps):
                pts.append((x, y))
                ang += rng.normal(0, 0.35)
                x += seg * np.cos(ang)
                y += seg * np.sin(ang)
            depth = float(rng.uniform(0.25, 0.7))
            cv2.polylines(m, [np.array(pts, np.int32)], False, depth,
                          int(rng.integers(t0, t1 + 1)), cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def make_scene(seed=1):
    rng = np.random.default_rng(seed)
    h, w = H + 64, W + 64  # margin for motion
    illum = rng.random((h // 16, w // 16)).astype(np.float32)
    illum = cv2.resize(cv2.GaussianBlur(illum, (0, 0), 3), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    illum = 0.5 + (illum - illum.min()) / (np.ptp(illum) + 1e-6)
    tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32),
                           (0, 0), 1.5)
    tex /= tex.std()
    return 1800.0 * illum * (1 + 0.08 * tex) * (1 - _vessels(rng, h, w))


def make_frames(n, e_per_dn=8.0, flat=False, seed=1):
    """n distinct uint16 frames. e_per_dn: ~8 = gain 0 dB on the IMX249
    (shot noise ~15 DN at mid-grey), ~2 = 12 dB gain (~30 DN)."""
    rng = np.random.default_rng(seed + 100)
    scene = None if flat else make_scene(seed)
    frames, dx, dy = [], 32.0, 32.0
    for _ in range(n):
        if flat:
            f = np.full((H, W), 1800.0, np.float32)
        else:
            dx = float(np.clip(dx + rng.normal(0, 1.5), 4, 60))
            dy = float(np.clip(dy + rng.normal(0, 1.5), 4, 60))
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            f = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR)
            f += rng.standard_normal((H, W), dtype=np.float32) * np.sqrt(
                np.maximum(f, 0) / e_per_dn)
        frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16) << 4)
    return frames


def app_cmd(path, fps=40.0, threads="3", crf=20, out_fmt="yuv420p10le",
            encoder="libx264", extra=()):
    """The VideoWriter ffmpeg command from camera_recorder.py (16-bit mono,
    High quality preset), with threads/encoder parameterised."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray16le",
           "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-"]
    if threads is not None:
        cmd += ["-threads", str(threads)]
    if encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf)]
    else:
        cmd += ["-c:v", encoder, *extra]
    return cmd + ["-pix_fmt", out_fmt, path]


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,profile,pix_fmt,"
         "bits_per_raw_sample,width,height,nb_read_frames",
         "-of", "default=nw=1", path], capture_output=True, text=True).stdout
    return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)


class CpuSampler:
    """Samples total CPU %, busiest-core %, and cores used by a process."""

    def __init__(self, pid=None):
        # imported here, not at the top: the analysis tests use this module's
        # vessel generator and shouldn't need psutil installed
        global psutil
        import psutil
        self.pid = pid
        self.samples = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        proc = psutil.Process(self.pid) if self.pid else None
        if proc:
            proc.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)
        while not self._stop.wait(1.0):
            per = psutil.cpu_percent(None, percpu=True)
            try:
                cores = proc.cpu_percent(None) / 100 if proc else 0.0
            except psutil.Error:
                cores = 0.0
            freq = psutil.cpu_freq()
            self.samples.append((sum(per) / len(per), cores,
                                 freq.current if freq else 0))

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()

    def summary(self):
        s = self.samples[2:] or self.samples  # skip ramp-up
        if not s:
            return "no samples"
        tot = np.mean([x[0] for x in s])
        cores = np.mean([x[1] for x in s])
        return (f"system CPU {tot:.0f}% of {psutil.cpu_count()} threads, "
                f"encoder using {cores:.1f} threads-worth")
