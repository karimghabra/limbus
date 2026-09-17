"""Step 3 without a camera: drive the app's real VideoWriter class at a fixed
frame rate with a paced synthetic source, including the grab thread's
per-frame copy and the 30 fps preview downscale, and report
frames_written vs frames_dropped.

usage: python bench_record_sim.py [fps] [seconds] [threads] [e_per_dn]
"""
import os
import sys
import time

import cv2
import numpy as np

from bench_common import REPO, CpuSampler, make_frames, probe

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402

FPS = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
THREADS = sys.argv[3] if len(sys.argv) > 3 else None   # None = as the app has it
E_PER_DN = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

if THREADS is not None:
    # override the thread count in the app's ffmpeg command for this run
    _popen = cr.subprocess.Popen

    def popen(cmd, *a, **kw):
        if "-threads" in cmd:
            cmd = list(cmd)
            cmd[cmd.index("-threads") + 1] = THREADS
        return _popen(cmd, *a, **kw)
    cr.subprocess.Popen = popen

frames = make_frames(80, e_per_dn=E_PER_DN)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   f"sim_{FPS:.0f}fps_t{THREADS or 'app'}.mp4")
w = cr.VideoWriter(out, cr.np.uint16(0).size and 1928, 1208, FPS, False,
                   cr.QUALITY_PRESETS[0][1], bit16=True)

period = 1.0 / FPS
t0 = time.perf_counter()
last_preview = 0.0
i = 0
with CpuSampler(w.proc.pid) as cpu:
    while True:
        target = t0 + i * period
        now = time.perf_counter()
        if target - now > 0:
            time.sleep(max(0.0, target - now - 0.002))
            while time.perf_counter() < target:
                pass
        if time.perf_counter() - t0 >= SECS:
            break
        frame = np.ascontiguousarray(frames[i % len(frames)].copy())  # converter output copy
        w.write(frame)
        if time.perf_counter() - last_preview >= 1.0 / cr.PREVIEW_MAX_FPS:
            last_preview = time.perf_counter()
            pf = (frame >> 8).astype(np.uint8)
            cv2.resize(pf, (1280, int(pf.shape[0] * 1280 / pf.shape[1])),
                       interpolation=cv2.INTER_AREA)
        i += 1
    t_offered = time.perf_counter() - t0
    err = w.close()
p = probe(out)
print(f"offered {i} frames at {i / t_offered:.2f} fps for {t_offered:.1f}s "
      f"(ffmpeg threads={THREADS or '3 (app default)'}, e/DN={E_PER_DN}) -> "
      f"written {w.frames_written}, dropped {w.frames_dropped} "
      f"({100 * w.frames_dropped / max(i, 1):.1f}%) | {cpu.summary()} | "
      f"ffprobe {p.get('pix_fmt')} frames={p.get('nb_read_frames')}"
      + (f" | ffmpeg: {err}" if err else ""), flush=True)
