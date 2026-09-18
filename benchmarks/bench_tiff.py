"""TIFF burst capture: can this machine save every 12-bit frame at 40 fps as
individual uncompressed 16-bit TIFF files?

Same design as the app's VideoWriter (bounded ~240 MB queue, frames that
don't fit are dropped and counted), with N writer threads. Runs long enough
(default 60 s = ~11 GB) that the Windows file cache can't hide a slow disk.

usage: python bench_tiff.py out_dir [fps] [seconds] [writer_threads]
       fps=0 -> unpaced: find the ceiling
"""
import os
import queue
import shutil
import sys
import threading
import time

import tifffile

from bench_common import FRAME_MB, CpuSampler, make_frames

OUT = sys.argv[1]
FPS = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
SECS = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
NTHREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 2

os.makedirs(OUT, exist_ok=True)
frames = make_frames(40)
q = queue.Queue(maxsize=max(30, int(240e6 // (FRAME_MB * 1e6))))
written = dropped = 0
lock = threading.Lock()


def drain():
    global written
    while True:
        item = q.get()
        if item is None:
            return
        idx, frame = item
        tifffile.imwrite(os.path.join(OUT, f"frame_{idx:06d}.tif"), frame,
                         photometric="minisblack")
        with lock:
            written += 1


threads = [threading.Thread(target=drain, daemon=True) for _ in range(NTHREADS)]
for t in threads:
    t.start()

t0 = time.perf_counter()
i = 0
with CpuSampler() as cpu:
    while time.perf_counter() - t0 < SECS:
        if FPS > 0:
            target = t0 + i / FPS
            d = target - time.perf_counter()
            if d > 0:
                time.sleep(max(0.0, d - 0.002))
                while time.perf_counter() < target:
                    pass
            try:
                q.put_nowait((i, frames[i % len(frames)]))
            except queue.Full:
                dropped += 1
        else:
            q.put((i, frames[i % len(frames)]))
        i += 1
    t_offer = time.perf_counter() - t0
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
t_all = time.perf_counter() - t0
mode = f"paced {FPS:.0f} fps" if FPS > 0 else "unpaced"
print(f"[TIFF {mode}, {NTHREADS} writer threads] offered {i} in {t_offer:.1f}s, "
      f"written {written}, dropped {dropped} ({100 * dropped / max(i, 1):.1f}%), "
      f"drain finished {t_all - t_offer:.1f}s after capture stopped | "
      f"write rate {written / t_all:.1f} fps = {written * FRAME_MB / t_all:.0f} MB/s | "
      f"{cpu.summary()}", flush=True)
shutil.rmtree(OUT, ignore_errors=True)  # benchmark output only
