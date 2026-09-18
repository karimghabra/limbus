"""The frame-rate restamp: when frames arrive slower than declared, the saved
file must be re-timed so it plays at true speed. This used to work by
unwrapping the raw H.264 bitstream, which would break for FFV1/HEVC in
Matroska; it now rescales timestamps instead. Verify in both containers.

usage: python test_restamp.py
"""
import os
import subprocess
import sys
import tempfile
import time

from bench_common import REPO, make_frames

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402

DECLARED = 40.0     # what the writer is told
REAL = 20.0         # the rate frames actually arrive at
N = 60

frames = make_frames(20, e_per_dn=8.0)
fail = []
tmp = tempfile.mkdtemp(prefix="restamp_")

for label, quality in (("H.264 / mp4", cr.QUALITY_PRESETS[0][1]),
                       ("HEVC 12-bit / mkv", cr.QUALITY_PRESETS[3][1]),
                       ("FFV1 lossless / mkv", cr.QUALITY_PRESETS[4][1])):
    path = os.path.join(tmp, "clip" + cr.video_extension(quality))
    w = cr.VideoWriter(path, 1928, 1208, DECLARED, False, quality, bit16=True)
    period = 1.0 / REAL
    t0 = time.perf_counter()
    for i in range(N):
        while time.perf_counter() - t0 < i * period:
            time.sleep(0.001)
        w.write(frames[i % len(frames)])
    w.close()

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,nb_read_frames:format=duration",
         "-of", "default=nw=1", path], capture_output=True, text=True).stdout
    info = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    dur = float(info.get("duration", 0) or 0)
    n = int(info.get("nb_read_frames", 0) or 0)
    plays_at = n / dur if dur else 0
    print(f"{label}: wrote {w.frames_written} frames at {REAL} fps but "
          f"declared {DECLARED} -> file has {n} frames, {dur:.2f}s, "
          f"plays at {plays_at:.2f} fps")
    if abs(plays_at - REAL) / REAL > 0.10:
        fail.append(f"{label}: plays at {plays_at:.2f} fps, expected ~{REAL}")
    os.remove(path)

os.rmdir(tmp)
print("\nRESULT:", "FAIL " + "; ".join(fail) if fail else "PASS")
sys.exit(1 if fail else 0)
