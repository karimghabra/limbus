"""Throughput of the 12-bit-capable video encoders, on realistic noisy
content, so the app's presets can carry honest frame-rate guidance.

usage: python bench_codecs.py [seconds] [out_dir]
"""
import os
import subprocess
import sys
import time

from bench_common import FRAME_MB, CpuSampler, W, H, make_frames, probe

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

CODECS = (
    ("x264_10bit_mp4", "mp4", ["-c:v", "libx264", "-preset", "ultrafast",
                               "-crf", "20"], "yuv420p10le"),
    ("x265_12bit_mkv", "mkv", ["-c:v", "libx265", "-preset", "ultrafast",
                               "-crf", "20", "-x265-params", "log-level=none"],
     "gray12le"),
    ("ffv1_lossless_mkv", "mkv", ["-c:v", "ffv1", "-level", "3", "-g", "1",
                                  "-slices", "4", "-slicecrc", "1"],
     "gray16le"),
)


def run(label, ext, codec_args, out_fmt, frames, threads="3"):
    path = os.path.join(OUT, f"codec_{label}.{ext}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{W}x{H}",
           "-r", "40", "-i", "-", "-threads", threads, *codec_args,
           "-pix_fmt", out_fmt, path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    n = 0
    t0 = time.perf_counter()
    with CpuSampler(proc.pid) as cpu:
        try:
            while time.perf_counter() - t0 < SECS:
                proc.stdin.write(frames[n % len(frames)].tobytes())
                n += 1
        except (BrokenPipeError, OSError):
            pass
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
    dt = time.perf_counter() - t0
    p = probe(path)
    size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"[{label} threads={threads}] {n / dt:6.1f} fps, "
          f"{n / dt * FRAME_MB:4.0f} MB/s raw | file {size / dt:6.1f} MB/s "
          f"({size / max(n, 1):.2f} MB/frame) | {cpu.summary()} | "
          f"ffprobe {p.get('codec_name')} {p.get('pix_fmt')} "
          f"bits={p.get('bits_per_raw_sample')}"
          + (f" | {err[:200]}" if err else ""), flush=True)
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    print("generating realistic noisy frames...", flush=True)
    frames = make_frames(60, e_per_dn=2.0, seed=3)   # high gain = worst case
    for label, ext, args, fmt in CODECS:
        run(label, ext, args, fmt, frames, "3")
    print("--- with all cores ---", flush=True)
    for label, ext, args, fmt in CODECS[1:]:
        run(label, ext, args, fmt, frames, "0")
