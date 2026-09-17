"""Step 2: encoder throughput ceiling.

Feeds 1928x1208 16-bit frames as fast as ffmpeg will take them, through the
same command the app uses, for a sustained period (a 15 W laptop CPU slows
down after its short turbo window, so short bursts overstate it).

usage: python bench_encoder.py [seconds] [out_dir]
"""
import os
import subprocess
import sys
import time

from bench_common import FRAME_MB, CpuSampler, app_cmd, make_frames, probe

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))


def run(label, frames, secs=SECS, **cmd_kw):
    path = os.path.join(OUT, f"enc_{label}.mp4")
    proc = subprocess.Popen(app_cmd(path, **cmd_kw), stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    n = 0
    t0 = time.perf_counter()
    with CpuSampler(proc.pid) as cpu:
        try:
            while time.perf_counter() - t0 < secs:
                proc.stdin.write(frames[n % len(frames)].tobytes())
                n += 1
        except (BrokenPipeError, OSError):
            pass
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.wait()
    dt = time.perf_counter() - t0
    fps = n / dt
    p = probe(path) if os.path.exists(path) else {}
    size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"[{label}] {n} frames in {dt:.1f}s -> {fps:.1f} fps, "
          f"{fps * FRAME_MB:.0f} MB/s raw | {cpu.summary()} | "
          f"out {size:.0f} MB ({size / dt:.1f} MB/s) | "
          f"ffprobe: {p.get('codec_name')} {p.get('profile')} "
          f"{p.get('pix_fmt')} bits={p.get('bits_per_raw_sample')} "
          f"frames={p.get('nb_read_frames')}"
          + (f" | ffmpeg stderr: {err[:300]}" if err else ""), flush=True)
    return fps


if __name__ == "__main__":
    print("generating frames...", flush=True)
    real = make_frames(80, e_per_dn=8.0)            # gain 0 dB
    noisy = make_frames(80, e_per_dn=2.0, seed=2)   # ~12 dB gain
    flat = make_frames(4, flat=True)

    # does libx264 here really do 12-bit? (short run, just for ffprobe)
    run("request_12bit", real, secs=4, out_fmt="yuv420p12le")

    run("app_threads3_gain0", real, threads="3")
    run("auto_threads_gain0", real, threads="0")
    run("app_threads3_gain12dB", noisy, threads="3")
    run("auto_threads_gain12dB", noisy, threads="0")
    run("app_threads3_FLAT", flat, threads="3", secs=15)
