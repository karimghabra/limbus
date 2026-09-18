"""FFV1 is lossless but slow by default: it parallelises across slices, so
slice count and thread count decide whether it can keep up with the camera.

usage: python bench_ffv1_tune.py [seconds]
"""
import os
import sys

from bench_codecs import run
from bench_common import make_frames

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
import bench_codecs  # noqa: E402
bench_codecs.SECS = SECS

BASE = ["-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1"]

if __name__ == "__main__":
    frames = make_frames(60, e_per_dn=2.0, seed=3)
    for slices, threads in ((4, "3"), (4, "0"), (12, "0"), (24, "0"),
                            (30, "0"), (24, "6")):
        run(f"ffv1_s{slices}_t{threads}", "mkv",
            BASE + ["-slices", str(slices)], "gray16le", frames, threads)
    # coder 0 (Golomb-Rice) trades compression for speed against the
    # default range coder
    run("ffv1_s24_t0_coder0", "mkv",
        BASE + ["-slices", "24", "-coder", "0"], "gray16le", frames, "0")
