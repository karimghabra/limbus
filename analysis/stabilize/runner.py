"""The per-burst worker for parallel runs.

Lives in its own module rather than __main__.py: Windows starts worker
processes by spawning fresh interpreters that must import the worker
function by module path, and a function defined in the __main__ of a
`python -m` run can't be found that way.
"""
import os
import traceback

from .config import Params


def process_one(burst_path, out_root):
    # one OpenCV thread per process: parallelism comes from running bursts
    # side by side, and nested thread pools would just oversubscribe the CPU
    import cv2
    cv2.setNumThreads(1)
    from .pipeline import process_burst

    lines = []
    try:
        record = process_burst(burst_path, out_root, Params(), log=lines.append)
    except Exception as exc:                     # keep the rest of the run going
        record = {"status": "error", "skip_reason": f"{type(exc).__name__}: {exc}",
                  "burst": {"name": os.path.basename(burst_path), "path": burst_path}}
        lines.append(traceback.format_exc())
    return record, lines
