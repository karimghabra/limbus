"""The per-burst worker for parallel runs.

Lives in its own module rather than __main__.py: Windows starts worker
processes by spawning fresh interpreters that must import the worker
function by module path, and a function defined in the __main__ of a
`python -m` run can't be found that way.
"""
import os
import traceback


def run_one(burst_path, out_base, method, log):
    from .methods import run
    try:
        return run(method, burst_path, out_base, log=log)
    except Exception as exc:                     # keep the rest of the run going
        log(traceback.format_exc())
        return {"status": "error", "method": method,
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "burst": {"name": os.path.basename(os.path.normpath(burst_path)),
                          "path": burst_path}}


def process_one(burst_path, out_base, method):
    # one OpenCV thread per process: parallelism comes from running bursts
    # side by side, and nested thread pools would just oversubscribe the CPU
    import cv2
    cv2.setNumThreads(1)
    lines = []
    record = run_one(burst_path, out_base, method, lines.append)
    return record, lines
