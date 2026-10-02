"""Run a segmenter on a stabilization result, or list the segmenters.

    cd analysis
    python -m segmentation list
    python -m segmentation run <id> <result folder> [--burst DIR]

<result folder> is a stabilization result, e.g.
stabilization/nonrigid/burst_2026-09-16_15-50-52. The overlay goes to
<result folder>/segmentation/<id>/overlay.json, beside anything else the
segmenter writes there. Exit code 0 on success, 1 if the segmenter failed,
2 for bad arguments.
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

from . import find, list_segmenters, load, result_dir
from .inputs import Inputs
from .overlay import Overlay


def cmd_list(args):
    segs = list_segmenters()
    if args.json:
        print(json.dumps([dict(id=s.id, label=s.label, description=s.description,
                               version=s.version, path=s.path) for s in segs], indent=1))
        return 0
    for s in segs:
        print(f"{s.id:24s} {s.label}  ({s.path})\n{'':24s} {s.description}")
    return 0


def cmd_run(args):
    try:
        seg = find(args.id)
    except KeyError as exc:
        print(exc.args[0], flush=True)
        return 2
    folder = os.path.abspath(args.result)
    try:
        with open(os.path.join(folder, "metrics.json"), encoding="utf-8") as f:
            status = json.load(f).get("status")
    except (OSError, ValueError):
        status = None
    if status != "ok" or not os.path.exists(os.path.join(folder, "mean_stabilized.tif")):
        print(f"not a finished stabilization result: {folder}", flush=True)
        return 2
    out = result_dir(folder, seg.id)
    # work in a folder of its own, which replaces the previous result only
    # when this run succeeds: cancelling a long run keeps the last overlay,
    # and a failed one leaves its partial files there to look at
    work = out + ".running"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)     # a cancelled run's files may still be held open
    print(f"segmenter: {seg.label} ({seg.path})", flush=True)
    started = time.time()
    inputs = Inputs(folder, work, args.burst, log=lambda s: print(s, flush=True))
    try:
        overlay = load(seg).segment(inputs)
        if not isinstance(overlay, Overlay):
            raise TypeError(f"segment() returned {type(overlay).__name__}, not an Overlay")
        if overlay.shape != inputs.shape:
            raise ValueError(f"the overlay is {overlay.shape}, the stabilized frame {inputs.shape}")
    except ImportError as exc:
        traceback.print_exc()
        print(f"{seg.id} needs a package that isn't installed: {exc.name or exc}", flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    st = os.stat(inputs.mean_path)
    overlay.write(work, {
        "segmenter": {"id": seg.id, "label": seg.label, "version": seg.version,
                      "sha256": seg.source_hash()},
        "source": {"result": folder, "method": inputs.method,
                   "burst": (inputs.metrics.get("burst") or {}).get("name"),
                   "stabilization_params_hash": inputs.metrics.get("processing", {}).get("params_hash"),
                   "mean_mtime_ns": st.st_mtime_ns, "mean_size": st.st_size},
        "made_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.time() - started, 1),
    })
    shutil.rmtree(out, ignore_errors=True)
    try:
        os.replace(work, out)
    except OSError as exc:
        print(f"could not replace the previous result ({exc}); this one is in {work}", flush=True)
        return 1
    print(f"done: {overlay.summary or overlay.default_summary()} ({time.time() - started:.0f}s)",
          flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m segmentation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="the segmenters found")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("run", help="run one on a stabilization result")
    p.add_argument("id")
    p.add_argument("result", help="a stabilization result folder")
    p.add_argument("--burst", default=None, help="the raw burst, if not where the result says")
    args = ap.parse_args(argv)
    sys.exit(cmd_list(args) if args.cmd == "list" else cmd_run(args))


if __name__ == "__main__":
    main()
