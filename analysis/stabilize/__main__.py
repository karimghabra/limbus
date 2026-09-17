"""Stabilize one burst, or every burst under a folder.

    cd analysis
    python -m stabilize ../recordings                                  # all bursts
    python -m stabilize ../recordings/burst_2026-09-16_15-50-52 --method nonrigid

Results go to <out>/<method>/<burst>/ (default <out>: a 'stabilization'
folder beside the recordings folder), with summary.csv / summary.html per
method covering every result there. Re-running skips bursts whose results
are current — same code version and parameters — so an interrupted run
resumes. Raw burst folders are only ever read.

Bursts run one at a time by default: OpenCV then uses every core within a
burst, which measured ~11x faster than one single-threaded worker per core.
"""
import argparse
import glob
import json
import os
import sys
import time


def _is_burst(path):
    return (os.path.basename(os.path.normpath(path)).startswith("burst_")
            and bool(glob.glob(os.path.join(path, "*.tif"))))


def _inside(path, folder):
    try:
        return (os.path.normcase(os.path.commonpath([path, folder]))
                == os.path.normcase(folder))
    except ValueError:              # different drives: can't be inside
        return False


def default_out(root):
    recordings = os.path.dirname(os.path.normpath(root)) if _is_burst(root) else root
    return os.path.join(os.path.dirname(os.path.abspath(recordings)), "stabilization")


def write_method_summary(out_base, method):
    """summary.csv/html over every result on disk for this method — not just
    this run's bursts, so a single-burst run never truncates it."""
    from .report import write_summary
    root = os.path.join(out_base, method)
    records = []
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except (OSError, ValueError):
            pass
    if not records:
        return []
    return write_summary(root, records)


def main(argv=None):
    from .bursts import discover
    from .methods import METHODS, is_current
    from .runner import process_one, run_one

    ap = argparse.ArgumentParser(prog="python -m stabilize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a burst_* folder, or a folder containing them")
    ap.add_argument("--method", choices=sorted(METHODS), default="translation")
    ap.add_argument("--out", default=None,
                    help="results folder (default: 'stabilization' beside the recordings folder)")
    ap.add_argument("--workers", type=int, default=1,
                    help="bursts processed in parallel (default 1; see above)")
    ap.add_argument("--only", default="", help="process only bursts whose name contains this")
    ap.add_argument("--force", action="store_true", help="reprocess even if current")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"not a folder: {root}")
    out_base = os.path.abspath(args.out or default_out(root))
    recordings = os.path.dirname(root) if _is_burst(root) else root
    if _inside(out_base, recordings):
        sys.exit("--out must not be inside the recordings folder: raw bursts are never written to.")
    os.makedirs(out_base, exist_ok=True)

    bursts = [b for b in discover(root) if args.only in os.path.basename(b)]
    todo = [b for b in bursts if args.force or not is_current(b, out_base, args.method)]
    label = METHODS[args.method]["label"]
    print(f"{label}: {len(bursts)} bursts found, {len(todo)} to process "
          f"({len(bursts) - len(todo)} already current) -> {os.path.join(out_base, args.method)}",
          flush=True)

    started = time.time()
    errors = 0
    if args.workers <= 1:
        for n, b in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}]", flush=True)
            record = run_one(b, out_base, args.method, lambda s: print(s, flush=True))
            errors += record.get("status") == "error"
            _report(record)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, b, out_base, args.method) for b in todo]
            for n, fut in enumerate(as_completed(futures), 1):
                record, lines = fut.result()
                print(f"\n[{n}/{len(todo)}] " + "\n".join(lines), flush=True)
                errors += record.get("status") == "error"
                _report(record)

    rows = write_method_summary(out_base, args.method)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nsummary: {ok} stabilized, {len(rows) - ok} skipped -> "
          f"{os.path.join(out_base, args.method, 'summary.html')}  ({time.time() - started:.0f}s)",
          flush=True)
    sys.exit(1 if errors else 0)


def _report(record):
    status = record.get("status")
    if status == "error":
        print(f"  ERROR: {record.get('skip_reason')}", flush=True)
    elif status == "skipped":
        print(f"  skipped: {record.get('skip_reason')}", flush=True)


if __name__ == "__main__":
    main()
