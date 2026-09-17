"""Bulk-stabilize every burst under a folder.

    python -m stabilize "E:\\Conjunctiva Code\\limbus\\recordings" --out E:\\Analysis\\stabilization

Bursts run in parallel, one worker each. Re-running skips bursts whose
results are newer than the burst itself, so an interrupted run resumes.
Raw burst folders are only ever read.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from stabilize.bursts import discover
from stabilize.runner import process_one


def _up_to_date(burst_path, out_root):
    result = os.path.join(out_root, os.path.basename(burst_path), "metrics.json")
    return (os.path.exists(result)
            and os.path.getmtime(result) >= os.path.getmtime(burst_path))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stabilize", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder containing burst_* folders")
    ap.add_argument("--out", default=r"E:\Analysis\stabilization",
                    help="where results are written (never inside the bursts)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--only", default="", help="process only bursts whose name contains this")
    ap.add_argument("--force", action="store_true", help="reprocess even if up to date")
    args = ap.parse_args(argv)

    out_root = os.path.abspath(args.out)
    if os.path.commonpath([out_root, os.path.abspath(args.root)]) == os.path.abspath(args.root):
        sys.exit("--out must not be inside the recordings folder: raw bursts are never written to.")
    os.makedirs(out_root, exist_ok=True)

    bursts = [b for b in discover(args.root) if args.only in os.path.basename(b)]
    todo = [b for b in bursts if args.force or not _up_to_date(b, out_root)]
    print(f"{len(bursts)} bursts found, {len(todo)} to process "
          f"({len(bursts) - len(todo)} already up to date), {args.workers} workers", flush=True)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, b, out_root): b for b in todo}
        for done, fut in enumerate(as_completed(futures), 1):
            record, lines = fut.result()
            print(f"\n[{done}/{len(todo)}] " + "\n".join(lines), flush=True)
            if record.get("status") == "error":
                print(f"  ERROR: {record['skip_reason']}", flush=True)

    # the summary covers every burst with results, not just this run's
    from stabilize.report import write_summary
    records = []
    for b in bursts:
        path = os.path.join(out_root, os.path.basename(b), "metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
    rows = write_summary(out_root, records)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nsummary: {ok} stabilized, {len(rows) - ok} skipped -> "
          f"{os.path.join(out_root, 'summary.html')}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
