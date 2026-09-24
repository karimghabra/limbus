"""Lucky fusion and vessel annotation of every stabilized burst.

    cd analysis
    python -m stabilize ../recordings --method nonrigid     # first, once
    python -m lucky ../recordings                           # then this

Reads the stabilization results (default: the 'stabilization' folder beside
the recordings folder, as python -m stabilize writes it) and writes
<stab>/lucky/<burst>/ plus a summary.csv. Uses the non-rigid result where it
exists and succeeded, the translation result otherwise. Re-running skips
bursts whose results are current — same code version and parameters.
Raw burst folders are only ever read.
"""
import argparse
import csv
import json
import os
import sys
import time


def main(argv=None):
    from stabilize.__main__ import _inside, _is_burst, default_out
    from stabilize.bursts import discover

    from . import __version__
    from .config import AnnotateParams, LuckyParams, params_hash
    from .pipeline import process

    ap = argparse.ArgumentParser(prog="python -m lucky", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a burst_* folder, or a folder containing them")
    ap.add_argument("--stab", default=None,
                    help="stabilization results (default: 'stabilization' beside the recordings)")
    ap.add_argument("--out", default=None, help="results folder (default: <stab>/lucky)")
    ap.add_argument("--method", choices=["auto", "nonrigid", "translation"], default="auto",
                    help="which saved alignment to use (default: non-rigid, else translation)")
    ap.add_argument("--only", default="", help="process only bursts whose name contains this")
    ap.add_argument("--force", action="store_true", help="reprocess even if current")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"not a folder: {root}")
    stab = os.path.abspath(args.stab or default_out(root))
    out = os.path.abspath(args.out or os.path.join(stab, "lucky"))
    recordings = os.path.dirname(root) if _is_burst(root) else root
    if _inside(out, recordings):
        sys.exit("--out must not be inside the recordings folder: raw bursts are never written to.")
    lp, anp = LuckyParams(), AnnotateParams()
    phash = params_hash(lp, anp)

    bursts = [b for b in discover(root) if args.only in os.path.basename(b)]
    todo = [b for b in bursts if args.force or not _current(out, b, phash, __version__)]
    print(f"lucky {__version__}: {len(bursts)} bursts found, {len(todo)} to process -> {out}",
          flush=True)
    started = time.time()
    errors = 0
    for n, b in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {os.path.basename(b)}", flush=True)
        try:
            process(b, stab, out, lp, anp, args.method, log=lambda s: print(s, flush=True))
        except FileNotFoundError as exc:            # not stabilized (or skipped) yet
            print(f"  skipped: {exc}", flush=True)
        except Exception as exc:                    # keep the rest of the run going
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            errors += 1
    rows = write_summary(out)
    print(f"\nsummary: {len(rows)} bursts -> {os.path.join(out, 'summary.csv')} "
          f"({time.time() - started:.0f}s)", flush=True)
    sys.exit(1 if errors else 0)


def _current(out, burst, phash, version):
    path = os.path.join(out, os.path.basename(os.path.normpath(burst)), "lucky.json")
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return False
    proc = rec.get("processing", {})
    return proc.get("params_hash") == phash and proc.get("lucky_version") == version


def write_summary(out):
    """One row per burst with a result on disk: the lucky annotation next
    to the plain mean's and the consensus-mask skeleton's."""
    import glob
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "*", "lucky.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        m, l_ = r["by_image"]["p0"], r["by_image"][r["output_image"]]
        cons = r.get("consensus_skeleton") or {}
        rows.append({
            "burst": r["burst"], "alignment": r["alignment"], "frames": r["frames_fused"],
            "segments": r["segments"],
            "centreline_px_per_mpx_consensus": round(cons.get("centreline_px_per_mpx", 0)),
            "centreline_px_per_mpx_mean": round(m["centreline_px_per_mpx"]),
            "centreline_px_per_mpx_lucky": round(l_["centreline_px_per_mpx"]),
            "split_half_f1_mean": round(m["split_half_f1"], 3),
            "split_half_f1_lucky": round(l_["split_half_f1"], 3),
            "edge_sharpness_mean": round(m["edge_sharpness"], 4),
            "edge_sharpness_lucky": round(l_["edge_sharpness"], 4),
            "consensus_found_by_lucky": round(cons.get("found_by_lucky_annotation", 0), 3),
        })
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
