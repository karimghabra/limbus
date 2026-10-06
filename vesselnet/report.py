"""Markdown tables for RESULTS.md from the JSONL logs of energy_check.py and baselines.py.

    python vesselnet/report.py energy   runs/energy_check/energy_check.jsonl
    python vesselnet/report.py detector runs/detector_tune/detector.jsonl
    python vesselnet/report.py baselines runs/baselines
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np


def rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def f(v, d=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def energy(path):
    from vesselnet.energy_check import summarize
    R = rows(path)
    print(f"{len(R)} windows from seeds {sorted({r['seed'] for r in R})}; tau "
          f"{f(float(np.median([r['tau'] for r in R])), 1)} (median), price per px "
          f"{f(float(np.median([r['price'] for r in R])), 0)} (median), oracle fit "
          f"{f(float(np.median([r['oracle']['fit']['seconds'] for r in R])), 0)} s (median)")
    summarize(path)


def detector(path):
    R = rows(path)
    by = defaultdict(list)
    for r in R:
        by[json.dumps(r["params"], sort_keys=True)].append(r)
    out = []
    for k, rs in by.items():
        tp = sum(r["junction_recall"] * r["n_truth_junctions"] for r in rs if r["n_truth_junctions"])
        n = sum(r["n_truth_junctions"] for r in rs)
        fp = sum(r["junction_false_positives"] for r in rs)
        rec, prec = tp / max(n, 1), tp / max(tp + fp, 1e-9)
        ev = sum(r["events_found"] for r in rs) / max(sum(r["events"] for r in rs), 1)
        out.append((2 * rec * prec / max(rec + prec, 1e-9), k, rec, prec, ev, len(rs), sum(r["n_dets"] for r in rs),
                    float(np.mean([r["seconds"] for r in rs]))))
    print("| settings | windows | detections | recall | precision | F1 | event recall | s / window |")
    print("|---|---|---|---|---|---|---|---|")
    for F1, k, rec, prec, ev, nw, nd, s in sorted(out, reverse=True):
        print(f"| `{k}` | {nw} | {nd} | {rec:.3f} | {prec:.3f} | {F1:.3f} | {ev:.3f} | {s:.0f} |")


ARCH = ("centreline_recall", "centreline_precision", "junction_recall", "junction_precision",
        "crossings_as_nodes_rate", "forks_found_rate", "pairing_accuracy", "pairing_accuracy_unambiguous",
        "vessels_fragments_weighted", "vessels_purity", "vessels_best_cover", "vessels_mixed_edges", "n_edges")


def _pool(dicts, key):
    v = [d[key] for d in dicts if isinstance(d.get(key), (int, float)) and math.isfinite(d[key])]
    return float(np.mean(v)) if v else float("nan")


def baselines(d):
    O = rows(os.path.join(d, "oracle.jsonl"))
    V = rows(os.path.join(d, "vesselmap.jsonl"))
    print(f"oracle: {len(O)} windows; vesselmap: {len(V)} windows (mean over windows)\n")
    cols = [("oracle (truth)", [r["arch"] for r in O])]
    for st in ("consolidate", "consolidate_segments", "search", "search_segments"):
        cols.append((f"vesselmap {st}", [r["arch"][st] for r in V if st in r.get("arch", {})]))
    print("| metric | " + " | ".join(c for c, _ in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for k in ARCH:
        print(f"| {k} | " + " | ".join(f(_pool(a, k)) for _, a in cols) + " |")
    if V:
        print("\n| stage | median dE to the oracle | median dNLL | vessels (median) | seconds (median) |")
        print("|---|---|---|---|---|")
        for st in ("map", "refine", "consolidate", "search"):
            s = [r["stages"][st] for r in V if st in r["stages"]]
            print(f"| {st} | {np.median([x['dE'] for x in s]):+.0f} | {np.median([x['dnll'] for x in s]):+.0f} | "
                  f"{np.median([x['n_vessels'] for x in s]):.0f} | {np.median([x['seconds'] for x in s]):.0f} |")
    if O:
        print(f"\noracle: E median {np.median([r['E']['total'] for r in O]):.0f}, tau median "
              f"{np.median([r['tau'] for r in O]):.1f}, reduced chi2 median "
              f"{np.median([r['E']['chi2_reduced'] for r in O]):.1f}, fit median "
              f"{np.median([r['fit']['seconds'] for r in O]):.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("energy", "detector", "baselines"))
    ap.add_argument("path")
    a = ap.parse_args()
    dict(energy=energy, detector=detector, baselines=baselines)[a.what](a.path)


if __name__ == "__main__":
    main()
