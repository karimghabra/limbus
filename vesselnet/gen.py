"""Generate vesselnet scenes from vesselscene: resumable, by split and seed range, in parallel processes.

    python vesselnet/gen.py --split val --out data                       # the whole split
    python vesselnet/gen.py --split train --start 0 --stop 500 --procs 3 --out data

Each scene is written by data.export_compact into data.scene_dir(out, seed) (the test split also gets the full
save_scene folder in `full/`). A scene is first written to `<folder>.tmp` and renamed when finished, so an
interrupted run leaves no half scenes; finished scenes are skipped. Failures go to `<out>/<split>/failures.jsonl`
and do not stop the run. With --procs N, N worker processes take every N-th seed; each logs to
`<out>/logs/gen_<split>_<i>.log`. Every scene.json records the seed, preset, git hashes, stage timings and the
peak VRAM of its process.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # the LIMBUS checkout: vesselmap, vesselnet

import torch  # noqa: E402

from vesselnet import data as D  # noqa: E402


def one(seed, out, device, prov):
    from vesselscene.scene import make_scene, save_scene
    folder = D.scene_dir(out, seed)
    if D.is_done(folder):
        return None
    tmp = folder + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    preset = D.preset_of(seed)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sc = make_scene(seed, preset, shape=(1200, 1920), device=device, kinds=(D.KIND,))
    t_make = time.perf_counter() - t0
    t1 = time.perf_counter()
    full = {}
    if D.split_of(seed) == "test":
        full = save_scene(sc, os.path.join(tmp, "full"), overlay=True)
    t_full = time.perf_counter() - t1
    extra = dict(vesselnet=dict(seed=seed, preset=preset, split=D.split_of(seed), device=device,
                                provenance=prov, seconds_make=round(t_make, 2), seconds_full_save=round(t_full, 2),
                                peak_vram_mb=(round(torch.cuda.max_memory_allocated() / 2**20)
                                              if device.startswith("cuda") else None),
                                peak_vram_reserved_mb=(round(torch.cuda.max_memory_reserved() / 2**20)
                                                       if device.startswith("cuda") else None),
                                full_files=sorted(os.path.basename(p) for p in full.values())))
    t2 = time.perf_counter()
    meta = D.export_compact(sc, tmp, extra)
    t_save = time.perf_counter() - t2
    # the compact save's own time, recorded after the fact
    p = os.path.join(tmp, "scene.json")
    meta["vesselnet"]["seconds_compact_save"] = round(t_save, 2)
    meta["vesselnet"]["seconds_total"] = round(time.perf_counter() - t0, 2)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(D._jsonable(meta), fh, indent=1)
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    os.replace(tmp, folder)
    return meta


def seeds_of(a):
    lo, hi = D.SPLITS[a.split]
    start = lo if a.start is None else a.start
    stop = hi if a.stop is None else a.stop
    if not (lo <= start <= stop <= hi):
        raise SystemExit(f"--start/--stop must lie in the {a.split} split's range {lo}-{hi}")
    return list(range(start, stop))


def worker(a):
    if a.threads:
        torch.set_num_threads(a.threads)
    prov = D.provenance()
    seeds = [s for s in seeds_of(a) if s % a.of == a.worker]
    fail_log = os.path.join(a.out, a.split, "failures.jsonl")
    os.makedirs(os.path.dirname(fail_log), exist_ok=True)
    print(f"worker {a.worker}/{a.of}: {len(seeds)} seeds, device {a.device}, git {json.dumps(prov)}", flush=True)
    for seed in seeds:
        try:
            m = one(seed, a.out, a.device, prov)
        except Exception as e:                        # one bad scene must not stop a long run
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(seed=seed, error=repr(e), trace=traceback.format_exc(),
                                         time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
            print(f"seed {seed} FAILED: {e!r}", flush=True)
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        if m is not None:
            v = m["vesselnet"]
            print(f"seed {seed} {v['preset']} {v['seconds_total']:.0f} s (make {v['seconds_make']:.0f}, "
                  f"save {v['seconds_compact_save']:.0f}), peak VRAM {v['peak_vram_mb']} MB, "
                  f"{m['counts']['vessels']} vessels, {m['counts']['junctions']} junctions", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True, choices=tuple(D.SPLITS))
    ap.add_argument("--start", type=int)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--out", default="data")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads per process (0: cores / procs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--of", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.threads:
        a.threads = max(1, (os.cpu_count() or 1) // max(a.procs, a.of))
    if a.procs <= 1 or a.of > 1:
        return worker(a)
    logs = os.path.join(a.out, "logs")
    os.makedirs(logs, exist_ok=True)
    procs = []
    for i in range(a.procs):
        cmd = [sys.executable, os.path.abspath(__file__), "--split", a.split, "--out", a.out, "--device", a.device,
               "--threads", str(a.threads), "--worker", str(i), "--of", str(a.procs)]
        if a.start is not None:
            cmd += ["--start", str(a.start)]
        if a.stop is not None:
            cmd += ["--stop", str(a.stop)]
        fh = open(os.path.join(logs, f"gen_{a.split}_{i}.log"), "a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
    print(f"{a.procs} workers started; logs in {logs}", flush=True)
    codes = [p.wait() for p, _ in procs]
    for _, fh in procs:
        fh.close()
    done = len(D.list_scenes(a.out, a.split))
    print(f"workers exited {codes}; {done} finished scenes in {a.split}", flush=True)


if __name__ == "__main__":
    main()
