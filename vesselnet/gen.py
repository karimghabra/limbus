"""Generate vesselnet scenes from vesselscene: resumable, by split and seed range, in parallel processes.

    python vesselnet/gen.py --split val --out data                       # the whole split
    python vesselnet/gen.py --split train --start 0 --stop 500 --procs 3 --out data

Each scene is written by data.export_compact into data.scene_dir(out, seed) (the test split also gets the full
save_scene folder in `full/`). A scene is first written to `<folder>.tmp` and renamed when finished, so an
interrupted run leaves no half scenes; finished scenes are skipped. Failures go to `<out>/<split>/failures.jsonl`
and do not stop the run; a scene running longer than --max-seconds is stopped and logged (and not retried
unless --retry-failed). With --procs N, N worker processes take every N-th seed; each logs to
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


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def thread_env(n):
    """The environment with every native thread pool (OpenMP, MKL, OpenBLAS) capped at n: torch's
    set_num_threads caps torch alone, and NumPy / SciPy / OpenCV pools sized to all cores in every worker
    oversubscribe the CPU (a 46 s scene took over an hour with 3 workers)."""
    env = dict(os.environ)
    env.update({k: str(n) for k in THREAD_VARS})
    return env


def seeds_of(a):
    lo, hi = D.SPLITS[a.split]
    start = lo if a.start is None else a.start
    stop = hi if a.stop is None else a.stop
    if not (lo <= start <= stop <= hi):
        raise SystemExit(f"--start/--stop must lie in the {a.split} split's range {lo}-{hi}")
    return list(range(start, stop))


def _one_child(args):
    """Entry of a child process making one scene (worker --max-seconds)."""
    seed, out, device, prov, threads = args
    _cap_threads(threads)
    return one(seed, out, device, prov)


def _fail(fail_log, seed, error, trace=""):
    with open(fail_log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(seed=seed, preset=D.preset_of(seed), error=error, trace=trace,
                                 time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print(f"seed {seed} FAILED: {error}", flush=True)


def _cap_threads(n):
    if n:
        import cv2
        torch.set_num_threads(n)
        cv2.setNumThreads(n)
        os.environ.update({k: str(n) for k in THREAD_VARS})   # inherited by spawned scene processes


def worker(a):
    _cap_threads(a.threads)
    prov = D.provenance()
    seeds = [s for s in seeds_of(a) if s % a.of == a.worker]
    fail_log = os.path.join(a.out, a.split, "failures.jsonl")
    os.makedirs(os.path.dirname(fail_log), exist_ok=True)
    failed = set()
    if os.path.exists(fail_log) and not a.retry_failed:  # scenes that timed out before are not retried
        failed = {json.loads(l)["seed"] for l in open(fail_log, encoding="utf-8") if "timed out" in l}
    print(f"worker {a.worker}/{a.of}: {len(seeds)} seeds, device {a.device}, max {a.max_seconds} s a scene, "
          f"git {json.dumps(prov)}", flush=True)
    for seed in seeds:
        if D.is_done(D.scene_dir(a.out, seed)) or seed in failed:
            continue
        try:
            if a.max_seconds > 0:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(1) as pool:
                    r = pool.apply_async(_one_child, ((seed, a.out, a.device, prov, a.threads),))
                    try:
                        m = r.get(timeout=a.max_seconds)
                    except mp.TimeoutError:
                        pool.terminate()
                        shutil.rmtree(D.scene_dir(a.out, seed) + ".tmp", ignore_errors=True)
                        _fail(fail_log, seed, f"timed out after {a.max_seconds} s")
                        continue
            else:
                m = one(seed, a.out, a.device, prov)
        except Exception as e:                        # one bad scene must not stop a long run
            _fail(fail_log, seed, repr(e), traceback.format_exc())
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
    ap.add_argument("--max-seconds", type=int, default=1800,
                    help="a scene taking longer is stopped and logged as failed (0: no limit)")
    ap.add_argument("--retry-failed", action="store_true", help="retry scenes that timed out before")
    ap.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--of", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.threads:
        a.threads = max(1, (os.cpu_count() or 1) // max(a.procs, a.of))
    if a.of > 1:
        return worker(a)
    if a.procs <= 1:
        worker(a)
        return missing(a)
    logs = os.path.join(a.out, "logs")
    os.makedirs(logs, exist_ok=True)
    procs = []
    for i in range(a.procs):
        cmd = [sys.executable, os.path.abspath(__file__), "--split", a.split, "--out", a.out, "--device", a.device,
               "--threads", str(a.threads), "--worker", str(i), "--of", str(a.procs),
               "--max-seconds", str(a.max_seconds)] + (["--retry-failed"] if a.retry_failed else [])
        if a.start is not None:
            cmd += ["--start", str(a.start)]
        if a.stop is not None:
            cmd += ["--stop", str(a.stop)]
        fh = open(os.path.join(logs, f"gen_{a.split}_{i}.log"), "a", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=thread_env(a.threads)), fh))
    print(f"{a.procs} workers started; logs in {logs}", flush=True)
    codes = [p.wait() for p, _ in procs]
    for _, fh in procs:
        fh.close()
    done = len(D.list_scenes(a.out, a.split))
    print(f"workers exited {codes}; {done} finished scenes in {a.split}", flush=True)
    return missing(a)


def missing(a):
    """Exit status: 0 when every seed of the range is finished or timed out before (those are not retried
    without --retry-failed), else 3, so a runner (resume.ps1) tries again."""
    fail_log = os.path.join(a.out, a.split, "failures.jsonl")
    timed = set()
    if os.path.exists(fail_log) and not a.retry_failed:
        timed = {json.loads(l)["seed"] for l in open(fail_log, encoding="utf-8") if "timed out" in l}
    left = [s for s in seeds_of(a) if not D.is_done(D.scene_dir(a.out, s)) and s not in timed]
    print(f"{len(left)} seeds left in {a.split}" + (f": {left[:20]}" if left else ""), flush=True)
    return 3 if left else 0


if __name__ == "__main__":
    sys.exit(main())
