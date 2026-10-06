"""Scenes for vesselnet: splits, presets, the compact export and its loading.

Splits are fixed by seed range (PLAN.md §3) and never mixed:

    train        0 - 99,999            compact
    val          1,000,000 - 1,000,049 compact (map.json included)
    test         2,000,000 - 2,000,099 compact + the full save_scene folder (complete truth, score_fit)

The preset follows the seed (healthy 50 %, random 35 %, pathologic 15 %), drawn from the seed alone, so a
seed is always the same scene whatever range it is generated in.

A compact scene is a folder `<out>/<split>/<shard>/s<seed>/` (shards of 100 seeds) holding:

    scene.npz                 the average still (float32 DN, NaN = no data) and its valid mask, and the truth
                              rasters for learning (see COMPACT_KEYS), bool / int / float16
    observable_average.json   the observable truth (truth.load_observable)
    junctions_average.json    the image-junction truth (junctions.load_junctions)
    map.json                  the truth as a vesselmap VesselNetwork (to_vesselnetwork)
    truth.json                the VesselGraph (VesselGraph.load)
    scene.json                parameters, seeds, timings, peak VRAM, code versions and git hashes; written
                              last, so its presence marks a finished scene

Orientation and per-pixel radius / blur / contrast targets are derived when loading (from the observable
runs and map.json), not stored.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

SPLITS = {"train": (0, 100_000), "val": (1_000_000, 1_000_050), "test": (2_000_000, 2_000_100)}
PRESET_SHARES = (("healthy", 0.50), ("random", 0.35), ("pathologic", 0.15))
PRESET_SALT = 0x7E55E1
SHARD = 100
KIND = "average"
JTYPES = ("bifurcation", "confluence", "pseudo_T", "crossing", "compound", "any")

# labels.npz key -> stored dtype (the average's rasters; PLAN.md §3)
COMPACT_KEYS = {
    "observable_lumen_average": bool,
    "observable_centreline_average": np.int32,          # vessel id on the observable runs, -1 elsewhere
    "observable_edge_average": np.int32,                # observable image-graph edge id
    "observable_dont_care_average": bool,
    "observable_junction_dont_care_average": bool,
    "junction_arm_dir": np.float16,                     # (H, W, 2)
    "junction_arm_group": np.int16,
    "junction_arm_vis": np.uint8,
    "dominant": np.int32, "runner_up": np.int32,        # the vessels darkening a pixel most and second most
    "lumen_top": np.int32, "lumen_second": np.int32,    # depth order of overlapping lumens
    "radius": np.float16, "depth": np.float16,          # of the dominant vessel (d_c)
    "centreline": np.int32,                             # every vessel's 1 px centreline (complete truth)
}
# Not stored: the junction heatmaps (Gaussians at the junctions and events of the JSON files; 11 MB a scene)
# are rebuilt when loading, and the OD rasters (5 MB) are not a target.


def split_of(seed: int) -> str:
    for name, (a, b) in SPLITS.items():
        if a <= seed < b:
            return name
    raise ValueError(f"seed {seed} is in no split")


def preset_of(seed: int) -> str:
    u = np.random.default_rng([int(seed), PRESET_SALT]).random()
    acc = 0.0
    for name, share in PRESET_SHARES:
        acc += share
        if u < acc:
            return name
    return PRESET_SHARES[-1][0]


def scene_dir(out: str, seed: int) -> str:
    return os.path.join(out, split_of(seed), "%07d" % (seed // SHARD * SHARD), "s%07d" % seed)


def is_done(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, "scene.json"))


def git_info(path: str) -> dict:
    """HEAD and whether the working tree has changes, for the repository holding `path`."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=path, capture_output=True, text=True, timeout=30).stdout.strip()
    try:
        return dict(head=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception as e:                       # noqa: BLE001 (provenance must never stop a run)
        return dict(error=repr(e))


def provenance() -> dict:
    """Git state of LIMBUS (this file's repository) and of the vesselscene in use."""
    import vesselscene
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(limbus=git_info(here), vesselscene=git_info(os.path.dirname(vesselscene.__path__[0])))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<array {o.dtype} {o.shape}>"
    return o


def export_compact(sc, folder: str, extra: dict | None = None) -> dict:
    """Write the compact form of a scene (module docstring) into `folder`; scene.json last."""
    from vesselscene import junctions as J
    from vesselscene import truth as T
    from vesselscene.scene import _versions, to_vesselnetwork
    from vesselscene import render as R
    os.makedirs(folder, exist_ok=True)
    L = sc.labels
    arrs = dict(img=np.asarray(sc.images[KIND], np.float32), valid=np.asarray(sc.formed[KIND]["valid"], bool))
    missing = []
    for key, dt in COMPACT_KEYS.items():
        if key in L and isinstance(L[key], np.ndarray):
            arrs[key] = np.asarray(L[key]).astype(dt)
        else:
            missing.append(key)
    np.savez_compressed(os.path.join(folder, "scene.npz"), **arrs)
    T.save_observable(sc.observable[KIND], os.path.join(folder, f"observable_{KIND}.json"))
    J.save_junctions(sc.junctions[KIND], os.path.join(folder, f"junctions_{KIND}.json"))
    H, W = sc.params["shape"]
    net = to_vesselnetwork(sc.graph, float(sc.params["k"]), (0.0, 0.0), (H, W), R.Optics(**sc.params["optics"]),
                           blur_px=sc.params["formed"][KIND]["blur_px"], hct_mod=sc.fills.get(KIND))
    net.save(os.path.join(folder, "map.json"))
    sc.graph.save(os.path.join(folder, "truth.json"))
    meta = dict(_jsonable(sc.params), versions=_versions(), missing_keys=missing,
                counts=dict(vessels=len(sc.graph.vessels), crossings_in_frame=len(sc.crossings),
                            junctions=len(sc.junctions[KIND]), observable=sc.observable[KIND].get("summary")),
                **(extra or {}))
    with open(os.path.join(folder, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(meta), fh, indent=1)
    return meta


def load_compact(folder: str) -> dict:
    """The arrays of scene.npz and the parsed JSON files of a compact scene."""
    z = np.load(os.path.join(folder, "scene.npz"))
    out = {k: z[k] for k in z.files}
    for name in ("scene", f"observable_{KIND}", f"junctions_{KIND}"):
        p = os.path.join(folder, name + ".json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh)
    if isinstance(out.get(f"junctions_{KIND}"), dict):          # junctions.load_junctions' list
        out[f"junctions_{KIND}"] = out[f"junctions_{KIND}"]["junctions"]
    out["folder"] = folder
    return out


def list_scenes(out: str, split: str) -> list[str]:
    """Finished scene folders of a split, by seed."""
    root = os.path.join(out, split)
    if not os.path.isdir(root):
        return []
    found = []
    for shard in sorted(os.listdir(root)):
        d = os.path.join(root, shard)
        if os.path.isdir(d):
            found += [os.path.join(d, s) for s in sorted(os.listdir(d)) if is_done(os.path.join(d, s))]
    return found
