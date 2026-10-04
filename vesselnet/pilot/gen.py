"""Training scenes from vesselscene (pilot): one image kind per scene, its
valid mask, and its truth.  Kept per scene: the image (float32), the
observable truth's junctions and rasters (lumen, centreline, don't-care,
junction don't-care), and the image-junction truth's junctions with their
arms (direction, visibility, vessel, partition group).

    python vesselnet/pilot/gen.py --start 0 --stop 20 --kind average --size 512 --device cuda --out data/pilot

vesselscene must be importable (pip install -e path/to/vesselscene, or set
VESSELSCENE to its checkout).  The preset follows the seed (healthy, random,
healthy, random, pathologic).  Existing outputs are skipped, so an
interrupted run resumes.  CPU timing (4 cores, frame only): about 150 s for
512 x 512; one thread per process: 4-20 min.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

if os.environ.get("VESSELSCENE"):
    sys.path.insert(0, os.environ["VESSELSCENE"])
import torch                                   # noqa: E402
from vesselscene.scene import make_scene        # noqa: E402

PRESETS = ("healthy", "random", "healthy", "random", "pathologic")


def preset_of(seed):
    return PRESETS[seed % len(PRESETS)]


def compact_junctions(js):
    out = []
    for j in js:
        out.append(dict(x=float(j["x"]), y=float(j["y"]), radius=float(j["radius"]), type=j["type"],
                        type_visible=j.get("type_visible"), ambiguous=bool(j.get("ambiguous")),
                        n_visible_lines=j.get("n_visible_lines"), partition=j.get("partition"),
                        members=j.get("members"),
                        arms=[dict(dir=a.get("dir"), xy=a.get("xy"), vis=a.get("visibility"), group=a.get("group"),
                                   vid=a.get("vid"), width=a.get("width_px")) for a in j["arms"]]))
    return out


def one(seed, kind, size, device, out):
    path = os.path.join(out, "s%05d.npz" % seed)
    if os.path.exists(path):
        return None
    t0 = time.time()
    sc = make_scene(seed, preset_of(seed), shape=(size, size), device=device, kinds=(kind,))
    L = sc.labels
    obs = sc.observable[kind]
    ojs = [dict(x=float(j["x"]), y=float(j["y"]), radius=float(j["radius"]), type=j.get("type"),
                type_observable=j.get("type_observable"), n_lines=j.get("n_lines")) for j in obs["junctions"]]
    meta = dict(seed=seed, preset=preset_of(seed), kind=kind, size=size, obs_junctions=ojs,
                dont_care_junctions=[dict(x=float(j["x"]), y=float(j["y"]), radius=float(j["radius"]))
                                     for j in obs.get("dont_care_junctions", []) if "x" in j],
                junctions=compact_junctions(sc.junctions[kind]), seconds=round(time.time() - t0, 1))
    np.savez_compressed(path, img=np.asarray(sc.images[kind], np.float32),
                        lumen=np.asarray(L["observable_lumen_%s" % kind], bool),
                        centreline=np.asarray(L["observable_centreline_%s" % kind]) >= 0,
                        dont_care=np.asarray(L["observable_dont_care_%s" % kind], bool),
                        jdont_care=np.asarray(L["observable_junction_dont_care_%s" % kind], bool),
                        meta=np.frombuffer(pickle.dumps(meta), np.uint8))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--kind", default="average", choices=("average", "frame"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch's default)")
    ap.add_argument("--out", default="data/pilot")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    for seed in range(a.start, a.stop):
        try:
            m = one(seed, a.kind, a.size, a.device, a.out)
        except Exception as e:                  # one bad scene must not stop a long run
            print("seed %d failed: %r" % (seed, e), flush=True)
            continue
        if m is not None:
            print("seed %d %s %.0f s, %d observable junctions" % (seed, m["preset"], m["seconds"],
                                                                  len(m["obs_junctions"])), flush=True)


if __name__ == "__main__":
    main()
