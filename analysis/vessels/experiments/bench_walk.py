"""Injection-recovery for the walker ensemble, beside the staged search.

The planted vessels are isolated by construction - they are put where nothing
else is - so this is the hardest possible case for a method that only finds
what connects to something it is already sure of. That is exactly why it has
to be measured.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"E:\Conjunctiva Code\limbus\analysis")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench, bench2
import walkers as wk
from seeded import seeds_of
from vessels import image as vimg, network as net, model, evidence as ev, ridges as rg

bench2.prepare = lambda mean: vimg.prepare(mean)
bench.GRID = [(1.0, 0.03), (1.0, 0.05), (1.5, 0.03), (2.5, 0.03), (1.5, 0.08),
              (5.0, 0.08), (8.0, 0.12), (12.0, 0.15)]
CFG = net.NetConfig()


def walker_network(A, valid, t_seed=5.0, L_seed=60):
    sup, z, scale, dirs = wk.direction_field(A, valid, CFG.sigmas_small + CFG.sigmas_large, n_dir=16)
    cands, _ = seeds_of(A, valid, CFG, t_seed, L_seed)
    segs = [{"points": model.catmull(np.array(C, float))[0] if len(C) > 3 else np.asarray(C, float)}
            for _, C in cands]
    if not segs:
        return []
    visit = wk.propagate(sup, dirs, wk.seeds_from(segs), steps=120, step_px=2.0)
    v = np.log1p(visit / max(visit.max(), 1e-12) * 1000).astype(np.float32)
    zz, aa, _ = ev.ridge_z(v, valid, (1.0, 1.5, 2.0, 3.0, 4.5), with_angle=True, with_scale=True)
    return rg.detect(zz, aa, t_hi=2.0, L_hi=30, t_lo=1.0, L_lo=15, min_len=25.0, reach=60)


def multi(A, valid):
    return {"walkers seed z5 L60": walker_network(A, valid, 5.0, 60),
            "walkers seed z4 L40": walker_network(A, valid, 4.0, 40)}


if __name__ == "__main__":
    bench2.run(multi, sys.argv[1:] or ["1522L"], "walkers", per=6, seeds=(1, 2))
