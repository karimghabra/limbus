"""Seed-and-grow as the PRIMARY detector, not as a finishing touch.

The staged search decides the whole network by thresholding the evidence, and
everything after it - junctions, consolidation, growth - can only rearrange
what that decision produced. Growth added 0.2% because by the time it ran
there was nothing left to find.

This turns the order round. Only the most certain fragments are taken as
seeds; the network is then grown out of them, each step judged against THAT
vessel's own profile and the direction of the local spectrum, adapting as it
goes. A faint vessel is followed because its own seed says what to look for,
not because a global threshold was lowered until it appeared.

usage: python seeded.py <crop key> [seed z] [seed length]
"""
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import numpy as np

sys.path[:0] = [".", ".."]
from vessels import benchmark as bench  # noqa: E402
from vessels import crystal as cr  # noqa: E402
from vessels import evidence as ev  # noqa: E402
from vessels import fit as vfit  # noqa: E402
from vessels import graph as gr  # noqa: E402
from vessels import image as vimg  # noqa: E402
from vessels import model  # noqa: E402
from vessels import network as net  # noqa: E402
from vessels import ridges as rg  # noqa: E402


def seeds_of(A, valid, cfg, t_seed, L_seed):
    """The fragments we are most sure of: strong, long, and fitted."""
    zl, angl, _ = ev.ridge_z(A, valid, cfg.sigmas_large, with_angle=True, with_scale=True)
    large = [C for C in rg.detect(zl, angl, t_seed, L_seed, t_seed, L_seed,
                                  cfg.gap, L_seed, reach=0) if rg.single_peaked(A, C)]
    zs, angs, _ = ev.ridge_z(A, valid, cfg.sigmas_small, with_angle=True, with_scale=True)
    small = rg.detect(zs, angs, t_seed, L_seed, t_seed, L_seed, cfg.gap, L_seed, reach=0)
    H, W = A.shape

    def strength(C):
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        return float(np.median(A[yi, xi])) * float(np.hypot(*np.diff(C, axis=0).T).sum())

    return ([(1e6 + strength(C), C) for C in large] + [(strength(C), C) for C in small],
            np.maximum(zl, zs))


def grow_network(A, valid, cands, cfg, gcfg, growcfg, rounds=4, log=print):
    """Fit the seeds, then grow every free end, repeatedly, until nothing more
    is found. Each round's growths become the next round's seeds."""
    acc, mdl, rej, occ = vfit.run(A, valid, cands, psf=cfg.psf, kappa=cfg.kappa,
                                  min_depth=cfg.min_depth, max_move=cfg.fit_move)
    H, W = A.shape
    segs = []
    for P, R, D, rec in acc:
        C = model.catmull(np.array(P, float))[0]
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        segs.append({"points": C, "radius": float(np.median(R)),
                     "darkness": float(np.median(A[yi, xi])), "depth": float(D),
                     "fit": (P, R, D, rec)})
    log(f"  seeds: {len(cands)} -> {len(segs)} fitted, {sum(len(s['points']) for s in segs) * 0.5:.0f} px")
    total = 0.0
    for rnd in range(rounds):
        grown = 0.0
        for seg in segs:
            for at_start in (True, False):
                P = np.asarray(seg["points"], float)
                p0 = P[0] if at_start else P[-1]
                # do not grow into a vessel already traced
                others = [np.asarray(s["points"], float) for s in segs if s is not seg]
                if any(float(np.hypot(*(L - p0).T).min()) <= gcfg.node_tol for L in others):
                    continue
                add = cr.grow_end(A, valid, p0, gr._tangent(P, at_start),
                                  float(seg["radius"]), float(seg["depth"]), None,
                                  growcfg, cfg.psf)
                if len(add) >= 2:
                    seg["points"] = np.vstack([add[::-1], P]) if at_start else np.vstack([P, add])
                    grown += float(np.hypot(*np.diff(np.vstack([p0, add]), axis=0).T).sum())
        total += grown
        log(f"  round {rnd + 1}: +{grown:.0f} px")
        if grown < 20:
            break
    log(f"  grown in total: {total:.0f} px")
    return segs, occ, total


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1522L"
    t_seed = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    L_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    cfg = net.NetConfig()
    gcfg = gr.GraphConfig()
    growcfg = cr.GrowConfig()
    A, valid = vimg.prepare(bench.load_crop(key))
    cands, z = seeds_of(A, valid, cfg, t_seed, L_seed)
    segs, occ, grown = grow_network(A, valid, cands, cfg, gcfg, growcfg)
    total = sum(float(np.hypot(*np.diff(np.asarray(s["points"], float), axis=0).T).sum()) for s in segs)
    mp = valid.sum() / 1e6
    print(f"{key}: seed z>{t_seed} len>{L_seed}: {len(segs)} vessels, {total:.0f} px "
          f"({total / mp:.0f} px/MP), of which {grown:.0f} px grown")
    import pickle
    pickle.dump([s["points"] for s in segs], open(f"seeded_{key}.pkl", "wb"))
