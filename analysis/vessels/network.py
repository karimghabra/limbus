"""The vessel network of an averaged stabilized frame.

  evidence   fine-scale Hessian ridge strength, each scale in units of its
             own robust spread                                 (evidence.py)
  ridges     non-maximum suppression across the ridge, path opening for
             length, hysteresis for the faint stretches          (ridges.py)
  join       ridge pieces whose ends point at each other across a short gap
             are one vessel - identity only, no invented centreline (join.py)
  fit        the physical model measures radius and peak absorbance, trims
             echoes inside wider vessels and drops what is too shallow (fit.py)

Vessel ids are assigned by position (row band, then x) of each vessel's
first point, so the same network gives the same labels on every run, and the
labels can be carried into every raw frame of the burst through the
stabilization fields.

Every stage can be switched off through NetConfig.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import evidence as ev
from . import fit as vfit
from . import join as joinmod
from . import model
from . import ridges


@dataclass
class NetConfig:
    sigmas: tuple = ev.SIGMAS  # Hessian scales (px)
    t_hi: float = 3.0          # strong ridge threshold (robust z)
    L_hi: int = 40             # strong ridge minimum path length (px)
    t_lo: float = 1.5          # faint ridge threshold, kept when connected
    L_lo: int = 20
    gap: int = 2               # path-opening gap tolerance (px)
    min_len: float = 25.0      # shortest centreline kept
    join: bool = True
    join_gap: float = 50.0     # longest gap joined (px)
    join_turn: float = 40.0    # how far an end may point off the gap (degrees)
    join_min_z: float = 1.0    # mean evidence required along the connector
    fit: bool = True
    fit_move: float = 0.0      # how far the fit may move the centreline (0 = keep the ridge)
    kappa: float = 0.0         # fit-gain required per free parameter
    min_depth: float = 0.015   # shallowest vessel accepted (peak absorbance)
    psf: float = 1.8           # optical blur (px), measured bound 2.1


def detect(A, valid, cfg=None, log=None):
    """Returns (vessels, model_image, z). Each vessel is a dict with id,
    pieces [(control points, radii, peak absorbance, record)], centrelines
    and anchor."""
    cfg = cfg or NetConfig()
    z, ang = ev.ridge_z(A, valid, cfg.sigmas, with_angle=True)
    cl = ridges.detect(z, ang, cfg.t_hi, cfg.L_hi, cfg.t_lo, cfg.L_lo, cfg.gap, cfg.min_len)
    if cfg.join:
        groups, joins = joinmod.join_ends(cl, z, cfg.join_gap, np.radians(cfg.join_turn), cfg.join_min_z)
    else:
        groups, joins = [[i] for i in range(len(cl))], []
    if log:
        log(f"  ridges: {len(cl)} centrelines in {len(groups)} vessels ({len(joins)} joins)")
    group_of = {i: gi for gi, m in enumerate(groups) for i in m}
    H, W = A.shape
    cands = []
    for C in cl:
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        L = float(np.hypot(*np.diff(C, axis=0).T).sum())
        cands.append((float(np.clip(z[yi, xi], 0, None).mean()) * L, C))
    if not cfg.fit:
        pieces = [(C, None, None, {"cand": group_of.get(i, i)}) for i, (_, C) in enumerate(cands)]
        return group(pieces, raw=True), None, z
    acc, mdl, rej, occ = vfit.run(A, valid, cands, psf=cfg.psf, kappa=cfg.kappa,
                                  min_depth=cfg.min_depth, max_move=cfg.fit_move, log=log)
    for p in acc:
        p[3]["cand"] = group_of.get(p[3].get("cand", -1), -1)
    return group(list(acc)), mdl, z


def group(pieces, raw=False):
    """Pieces sharing a parent ridge group are one vessel; ids by position."""
    groups = {}
    for k, p in enumerate(pieces):
        cid = p[3].get("cand", -1)
        groups.setdefault(("c", cid) if cid >= 0 else ("p", k), []).append(p)
    out = []
    for ps in groups.values():
        cls = [np.asarray(P, float) if raw else model.catmull(np.array(P, float))[0] for P, *_ in ps]
        first = min((c[0] for c in cls), key=lambda q: (int(q[1] // 40), q[0]))
        out.append({"pieces": ps, "centrelines": cls, "anchor": (float(first[0]), float(first[1]))})
    out.sort(key=lambda v: (int(v["anchor"][1] // 40), v["anchor"][0]))
    for i, v in enumerate(out, start=1):
        v["id"] = i
    return out


def centrelines(vessels):
    return [c for v in vessels for c in v["centrelines"]]


def labels(vessels, shape, psf=1.8):
    """Per-vessel label image: each fitted vessel's lumen carries its id.
    Longer vessels are drawn first, so a short piece cannot overwrite them."""
    lab = np.zeros(shape, np.int32)
    for v in sorted(vessels, key=lambda v: -sum(len(c) for c in v["centrelines"])):
        for (P, R, D, rec) in v["pieces"]:
            if R is None:
                continue
            P = np.asarray(P, float)
            ve = model.Vessel(P, 1.0, float(D))
            ve.p[:] = np.concatenate([np.zeros(len(P)), np.asarray(R, float), [float(D), 0.0]])
            x0, y0, x1, y1 = ve.box(shape, psf)
            xs, ys = np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32)
            lumen = ve.render(xs, ys, 0.0, with_offset=False) > 0.25 * D
            sub = lab[y0:y1, x0:x1]
            sub[(sub == 0) & lumen] = v["id"]
    return lab
