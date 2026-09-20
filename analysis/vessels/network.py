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
from . import graph as gr
from . import join as joinmod
from . import model
from . import ridges


@dataclass
class NetConfig:
    sigmas: tuple = ev.SIGMAS  # Hessian scales (px)
    drop_echoes: bool = True   # drop the fine-scale lines a wide vessel throws along its walls
    t_hi: float = 3.0          # strong ridge threshold (robust z)
    L_hi: int = 40             # strong ridge minimum path length (px)
    t_lo: float = 1.5          # faint ridge threshold, kept when connected
    L_lo: int = 20
    gap: int = 2               # path-opening gap tolerance (px)
    min_len: float = 25.0      # shortest centreline kept
    reach: int = 60            # how far a faint ridge may extend from a strong one (px)
    join: bool = True
    join_gap: float = 50.0     # longest gap joined (px)
    join_turn: float = 40.0    # how far an end may point off the gap (degrees)
    join_min_z: float = 1.0    # mean evidence required along the connector
    graph: bool = True         # resolve crossings and bifurcations, thread pieces into vessels
    attach: bool = True        # extend a branch that stops short of the vessel it leaves
    attach_gap: float = 30.0   # how far it may be extended (px)
    under: bool = True         # rejoin a vessel across the shadow of a wider one
    under_gap: float = 45.0    # widest such shadow crossed (px)
    node_tol: float = 10.0     # piece ends this close meet at one junction (px)
    touch_tol: float = 8.0     # an end this close to another piece splits it (px)
    fit: bool = True
    fit_move: float = 0.0      # how far the fit may move the centreline (0 = keep the ridge)
    kappa: float = 0.0         # fit-gain required per free parameter
    min_depth: float = 0.015   # shallowest vessel accepted (peak absorbance)
    psf: float = 1.8           # optical blur (px), measured bound 2.1


def detect(A, valid, cfg=None, log=None):
    """Returns (vessels, model_image, z).

    Each vessel is a dict with id, label ("3.2"), parent, children, strahler,
    the fitted pieces it is made of, one continuous centreline, and the
    junctions along it."""
    cfg = cfg or NetConfig()
    z, ang, sc = ev.ridge_z(A, valid, cfg.sigmas, with_angle=True, with_scale=True)
    cl = ridges.detect(z, ang, cfg.t_hi, cfg.L_hi, cfg.t_lo, cfg.L_lo, cfg.gap, cfg.min_len,
                       reach=cfg.reach, scale=sc if cfg.drop_echoes else None, A=A)
    if log:
        log(f"  ridges: {len(cl)} centrelines")
    H, W = A.shape
    cands = []
    for C in cl:
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        L = float(np.hypot(*np.diff(C, axis=0).T).sum())
        cands.append((float(np.clip(z[yi, xi], 0, None).mean()) * L, C))
    if cfg.fit:
        acc, mdl, rej, occ = vfit.run(A, valid, cands, psf=cfg.psf, kappa=cfg.kappa,
                                      min_depth=cfg.min_depth, max_move=cfg.fit_move, log=log)
    else:
        acc = [(C, None, None, {}) for _, C in cands]
        mdl, occ = None, np.zeros(A.shape, bool)
    pieces = []
    for P, R, D, rec in acc:
        C = model.catmull(np.array(P, float))[0] if R is not None else np.asarray(P, float)
        pieces.append({"points": C, "radius": float(np.median(R)) if R is not None else 2.0,
                       "fit": (P, R, D, rec)})
    if not cfg.graph:
        return group([p["fit"] for p in pieces]), mdl, z
    return _thread(pieces, A, z, occ, cfg, log), mdl, z


def _thread(pieces, A, z, occupied, cfg, log=None):
    """Pieces -> vessels: split at touches, classify every junction, add the
    long-gap joins as further pairings, then walk the chains."""
    gcfg = gr.GraphConfig(node_tol=cfg.node_tol, touch_tol=cfg.touch_tol)
    segs = list(pieces)
    n_merged = 0
    if cfg.join:                    # repair broken pieces before reading junctions
        before = len(segs)
        segs = gr.merge_gaps(segs, z, joinmod.join_ends, cfg.join_gap,
                             np.radians(cfg.join_turn), cfg.join_min_z)
        n_merged = before - len(segs)
    if cfg.attach:
        segs = gr.attach_to_body(segs, A, gcfg, cfg.attach_gap)
    segs = gr.split_at_touches(segs, gcfg)
    ends, nodes, pairs, records = gr.build(segs, gcfg)
    kinds = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    if cfg.under:
        under = gr.crossing_under(segs, ends, pairs, occupied, gcfg, cfg.under_gap)
        records += under
        kinds["crossing under"] = len(under)
    n_join = 0
    if cfg.join:                    # any gap still open after the junctions were read
        _, joins = joinmod.join_ends([s["points"] for s in segs], z, cfg.join_gap,
                                     np.radians(cfg.join_turn), cfg.join_min_z)
        for j in joins:
            ea = 2 * j["a"] + (0 if j["a_start"] else 1)
            eb = 2 * j["b"] + (0 if j["b_start"] else 1)
            if pairs.get(ea, -1) < 0 and pairs.get(eb, -1) < 0:
                pairs[ea], pairs[eb] = eb, ea
                n_join += 1
    chains = gr.chains(segs, pairs)
    h = gr.hierarchy(segs, chains, records, ends)
    if log:
        log(f"  junctions: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
            + f"; {n_merged} gaps repaired, {n_join} gap joins")
        log(f"  {len(segs)} segments -> {len(chains)} vessels")
    seg_chain = {s: ci for ci, ch in enumerate(chains) for s, _ in ch}
    out = []
    for ci, chain in enumerate(chains):
        C = gr.centreline(segs, chain)
        fits, seen = [], set()
        for s, _ in chain:                       # a piece split at a touch appears once
            key = id(segs[s]["fit"])
            if key not in seen:
                seen.add(key)
                fits.append(segs[s]["fit"])
        rr = [float(segs[s]["radius"]) for s, _ in chain]
        out.append({"chain": ci, "pieces": fits, "centrelines": [C], "centreline": C,
                    "radius_px": round(float(h["calibre"][ci]), 2),
                    "radius_max_px": round(float(np.max(rr)), 2),
                    "length_px": round(float(np.hypot(*np.diff(C, axis=0).T).sum()), 1),
                    "label": h["label"].get(ci), "strahler": h["strahler"].get(ci),
                    "parent_chain": h["parent"][ci], "branch_point": h["branch_node"][ci],
                    "anchor": (float(C[0][0]), float(C[0][1]))})
    # number them in hierarchy order: thickest trunk first, then its branches.
    # out is indexed by CHAIN here; ids are assigned, then every lookup goes
    # through id_of_chain, and only the returned copy is sorted by id.
    for new_id, (ci, depth) in enumerate(h["order"], start=1):
        out[ci]["id"] = new_id
        out[ci]["depth"] = depth
    id_of_chain = {ci: v["id"] for ci, v in enumerate(out)}
    for ci, v in enumerate(out):
        p_ = v["parent_chain"]
        v["parent"] = id_of_chain[p_] if p_ is not None else None
        v["children"] = [id_of_chain[c] for c in h["children"][ci]]
        v["junctions"] = []
    for r in records:
        if r["kind"] not in ("crossing", "crossing under", "bifurcation", "unresolved"):
            continue
        touching = sorted({id_of_chain[seg_chain[ends[e]["seg"]]] for e in r["ends"]
                           if ends[e]["seg"] in seg_chain})
        j = {k: val for k, val in r.items() if k not in ("ends", "pairs")}
        j["vessels"] = touching
        for ci, v in enumerate(out):
            if v["id"] in touching:
                v["junctions"].append(j)
    out.sort(key=lambda v: v["id"])
    return out


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
