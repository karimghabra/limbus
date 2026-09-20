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

from . import crystal as cryst
from . import evidence as ev
from . import fit as vfit
from . import graph as gr
from . import join as joinmod
from . import model
from . import ridges


@dataclass
class NetConfig:
    sigmas: tuple = ev.SIGMAS         # (kept for single-pass use)
    sigmas_large: tuple = ev.SIGMAS_LARGE   # scales of the large-vessel pass
    sigmas_small: tuple = ev.SIGMAS_SMALL   # scales of the small-vessel pass
    staged: bool = True               # search large vessels first, then small ones
    t_hi_large: float = 3.0           # the large pass has its own thresholds: a wide
    L_hi_large: int = 40              # vessel is strong, so completeness matters more
    t_lo_large: float = 1.0           # than caution here
    L_lo_large: int = 20
    min_len_large: float = 30.0
    drop_echoes: bool = False  # suppress wall echoes in the EVIDENCE (blunt: also eats
                               # thin vessels running beside a wide one). Off by default -
                               # the model fit removes echoes instead, by trimming whatever
                               # lies inside an accepted vessel's lumen, which is a physical
                               # boundary rather than a fixed radius around a coarse ridge.
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
    crystallize: bool = True   # grow each free end using that vessel's own profile
                               # and the direction of the local spectrum
    grow_seed_len: float = 80.0    # only vessels at least this long may seed growth
    grow_seed_depth: float = 0.03  # ... and at least this deep (peak absorbance)
    consolidate: bool = True   # join fragments only when diameter, darkness and
                               # direction agree, and drop a vessel traced twice
    attach: bool = True        # extend a branch that stops short of the vessel it leaves
    attach_gap: float = 30.0   # how far it may be extended (px)
    under: bool = True         # rejoin a vessel across the shadow of a wider one
    under_gap: float = 45.0    # widest such shadow crossed (px)
    node_tol: float = 10.0     # piece ends this close meet at one junction (px)
    evidence: str = "hessian"  # "hessian" or "orientation": one strength and one
                               # direction per pixel, or a plane per direction. At a
                               # right-angle crossing the Hessian falls to 0.12 of the
                               # vessel's own strength and the orientation stack to 1.00.
    trim_split_fits: bool = True  # a piece split at a touch carries its own fit, covering
                               # only the span it kept, instead of sharing the original
                               # with its other half (which made two vessels export the
                               # same geometry, and read the same kymograph)
    touch_tol: float = 8.0     # an end this close to another piece splits it (px)
    fit: bool = True
    fit_move: float = 0.0      # how far the fit may move the centreline (0 = keep the ridge)
    kappa: float = 0.0         # fit-gain required per free parameter
    min_depth: float = 0.015   # shallowest vessel accepted (peak absorbance)
    psf: float = 1.8           # optical blur (px), measured bound 2.1
    grow_cfg: object = None    # crystal.GrowConfig, or None for its defaults


def detect(A, valid, cfg=None, log=None):
    """Returns (vessels, model_image, z).

    Each vessel is a dict with id, label ("3.2"), parent, children, strahler,
    the fitted pieces it is made of, one continuous centreline, and the
    junctions along it."""
    cfg = cfg or NetConfig()
    if cfg.staged:
        large, small, z, _, sc = ridges.detect_staged(A, valid, cfg, log=log)
    else:
        z, ang, sc = ev.ridge_z(A, valid, cfg.sigmas, with_angle=True, with_scale=True)
        large, small = [], ridges.detect(z, ang, cfg.t_hi, cfg.L_hi, cfg.t_lo, cfg.L_lo,
                                         cfg.gap, cfg.min_len, reach=cfg.reach,
                                         scale=sc if cfg.drop_echoes else None, A=A)
    if log and not cfg.staged:
        log(f"  ridges: {len(small)} centrelines")
    H, W = A.shape
    # Order candidates by how much absorbance they carry: depth x length. The
    # centre of a wide vessel is its darkest line, so it is fitted before the
    # wall echoes beside it, and its lumen then trims them away. Ordering by
    # evidence z instead let a fine-scale wall echo be fitted first, and the
    # vessel came back with the radius of its own wall.
    # Large vessels are fitted first - every one of them, before any small
    # candidate - so a wide vessel is measured on its own centreline and its
    # lumen is marked before the small pass's lines beside it are considered.
    # Within a pass, the order is depth x length: the darkest, longest line
    # first.
    def strength(C):
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        L = float(np.hypot(*np.diff(C, axis=0).T).sum())
        return float(np.median(A[yi, xi])) * L

    cands = [(1e6 + strength(C), C) for C in large] + [(strength(C), C) for C in small]
    if cfg.fit:
        acc, mdl, rej, occ = vfit.run(A, valid, cands, psf=cfg.psf, kappa=cfg.kappa,
                                      min_depth=cfg.min_depth, max_move=cfg.fit_move, log=log)
    else:
        acc = [(C, None, None, {}) for _, C in cands]
        mdl, occ = None, np.zeros(A.shape, bool)
    pieces = []
    for P, R, D, rec in acc:
        C = model.catmull(np.array(P, float))[0] if R is not None else np.asarray(P, float)
        xi = np.clip(np.rint(C[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int), 0, H - 1)
        pieces.append({"points": C, "radius": float(np.median(R)) if R is not None else 2.0,
                       "darkness": float(np.median(A[yi, xi])),   # absorbance along it
                       "fit": (P, R, D, rec)})
    if not cfg.graph:
        return group([p["fit"] for p in pieces]), mdl, z
    return _thread(pieces, A, valid, z, occ, cfg, log), mdl, z


def _thread(pieces, A, valid, z, occupied, cfg, log=None):
    """Pieces -> vessels: split at touches, classify every junction, add the
    long-gap joins as further pairings, then walk the chains."""
    gcfg = gr.GraphConfig(node_tol=cfg.node_tol, touch_tol=cfg.touch_tol,
                          trim_split_fits=cfg.trim_split_fits)
    segs = list(pieces)
    n_dup = 0
    if cfg.consolidate:             # the same vessel found by both passes
        before = len(segs)
        segs = gr.drop_duplicates(segs, gcfg)
        n_dup = before - len(segs)
    n_merged = 0
    if cfg.join:                    # repair broken pieces before reading junctions
        before = len(segs)
        segs = gr.merge_gaps(segs, z, joinmod.join_ends, cfg.join_gap,
                             np.radians(cfg.join_turn), cfg.join_min_z, gcfg,
                             A if cfg.consolidate else None)
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
            + f"; {n_dup} duplicates dropped, {n_merged} gaps repaired, {n_join} gap joins")
        log(f"  {len(segs)} segments -> {len(chains)} vessels")
    seg_chain = {s: ci for ci, ch in enumerate(chains) for s, _ in ch}
    out = []
    for ci, chain in enumerate(chains):
        C = gr.centreline(segs, chain)
        prof = np.concatenate([np.full(len(np.asarray(segs[sg]["points"], float)),
                                       float(segs[sg]["radius"])) for sg, _ in chain])
        fits, seen = [], set()
        for s, _ in chain:                       # a piece split at a touch appears once
            for f in (segs[s].get("fits") or ([segs[s]["fit"]]
                                              if segs[s].get("fit") is not None else [])):
                if id(f) not in seen:
                    seen.add(id(f))
                    fits.append(f)
        rr = [float(segs[s]["radius"]) for s, _ in chain]
        dd = [float(segs[s]["fit"][2]) for s, _ in chain
              if segs[s].get("fit") is not None and segs[s]["fit"][2] is not None]
        out.append({"chain": ci, "pieces": fits, "centrelines": [C], "centreline": C,
                    "radius_profile": prof,
                    "radius_px": round(float(h["calibre"][ci]), 2),
                    "depth_abs": round(float(np.median(dd)), 4) if dd else 0.0,
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
    # Crystallisation runs LAST, on finished vessels. Growing before the
    # junctions are read means every grown tip lands on another vessel and
    # splits it, and the network comes apart: 52 vessels became 85 and the
    # longest fell from 941 px to 585. Growing afterwards extends what has
    # already been decided and leaves the topology alone.
    n_grown, grown_px = 0, 0.0
    if cfg.crystallize:
        lines = [np.asarray(v["centreline"], float) for v in out]
        for vi, v in enumerate(out):
            r = float(v["radius_px"])
            D = max(float(v.get("depth_abs", 0.0)), 0.01)
            # Only a vessel we are sure of may seed growth. Growing from
            # everything extends the detector's own false positives, and on a
            # vessel-free texture surrogate that more than doubled the
            # centreline while adding almost nothing on real data.
            if v["length_px"] < cfg.grow_seed_len or D < cfg.grow_seed_depth:
                continue
            for at_start in (True, False):
                P = np.asarray(v["centreline"], float)
                if len(P) < 3:
                    continue
                p0 = P[0] if at_start else P[-1]
                if any(float(np.hypot(*(L - p0).T).min()) <= gcfg.node_tol
                       for j, L in enumerate(lines) if j != vi):
                    continue                        # this end already meets a vessel
                add = cryst.grow_end(A, valid, p0, gr._tangent(P, at_start), r, D,
                                     occupied, cfg.grow_cfg, cfg.psf)
                if len(add) >= 2:
                    v.setdefault("grown_parts", []).append(np.vstack([p0, add]))
                    prof = np.asarray(v["radius_profile"], float)
                    edge = np.full(len(add), prof[0] if at_start else prof[-1])
                    v["centreline"] = np.vstack([add[::-1], P]) if at_start else np.vstack([P, add])
                    v["radius_profile"] = (np.concatenate([edge, prof]) if at_start
                                           else np.concatenate([prof, edge]))
                    v["centrelines"] = [v["centreline"]]
                    v["length_px"] = round(float(np.hypot(*np.diff(v["centreline"], axis=0).T).sum()), 1)
                    lines[vi] = v["centreline"]
                    n_grown += 1
                    grown_px += float(np.hypot(*np.diff(np.vstack([p0, add]), axis=0).T).sum())
        if log:
            log(f"  crystallised: {n_grown} ends grown, {grown_px:.0f} px added")

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
    """Per-vessel label image: each vessel's lumen carries its id.

    Painted from the vessel's own centreline and the radius measured along it,
    so the labels always match the geometry that is reported. Painting from
    the fitted pieces instead left gaps wherever two pieces had been threaded
    into one vessel. Longer vessels are painted first, so a short piece cannot
    overwrite them.
    """
    lab = np.zeros(shape, np.int32)
    H, W = shape
    for v in sorted(vessels, key=lambda v: -float(v.get("length_px", 0))):
        C = np.asarray(v.get("centreline", v["centrelines"][0]), float)
        if len(C) < 2:
            continue
        r = np.asarray(v.get("radius_profile", np.full(len(C), v.get("radius_px", 2.0))), float)
        if len(r) != len(C):
            r = np.interp(np.linspace(0, 1, len(C)), np.linspace(0, 1, len(r)), r)
        pad = int(np.ceil(r.max() + 3))
        x0 = max(0, int(C[:, 0].min()) - pad); x1 = min(W, int(C[:, 0].max()) + pad + 1)
        y0 = max(0, int(C[:, 1].min()) - pad); y1 = min(H, int(C[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        seeds = np.ones((y1 - y0, x1 - x0), np.uint8)
        xi = np.clip(np.rint(C[:, 0]).astype(int) - x0, 0, x1 - x0 - 1)
        yi = np.clip(np.rint(C[:, 1]).astype(int) - y0, 0, y1 - y0 - 1)
        seeds[yi, xi] = 0
        dist, idx = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        # map each pixel's nearest seed to that centreline sample's radius
        table = np.zeros(int(idx.max()) + 1, np.float32)
        table[idx[yi, xi]] = r.astype(np.float32)
        lumen = dist <= table[idx]
        sub = lab[y0:y1, x0:x1]
        sub[(sub == 0) & lumen] = v["id"]
    return lab
