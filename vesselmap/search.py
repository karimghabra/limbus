"""Energy search: the fewest vessels that still render the image.

``consolidate_map`` (consolidate.py) joins segments that pass fixed
geometric tests (turn <= 40 deg, calibre and blur within bounds, an
unambiguous pair at a fork) and a per-link image test.  What it cannot join
stays in pieces, and nothing removes a vessel that is too weak to be one.
``search_map`` goes further by minimising one explicit objective over the
structure of the map:

    E(M) = NLL(image | M) + prior(M) + sum_v (tau * lam_vessel + price * L_v) + Phi(M)

* NLL and prior are exactly the renderer's score (render.py), with the
  windowed calibre prior of consolidation (a long vessel may taper).
* Existence.  Every vessel pays a fixed ``tau * lam_vessel``, and every
  pixel of centreline pays ``price``, so a vessel must explain enough of
  the image per pixel to be kept.  ``tau`` is the data temperature, the
  reduced chi-square of the map's residual: structured misfit inflates NLL
  differences by about that factor.  The price is set by the *texture
  null*: tissue texture makes bright ridges as often as dark ones, and
  vessels are only dark, so 'anti-vessels' fitted to the bright ridges of
  the residual show how much a dark trace of texture explains per pixel.
  A vessel that explains no more than that (``null_quantile``) is not told
  apart from texture.  (At least ``tau * lam_length``.)
* Consolidation.  No cost per vessel can make fragments join without also
  deleting them: for any cost c, c(a) + c(b) - c(a + b) <= c(b), so
  deleting a piece always saves at least what joining it saves.  The
  pieces of a vessel are instead told by their ends: Phi charges vessel
  ends that face each other (the pairs a join would try), each pair
  g * min(om, om'), over a maximum-weight matching of the ends (so an end
  pays once, and at a fork or crossing any buildable join pays about the
  same: the data picks the partner).  om = lam_frag times the evidence of
  the end's last frag_len px above the price (end_evidence; at most
  tau * lam_cap), so an end of texture-grade junk earns a join nothing, and
  deleting a vessel releases at most lam_frag of its evidence: a vessel
  survives whenever (1 - lam_frag) of what it explains beyond the price
  exceeds tau * lam_vessel.  g in [0, 1] is how clearly the ends face each
  other: across a gap only up to frag_gap (farther apart, facing ends are
  as often two vessels as one: the data decides those joins alone), and
  overlapping ends only when each lies on the other's line.  Phi depends
  on the map alone (no history), so the search still samples one E, and
  the fits do not see it (in the gradient it would dim fragment ends).

Moves.  Each move is followed by a short gradient fit of what it changes:

    join     two vessel ends into one spline (ends that met at a node, a
             bridged gap, or overlapping ends of a vessel found twice)
    delete   a vessel (vessels duplicating it are re-fitted to take over)
    split    a vessel at a sharp bend, where the trace may switch vessels
    reroute  at a branch point, which arm continues the parent: an end
             touching a vessel takes over the vessel's far part, and the
             near part ends there
    revive   a vessel deleted earlier (so every move can be undone)

Exact local energy.  A move changes the image only near the vessels it
touches, so its energy change is computed on their footprint, rendering
only those vessels over the fixed rest of the model (the other vessels'
optical density, halo included, and the background).  A join, split or
reroute is fitted cheaply: only the parameters within ``focus_radius`` of
the junction, on a window around it (joined vessels are rebuilt
faithfully, so elsewhere they render as their pieces did).  The fitted
result is then scored on the whole footprint, so dE is exactly the change
the move makes.  The change of Phi is exact too (EndGraph.delta: only
the matching components a move touches are matched again), and computed
afresh at every step, as it depends on ends up to join_gap away.

Search.  Rejection-free annealing: at every step every candidate move is
scored and one is drawn with probability proportional to exp(-dE / T),
staying put (dE = 0) included; T starts at t_start * tau * t_scale.  A
data score is cached and corrected exactly for changes around it, and
re-fitted (in parallel worker processes) when the correction is large.
At a high temperature the search can leave a local minimum, e.g. join two
pieces whose join only pays once a duplicate between them is gone.  As T falls it becomes steepest descent,
and ends when no move lowers E.  A joint fit of every vessel and the
background follows, then a last greedy pass.

The result uses the map's usual representation (network.py): one edge per
vessel, and a branch ends at a node its parent passes through
(``info["through"]``), so ``to_segments`` / ``to_digraph`` still give the
segment graph.  Like the rest of vesselmap, only the intensities of one
image are used.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import spline as sp
from .fit import MapConfig, optimize
from .image import Prepared, prepare
from .network import (A_MIN, PROFILE_SPACING, R_MIN, S_MIN, Edge, Node, VesselNetwork,
                      _fit_edge_params, despike)
from .render import NetworkModel, _cv_blur

BG_SPACING = 64.0


@dataclass
class SearchConfig:
    lam_vessel: float = 50.0        # existence cost of one vessel (x tau)
    lam_length: float = 1.0         # cost per px of centreline (x tau), at least ...
    texture_null: bool = True       # ... what background texture alone explains per px:
    null_quantile: float = 0.9      # this quantile of the evidence per px of 'anti-vessels'
    null_bands: tuple = ((1.2, 2.5), (2.5, 5.0), (5.0, 10.0))   # fitted to bright ridges
    lam_frag: float = 0.4           # fragmentation: a vessel end facing another pays this
                                    # share of its evidence (Phi, see the module docstring) ...
    lam_cap: float = 400.0          # ... at most this (x tau) per end ...
    lam_pair: float = 0.0           # ... plus this (x tau), up to the end's own evidence
    frag_len: float = 30.0          # the evidence of an end: its last frag_len px (<= L / 2)
    frag_margin: float = 0.25       # a facing pair counts fully this far inside the join test,
    frag_gap: float = 6.0           # and across a gap up to this (px), not at all beyond twice it
    frag_mode: str = "match"        # "match": Phi over a matching of facing ends; "ends":
                                    # every end pays (an ablation)
    strict_forks: bool = False      # three vessel ends meeting at one point are a fork (each
                                    # vessel ends there), not a vessel in pieces: Phi does not
                                    # charge pairs among them (fork_clusters); and an end on
                                    # another vessel's interior pays tau * lam_vessel (that
                                    # vessel is two vessels there), so E counts vessels as the
                                    # strict convention does
    tau: float = 0.0                # data temperature; <= 0: reduced chi-square of the map
    t_scale: float = 150.0          # energy scale (x tau) of the schedule and the cache:
    t_start: float = 0.1            # first temperature (x tau * t_scale)
    hot_steps: float = 0.5          # annealing steps (x number of input edges) before T = 0
    hopeless: float = 10.0          # a move scored worse than this (x tau * t_scale), even
                                    # with all the Phi it may release, is not re-fitted while
                                    # its vessels exist
    sens_budget: float = 0.5        # nats a cached score's correction may lose to sparsity
    refit_tol: float = 0.05         # a cached score is corrected for changes around the move
                                    # (exact for the fitted vessels); beyond this (x tau *
                                    # t_scale) the move is re-fitted, as its fit may adapt
    local_iters: int = 40           # gradient steps after a move
    focus_radius: float = 30.0      # a join / split / reroute re-fits the vessel within this
                                    # distance (px) of the junction; the rest stays as it was
    refine_below: float = 2.0       # a move scored within +-this (x tau * t_scale) of 0 is
                                    # fitted again, longer (a join / split / reroute also wider: a
    refine_iters: int = 150         # bend at a junction needs room to relax; the pieces it
    refine_radius: float = 60.0     # replaces were fitted for hundreds of steps)
    relax_radius: float = 0.0       # a move of relax_kinds scored above 0 but below
    relax_iters: int = 120          # relax_below (x tau * t_scale) is scored again with the
    relax_below: float = 1.0        # other vessels within this distance (px) re-fitted too,
    relax_kinds: tuple = ("join", "reroute", "delete", "split", "swap", "extend")
                                    # before and after alike (a branch whose parent changes,
                                    # a crossing vessel, a vessel an end grows into, may have
                                    # to move for the move to pay); relax_radius 0: off
    lr_pos: float = 0.1
    lr_prof: float = 0.1            # a wide blurred vessel's calibre / blur / contrast must move
                                    # far within a move's short fit (0.03 left a duplicate's
                                    # delete 600 above 0 where a long fit puts it below)
    lr_prof_global: float = 0.03    # ... while the joint fit of every vessel keeps to this
    extend_turn: float = 20.0       # an extend trace turns at most this much a 2 px step (deg),
    extend_first_turn: float = 20.0     # its first step this much (a hairpin needs ~150),
    extend_through: bool = False    # and goes on through a vessel it reaches (else ends there)
    join_gap: float = 40.0          # largest gap bridged (px) ...
    join_gap_factor: float = 6.0    # ... and at most this many vessel widths
    join_cone_deg: float = 60.0     # a gap is bridged only roughly straight ahead
    max_turn_deg: float = 120.0     # joins turning more sharply are not tried
    trim_max: float = 25.0          # overlap of two ends that a join may trim
    split_turn_deg: float = 35.0    # bends sharper than this are split candidates
    init_iters: int = 0             # joint fit before the search (unfitted input)
    global_iters: int = 150         # joint fit of everything after the search
    attach_tol: float = 1.5         # px beyond a vessel's r + s where a branch end attaches
    attach_max: float = 15.0        # ... unless attaching it (fitted) changes NLL + prior by
                                    # more than this (x tau * lam_vessel): the end then stays
                                    # free (frame 20: median cost 150-200 nats, a tail of 10%
                                    # beyond 2000 that makes a third of the damage)
    through_iters: int = 40         # joint fit once branch ends sit on their parents
    calibre_window: int = 4         # profile knots the calibre prior averages over (as in
                                    # consolidation: a long vessel may taper)
    local_bg: bool = False          # a move also re-fits the background near it (a correction
                                    # on the window, before and after alike, in closed form):
                                    # a vessel that explains background texture can then go
    births: bool = False            # propose new vessels where the map leaves dark ridges
    birth_bands: tuple = ((0.8, 1.2), (1.2, 2.5), (2.5, 5.0), (5.0, 10.0))   # (px) of
    birth_z: float = 4.0            # the residual (ridge z above this), and faint paths
    birth_faint: bool = True        # traced along the residual (faint.trace_faint)
    birth_rounds: int = 1           # greedy passes with fresh proposals after annealing
    greedy_batch: bool = True       # at T = 0 take every improving move on a window of its
                                    # own in one step (exact: they do not interact)
    min_gain: float = 3.0           # at T = 0 a move must lower E by more than this (nats): less
                                    # is the local fit's own noise (a trim whose end the fit
                                    # slides back out is a re-fit, not a change)
    lam_bend: float = 0.0           # bending prior weight; 0: the map's (MapConfig)
    lam_bg: float = 0.0             # background smoothness prior weight; 0: the map's
    bg_spacing: float = 0.0         # background grid spacing (px); 0: the map's
    workers: int = 0                # processes scoring moves in parallel; 0: one per CPU
    seed: int = 0
    verbose: bool = True


# ------------------------------------------------------------------ helpers
def edge_samples(e: Edge, spacing: float = 1.0) -> dict:
    """Samples of an edge evenly spaced in arclength (like
    VesselNetwork.sample, for edges that are not in a network)."""
    n = len(e.ctrl)
    m0 = max(128, 16 * n)
    xy_d = sp.design(n, m0) @ e.ctrl
    prof_d = sp.design(len(e.r), m0) @ np.stack([e.r, e.s, e.a], 1)
    s_d = sp.arclength(xy_d)
    L = float(s_d[-1])
    m = sp.n_samples_for_length(L, spacing)
    if L <= 0:
        xy = np.repeat(xy_d[:1], m, 0)
        pr = np.repeat(prof_d[:1], m, 0)
    else:
        q = np.linspace(0.0, L, m)
        f = lambda a: np.stack([np.interp(q, s_d, a[:, j]) for j in range(a.shape[1])], 1)
        xy, pr = f(xy_d), f(prof_d)
    tan = np.gradient(xy, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    return dict(xy=xy, tan=tan, s_arc=sp.arclength(xy), r=np.maximum(pr[:, 0], 0.3),
                s=np.maximum(pr[:, 1], 0.6), a=np.maximum(pr[:, 2], 1e-3), L=L)


def _flip(smp):
    out = {k: (v[::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in smp.items()}
    out["tan"] = -out["tan"]
    out["s_arc"] = out["L"] - out["s_arc"]
    return out


def _direction(xy, i, forward=True, look=4):
    """Unit direction of a polyline at index i, averaged over a few points."""
    j = min(len(xy) - 1, i + look) if forward else max(0, i - look)
    v = xy[j] - xy[i] if forward else xy[i] - xy[j]
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = xy[-1] - xy[0]
        n = np.linalg.norm(v) + 1e-9
    return v / n


def _faces(p1, t1, w1, p2, t2, w2, cfg, gap=True) -> bool:
    """The join test of end 1 (position, outward tangent, width) towards
    end 2, as join_moves applies it: the ends are close, or face each other
    across a gap (roughly straight ahead), or overlap.  (gap: also test the
    distance, which join_moves leaves to its ball query.)"""
    v = p2 - p1
    d = float(np.linalg.norm(v))
    if gap and d > min(cfg.join_gap, max(8.0, cfg.join_gap_factor * w1)):
        return False
    if np.dot(t1, t2) > 0.5:                        # both ends point the same way
        return False
    if d <= max(2.0, 0.5 * (w1 + w2)):
        return True
    u = v / d
    cone = math.cos(math.radians(cfg.join_cone_deg))
    ahead = np.dot(t1, u) > cone and np.dot(t2, -u) > cone
    overlap = np.dot(t1, t2) < -0.7 and d <= cfg.trim_max and \
        abs(t1[0] * v[1] - t1[1] * v[0]) <= max(w1, w2) + 2.0
    return bool(ahead or overlap)


def face_weight(a, b, cfg):
    """Continuity weight g in [0, 1] of two vessel ends a, b = (position,
    outward tangent, width) that face each other (the join test passes in
    either direction), None when they do not.  g is how far inside the
    test the pair lies, over frag_margin: 0 at its edge, 1 once the pair is
    well inside, so a small refit never switches a pair's weight on or off
    at once.  Two relations are held to more than the join test, which
    only proposes: ends across a gap count fully up to frag_gap and not at
    all beyond twice it (facing ends farther apart are as often two vessels
    as one), and overlapping ends must each lie on the other's line and
    have passed each other (a vessel found twice), not merely one of them."""
    (p1, t1, w1), (p2, t2, w2) = a, b
    if not (_faces(p1, t1, w1, p2, t2, w2, cfg) or _faces(p2, t2, w2, p1, t1, w1, cfg)):
        return None
    fm = cfg.frag_margin
    v = p2 - p1
    d = float(np.linalg.norm(v))
    c12 = float(np.dot(t1, t2))
    near = max(2.0, 0.5 * (w1 + w2))
    G = min(cfg.join_gap, max(8.0, cfg.join_gap_factor * max(w1, w2)))
    m_ahead = m_ov = -1.0
    if d > 0:
        u = v / d
        cone = math.cos(math.radians(cfg.join_cone_deg))
        m_ahead = (min(float(np.dot(t1, u)), float(np.dot(t2, -u))) - cone) / (1.0 - cone)
        cross = max(abs(t1[0] * v[1] - t1[1] * v[0]), abs(t2[0] * v[1] - t2[1] * v[0]))
        m_ov = min((-0.7 - c12) / 0.3, 1.0 - d / cfg.trim_max,
                   1.0 - cross / (max(w1, w2) + 2.0))
        D0 = max(cfg.frag_gap, near)
        taper = fm * (2.0 * D0 - d) / D0
        m_ahead = min(m_ahead, taper)
        # ends that have not passed each other (one lies ahead of the
        # other) are a gap: tapered, blending in over a width so that
        # the weight does not jump where they just pass
        ahead_by = max(float(np.dot(t1, v)), float(np.dot(t2, -v)))
        if ahead_by > 0:
            m_ov = min(m_ov, max(taper, m_ov - ahead_by / (max(w1, w2) + 2.0)))
    m = min(1.0 - d / G, (0.5 - c12) / 1.5, max(1.0 - d / near, m_ahead, m_ov))
    return float(min(1.0, max(0.0, m) / fm))


def _mwm(edges):
    """Value of a maximum-weight matching of a graph given as edges
    (i, j, w > 0): a bitmask recursion for components of up to 12 nodes,
    networkx beyond."""
    if not edges:
        return 0.0
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in edges:
        parent[find(i)] = find(j)
    comps = {}
    for e in edges:
        comps.setdefault(find(e[0]), []).append(e)
    total = 0.0
    for ce in comps.values():
        nodes = sorted({x for i, j, _ in ce for x in (i, j)})
        if len(nodes) <= 12:
            pos = {x: q for q, x in enumerate(nodes)}
            adj = [dict() for _ in nodes]
            for i, j, w in ce:
                a, b = pos[i], pos[j]
                if w > adj[a].get(b, 0.0):
                    adj[a][b] = adj[b][a] = w
            memo = {0: 0.0}

            def f(mask):
                if mask in memo:
                    return memo[mask]
                i = (mask & -mask).bit_length() - 1
                rest = mask & ~(1 << i)
                best = f(rest)
                for j, w in adj[i].items():
                    if rest >> j & 1:
                        best = max(best, w + f(rest & ~(1 << j)))
                memo[mask] = best
                return best

            total += f((1 << len(nodes)) - 1)
        else:
            import networkx as nx
            G = nx.Graph()
            for i, j, w in ce:
                if w > G.get_edge_data(i, j, {}).get("weight", 0.0):
                    G.add_edge(i, j, weight=w)
            total += float(sum(G[i][j]["weight"] for i, j in nx.max_weight_matching(G)))
    return total


@torch.no_grad()
def end_evidence(m: NetworkModel, hw: float, frag_len: float, entries=None):
    """(S0, S1, l) for every edge of a model, in the order of m.eids: the
    signal energy 0.5 * sum w ((1 - hw) m)^2 of the pixels whose nearest
    centreline sample lies within arclength l = min(frag_len, L / 2) of the
    start (S0) and of the end (S1).  It is the first term of the NLL gain
    of those pixels (render._entry_gains), what the stretch explains when
    its amplitude is fitted, and depends on the vessel's own rendering
    only."""
    n = len(m.eids)
    if n == 0:
        return []
    ent = m.vessel_entries() if entries is None else torch.as_tensor(entries)
    if len(m.e_samp) == 0:
        return [(0.0, 0.0, 0.0)] * n
    arc, Ledge = m.samples()[5:7]
    j, k = m.e_samp, m.e_edge
    w = m.weight.reshape(-1)[m.e_pix]
    v = 0.5 * w * ((1 - hw) * ent) ** 2 * m.stride ** 2
    le = torch.clamp(0.5 * Ledge, max=frag_len)
    a, L = arc[j], Ledge[k]
    # a vessel shorter than 2 frag_len is split in halves by sample index
    # (the middle sample shared), not by arclength, whose rounding depends
    # on the model the vessel is rendered in
    r = (j - m.samp_first_t[k]).to(v.dtype)
    n1 = (m.samp_last_t[k] - m.samp_first_t[k]).to(v.dtype)
    h0 = torch.where(2 * r < n1, 1.0, torch.where(2 * r == n1, 0.5, 0.0)).to(v.dtype)
    short = 0.5 * L <= frag_len
    w0 = torch.where(short, h0, (a <= frag_len).to(v.dtype))
    w1 = torch.where(short, 1.0 - h0, (L - a <= frag_len).to(v.dtype))
    S0 = torch.zeros(n, dtype=v.dtype).index_add(0, k, v * w0)
    S1 = torch.zeros(n, dtype=v.dtype).index_add(0, k, v * w1)
    return [(float(S0[q]), float(S1[q]), float(le[q])) for q in range(n)]


def fork_clusters(rows) -> np.ndarray:
    """Per row (vessel id, position, tangent, width, ...): a fork id, or -1.
    Ends of different vessels within max(3, (w + w') / 2) px of each other are
    linked; a connected group of exactly three ends of three vessels is a fork
    (each vessel ends at the branch point).  Two ends of one vessel, or four
    or more ends (a crossing cut at its centre), are not."""
    n = len(rows)
    lab = np.full(n, -1)
    if n < 3:
        return lab
    P = np.array([r[1] for r in rows]).reshape(-1, 2)
    wmax = max(float(r[3]) for r in rows)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cKDTree(P).query_pairs(max(3.0, wmax)):
        if rows[a][0] != rows[b][0] and \
                np.linalg.norm(P[a] - P[b]) <= max(3.0, 0.5 * (rows[a][3] + rows[b][3])):
            parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    for g in groups.values():
        if len(g) == 3 and len({rows[i][0] for i in g}) == 3:
            lab[g] = g[0]
    return lab


class EndGraph:
    """The facing-end graph of a map and its fragmentation term Phi.

    Nodes are vessel ends not at the image border, as rows (vessel id,
    position, outward tangent, width, om, end); edges are the pairs
    join_moves would try (the join test passes in either direction; ends
    of one vessel never pair), weighted g * min(om, om').  Phi is the value
    of a maximum-weight matching (mode "match"), or the sum of om over all
    ends (mode "ends").  Components of the positive-weight graph are
    matched separately, so the change of Phi by a move is computed on the
    components it touches alone (delta)."""

    def __init__(self, rows, cfg):
        self.cfg, self.rows = cfg, rows
        n = len(rows)
        self.by_vid, self.by_end = {}, {}
        for i, r in enumerate(rows):
            self.by_vid.setdefault(r[0], []).append(i)
            self.by_end[(r[0], r[5])] = i
        self.tree = cKDTree(np.array([r[1] for r in rows]).reshape(-1, 2)) if n else None
        self.adj = [dict() for _ in range(n)]
        self.n_facing = 0
        self.fork = fork_clusters(rows) if getattr(cfg, "strict_forks", False) else np.full(n, -1)
        if n:
            for a, b in sorted(self.tree.query_pairs(cfg.join_gap)):
                if rows[a][0] == rows[b][0]:
                    continue
                if self.fork[a] >= 0 and self.fork[a] == self.fork[b]:
                    continue                        # three vessels ending at a branch point
                g = face_weight(rows[a][1:4], rows[b][1:4], cfg)
                if g is None:
                    continue
                self.n_facing += 1
                w = g * min(rows[a][4], rows[b][4])
                if w > 0:
                    self.adj[a][b] = self.adj[b][a] = w
        self.comp = list(range(n))
        for a in range(n):                          # components: label propagation
            if self.comp[a] != a:
                continue
            stack = [a]
            while stack:
                x = stack.pop()
                for y in self.adj[x]:
                    if self.comp[y] != a:
                        self.comp[y] = a
                        stack.append(y)
        self.members = {}
        for i, c in enumerate(self.comp):
            self.members.setdefault(c, []).append(i)
        self.value = {c: _mwm(self._edges(ms)) for c, ms in self.members.items()
                      if len(ms) > 1}
        self._matched = None
        self.total = sum(r[4] for r in rows) if cfg.frag_mode == "ends" else \
            float(sum(self.value.values()))

    def _edges(self, nodes, gone=()):
        s = set(nodes)
        return [(i, j, w) for i in nodes if i not in gone
                for j, w in self.adj[i].items() if j > i and j in s and j not in gone]

    @property
    def matched(self):
        """Number of pairs in the maximum-weight matching (for the logs)."""
        if self._matched is None:
            import networkx as nx
            n = 0
            for ms in self.members.values():
                if len(ms) > 1:
                    G = nx.Graph()
                    G.add_weighted_edges_from(self._edges(ms))
                    n += len(nx.max_weight_matching(G))
            self._matched = n
        return self._matched

    def weight(self, a, b):
        """Weight of the pair of ends a, b = (vessel id, end), 0 if none."""
        i, j = self.by_end.get(a), self.by_end.get(b)
        return 0.0 if i is None or j is None else self.adj[i].get(j, 0.0)

    def bound(self, vids):
        """-dPhi of any move removing the vessels vids is at most this.  With
        strict forks, removing an end can make the ends near it a fork (no
        longer charged), so their om counts too."""
        idx = {i for v in vids for i in self.by_vid.get(v, [])}
        if getattr(self.cfg, "strict_forks", False) and idx and self.tree is not None:
            for i in list(idx):
                idx.update(self.tree.query_ball_point(self.rows[i][1], self.cfg.join_gap))
        return float(sum(self.rows[i][4] for i in idx))

    def _strict_delta_rows(self, gone, new):
        """With strict forks: the existing ends whose fork cluster a change
        alters (they are removed and re-added, their pairs recomputed), and
        the clusters after the change, keyed by row identity (an existing
        index, or n0 + position in new + re-added)."""
        n0 = len(self.rows)
        keep = [i for i in range(n0) if i not in gone]
        post = fork_clusters([self.rows[i] for i in keep] + list(new))
        ident = keep + [n0 + q for q in range(len(new))]
        members_post, members_pre = {}, {}
        for x, l in zip(ident, post):
            if l >= 0:
                members_post.setdefault(int(l), set()).add(x)
        for i in range(n0):
            if self.fork[i] >= 0:
                members_pre.setdefault(int(self.fork[i]), set()).add(i)
        lab_post = {x: int(l) for x, l in zip(ident, post)}
        pre_of = lambda i: frozenset(members_pre[self.fork[i]]) if self.fork[i] >= 0 else frozenset([i])
        post_of = lambda x: frozenset(members_post[lab_post[x]]) if lab_post[x] >= 0 else frozenset([x])
        affected = [i for i in keep if pre_of(i) != post_of(i)]
        # re-added rows get identities after the new ones; their cluster label carries over
        readd = {i: n0 + len(new) + q for q, i in enumerate(affected)}
        label = {readd.get(x, x): l for x, l in lab_post.items()}
        return affected, label

    def delta(self, gone, new):
        """Change of Phi when the rows `gone` (indices) are removed and the
        rows `new` added (their vessel ids distinct from the current ones).
        Exact: a component that loses no node and gains no edge keeps its
        matching value."""
        gone = set(gone)
        if self.cfg.frag_mode == "ends":
            return sum(r[4] for r in new) - sum(self.rows[i][4] for i in gone)
        n0 = len(self.rows)
        new = list(new)
        exempt = lambda x, y: False
        if getattr(self.cfg, "strict_forks", False):
            affected, label = self._strict_delta_rows(gone, new)
            gone |= set(affected)
            new += [self.rows[i] for i in affected]
            exempt = lambda x, y: label.get(x, -1) >= 0 and label.get(x) == label.get(y)
        comps = {self.comp[i] for i in gone}
        add = []
        for q, r in enumerate(new):
            if r[4] <= 0:
                continue
            if self.tree is not None:
                for i in self.tree.query_ball_point(r[1], self.cfg.join_gap):
                    if i in gone or self.rows[i][4] <= 0 or self.rows[i][0] == r[0] or exempt(i, n0 + q):
                        continue
                    g = face_weight(self.rows[i][1:4], r[1:4], self.cfg)
                    if g:
                        add.append((i, n0 + q, g * min(self.rows[i][4], r[4])))
                        comps.add(self.comp[i])
            for q2 in range(q + 1, len(new)):
                r2 = new[q2]
                if r2[0] == r[0] or r2[4] <= 0 or exempt(n0 + q, n0 + q2) or \
                        np.linalg.norm(r2[1] - r[1]) > self.cfg.join_gap:
                    continue
                g = face_weight(r[1:4], r2[1:4], self.cfg)
                if g:
                    add.append((n0 + q, n0 + q2, g * min(r[4], r2[4])))
        if not comps and not add:
            return 0.0
        nodes = [i for c in comps for i in self.members[c]]
        after = _mwm(self._edges(nodes, gone) + add)
        return after - sum(self.value.get(c, 0.0) for c in comps)


def join_path(A: dict, B: dict, trim_max: float, max_turn_deg: float, spacing=0.7):
    """Dense path of the vessel formed by A (joining end last) and B
    (joining end first): A's tail and B's head are cut where they come
    closest (so overlapping ends of a duplicate are trimmed), and a gap is
    bridged by a cubic Hermite curve that leaves A and enters B along their
    own directions.  Returns (xy, r, s, a) or None when the join would turn
    back on itself."""
    sA, sB = A["s_arc"], B["s_arc"]
    tail = np.flatnonzero(sA >= sA[-1] - trim_max)
    head = np.flatnonzero(sB <= trim_max)
    D = np.linalg.norm(A["xy"][tail][:, None] - B["xy"][head][None], axis=2)
    cost = D + 0.15 * ((sA[-1] - sA[tail])[:, None] + sB[head][None])
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    ia, jb = int(tail[i]), int(head[j])
    if ia < 1 or jb > len(B["xy"]) - 2:
        return None
    pA, pB = A["xy"][ia], B["xy"][jb]
    tA = _direction(A["xy"], ia, forward=False)
    tB = _direction(B["xy"], jb, forward=True)
    if np.dot(tA, tB) < math.cos(math.radians(max_turn_deg)):
        return None
    h = float(np.linalg.norm(pB - pA))
    if h > 1.5:                          # a real gap (not two samples of one point)
        c = (pB - pA) / h
        if np.dot(tA, c) < -0.2 or np.dot(tB, c) < -0.2:     # the bridge would double back
            return None
    nb = int(math.ceil(h / spacing))
    a1, a2 = A["xy"][:ia + 1], B["xy"][jb:]
    if nb > 1:
        t = np.linspace(0, 1, nb + 1)[1:-1]
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        br = (h00[:, None] * pA + h10[:, None] * h * tA +
              h01[:, None] * pB + h11[:, None] * h * tB)
        xy = np.concatenate([a1, br, a2], 0)
    else:
        xy = np.concatenate([a1, a2], 0)
    r, s, a = _join_profiles(A, ia, B, jb, nb)
    xy, r, s, a = despike(xy, r, s, a)
    return xy, r, s, a


def fit_vessel(xy, r, s, a, spacing):
    """Spline parameters of a vessel assembled from pieces.  Everything is
    fitted faithfully (the centreline, and the profiles in the log domain
    on the dense knots of a faithful re-fit), so that away from the
    junction the vessel renders as its pieces did: a join changes the image
    only near the junction.  The profile values within one vessel width of
    each piece's free end are replaced beforehand, because the end cap makes
    them unreliable (see `_join_profiles`)."""
    ctrl, _, _, _ = _fit_edge_params(xy, r, s, a, spacing, faithful=True)
    arc = sp.arclength(np.asarray(xy, float))
    L = float(arc[-1])
    n_p = sp.n_ctrl_for_length(L, PROFILE_SPACING / 2.5, minimum=2)
    m = max(4 * n_p, int(np.ceil(L / 0.5)) + 1)
    q = np.linspace(0, L, m) if L > 0 else np.zeros(m)
    out = []
    for vals, lo in ((r, R_MIN), (s, S_MIN), (a, A_MIN)):
        lv = np.log(np.maximum(np.asarray(vals, float), lo))
        vu = np.interp(q, arc, lv) if L > 0 else np.full(m, lv.mean())
        c = sp.fit_ctrl(vu, n_p, smooth=1e-4, pin_ends=False)[:, 0]
        out.append(np.maximum(np.exp(c), lo))
    return ctrl, out[0], out[1], out[2]


def _join_profiles(A, ia, B, jb, nb):
    """Profile samples (r, s, a) along a joined path: A up to ia, a bridge of
    nb - 1 points, B from jb.  Within one vessel width of the joined ends
    the fitted profiles are distorted by the end caps, so there they are
    replaced by values from further inside, interpolated across the bridge."""
    out = []
    for k in ("r", "s", "a"):
        a1 = A[k][:ia + 1].copy()
        a2 = B[k][jb:].copy()
        wa = float(A["r"][ia] + 2 * A["s"][ia])
        wb = float(B["r"][jb] + 2 * B["s"][jb])
        sa = A["s_arc"][:ia + 1]
        sb = B["s_arc"][jb:] - B["s_arc"][jb]
        ka = np.flatnonzero(sa <= sa[-1] - wa)
        kb = np.flatnonzero(sb >= wb)
        va = a1[ka[-1]] if len(ka) else np.median(a1)
        vb = a2[kb[0]] if len(kb) else np.median(a2)
        if len(ka):
            a1[ka[-1] + 1:] = va
        if len(kb):
            a2[:kb[0]] = vb
        t = np.linspace(0, 1, nb + 1)[1:-1] if nb > 1 else np.zeros(0)
        out.append(np.concatenate([a1, va * (1 - t) + vb * t, a2]))
    return out


def _link_pos(L):
    for k in ("xy", "xy0"):
        if L.get(k) is not None:
            return np.asarray(L[k], float)[:2]
    return None


def _dedup_links(links):
    seen, out = set(), []
    for L in links:
        key = json.dumps(L, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(L)
    return out


def _split_links(links, pieces, cut, tol=8.0):
    """Link records of a vessel cut into pieces (sample arrays) at the point
    cut: a record within tol of the cut records the link the cut undoes and
    is dropped; every other record goes to the one piece nearest to it
    (records without a position go to the first piece)."""
    out = [[] for _ in pieces]
    trees = [cKDTree(xy) for xy in pieces]
    for L in links:
        p = _link_pos(L)
        if p is None:
            out[0].append(L)
            continue
        if np.linalg.norm(p - np.asarray(cut, float)) <= tol:
            continue
        out[int(np.argmin([t.query(p)[0] for t in trees]))].append(L)
    return out


def _with_links(info, links):
    out = dict(info)
    if links:
        out["links"] = links
    else:
        out.pop("links", None)
    return out


def _merge_info(infos):
    info = {}
    bands = [i["band"] for i in infos if i.get("band")]
    if bands:
        info["band"] = [min(b[0] for b in bands), max(b[1] for b in bands)]
    info["consolidated_from"] = sorted({f for i in infos for f in i.get("consolidated_from", [])})
    info["spacing"] = min(i.get("spacing", 12.0) for i in infos)
    links = _dedup_links([L for i in infos for L in i.get("links", [])])
    if links:
        info["links"] = links
    return info


def explode(net: VesselNetwork) -> VesselNetwork:
    """Copy of net in which every edge is a free vessel with its own two end
    nodes and no through nodes.  The search works on vessels alone; which
    vessel a branch leaves is worked out again at the end (`to_through`)."""
    out = net.copy()
    for e in out.edges.values():
        e.info.pop("through", None)
    out._touch()
    for nid in [n for n in out.nodes if not out.incident(n)]:
        del out.nodes[nid]
    for nid in list(out.nodes):
        inc = out.incident(nid)
        for eid, end in inc[1:]:
            new = out.add_node(*out.nodes[nid].xy)
            e = out.edges[eid]
            if end == 0:
                e.u = new
            else:
                e.v = new
        out._touch()
    for eid, e in out.edges.items():
        e.info.setdefault("consolidated_from", [int(eid)])
        e.info.pop("gain", None)
    return out


# ------------------------------------------------------------------ search
class VesselSearch:
    """State of the search: the network (one edge per vessel, private end
    nodes), the fixed background image and the running sharp-core vessel
    image V = sum of every vessel's rendered patch."""

    def __init__(self, net: VesselNetwork, P: Prepared, cfg: SearchConfig):
        self.cfg = cfg
        self.P = P
        self.H, self.W = P.shape
        self.net = explode(net)
        self.priors = dict(MapConfig().priors(), calibre_window=cfg.calibre_window)
        if cfg.lam_bend > 0:
            self.priors["lam_bend"] = cfg.lam_bend
        if cfg.lam_bg > 0:
            self.priors["lam_bg"] = cfg.lam_bg
        if cfg.bg_spacing > 0 and self.net.bg_spacing != cfg.bg_spacing:
            self.net.background, self.net.bg_spacing = None, float(cfg.bg_spacing)
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = time.time()
        self._smp = {}
        self._cache = {}
        self.graveyard = {}             # deleted vessels, which a revive move may restore
        self.births, self._bid = {}, 0  # proposed new vessels (refresh_births)
        self._gsmp = {}
        self._gid = 0
        self._index = None
        self.n_evaluated = 0
        self.endS = {}                  # (S0, S1, l) of every vessel's ends (end_evidence)
        self._egraph = self._att = None
        self.move_log = []
        if cfg.init_iters > 0:
            self.global_fit(cfg.init_iters)
        else:
            self._render_all()
        self.tau = cfg.tau if cfg.tau > 0 else self.chi2()
        self.null = self.texture_null() if cfg.texture_null else dict(per_px=0.0)
        self.price = max(self.tau * cfg.lam_length, self.null["per_px"])

    # ------------------------------------------------------------ bookkeeping
    def log(self, *a):
        if self.cfg.verbose:
            print(f"[{time.time() - self.t0:6.0f}s]", *a, flush=True)

    def samples(self, eid):
        if eid not in self._smp:
            self._smp[eid] = edge_samples(self.net.edges[eid], 1.0)
        return self._smp[eid]

    def _global_model(self):
        net = self.net
        return NetworkModel(net, self.P.logI, self.P.weight, stride=1,
                            bg_spacing=net.bg_spacing or BG_SPACING, ref=net, inside=True)

    @torch.no_grad()
    def _render_all(self, model=None):
        """Background image, halo and per-vessel patches of the current
        network (from a fitted global model when given)."""
        model = model or self._global_model()
        if self.net.background is None:
            self.net.background = model.bg.detach().numpy().copy()
            self.net.bg_spacing = model.bg_spacing
        self.B = model.background().numpy()
        hw, hs = model.halo()
        self.hw, self.hs = float(hw), float(hs)
        self.net.meta["optics"] = dict(halo_weight=self.hw, halo_sigma=self.hs)
        ent = model.vessel_entries().numpy()
        pix, eidx = model.e_pix.numpy(), model.e_edge.numpy()
        order = np.argsort(eidx, kind="stable")
        bounds = np.searchsorted(eidx[order], np.arange(len(model.eids) + 1))
        self.patch = {}
        for k, eid in enumerate(model.eids):
            sel = order[bounds[k]:bounds[k + 1]]
            self.patch[eid] = (pix[sel].copy(), ent[sel].astype(np.float32))
        self.V = np.zeros(self.H * self.W, np.float32)
        for p, v in self.patch.values():
            self.V[p] += v
        self.endS = dict(zip(model.eids, end_evidence(model, self.hw, self.cfg.frag_len,
                                                      torch.as_tensor(ent))))
        self._smp.clear()
        self._cache.clear()
        self._index = None
        self._egraph = self._att = None

    def optical_density(self):
        V = self.V.reshape(self.H, self.W)
        return (1 - self.hw) * V + self.hw * _cv_blur(V, self.hs)

    def chi2(self):
        """Reduced chi-square of the current model's residual."""
        res = self.P.logI - (self.B - self.optical_density())
        w = self.P.weight
        m = w > 0
        return float(max(1.0, (w[m] * res[m] ** 2).mean()))

    def cost(self, e: Edge, L=None) -> float:
        L = edge_samples(e, 1.0)["L"] if L is None else L
        return self.tau * self.cfg.lam_vessel + self.price * L

    def texture_null(self):
        """What background texture alone buys a 'vessel', per px of
        centreline.  Texture makes bright ridges as often as dark ones, and
        vessels are only dark, so 'anti-vessels' fitted by the same renderer
        to the bright ridges of the map's residual (away from mapped
        vessels, whose misfit rims would bias it) show how much evidence a
        dark trace of texture gets.  A vessel explaining less per px than
        the null_quantile of the anti-vessels (length-weighted) is not told
        apart from texture: that is the price of a px of centreline."""
        import cv2
        from .fit import band_scales, seeds_to_edges
        from .ridges import detect
        cfg = self.cfg
        R = self.P.logI - (self.B - self.optical_density())
        foot = np.zeros(self.P.shape, np.uint8)
        for k in self.net.edges:
            s = self.samples(k)
            w = int(math.ceil(float(np.median(s["r"] + 2 * s["s"])) + 2))
            cv2.polylines(foot, [np.round(s["xy"] * 4).astype(np.int32)], False, 1,
                          thickness=2 * w + 1, shift=2)
        null = VesselNetwork(self.P.shape)
        null.meta = {"optics": dict(halo_weight=self.hw, halo_sigma=self.hs)}
        for band in cfg.null_bands:
            ml = max(6.0, 2.5 * band[0])
            seeds = detect(R, self.P.sigma, band_scales(band, 3), min_len=ml, valid=self.P.valid,
                           exclude=foot.astype(bool))
            seeds_to_edges(null, seeds, band)
        if not null.edges:
            return dict(n=0, per_px=0.0)
        gh = int(math.ceil((self.H - 1) / BG_SPACING)) + 1
        gw = int(math.ceil((self.W - 1) / BG_SPACING)) + 1
        null.background = np.zeros((gh, gw), np.float32)
        null.bg_spacing = BG_SPACING
        m = NetworkModel(null, -R, self.P.weight, stride=1, bg_spacing=BG_SPACING,
                         fit_background=False)
        m.raw_hw.requires_grad_(False)
        m.raw_hs.requires_grad_(False)
        self._fit(m, 60)
        gains, _ = m.edge_gains()
        m.write_back()
        L = np.array([max(null.length(k), 1.0) for k in m.eids])
        g = gains / L
        ok = gains > 0
        if not ok.any():
            return dict(n=0, per_px=0.0)
        o = np.argsort(g[ok])
        cw = np.cumsum(L[ok][o]) / L[ok].sum()
        q = float(g[ok][o][min(len(o) - 1, int(np.searchsorted(cw, cfg.null_quantile)))])
        out = dict(n=int(ok.sum()), length=round(float(L[ok].sum()), 1), per_px=q,
                   median_per_px=float(np.median(g[ok])))
        self.log(f"texture null: {out['n']} anti-vessels, {out['length']:.0f} px; evidence per px "
                 f"median {out['median_per_px']:.1f}, {cfg.null_quantile:.0%} {q:.1f}")
        return out

    def energy_total(self):
        """E of the whole network (data NLL + prior + structural cost)."""
        res = self.P.logI - (self.B - self.optical_density())
        nll = 0.5 * float((self.P.weight * res ** 2).sum())
        m = NetworkModel(self.net, self.P.logI, self.P.weight, stride=1,
                         bg_spacing=self.net.bg_spacing or BG_SPACING)
        with torch.no_grad():
            pr = float(m.prior(**self.priors))
        cst = sum(self.cost(e, self.samples(k)["L"]) for k, e in self.net.edges.items())
        G = self._graph()
        att = self._attach_total()
        return dict(total=nll + pr + cst + G.total + att, nll=nll, prior=pr, cost=cst, frag=G.total + att,
                    phi=G.total, attach=att, matched_pairs=G.matched, facing_pairs=G.n_facing)

    # ------------------------------------------------------------ attachments (strict forks)
    def _attach_on(self):
        return bool(self.cfg.strict_forks)

    def _body(self, smp):
        """(xy, reach, interior) of a vessel's samples: an end of another vessel
        attaches where it lies within reach = r + s + attach_tol of a sample
        more than max(4, r + s) px from this vessel's own ends."""
        rs = smp["r"] + smp["s"]
        a, L = smp["s_arc"], smp["s_arc"][-1]
        m = np.maximum(4.0, rs)
        return smp["xy"], rs + self.cfg.attach_tol, (a > m) & (a < L - m)

    def _ends_xy(self, vid, smp):
        return [(vid, end, smp["xy"][0 if end == 0 else -1]) for end in (0, 1)
                if not self._border(smp["xy"][0 if end == 0 else -1])]

    def _bodies_index(self, bodies):
        """KD-tree over the interior samples of bodies {vid: (xy, reach, interior)}."""
        X, R, V = [], [], []
        for v, (xy, reach, inner) in bodies.items():
            X.append(xy[inner])
            R.append(reach[inner])
            V.append(np.full(int(inner.sum()), v))
        X = np.concatenate(X) if X else np.zeros((0, 2))
        return (cKDTree(X) if len(X) else None, np.concatenate(R) if R else np.zeros(0),
                np.concatenate(V) if V else np.zeros(0, int), float(max([r.max() for r in R if len(r)] + [0.0])))

    @staticmethod
    def _attached(p, vid, index, extra=None, skip=()):
        """Whether the end p of vessel vid lies on another vessel's interior (index, and the bodies in
        extra {vid: (xy, reach, interior)}), ignoring the vessels in skip."""
        tree, R, V, rmax = index
        if tree is not None:
            for i in tree.query_ball_point(p, rmax):
                if V[i] != vid and V[i] not in skip and np.linalg.norm(tree.data[i] - p) <= R[i]:
                    return True
        for v, (xy, reach, inner) in (extra or {}).items():
            if v == vid:
                continue
            d = np.linalg.norm(xy - p, axis=1)
            if np.any(inner & (d <= reach)):
                return True
        return False

    def _attach_state(self):
        """{(vid, end): attached} of the current map, and its body index (cached with the end graph)."""
        if getattr(self, "_att", None) is None:
            bodies = {k: self._body(self.samples(k)) for k in self.net.edges}
            index = self._bodies_index(bodies)
            st = {}
            for k in self.net.edges:
                for v, end, p in self._ends_xy(k, self.samples(k)):
                    st[(v, end)] = self._attached(p, v, index)
            self._att = (st, index)           # cleared with the end graph after every change
        return self._att

    def _attach_total(self):
        if not self._attach_on():
            return 0.0
        st, _ = self._attach_state()
        return self.tau * self.cfg.lam_vessel * sum(st.values())

    def _attach_delta(self, gone, new_edges):
        """Change of the attachment charge when the vessels `gone` are replaced by `new_edges`: the new
        vessels' ends, and every remaining end within reach of a removed or added body, are tested again."""
        if not self._attach_on():
            return 0.0
        st, index = self._attach_state()
        gone = set(gone)
        new = {-1 - q: edge_samples(e, 1.0) for q, e in enumerate(new_edges)}
        nb = {v: self._body(s) for v, s in new.items()}
        n_new = sum(self._attached(p, v, index, nb, skip=gone)
                    for v, s in new.items() for _, _, p in self._ends_xy(v, s))
        changed = [self._body(self.samples(k)) for k in gone] + list(nb.values())
        G = self._graph()
        near = set()
        for xy, reach, _ in changed:
            if G.tree is not None and len(xy):
                for h in G.tree.query_ball_point(xy, float(reach.max()) + 1e-6):
                    near.update(h)
        before = after = 0
        for i in near:
            v, end = G.rows[i][0], G.rows[i][5]
            if v in gone:
                continue
            before += st.get((v, end), False)
            after += self._attached(G.rows[i][1], v, index, nb, skip=gone)
        lost = sum(st.get((k, end), False) for k in gone for end in (0, 1))
        return self.tau * self.cfg.lam_vessel * (n_new + after - before - lost)

    # ------------------------------------------------------------ fragmentation
    def _phi_on(self):
        return self.cfg.lam_frag > 0 or self.cfg.lam_pair > 0

    def _omega(self, S, l):
        """What an end facing another pays: lam_frag of its evidence above
        the price of its stretch (plus lam_pair, up to that evidence), at
        most lam_cap."""
        cfg = self.cfg
        if not self._phi_on():
            return 0.0
        St = max(0.0, S - self.price * l)
        return min(self.tau * cfg.lam_cap, cfg.lam_frag * St + min(St, self.tau * cfg.lam_pair))

    def _end_rows(self, vid, smp, S):
        """EndGraph rows of a vessel's ends that are not at the image border
        (geometry as in _ends)."""
        out = []
        for end in (0, 1):
            i = 0 if end == 0 else -1
            p = smp["xy"][i]
            if self._border(p):
                continue
            t = -_direction(smp["xy"], 0, True) if end == 0 else \
                _direction(smp["xy"], len(smp["xy"]) - 1, False)
            out.append((vid, p, t, float(smp["r"][i] + smp["s"][i]),
                        self._omega(S[end], S[2]), end))
        return out

    def _graph(self):
        """EndGraph of the current map (rebuilt after every change)."""
        if self._egraph is None:
            rows = []
            for k in self.net.edges:
                rows += self._end_rows(k, self.samples(k), self.endS[k])
            self._egraph = EndGraph(rows, self.cfg)
        return self._egraph

    def _phi_delta(self, prop):
        """Change of Phi a proposal makes.  Computed afresh for the current
        state (Phi depends on ends up to join_gap away), never cached."""
        att = self._attach_delta(prop[0], prop[1])
        if not self._phi_on():
            return att
        G = self._graph()
        gone = [i for k in prop[0] for i in G.by_vid.get(k, [])]
        new = [(-1 - q, p, t, w, self._omega(S[end], S[2]), end)
               for q, (S, ends) in enumerate(zip(prop[5]["S"], prop[5]["ends"]))
               for end, p, t, w in ends]
        return G.delta(gone, new) + att

    def _release_bound(self, old):
        """-dPhi of any move removing the vessels old is at most this (with
        strict forks, plus the attachment charges it can release: the removed
        vessels' ends, and every end near their bodies)."""
        b = self._graph().bound(old) if self._phi_on() else 0.0
        if self._attach_on():
            G = self._graph()
            n = 2 * len(old)
            for k in old:
                xy, reach, _ = self._body(self.samples(k))
                if G.tree is not None:
                    n += len({i for h in G.tree.query_ball_point(xy, float(reach.max())) for i in h})
            b += self.tau * self.cfg.lam_vessel * n
        return b

    # ------------------------------------------------------------ local model
    def _window(self, smps, extra=4.0):
        xy = np.concatenate([s["xy"] for s in smps])
        reach = max(float((s["r"] + 3 * s["s"]).max()) for s in smps) + 1.5
        pad = reach + 3 * self.hs + extra
        x0 = int(max(0, math.floor(xy[:, 0].min() - pad)))
        y0 = int(max(0, math.floor(xy[:, 1].min() - pad)))
        x1 = int(min(self.W, math.ceil(xy[:, 0].max() + pad) + 1))
        y1 = int(min(self.H, math.ceil(xy[:, 1].max() + pad) + 1))
        return x0, y0, x1, y1

    def _local(self, edges, win, removed, bg=False):
        """NetworkModel of `edges` alone on the window, fitted against the
        image minus the background and every vessel except `removed`.
        bg: with a background correction on the window (zero to start),
        on the map's background grid spacing, which may be fitted."""
        x0, y0, x1, y1 = win
        pad = int(math.ceil(3 * self.hs)) + 2
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(self.W, x1 + pad), min(self.H, y1 + pad)
        Vw = self.V.reshape(self.H, self.W)[Y0:Y1, X0:X1].copy()
        for eid in removed:
            p, v = self.patch[eid]
            ys, xs = np.divmod(p, self.W)
            m = (ys >= Y0) & (ys < Y1) & (xs >= X0) & (xs < X1)
            Vw[ys[m] - Y0, xs[m] - X0] -= v[m]
        OD = (1 - self.hw) * Vw + self.hw * _cv_blur(Vw, self.hs)
        OD = OD[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
        target = self.P.logI[y0:y1, x0:x1] - self.B[y0:y1, x0:x1] + OD
        sub = VesselNetwork((y1 - y0, x1 - x0))
        sub.meta = {"optics": dict(halo_weight=self.hw, halo_sigma=self.hs)}
        off = np.array([x0, y0], float)
        for e in edges:
            c = e.ctrl - off
            u, v = sub.add_node(*c[0]), sub.add_node(*c[-1])
            k = sub._eid
            sub._eid += 1
            sub.edges[k] = Edge(u, v, c.copy(), e.r.copy(), e.s.copy(), e.a.copy(), dict(e.info))
        spacing = (self.net.bg_spacing or BG_SPACING) if bg else BG_SPACING
        gh = int(math.ceil((y1 - y0 - 1) / spacing)) + 1
        gw = int(math.ceil((x1 - x0 - 1) / spacing)) + 1
        sub.background = np.zeros((gh, gw), np.float32)
        sub.bg_spacing = spacing
        m = NetworkModel(sub, target, self.P.weight[y0:y1, x0:x1], stride=1,
                         bg_spacing=spacing, fit_background=bg,
                         image_box=(-x0, -y0, self.W - 1.0 - x0, self.H - 1.0 - y0), inside=True)
        m.raw_hw.requires_grad_(False)
        m.raw_hs.requires_grad_(False)
        return m

    def _local_energy(self, m):
        """(data NLL, prior) of a local model."""
        with torch.no_grad():
            loss, nll, _, _ = m.loss(priors=self.priors)
        return float(nll), float(loss - nll)

    def _fitted_edges(self, m, win):
        """The local model's edges back in image coordinates."""
        m.write_back()
        off = np.array([win[0], win[1]], float)
        out = []
        for eid in m.eids:
            e = m.net.edges[eid]
            out.append(Edge(-1, -1, e.ctrl + off, e.r.copy(), e.s.copy(), e.a.copy(),
                            dict(e.info)))
        return out

    def _patches(self, m, win):
        with torch.no_grad():
            ent = m.vessel_entries().numpy()
        py, px = np.divmod(m.e_pix.numpy(), m.W)
        gpix = (py + win[1]) * self.W + px + win[0]
        eidx = m.e_edge.numpy()
        return [(gpix[eidx == k], ent[eidx == k].astype(np.float32)) for k in range(len(m.eids))]

    # ------------------------------------------------------------ spatial index
    def _idx(self):
        """Per-state index of every vessel: bounding boxes, and a KD-tree of
        all centreline samples.  Rebuilt after every change."""
        if self._index is None:
            ids = list(self.net.edges)
            smp = [self.samples(k) for k in ids]
            bb = np.array([[s["xy"][:, 0].min(), s["xy"][:, 1].min(),
                            s["xy"][:, 0].max(), s["xy"][:, 1].max()] for s in smp]).reshape(-1, 4)
            X = np.concatenate([s["xy"] for s in smp]) if smp else np.zeros((0, 2))
            lab = np.concatenate([np.full(len(s["xy"]), k) for k, s in zip(ids, smp)]) \
                if smp else np.zeros(0, int)
            idx = np.concatenate([np.arange(len(s["xy"])) for s in smp]) if smp else np.zeros(0, int)
            self._index = dict(ids=np.array(ids, int), bb=bb, X=X, lab=lab, idx=idx,
                               tree=cKDTree(X) if len(X) else None)
        return self._index

    def _near(self, lo, hi, margin):
        """Vessels whose bounding box comes within margin of [lo, hi]."""
        I = self._idx()
        bb = I["bb"]
        sel = (bb[:, 2] >= lo[0] - margin) & (bb[:, 0] <= hi[0] + margin) & \
              (bb[:, 3] >= lo[1] - margin) & (bb[:, 1] <= hi[1] + margin)
        return I["ids"][sel]

    def _lookup(self, move):
        """(True, result) for a usable cached score, (False, None) when the
        move must be scored.  A move's dE depends on the other vessels only
        through the target of its local models, and linearly: with the new
        vessels' parameters fixed, dNLL changes by sens . (V_now - V_then)
        (see _sensitivity).  So a cached dE is corrected exactly for every
        change around the move since it was scored, and the move is only
        re-fitted when that correction is large enough for the fit itself
        to adapt (refit_tol), or when a vessel it involves has gone."""
        key = move["key"]
        if key not in self._cache:
            return False, None
        out = self._cache[key]
        if out is None:                    # cannot be built: depends on its own vessels only
            return True, None
        dE, prop = out
        if any(k not in self.net.edges for k in prop[0]):
            return False, None
        idx, vals, S0 = prop[4]
        delta = float(vals @ self.V[idx]) - S0 if len(idx) else 0.0
        scale = self.tau * self.cfg.t_scale
        if abs(delta) > self.cfg.refit_tol * scale and \
                dE + delta - self._release_bound(prop[0]) < self.cfg.hopeless * scale:
            return False, None
        return True, (dE + delta, prop)

    def _focus_reach(self, move, radius=None):
        """Radius of a move's focus window: the re-fitted stretch plus the
        widest footprint among the vessels it touches."""
        rch = max(float((self.samples(k)["r"] + 3 * self.samples(k)["s"]).max())
                  for k in move["anchor"])
        return (radius or self.cfg.focus_radius) + rch + 1.5

    # ------------------------------------------------------------ moves
    def evaluate(self, move):
        """Data and cost part of a move's dE, with the new vessels fitted
        locally (the change of Phi is added by _phi_delta).  Returns (dE,
        proposal) with proposal = (old ids, new edges, patches of the new
        edges or None, window, sensitivity, dict(S=their end_evidence,
        ends=their non-border ends (end, position, tangent, width),
        parts=dE split into nll / prior / cost)), or None when the move
        cannot be built.  Cached, and corrected for later changes (see
        _lookup)."""
        hit, out = self._lookup(move)
        if hit:
            return out
        out = self._compute(move)
        self._cache[move["key"]] = out
        self.n_evaluated += 1
        return out

    @torch.no_grad()
    def _sensitivity(self, m_old, m_new, win):
        """d(dE)/dV, V the sharp-core image of the fixed vessels, as sparse
        (pixel indices, values) plus their dot product with V now.  The local
        models are fitted against target = logI - B + OD_fixed, and dNLL =
        sum w (OD_new - OD_old) target + (a term without target), so the
        derivative w.r.t. OD_fixed is A = w dOD, and w.r.t. V the halo
        operator's adjoint (itself) applied to A."""
        d = (m_new.optical_density() - m_old.optical_density()).numpy()
        A = m_new.weight.numpy() * d
        x0, y0, x1, y1 = win
        pad = int(math.ceil(3 * self.hs)) + 2
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        X1, Y1 = min(self.W, x1 + pad), min(self.H, y1 + pad)
        Ap = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        Ap[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = A
        Ap = (1 - self.hw) * Ap + self.hw * _cv_blur(Ap, self.hs)
        # keep all but the smallest entries: those dropped change a corrected
        # dE by at most sum|dropped| * max V <= sens_budget nats
        flat = np.abs(Ap).ravel()
        nz = np.flatnonzero(flat)
        if len(nz):
            o = nz[np.argsort(flat[nz], kind="stable")]
            vmax = max(float(self.V.max()), 1e-3)
            cut = int(np.searchsorted(np.cumsum(flat[o], dtype=np.float64),
                                      self.cfg.sens_budget / vmax, side="right"))
            keep = o[cut:]
        else:
            keep = nz
        ys, xs = np.divmod(keep, X1 - X0)
        idx = ((ys + Y0) * self.W + xs + X0).astype(np.int64)
        vals = Ap.ravel()[keep].astype(np.float32)
        return idx, vals, float(vals @ self.V[idx]) if len(idx) else 0.0

    def _compute(self, move):
        """(dE, (old ids, new edges, their patches, window)) of a move.

        The new vessels are fitted on a window: for a join, split or
        reroute only the parameters near the junction, on the focus window;
        otherwise all of them, on the whole footprint.  dE is then always
        scored on the whole footprint of the old and new vessels, so it is
        exactly the change of E that apply() makes, wherever the new
        vessels differ from the old.  The proposal carries the sensitivity
        of dE to the other vessels (see _sensitivity, _lookup)."""
        built = move["build"]()
        if built is None:
            return None
        old, new0 = built
        focus = move.get("focus")
        full = lambda new: self._window([self.samples(k) for k in old] +
                                        [edge_samples(e) for e in new])
        def fit(start, radius, iters):
            if focus is None:
                wfit = full(start)
            else:
                wfit = self._focus_window(focus, self._focus_reach(move, radius))
            m_fit = self._local(start, wfit, old)
            free = None if focus is None else \
                self._free(m_fit, np.asarray(focus, float) - [wfit[0], wfit[1]], radius)
            self._fit(m_fit, iters, free)
            return self._fitted_edges(m_fit, wfit)

        lbg = self.cfg.local_bg
        bg_grid = [None]

        def score(new):
            win = full(new)
            m_old = self._local([self.net.edges[k] for k in old], win, old)
            n0, p0 = self._local_energy(m_old)
            c0 = sum(self.cost(self.net.edges[k], self.samples(k)["L"]) for k in old)
            m_new = self._local(new, win, old)
            n1, p1 = self._local_energy(m_new)
            if lbg:                       # the best background correction for each, in closed form
                g0, _ = self._bg_gain(m_old)
                g1, grid = self._bg_gain(m_new)
                n0, n1 = n0 - g0, n1 - g1
                bg_grid[0] = grid
            c1 = sum(self.cost(e) for e in new)
            return win, m_old, m_new, dict(nll=n1 - n0, prior=p1 - p0, cost=c1 - c0)

        new = fit(new0, None, self.cfg.local_iters) if new0 else []
        win, m_old, m_new, parts = score(new)
        scale = self.tau * self.cfg.t_scale
        if new and self.cfg.refine_below > 0 and \
                -self.cfg.refine_below * scale < sum(parts.values()) < self.cfg.refine_below * scale:
            # promising: fit again, longer (and wider around a junction), from
            # where the first fit got
            new = fit(new, self.cfg.refine_radius, self.cfg.refine_iters)
            win, m_old, m_new, parts = score(new)
        nb_S, relaxed = {}, False
        if self.cfg.relax_radius > 0 and move["kind"] in self.cfg.relax_kinds and \
                0 < sum(parts.values()) < self.cfg.relax_below * scale:
            rel = self._relaxed(old, new)
            if rel is not None and sum(rel[4].values()) < sum(parts.values()):
                old, new, win, m_old, m_new, parts, nb_S = rel[0], rel[1], rel[2], rel[3][0], \
                    rel[3][1], rel[4], rel[5]
                bg_grid[0] = rel[6]
                relaxed = True
        self.last_parts = parts
        # full-footprint patches are kept only for moves without a focus (few);
        # a join / split / reroute (or a relaxed move) is rendered again when applied
        keep = focus is None and not relaxed
        return self._proposal(old, new, win, m_old, m_new, parts, keep, nb_S,
                              bg_grid[0] if lbg else None)

    def _proposal(self, old, new, win, m_old, m_new, parts, keep_patches=False, nb_S=None,
                  bg=None):
        """(dE, proposal) of a scored move (see evaluate)."""
        sens = self._sensitivity(m_old, m_new, win)
        patches = self._patches(m_new, win) if new and keep_patches else ([] if not new else None)
        ends = []
        for e in new:
            s = edge_samples(e, 1.0)
            ends.append([(r[5], r[1], r[2], r[3]) for r in self._end_rows(-1, s, (0.0, 0.0, 0.0))])
        S = end_evidence(m_new, self.hw, self.cfg.frag_len) if new else []
        if nb_S:                   # a relaxed neighbour's ends lie beyond the window: as they were
            S = [nb_S.get(i, S[i]) for i in range(len(S))]
        extra = dict(S=S, ends=ends, parts=parts)
        if bg is not None:
            extra["bg"] = bg
        return (sum(parts.values()), (old, new, patches, win, sens, extra))

    def _relaxed(self, old, new):
        """A move scored with the vessels near it re-fitted as well: those
        with centreline within relax_radius (+8 px) of the old or new
        vessels, their parameters within relax_radius of them free (the
        old and new vessels' all), fitted for relax_iters steps before and
        after the move alike on a window over both.  Returns (old ids and
        the neighbours', new edges and the neighbours re-fitted, window,
        (model before, model after), parts, {index in new: end evidence of
        a neighbour as it was}, background correction with local_bg or
        None), or None without neighbours."""
        R = self.cfg.relax_radius
        smp = [self.samples(k) for k in old] + [edge_samples(e) for e in new]
        pts = np.concatenate([s["xy"] for s in smp])
        I = self._idx()
        if I["tree"] is None:
            return None
        nb = sorted({int(I["lab"][c]) for h in I["tree"].query_ball_point(pts, R + 8.0) for c in h}
                    - set(old))
        if not nb:
            return None
        tree = cKDTree(pts)
        ext = []
        for j in nb:
            s = self.samples(j)
            d, _ = tree.query(s["xy"])
            sel = d <= R + 30.0             # a free control point moves the curve about this far
            if sel.any():
                ext.append(dict(xy=s["xy"][sel], r=s["r"][sel], s=s["s"][sel]))
        win = self._window(smp + ext)
        removed = list(old) + nb
        nbe = [self.net.edges[j] for j in nb]
        off = np.array([win[0], win[1]], float)
        models = []
        for edges in ([self.net.edges[k] for k in old] + nbe, list(new) + nbe):
            m = self._local(edges, win, removed)
            if len(m.eids):
                self._fit(m, self.cfg.relax_iters, self._free_near(m, pts - off, R))
            models.append(m)
        m0, m1 = models
        n0, p0 = self._local_energy(m0)
        n1, p1 = self._local_energy(m1)
        grid = None
        if self.cfg.local_bg:                 # as in _compute: each with its best correction
            g0, _ = self._bg_gain(m0)
            g1, grid = self._bg_gain(m1)
            n0, n1 = n0 - g0, n1 - g1
        fitted = self._fitted_edges(m1, win)
        c0 = sum(self.cost(self.net.edges[k], self.samples(k)["L"]) for k in removed)
        c1 = sum(self.cost(e) for e in fitted)
        nb_S = {len(new) + q: self.endS[j] for q, j in enumerate(nb) if j in self.endS}
        return (removed, fitted, win, (m0, m1), dict(nll=n1 - n0, prior=p1 - p0, cost=c1 - c0),
                nb_S, grid)

    @torch.no_grad()
    def _free_near(self, m, pts, radius):
        """As _free, for the parameters within radius of any of pts."""
        T = cKDTree(pts)
        near = lambda P: torch.as_tensor(T.query(P.numpy())[0] <= radius)
        ctrl = m.ctrl_all().numpy()
        prof = []
        for k in range(len(m.eids)):
            n, mp = m.edge_nctrl[k], m.edge_nprof[k]
            c = ctrl[m.edge_ctrl_off[k]:m.edge_ctrl_off[k] + n]
            prof.append(sp.design(n, max(mp, 2))[:mp] @ c if mp > 1 else c[:1])
        prof = torch.as_tensor(np.concatenate(prof), dtype=torch.float32)
        return dict(node=near(m.node_xy), inner=near(m.inner), prof=near(prof))

    def _focus_window(self, focus, R):
        R = R + 3 * self.hs + 4.0
        x0 = int(max(0, math.floor(focus[0] - R)))
        y0 = int(max(0, math.floor(focus[1] - R)))
        x1 = int(min(self.W, math.ceil(focus[0] + R) + 1))
        y1 = int(min(self.H, math.ceil(focus[1] + R) + 1))
        return x0, y0, x1, y1

    @torch.no_grad()
    def _free(self, m, f, radius=None):
        """Masks of the parameters of a local model within focus_radius (or
        radius) of the point f (window coordinates): nodes, inner control
        points and profile knots.  The others are held fixed."""
        rad2 = (radius or self.cfg.focus_radius) ** 2
        near = lambda P: ((P - torch.as_tensor(f, dtype=P.dtype)) ** 2).sum(1) <= rad2
        ctrl = m.ctrl_all().numpy()
        prof = []
        for k in range(len(m.eids)):
            n, mp = m.edge_nctrl[k], m.edge_nprof[k]
            c = ctrl[m.edge_ctrl_off[k]:m.edge_ctrl_off[k] + n]
            prof.append(sp.design(n, max(mp, 2))[:mp] @ c if mp > 1 else c[:1])
        prof = torch.as_tensor(np.concatenate(prof), dtype=torch.float32)
        return dict(node=near(m.node_xy), inner=near(m.inner), prof=near(prof))

    @torch.no_grad()
    def _bg_gain(self, m):
        """The background correction c (a grid on the local model's window,
        at the map's background spacing, upsampled bicubically like the
        background) that best explains the local model's residual under the
        background prior, in closed form: it minimises
        0.5 sum w (res - U c)^2 + lam_bg |second differences of c|^2, and
        lowers NLL + prior by 0.5 b' A^-1 b (A = U'WU + 2Q, b = U'W res).
        Returns (that decrease, c)."""
        import torch.nn.functional as F
        H, W = m.H, m.W
        spacing = self.net.bg_spacing or BG_SPACING
        gh = int(math.ceil((H - 1) / spacing)) + 1
        gw = int(math.ceil((W - 1) / spacing)) + 1
        n = gh * gw
        basis = torch.eye(n, dtype=torch.float32).reshape(n, 1, gh, gw)
        U = F.interpolate(basis, size=(H, W), mode="bicubic", align_corners=True)
        U = U.reshape(n, -1).T.double()                          # pixels x coefficients
        res = (m.logI - m.predict()).reshape(-1).double()
        w = m.weight.reshape(-1).double()
        A = U.T @ (w[:, None] * U)
        lam = float(self.priors.get("lam_bg", MapConfig().lam_bg))
        D = []
        for i in range(gh):
            for j in range(gw - 2):
                row = torch.zeros(n, dtype=torch.float64)
                row[i * gw + j], row[i * gw + j + 1], row[i * gw + j + 2] = 1, -2, 1
                D.append(row)
        for i in range(gh - 2):
            for j in range(gw):
                row = torch.zeros(n, dtype=torch.float64)
                row[i * gw + j], row[(i + 1) * gw + j], row[(i + 2) * gw + j] = 1, -2, 1
                D.append(row)
        if D:
            D = torch.stack(D)
            A = A + 2 * lam * (D.T @ D)
        b = U.T @ (w * res)
        c = torch.linalg.solve(A + 1e-9 * torch.eye(n, dtype=torch.float64), b)
        return float(0.5 * (b @ c)), c.reshape(gh, gw).float().numpy()

    def _fit(self, m, iters, free=None):
        """Adam on a local model's vessels (positions and profiles; the
        halo and background stay fixed), only on `free` parameters."""
        cfg = self.cfg
        opt = torch.optim.Adam([dict(params=[m.node_xy, m.inner], lr=cfg.lr_pos),
                                dict(params=[m.raw_r, m.raw_s, m.raw_a], lr=cfg.lr_prof)])
        base = [g["lr"] for g in opt.param_groups]
        for it in range(iters):
            f = 0.5 * (1 + math.cos(math.pi * it / iters)) * 0.9 + 0.1
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            opt.zero_grad(set_to_none=True)
            loss, _, _, _ = m.loss(priors=self.priors)
            loss.backward()
            if free is not None:
                m.node_xy.grad[~free["node"]] = 0
                m.inner.grad[~free["inner"]] = 0
                for q in (m.raw_r, m.raw_s, m.raw_a):
                    q.grad[~free["prof"]] = 0
            opt.step()
            m.project()
            if (it + 1) % 20 == 0 and it + 1 < iters:
                m.rebuild()
        m.rebuild()

    def apply(self, proposal):
        old, new, patches, win = proposal[:4]
        if len(proposal) > 5 and "bg" in proposal[5]:
            self._apply_bg(proposal[5]["bg"], win)
        if patches is None:            # render the new vessels again, each on its own footprint
            patches = []               # (a relaxed neighbour reaches beyond the move's window)
            for e in new:
                w = self._window([edge_samples(e)])
                patches += self._patches(self._local([e], w, []), w)
        for k in old:
            p, v = self.patch.pop(k)
            self.V[p] -= v
            self.net.remove_edge(k)
            self._smp.pop(k, None)
            self.endS.pop(k, None)
        ids = []
        for e, (p, v), S in zip(new, patches, proposal[5]["S"]):
            e = e.copy()
            u = self.net.add_node(*e.ctrl[0])
            w = self.net.add_node(*e.ctrl[-1])
            k = self.net._eid
            self.net._eid += 1
            e.u, e.v = u, w
            self.net.edges[k] = e
            self.net._touch()
            self.patch[k] = (p, v)
            self.V[p] += v
            self.endS[k] = S
            ids.append(k)
        self._index = None
        self._egraph = self._att = None
        return ids

    def _apply_bg(self, grid, win):
        """Add a move's background correction (its grid on the window) to
        the background image; scores fitted against the old background
        there are dropped."""
        import torch.nn.functional as F
        x0, y0, x1, y1 = win
        g = torch.as_tensor(grid, dtype=torch.float32)
        img = F.interpolate(g[None, None], size=(y1 - y0, x1 - x0), mode="bicubic",
                            align_corners=True)[0, 0].numpy()
        if not np.any(img):
            return
        self.B = self.B.copy()
        self.B[y0:y1, x0:x1] += img
        keep = {}
        for key, out in self._cache.items():
            if out is None:
                keep[key] = out
                continue
            a0, b0, a1, b1 = out[1][3]
            if a0 < x1 and x0 < a1 and b0 < y1 and y0 < b1:
                continue
            keep[key] = out
        self._cache = keep

    def _ends(self):
        """Every vessel end: (eid, end, position, outward tangent, width)."""
        out = []
        for k in self.net.edges:
            s = self.samples(k)
            for end in (0, 1):
                i = 0 if end == 0 else -1
                t = -_direction(s["xy"], 0, True) if end == 0 else \
                    _direction(s["xy"], len(s["xy"]) - 1, False)
                out.append((k, end, s["xy"][i], t, float(s["r"][i] + s["s"][i])))
        return out

    def _border(self, p, margin=3.0):
        return p[0] < margin or p[1] < margin or p[0] > self.W - 1 - margin or \
            p[1] > self.H - 1 - margin

    def join_moves(self):
        cfg = self.cfg
        ends = [e for e in self._ends() if not self._border(e[2])]
        if len(ends) < 2:
            return []
        P = np.array([e[2] for e in ends])
        tree = cKDTree(P)
        moves, seen = [], set()
        for i, (k1, end1, p1, t1, w1) in enumerate(ends):
            G = min(cfg.join_gap, max(8.0, cfg.join_gap_factor * w1))
            for j in tree.query_ball_point(p1, G):
                k2, end2, p2, t2, w2 = ends[j]
                if k2 == k1:
                    continue
                pair = tuple(sorted([(k1, end1), (k2, end2)]))
                if pair in seen:
                    continue
                if not _faces(p1, t1, w1, p2, t2, w2, cfg, gap=False):
                    continue
                seen.add(pair)
                moves.append(dict(kind="join", anchor=[k1, k2], key=("join",) + pair,
                                  focus=0.5 * (p1 + p2),
                                  build=self._join_builder(k1, end1, k2, end2)))
        return moves

    def _join_builder(self, k1, end1, k2, end2):
        def build():
            e1, e2 = self.net.edges[k1], self.net.edges[k2]
            A = edge_samples(e1, 0.7)
            B = edge_samples(e2, 0.7)
            A = A if end1 == 1 else _flip(A)
            B = B if end2 == 0 else _flip(B)
            path = join_path(A, B, self.cfg.trim_max, self.cfg.max_turn_deg)
            if path is None:
                return None
            info = _merge_info([e1.info, e2.info])
            ctrl, r, s, a = fit_vessel(*path, info["spacing"])
            return [k1, k2], [Edge(-1, -1, ctrl, r, s, a, info)]
        return build

    def delete_moves(self):
        return [dict(kind="delete", anchor=[k], key=("delete", k),
                     build=self._delete_builder(k)) for k in self.net.edges]

    def _duplicates(self, k, frac=0.3, cos_par=0.8):
        """Vessels running inside and parallel to k along >= frac of k."""
        s = self.samples(k)
        xy, tan = s["xy"], s["tan"]
        out = []
        for j in self._near(xy.min(0), xy.max(0), 30.0):
            j = int(j)
            if j == k:
                continue
            o = self.samples(j)
            if "tree" not in o:
                o["tree"] = cKDTree(o["xy"])
            d, i = o["tree"].query(xy)
            inside = (d < o["r"][i] + 0.7 * o["s"][i] + 1.0) & \
                (np.abs((o["tan"][i] * tan).sum(1)) > cos_par)
            if inside.mean() >= frac:
                out.append(j)
        return out

    def _delete_builder(self, k):
        """Delete k; vessels duplicating it are re-fitted to take over."""
        def build():
            dup = self._duplicates(k)
            return [k] + dup, [self.net.edges[j].copy() for j in dup]
        return build

    def revive_moves(self):
        """Put back a vessel deleted earlier.  Deletions accepted at a high
        temperature would otherwise be final; with this move every move
        of the search can be undone."""
        out = []
        for g, e in self.graveyard.items():
            if g not in self._gsmp:
                self._gsmp[g] = edge_samples(e, 1.0)
            out.append(dict(kind="revive", anchor=[], key=("revive", g), grave=g,
                            smp=self._gsmp[g], build=(lambda e=e: ([], [e.copy()]))))
        return out

    def refresh_births(self, bands=None):
        """Candidate new vessels from what the map does not explain yet:
        dark ridges of the residual away from the mapped vessels (per band
        of widths) and faint line paths traced along it.  Each becomes a
        birth move, fitted and scored like any other; the energy decides
        which exist (a vessel must pay its price and cost)."""
        from .faint import FaintConfig, map_zone, trace_faint
        from .fit import band_scales, seeds_to_edges
        from .ridges import detect
        from .render import profile_peak
        cfg = self.cfg
        R = self.P.logI - (self.B - self.optical_density())
        zone = map_zone(self.net, self.P.shape, 0.0, 1.0)   # a centreline itself: not found again
        tmp = VesselNetwork(self.P.shape)
        for band in (cfg.birth_bands if bands is None else bands):
            seeds = detect(-R, self.P.sigma, band_scales(band, 3), z_hi=cfg.birth_z,
                           z_lo=0.5 * cfg.birth_z, min_len=max(8.0, 2.5 * band[0]),
                           valid=self.P.valid, exclude=zone)
            seeds_to_edges(tmp, seeds, band)
        if cfg.birth_faint and (bands is None or min(b[0] for b in bands) <= 1.2):
            for t in trace_faint(R, zone, FaintConfig(verbose=False), self.P.valid):
                xy = t["xy"]
                n = len(xy)
                sc = float(np.median(t["scale"]))
                r, s_ = max(R_MIN, 0.8 * sc), max(S_MIN, 0.6 * sc)
                ix = np.clip(np.round(xy).astype(int), 0, [self.W - 1, self.H - 1])
                a = max(0.01, float(np.mean(-R[ix[:, 1], ix[:, 0]])) / float(profile_peak(r, s_)))
                tmp.add_edge_dense(xy, np.full(n, r), np.full(n, s_), np.full(n, a),
                                   info=dict(tier="birth-faint"))
        self.births = {}
        for k in tmp.edges:
            e = tmp.edges[k]
            if len(e.ctrl) >= 2:
                self.births[self._bid] = e
                self._bid += 1
        self.log(f"births: {len(self.births)} proposals")

    def birth_moves(self):
        return [dict(kind="birth", anchor=[], key=("birth", b), bid=b,
                     build=(lambda e=e: ([], [e.copy()]))) for b, e in self.births.items()]

    def split_moves(self):
        """Split a vessel at a sharp bend (the path may switch vessels there)."""
        cfg = self.cfg
        moves = []
        cmax = math.cos(math.radians(cfg.split_turn_deg))
        look = 5
        for k in self.net.edges:
            s = self.samples(k)
            xy = s["xy"]
            n = len(xy)
            if n < 4 * look:
                continue
            a = xy[look:n - look] - xy[:n - 2 * look]
            b = xy[2 * look:] - xy[look:n - look]
            c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
            idx = np.arange(look, n - look)
            cand = idx[c < cmax]
            if not len(cand):
                continue
            # one candidate per bend: the sharpest point of each run
            for run in np.split(cand, np.flatnonzero(np.diff(cand) > 1) + 1):
                i = int(run[np.argmin(c[run - look])])
                if s["s_arc"][i] < 6 or s["L"] - s["s_arc"][i] < 6:
                    continue
                moves.append(dict(kind="split", anchor=[k], key=("split", k, i),
                                  focus=xy[i], build=self._split_builder(k, s["s_arc"][i])))
        return moves + self.border_split_moves()

    def border_split_moves(self, near=3.0):
        """Split a vessel where it runs along the image border, both new ends
        put on the border (open there): a vessel that leaves the image and
        comes back is two traces, not one bridging along the border."""
        moves = []
        lo, hi = near, np.array([self.W - 1.0 - near, self.H - 1.0 - near])
        for k in self.net.edges:
            s = self.samples(k)
            xy = s["xy"]
            at = np.flatnonzero((xy[:, 0] < lo) | (xy[:, 1] < lo) | (xy[:, 0] > hi[0]) |
                                (xy[:, 1] > hi[1]))
            at = at[(s["s_arc"][at] >= 6.0) & (s["L"] - s["s_arc"][at] >= 6.0)]
            if not len(at):
                continue
            for run in np.split(at, np.flatnonzero(np.diff(at) > 1) + 1):
                d = np.minimum.reduce([xy[run, 0], xy[run, 1], self.W - 1.0 - xy[run, 0],
                                       self.H - 1.0 - xy[run, 1]])
                i = int(run[np.argmin(d)])
                moves.append(dict(kind="split", anchor=[k], key=("bsplit", k, i), focus=xy[i],
                                  build=self._split_builder(k, s["s_arc"][i], border=True)))
        return moves

    def _split_builder(self, k, s_cut, border=False):
        def build():
            e = self.net.edges[k]
            s = edge_samples(e, 0.7)
            i = int(np.searchsorted(s["s_arc"], s_cut))
            out = []
            sls = (slice(0, i + 1), slice(i, None))
            links = _split_links(e.info.get("links", []), [s["xy"][sl] for sl in sls],
                                 s["xy"][min(i, len(s["xy"]) - 1)])
            for n, (sl, lk) in enumerate(zip(sls, links)):
                if len(s["xy"][sl]) < 4:
                    return None
                xy = s["xy"][sl].copy()
                if border:                      # the cut end onto the nearest border
                    j = -1 if n == 0 else 0
                    x, y = xy[j]
                    d = [x, y, self.W - 1.0 - x, self.H - 1.0 - y]
                    m = int(np.argmin(d))
                    xy[j] = [0.0 if m == 0 else self.W - 1.0 if m == 2 else x,
                             0.0 if m == 1 else self.H - 1.0 if m == 3 else y]
                ctrl, r, s_, a = _fit_edge_params(xy, s["r"][sl], s["s"][sl],
                                                  s["a"][sl], e.info.get("spacing", 12.0),
                                                  faithful=True)
                out.append(Edge(-1, -1, ctrl, r, s_, a, _with_links(e.info, lk)))
            return [k], out
        return build

    def trim_moves(self, cuts=(8.0, 16.0, 32.0)):
        """Cut a stretch off a vessel end: where a trace runs on past its
        vessel, into a neighbour or along texture, the rest stands alone."""
        moves = []
        for k in self.net.edges:
            s = self.samples(k)
            L = float(s["L"])
            for end in (0, 1):
                if end == 0 and self.net.node_kind(self.net.edges[k].u, 1) == "border" or \
                        end == 1 and self.net.node_kind(self.net.edges[k].v, 1) == "border":
                    continue
                for c in cuts:
                    if L - c < 10.0:
                        continue
                    at = c if end == 0 else L - c
                    i = int(np.searchsorted(s["s_arc"], at))
                    moves.append(dict(kind="trim", anchor=[k], key=("trim", k, end, c),
                                      focus=s["xy"][min(i, len(s["xy"]) - 1)],
                                      build=self._trim_builder(k, end, at)))
        return moves

    def _trim_builder(self, k, end, at):
        def build():
            e = self.net.edges[k]
            s = edge_samples(e, 0.7)
            i = int(np.searchsorted(s["s_arc"], at))
            sl = slice(i, None) if end == 0 else slice(0, i + 1)
            if len(s["xy"][sl]) < 4:
                return None
            links = _split_links(e.info.get("links", []), [s["xy"][sl]], s["xy"][min(i, len(s["xy"]) - 1)])
            ctrl, r, s_, a = _fit_edge_params(s["xy"][sl], s["r"][sl], s["s"][sl], s["a"][sl],
                                              e.info.get("spacing", 12.0), faithful=True)
            return [k], [Edge(-1, -1, ctrl, r, s_, a, _with_links(e.info, links[0]))]
        return build

    def extend_moves(self, lengths=(4.0, 8.0, 16.0, 24.0), z=1.0):
        """Lengthen a vessel end along the dark ridge the model leaves
        unexplained beyond it (a trace that stopped short of its vessel's
        end, or of the vessel it runs into): the ridge is followed from the
        end (see _trace) and each length it reaches is a candidate; the
        shortest is tried straight ahead in any case (the end's own taper
        explains part of what lies just beyond it)."""
        moves = []
        ridge = self._ridge_maps()
        for (k, end, p, t, w) in self._ends():
            if self._border(p):
                continue
            stops = []
            path = self._trace(ridge, k, p, t, float(self.samples(k)["r"][0 if end == 0 else -1]),
                               max(lengths), z, stops=stops)
            L = 0.0 if path is None else float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if L < lengths[1]:
                path = np.asarray(p, float) + np.outer(np.linspace(0.0, lengths[1], 9), t)
            inside = (path[:, 0] >= 0) & (path[:, 0] <= self.W - 1) & \
                (path[:, 1] >= 0) & (path[:, 1] <= self.H - 1)
            if not inside.all():                     # to the border, where the vessel stays open
                n = int(np.argmin(inside))
                a, b = path[n - 1], path[n]
                lo = np.array([0.0, 0.0]) - a
                hi = np.array([self.W - 1.0, self.H - 1.0]) - a
                d = b - a
                f = min([1.0] + [float(v) for v in np.r_[lo / np.where(d < 0, d, -1e-12),
                                                         hi / np.where(d > 0, d, 1e-12)] if 0 <= v <= 1])
                path = np.vstack([path[:n], a + f * d])
            L = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if L < 1.0:
                continue
            cands = sorted({round(min(c, L), 1) for c in lengths} |
                           {round(c, 1) for c in stops if 1.0 <= c <= L})
            for c in cands:
                moves.append(dict(kind="extend", anchor=[k], key=("extend", k, end, c),
                                  focus=p, build=self._extend_builder(k, end, path, c)))
        return moves + self._extend_births()

    def _extend_births(self, reach=6.0, cos_min=0.5):
        """Extend a vessel end along a proposed birth that starts at it
        (within reach px, heading on from the end): a stretch too short or
        faint to pay for a vessel of its own may still pay as more of one."""
        moves = []
        if not self.births:
            return moves
        for (k, end, p, t, w) in self._ends():
            if self._border(p):
                continue
            for bid, b in self.births.items():
                xy = edge_samples(b, 0.7)["xy"]
                for bxy in (xy, xy[::-1]):
                    if np.linalg.norm(bxy[0] - p) > reach + w:
                        continue
                    d = bxy[min(len(bxy) - 1, 6)] - bxy[0]
                    if np.dot(d, t) < cos_min * (np.linalg.norm(d) + 1e-9):
                        continue
                    path = np.vstack([p, bxy])
                    L = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                    moves.append(dict(kind="extend", anchor=[k], key=("extend_birth", k, end, bid),
                                      focus=p, bid=bid, build=self._extend_builder(k, end, path, L)))
        return moves

    def _ridge_maps(self, sigmas=(1.0, 2.0, 4.0, 8.0)):
        """The residual's dark ridges at a few scales: the image less the
        model, negated and smoothed, over its robust noise level."""
        import cv2
        res = -(self.P.logI - (self.B - self.optical_density())).astype(np.float32)
        out = {}
        ok = self.P.weight > 0
        for sg in sigmas:
            m = cv2.GaussianBlur(res, (0, 0), sg)
            v = m[ok]
            noise = 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 1e-9
            out[sg] = (m, noise)
        return out

    def _trace(self, ridge, k, p, t, r, Lmax, z, step=2.0, turn=None, stops=None,
               first_turn=None):
        """Follow the residual's dark ridge from p along t, a step at a
        time, turning at most `turn` degrees a step (the first step up to
        first_turn: a tortuous vessel's trace may stop where it turns back),
        while it stays z noise levels above zero (at the scale of the
        vessel's radius r).  Where it
        reaches another vessel it goes on (a vessel may cross, or end
        inside, another), noting the arc length in `stops`.  None when not
        a step is taken."""
        turn = self.cfg.extend_turn if turn is None else turn
        first_turn = self.cfg.extend_first_turn if first_turn is None else first_turn
        sg = min(ridge, key=lambda q: abs(math.log(q / max(r, 0.8))))
        M, noise = ridge[sg]
        H, W = M.shape

        def at(q):
            x, y = float(q[0]), float(q[1])
            if not (0 <= x < W - 1 and 0 <= y < H - 1):
                return -np.inf
            x0, y0 = int(x), int(y)
            fx, fy = x - x0, y - y0
            return float((1 - fx) * (1 - fy) * M[y0, x0] + fx * (1 - fy) * M[y0, x0 + 1] +
                         (1 - fx) * fy * M[y0 + 1, x0] + fx * fy * M[y0 + 1, x0 + 1])

        I = self._idx()
        pts, d = [np.asarray(p, float)], np.asarray(t, float) / (np.linalg.norm(t) + 1e-9)
        angs = np.radians(np.linspace(-turn, turn, 7))
        first = np.radians(np.linspace(-first_turn, first_turn, 21))   # an end may sit at a
        while (len(pts) - 1) * step < Lmax:                              # hairpin
            best, bv = None, -np.inf
            for a in (first if len(pts) == 1 else angs):
                c, s_ = math.cos(a), math.sin(a)
                u = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])
                q = pts[-1] + step * u
                v = at(q)
                if v > bv:
                    best, bv = q, v
            if best is None or bv < z * noise:
                break
            d = (best - pts[-1]) / step
            pts.append(best)
            if I["tree"] is not None:            # reached another vessel: end there, or
                hit = [c for c in I["tree"].query_ball_point(best, 2.0 + r)    # (through) note
                       if int(I["lab"][c]) != k]                         # it as a candidate
                if hit and not self.cfg.extend_through:
                    break
                if hit and stops is not None and (not stops or stops[-1] < (len(pts) - 2) * step):
                    stops.append((len(pts) - 1) * step)
        return np.array(pts) if len(pts) > 1 else None

    def _extend_builder(self, k, end, path, c):
        def build():
            e = self.net.edges[k]
            s = edge_samples(e, 0.7)
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            sa = np.r_[0, np.cumsum(seg)]
            q = np.arange(0.7, min(c, sa[-1]) + 1e-6, 0.7)
            if not len(q):
                return None
            ext = np.stack([np.interp(q, sa, path[:, 0]), np.interp(q, sa, path[:, 1])], 1)
            i = 0 if end == 0 else -1
            n = len(ext)
            prof = [np.full(n, s[key][i]) for key in ("r", "s", "a")]
            if end == 1:
                xy = np.concatenate([s["xy"], ext])
                r, s_, a = (np.concatenate([s[key], pr]) for key, pr in zip(("r", "s", "a"), prof))
            else:
                xy = np.concatenate([ext[::-1], s["xy"]])
                r, s_, a = (np.concatenate([pr, s[key]]) for key, pr in zip(("r", "s", "a"), prof))
            ctrl, r, s_, a = _fit_edge_params(xy, r, s_, a, e.info.get("spacing", 12.0),
                                              faithful=True)
            return [k], [Edge(-1, -1, ctrl, r, s_, a, dict(e.info))]
        return build

    def swap_moves(self, touch=1.0, apart=10.0):
        """Two vessels that cross exchange their parts beyond the crossing,
        each part going on with the one that keeps its direction (a trace
        that switched vessels where two cross at a shallow angle: two
        traces that touch there, centrelines within their radii + touch,
        and part again within `apart` px either side; one vessel traced
        twice stays close)."""
        moves = []
        I = self._idx()
        if I["tree"] is None or len(I["ids"]) < 2:
            return moves
        X, lab, idx = I["X"], I["lab"], I["idx"]
        rmax = max(float(self.samples(j)["r"].max()) for j in self.net.edges)
        seen = set()
        for k in self.net.edges:
            s = self.samples(k)
            best = {}                                   # (j, crossing cluster) -> closest pair
            for i, h in enumerate(I["tree"].query_ball_point(s["xy"], s["r"].max() + rmax + touch)):
                for c in h:
                    j = int(lab[c])
                    if j == k:
                        continue
                    d = float(np.linalg.norm(X[c] - s["xy"][i]))
                    if d > s["r"][i] + self.samples(j)["r"][int(idx[c])] + touch:
                        continue
                    key = (j, int(s["s_arc"][i] // 12.0))     # one candidate per 12 px of k
                    if key not in best or d < best[key][0]:
                        best[key] = (d, i, int(idx[c]))
            for (j, _), (d, i, ij) in best.items():
                o = self.samples(j)
                if min(s["s_arc"][i], s["L"] - s["s_arc"][i], o["s_arc"][ij],
                       o["L"] - o["s_arc"][ij]) < 6.0:
                    continue                            # near an end: a join or reroute
                if "tree" not in o:
                    o["tree"] = cKDTree(o["xy"])
                far = [int(np.clip(np.searchsorted(s["s_arc"], s["s_arc"][i] + dd), 0,
                                   len(s["xy"]) - 1)) for dd in (-apart, apart)]
                if min(float(o["tree"].query(s["xy"][f])[0]) for f in far) < d + 2.0:
                    continue                            # still together: one vessel traced twice
                pair = (min(k, j), max(k, j), round(float(s["xy"][i][0]) / 4),
                        round(float(s["xy"][i][1]) / 4))
                if pair in seen:
                    continue
                seen.add(pair)
                moves.append(dict(kind="swap", anchor=[k, j], key=("swap",) + pair,
                                  focus=s["xy"][i],
                                  build=self._swap_builder(k, s["s_arc"][i], j, o["s_arc"][ij])))
        return moves

    def _swap_builder(self, k, sk, j, sj):
        def build():
            A = edge_samples(self.net.edges[k], 0.7)
            B = edge_samples(self.net.edges[j], 0.7)
            ia = int(np.clip(np.searchsorted(A["s_arc"], sk), 1, len(A["xy"]) - 2))
            ib = int(np.clip(np.searchsorted(B["s_arc"], sj), 1, len(B["xy"]) - 2))
            if np.dot(_direction(A["xy"], ia), _direction(B["xy"], ib)) < 0:
                B = _flip(B)                            # B runs the way A does
                ib = len(B["xy"]) - 1 - ib
            cut = lambda S, sl: {q: (v[sl] if isinstance(v, np.ndarray) else v)
                                 for q, v in S.items()}
            parts = [(cut(A, slice(0, ia + 1)), cut(B, slice(ib, None))),
                     (cut(B, slice(0, ib + 1)), cut(A, slice(ia, None)))]
            out = []
            for head, tail in parts:
                if len(head["xy"]) < 4 or len(tail["xy"]) < 4:
                    return None
                xy = np.concatenate([head["xy"], tail["xy"][1:]])
                r, s_, a = (np.concatenate([head[q], tail[q][1:]]) for q in ("r", "s", "a"))
                ctrl, r, s_, a = _fit_edge_params(xy, r, s_, a,
                                                  min(self.net.edges[k].info.get("spacing", 12.0),
                                                      self.net.edges[j].info.get("spacing", 12.0)),
                                                  faithful=True)
                out.append(Edge(-1, -1, ctrl, r, s_, a,
                                _merge_info([self.net.edges[k].info, self.net.edges[j].info])))
            return [k, j], out
        return build

    def reroute_moves(self):
        """A vessel end E that touches vessel V in its interior: E continues
        into one part of V, the other part of V ends at the junction.  This
        changes which arm of a branch point continues the parent."""
        moves = []
        I = self._idx()
        if I["tree"] is None or len(I["ids"]) < 2:
            return moves
        X, lab, idx = I["X"], I["lab"], I["idx"]
        for (k, end, p, t, w) in self._ends():
            if self._border(p):
                continue
            near = {}
            for c in I["tree"].query_ball_point(p, 25.0):
                j = int(lab[c])
                if j == k:
                    continue
                d = float(np.linalg.norm(X[c] - p))
                if j not in near or d < near[j][0]:
                    near[j] = (d, int(idx[c]))
            for j, (d, i) in near.items():
                o = self.samples(j)
                if d > o["r"][i] + o["s"][i] + 2.0:
                    continue
                if o["s_arc"][i] < 8 or o["L"] - o["s_arc"][i] < 8:
                    continue                            # near V's end: a join, not a reroute
                for part in (0, 1):                     # E continues into V's part 0 or 1
                    moves.append(dict(kind="reroute", anchor=[k, j],
                                      key=("reroute", k, end, j, i, part), focus=p,
                                      build=self._reroute_builder(k, end, j, o["s_arc"][i],
                                                                  part)))
        return moves

    def _reroute_builder(self, k, end, j, s_cut, part):
        def build():
            e, v = self.net.edges[k], self.net.edges[j]
            s = edge_samples(v, 0.7)
            i = int(np.searchsorted(s["s_arc"], s_cut))
            if i < 4 or i > len(s["xy"]) - 5:
                return None
            cut = lambda sl: {kk: (vv[sl] if isinstance(vv, np.ndarray) else vv)
                              for kk, vv in s.items()}
            pieces = {0: cut(slice(0, i + 1)), 1: cut(slice(i, None))}
            pieces[1]["s_arc"] = pieces[1]["s_arc"] - pieces[1]["s_arc"][0]
            cont, stay = pieces[part], pieces[1 - part]
            if part == 0:                                # continue into V backwards
                cont = _flip(dict(cont, L=float(cont["s_arc"][-1])))
            A = edge_samples(e, 0.7)
            A = A if end == 1 else _flip(A)
            path = join_path(A, cont, self.cfg.trim_max, self.cfg.max_turn_deg)
            if path is None:
                return None
            info = _merge_info([e.info, v.info])
            lk = _split_links(info.get("links", []), [path[0], stay["xy"]], s["xy"][i])
            ctrl, r, s_, a = fit_vessel(*path, info["spacing"])
            c2, r2, s2, a2 = _fit_edge_params(stay["xy"], stay["r"], stay["s"], stay["a"],
                                              v.info.get("spacing", 12.0), faithful=True)
            return [k, j], [Edge(-1, -1, ctrl, r, s_, a, _with_links(info, lk[0])),
                            Edge(-1, -1, c2, r2, s2, a2, _with_links(v.info, lk[1]))]
        return build

    def all_moves(self, kinds=("join", "delete", "split", "reroute", "revive", "birth", "trim",
                               "extend", "swap")):
        gen = dict(join=self.join_moves, delete=self.delete_moves, split=self.split_moves,
                   reroute=self.reroute_moves, revive=self.revive_moves, birth=self.birth_moves,
                   trim=self.trim_moves, extend=self.extend_moves, swap=self.swap_moves)
        return [m for k in kinds for m in gen[k]()]

    # ------------------------------------------------------------ driver
    def global_fit(self, iters):
        if not self.net.edges:
            self._render_all()
            return
        m = self._global_model()
        optimize(m, iters, self.cfg.lr_pos * 0.5, self.cfg.lr_prof_global, MapConfig().lr_bg,
                 rebuild_every=25, priors=self.priors, track=MapConfig().anchor())
        m.write_back()
        self._render_all(m)

    def _evaluate_all(self, moves):
        """Score every move whose cached score is stale, in worker processes
        when there are enough of them.  The workers are forked from the
        current state, so they see it without copying; only the results
        come back.  Where processes cannot fork (Windows) the moves are
        scored here, one after another."""
        todo = [i for i, m in enumerate(moves) if not self._lookup(m)[0]]
        nw = self.cfg.workers or os.cpu_count() or 1
        if nw <= 1 or len(todo) < 2 * nw or "fork" not in multiprocessing.get_all_start_methods():
            for i in todo:
                self.evaluate(moves[i])
            return
        global _SHARED
        _SHARED = (self, moves)
        chunks = [todo[j::nw] for j in range(nw)]
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(nw) as pool:
            for part in pool.map(_evaluate_chunk, chunks):
                for i, r in part:
                    self._cache[moves[i]["key"]] = r
        _SHARED = None
        self.n_evaluated += len(todo)

    def anneal(self, T0, steps_hot, kinds=("join", "delete", "split", "reroute", "revive", "birth",
                                           "trim", "extend", "swap"),
               max_steps=None):
        """Rejection-free annealing.  At every step all candidate moves are
        scored (cached: only moves near the last change are re-fitted) and
        one is drawn with probability proportional to exp(-dE / T), the
        current state ("stay", dE = 0) included.  T falls geometrically from
        T0 to T0 / 100 over steps_hot steps and is then 0: from there on the
        best improving move is taken until none is left."""
        with _one_thread():
            return self._anneal(T0, steps_hot, kinds, max_steps)

    def _anneal(self, T0, steps_hot, kinds, max_steps):
        decay = 0.01 ** (1.0 / max(1, steps_hot))
        T = T0
        max_steps = max_steps or (3 * steps_hot + 100 + len(self.net.edges) + len(self.births))
        acc, stays = {}, 0
        for step in range(max_steps):
            moves = self.all_moves(kinds)
            self._evaluate_all(moves)
            res = []
            for m in moves:
                r = self._lookup(m)[1]
                if r is not None:
                    dp = self._phi_delta(r[1])
                    res.append((r[0] + dp, m, r[1], dp))
            live = {m["key"] for m in moves}
            self._cache = {k: v for k, v in self._cache.items() if k in live}
            if not res:
                break
            dE = np.array([r[0] for r in res])
            T_draw = T
            if T <= 0:
                chosen = self._batch(res, dE)
                if not chosen:
                    break
            else:
                logit = np.r_[-dE, 0.0] / T
                p = np.exp(logit - logit.max())
                i = int(self.rng.choice(len(p), p=p / p.sum()))
                T = T * decay if step < steps_hot else 0.0
                if i == len(res):
                    stays += 1
                    continue
                chosen = [i]
            for i in chosen:
                d, m, prop, dp = res[i]
                rec = self._record(step, T_draw, m, prop, d, dp)
                if m["kind"] == "delete":
                    self.graveyard[self._gid] = self.net.edges[m["anchor"][0]].copy()
                    rec["grave"] = self._gid
                    self._gid += 1
                elif m["kind"] == "revive":
                    del self.graveyard[m["grave"]]
                    rec["grave"] = int(m["grave"])
                elif m["kind"] == "birth" or "bid" in m:       # a birth taken, or grown into
                    self.births.pop(m["bid"], None)
                ids = self.apply(prop)
                rec["new"] = [int(k) for k in ids]
                self.move_log.append(rec)
                if m["kind"] in ("join", "reroute") and ids:
                    e = self.net.edges[ids[0]]
                    e.info["links"] = list(e.info.get("links", [])) + [dict(
                        kind=m["kind"], evidence="energy", dE=round(float(d), 1),
                        frag=round(float(dp), 1), xy=[round(float(v), 1) for v in m["focus"]])]
                acc[m["kind"]] = acc.get(m["kind"], 0) + 1
            if self.cfg.verbose and (step % 25 == 0):
                G = self._graph()
                self.log(f"step {step}: T = {T:.1f}, {len(res)} moves, took {len(chosen)} "
                         f"({m['kind']}, dE = {d:.0f}, of which Phi {dp:.0f}); "
                         f"{len(self.net.edges)} vessels; Phi {G.total:.0f} ({G.matched} "
                         f"matched of {G.n_facing} facing pairs); {self.n_evaluated} local fits")
        return dict(steps=step + 1, accepted=acc, stays=stays)

    def _batch(self, res, dE):
        """At T = 0: the improving moves to take in one step, best first,
        each only if its window (padded by join_gap while Phi is on, as Phi
        couples ends that far apart) overlaps no window taken before it and
        it touches no vessel taken before.  Moves on disjoint windows do not
        interact, so their scores add exactly."""
        pad = self.cfg.join_gap if self._phi_on() else 0.0
        taken, boxes, used = [], [], set()
        for i in np.argsort(dE, kind="stable"):
            if dE[i] >= -max(self.cfg.min_gain, 1e-3) or (taken and not self.cfg.greedy_batch):
                break
            prop = res[i][2]
            x0, y0, x1, y1 = prop[3]
            box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
            ids = set(prop[0])
            if ids & used or any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                                 for b in boxes):
                continue
            taken.append(int(i))
            boxes.append(box)
            used |= ids
        return taken

    def _attach_scorer(self, log):
        """accept() for to_through: an edit of the representation (a branch
        end brought onto its parent, two ends made one) is scored like a
        move: the changed vessels are fitted near the junction (the nodes
        held where the edit put them, so shared ends stay shared) on their
        window over the rest of the model, and the edit is kept only if it
        then changes NLL + prior by at most attach_max * tau * lam_vessel.
        A kept edit takes the fitted geometry and updates the rendered
        state."""
        limit = self.cfg.attach_max * self.tau * self.cfg.lam_vessel

        def accept(before, ids, focus):
            old = [before[k] for k in ids if k in before]
            new_ids = [k for k in ids if k in self.net.edges]
            new = [self.net.edges[k] for k in new_ids]
            if not old and not new:
                return True
            win = self._window([edge_samples(e) for e in old + new])
            removed = [k for k in ids if k in self.patch]
            n0, p0 = self._local_energy(self._local(old, win, removed))
            m_new = self._local(new, win, removed)
            if new:
                free = self._free(m_new, np.asarray(focus, float) - [win[0], win[1]])
                free["node"][:] = False
                self._fit(m_new, self.cfg.local_iters, free)
            n1, p1 = self._local_energy(m_new)
            d = (n1 + p1) - (n0 + p0)
            log.append(round(d, 1))
            if d > limit:
                return False
            for k, f in zip(new_ids, self._fitted_edges(m_new, win) if new else []):
                e = self.net.edges[k]
                e.ctrl, e.r, e.s, e.a = f.ctrl, f.r, f.s, f.a
            for k in removed:
                p, v = self.patch.pop(k)
                self.V[p] -= v
            for k, (p, v) in zip(new_ids, self._patches(m_new, win) if new else []):
                self.patch[k] = (p, v)
                self.V[p] += v
            return True
        return accept

    def _record(self, step, T, m, prop, d, dp):
        """move_log entry of an accepted move (before it is applied): its
        dE split into data (nll, prior), cost and Phi; for a join the
        weight of the pair it joins (a join releasing more Phi than that
        strands other ends), for a delete what the vessel explained."""
        parts = dict(prop[5]["parts"])
        # a cached score's correction for changes around the move (see
        # _lookup) is a change of the data NLL alone
        parts["nll"] += (d - dp) - (parts["nll"] + parts["prior"] + parts["cost"])
        rec = dict(step=int(step), T=round(float(T), 2), kind=m["kind"],
                   anchor=[int(k) for k in m["anchor"]], dE=round(float(d), 1),
                   nll=round(parts["nll"], 1), prior=round(parts["prior"], 1),
                   cost=round(parts["cost"], 1), frag=round(float(dp), 1))
        if m["kind"] == "join" and self._phi_on():
            a, b = m["key"][1:]
            G = self._graph()
            rec["pair_weight"] = round(G.weight(a, b), 1)
            rec["ends_S"] = [round(self.endS[k][e], 1) for k, e in (a, b)]
        if m["kind"] == "delete":
            k = m["anchor"][0]
            s = self.samples(k)
            rec.update(L=round(float(s["L"]), 1), vessel_cost=round(self.cost(self.net.edges[k],
                                                                              s["L"]), 1),
                       loss=round(parts["nll"] + parts["prior"], 1),
                       release=round(self._release_bound(prop[0]), 1),
                       duplicates=[int(j) for j in prop[0][1:]],
                       xy=np.round(edge_samples(self.net.edges[k], 2.0)["xy"], 1).tolist())
        return rec

    def run(self):
        cfg = self.cfg
        n0 = len(self.net.edges)
        E0 = self.energy_total()
        oms = [r[4] for r in self._graph().rows]
        self.log(f"search: {n0} vessels, tau = {self.tau:.2f}, price {self.price:.1f} per px, "
                 f"vessel {self.tau * cfg.lam_vessel:.0f}, end om median "
                 f"{np.median(oms) if oms else 0:.0f}; E = {E0['total']:.0f} "
                 f"(nll {E0['nll']:.0f}, prior {E0['prior']:.0f}, cost {E0['cost']:.0f}, "
                 f"Phi {E0['frag']:.0f}: {E0['matched_pairs']} matched of "
                 f"{E0['facing_pairs']} facing pairs)")
        T0 = cfg.t_start * self.tau * cfg.t_scale
        if cfg.births:
            self.refresh_births()
        hist = [self.anneal(T0, max(10, int(cfg.hot_steps * n0)))]
        self.log(f"annealed: {hist[-1]}; {len(self.net.edges)} vessels")
        for _ in range(cfg.birth_rounds if cfg.births else 0):
            self.refresh_births()               # what the map still leaves unexplained
            hist.append(self.anneal(0.0, 0))
            self.log(f"with fresh births: {hist[-1]}; {len(self.net.edges)} vessels")
        if cfg.global_iters > 0:
            phi = self._graph().total
            self.global_fit(cfg.global_iters)
            # the fits do not see Phi (in the gradient it would dim the
            # ends of fragments), so it may jump here
            self.log(f"joint fit: Phi {phi:.0f} -> {self._graph().total:.0f}")
            if cfg.births:
                self.refresh_births()
            hist.append(self.anneal(0.0, 0))
            self.log(f"after the joint fit: {hist[-1]}; {len(self.net.edges)} vessels")
        E_an = self.energy_total()
        attach_log = []
        with _one_thread():
            to_through(self.net, cfg.attach_tol, accept=self._attach_scorer(attach_log))
        limit = cfg.attach_max * self.tau * cfg.lam_vessel
        refused = sum(d > limit for d in attach_log)
        self.log(f"branch points: {len(attach_log) - refused} edits, {refused} refused "
                 f"(they would change NLL + prior by more than {limit:.0f})")
        self.net.snap_through_nodes()
        self.global_fit(cfg.through_iters)
        E1 = self.energy_total()
        self.net.orient_structural()
        m = NetworkModel(self.net, self.P.logI, self.P.weight, stride=1,
                         bg_spacing=self.net.bg_spacing or BG_SPACING)
        self._render_all(m)              # edges may be reversed: renew what is kept per end
        gains, _ = m.edge_gains()
        for k, eid in enumerate(m.eids):
            L = max(self.samples(eid)["L"], 1.0)
            self.net.edges[eid].info.update(gain=float(gains[k]), gain_per_px=float(gains[k] / L))
        self.net.meta["search"] = dict(
            config={k: v for k, v in asdict(cfg).items()}, tau=self.tau,
            texture_null=self.null, price_per_px=self.price,
            vessels_before=n0, vessels_after=len(self.net.edges),
            energy_before=E0, energy_annealed=E_an, energy_after=E1, steps=hist,
            local_fits=self.n_evaluated, seconds=round(time.time() - self.t0, 1),
            attach=dict(edits=len(attach_log) - refused, refused=refused,
                        dE=sorted(attach_log)),
            moves=[{k: v for k, v in r.items() if k != "xy"} for r in self.move_log],
            deleted=[r for r in self.move_log if r["kind"] == "delete"])
        self.net.meta["final_nll"] = E1["nll"]
        self.log(f"done: {n0} -> {len(self.net.edges)} vessels; E {E0['total']:.0f} -> "
                 f"{E1['total']:.0f} (nll {E0['nll']:.0f} -> {E1['nll']:.0f}); "
                 f"{self.n_evaluated} local fits")
        return self.net


_SHARED = None


class _one_thread:
    """Local models are small: torch's intra-op threads only add overhead
    to them (tens of times slower when the cores are busy, as they are
    while workers score moves).  The joint fits keep their threads."""
    def __enter__(self):
        self.n = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self.n)


def _evaluate_chunk(idx):
    """Worker: score some of the moves of the forked search state."""
    torch.set_num_threads(1)
    try:
        # torch.compile (render.entry_core) may recompile here; its compile
        # workers belong to the parent and would never answer a forked child
        import torch._inductor.config as inductor_config
        inductor_config.compile_threads = 1
    except Exception:
        pass
    C, moves = _SHARED
    return [(i, C._compute(moves[i])) for i in idx]


def _end_path(smp, a, p, slack=2.0):
    """(xy, r, s, a) an edge keeps when end `a` is brought to p along its
    own course.  Only the end stretch is looked at (arclength up to
    |p - end| + slack from the end): the edge is cut at the sample of that
    stretch nearest p, which becomes p, or, when that sample is the end
    itself, extended straight to p.  None if too little would be left."""
    xy, arc = smp["xy"], smp["s_arc"]
    p = np.asarray(p, float)
    i_end = 0 if a == 0 else len(xy) - 1
    D = float(np.linalg.norm(p - xy[i_end]))
    reach = D + slack
    idx = np.flatnonzero(arc <= reach) if a == 0 else np.flatnonzero(arc >= arc[-1] - reach)
    i = int(idx[np.argmin(np.linalg.norm(xy[idx] - p, axis=1))])
    prof = [smp[k] for k in ("r", "s", "a")]
    if i == i_end and D > 0.5:                    # extend straight to p
        if a == 0:
            return (np.vstack([p, xy]), *[np.r_[q[0], q] for q in prof])
        return (np.vstack([xy, p]), *[np.r_[q, q[-1]] for q in prof])
    sl = slice(i, None) if a == 0 else slice(0, i + 1)
    out = xy[sl].copy()
    if len(out) < 3:
        return None
    out[0 if a == 0 else -1] = p
    return (out, *[q[sl] for q in prof])


def _move_node(net: VesselNetwork, nid, p, limit):
    """Move node nid to p, bringing every edge that ends there along its own
    course (_end_path: cut back, or extended straight) and re-fitting it
    faithfully, so no edge folds back as it would if only the node moved.
    Refused (nothing changes) when any of those ends is farther than
    `limit` from p: bigger edits would change what the map renders.
    Through nodes left off a cut edge (they were on the cut-away stub) are
    dropped from its through list.  Returns (ok, the ids of the edges
    changed, the through nodes dropped)."""
    inc = net.incident(nid)
    p = np.asarray(p, float)
    paths = []
    for f, end in inc:                       # sampled before the node moves (sampling
        smp = net.sample(f, 0.5)             # syncs the ends to the nodes)
        if np.linalg.norm(smp["xy"][0 if end == 0 else -1] - p) > limit:
            return False, [], []
        path = _end_path(smp, end, p)
        if path is None:                     # every edge must keep enough of itself
            return False, [], []
        paths.append((f, path))
    net.nodes[nid].x, net.nodes[nid].y = float(p[0]), float(p[1])
    for f, path in paths:
        net.refit_edge(f, *path, faithful=True)
    dropped = []
    for f, _ in inc:
        e = net.edges[f]
        if not e.through:
            continue
        smp = net.sample(f, 1.0)
        keep = []
        for t in e.through:
            d = np.linalg.norm(smp["xy"] - net.nodes[t].xy, axis=1)
            j = int(np.argmin(d))
            if d[j] <= smp["r"][j] + smp["s"][j] + 2.0 and 0 < j < len(d) - 1:
                keep.append(t)
            else:
                dropped.append(t)
        if keep:
            e.info["through"] = keep
        else:
            e.info.pop("through", None)
    net._touch()
    return True, [f for f, _ in inc], dropped


def _snapshot(net: VesselNetwork):
    return ({k: Node(v.x, v.y, v.fixed_kind) for k, v in net.nodes.items()},
            {k: e.copy() for k, e in net.edges.items()}, net._nid, net._eid)


def _restore(net: VesselNetwork, snap):
    net.nodes, net.edges, net._nid, net._eid = snap[0], snap[1], snap[2], snap[3]
    net._touch()


def _can_merge(net: VesselNetwork, keep, drop) -> bool:
    """Whether node drop may be merged into keep without breaking the
    representation: no edge ending at both (a loop), no edge ending at a
    node it passes through, nothing attached to an image-border node."""
    if keep == drop or keep not in net.nodes or drop not in net.nodes:
        return False
    if "border" in (net.node_kind(keep, 1), net.node_kind(drop, 1)):
        return False
    for f_id, end in net.incident(drop):
        f = net.edges[f_id]
        other = f.v if end == 0 else f.u
        if other == keep or keep in f.through:
            return False
    for f_id in net.passing(drop):
        if keep in (net.edges[f_id].u, net.edges[f_id].v):
            return False
    for f_id, end in net.incident(keep):
        if drop in net.edges[f_id].through:
            return False
    return True


def _merge_ends(net: VesselNetwork, keep, drop, limit):
    """Merge node drop into keep, if _can_merge allows it: the edges ending
    at drop are first brought along their own course onto keep (at most
    `limit` px, see _move_node).  Through lists that named drop name keep.
    Returns (ok, edges changed, through nodes dropped)."""
    if not _can_merge(net, keep, drop):
        return False, [], []
    ok, changed, dropped = _move_node(net, drop, net.nodes[keep].xy, limit)
    if not ok:
        return False, [], []
    for f_id in net.passing(drop):
        f = net.edges[f_id]
        f.info["through"] = [keep if n == drop else n for n in f.through]
    net.merge_nodes(keep, drop)
    return True, changed, dropped


def to_through(net: VesselNetwork, tol=1.5, end_tol=3.0, parallel_deg=12.0, accept=None):
    """Branch points in the map's representation (network.py), from vessel
    ends that lie on another vessel (within its r + s + tol).  The ends are
    settled one at a time, closest contact first, each against the current
    geometry (every change re-samples the edges it touched):

    * continuation: the other vessel has an end lying on this vessel's
      centreline (within half their calibre, as this end lies on the
      other's) that faces this end (their outward directions oppose): the
      vessels overlap at their ends.  Both ends are brought along their own
      course to the point midway between them and share one node, a joint;
    * side by side: the end runs (nearly) parallel to the other vessel and
      is no continuation of it: two vessels next to each other, not a branch
      point; nothing is recorded;
    * on the other vessel's own end (within end_tol px of arclength): the
      end shares that end's node;
    * otherwise a branch: the end becomes a node the other vessel passes
      through (``info["through"]``), brought along its own course onto the
      other's centreline, or joins a through node already within 3 px.

    Every edit is bounded: an end moves at most about the footprint it lies
    in (a continuation: half the calibre + end_tol; otherwise the other
    vessel's r + s + tol, + end_tol at its end), cut back along its end
    stretch or extended straight (_move_node).  An end that would need more
    stays free: this is a change of representation, and must not change
    what the map renders (long overlaps are left to the search's joins).

    A vessel never attaches both its ends to the same vessel (a stub lying
    along it is no branch), no merge may make an edge end at a node it
    passes through, form a loop or touch the image border (_can_merge), and
    ends whose through node is dropped by a later cut are settled again.

    accept(edges before, ids of the edges changed or removed, the junction)
    -> bool, if given, sees every edit and may refuse it: the edit is undone
    and that end stays free (VesselSearch refuses edits that change the fit
    much, see VesselSearch._attach_scorer; it may also refine the changed
    edges' geometry)."""
    import heapq
    geo, index = {}, [None]
    node_of = lambda k, a: net.edges[k].u if a == 0 else net.edges[k].v

    def smp(k):
        if k not in geo:
            geo[k] = net.sample(k, 1.0)
        return geo[k]

    def changed(ks):
        for k in ks:
            geo.pop(k, None)
        index[0] = None

    def tree():
        if index[0] is None:
            ids = list(net.edges)
            X = np.concatenate([smp(k)["xy"] for k in ids])
            lab = np.concatenate([np.full(len(smp(k)["xy"]), k) for k in ids])
            idx = np.concatenate([np.arange(len(smp(k)["xy"])) for k in ids])
            index[0] = (cKDTree(X), X, lab, idx)
        return index[0]

    def end_geom(k, a):
        xy = smp(k)["xy"]
        if a == 0:
            return xy[0], -_direction(xy, 0, True)
        return xy[-1], _direction(xy, len(xy) - 1, False)

    def dist_to(p, k):
        s = smp(k)
        d = np.linalg.norm(s["xy"] - p, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), i

    def contact(k, a):
        if len(net.edges) < 2 or net.node_kind(node_of(k, a), 1) == "border":
            return None
        p, _ = end_geom(k, a)
        T, X, lab, idx = tree()
        best = None
        for c in T.query_ball_point(p, 30.0):
            j, i = int(lab[c]), int(idx[c])
            if j == k:
                continue
            s = smp(j)
            d = float(np.linalg.norm(X[c] - p))
            if d <= s["r"][i] + s["s"][i] + tol and (best is None or d < best[0]):
                best = (d, j, i)
        return best

    heap, attached, refused = [], {}, set()
    for k in list(net.edges):
        for a in (0, 1):
            c = contact(k, a)
            if c is not None:
                heap.append((c[0], k, a))
    heapq.heapify(heap)
    cos_par = math.cos(math.radians(parallel_deg))
    guard = 0
    while heap and guard < 20 * len(net.edges) + 100:
        guard += 1
        _, k, a = heapq.heappop(heap)
        if k not in net.edges or (k, a) in attached or (k, a) in refused:
            continue
        c = contact(k, a)
        if c is None:
            continue
        d, j, i = c
        if attached.get((k, 1 - a)) == j:          # a stub along j, not a branch
            continue
        nid = node_of(k, a)
        p, t = end_geom(k, a)
        sj = smp(j)
        done, made = None, []
        snap = _snapshot(net) if accept is not None else None
        # continuation: an end of j lies on k and faces this end
        cont = []
        sk = smp(k)
        for b in (0, 1):
            nb = node_of(j, b)
            if nb == nid or net.node_kind(nb, 1) == "border" or attached.get((j, 1 - b)) == k:
                continue
            pb, tb = end_geom(j, b)
            db, ib = dist_to(pb, k)
            # one vessel traced twice: the centrelines coincide (within half
            # the calibre); farther apart they are two vessels side by side
            lat = max(1.5, 0.5 * (sk["r"][ib] + sj["r"][i]))
            if np.dot(t, tb) < -0.5 and max(d, db) <= lat:
                cont.append((float(np.linalg.norm(pb - p)), b, nb, pb))
        if cont:
            _, b, nb, pb = min(cont)
            if _can_merge(net, nb, nid):
                ok, ch1, dr1 = _move_node(net, nb, 0.5 * (p + pb), lat + end_tol)
                if ok:
                    ok2, ch2, dr2 = _merge_ends(net, nb, nid, lat + end_tol)
                    done = (ch1 + ch2, dr1 + dr2)
                    if ok2:
                        made = [((k, a), j), ((j, b), k)]
        elif abs(float(np.dot(t, sj["tan"][i]))) > cos_par:
            continue                                # side by side: no branch point
        else:
            L, sa = float(sj["s_arc"][-1]), float(sj["s_arc"][i])
            reach = float(sj["r"][i] + sj["s"][i]) + tol      # an end moves at most this
            if min(sa, L - sa) < end_tol:           # on j's own end: share its node
                ok, ch, dr = _merge_ends(net, net.edges[j].u if sa < 0.5 * L else net.edges[j].v,
                                         nid, reach + end_tol)
                done = (ch, dr)
                if ok:
                    made = [((k, a), j)]
            else:
                q = sj["xy"][i]
                ej = net.edges[j]
                near = [x for x in ej.through if x in net.nodes and
                        np.linalg.norm(net.nodes[x].xy - q) < 3.0]
                if near:                            # two branches leave at one point
                    ok, ch, dr = _merge_ends(net, near[0], nid, reach + 3.0)
                    done = (ch, dr)
                    if ok:
                        made = [((k, a), j)]
                elif nid not in ej.through and nid not in (ej.u, ej.v) and not net.passing(nid):
                    ok, ch, dr = _move_node(net, nid, q, reach)
                    done = (ch, dr)
                    if ok:
                        ej.info["through"] = list(ej.through) + [nid]
                        net._touch()
                        made = [((k, a), j)]
        if done and accept is not None:
            gone = [x for x in snap[1] if x not in net.edges]
            if not made or not accept(snap[1], sorted(set(done[0]) | set(gone)),
                                      net.nodes[node_of(k, a)].xy):
                _restore(net, snap)                 # undone: this end stays free
                refused.add((k, a))
                geo.clear()
                index[0] = None
                continue
        for key, val in made:
            attached[key] = val
        if done:
            ch, dr = done
            changed(set(ch) | {j})
            for x in dr:                            # branches whose through node was dropped
                if x in net.nodes:
                    for f, end in net.incident(x):
                        attached.pop((f, end), None)
                        cc = contact(f, end)
                        if cc is not None:
                            heapq.heappush(heap, (cc[0], f, end))
    # an edge never passes through its own end, nor twice through a node
    for e in net.edges.values():
        if e.through:
            keep = [n for n in dict.fromkeys(e.through) if n in net.nodes and n not in (e.u, e.v)]
            if keep:
                e.info["through"] = keep
            else:
                e.info.pop("through", None)
    net._touch()


def search_map(intensity: np.ndarray, net: VesselNetwork, cfg: SearchConfig | None = None,
               prepared: Prepared | None = None) -> VesselNetwork:
    """The fewest vessels (one spline each) that render the image as well as
    `net` does, by annealing over joins, deletions, splits and reroutes
    under the vessel-count energy (see the module docstring).  `net` is any
    map of the same image: segments from build_map / refine_map, or better
    the output of consolidate_map, which the search then continues.  It is
    not modified."""
    cfg = cfg or SearchConfig()
    P = prepared or prepare(intensity)
    return VesselSearch(net, P, cfg).run()
