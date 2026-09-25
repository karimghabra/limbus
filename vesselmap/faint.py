"""Faint vessels by path tracking: a second, recall-oriented tier.

Many thin, defocused vessels sit at the noise/texture floor pixel by pixel
(their ridge z-score is no higher than that of the tissue texture around
them), so neither the ridge detector nor the MDL test of `build_map` can
accept them.  Along their length, though, the evidence adds up.  This module
follows lines in the residual of an existing map (what the map does not
explain yet) and keeps paths whose evidence, integrated along the path, is
consistently vessel-like:

1. oriented line responses Z(theta, x) (elongated second-derivative filters,
   several scales, standardised by a robust local scale);
2. seeds at strong local maxima away from the mapped vessels;
3. a tracker that steps along the line with limited turning, stopping when
   the running evidence fades, when it reaches a mapped vessel (a join) or
   an already accepted track;
4. the path is trimmed to its maximal-evidence stretch, smoothed, and
   re-scored along its OWN tangent (removing most of the selection bias of
   the greedy steps); paths that are long enough with enough mean evidence
   are kept.

Accepted paths join the network as edges with info["tier"] = "faint".
They are meant to widen the search mask for later (velocity) analysis,
not as confirmed vessels: `score_and_prune` would reject most of them.

Like everything in vesselmap, only the single image is used.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .network import VesselNetwork
from .ridges import oriented_kernel


@dataclass
class FaintConfig:
    scales: tuple = (1.2, 2.0, 3.0)   # filter sigma across the line (px)
    n_orient: int = 16
    elong: float = 6.0                # filter length / width
    zone_scale: float = 1.0           # no tracking within zone_scale*r+zone_pad
    zone_pad: float = 2.0             # of a mapped centreline
    seed_z: float = 4.0               # seeds: local maxima of Z above this
    step: float = 2.0                 # tracking step (px)
    turn_bins: int = 1                # max turn per step, in orientation bins
    window: int = 8                   # steps in the running-evidence window
    stop_z: float = 1.2               # stop when the running mean falls below
    drift: float = 2.0                # trimming: keep the stretch maximising sum(z - drift)
    smooth: float = 6.0               # px, smoothing of accepted paths
    min_length: float = 30.0          # px
    min_mean: float = 2.5             # mean tangent-oriented Z along the path must
    sig_a: float = 2.0                # exceed max(min_mean, sig_a + sig_b/sqrt(L)):
    sig_b: float = 12.0               # calibrated on pure (white and correlated)
                                      # noise, where this admits ~4 paths per
                                      # 1920x1200 image (sig_a 2.2: < 1)
    join_px: float = 6.0              # an end this close to the zone joins the map
    bridge_cos: float = 0.5           # a join bridge must continue the path (cos of turn)
    hug_px: float = 4.0               # a path running parallel within zone + hug_px
    hug_frac: float = 0.6             # of a mapped vessel for this fraction of its
                                      # length is a misfit edge, not a new vessel
    max_length: float = 3000.0
    fit_profiles: bool = True         # fit width/blur/contrast of the new edges
    fit_iters: int = 40
    verbose: bool = True


# ------------------------------------------------------------------ evidence
def _robust_tile_scale(R: np.ndarray, tile: int = 96) -> np.ndarray:
    """Robust (MAD) scale of the responses R[orient, y, x] over tiles."""
    _, h, w = R.shape
    ny, nx = max(1, h // tile), max(1, w // tile)
    est = np.zeros((ny, nx), np.float32)
    for iy in range(ny):
        for ix in range(nx):
            blk = R[:, iy * h // ny:(iy + 1) * h // ny, ix * w // nx:(ix + 1) * w // nx][:, ::2, ::2]
            est[iy, ix] = 1.4826 * np.median(np.abs(blk - np.median(blk)))
    est = ndi.median_filter(np.maximum(est, 1e-9), size=3, mode="nearest")
    return cv2.resize(est, (w, h), interpolation=cv2.INTER_CUBIC)


def orientation_scores(V: np.ndarray, scales, n_orient=16, elong=6.0, valid=None):
    """Z[k] = standardised line response of V (lines bright) in orientation
    k*pi/n_orient, maximised over scales; also the arg-max scale."""
    vv = V.astype(np.float32)
    if valid is not None:
        vv = np.where(valid, vv, ndi.median_filter(vv, 5)).astype(np.float32)
    thetas = np.arange(n_orient) * math.pi / n_orient
    Z = S = None
    for s in scales:
        R = np.stack([cv2.filter2D(vv, cv2.CV_32F, oriented_kernel(float(s), float(t), elong),
                                   borderType=cv2.BORDER_REFLECT) for t in thetas])
        R /= _robust_tile_scale(R)[None]
        if Z is None:
            Z, S = R, np.full(R.shape, s, np.float32)
        else:
            upd = R > Z
            Z = np.where(upd, R, Z)
            S = np.where(upd, np.float32(s), S)
    if valid is not None:
        Z[:, ~valid] = 0
    return Z, S, thetas


def _bilinear(A, x, y):
    return ndi.map_coordinates(A, [np.atleast_1d(y), np.atleast_1d(x)], order=1, mode="nearest")


def path_evidence(Z: np.ndarray, xy: np.ndarray, S: np.ndarray | None = None):
    """Z along a path in the orientation of the path's own tangent
    (interpolated between orientation bins); optionally the matching scale."""
    n = Z.shape[0]
    d = np.gradient(xy, axis=0)
    a = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi) / (np.pi / n)
    k0 = np.floor(a).astype(int) % n
    k1 = (k0 + 1) % n
    f = a - np.floor(a)
    z0 = np.array([_bilinear(Z[k], x, y)[0] for k, x, y in zip(k0, xy[:, 0], xy[:, 1])])
    z1 = np.array([_bilinear(Z[k], x, y)[0] for k, x, y in zip(k1, xy[:, 0], xy[:, 1])])
    z = (1 - f) * z0 + f * z1
    if S is None:
        return z
    s = np.where(f < 0.5,
                 [_bilinear(S[k], x, y)[0] for k, x, y in zip(k0, xy[:, 0], xy[:, 1])],
                 [_bilinear(S[k], x, y)[0] for k, x, y in zip(k1, xy[:, 0], xy[:, 1])])
    return z, s


# ------------------------------------------------------------------ tracking
class _Tracker:
    """Greedy line follower on the orientation scores."""
    FREE, MAP = 0, -1

    def __init__(self, Z, block, cfg: FaintConfig):
        self.Z, self.block, self.cfg = Z, block, cfg
        self.n = Z.shape[0]
        self.h, self.w = Z.shape[1:]
        self.max_steps = int(cfg.max_length / cfg.step)

    def _z(self, k, x, y):
        if not (1 <= x < self.w - 2 and 1 <= y < self.h - 2):
            return None
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        Zk = self.Z[k % self.n]
        return ((1 - fx) * (1 - fy) * Zk[y0, x0] + fx * (1 - fy) * Zk[y0, x0 + 1]
                + (1 - fx) * fy * Zk[y0 + 1, x0] + fx * fy * Zk[y0 + 1, x0 + 1])

    def _follow(self, x, y, kdir):
        """Follow the line from (x, y) in directed orientation bin kdir
        (angle kdir*pi/n, kdir in [0, 2n))."""
        c = self.cfg
        pts, zs = [], []
        n2 = 2 * self.n
        for _ in range(self.max_steps):
            best = None
            for dk in range(-c.turn_bins, c.turn_bins + 1):
                kd = (kdir + dk) % n2
                a = kd * math.pi / self.n
                cx, cy = x + c.step * math.cos(a), y + c.step * math.sin(a)
                nx, ny = -math.sin(a), math.cos(a)
                for off in (-1.0, 0.0, 1.0):
                    qx, qy = cx + off * nx, cy + off * ny
                    z = self._z(kd, qx, qy)
                    if z is None:
                        continue
                    z -= 0.15 * abs(dk) + 0.1 * abs(off)
                    if best is None or z > best[0]:
                        best = (z, qx, qy, kd)
            if best is None:
                break                                   # image border
            z, x, y, kdir = best
            pts.append((x, y))
            zs.append(z)
            if self.block[int(round(y)), int(round(x))] != self.FREE:
                break                                   # map or another track
            if len(zs) >= c.window and np.mean(zs[-c.window:]) < c.stop_z:
                break
        return pts, zs

    def trace(self, x, y, k):
        f, zf = self._follow(x, y, k)
        b, zb = self._follow(x, y, k + self.n)
        pts = np.array(b[::-1] + [(x, y)] + f)
        zs = np.array(zb[::-1] + [float(self.Z[k, int(y), int(x)])] + zf)
        return pts, zs


def _max_stretch(z, c):
    """[i, j) maximising sum(z[i:j] - c) (Kadane)."""
    best, cur, bi, bj, ci = -np.inf, 0.0, 0, 0, 0
    for j, v in enumerate(z - c):
        if cur <= 0:
            cur, ci = v, j
        else:
            cur += v
        if cur > best:
            best, bi, bj = cur, ci, j + 1
    return bi, bj


def smooth_path(xy, spacing=2.0, sm=6.0):
    """Gaussian-smooth a polyline along its arclength and resample it."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if s[-1] < 2 * spacing:
        return np.asarray(xy, float)
    q = np.arange(0, s[-1] + 1e-9, 1.0)
    p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
    p = ndi.gaussian_filter1d(p, sm, axis=0, mode="nearest")
    s2 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    q2 = np.arange(0, s2[-1] + 1e-9, spacing)
    return np.stack([np.interp(q2, s2, p[:, 0]), np.interp(q2, s2, p[:, 1])], 1)


def map_zone(net: VesselNetwork, shape, scale=1.0, pad=2.0) -> np.ndarray:
    """Pixels within scale*r + pad of a mapped centreline."""
    m = np.zeros(shape, np.uint8)
    for eid in net.edges:
        smp = net.sample(eid, 1.0)
        rad = scale * float(np.median(smp["r"])) + pad
        cv2.polylines(m, [np.round(smp["xy"] * 8).astype(np.int32)], False, 1,
                      max(1, int(round(2 * rad)) + 1), shift=3)
    return m > 0


class _MapRef:
    """Nearest mapped centreline point, its tangent and zone radius."""

    def __init__(self, net: VesselNetwork, cfg: FaintConfig):
        xy, tan, rad = [], [], []
        for eid, e in net.edges.items():
            if e.info.get("tier") == "faint":
                continue
            smp = net.sample(eid, 1.0)
            xy.append(smp["xy"])
            tan.append(smp["tan"])
            rad.append(np.full(len(smp["xy"]), cfg.zone_scale * float(np.median(smp["r"]))
                               + cfg.zone_pad))
        self.xy, self.tan, self.rad = np.concatenate(xy), np.concatenate(tan), np.concatenate(rad)
        self.tan = self.tan / np.maximum(np.linalg.norm(self.tan, axis=1, keepdims=True), 1e-9)
        self.tree = cKDTree(self.xy)

    def hug_fraction(self, xy, hug_px):
        d, j = self.tree.query(xy, k=4)
        t = np.gradient(xy, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        cos = np.abs(np.einsum("nkc,nc->nk", self.tan[j], t))
        hug = (d <= self.rad[j] + hug_px) & (cos > 0.85)
        return float(hug.any(1).mean())


def trace_faint(res: np.ndarray, zone: np.ndarray, cfg: FaintConfig | None = None,
                valid=None, polarity: int = 1, log=None, ref: _MapRef | None = None):
    """Find faint line paths in a residual image (logI - model; vessels are
    negative there).  polarity=-1 looks for bright lines instead, which is
    the null used to calibrate the thresholds.  Returns a list of dicts
    (xy, z, scale, L, mean, cv, ends) for the accepted paths."""
    cfg = cfg or FaintConfig()
    t0 = time.time()
    Z, S, thetas = orientation_scores(-polarity * res, cfg.scales, cfg.n_orient, cfg.elong, valid)
    block = np.where(zone, _Tracker.MAP, _Tracker.FREE).astype(np.int32)
    dist_zone = ndi.distance_transform_edt(~zone)
    zmax, kmax = Z.max(0), Z.argmax(0)
    loc = (zmax == ndi.maximum_filter(zmax, 7)) & (zmax > cfg.seed_z) & ~zone
    ys, xs = np.nonzero(loc)
    order = np.argsort(-zmax[ys, xs])
    T = _Tracker(Z, block, cfg)
    H, W = zone.shape
    tracks, n_tried = [], 0
    for i in order:
        y, x = ys[i], xs[i]
        if block[y, x] != _Tracker.FREE:
            continue
        n_tried += 1
        pts, zs = T.trace(float(x), float(y), int(kmax[y, x]))
        i0, i1 = _max_stretch(zs, cfg.drift)
        if cfg.step * (i1 - i0 - 1) < cfg.min_length:
            continue
        xy = smooth_path(pts[i0:i1], cfg.step, cfg.smooth)
        L = cfg.step * (len(xy) - 1)
        if L < cfg.min_length:
            continue
        z, sc = path_evidence(Z, xy, S)
        m = float(z.mean())
        if m < max(cfg.min_mean, cfg.sig_a + cfg.sig_b / math.sqrt(L)):
            continue
        if ref is not None and ref.hug_fraction(xy, cfg.hug_px) > cfg.hug_frac:
            continue
        ends = []
        for p in (xy[0], xy[-1]):
            ix, iy = int(np.clip(round(p[0]), 0, W - 1)), int(np.clip(round(p[1]), 0, H - 1))
            if dist_zone[iy, ix] <= cfg.join_px:
                ends.append("map")
            elif (block[max(iy - 4, 0):iy + 5, max(ix - 4, 0):ix + 5] > 0).any():
                ends.append("faint")
            else:
                ends.append("free")
        tracks.append(dict(xy=xy, z=z, scale=sc, L=L, mean=m,
                           cv=float(z.std() / max(m, 1e-3)), ends=ends))
        claim = np.zeros(zone.shape, np.uint8)
        cv2.polylines(claim, [np.round(xy * 8).astype(np.int32)], False, 1, 5, shift=3)
        block[(claim > 0) & (block == _Tracker.FREE)] = len(tracks)
    if log:
        log(f"faint tracking: {len(ys)} seeds, {n_tried} traced -> {len(tracks)} paths, "
            f"{sum(t['L'] for t in tracks):.0f} px in {time.time() - t0:.0f}s")
    return tracks


# ------------------------------------------------------------------ network
def _nearest_on_map(net: VesselNetwork, eids):
    xy, owner = [], []
    for eid in eids:
        p = net.sample(eid, 1.0)["xy"]
        xy.append(p)
        owner.append(np.full(len(p), eid))
    return cKDTree(np.concatenate(xy)), np.concatenate(xy), np.concatenate(owner)


def add_faint(net: VesselNetwork, tracks, cfg: FaintConfig | None = None, reach=None):
    """Add accepted paths as edges (info["tier"] = "faint").  An end that
    stopped at the map is extended straight onto the nearest mapped
    centreline (within `reach` px) and joined there.  Returns the new edge ids."""
    cfg = cfg or FaintConfig()
    base = [e for e in net.edges if net.edges[e].info.get("tier") != "faint"]
    if not base:
        return []
    new = []
    for t in tracks:
        xy = np.asarray(t["xy"], float)
        sc = np.asarray(t["scale"], float)
        # rough profile: a line of Gaussian profile s is found at filter
        # scale ~s; the contrast is refitted below when fit_profiles is set
        r = np.full(len(xy), 1.0)
        s = np.clip(sc, 0.8, 4.0)
        a = np.full(len(xy), 0.05)
        nodes = [None, None]
        for side, end in ((0, 0), (1, -1)):
            if t["ends"][side] != "map":
                continue
            tree, pts, owner = _nearest_on_map(net, base)
            d, j = tree.query(xy[end])
            lim = reach if reach is not None else (
                cfg.zone_scale * float(np.median(net.sample(int(owner[j]), 1.0)["r"]))
                + cfg.zone_pad + cfg.join_px + 2.0)
            if d > lim:
                continue
            if d > 3.0:
                # the bridge must roughly continue the path's direction
                k3 = min(3, len(xy) - 1)
                out = xy[0] - xy[k3] if side == 0 else xy[-1] - xy[-1 - k3]
                if np.dot(out, pts[j] - xy[end]) < cfg.bridge_cos * np.linalg.norm(out) * d:
                    continue
            nid = net.split_edge(int(owner[j]), pts[j])
            base = [e for e in net.edges if net.edges[e].info.get("tier") != "faint"]
            q = net.nodes[nid].xy
            # straight bridge from the path end to the join point
            k = max(1, int(math.ceil(d / cfg.step)))
            br = np.linspace(q, xy[end], k + 1)[:-1]
            if side == 0:
                xy = np.vstack([br, xy])
                r, s, a = (np.r_[np.full(k, v[0]), v] for v in (r, s, a))
            else:
                xy = np.vstack([xy, br[::-1]])
                r, s, a = (np.r_[v, np.full(k, v[-1])] for v in (r, s, a))
            nodes[side] = nid
        info = dict(tier="faint", evidence=round(float(t["mean"]), 2), protected=True)
        eid = net.add_edge_dense(xy, r, s, a, u=nodes[0], v=nodes[1], info=info, faithful=True)
        new.append(eid)
    return new


def fit_faint_profiles(net: VesselNetwork, P, cfg: FaintConfig, bg_spacing=64):
    """Fit width/blur/contrast of all edges with positions and background
    frozen (the mapped edges are already at their optimum, so in effect it
    fits the new ones)."""
    from .fit import MapConfig, optimize
    from .render import NetworkModel
    model = NetworkModel(net, P.logI, P.weight, stride=2, bg_spacing=bg_spacing)
    optimize(model, cfg.fit_iters, lr_prof=0.04, rebuild_every=0, fit_pos=False,
             fit_bg=False, priors=MapConfig().priors())
    model.write_back()
    return model


def add_faint_tier(intensity: np.ndarray, net: VesselNetwork, cfg: FaintConfig | None = None,
                   prepared=None, log=print) -> VesselNetwork:
    """Add the faint tier to a map of the same image (in place; returned)."""
    from .fit import residual_image
    from .image import prepare
    from .render import NetworkModel
    cfg = cfg or FaintConfig()
    P = prepared or prepare(intensity)
    t0 = time.time()
    say = (lambda *a: log(f"[{time.time() - t0:6.0f}s]", *a)) if (cfg.verbose and log) else None
    if net.has_through():
        seg = net.to_segments()
        net.nodes, net.edges = seg.nodes, seg.edges
        net._nid, net._eid, net._adj = seg._nid, seg._eid, None
    bg_spacing = net.bg_spacing or 64
    model = NetworkModel(net, P.logI, P.weight, stride=1, bg_spacing=bg_spacing)
    res = residual_image(model)
    zone = map_zone(net, P.shape, cfg.zone_scale, cfg.zone_pad)
    tracks = trace_faint(res, zone, cfg, P.valid, log=say, ref=_MapRef(net, cfg))
    new = add_faint(net, tracks, cfg)
    if cfg.fit_profiles and new:
        fit_faint_profiles(net, P, cfg, bg_spacing)
    for e in net.edges.values():
        e.info.pop("protected", None)
    L = sum(net.length(e) for e in new)
    net.meta.setdefault("faint", []).append(dict(
        seconds=round(time.time() - t0, 1), edges=len(new), length=round(L, 1),
        config={k: v for k, v in cfg.__dict__.items() if k != "verbose"}))
    if say:
        say(f"faint tier: +{len(new)} edges, +{L:.0f} px; {net.summary()}")
    return net


def search_mask(net: VesselNetwork, shape, pad=3.0, faint_radius=2.0) -> np.ndarray:
    """Label image of where to look for vessels: 1 = mapped vessel tube
    (radius r + pad), 2 = faint-tier tube (radius faint_radius + pad).
    Mapped vessels win where the two overlap."""
    out = np.zeros(shape, np.uint8)
    for want_faint, val in ((True, 2), (False, 1)):
        for eid, e in net.edges.items():
            if (e.info.get("tier") == "faint") != want_faint:
                continue
            smp = net.sample(eid, 1.0)
            rad = (faint_radius if want_faint else float(np.median(smp["r"]))) + pad
            cv2.polylines(out, [np.round(smp["xy"] * 8).astype(np.int32)], False, val,
                          max(1, int(round(2 * rad)) + 1), shift=3)
    return out
