"""Synthetic vessel images with known ground truth, and centreline metrics.

The generator is deliberately independent of render.py: every vessel is a
cylinder whose optical path is computed per pixel on a 2x super-sampled
grid, blurred with its own depth-dependent Gaussian, and the scene has a
lumpy textured background, illumination falloff and shot + read noise.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def _smooth_path(rng, start, heading, length, step=1.0, wiggle=0.08, corr=0.9):
    pts = [np.array(start, float)]
    h = heading
    dh = 0.0
    for _ in range(int(length / step)):
        dh = corr * dh + wiggle * rng.standard_normal()
        h += dh
        pts.append(pts[-1] + step * np.array([math.cos(h), math.sin(h)]))
    return np.array(pts)


def make_network(rng, shape, n_trees=4, n_cross=4, n_capillary=25):
    """Ground-truth vessels: list of dict(xy, r (per point), blur, amp)."""
    H, W = shape
    vessels = []
    bif = []

    def add_tree(start, heading, r0, blur, depth, length, parent=None):
        xy = _smooth_path(rng, start, heading, length, wiggle=0.015)
        r = np.linspace(r0, max(0.6, 0.8 * r0), len(xy))
        me = len(vessels)
        vessels.append(dict(xy=xy, r=r, blur=blur, amp=0.35 + 0.1 * rng.random(), parent=parent))
        if depth <= 0 or r0 < 1.0:
            return
        for _ in range(rng.integers(1, 3)):
            i = int(rng.integers(len(xy) // 5, 4 * len(xy) // 5))
            tan = xy[min(i + 3, len(xy) - 1)] - xy[max(i - 3, 0)]
            ang = math.atan2(tan[1], tan[0]) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.3)
            add_tree(xy[i], ang, max(0.7, r[i] * rng.uniform(0.45, 0.75)), blur, depth - 1,
                     length * rng.uniform(0.4, 0.7), parent=(me, i))

    for _ in range(n_trees):
        start = (rng.uniform(0, W), rng.choice([0, H - 1]))
        heading = -math.pi / 2 if start[1] > 0 else math.pi / 2
        heading += rng.uniform(-0.6, 0.6)
        add_tree(start, heading, rng.uniform(3, 9), rng.uniform(0.7, 1.5), 3, rng.uniform(0.6, 1.0) * H)
    # out-of-focus vessels that cross everything at another depth
    for _ in range(n_cross):
        start = (0, rng.uniform(0, H))
        xy = _smooth_path(rng, start, rng.uniform(-0.4, 0.4), W * 1.1, wiggle=0.008)
        r0 = rng.uniform(2, 10)
        vessels.append(dict(xy=xy, r=np.full(len(xy), r0), blur=rng.uniform(2.5, 9.0),
                            amp=0.25 + 0.25 * rng.random()))
    # small capillaries, tortuous, faint and sharp
    for _ in range(n_capillary):
        start = (rng.uniform(0, W), rng.uniform(0, H))
        xy = _smooth_path(rng, start, rng.uniform(0, 2 * math.pi), rng.uniform(40, 200),
                          wiggle=0.07, corr=0.85)
        vessels.append(dict(xy=xy, r=np.full(len(xy), rng.uniform(0.6, 1.4)),
                            blur=rng.uniform(0.6, 1.5), amp=0.12 + 0.2 * rng.random()))
    vessels = _cut_overlaps(vessels)
    for v in vessels:
        if v.get("parent") is not None:
            bif.append(v["xy"][0].copy())
        v.pop("parent", None)
        inside = (v["xy"][:, 0] >= -20) & (v["xy"][:, 0] < W + 20) & \
                 (v["xy"][:, 1] >= -20) & (v["xy"][:, 1] < H + 20)
        v["xy"], v["r"] = v["xy"][inside], v["r"][inside]
    vessels = [v for v in vessels if len(v["xy"]) > 10]
    return vessels, np.array(bif).reshape(-1, 2)


def _cut_overlaps(vessels, blur_max=2.0, run_factor=4.0, run_px=6.0, min_len=11):
    """In-focus vessels do not run inside one another (no image, and no
    annotator, can tell two vessels apart there).  In the order drawn (a
    parent before its branches), a vessel is cut where its lumen starts to
    run along an earlier in-focus vessel's for longer than a crossing would
    (run_factor * (r1 + r2) + run_px px); its branches off the part cut go
    too.  A branch leaving its parent (the stretch at its start) and a
    vessel out of focus (blur >= blur_max: at another depth) are exempt."""
    out, new_index = [], {}
    for k, v in enumerate(vessels):
        v = dict(v)
        par = v.get("parent")
        if par is not None:
            pk, pi = par
            if pk not in new_index or pi >= len(out[new_index[pk]]["xy"]):
                continue                                    # its parent was cut before it
            v["parent"] = (new_index[pk], pi)
        cut = len(v["xy"])
        if v["blur"] < blur_max:
            for j, u in enumerate(out):
                if u["blur"] >= blur_max:
                    continue
                d, i = cKDTree(u["xy"]).query(v["xy"])
                reach = u["r"][i] + v["r"]
                ov = d < reach
                n = 0
                while n < len(ov):
                    if not ov[n]:
                        n += 1
                        continue
                    b = n
                    while b < len(ov) and ov[b]:
                        b += 1
                    departing = n == 0 and par is not None and v["parent"][0] == j
                    if not departing and b - n > run_factor * float(reach[n:b].max()) + run_px:
                        cut = min(cut, n)
                        break
                    n = b
        if cut < min_len:
            continue
        v["xy"], v["r"] = v["xy"][:cut], v["r"][:cut]
        new_index[k] = len(out)
        out.append(v)
    return out


def render(vessels, shape, rng, ss=2, noise=True, bright=0.8):
    H, W = shape
    Hs, Ws = H * ss, W * ss
    od = np.zeros((Hs, Ws), np.float32)
    for v in vessels:
        xy = v["xy"] * ss + (ss - 1) / 2.0
        r = v["r"] * ss
        R = r.max() + 1
        x0, x1 = int(max(0, xy[:, 0].min() - R - 2)), int(min(Ws, xy[:, 0].max() + R + 3))
        y0, y1 = int(max(0, xy[:, 1].min() - R - 2)), int(min(Hs, xy[:, 1].max() + R + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        # dense resample so the nearest-point distance is accurate
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.linspace(0, s[-1], int(s[-1] / 0.3) + 2)
        dx = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        dr = np.interp(q, s, r)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
        d, j = cKDTree(dx).query(P, distance_upper_bound=R + 1)
        ok = np.isfinite(d)
        layer = np.zeros(len(P), np.float32)
        rr = dr[j[ok]]
        layer[ok] = np.sqrt(np.clip(1 - (d[ok] / rr) ** 2, 0, 1))
        layer = layer.reshape(y1 - y0, x1 - x0)
        full = np.zeros((Hs, Ws), np.float32)
        full[y0:y1, x0:x1] = layer * v["amp"]
        od += ndi.gaussian_filter(full, v["blur"] * ss, truncate=3.5)
    od = cv2.resize(od, (W, H), interpolation=cv2.INTER_AREA)
    # background: lumpy scleral texture + illumination falloff
    yy, xx = np.mgrid[0:H, 0:W]
    illum = bright * np.exp(-(((xx - W * rng.uniform(0.3, 0.7)) / (0.9 * W)) ** 2 +
                              ((yy - H * rng.uniform(0.3, 0.7)) / (0.9 * H)) ** 2))
    tex = ndi.gaussian_filter(rng.standard_normal((H, W)), 45)
    tex = 0.12 * tex / tex.std()
    fine = ndi.gaussian_filter(rng.standard_normal((H, W)), 4)
    fine = 0.006 * fine / fine.std()
    I = illum * np.exp(tex + fine - od)
    if noise:
        e = 4095.0 * I / 1.2
        e = rng.poisson(np.maximum(e, 0) * 1.0) + rng.normal(0, 3.0, e.shape)
        I = np.clip(e / 4095.0 * 1.2, 0, 1)
    return I.astype(np.float32), od


def make_scene(seed=0, shape=(512, 768), **kw):
    rng = np.random.default_rng(seed)
    vessels, bif = make_network(rng, shape, **kw)
    I, od = render(vessels, shape, rng)
    return I, vessels, bif


def centreline_metrics(net, vessels, shape, tol_min=2.0, tol_frac=0.5, spacing=1.0,
                       min_amp_visible=0.0):
    """Centreline recall (per true vessel point) and precision (per detected
    point), with tolerance max(tol_min, tol_frac * true diameter)."""
    H, W = shape
    tx, tr, tb, ta = [], [], [], []
    for v in vessels:
        xy = v["xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], spacing)
        p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        ok = (p[:, 0] >= 2) & (p[:, 0] < W - 2) & (p[:, 1] >= 2) & (p[:, 1] < H - 2)
        tx.append(p[ok])
        tr.append(np.interp(q, s, v["r"])[ok])
        tb.append(np.full(ok.sum(), v["blur"]))
        ta.append(np.full(ok.sum(), v["amp"]))
    tx, tr, tb, ta = map(np.concatenate, (tx, tr, tb, ta))
    dx = [net.sample(e, spacing)["xy"] for e in net.edges]
    dx = np.concatenate(dx) if dx else np.zeros((0, 2))
    tol_t = np.maximum(tol_min, tol_frac * 2 * tr)
    if len(dx):
        d_t, _ = cKDTree(dx).query(tx)
    else:
        d_t = np.full(len(tx), np.inf)
    hit = d_t <= tol_t
    if len(dx):
        d_d, j = cKDTree(tx).query(dx)
        prec_hit = d_d <= tol_t[j]
    else:
        prec_hit = np.zeros(0, bool)
    out = dict(recall=float(hit.mean()), precision=float(prec_hit.mean()) if len(prec_hit) else 0.0,
               true_len=int(len(tx)), detected_len=int(len(dx)))
    bins = [(0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 20)]
    out["recall_by_radius"] = {f"{a}-{b}": float(hit[(tr >= a) & (tr < b)].mean())
                               for a, b in bins if ((tr >= a) & (tr < b)).any()}
    bb = [(0, 1.5), (1.5, 4.0), (4.0, 20)]
    out["recall_by_blur"] = {f"{a}-{b}": float(hit[(tb >= a) & (tb < b)].mean())
                             for a, b in bb if ((tb >= a) & (tb < b)).any()}
    return out


def _true_points(vessels, shape, spacing=1.0, margin=2):
    """Ground-truth centreline points inside the image, 1 px apart, with the
    vessel index, unit tangent and radius of each point."""
    H, W = shape
    pts, vid, tan, rad = [], [], [], []
    for k, v in enumerate(vessels):
        xy = v["xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], spacing)
        p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        ok = (p[:, 0] >= margin) & (p[:, 0] < W - margin) & \
             (p[:, 1] >= margin) & (p[:, 1] < H - margin)
        pts.append(p[ok])
        tan.append(t[ok])
        rad.append(np.interp(q, s, v["r"])[ok])
        vid.append(np.full(ok.sum(), k))
    return (np.concatenate(pts), np.concatenate(vid), np.concatenate(tan),
            np.concatenate(rad))



def vessel_metrics(net, vessels, shape, tol_min=2.0, tol_frac=0.5, spacing=1.0,
                   min_piece=10.0, min_frac=0.2):
    """How completely single splines annotate single vessels.

    Every detected centreline sample is assigned to the nearest true vessel
    point within tolerance (as in centreline_metrics).  An edge *annotates*
    a true vessel when at least max(min_piece, min_frac * its length) px of
    it are assigned to that vessel.  Returns

    * fragments         mean number of edges annotating each detected
                        vessel (1 is ideal), and weighted by vessel length
    * best_cover        fraction of each true vessel's length covered by its
                        single best edge, length-weighted (at most recall)
    * purity            fraction of each edge's assigned length that belongs
                        to its main vessel, length-weighted (1 is ideal:
                        no edge mixes two vessels)
    * mixed_edges       edges that annotate two or more vessels
    * excess            n_edges - n_vessels_seen: splines beyond one per
                        vessel found (0 is ideal: no fragments, nothing
                        spurious)
    * spurious          edges with less than half their length on any true
                        vessel (and their total length, spurious_len)

    A vessel that forks keeps its identity through the fork in the ground
    truth (the parent runs on, the branch is a separate vessel), which is
    exactly what consolidation aims for."""
    H, W = shape
    tx, tv, tr = [], [], []
    for k, v in enumerate(vessels):
        xy = v["xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], spacing)
        p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        ok = (p[:, 0] >= 2) & (p[:, 0] < W - 2) & (p[:, 1] >= 2) & (p[:, 1] < H - 2)
        tx.append(p[ok])
        tv.append(np.full(ok.sum(), k))
        tr.append(np.interp(q, s, v["r"])[ok])
    tx, tv, tr = map(np.concatenate, (tx, tv, tr))
    tol_t = np.maximum(tol_min, tol_frac * 2 * tr)
    nv = len(vessels)
    true_len = np.bincount(tv, minlength=nv).astype(float)
    eids = list(net.edges)
    if not eids:
        return dict(fragments=0.0, fragments_weighted=0.0, best_cover=0.0, purity=0.0,
                    mixed_edges=0, n_edges=0, n_vessels_seen=0, excess=0, spurious=0,
                    spurious_len=0.0)
    tree = cKDTree(tx)
    counts = np.zeros((len(eids), nv))
    cover = np.zeros((len(eids), nv))
    elen = np.zeros(len(eids))
    for i, eid in enumerate(eids):
        xy = net.sample(eid, spacing)["xy"]
        elen[i] = len(xy)
        d, j = tree.query(xy)
        ok = d <= tol_t[j]
        counts[i] = np.bincount(tv[j[ok]], minlength=nv)
        # true points of each vessel within tolerance of this edge
        d2, _ = cKDTree(xy).query(tx, distance_upper_bound=float(tol_t.max()) + 1)
        hit = d2 <= tol_t
        cover[i] = np.bincount(tv[hit], minlength=nv)
    annot = counts >= np.maximum(min_piece, min_frac * elen)[:, None]
    # splines with less than half their length on any vessel annotate nothing
    spurious = counts.sum(1) < 0.5 * elen
    n_frag = annot.sum(0)
    seen = n_frag > 0
    frag = float(n_frag[seen].mean()) if seen.any() else 0.0
    frag_w = float((n_frag[seen] * true_len[seen]).sum() / true_len[seen].sum()) if seen.any() else 0.0
    best = cover.max(0)
    best_cover = float(best.sum() / true_len.sum())
    assigned = counts.sum(1)
    main = counts.max(1)
    purity = float(main.sum() / max(assigned.sum(), 1))
    return dict(fragments=round(frag, 3), fragments_weighted=round(frag_w, 3),
                best_cover=round(best_cover, 4), purity=round(purity, 4),
                mixed_edges=int((annot.sum(1) >= 2).sum()), n_edges=len(eids),
                n_vessels_seen=int(seen.sum()),
                excess=len(eids) - int(seen.sum()), spurious=int(spurious.sum()),
                spurious_len=round(float(elen[spurious].sum() * spacing), 1))


def _true_runs(vessels, shape, margin=2.0, visible=False):
    """Each true vessel's in-image runs: (vessel index, points 1 px apart,
    radius per point).  A vessel leaving the image and coming back is two
    runs: no map can join them.  visible: also keep the stretches outside
    the image whose body still shows in it (within r + 3 blur of the
    border)."""
    H, W = shape
    out = []
    for k, v in enumerate(vessels):
        xy = np.asarray(v["xy"], float)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s_ = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s_[-1], 1.0)
        p = np.stack([np.interp(q, s_, xy[:, 0]), np.interp(q, s_, xy[:, 1])], 1)
        r = np.interp(q, s_, v["r"])
        m = -(r + 3.0 * v.get("blur", 0.0)) if visible else np.full(len(p), margin)
        ok = (p[:, 0] >= m) & (p[:, 0] < W - 1 - m + 1e-9) & \
             (p[:, 1] >= m) & (p[:, 1] < H - 1 - m + 1e-9) if visible else \
            (p[:, 0] >= margin) & (p[:, 0] < W - margin) & \
            (p[:, 1] >= margin) & (p[:, 1] < H - margin)
        idx = np.flatnonzero(ok)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) >= 2:
                out.append((k, p[run], r[run]))
    return out


def truth_network(vessels, shape):
    """The ground truth as a map: one edge per run of every vessel that
    shows in the image (its true centreline, radius, blur and amplitude;
    the centreline may run just outside the image where the vessel's body
    still shows in it), no branch points."""
    from .network import VesselNetwork
    net = VesselNetwork(shape)
    net.meta["optics"] = dict(halo_weight=0.02, halo_sigma=3.0)
    for k, p, r in _true_runs(vessels, shape, visible=True):
        if len(p) < 4:
            continue
        v = vessels[k]
        n = len(p)
        net.add_edge_dense(p, r, np.full(n, v["blur"]), np.full(n, v["amp"]),
                           info=dict(true_vessel=int(k)), faithful=True)
    return net


def resolution_report(net, vessels, shape, tol_min=1.5, tol_frac=0.5, min_frac=0.97,
                      slack=2.0, min_len=10.0, margin=2.0):
    """Whether a map resolves the scene exactly: every true vessel one
    spline, and nothing else.

    The units are the in-image runs of the true vessels (a vessel leaving
    the image and coming back is two).  A run's centreline (1 px apart),
    less the stretch at either end that lies inside another vessel (a
    branch starts on its parent's centreline, where either may be traced),
    is compared with the edge covering most of it, point by point within
    tol = max(tol_min, tol_frac * max(r, blur)) (a blurred vessel's
    centreline is known less precisely):

    * cover: the fraction of the run's points that edge comes within tol of
    * own: the fraction of that edge's length in the image (beyond margin)
      within tol of this vessel, less a stretch at either end lying inside
      another vessel (where one vessel ends in another's lumen, the image
      cannot say where)

    A run is resolved when both are at least min_frac (or miss at most
    slack px) and its edge is the best edge of no other vessel's run.  Edges that are no run's best edge are extra
    (fragments, duplicates, spurious, mixed pieces), unless they trace a
    run too short to be required (under min_len px).  exact: every
    required run resolved and no extra edge."""
    H, W = shape
    eids = list(net.edges)
    esmp = {}
    for k in eids:
        xy = net.sample(k, 1.0)["xy"]
        inn = (xy[:, 0] >= margin) & (xy[:, 0] < W - margin) & \
              (xy[:, 1] >= margin) & (xy[:, 1] < H - margin)
        esmp[k] = (xy, xy[inn])
    etree = {k: cKDTree(esmp[k][0]) for k in eids}
    runs = _true_runs(vessels, shape, margin)
    vpts = {}
    for k, p, r in runs:
        P, R = vpts.get(k, (np.zeros((0, 2)), np.zeros(0)))
        vpts[k] = (np.vstack([P, p]), np.r_[R, r])
    vtree = {k: cKDTree(P) for k, (P, _) in vpts.items()}

    blur = {k: float(vessels[k].get("blur", 0.0)) for k in vpts}
    tol_of = lambda k, r: np.maximum(tol_min, tol_frac * np.maximum(r, blur[k]))

    def inside_others(xy, k):
        """Per sample, the half-width r + blur + 1 of another vessel (than
        k) whose footprint it lies in (0: none)."""
        out = np.zeros(len(xy))
        for j, (Pj, Rj) in vpts.items():
            if j == k:
                continue
            fw = Rj + blur[j] + 1.0
            d, i = vtree[j].query(xy, distance_upper_bound=float(fw.max()) + 1.0)
            hit = np.isfinite(d)
            h = np.flatnonzero(hit)
            inn = d[hit] <= fw[i[hit]]
            out[h[inn]] = np.maximum(out[h[inn]], fw[i[hit]][inn])
        return out

    def end_trim(inside):
        """True where a sample is not in a stretch at either end lying
        inside another vessel, of at most twice that vessel's half-width
        (where one vessel ends in another's lumen, the image cannot say
        where)."""
        keep = np.ones(len(inside), bool)
        for order in (list(range(len(inside))), list(range(len(inside) - 1, -1, -1))):
            if not len(order) or inside[order[0]] <= 0:
                continue
            cap = 2.0 * inside[order[0]]
            for n, i in enumerate(order):
                if inside[i] <= 0 or n >= cap:
                    break
                keep[i] = False
        return keep if keep.any() else np.ones(len(inside), bool)

    def own_frac(e, k):
        xy = esmp[e][1]
        if not len(xy):
            return 0.0
        xy = xy[end_trim(inside_others(xy, k))]
        d, i = vtree[k].query(xy)
        return float((d <= tol_of(k, vpts[k][1][i])).mean())

    def ok_frac(f, n):                              # at least min_frac, or all but slack px
        return f >= min(min_frac, 1.0 - slack / max(n, 1.0))

    rows, best_of, allowed = [], {}, set()
    for u, (k, p, r) in enumerate(runs):
        tol = tol_of(k, r)
        inside = inside_others(p, k)
        if (inside > 0).all():                      # all inside other vessels
            need = np.ones(len(p), bool)
            short = True
        else:
            need = end_trim(inside)
            short = need.sum() < min_len
        best = None
        for e in eids:
            d, _ = etree[e].query(p[need])
            c = float((d <= tol[need]).mean())
            if best is None or c > best[0]:
                best = (c, e)
        cover, e = best if best else (0.0, None)
        own = own_frac(e, k) if e is not None else 0.0
        if short:                                   # not required; may be traced
            for f in eids:
                d, _ = etree[f].query(p[need])
                if ok_frac((d <= tol[need]).mean(), need.sum()) and \
                        ok_frac(own_frac(f, k), len(esmp[f][1])):
                    allowed.add(f)
            continue
        rows.append(dict(vessel=k, run=u, length=int(need.sum()),
                         edge_len=int(len(esmp[e][1])) if e is not None else 0,
                         r=round(float(np.median(r)), 2), blur=round(float(vessels[k]["blur"]), 2),
                         amp=round(float(vessels[k]["amp"]), 3), edge=e,
                         cover=round(cover, 3), own=round(own, 3), _c=cover, _o=own))
        if e is not None:
            best_of.setdefault(e, []).append(u)
    run_vessel = {row["run"]: row["vessel"] for row in rows}
    for row in rows:
        e = row["edge"]
        # an edge may cover several runs of one vessel (one dipping into the margin)
        row["resolved"] = bool(e is not None and ok_frac(row.pop("_c"), row["length"]) and
                               ok_frac(row.pop("_o"), row["edge_len"]) and
                               len({run_vessel[u] for u in best_of[e]}) == 1)
    extra = [e for e in eids if e not in best_of and e not in allowed and len(esmp[e][1])]
    n_res = sum(r["resolved"] for r in rows)
    return dict(n_runs=len(rows), resolved=n_res, extra_edges=len(extra),
                extra_len=round(float(sum(len(esmp[e][1]) for e in extra)), 1),
                exact=bool(n_res == len(rows) and not extra),
                unresolved=[r for r in rows if not r["resolved"]], runs=rows, extra=extra)


def search_report(net, vessels, shape, tol_min=2.0, tol_frac=0.5):
    """What the energy search (search.py) did, against the ground truth,
    from net.meta["search"]:

    * deletions (not revived later): how many, how many of vessels (half
      their length or more on a true vessel), and what decided each of
      those: 'hot' (accepted uphill, by the temperature), the 'price' (it
      explained less than price * L), the existence cost ('lam': less than
      the cost it saved, duplicates' refits included), or the
      fragmentation term it released ('phi': less than that plus the Phi
      released); with the evidence per px of the true ones, next to the
      price
    * fidelity: what the accepted joins cost in NLL + prior, next to the
      signal energy of the ends they joined (their S)
    * stranding joins: accepted joins that released more Phi than their
      own pair's weight (they left other facing ends unmatched)"""
    from scipy.spatial import cKDTree
    meta = net.meta.get("search", {})
    tx, tv, _, tr = _true_points(vessels, shape)
    tree = cKDTree(tx)
    tol = np.maximum(tol_min, tol_frac * 2 * tr)
    price = meta.get("price_per_px", 0.0)
    moves = meta.get("moves", [])
    revived = {r.get("grave") for r in moves if r["kind"] == "revive"}
    dels = []
    for r in meta.get("deleted", []):
        xy = np.asarray(r.get("xy", []), float).reshape(-1, 2)
        if not len(xy) or r.get("grave") in revived:
            continue
        d, j = tree.query(xy)
        on = float((d <= tol[j]).mean())
        L, loss = r["L"], r["loss"]
        saved = -r["cost"]
        driver = "hot" if r["dE"] >= 0 else "price" if loss < price * L else \
            "lam" if loss < saved else "phi" if loss < saved - r["frag"] else "other"
        dels.append(dict(on_truth=round(on, 2), true=on >= 0.5, L=L,
                         per_px=round(loss / max(L, 1.0), 2), driver=driver, dE=r["dE"],
                         frag=r["frag"]))
    true = [x for x in dels if x["true"]]
    joins = [r for r in moves if r["kind"] == "join"]
    return dict(
        deleted=len(dels), deleted_true=len(true),
        deleted_true_by={k: sum(x["driver"] == k for x in true)
                         for k in ("hot", "price", "lam", "phi", "other")},
        deleted_true_per_px=sorted(x["per_px"] for x in true), price=round(price, 2),
        joins=len(joins),
        join_cost=round(sum(r["nll"] + r["prior"] for r in joins), 1),
        joined_ends_S=round(sum(sum(r.get("ends_S", [])) for r in joins), 1),
        stranding_joins=sum(-r["frag"] > r.get("pair_weight", 0.0) + 1.0 for r in joins
                            if "pair_weight" in r),
        accepted={k: sum(r["kind"] == k for r in moves)
                  for k in ("join", "delete", "split", "reroute", "revive")})


def fragment_network(vessels, shape, rng, piece_len=(25.0, 80.0), gap=(0.0, 6.0),
                     jitter=0.3, dup_frac=0.15, n_spurious=6, min_len=12.0):
    """A deliberately fragmented network built from the ground truth, as a
    fast stand-in for a noisy map: every true vessel is cut into pieces of
    random length separated by small gaps, some pieces get a slightly
    offset duplicate (as found again in another scale band), and a few short
    faint edges are scattered over the background.  Profiles are the true
    ones (r, blur, peak density)."""
    from scipy.ndimage import gaussian_filter1d
    from .network import VesselNetwork
    H, W = shape
    net = VesselNetwork(shape)
    net.meta["optics"] = dict(halo_weight=0.02, halo_sigma=3.0)
    for v in vessels:
        xy = np.asarray(v["xy"], float)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        q = np.arange(0, s[-1], 0.7)
        p = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], 1)
        r = np.interp(q, s, v["r"])
        inside = (p[:, 0] >= 0) & (p[:, 0] <= W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= H - 1)
        if inside.sum() < 2:
            continue
        idx = np.flatnonzero(inside)
        p, r, q = p[idx[0]:idx[-1] + 1], r[idx[0]:idx[-1] + 1], q[idx[0]:idx[-1] + 1]
        q = q - q[0]
        off = gaussian_filter1d(rng.standard_normal((len(p), 2)), 8.0, axis=0) * jitter * 4
        p = p + off
        a0 = 0.0
        while a0 < q[-1]:
            a1 = min(q[-1], a0 + rng.uniform(*piece_len))
            if q[-1] - a1 < min_len:
                a1 = q[-1]
            m = (q >= a0) & (q <= a1)
            if m.sum() >= 3 and a1 - a0 >= min_len:
                n = int(m.sum())
                amp = np.full(n, v["amp"])
                net.add_edge_dense(p[m], r[m], np.full(n, v["blur"]), amp,
                                   info=dict(true_id=int(id(v) % 100000)))
                if rng.random() < dup_frac:
                    shift = rng.normal(0, 0.8, 2)
                    net.add_edge_dense(p[m] + shift, r[m] * 1.2, np.full(n, v["blur"] * 1.1),
                                       amp * 0.5)
            a0 = a1 + rng.uniform(*gap)
    for _ in range(n_spurious):
        c = rng.uniform([20, 20], [W - 20, H - 20])
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(15, 40)
        t = np.linspace(-L / 2, L / 2, int(L))
        p = c + np.stack([t * math.cos(ang), t * math.sin(ang)], 1)
        n = len(p)
        net.add_edge_dense(p, np.full(n, 1.0), np.full(n, 1.0), np.full(n, 0.02))
    return net
