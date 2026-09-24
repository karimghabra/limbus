"""Differentiable rendering of a whole vessel network and its score.

Forward model (log domain, see image.py):

    logI_hat(x) = B(x) - sum_e a_e(t) * P(d_e(x); r_e(t), s_e(t)) * T_e(x)

* B       smooth background: bicubic interpolation of a coarse grid.
* d_e(x)  distance of pixel x from the centreline spline of edge e, t the
          parameter of the closest centreline point.
* P       cross-section of a cylinder of radius r (chord length, i.e. the
          optical path through blood, normalised to 1 at the centre)
          convolved with a Gaussian of std s (defocus/optics).  The cylinder
          is represented as a stack of K nested boxes, each of which blurs to
          a difference of erf's, so P is exact to the staircase and smooth.
* T_e     end caps: the vessel fades along its axis with the same blur, so
          an edge ending at a free end, or two edges meeting at a node,
          produce no seams (two collinear caps sum exactly to one).
* halo    light scattered in the overlying tissue gives every vessel a
          shallow skirt that one Gaussian cannot describe.  The optical
          blur is therefore (1 - h) G(s) + h G(sqrt(s^2 + s_h^2)), with the
          halo weight h and width s_h shared by the whole image (fitted).

Overlapping vessels add in optical density.  The score of a network is the
weighted sum of squared residuals over all valid pixels plus smoothness
priors, i.e. a negative log-likelihood; see fit.py for how edges are added
and removed by the change in this score.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from . import spline as sp
from .network import A_MIN, R_MIN, S_MIN, VesselNetwork

# nested boxes approximating a semicircle chord profile sqrt(1 - rho^2)
_K = 6
_C = np.sin(0.5 * np.pi * np.arange(1, _K + 1) / _K)             # box half-widths / r
_MID = 0.5 * (np.r_[0.0, _C[:-1]] + _C)
_H = np.sqrt(1.0 - _MID ** 2)
_W = _H - np.r_[_H[1:], 0.0]                                     # box heights
PROFILE_C = torch.tensor(_C, dtype=torch.float32)
PROFILE_W = torch.tensor(_W, dtype=torch.float32)
SQRT2 = math.sqrt(2.0)


def profile(d, r, s):
    """Blurred cylinder profile; broadcasting over the last axis of boxes."""
    c = PROFILE_C.to(d.dtype)
    w = PROFILE_W.to(d.dtype)
    rr = r[..., None] * c
    ss = (SQRT2 * s)[..., None]
    dd = d[..., None]
    return (0.5 * (torch.erf((rr - dd) / ss) + torch.erf((rr + dd) / ss)) * w).sum(-1)


def _entry_core(xy, C, T, R, S, A, arc, Le, hw, hs):
    """Optical density contributed by one (pixel, edge) pair; all inputs are
    per pair except the two global halo parameters."""
    dv = xy - C
    u = (dv * T).sum(1)
    d = (dv[:, 0] * T[:, 1] - dv[:, 1] * T[:, 0]).abs()
    along = arc + u
    ss = SQRT2 * S
    taper = 0.25 * torch.erfc(-along / ss) * torch.erfc((along - Le) / ss)
    Sh = torch.sqrt(S * S + hs * hs)
    prof = (1.0 - hw) * profile(d, R, S) + hw * profile(d, R, Sh)
    return A * prof * taper


_CORE = None


def entry_core(*args):
    """_entry_core, fused by torch.compile when a C++ compiler is available
    (about 4x faster on CPU); falls back to eager mode otherwise.  Set
    VESSELMAP_COMPILE=0 to disable."""
    global _CORE
    if _CORE is None:
        import os
        _CORE = _entry_core
        if os.environ.get("VESSELMAP_COMPILE", "1") != "0":
            try:
                cc = torch.compile(_entry_core, dynamic=True)
                t = [torch.zeros(4, 2), torch.zeros(4, 2), torch.ones(4, 2) / SQRT2] + \
                    [torch.ones(4) for _ in range(5)] + [torch.tensor(0.2), torch.tensor(3.0)]
                t[3].requires_grad_(True)
                cc(*t).sum().backward()
                _CORE = cc
            except Exception:                       # no compiler, old torch, ...
                _CORE = _entry_core
    return _CORE(*args)


def profile_peak(r, s):
    r = torch.as_tensor(r, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    return profile(torch.zeros_like(r), r, s)


def inv_softplus(y):
    y = np.maximum(np.asarray(y, float), 1e-6)
    return np.where(y > 20, y, np.log(np.expm1(y)))


def _block_sparse(blocks, row_off, col_off, n_rows, n_cols):
    rows, cols, vals = [], [], []
    for M, ro, co in zip(blocks, row_off, col_off):
        nz = np.nonzero(M)
        rows.append(nz[0] + ro)
        cols.append(nz[1] + co)
        vals.append(M[nz])
    if rows:
        idx = np.stack([np.concatenate(rows), np.concatenate(cols)])
        v = np.concatenate(vals)
    else:
        idx = np.zeros((2, 0), int)
        v = np.zeros(0)
    return torch.sparse_coo_tensor(torch.as_tensor(idx, dtype=torch.long),
                                   torch.as_tensor(v, dtype=torch.float32),
                                   (n_rows, n_cols)).coalesce()


class NetworkModel(torch.nn.Module):
    """Holds the parameters of a network + background for one image."""

    def __init__(self, net: VesselNetwork, logI: np.ndarray, weight: np.ndarray,
                 stride: int = 1, sample_spacing: float = 0.75,
                 bg_spacing: float = 64.0, ref: VesselNetwork | None = None,
                 fit_background: bool = True):
        super().__init__()
        self.net = net
        self.H, self.W = logI.shape
        self.stride = int(stride)
        self.sample_spacing = sample_spacing
        self.logI = torch.as_tensor(logI, dtype=torch.float32)
        self.weight = torch.as_tensor(weight, dtype=torch.float32)
        self.eids = list(net.edges)
        self.nids = list(net.nodes)
        nidx = {n: i for i, n in enumerate(self.nids)}
        self.node_xy = torch.nn.Parameter(torch.tensor(
            np.array([net.nodes[n].xy for n in self.nids]).reshape(-1, 2), dtype=torch.float32))
        inner, gather, rprof, sprof, aprof, pidx = [], [], [], [], [], []
        rmax_l, smax_l = [], []
        n_nodes = len(self.nids)
        off = 0
        poff = 0
        self.edge_nctrl, self.edge_nprof, self.edge_ctrl_off, self.edge_prof_off = [], [], [], []
        c_off = 0
        for eid in self.eids:
            net.sync_ends(eid)
            e = net.edges[eid]
            n = len(e.ctrl)
            inner.append(e.ctrl[1:-1])
            g = [nidx[e.u]] + list(range(n_nodes + off, n_nodes + off + n - 2)) + [nidx[e.v]]
            gather += g
            off += n - 2
            self.edge_nctrl.append(n)
            self.edge_ctrl_off.append(c_off)
            c_off += n
            m = len(e.r)
            rprof.append(inv_softplus(np.maximum(e.r, R_MIN + 1e-3) - R_MIN))
            sprof.append(inv_softplus(np.maximum(e.s, S_MIN + 1e-3) - S_MIN))
            aprof.append(inv_softplus(np.maximum(e.a, A_MIN)))
            self.edge_nprof.append(m)
            self.edge_prof_off.append(poff)
            poff += m
            wmax = edge_wmax(e.info)
            rmax_l.append(np.full(m, inv_softplus(2.0 * wmax - R_MIN)))
            smax_l.append(np.full(m, inv_softplus(wmax - S_MIN)))
        cat = (lambda L, shape: np.concatenate(L, 0) if L else np.zeros(shape))
        self.inner = torch.nn.Parameter(torch.tensor(cat(inner, (0, 2)), dtype=torch.float32))
        self.gather = torch.tensor(gather, dtype=torch.long)
        self.raw_r = torch.nn.Parameter(torch.tensor(cat(rprof, (0,)), dtype=torch.float32))
        self.raw_s = torch.nn.Parameter(torch.tensor(cat(sprof, (0,)), dtype=torch.float32))
        self.raw_a = torch.nn.Parameter(torch.tensor(cat(aprof, (0,)), dtype=torch.float32))
        self.raw_r_max = torch.tensor(cat(rmax_l, (0,)), dtype=torch.float32)
        optics = net.meta.get("optics", {})
        hw0 = float(np.clip(optics.get("halo_weight", 0.25), 0.01, 0.79))
        hs0 = float(max(optics.get("halo_sigma", 5.0), 1.05))
        self.raw_hw = torch.nn.Parameter(torch.tensor(math.log(hw0 / (0.8 - hw0)), dtype=torch.float32))
        self.raw_hs = torch.nn.Parameter(torch.tensor(float(inv_softplus(hs0 - 1.0)), dtype=torch.float32))
        self.raw_s_max = torch.tensor(cat(smax_l, (0,)), dtype=torch.float32)
        self.project()
        # background grid
        if net.background is not None and net.bg_spacing:
            grid = np.asarray(net.background, np.float32)
            self.bg_spacing = net.bg_spacing
            gh, gw = grid.shape
            if (gh, gw) != self._grid_shape(self.bg_spacing):
                grid = self._fit_grid(self._upsample_np(grid), bg_spacing)
                self.bg_spacing = bg_spacing
        else:
            self.bg_spacing = bg_spacing
            grid = self._fit_grid(robust_background(logI, weight), bg_spacing)
        self.bg = torch.nn.Parameter(torch.tensor(grid, dtype=torch.float32),
                                     requires_grad=fit_background)
        # optional reference for per-frame tracking priors
        self.ref = None
        if ref is not None:
            self.ref = {k: v.detach().clone() for k, v in (
                ("node_xy", self.node_xy), ("inner", self.inner), ("raw_r", self.raw_r),
                ("raw_s", self.raw_s), ("raw_a", self.raw_a))}
            if ref is not net:
                self.ref.update(self._pack_reference(ref))
        # global affine (identity unless optimised; used to pre-align a frame)
        self.aff_A = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.aff_t = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self._pix_mask = self._stride_mask()
        self._index_edges()
        self.rebuild()

    def _pack_reference(self, ref: VesselNetwork) -> dict:
        """Parameters of another network with the same ids and control-point
        counts, packed like this model's (for the tracking prior)."""
        node_xy = np.array([ref.nodes[n].xy for n in self.nids]).reshape(-1, 2)
        inner, rr, ss, aa = [], [], [], []
        for k, eid in enumerate(self.eids):
            e = ref.edges[eid]
            if len(e.ctrl) != self.edge_nctrl[k] or len(e.r) != self.edge_nprof[k]:
                raise ValueError(f"edge {eid} differs in structure from the reference")
            inner.append(e.ctrl[1:-1])
            rr.append(inv_softplus(np.maximum(e.r, R_MIN + 1e-3) - R_MIN))
            ss.append(inv_softplus(np.maximum(e.s, S_MIN + 1e-3) - S_MIN))
            aa.append(inv_softplus(np.maximum(e.a, A_MIN)))
        t = lambda L, shape: torch.tensor(np.concatenate(L, 0) if L else np.zeros(shape),
                                          dtype=torch.float32)
        return dict(node_xy=torch.tensor(node_xy, dtype=torch.float32), inner=t(inner, (0, 2)),
                    raw_r=t(rr, (0,)), raw_s=t(ss, (0,)), raw_a=t(aa, (0,)))

    # ------------------------------------------------------------ background
    def _grid_shape(self, spacing):
        return (int(math.ceil((self.H - 1) / spacing)) + 1,
                int(math.ceil((self.W - 1) / spacing)) + 1)

    def _upsample_np(self, grid):
        g = torch.tensor(grid, dtype=torch.float32)[None, None]
        return F.interpolate(g, size=(self.H, self.W), mode="bicubic",
                             align_corners=True)[0, 0].numpy()

    def _fit_grid(self, img, spacing):
        """Grid whose bicubic upsampling approximates img (a few GD steps)."""
        gh, gw = self._grid_shape(spacing)
        import cv2
        g0 = cv2.resize(img.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        g = torch.tensor(g0, requires_grad=True)
        target = torch.tensor(img, dtype=torch.float32)
        opt = torch.optim.Adam([g], lr=0.01)
        for _ in range(60):
            opt.zero_grad()
            up = F.interpolate(g[None, None], size=(self.H, self.W), mode="bicubic",
                               align_corners=True)[0, 0]
            loss = ((up - target) ** 2).mean()
            loss.backward()
            opt.step()
        return g.detach().numpy()

    def background(self):
        return F.interpolate(self.bg[None, None], size=(self.H, self.W), mode="bicubic",
                             align_corners=True)[0, 0]

    def _stride_mask(self):
        m = torch.zeros(self.H, self.W, dtype=torch.bool)
        m[::self.stride, ::self.stride] = True
        return m & (self.weight > 0)

    @torch.no_grad()
    def project(self):
        """Hard limits on calibre and blur (see edge_wmax)."""
        self.raw_r.data = torch.minimum(self.raw_r.data, self.raw_r_max)
        self.raw_s.data = torch.minimum(self.raw_s.data, self.raw_s_max)

    # ------------------------------------------------------------ geometry
    def ctrl_all(self):
        pool = torch.cat([self.node_xy, self.inner], 0)
        if self.aff_A.requires_grad or self.aff_t.requires_grad:
            c = torch.tensor([self.W / 2.0, self.H / 2.0])
            pool = (pool - c) @ (torch.eye(2) + self.aff_A).T + c + self.aff_t
        return pool[self.gather]

    def halo(self):
        return 0.8 * torch.sigmoid(self.raw_hw), 1.0 + F.softplus(self.raw_hs)

    def profiles(self):
        r = R_MIN + F.softplus(self.raw_r)
        s = S_MIN + F.softplus(self.raw_s)
        a = F.softplus(self.raw_a)
        return r, s, a

    def rebuild(self):
        """(Re)create the dense sampling matrices and the pixel association.
        Called at start and periodically during optimisation."""
        with torch.no_grad():
            ctrl = self.ctrl_all().numpy().astype(float)
            r_c, s_c, a_c = (t.numpy().astype(float) for t in self.profiles())
        Bs, Ds, Ps, roff, coff, poff = [], [], [], [], [], []
        n_s = 0
        self.samp_edge, self.samp_first, self.samp_last = [], [], []
        for k, eid in enumerate(self.eids):
            n = self.edge_nctrl[k]
            c = ctrl[self.edge_ctrl_off[k]:self.edge_ctrl_off[k] + n]
            xy = sp.design(n, 64) @ c
            L = sp.arclength(xy)[-1]
            m = sp.n_samples_for_length(L, self.sample_spacing)
            Bs.append(sp.design(n, m))
            Ds.append(sp.design(n, m, 1))
            Ps.append(sp.design(self.edge_nprof[k], m))
            roff.append(n_s)
            coff.append(self.edge_ctrl_off[k])
            poff.append(self.edge_prof_off[k])
            self.samp_edge.append(np.full(m, k))
            self.samp_first.append(n_s)
            n_s += m
            self.samp_last.append(n_s - 1)
        n_c = int(sum(self.edge_nctrl))
        n_p = int(sum(self.edge_nprof))
        self.B = _block_sparse(Bs, roff, coff, n_s, n_c)
        self.D = _block_sparse(Ds, roff, coff, n_s, n_c)
        self.Bp = _block_sparse(Ps, roff, poff, n_s, n_p)
        self.n_samp = n_s
        self.samp_edge_t = torch.tensor(np.concatenate(self.samp_edge) if self.samp_edge
                                        else np.zeros(0, int), dtype=torch.long)
        self.samp_first_t = torch.tensor(self.samp_first, dtype=torch.long)
        self.samp_last_t = torch.tensor(self.samp_last, dtype=torch.long)
        self._associate()

    def samples(self):
        ctrl = self.ctrl_all()
        C = torch.sparse.mm(self.B, ctrl)
        Dv = torch.sparse.mm(self.D, ctrl)
        T = Dv / (Dv.norm(dim=1, keepdim=True) + 1e-6)
        r, s, a = self.profiles()
        R = torch.sparse.mm(self.Bp, r[:, None])[:, 0]
        S = torch.sparse.mm(self.Bp, s[:, None])[:, 0]
        A = torch.sparse.mm(self.Bp, a[:, None])[:, 0]
        # arclength per edge
        seg = (C[1:] - C[:-1]).norm(dim=1)
        same = self.samp_edge_t[1:] == self.samp_edge_t[:-1]
        seg = torch.where(same, seg, torch.zeros_like(seg))
        cum = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        arc = cum - cum[self.samp_first_t][self.samp_edge_t]
        Ledge = arc[self.samp_last_t]
        return C, T, R, S, A, arc, Ledge

    def _associate(self):
        """For each edge, the pixels within its reach and their nearest
        centreline sample (no gradient; refreshed by rebuild())."""
        import cv2
        with torch.no_grad():
            C, T, R, S, A, arc, Ledge = self.samples()
        C = C.numpy()
        hw, hs = (float(v) for v in self.halo())
        # the halo tail beyond 2 of its sigmas is far below the noise
        Sh = torch.sqrt(S * S + hs * hs) if hw > 0.03 else S
        reach = (R + torch.maximum(3.0 * S, 2.0 * Sh)).numpy() + 1.5
        mask = self._pix_mask.numpy()
        pix_l, samp_l, edge_l = [], [], []
        for k in range(len(self.eids)):
            a0, a1 = self.samp_first[k], self.samp_last[k] + 1
            xy = C[a0:a1]
            Rk = float(reach[a0:a1].max())
            x0 = int(max(0, math.floor(xy[:, 0].min() - Rk - 1)))
            x1 = int(min(self.W, math.ceil(xy[:, 0].max() + Rk + 2)))
            y0 = int(max(0, math.floor(xy[:, 1].min() - Rk - 1)))
            y1 = int(min(self.H, math.ceil(xy[:, 1].max() + Rk + 2)))
            if x1 <= x0 or y1 <= y0:
                continue
            canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
            pts = np.round((xy - [x0, y0]) * 4).astype(np.int32)
            cv2.polylines(canvas, [pts], False, 1, thickness=int(2 * math.ceil(Rk) + 1),
                          lineType=cv2.LINE_8, shift=2)
            canvas &= mask[y0:y1, x0:x1]
            ys, xs = np.nonzero(canvas)
            if len(ys) == 0:
                continue
            P = np.stack([xs + x0, ys + y0], 1).astype(float)
            d, j = cKDTree(xy).query(P, k=1)
            ok = d <= Rk
            pix_l.append((ys[ok] + y0) * self.W + xs[ok] + x0)
            samp_l.append(j[ok] + a0)
            edge_l.append(np.full(int(ok.sum()), k))
        cat = lambda L: np.concatenate(L) if L else np.zeros(0, int)
        self.e_pix = torch.tensor(cat(pix_l), dtype=torch.long)
        self.e_samp = torch.tensor(cat(samp_l), dtype=torch.long)
        self.e_edge = torch.tensor(cat(edge_l), dtype=torch.long)
        py, px = np.divmod(self.e_pix.numpy(), self.W)
        self.e_xy = torch.tensor(np.stack([px, py], 1), dtype=torch.float32)

    # ------------------------------------------------------------ rendering
    def vessel_entries(self):
        """Per (pixel, edge) contributions m >= 0 to the optical density."""
        C, T, R, S, A, arc, Ledge = self.samples()
        j = self.e_samp
        hw, hs = self.halo()
        if len(j) == 0:
            return torch.zeros(0)
        return entry_core(self.e_xy, C[j], T[j], R[j], S[j], A[j], arc[j],
                          Ledge[self.samp_edge_t[j]], hw, hs)

    def vessel_image(self, entries=None):
        m = self.vessel_entries() if entries is None else entries
        V = torch.zeros(self.H * self.W, dtype=torch.float32)
        V = V.index_add(0, self.e_pix, m)
        return V.view(self.H, self.W)

    def predict(self, entries=None):
        return self.background() - self.vessel_image(entries)

    # ------------------------------------------------------------ scoring
    def data_nll(self, pred):
        res = self.logI - pred
        m = self._pix_mask
        return 0.5 * (self.weight[m] * res[m] ** 2).sum() * self.stride ** 2

    def prior(self, lam_bend=300.0, lam_prof=20.0, lam_bg=2e4, track=None):
        ctrl = self.ctrl_all()
        pen = torch.zeros(())
        # bending energy ~ sum |d2 P|^2 / h^3, per edge
        if len(self.eids):
            i = torch.arange(len(ctrl) - 2)
            same = self._ctrl_edge
            ok = (same[i] == same[i + 2])
            d2 = ctrl[i] - 2 * ctrl[i + 1] + ctrl[i + 2]
            h = self._ctrl_h[same[i]]
            pen = pen + lam_bend * ((d2 ** 2).sum(1)[ok] / h[ok] ** 3).sum()
            r, s, a = self.profiles()
            for q in (r, s, a):
                lq = torch.log(q)
                jj = torch.arange(len(lq) - 1)
                same_p = self._prof_edge[jj] == self._prof_edge[jj + 1]
                pen = pen + lam_prof * ((lq[jj + 1] - lq[jj]) ** 2)[same_p].sum()
        # background smoothness (second differences of the grid)
        g = self.bg
        pen = pen + lam_bg * (((g[2:] - 2 * g[1:-1] + g[:-2]) ** 2).sum()
                              + ((g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) ** 2).sum())
        if track is not None and self.ref is not None:
            pen = pen + track_penalty(self, **track)
        return pen

    def _index_edges(self):
        h = [self.net.edges[eid].info.get("spacing", 12.0) for eid in self.eids]
        self._ctrl_h = torch.tensor(h if h else [1.0], dtype=torch.float32)
        ce = [np.full(n, k) for k, n in enumerate(self.edge_nctrl)]
        pe = [np.full(m, k) for k, m in enumerate(self.edge_nprof)]
        self._ctrl_edge = torch.tensor(np.concatenate(ce) if ce else np.zeros(1, int), dtype=torch.long)
        self._prof_edge = torch.tensor(np.concatenate(pe) if pe else np.zeros(1, int), dtype=torch.long)

    def loss(self, track=None, priors=None):
        entries = self.vessel_entries()
        pred = self.predict(entries)
        nll = self.data_nll(pred)
        pen = self.prior(track=track, **(priors or {}))
        return nll + pen, nll, entries, pred

    @torch.no_grad()
    def edge_gains(self, entries=None, pred=None):
        """Decrease of the data NLL that each edge is responsible for, with all
        other edges held fixed: 0.5 * sum w [(res - m)^2 - res^2]."""
        if entries is None:
            entries = self.vessel_entries()
        if pred is None:
            pred = self.predict(entries)
        res = (self.logI - pred).view(-1)[self.e_pix]
        w = self.weight.view(-1)[self.e_pix]
        g = 0.5 * w * (entries ** 2 - 2 * res * entries) * self.stride ** 2
        out = torch.zeros(len(self.eids)).index_add(0, self.e_edge, g)
        cnt = torch.zeros(len(self.eids)).index_add(0, self.e_edge, torch.ones_like(g))
        return out.numpy(), cnt.numpy() * self.stride ** 2

    # ------------------------------------------------------------ write back
    @torch.no_grad()
    def write_back(self):
        net = self.net
        ctrl = self.ctrl_all().numpy().astype(float)
        if self.aff_A.requires_grad or self.aff_t.requires_grad:
            pool = torch.cat([self.node_xy, self.inner], 0)
            c = torch.tensor([self.W / 2.0, self.H / 2.0])
            pool = (pool - c) @ (torch.eye(2) + self.aff_A).T + c + self.aff_t
            n_nodes = len(self.nids)
            self.node_xy.data = pool[:n_nodes].clone()
            self.inner.data = pool[n_nodes:].clone()
            self.aff_A.data.zero_()
            self.aff_t.data.zero_()
            self.aff_A.requires_grad_(False)
            self.aff_t.requires_grad_(False)
        r, s, a = (t.numpy().astype(float) for t in self.profiles())
        for i, n in enumerate(self.nids):
            net.nodes[n].x, net.nodes[n].y = (float(v) for v in self.node_xy[i])
        for k, eid in enumerate(self.eids):
            e = net.edges[eid]
            o, n = self.edge_ctrl_off[k], self.edge_nctrl[k]
            e.ctrl = ctrl[o:o + n].copy()
            po, m = self.edge_prof_off[k], self.edge_nprof[k]
            e.r, e.s, e.a = r[po:po + m].copy(), s[po:po + m].copy(), a[po:po + m].copy()
        net.background = self.bg.detach().numpy().copy()
        net.bg_spacing = self.bg_spacing
        hw, hs = self.halo()
        net.meta["optics"] = dict(halo_weight=float(hw), halo_sigma=float(hs))


W_GLOBAL_MAX = 20.0


def edge_wmax(info) -> float:
    """Largest Gaussian-equivalent half-width sqrt(s^2 + r^2/4) an edge may
    take.  An edge proposed from ridges at scales <= b may grow to twice the
    width a Gaussian line detected at b would have (b / sqrt 2) but not
    more: otherwise a thin proposal can inflate into a blob that models
    background texture instead of a vessel."""
    band = info.get("band")
    if not band:
        return W_GLOBAL_MAX
    return float(min(W_GLOBAL_MAX, math.sqrt(2.0) * float(band[1])))


def track_penalty(model: NetworkModel, sigma_pos=3.0, sigma_logprof=0.35, sigma_amp=0.6):
    """Gaussian prior tying a per-frame fit to the reference network."""
    ref = model.ref
    pen = ((model.node_xy - ref["node_xy"]) ** 2).sum() / (2 * sigma_pos ** 2)
    pen = pen + ((model.inner - ref["inner"]) ** 2).sum() / (2 * sigma_pos ** 2)
    r, s, a = model.profiles()
    r0 = R_MIN + F.softplus(ref["raw_r"])
    s0 = S_MIN + F.softplus(ref["raw_s"])
    a0 = F.softplus(ref["raw_a"])
    pen = pen + ((torch.log(r / r0)) ** 2).sum() / (2 * sigma_logprof ** 2)
    pen = pen + ((torch.log(s / s0)) ** 2).sum() / (2 * sigma_logprof ** 2)
    pen = pen + ((torch.log(a / a0)) ** 2).sum() / (2 * sigma_amp ** 2)
    return pen


def robust_background(logI, weight, max_vessel_width=70.0, smooth=12.0):
    """Initial background: an upper envelope that ignores dark structures
    narrower than max_vessel_width (px).

    A grey-level closing (dilation then erosion) with a disc wider than the
    widest vessel fills every vessel in, independent of how close it runs to
    the image border (reflective padding); the result is then smoothed.
    Computed at half resolution for speed.
    """
    import cv2
    L = logI.astype(np.float32)
    H, W = L.shape
    valid = weight > 0
    if (~valid).any():
        # fill invalid pixels (specular spots, black warp borders) from their
        # surroundings so they neither raise nor lower the envelope
        L = L.copy()
        L[~valid] = float(np.median(L[valid])) if valid.any() else 0.0
    small = 2
    Ls = cv2.resize(L, (W // small, H // small), interpolation=cv2.INTER_AREA)
    Ls = cv2.GaussianBlur(Ls, (0, 0), 1.5)            # noise would bias the max
    d = int(round(max_vessel_width / small)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    pad = d
    Lp = cv2.copyMakeBorder(Ls, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    closed = cv2.morphologyEx(Lp, cv2.MORPH_CLOSE, k)[pad:-pad, pad:-pad]
    closed = cv2.GaussianBlur(closed, (0, 0), smooth / small)
    return cv2.resize(closed, (W, H), interpolation=cv2.INTER_CUBIC)
