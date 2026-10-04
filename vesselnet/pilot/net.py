"""Junction network (pilot): data, model, losses, detection, scoring.

CPU pilot of vesselnet/PLAN.md (iteration 0 starts from it).  Input: one
image's log intensity, flattened (less a wide Gaussian of itself) and scaled
by its robust spread, so a vesselscene image, a zoo sheet and a real frame
look alike to the network.  Outputs, per pixel:
  0 junction heat (any observable junction; target a Gaussian of sigma 3 px
    at its centre),
  1-3 junction type (crossing / branch: bifurcation, confluence, pseudo-T /
    compound; trained near junction centres only),
  4 vessel lumen, 5 vessel centreline (observable truth; don't-care masked).
"""
import math, pickle, glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

TYPES = {"crossing": 0, "bifurcation": 1, "confluence": 1, "pseudo-T": 1, "pseudo_T": 1, "compound": 2}
LAMBDA = 11.9
SIGMA = 3.0
GAIN = tuple(float(g) for g in os.environ.get("JNET_GAIN", "0.7,1.4").split(","))   # contrast augmentation


def preprocess(img, valid=None, flat_sigma=24.0):
    """log intensity, less its wide Gaussian, over its robust spread; invalid -> 0."""
    I = np.asarray(img, np.float64)
    v = np.isfinite(I) & (I > 0) if valid is None else (valid & np.isfinite(I) & (I > 0))
    L = np.zeros(I.shape)
    L[v] = np.log(I[v])
    m = L[v].mean() if v.any() else 0.0
    L[~v] = m
    w = ndi.gaussian_filter(v.astype(float), flat_sigma) + 1e-6
    B = ndi.gaussian_filter(np.where(v, L, 0.0), flat_sigma) / w
    X = L - B
    s = 1.4826 * np.median(np.abs(X[v] - np.median(X[v]))) + 1e-9
    X = np.clip(X / s, -8, 8)
    X[~v] = 0.0
    return X.astype(np.float32), v


def load_scene(path):
    z = np.load(path)
    meta = pickle.loads(z["meta"].tobytes())
    X, v = preprocess(z["img"])
    js = [j for j in meta["obs_junctions"] if j["type_observable"] in TYPES]
    H, W = X.shape
    heat = np.zeros((H, W), np.float32)
    typ = np.full((H, W), -1, np.int64)
    yy, xx = np.mgrid[0:H, 0:W]
    for j in js:
        x0, y0 = j["x"], j["y"]
        r = int(3 * SIGMA + 1)
        ys, xs = slice(max(int(y0) - r, 0), min(int(y0) + r + 2, H)), slice(max(int(x0) - r, 0), min(int(x0) + r + 2, W))
        g = np.exp(-((xx[ys, xs] - x0) ** 2 + (yy[ys, xs] - y0) ** 2) / (2 * SIGMA ** 2))
        heat[ys, xs] = np.maximum(heat[ys, xs], g)
        iy, ix = int(round(y0)), int(round(x0))
        if 0 <= iy < H and 0 <= ix < W:
            heat[iy, ix] = 1.0                        # the peak pixel
        near = (xx[ys, xs] - x0) ** 2 + (yy[ys, xs] - y0) ** 2 <= max(4.0, 0.5 * j["radius"]) ** 2
        typ[ys, xs][near] = TYPES[j["type_observable"]]
    jcare = ~z["jdont_care"]
    for j in meta["dont_care_junctions"]:          # no loss near junctions the truth leaves open
        jcare &= (xx - j["x"]) ** 2 + (yy - j["y"]) ** 2 > max(6.0, j["radius"]) ** 2
    vcare = ~z["dont_care"]
    return dict(X=X, valid=v, heat=heat, typ=typ, jcare=jcare & v, lumen=z["lumen"], centre=z["centreline"],
                vcare=vcare & v, junctions=js, dc_junctions=meta["dont_care_junctions"], meta=meta)


class Crops(torch.utils.data.Dataset):
    def __init__(self, scenes, size=256, n=4000, seed=0):
        self.S, self.size, self.n = scenes, size, n
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
        S = self.S[rng.integers(len(self.S))]
        H, W = S["X"].shape
        c = self.size
        y0, x0 = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        a = [S["X"][sl], S["heat"][sl], S["typ"][sl], S["jcare"][sl], S["lumen"][sl], S["centre"][sl], S["vcare"][sl]]
        k = rng.integers(4)
        flip = rng.integers(2)
        a = [np.rot90(x, k) for x in a]
        if flip:
            a = [x[:, ::-1] for x in a]
        lo, hi = GAIN
        X = a[0] * math.exp(rng.uniform(math.log(lo), math.log(hi))) + rng.normal(0, rng.uniform(0, 0.3), a[0].shape)
        t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(x).astype(dt))
        return (t(X, np.float32)[None], t(a[1], np.float32)[None], t(a[2], np.int64), t(a[3], np.float32)[None],
                t(a[4], np.float32)[None], t(a[5], np.float32)[None], t(a[6], np.float32)[None])


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128, 192), out=6):
        super().__init__()
        self.down = nn.ModuleList()
        i = 1
        for c in ch:
            self.down.append(block(i, c))
            i = c
        self.up = nn.ModuleList()
        self.upc = nn.ModuleList()
        for c_hi, c_lo in zip(ch[::-1][:-1], ch[::-1][1:]):
            self.up.append(nn.ConvTranspose2d(c_hi, c_lo, 2, stride=2))
            self.upc.append(block(2 * c_lo, c_lo))
        self.head = nn.Conv2d(ch[0], out, 1)
        with torch.no_grad():
            self.head.bias[0] = -4.0           # junctions are rare

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** (len(self.down) - 1)
        ph, pw = (-H) % m, (-W) % m
        x = F.pad(x, (0, pw, 0, ph))
        skips = []
        for k, d in enumerate(self.down):
            x = d(x)
            if k < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for u, c, s in zip(self.up, self.upc, skips[::-1]):
            x = c(torch.cat([u(x), s], 1))
        return self.head(x)[..., :H, :W]


def focal_heat(logit, heat, care, alpha=2.0, beta=4.0):
    """CenterNet focal loss on a Gaussian heatmap, over the cared-for pixels."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (heat > 0.99).float() * care
    neg = (1 - pos) * care
    lp = -((1 - p) ** alpha) * torch.log(p) * pos
    ln = -((1 - heat) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1.0)


def losses(out, heat, typ, jcare, lumen, centre, vcare):
    lj = focal_heat(out[:, 0:1], heat, jcare)
    lt = F.cross_entropy(out[:, 1:4], typ, ignore_index=-1) if (typ >= 0).any() else out.sum() * 0
    ll = (F.binary_cross_entropy_with_logits(out[:, 4:5], lumen, reduction="none") * vcare).sum() / vcare.sum().clamp(min=1)
    w = 1 + 9 * centre
    lc = (F.binary_cross_entropy_with_logits(out[:, 5:6], centre, weight=w, reduction="none") * vcare).sum() / (w * vcare).sum().clamp(min=1)
    return lj, lt, ll, lc


@torch.no_grad()
def predict(model, X, tile=512, pad=32):
    """Full image, in overlapping tiles: sigmoid heat, type probs, lumen, centreline."""
    model.eval()
    dev = next(model.parameters()).device
    H, W = X.shape
    out = np.zeros((6, H, W), np.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
            yb, xb = min(y0 + tile + pad, H), min(x0 + tile + pad, W)
            o = model(torch.from_numpy(np.ascontiguousarray(X[ya:yb, xa:xb]))[None, None].to(dev))[0].float().cpu().numpy()
            ye, xe = min(y0 + tile, H), min(x0 + tile, W)
            out[:, y0:ye, x0:xe] = o[:, y0 - ya:ye - ya, x0 - xa:xe - xa]
    o = out
    res = np.empty_like(o)
    res[0] = 1 / (1 + np.exp(-o[0]))
    e = np.exp(o[1:4] - o[1:4].max(0))
    res[1:4] = e / e.sum(0)
    res[4:6] = 1 / (1 + np.exp(-o[4:6]))
    return res


def peaks(P, thr=0.3, nms=5, valid=None):
    h = P[0]
    mx = ndi.maximum_filter(h, size=2 * nms + 1)
    ok = (h == mx) & (h >= thr)
    if valid is not None:
        ok &= valid
    ys, xs = np.nonzero(ok)
    names = ("crossing", "branch", "compound")
    return [dict(xy=np.array([x, y], float), score=float(h[y, x]), type=names[int(np.argmax(P[1:4, y, x]))])
            for y, x in zip(ys, xs)]


def score_scene(dets, S, thr=0.3):
    """vesselscene's rule: a detection matches the nearest unmatched observable
    junction within max(radius, lambda); unmatched ones near a don't-care
    junction are ignored."""
    js = S["junctions"]
    used, tp, fp, type_ok = set(), 0, 0, 0
    for d in sorted([d for d in dets if d["score"] >= thr], key=lambda d: -d["score"]):
        best, bd = None, 1e9
        for k, j in enumerate(js):
            if k in used:
                continue
            dd = math.hypot(d["xy"][0] - j["x"], d["xy"][1] - j["y"])
            if dd <= max(j["radius"], LAMBDA) and dd < bd:
                best, bd = k, dd
        if best is not None:
            used.add(best)
            tp += 1
            t = js[best]["type_observable"]
            type_ok += {"crossing": "crossing", "compound": "compound"}.get(t, "branch") == d["type"]
        elif any(math.hypot(d["xy"][0] - j["x"], d["xy"][1] - j["y"]) <= max(j["radius"], LAMBDA) for j in S["dc_junctions"]):
            continue
        else:
            fp += 1
    return dict(tp=tp, fp=fp, n=len(js), type_ok=type_ok)
