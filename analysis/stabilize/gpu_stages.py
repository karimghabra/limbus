"""The stabilization stages on the GPU (METHODS.md §14).

Each function stands in for one CPU function, named in its docstring, and
returns the same values: it is the loop over frames that moves to the GPU,
batched. Everything the pipeline decides from those values — the gate,
candidate choice, registered flags, medians, the robust fits of the
non-rigid stage — stays in the pipeline code shared with the CPU path.
"""
import numpy as np

from .gpu import Maps

# GPU working memory per pixel of one frame in a batch, in bytes: the peak
# torch.cuda.max_memory_allocated of each stage on the reference bursts, per
# frame and pixel, rounded up (warps by non-rigid fields measured ~134)
READ_BYTES = 48          # full-resolution frames: frame statistics, masks
VESSEL_BYTES = 240       # per working-scale pixel: vesselness at every scale
REFINE_BYTES = 160       # per working-scale pixel: complex128 spectra
WARP_BYTES = 160         # per pixel: dense fields, warps and their indices
PATCH_BYTES = 96         # per pixel of all patches of a frame


def _batches(items, size):
    items = np.asarray(items)
    for s in range(0, len(items), size):
        yield items[s:s + size]


class ShiftWarp:
    """Align frames by translation: register.shift(img, traj[i])."""

    def __init__(self, gpu, traj):
        self.gpu, self.traj = gpu, np.asarray(traj, np.float64)

    def many(self, imgs, ids):
        d = self.traj[np.asarray(ids, np.int64)]
        return [self.gpu.shift(x, d) for x in imgs]

    def __call__(self, img, ids):
        return self.many([img], ids)[0]


class FieldWarp:
    """Align frames by non-rigid fields: FieldEvaluator.warp(img, *dense(A[k],
    L[k])), k the row of each frame. A and L are read when called, so updates
    to them in place are seen."""

    def __init__(self, fields, A, L, row_of):
        self.fields, self.A, self.L, self.row_of = fields, A, L, row_of

    def many(self, imgs, ids):
        rows = [self.row_of[int(i)] for i in np.ravel(ids)]
        fx, fy = self.fields.dense(self.A[rows], self.L[rows])
        return [self.fields.warp(x, fx, fy) for x in imgs]

    def __call__(self, img, ids):
        return self.many([img], ids)[0]


# ---- pipeline passes 1 and 2 -------------------------------------------------------

def frame_stats(gpu, burst, scale, params):
    """Pass 1: quality.frame_stats of every frame, as three arrays."""
    n = len(burst)
    out = np.zeros((3, n))
    batch = gpu.batch(burst.width * burst.height, READ_BYTES)
    for pos, x in gpu.read(burst.files, np.arange(n), batch):
        out[:, pos] = gpu.frame_stats(gpu.to_working(x, scale), burst.full_scale, params)
    return out[0], out[1], out[2]


def _vessels_and_glare(gpu, x, scale, sigmas, c, params, full_scale):
    img = gpu.to_working(x, scale)
    v = gpu.vesselness(img, sigmas, c, params)
    glare = gpu.glare(img, full_scale, params, scale)
    return gpu.torch.where(glare, 0.0, v), glare


def sample_vesselness(gpu, burst, frames, scale, sigmas, c, params):
    """The envelope threshold's sample: vesselness of `frames` with glare set
    to 0, stacked as a NumPy array (the percentile is taken by the caller)."""
    h = int(np.rint(burst.height * scale))
    w = int(np.rint(burst.width * scale))
    batch = gpu.batch(h * w, VESSEL_BYTES)
    out = []
    for pos, x in gpu.read(burst.files, np.asarray(frames), batch):
        v, _ = _vessels_and_glare(gpu, x, scale, sigmas, c, params, burst.full_scale)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


def vessel_maps(gpu, burst, scale, sigmas, c, threshold, params, V, M, O):
    """Pass 2: vesselness V, envelope M and observed mask O of every frame,
    written to the host arrays and, when they fit, kept on the GPU."""
    n, h, w = V.shape
    maps = Maps(gpu, V, M, O)
    batch = gpu.batch(h * w, VESSEL_BYTES)
    for pos, x in gpu.read(burst.files, np.arange(n), batch):
        v, glare = _vessels_and_glare(gpu, x, scale, sigmas, c, params, burst.full_scale)
        maps.put(pos, v, gpu.envelope(v, threshold), (~glare).to(gpu.torch.uint8))
    return maps


# ---- registration --------------------------------------------------------------------

def build_templates(gpu, maps, warp, frames):
    """register.build_templates: coverage-normalised mask consensus p and mean
    vesselness, as (h, w) float32 tensors, and the coverage."""
    torch = gpu.torch
    h, w = maps.shape
    msum = torch.zeros((h, w), dtype=torch.float32, device=gpu.dev)
    vsum = torch.zeros_like(msum)
    cover = torch.zeros_like(msum)
    for ids in _batches(frames, gpu.batch(h * w, WARP_BYTES)):
        m, v, o = warp.many([maps.M.get(ids), maps.V.get(ids), maps.O.get(ids)], ids)
        msum += m.sum(0)
        vsum += v.sum(0)
        cover += o.sum(0)
    valid = cover > 0
    p = torch.where(valid, msum / torch.where(valid, cover, 1.0), 0.0)
    vmean = torch.where(valid, vsum / torch.where(valid, cover, 1.0), 0.0)
    return p, vmean, cover


def chained_initial(gpu, maps, frames, correlator):
    """register.chained_initial: consecutive good frames correlated in batches,
    their shifts added up in order."""
    n = len(maps.V)
    traj = np.zeros((n, 2), np.float64)
    a_all, b_all = frames[:-1], frames[1:]
    steps = np.zeros((len(a_all), 2))
    size = gpu.batch(np.prod(maps.shape), 2 * WARP_BYTES)
    for s in range(0, len(a_all), size):
        a, b = a_all[s:s + size], b_all[s:s + size]
        dx, dy, _ = correlator(maps.V.get(a), maps.V.get(b))
        steps[s:s + size] = np.stack([dx, dy], 1)
    for a, b, step in zip(a_all, b_all, steps):
        traj[b] = traj[a] + step
    traj[frames] -= np.median(traj[frames], axis=0)
    return traj


def register_groupwise(gpu, maps, good, params, scale, log=None):
    """register.register_groupwise, with each iteration's per-frame work —
    coarse correlation against the mask template, the sub-pixel refinement
    of every candidate — done for a batch of frames at a time."""
    n = len(maps.V)
    frames = np.flatnonzero(good)
    correlator = gpu.phase_correlator(maps.shape)
    refiner = gpu.refiner(maps.shape)
    traj = chained_initial(gpu, maps, frames, correlator)

    coarse_resp = np.zeros(n)
    fine_resp = np.zeros(n)
    registered = good.copy()
    converge = params.converge_rms_px * scale
    max_step = params.fine_max_step_px * scale
    history = []
    size = gpu.batch(np.prod(maps.shape), REFINE_BYTES)

    for iteration in range(1, params.max_iterations + 1):
        use = np.flatnonzero(registered) if registered.any() else frames
        p, vmean, _ = build_templates(gpu, maps, ShiftWarp(gpu, traj), use)
        refiner.set_template(vmean)
        p_spec = correlator.spectrum(p[None])
        new = traj.copy()
        for ids in _batches(frames, size):
            dx, dy, r_c = correlator(p_spec, maps.M.get(ids))
            d_coarse = np.stack([dx, dy], 1)
            spectra = refiner.spectra(maps.V.get(ids))
            rows = np.arange(len(ids))
            # candidates as on the CPU: the current estimate always, the coarse
            # one too when it is confident and lands elsewhere; the fine stage's
            # confidence decides, and the current estimate wins ties
            best_d, best_conf = refiner(spectra, rows, traj[ids], max_step)
            alt = ((r_c >= params.coarse_min_response)
                   & (np.hypot(*(d_coarse - traj[ids]).T) > max_step))
            if alt.any():
                d2, c2 = refiner(spectra, rows[alt], d_coarse[alt], max_step)
                better = c2 > best_conf[alt]
                sel = rows[alt][better]
                best_d[sel], best_conf[sel] = d2[better], c2[better]
            coarse_resp[ids], fine_resp[ids] = r_c, best_conf
            registered[ids] = best_conf >= params.fine_min_response
            new[ids] = np.where(registered[ids][:, None], best_d, traj[ids])
        keep = np.flatnonzero(registered)
        if keep.size:
            new[frames] -= np.median(new[keep], axis=0)
        change = float(np.sqrt(np.mean(np.sum((new[frames] - traj[frames]) ** 2, axis=1))))
        traj = new
        history.append(change / scale)
        if log:
            log(f"    iteration {iteration}: RMS change {change / scale:.3f} px, "
                f"registered {keep.size}/{frames.size}")
        if change < converge:
            break

    return {
        "traj": traj,
        "coarse_response": coarse_resp,
        "fine_response": fine_resp,
        "registered": registered,
        "iterations": len(history),
        "convergence_px": history,
    }


# ---- metrics ----------------------------------------------------------------------------

def mask_consensus(gpu, maps, frames, params, aligned, warp=None):
    """metrics.mask_consensus. The sums are whole counts, so they are exact
    whatever the order of addition."""
    torch = gpu.torch
    h, w = maps.shape
    frames = np.asarray(frames)
    size = gpu.batch(h * w, WARP_BYTES)

    def masks(ids):
        m, obs = maps.M.get(ids), maps.O.get(ids)
        if aligned:
            m, obs = warp.many([m, obs], ids)
        return m >= 0.5, obs >= 0.5

    msum = torch.zeros((h, w), dtype=torch.float32, device=gpu.dev)
    cover = torch.zeros_like(msum)
    for ids in _batches(frames, size):
        mb, seen = masks(ids)
        msum += (mb & seen).sum(0)
        cover += seen.sum(0)
    valid = cover >= params.coverage_min_fraction * len(frames)
    p = torch.where(valid, msum / torch.where(valid, cover, 1.0), 0.0)
    consensus = (p >= 0.5) & valid
    k = msum[valid].double()
    nn = cover[valid].double()
    denom = float(torch.sum(k * (nn - 1)))
    overlap = float(torch.sum(k * (k - 1))) / denom if denom > 0 else 0.0

    dices = []
    for ids in _batches(frames, size):
        mb, seen = masks(ids)
        region = valid & seen
        mb = mb & region
        cons = consensus & region
        counts = torch.stack([mb.sum((1, 2)), cons.sum((1, 2)), (mb & cons).sum((1, 2))], 1)
        for n_m, n_c, n_both in counts.cpu().numpy().tolist():
            if n_m + n_c:
                dices.append(2 * n_both / (n_m + n_c))
    return {
        "overlap": overlap,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "vessel_fraction": float(consensus.sum()) / max(int(valid.sum()), 1),
    }, consensus.cpu().numpy()


def residual_motion(gpu, maps, frames, scale, params, warp):
    """metrics.residual_motion."""
    if len(frames) < 2:
        return {}
    pairs = list(zip(frames[:-1], frames[1:]))
    take = np.linspace(0, len(pairs) - 1, min(params.diagnostic_frames, len(pairs))).astype(int)
    a = np.array([pairs[k][0] for k in take])
    b = np.array([pairs[k][1] for k in take])
    correlator = gpu.phase_correlator(maps.shape)
    steps = []
    size = gpu.batch(np.prod(maps.shape), 2 * WARP_BYTES)
    for s in range(0, len(a), size):
        aa, bb = a[s:s + size], b[s:s + size]
        dx, dy, _ = correlator(warp(maps.V.get(aa), aa), warp(maps.V.get(bb), bb))
        steps.extend(np.hypot(dx, dy) / scale)
    steps = np.array(steps)
    return {"residual_step_median_px": float(np.median(steps)),
            "residual_step_p90_px": float(np.percentile(steps, 90))}


def closure_error(gpu, maps, frames, scale, params):
    """metrics.closure_error."""
    if len(frames) < 3:
        return {}
    triples = list(zip(frames[:-2], frames[1:-1], frames[2:]))
    take = np.linspace(0, len(triples) - 1,
                       min(params.diagnostic_frames, len(triples))).astype(int)
    a, b, c = (np.array([triples[k][j] for k in take]) for j in range(3))
    correlator = gpu.phase_correlator(maps.shape)
    errs = []
    size = gpu.batch(np.prod(maps.shape), 3 * WARP_BYTES)
    for s in range(0, len(a), size):
        Va, Vb, Vc = (maps.V.get(x[s:s + size]) for x in (a, b, c))
        ab = np.stack(correlator(Va, Vb)[:2], 1)
        bc = np.stack(correlator(Vb, Vc)[:2], 1)
        ac = np.stack(correlator(Va, Vc)[:2], 1)
        errs.extend(np.hypot(*(ab + bc - ac).T) / scale)
    errs = np.array(errs)
    return {"closure_median_px": float(np.median(errs)),
            "closure_p90_px": float(np.percentile(errs, 90))}


def rotation_diagnostic(gpu, maps, template_v, frames, scale, params, warp):
    """metrics.rotation_diagnostic; template_v is an (h, w) tensor."""
    h, w = maps.shape
    if h < 64 or w < 64 or len(frames) == 0:
        return {}
    quads = [(0, h // 2, 0, w // 2), (0, h // 2, w // 2, w),
             (h // 2, h, 0, w // 2), (h // 2, h, w // 2, w)]
    correlators = [gpu.phase_correlator((y1 - y0, x1 - x0)) for y0, y1, x0, x1 in quads]
    refs = [corr.spectrum(template_v[None, y0:y1, x0:x1])
            for (y0, y1, x0, x1), corr in zip(quads, correlators)]
    take = np.linspace(0, len(frames) - 1, min(params.diagnostic_frames, len(frames))).astype(int)
    sel = np.asarray(frames)[take]
    spreads = []
    for ids in _batches(sel, gpu.batch(h * w, 2 * WARP_BYTES)):
        aligned = warp(maps.V.get(ids), ids)
        shifts = np.stack([np.stack(corr(ref, aligned[:, y0:y1, x0:x1])[:2], 1)
                           for (y0, y1, x0, x1), corr, ref in zip(quads, correlators, refs)], 1)
        for s in shifts:
            spreads.append(np.max(np.hypot(*(s - s.mean(axis=0)).T)) / scale)
    spreads = np.array(spreads)
    return {"quadrant_disagreement_median_px": float(np.median(spreads)),
            "quadrant_disagreement_p90_px": float(np.percentile(spreads, 90))}


# ---- projections ------------------------------------------------------------------------

def projections(gpu, burst, frames, warp, params):
    """Pass 3 (translation and non-rigid alike): the sums behind the raw and
    stabilized means and SD, one resampling of each original frame, glare
    left out. Returns raw_sum, st_sum, st_sq, cover as float64 arrays."""
    torch = gpu.torch
    H, W = burst.height, burst.width
    acc = torch.zeros((4, H, W), dtype=torch.float64, device=gpu.dev)
    batch = gpu.batch(H * W, READ_BYTES + 2 * WARP_BYTES)
    for pos, img in gpu.read(burst.files, np.asarray(frames), batch):
        acc[0] += img.double().sum(0)
        clear = (~gpu.glare(img, burst.full_scale, params, 1.0)).float()
        wimg, seen = warp.many([img, clear], pos)
        wimg = wimg.double()
        seen = seen >= 0.5
        acc[1] += torch.where(seen, wimg, 0.0).sum(0)
        acc[2] += torch.where(seen, wimg * wimg, 0.0).sum(0)
        acc[3] += seen.sum(0)
    out = acc.cpu().numpy()
    return out[0], out[1], out[2], out[3]


# ---- non-rigid refinement ------------------------------------------------------------------

def aligned_thumbnails(gpu, maps, warp, frames, step=4):
    """nonrigid._refine's reference search: each frame aligned, every step-th
    pixel, flattened; (len(frames), pixels) float32."""
    out = []
    for ids in _batches(frames, gpu.batch(np.prod(maps.shape), WARP_BYTES)):
        out.append(warp(maps.V.get(ids), ids)[:, ::step, ::step].reshape(len(ids), -1).cpu().numpy())
    return np.concatenate(out)


def template(gpu, maps, warp, frames):
    """nonrigid._refine's template: mean aligned vesselness over the frames
    that saw each pixel. Returns T and the coverage, (h, w) float32 tensors."""
    torch = gpu.torch
    h, w = maps.shape
    acc = torch.zeros((h, w), dtype=torch.float32, device=gpu.dev)
    cov = torch.zeros_like(acc)
    for ids in _batches(frames, gpu.batch(h * w, WARP_BYTES)):
        v, o = warp.many([maps.V.get(ids), maps.O.get(ids)], ids)
        acc += v.sum(0)
        cov += (o >= 0.5).sum(0)
    T = torch.where(cov > 0, acc / torch.clamp(cov, min=1), 0.0)
    return T, cov


def patch_shifts(gpu, maps, warp, frames, T, patch, stride, grid_shape, np_):
    """nonrigid._refine's patch measurements for every frame: each patch of
    the aligned frame phase-correlated with the same patch of the template T.
    Returns rdx, rdy (NaN where not measured) and weights wts, each
    (len(frames), rows, cols) float32, exactly as the CPU loop fills them."""
    torch = gpu.torch
    R, C = grid_shape
    correlator = gpu.phase_correlator((patch, patch))
    tp = T.unfold(0, patch, stride).unfold(1, patch, stride)[:R, :C]
    t_ok = tp.amax((-2, -1)) > 0
    t_spec = correlator.spectrum(tp)
    m = len(frames)
    rdx = np.full((m, R, C), np.nan, np.float32)
    rdy = np.full((m, R, C), np.nan, np.float32)
    wts = np.zeros((m, R, C), np.float32)
    h, w = maps.shape
    size = gpu.batch(R * C * patch * patch + h * w, PATCH_BYTES)
    done = 0
    for ids in _batches(frames, size):
        a = warp(maps.V.get(ids), ids)
        ap = a.unfold(1, patch, stride).unfold(2, patch, stride)[:, :R, :C]
        ok = (t_ok[None] & (ap.amax((-2, -1)) > 0)).cpu().numpy()
        dx, dy, resp = correlator(t_spec[None], ap)
        keep = (ok & (resp >= np_.patch_min_response)
                & (np.abs(dx) < patch / 4) & (np.abs(dy) < patch / 4))
        sl = slice(done, done + len(ids))
        rdx[sl] = np.where(keep, dx, np.nan)
        rdy[sl] = np.where(keep, dy, np.nan)
        wts[sl] = np.where(keep, resp, 0)
        done += len(ids)
    return rdx, rdy, wts


def update_rms(gpu, fields, A_u, l_u):
    """RMS of each update field over the image, sqrt(mean(ufx^2 + ufy^2)):
    the per-frame change nonrigid._refine converges on."""
    out = []
    for s in range(0, len(A_u), 64):
        ufx, ufy = fields.dense(A_u[s:s + 64], l_u[s:s + 64])
        out.append(gpu.torch.sqrt((ufx * ufx + ufy * ufy).double().mean((1, 2))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def template_ncc(gpu, maps, warp, frames, T, region):
    """np.corrcoef(aligned frame[region], T[region]) for every frame, in float64."""
    t = T[region].double()
    t = t - t.mean()
    tn = gpu.torch.sqrt((t * t).sum())
    out = []
    for ids in _batches(frames, gpu.batch(np.prod(maps.shape), WARP_BYTES)):
        a = warp(maps.V.get(ids), ids)[:, region].double()
        a = a - a.mean(1, keepdim=True)
        r = (a * t).sum(1) / (gpu.torch.sqrt((a * a).sum(1)) * tn)
        out.append(r.clamp(-1, 1).cpu().numpy())
    return np.concatenate(out)


def observed_template(gpu, maps, warp, frames):
    """The non-rigid metrics' template_v: aligned vesselness summed, divided
    by the summed (interpolated) observed masks."""
    torch = gpu.torch
    h, w = maps.shape
    acc = torch.zeros((h, w), dtype=torch.float32, device=gpu.dev)
    cov = torch.zeros_like(acc)
    for ids in _batches(frames, gpu.batch(h * w, WARP_BYTES)):
        v, o = warp.many([maps.V.get(ids), maps.O.get(ids)], ids)
        acc += v.sum(0)
        cov += o.sum(0)
    return torch.where(cov > 0, acc / torch.clamp(cov, min=1e-6), 0.0)


def tile_spread(gpu, maps, frames, warp, grid_shape, scale):
    """nonrigid.tile_spread."""
    torch = gpu.torch
    rows, cols = grid_shape
    h, w = maps.shape
    th, tw = h // rows, w // cols
    if th < 32 or tw < 32 or len(frames) < 2:
        return None
    size = gpu.batch(h * w, 2 * WARP_BYTES)
    mean = torch.zeros((h, w), dtype=torch.float32, device=gpu.dev)
    for ids in _batches(frames, size):
        mean += warp(maps.V.get(ids), ids).sum(0)
    mean /= len(frames)

    def tiles(x):   # (..., h, w) -> (..., rows, cols, th, tw)
        lead = x.shape[:-2]
        t = x[..., :rows * th, :cols * tw].reshape(*lead, rows, th, cols, tw)
        return t.transpose(-3, -2)

    correlator = gpu.phase_correlator((th, tw))
    ref = correlator.spectrum(tiles(mean))
    res = []
    for ids in _batches(frames, size):
        dx, dy, _ = correlator(ref[None], tiles(warp(maps.V.get(ids), ids)))
        res.append(np.stack([dx / scale, dy / scale], -1))
    res = np.concatenate(res)
    det = res - np.median(res.reshape(len(frames), -1, 2), axis=1)[:, None, None, :]
    return np.hypot(*np.std(np.clip(det, -30, 30), axis=0).transpose(2, 0, 1))
