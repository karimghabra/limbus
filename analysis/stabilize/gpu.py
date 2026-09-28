"""GPU backend: the per-frame image work, batched on an NVIDIA GPU (METHODS.md §14).

The CPU path — OpenCV and NumPy, one frame at a time — is the validated
reference. This module does the same image work with PyTorch on a CUDA GPU,
many frames at once: filtering and vesselness, glare and envelope masks,
phase correlation, the sub-pixel refinement, displacement fields and every
warp. Each primitive names the CPU function it stands in for and reproduces
its arithmetic, including OpenCV's conventions (borders, window, spectrum
normalisation, peak centroid), so the two paths agree to rounding. The
stages built from these primitives are in gpu_stages.py; tests/test_gpu.py
checks both against the CPU path.

Only numbers leave the GPU: the decisions made on them (the gate, candidate
choice, robust fits, medians) run in NumPy exactly as on the CPU path.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .bursts import read_frame

ENV_VAR = "STABILIZE_DEVICE"
DEVICES = ("auto", "cuda", "cpu")
FLT_EPSILON = float(np.finfo(np.float32).eps)   # OpenCV's divSpectrums guard
DBL_EPSILON = float(np.finfo(np.float64).eps)   # OpenCV's weightedCentroid guard

_instance = None


def select(device=None):
    """The GPU to run on, or None for the CPU path.

    'auto' uses a CUDA GPU if PyTorch finds one and the CPU otherwise;
    'cuda' insists (and says why it can't); 'cpu' never uses the GPU.
    None reads the STABILIZE_DEVICE environment variable, default 'auto'.
    """
    global _instance
    choice = (device or os.environ.get(ENV_VAR) or "auto").strip().lower()
    if choice not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, not {choice!r}")
    if choice == "cpu":
        return None
    try:
        import torch
    except ImportError:
        if choice == "cuda":
            raise RuntimeError("device 'cuda' needs PyTorch with CUDA: pip install torch "
                               "--index-url https://download.pytorch.org/whl/cu128") from None
        return None
    if _instance is None:
        # is_available() alone can say yes with no usable device (e.g. an
        # empty CUDA_VISIBLE_DEVICES), so ask for a device and start it
        try:
            if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
                raise RuntimeError("PyTorch found no CUDA GPU")
            _instance = Gpu()
        except Exception as exc:
            if choice == "cuda":
                raise RuntimeError(f"device 'cuda': {exc} (torch {torch.__version__})") from None
            return None
    return _instance


def describe(gpu):
    """What a result was computed on, for metrics.json."""
    return "cpu" if gpu is None else gpu.describe()


def _reflect101(n, pad):
    """Source index of each position -pad .. n-1+pad under BORDER_REFLECT_101
    (OpenCV's default: dcb|abcd|cba), reflecting again as often as needed."""
    i = np.arange(-pad, n + pad)
    if n == 1:
        return np.zeros_like(i)
    period = 2 * (n - 1)
    i = np.mod(i, period)
    return np.where(i >= n, period - i, i)


def _cubic_matrix(g, n):
    """(len(g), n) weights of cv2.remap's INTER_CUBIC (A = -0.75) sampling a
    line of n nodes at positions g, with BORDER_REPLICATE: row j of the matrix
    times the node values is the interpolated value at g[j]."""
    g = np.asarray(g, np.float32).astype(np.float64)
    i0 = np.floor(g).astype(np.int64)
    t = g - i0
    a = -0.75
    w0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
    w1 = ((a + 2) * t - (a + 3)) * t * t + 1
    w2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
    rows = np.arange(g.size)
    out = np.zeros((g.size, n))
    for k, w in enumerate((w0, w1, w2, 1 - w0 - w1 - w2)):
        np.add.at(out, (rows, np.clip(i0 - 1 + k, 0, n - 1)), w)
    return out


class Gpu:
    """One CUDA device and the primitives that run on it."""

    def __init__(self):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.dev = torch.device("cuda", torch.cuda.current_device())
        # Full float32 everywhere. On Ampere and later GPUs PyTorch runs
        # float32 convolutions as TF32 by default: a 10-bit mantissa, ~1000x
        # coarser than the arithmetic of the CPU path.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.name = torch.cuda.get_device_name(self.dev)
        free, _ = torch.cuda.mem_get_info(self.dev)
        # a share of what is free now: working memory for one batch, and
        # (separately) room to keep a burst's vessel maps on the card; the
        # rest is headroom for PyTorch's allocator and other programs
        self.budget = int(0.3 * free)
        self.resident_budget = int(0.3 * free)
        self._cache = {}

    def describe(self):
        return f"cuda: {self.name} (PyTorch {self.torch.__version__})"

    def out_of_memory(self, exc):
        return (isinstance(exc, self.torch.cuda.OutOfMemoryError)
                or "CUDA out of memory" in str(exc))

    def release(self):
        """Hand cached blocks back to the driver, e.g. after running out."""
        self._cache.clear()
        self.torch.cuda.empty_cache()

    def batch(self, pixels, bytes_per_pixel, cap=64):
        """Frames per batch for a working set of bytes_per_pixel per pixel."""
        return int(max(1, min(cap, self.budget // max(1, int(pixels) * int(bytes_per_pixel)))))

    def tensor(self, a, dtype=None):
        a = np.ascontiguousarray(a)
        if not a.flags.writeable:        # read-only memmaps: PyTorch wants to own a copy
            a = a.copy()
        t = self.torch.as_tensor(a, device=self.dev)
        return t if dtype is None else t.to(dtype)

    def _cached(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # ---- frames ----------------------------------------------------------------

    def read(self, files, positions, batch, workers=6):
        """Yield (positions, frames) for files[positions], `batch` at a time:
        frames is a (B, H, W) float32 tensor of the raw sensor values. The
        next batches are read from disk in threads while this one is on the GPU."""
        groups = [list(positions[s:s + batch]) for s in range(0, len(positions), batch)]
        with ThreadPoolExecutor(workers) as pool:
            ahead = [[pool.submit(read_frame, files[i]) for i in g] for g in groups[:2]]
            for n, g in enumerate(groups):
                arrays = [f.result() for f in ahead.pop(0)]
                if n + 2 < len(groups):
                    ahead.append([pool.submit(read_frame, files[i]) for i in groups[n + 2]])
                yield g, self.tensor(np.stack(arrays)).float()

    def to_working(self, x, scale):
        """vessels.to_working: area downscaling. An exact halving is a 2x2
        mean, as in OpenCV; any other size is left to OpenCV on the CPU."""
        if scale == 1.0:
            return x
        B, h, w = x.shape
        oh, ow = int(np.rint(h * scale)), int(np.rint(w * scale))   # cv2's cvRound
        return self.resize_area(x, (oh, ow))

    def resize_area(self, x, size):
        """cv2.resize(..., INTER_AREA) to size (h, w)."""
        B, h, w = x.shape
        oh, ow = size
        if h == 2 * oh and w == 2 * ow:
            return self.F.avg_pool2d(x[:, None], 2)[:, 0]
        host = x.cpu().numpy()
        return self.tensor(np.stack([cv2.resize(im, (ow, oh), interpolation=cv2.INTER_AREA)
                                     for im in host]))

    def resize_linear(self, x, size):
        """cv2.resize(..., INTER_LINEAR) to size (h, w): the same half-pixel
        mapping and edge clamping as PyTorch's bilinear, align_corners=False."""
        return self.F.interpolate(x[:, None], size=size, mode="bilinear",
                                  align_corners=False)[:, 0]

    # ---- filters -----------------------------------------------------------------

    def _sep(self, x, kx, ky):
        """Separable correlation of (B, H, W) images with 1-D kernels kx along
        rows and ky down columns, BORDER_REFLECT_101 — cv2.sepFilter2D's
        arithmetic, which GaussianBlur and Sobel use."""
        F, torch = self.F, self.torch
        B, H, W = x.shape
        kx = np.asarray(kx, np.float32)
        ky = np.asarray(ky, np.float32)
        kxt = self._cached(("k", kx.tobytes()), lambda: self.tensor(kx)).view(1, 1, 1, -1)
        kyt = self._cached(("k", ky.tobytes()), lambda: self.tensor(ky)).view(1, 1, -1, 1)
        rx, ry = kx.size // 2, ky.size // 2
        xi = self._cached(("r101", W, rx), lambda: self.tensor(_reflect101(W, rx)))
        yi = self._cached(("r101", H, ry), lambda: self.tensor(_reflect101(H, ry)))
        y = F.conv2d(x.index_select(2, xi)[:, None], kxt)[:, 0]
        return F.conv2d(y.index_select(1, yi)[:, None], kyt)[:, 0]

    def gaussian_blur(self, x, sigma):
        """cv2.GaussianBlur(img, (0, 0), sigma) for float32 images: OpenCV's
        kernel size for non-8-bit images and its own kernel values."""
        k = int(round(sigma * 8 + 1)) | 1
        kern = cv2.getGaussianKernel(k, sigma, cv2.CV_32F).ravel()
        return self._sep(x, kern, kern)

    def laplacian(self, x):
        """cv2.Laplacian(img, CV_32F) with its default ksize=1 kernel."""
        k = np.float32([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        B, H, W = x.shape
        kt = self._cached(("lap",), lambda: self.tensor(k)).view(1, 1, 3, 3)
        xi = self._cached(("r101", W, 1), lambda: self.tensor(_reflect101(W, 1)))
        yi = self._cached(("r101", H, 1), lambda: self.tensor(_reflect101(H, 1)))
        return self.F.conv2d(x.index_select(2, xi).index_select(1, yi)[:, None], kt)[:, 0]

    def hessian_eigenvalues(self, x, sigma):
        """vessels._hessian_eigenvalues for a batch."""
        torch = self.torch
        g = self.gaussian_blur(x, sigma)
        norm = sigma * sigma
        d2, sm, d1 = (1, -2, 1), (1, 2, 1), (-1, 0, 1)   # cv2.Sobel's ksize=3 kernels
        dxx = self._sep(g, d2, sm) * norm
        dyy = self._sep(g, sm, d2) * norm
        dxy = self._sep(g, d1, d1) * norm
        root = torch.sqrt((dxx - dyy) ** 2 + 4.0 * dxy ** 2)
        a = 0.5 * (dxx + dyy - root)
        b = 0.5 * (dxx + dyy + root)
        swap = a.abs() > b.abs()
        return torch.where(swap, b, a), torch.where(swap, a, b)

    def _frangi(self, x, sigma, c, beta):
        """vessels._vesselness_one_scale. Float64 where NumPy's type promotion
        makes the original float64 (the ratio term and the product)."""
        torch = self.torch
        l1, l2 = self.hessian_eigenvalues(x, sigma)
        tiny = torch.tensor(1e-6, dtype=torch.float64, device=self.dev)
        rb = l1.double() / (l2.double() + torch.where(l2 >= 0, tiny, -tiny))
        s2 = l1 ** 2 + l2 ** 2
        v = torch.exp(-(rb ** 2) / (2 * beta ** 2)) * (1.0 - torch.exp(-s2 / (2 * c ** 2))).double()
        return torch.where(l2 <= 0, 0.0, v)

    def vesselness(self, x, sigmas, c, params):
        """vessels.vesselness for a batch of working-scale images."""
        torch = self.torch
        B, h, w = x.shape
        out = torch.zeros((B, h, w), dtype=torch.float32, device=self.dev)
        half = None
        for s in sigmas:
            if s >= 8 and min(h, w) >= 64:
                if half is None:
                    half = self.resize_area(x, (h // 2, w // 2))
                v = self.resize_linear(self._frangi(half, s / 2.0, c, params.frangi_beta), (h, w))
            else:
                v = self._frangi(x, s, c, params.frangi_beta)
            out = torch.maximum(out.double(), v).float()   # np.maximum(out, v, out=out)
        return out

    def glare(self, x, full_scale, params, scale):
        """vessels.glare_mask for a batch: clipped pixels, dilated by a disc.
        Dilation of a binary mask is 'any pixel of the disc set', counted by
        a convolution; the zero padding matches OpenCV, whose dilate ignores
        the border."""
        clipped = x >= params.glare_fraction_of_full_scale * full_scale
        if not bool(clipped.any()):
            return clipped
        k = max(3, int(round(params.glare_dilate_px * scale)) | 1)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32)
        kt = self._cached(("disk", k), lambda: self.tensor(disk)).view(1, 1, k, k)
        return self.F.conv2d(clipped[:, None].float(), kt, padding=k // 2)[:, 0] > 0.5

    def envelope(self, v, threshold):
        """vessels.envelope for a batch: threshold, then a 3x3 opening. Max
        pooling pads with -inf, so, as with OpenCV's defaults, the border
        neither erodes nor dilates."""
        F = self.F
        m = (v > threshold).float()[:, None]
        m = -F.max_pool2d(-m, 3, 1, 1)
        return F.max_pool2d(m, 3, 1, 1)[:, 0].to(self.torch.uint8)

    def frame_stats(self, x, full_scale, params):
        """quality.frame_stats for a batch of working-scale frames:
        (log sharpness, mean, clipped fraction), each a float64 array."""
        smooth = self.gaussian_blur(x, params.sharpness_blur_sigma)
        sharp = self.laplacian(smooth).double().var(dim=(1, 2), unbiased=False)
        clipped = (x >= params.glare_fraction_of_full_scale * full_scale).double().mean(dim=(1, 2))
        mean = x.double().mean(dim=(1, 2))
        return (np.log(sharp.cpu().numpy() + 1.0), mean.cpu().numpy(), clipped.cpu().numpy())

    # ---- warps ---------------------------------------------------------------------

    def bilinear(self, img, ix, iy, fx, fy):
        """Bilinear sampling with a zero border, as cv2.remap and warpAffine do
        with INTER_LINEAR and BORDER_CONSTANT 0 (OpenCV 5 interpolates exactly,
        with no 1/32-pixel table). Sample point = (ix + fx, iy + fy): integer
        and fractional parts, each broadcastable to (B, H, W); neighbours that
        fall outside the image count as 0."""
        torch = self.torch
        B, h, w = img.shape
        shape = torch.broadcast_shapes(ix.shape, iy.shape, fx.shape, fy.shape)
        H, W = shape[-2], shape[-1]
        flat = img.reshape(B, h * w)
        out = torch.zeros((B, H, W), dtype=img.dtype, device=self.dev)
        for dy, wy in ((0, 1 - fy), (1, fy)):
            yy = iy + dy
            in_y = (yy >= 0) & (yy < h)
            row = yy.clamp(0, h - 1) * w
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xx = ix + dx
                ok = in_y & (xx >= 0) & (xx < w)
                lin = (row + xx.clamp(0, w - 1)).expand(B, H, W).reshape(B, H * W)
                vals = flat.gather(1, lin).reshape(B, H, W)
                out += torch.where(ok, vals * (wy * wx), 0.0)
        return out

    def _grid(self, h, w):
        torch = self.torch
        return (self._cached(("xs", w), lambda: torch.arange(w, device=self.dev).view(1, 1, w)),
                self._cached(("ys", h), lambda: torch.arange(h, device=self.dev).view(1, h, 1)))

    def shift(self, img, d):
        """register.shift for a batch: warp each image by -d[b], undoing a
        content displacement d[b] = (dx, dy). OpenCV reads the offset from a
        float32 matrix, so it is rounded to float32 first; the sample point
        x + d is then split exactly into whole and fractional pixels."""
        torch = self.torch
        B, h, w = img.shape
        D = np.asarray(d, np.float64).reshape(B, 2).astype(np.float32).astype(np.float64)
        whole = np.floor(D)
        frac = self.tensor((D - whole).astype(np.float32))
        whole = self.tensor(whole.astype(np.int64))
        xs, ys = self._grid(h, w)
        return self.bilinear(img, xs + whole[:, 0].view(B, 1, 1), ys + whole[:, 1].view(B, 1, 1),
                             frac[:, 0].view(B, 1, 1), frac[:, 1].view(B, 1, 1))

    def remap(self, img, mapx, mapy):
        """cv2.remap(img, mapx, mapy, INTER_LINEAR, BORDER_CONSTANT 0) for a
        batch of float32 maps."""
        fx0, fy0 = self.torch.floor(mapx), self.torch.floor(mapy)
        return self.bilinear(img, fx0.long(), fy0.long(), mapx - fx0, mapy - fy0)

    # ---- phase correlation -----------------------------------------------------------

    def phase_correlator(self, shape):
        return self._cached(("pc", tuple(shape)), lambda: PhaseCorrelator(self, shape))

    def refiner(self, shape, factor=20, alpha=0.5):
        return SubpixelRefiner(self, shape, factor, alpha)

    def fields(self, shape, grid):
        return Fields(self, shape, grid)


class PhaseCorrelator:
    """cv2.phaseCorrelate for batches (register.phase_correlate): zero-pad to
    OpenCV's optimal DFT size, multiply by the Hanning window, normalise the
    cross-power spectrum (with OpenCV's FLT_EPSILON guard), inverse transform
    without scaling, fftshift, take the first maximum, and read the shift from
    the 5x5 weighted centroid around it. Float32 transforms, as in OpenCV.

    OpenCV 5 recentres with a true fftshift for every size; earlier versions
    swapped quadrants, which differs for odd sizes (checked against cv2 5.0).
    """

    def __init__(self, gpu, shape):
        self.gpu = gpu
        torch = gpu.torch
        self.h, self.w = h, w = shape
        self.M, self.N = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
        win = np.zeros((self.M, self.N), np.float32)
        win[:h, :w] = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self.window = gpu.tensor(win)
        self.offsets = torch.arange(-2, 3, device=gpu.dev)

    def spectrum(self, x):
        """Window and transform (..., h, w) images; the result can be passed to
        __call__ in place of the images, e.g. a template used many times."""
        xp = self.gpu.F.pad(x, (0, self.N - self.w, 0, self.M - self.h))
        return self.gpu.torch.fft.rfft2(xp * self.window)

    def __call__(self, ref, mov):
        """Shift of each mov relative to ref: arrays (dx, dy, response).
        ref and mov are (..., h, w) images or their spectra, with leading
        dimensions that broadcast (one template against many frames, say);
        the results have the broadcast leading shape."""
        torch = self.gpu.torch
        M, N = self.M, self.N
        F1 = ref if ref.is_complex() else self.spectrum(ref)
        F2 = mov if mov.is_complex() else self.spectrum(mov)
        P = F1 * F2.conj()
        lead = P.shape[:-2]
        mag = P.abs()
        C = torch.fft.irfft2(P * (mag / (mag * mag + FLT_EPSILON)), s=(M, N)) * (M * N)
        C = torch.roll(C.reshape(-1, M, N), (M // 2, N // 2), (1, 2))
        B = C.shape[0]
        peak = C.reshape(B, -1).argmax(1)
        py, px = peak // N, peak % N
        rows = py[:, None] + self.offsets
        cols = px[:, None] + self.offsets
        ok = (((rows >= 0) & (rows < M))[:, :, None] & ((cols >= 0) & (cols < N))[:, None, :])
        lin = rows.clamp(0, M - 1)[:, :, None] * N + cols.clamp(0, N - 1)[:, None, :]
        vals = torch.where(ok, C.reshape(B, -1).gather(1, lin.reshape(B, 25)).reshape(B, 5, 5),
                           0.0).double()
        total = vals.sum((1, 2))
        cx = (vals * cols[:, None, :]).sum((1, 2)) / (total + DBL_EPSILON)
        cy = (vals * rows[:, :, None]).sum((1, 2)) / (total + DBL_EPSILON)
        out = torch.stack([N / 2.0 - cx, M / 2.0 - cy, total / (M * N)]).cpu().numpy()
        return tuple(o.reshape(lead) for o in out)


class SubpixelRefiner:
    """register.SubpixelRefiner for batches, in float64 like the original:
    half-whitened cross-power spectrum, integer peak searched near a coarse
    estimate, then the upsampled DFT around it."""

    def __init__(self, gpu, shape, factor=20, alpha=0.5):
        self.gpu = gpu
        self.shape = np.array(shape)
        h, w = shape
        self.window = gpu.tensor(cv2.createHanningWindow((w, h), cv2.CV_64F))
        self.factor, self.alpha = factor, alpha
        self.size = int(np.ceil(factor * 1.5))
        self.mid = np.trunc(self.size / 2.0)
        self.freq_r = gpu.tensor(np.fft.fftfreq(h, factor))
        self.freq_c = gpu.tensor(np.fft.fftfreq(w, factor))
        self.ramp = gpu.tensor(np.arange(self.size, dtype=np.float64))

    def set_template(self, template):
        self.F = self.gpu.torch.fft.fft2(template.double() * self.window)

    def spectra(self, frames):
        """Everything the refinement needs from a batch of (B, h, w) frames
        that does not depend on the coarse estimate: the half-whitened
        spectrum, its correlation surface, and the fully whitened surface
        that gives the confidence."""
        torch = self.gpu.torch
        G = torch.fft.fft2(frames.double() * self.window)
        prod = self.F * G.conj()
        mag = prod.abs().clamp_min(1e-12)
        phase_cc = torch.fft.ifft2(prod / mag).abs()
        weighted = prod / mag ** self.alpha
        cc = torch.fft.ifft2(weighted).abs()
        return weighted, phase_cc, cc

    def __call__(self, spectra, rows, coarse_d, radius):
        """Refine batch rows `rows` of `spectra` from their coarse estimates
        (len(rows), 2). Returns (d, confidence) as NumPy arrays."""
        torch, gpu = self.gpu.torch, self.gpu
        weighted, phase_cc, cc = spectra
        rows = np.asarray(rows, np.int64)
        coarse_d = np.asarray(coarse_d, np.float64).reshape(-1, 2)
        n = rows.size
        h, w = self.shape
        # integer peak near the coarse estimate: the same candidates in the
        # same order as the CPU loop (dy outer, dx inner; first maximum wins)
        centre = np.stack([-coarse_d[:, 1], -coarse_d[:, 0]], 1)
        r = int(np.ceil(radius))
        steps = np.array([(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)], float)
        cand = np.round(centre)[:, None, :] + steps[None]
        ci = (cand % self.shape).astype(np.int64)
        brow = gpu.tensor(rows)
        vals = cc[brow[:, None], gpu.tensor(ci[..., 0]), gpu.tensor(ci[..., 1])]
        pick = vals.argmax(1).cpu().numpy()
        best = cand[np.arange(n), pick]
        bi = (best % self.shape).astype(np.int64)
        confidence = phase_cc[brow, gpu.tensor(bi[:, 0]), gpu.tensor(bi[:, 1])].cpu().numpy()
        # upsampled DFT around it: up = E_r @ conj(weighted) @ E_c^T
        off = gpu.tensor(self.mid - best * self.factor)
        e_r = torch.exp(-2j * np.pi * ((self.ramp[None, :] - off[:, 0:1])[:, :, None]
                                       * self.freq_r[None, None, :]))
        e_c = torch.exp(-2j * np.pi * ((self.ramp[None, :] - off[:, 1:2])[:, :, None]
                                       * self.freq_c[None, None, :]))
        up = (e_r @ weighted[brow].conj() @ e_c.transpose(1, 2)).conj()
        peak = up.abs().reshape(n, -1).argmax(1).cpu().numpy()
        peak = np.stack([peak // self.size, peak % self.size], 1).astype(float) - self.mid
        rowcol = best + peak / self.factor
        return np.stack([-rowcol[:, 1], -rowcol[:, 0]], 1), confidence


class Fields:
    """fields.FieldEvaluator on the GPU: displacement fields d(x) = A @ [x, y, 1]
    + L(x) for one image shape and patch grid, and warps by them.

    The affine part is evaluated exactly, in float64, then rounded to
    float32 as on the CPU. The local residual is cv2.remap's bicubic
    (A = -0.75, BORDER_REPLICATE) interpolation of the node grid; because
    the sample positions along x depend only on the column, and along y only
    on the row, it is separable: L(x) = W_y @ L_nodes @ W_x^T.
    """

    def __init__(self, gpu, shape, grid=None):
        self.gpu = gpu
        torch = gpu.torch
        self.shape = h, w = shape
        self.X = torch.arange(w, dtype=torch.float32, device=gpu.dev).view(1, 1, w)
        self.Y = torch.arange(h, dtype=torch.float32, device=gpu.dev).view(1, h, 1)
        self.grid = grid
        self.local = grid is not None and grid.rows and grid.cols
        if self.local:
            # the CPU path's node coordinates, float32 arithmetic included
            gx = (np.arange(w, dtype=np.float32) - grid.x0) / grid.stride
            gy = (np.arange(h, dtype=np.float32) - grid.y0) / grid.stride
            self.Wx = gpu.tensor(_cubic_matrix(gx.astype(np.float32), grid.cols))
            self.Wy = gpu.tensor(_cubic_matrix(gy.astype(np.float32), grid.rows))

    def dense(self, A, L=None):
        """Fields (fx, fy), each (B, h, w) float32, for affines A (B, 2, 3)
        and node residuals L (B, rows, cols, 2)."""
        A = self.gpu.tensor(np.asarray(A, np.float64).reshape(-1, 2, 3))
        X, Y = self.X.double(), self.Y.double()
        a = [A[:, r, k].view(-1, 1, 1) for r in (0, 1) for k in (0, 1, 2)]
        fx = (a[0] * X + a[1] * Y + a[2]).float()
        fy = (a[3] * X + a[4] * Y + a[5]).float()
        if L is not None and self.local:
            Lt = self.gpu.tensor(np.asarray(L, np.float32).reshape(
                -1, self.grid.rows, self.grid.cols, 2)).double()
            fx = fx + (self.Wy @ Lt[..., 0] @ self.Wx.T).float()
            fy = fy + (self.Wy @ Lt[..., 1] @ self.Wx.T).float()
        return fx, fy

    def warp(self, img, fx, fy):
        """output(x) = img(x + field(x)); outside the frame -> 0."""
        return self.gpu.remap(img, self.X + fx, self.Y + fy)


class Stack:
    """A (N, h, w) stack of working-scale maps — vesselness V, envelope M or
    observed mask O — handed to the GPU a batch of frames at a time. Kept on
    the card when it fits; otherwise read from the host array each time."""

    def __init__(self, gpu, host, device_copy=None):
        self.gpu, self.host = gpu, host
        self.shape = tuple(host.shape)
        self.dev = device_copy

    def __len__(self):
        return self.shape[0]

    def get(self, frames):
        """Frames as a (B, h, w) float32 tensor."""
        frames = np.asarray(frames, np.int64).ravel()
        if self.dev is not None:
            return self.dev.index_select(0, self.gpu.tensor(frames)).float()
        return self.gpu.tensor(np.asarray(self.host[frames])).float()


class Maps:
    """The vessel maps of one burst: Stacks V, M, O over the same frames."""

    def __init__(self, gpu, V, M, O, resident=None):
        self.gpu = gpu
        n, h, w = V.shape
        self.shape = (h, w)
        if resident is None:
            resident = n * h * w * 6 <= gpu.resident_budget
        copies = {}
        if resident:
            for key, arr, dt in (("V", V, gpu.torch.float32), ("M", M, gpu.torch.uint8),
                                 ("O", O, gpu.torch.uint8)):
                copies[key] = gpu.torch.empty((n, h, w), dtype=dt, device=gpu.dev)
        self.resident = resident
        self.V = Stack(gpu, V, copies.get("V"))
        self.M = Stack(gpu, M, copies.get("M"))
        self.O = Stack(gpu, O, copies.get("O"))

    def put(self, frames, v, m, o):
        """Store computed maps for frames: on the card if resident, and always
        in the host arrays (on-disk memmaps the non-rigid stage reads)."""
        frames = np.asarray(frames, np.int64)
        if self.resident:
            idx = self.gpu.tensor(frames)
            self.V.dev.index_copy_(0, idx, v)
            self.M.dev.index_copy_(0, idx, m)
            self.O.dev.index_copy_(0, idx, o)
        lo, hi = int(frames[0]), int(frames[-1]) + 1
        if hi - lo == frames.size:
            self.V.host[lo:hi] = v.cpu().numpy()
            self.M.host[lo:hi] = m.cpu().numpy()
            self.O.host[lo:hi] = o.cpu().numpy()
        else:
            self.V.host[frames] = v.cpu().numpy()
            self.M.host[frames] = m.cpu().numpy()
            self.O.host[frames] = o.cpu().numpy()

    @classmethod
    def load(cls, gpu, V, M, O):
        """Maps already on disk (the translation stage's, for non-rigid)."""
        maps = cls(gpu, V, M, O)
        if maps.resident:
            n = V.shape[0]
            step = max(1, gpu.batch(V.shape[1] * V.shape[2], 6, cap=256))
            for s in range(0, n, step):
                idx = gpu.tensor(np.arange(s, min(n, s + step)))
                maps.V.dev.index_copy_(0, idx, gpu.tensor(np.asarray(V[s:s + step])))
                maps.M.dev.index_copy_(0, idx, gpu.tensor(np.asarray(M[s:s + step])))
                maps.O.dev.index_copy_(0, idx, gpu.tensor(np.asarray(O[s:s + step])))
        return maps
