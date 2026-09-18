"""Displacement fields: compact storage and evaluation (METHODS.md §13).

A frame's correction is a displacement field d(x): the stabilized frame is

    output(x) = frame(x + d(x))

Non-rigid results store each field compactly rather than as a dense map:

    d(x) = A @ [x, y, 1]  +  L(x)

where A is a 2x3 affine (translation, rotation, magnification, shear),
evaluated exactly at every pixel, and L is a smooth local residual known at a
grid of patch centres and interpolated bicubically between them. A
translation is the special case A = [[0, 0, dx], [0, 0, dy]], L = 0.

Evaluating the affine exactly matters. Interpolating it from the patch
centres and replicating beyond the outermost ones under-corrects the corners
— exactly where rotation displaces most.

Only numpy and OpenCV are needed, so the camera app's Review tab can use this
module without the rest of the analysis stack.
"""
import cv2
import numpy as np


class Grid:
    """Patch-centre grid: node (r, c) sits at (x0 + c*stride, y0 + r*stride)."""

    def __init__(self, x0, y0, stride, rows, cols):
        self.x0, self.y0, self.stride = float(x0), float(y0), float(stride)
        self.rows, self.cols = int(rows), int(cols)

    def nodes(self):
        """Node coordinates, each (rows, cols)."""
        return np.meshgrid(self.x0 + self.stride * np.arange(self.cols),
                           self.y0 + self.stride * np.arange(self.rows))


class FieldEvaluator:
    """Dense fields and warps for one image shape; caches the coordinate maps."""

    def __init__(self, shape, grid=None):
        h, w = shape
        self.shape = (h, w)
        self.X, self.Y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        self.grid = grid
        if grid is not None and grid.rows and grid.cols:
            self.gx = ((self.X - grid.x0) / grid.stride).astype(np.float32)
            self.gy = ((self.Y - grid.y0) / grid.stride).astype(np.float32)

    def local(self, L):
        """Bicubic interpolation of a (rows, cols, 2) node grid to every pixel.
        Beyond the outermost nodes the residual is held constant."""
        return [cv2.remap(np.ascontiguousarray(L[..., k], dtype=np.float32),
                          self.gx, self.gy, cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE) for k in (0, 1)]

    def dense(self, A, L=None):
        A = np.asarray(A, np.float64)
        fx = (A[0, 0] * self.X + A[0, 1] * self.Y + A[0, 2]).astype(np.float32)
        fy = (A[1, 0] * self.X + A[1, 1] * self.Y + A[1, 2]).astype(np.float32)
        if L is not None and self.grid is not None and self.grid.rows and self.grid.cols:
            lx, ly = self.local(L)
            fx += lx
            fy += ly
        return fx, fy

    def warp(self, img, fx, fy, interpolation=cv2.INTER_LINEAR, border=0.0):
        """output(x) = img(x + field(x)); outside the frame -> border."""
        return cv2.remap(np.ascontiguousarray(img, dtype=np.float32),
                         self.X + fx, self.Y + fy, interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def to_full_resolution(A_ws, L_ws, grid_ws, scale):
    """Convert a field measured at working scale to full-resolution pixels.

    Area downscaling by `scale` maps pixel centres as
    x_ws = scale * (x_full + 0.5) - 0.5, and displacements scale by 1/scale.
    The affine's linear part is dimensionless and unchanged; its offset and
    the local residual are rescaled, and the grid nodes are moved.
    """
    A_ws = np.asarray(A_ws, np.float64)
    if scale == 1.0:
        return A_ws.copy(), np.asarray(L_ws, np.float32).copy(), grid_ws
    k = 0.5 * scale - 0.5
    A = A_ws.copy()
    lin = A_ws[..., :2]
    A[..., 2] = (A_ws[..., 2] + (lin @ np.array([k, k]))) / scale
    L = (np.asarray(L_ws, np.float32) / scale).astype(np.float32)
    grid = Grid((grid_ws.x0 + 0.5) / scale - 0.5, (grid_ws.y0 + 0.5) / scale - 0.5,
                grid_ws.stride / scale, grid_ws.rows, grid_ws.cols)
    return A, L, grid


def save_fields(path, frame_index, used, A, L, grid, shape, model, extra=None):
    np.savez_compressed(
        path, frame_index=np.asarray(frame_index, np.int32),
        used=np.asarray(used, bool), affine=np.asarray(A, np.float64),
        local=np.asarray(L, np.float32),
        grid=np.array([grid.x0, grid.y0, grid.stride, grid.rows, grid.cols], np.float64),
        shape=np.array(shape, np.int32), model=np.array(model),
        **(extra or {}))


class FieldSet:
    """The saved fields of one non-rigid result, for playback and analysis.

        fs = FieldSet("stabilization/nonrigid/burst_x/fields.npz")
        out = fs.warp_frame(i, frame)     # None if frame i has no field
    """

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.frame_index = z["frame_index"]
            self.used = z["used"]
            self.affine = z["affine"]
            self.local = z["local"]
            g = z["grid"]
            self.grid = Grid(g[0], g[1], g[2], int(g[3]), int(g[4]))
            self.shape = tuple(int(v) for v in z["shape"])
            self.model = str(z["model"])
        self._row = {int(i): k for k, i in enumerate(self.frame_index)}
        self._evaluator = None

    def has(self, i):
        return int(i) in self._row

    def is_used(self, i):
        k = self._row.get(int(i))
        return k is not None and bool(self.used[k])

    def field(self, i):
        k = self._row[int(i)]
        if self._evaluator is None:
            self._evaluator = FieldEvaluator(self.shape, self.grid)
        return self._evaluator.dense(self.affine[k], self.local[k])

    def warp_frame(self, i, img, border=0.0):
        if not self.has(i) or tuple(img.shape[:2]) != self.shape:
            return None
        fx, fy = self.field(i)
        return self._evaluator.warp(img, fx, fy, border=border)
