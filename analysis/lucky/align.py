"""Aligned frames from an existing stabilization result (METHODS.md §L1).

Nothing is re-registered here: the frames are warped with exactly the
transforms the stabilization stage saved — the non-rigid fields when that
result exists and is usable, otherwise the translation shifts — so the fused
image lands on the same pixel grid as that result's mean_stabilized.tif and
consensus_mask.tif. Each original frame is resampled once (stabilize §11).
"""
import csv
import json
import os

import numpy as np

from stabilize.bursts import load_burst, read_frame
from stabilize.config import Params
from stabilize.fields import FieldSet
from stabilize.register import shift
from stabilize.vessels import glare_mask


class AlignedBurst:
    """One burst plus the saved transforms that stabilize it.

        ab = AlignedBurst(burst_path, stab_root)       # prefers non-rigid
        for i, img, seen in ab.frames():                # float32, bool
            ...
    """

    def __init__(self, burst_path, stab_root, method="auto"):
        self.burst = load_burst(burst_path)
        name = self.burst.name
        self.method = None
        candidates = ["nonrigid", "translation"] if method == "auto" else [method]
        for m in candidates:
            d = os.path.join(stab_root, m, name)
            rec = _read_json(os.path.join(d, "metrics.json"))
            if rec and rec.get("status") == "ok":
                self.method, self.result_dir, self.record = m, d, rec
                break
        if self.method is None:
            raise FileNotFoundError(f"no usable stabilization result for {name} under "
                                    f"{stab_root} (tried {', '.join(candidates)})")
        with open(os.path.join(self.result_dir, "transforms.csv"), encoding="utf-8",
                  newline="") as f:
            rows = list(csv.DictReader(f))
        self.used = np.array([int(r["index"]) for r in rows if r["registered"] == "1"])
        self.traj_px = np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows])
        self.fields = None
        if self.method == "nonrigid":
            self.fields = FieldSet(os.path.join(self.result_dir, "fields.npz"))
            self.used = np.array([i for i in self.used if self.fields.is_used(i)])
        self.shape = (self.burst.height, self.burst.width)
        self.params = Params()

    def warp(self, i, img, border=0.0):
        if self.fields is not None:
            return self.fields.warp_frame(i, img, border=border)
        return shift(img, self.traj_px[i], border=border)

    def frame(self, i):
        """Frame i on the stabilized grid, and the pixels it truly observed:
        inside the field of view and not hidden by glare (stabilize §4)."""
        raw = read_frame(self.burst.files[i]).astype(np.float32)
        obs = (~glare_mask(raw, self.burst.full_scale, self.params, 1.0)).astype(np.float32)
        img = self.warp(i, raw)
        seen = self.warp(i, obs) >= 0.999     # strictly inside: no border blend
        return img, seen

    def frames(self, subset=None):
        for i in (self.used if subset is None else subset):
            img, seen = self.frame(i)
            yield i, img, seen


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
