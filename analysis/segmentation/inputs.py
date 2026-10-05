"""What a segmenter's segment(inputs) is given: one stabilization result."""
import csv
import json
import os

import numpy as np


class Inputs:
    """One stabilization result of one burst, and where to write.

    result_dir   the stabilization result (metrics.json, transforms.csv,
                 mean_stabilized.tif, consensus_mask.tif, and fields.npz for
                 non-rigid)
    out_dir      this segmenter's folder for the result: write any files of
                 your own here (overlay.json itself is written for you)
    burst_dir    the raw burst, when known and present; otherwise None
    method       the stabilization method, 'translation' or 'nonrigid'
    metrics      the result's metrics.json
    shape        (H, W) of the stabilized frame
    log(text)    progress, shown in the Review tab's log as it happens
    """

    def __init__(self, result_dir, out_dir, burst_dir=None, log=print):
        self.result_dir = os.path.abspath(result_dir)
        self.out_dir = out_dir
        self.log = log
        with open(os.path.join(self.result_dir, "metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.method = self.metrics.get("method")
        burst = burst_dir or (self.metrics.get("burst") or {}).get("path")
        self.burst_dir = burst if burst and os.path.isdir(burst) else None
        self.mean_path = os.path.join(self.result_dir, "mean_stabilized.tif")
        self._mean = None

    @property
    def shape(self):
        return self.mean().shape

    def overlay(self):
        """An empty Overlay the size of the stabilized frame, to fill and return."""
        from .overlay import Overlay
        return Overlay(self.shape)

    def mean(self):
        """The stabilized mean: float32 (H, W), sensor DN, NaN where too few
        frames saw the tissue."""
        if self._mean is None:
            import tifffile
            self._mean = tifffile.imread(self.mean_path).astype(np.float32)
        return self._mean

    def valid(self):
        """Where the mean has data."""
        return np.isfinite(self.mean())

    def vessel_mask(self):
        """The vessel mask the stabilization registered on (the consensus of
        its per-frame envelope masks), bool (H, W)."""
        import tifffile
        return tifffile.imread(os.path.join(self.result_dir, "consensus_mask.tif")) > 0

    def frames(self, used_only=True):
        """Yield (index, frame) for the burst's frames as stabilized: each raw
        frame warped by its correction, float32 in sensor DN, 0 outside the
        field. used_only: just the frames the stabilization used. Needs the
        raw burst (burst_dir)."""
        if self.burst_dir is None:
            raise FileNotFoundError("the raw burst isn't available to this segmenter")
        from stabilize.bursts import load_burst, read_frame
        from stabilize.fields import FieldSet
        from stabilize.register import shift
        with open(os.path.join(self.result_dir, "transforms.csv"), encoding="utf-8",
                  newline="") as f:
            rows = {int(r["index"]): r for r in csv.DictReader(f)}
        fields_path = os.path.join(self.result_dir, "fields.npz")
        fields = FieldSet(fields_path) if os.path.exists(fields_path) else None
        burst = load_burst(self.burst_dir)
        for i, path in enumerate(burst.files):
            row = rows.get(i)
            if row is None or (used_only and row.get("registered") != "1"):
                continue
            img = read_frame(path).astype(np.float32)
            if fields is not None:
                out = fields.warp_frame(i, img)
                if out is None:
                    continue
            else:
                out = shift(img, (float(row["dx_px"]), float(row["dy_px"])))
            yield i, out
