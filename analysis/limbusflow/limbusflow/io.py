"""Loading the reference (mean) image and a raw camera burst.

A burst directory produced by the LIMBUS recorder contains
  frame_XXXXXX.tif   one 12-bit mono frame per file (uint16, right aligned)
  frames.csv         per-frame chunk data (camera timestamp, exposure, gain, ...)
  manifest.json      camera + acquisition settings
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tifffile


@dataclass
class Burst:
    path: str
    frames: np.ndarray            # (T, H, W) uint16, possibly a memmap
    meta: pd.DataFrame            # per-frame table from frames.csv
    manifest: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        """Frame timestamps in seconds from the camera clock (t[0] == 0)."""
        ns = self.meta["camera_timestamp_ns"].to_numpy(np.int64)
        return (ns - ns[0]) * 1e-9

    @property
    def fps(self) -> float:
        return 1.0 / np.median(np.diff(self.t))

    @property
    def exposure_s(self) -> np.ndarray:
        return self.meta["exposure_us"].to_numpy(float) * 1e-6

    @property
    def gain_db(self) -> np.ndarray:
        return self.meta["gain"].to_numpy(float)

    def __len__(self):
        return len(self.frames)


def load_reference(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (image float32 with NaNs filled by the mean, valid-pixel mask)."""
    ref = tifffile.imread(path).astype(np.float32)
    valid = np.isfinite(ref)
    ref = np.where(valid, ref, np.nanmean(ref)).astype(np.float32)
    return ref, valid


def load_burst(path: str, cache_dir: str | None = "cache") -> Burst:
    """Load every frame of a burst into one (T, H, W) uint16 array.

    Reading ~1100 separate TIFFs is slow-ish, so the stack is cached once as a
    .npy file and memory-mapped on later calls.
    """
    meta = pd.read_csv(os.path.join(path, "frames.csv"))
    man_path = os.path.join(path, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}

    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, os.path.basename(os.path.normpath(path)) + "_stack.npy")
    if cache and os.path.exists(cache):
        frames = np.load(cache, mmap_mode="r")
    else:
        files = sorted(glob.glob(os.path.join(path, "frame_*.tif")))
        first = tifffile.imread(files[0])
        frames = np.empty((len(files),) + first.shape, first.dtype)
        for i, f in enumerate(files):
            frames[i] = tifffile.imread(f)
        if cache:
            np.save(cache, frames)
            frames = np.load(cache, mmap_mode="r")
    return Burst(path=path, frames=frames, meta=meta, manifest=manifest)
