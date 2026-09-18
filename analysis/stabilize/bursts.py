"""Finding and reading bursts written by the LIMBUS camera recorder.

Read-only by design: nothing here ever writes into a burst folder.
"""
import csv
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import tifffile


@dataclass
class Burst:
    path: str
    name: str
    files: list
    manifest: dict
    bit_depth: int
    fps: float
    width: int
    height: int
    pixel_format: str
    offset_y: object
    times_s: np.ndarray = field(repr=False)

    @property
    def full_scale(self):
        return (1 << self.bit_depth) - 1

    def __len__(self):
        return len(self.files)


def discover(root):
    """All burst folders under root: directories named burst_* holding TIFFs."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if (os.path.basename(dirpath).startswith("burst_")
                and any(f.endswith(".tif") for f in filenames)):
            found.append(dirpath)
            dirnames[:] = []
    return sorted(found)


def read_frame(path):
    return tifffile.imread(path)


def _frame_times(path, files, fps):
    """Per-frame capture times in seconds. The camera's own clock is used when
    frames.csv has it, so dropped frames leave a real gap in time rather than
    being silently treated as evenly spaced; otherwise fall back to 1/fps."""
    stamps = {}
    try:
        with open(os.path.join(path, "frames.csv"), encoding="utf-8",
                  newline="") as f:
            for row in csv.DictReader(f):
                if row.get("camera_timestamp_ns") and row.get("filename"):
                    stamps[row["filename"]] = int(row["camera_timestamp_ns"])
    except (OSError, ValueError):
        pass
    names = [os.path.basename(p) for p in files]
    if stamps and all(n in stamps for n in names):
        t = np.array([stamps[n] for n in names], dtype=np.float64) / 1e9
        return t - t[0]
    return np.arange(len(files), dtype=np.float64) / max(fps, 1e-6)


def load_burst(path):
    files = sorted(glob.glob(os.path.join(path, "frame_*.tif")))
    manifest = {}
    try:
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        pass
    capture = manifest.get("capture") or {}
    requested = manifest.get("acquisition_requested") or {}
    values = manifest.get("pixel_values") or {}

    first = read_frame(files[0]) if files else np.zeros((1, 1), np.uint16)
    default_bits = 8 if first.dtype == np.uint8 else 12
    bit_depth = int(values.get("bit_depth") or default_bits)
    fps = float(capture.get("effective_fps") or 0) \
        or float(requested.get("requested_fps") or 0) or 30.0
    return Burst(
        path=path,
        name=os.path.basename(path.rstrip("\\/")),
        files=files,
        manifest=manifest,
        bit_depth=bit_depth,
        fps=fps,
        width=int(first.shape[1]),
        height=int(first.shape[0]),
        pixel_format=str(requested.get("pixel_format") or ""),
        offset_y=requested.get("offset_y"),
        times_s=_frame_times(path, files, fps),
    )


def has_image_data(burst, samples=5):
    """False for a burst whose frames are entirely zero — e.g. Mono8 bursts
    written before the camera recorder's bit-shift bug was fixed."""
    if not burst.files:
        return False
    picks = np.linspace(0, len(burst.files) - 1, min(samples, len(burst.files)))
    return any(int(read_frame(burst.files[int(i)]).max()) > 0 for i in picks)
