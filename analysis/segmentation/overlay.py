"""The overlay file: what a segmenter hands the Review tab.

overlay.json (format "limbus-overlay", version 1), in one folder per
segmenter and stabilization result. Coordinates are full-resolution pixels
of the stabilized frame — the frame of mean_stabilized.tif — with x to the
right, y down and pixel centres on whole numbers:

    {"format": "limbus-overlay", "version": 1,
     "shape": [H, W],
     "vessels":   [{"id": 7, "points": [[x, y], ...],        # >= 2 points
                    "radius": [r, ...] | null,               # per point, px
                    "group": 3 | null,                       # e.g. the vessel a segment belongs to
                    "props": {...}}],                        # anything else, JSON
     "junctions": [{"x": .., "y": .., "kind": "bifurcation", "vessels": [7, 9]}],
     "masks":     [{"name": "vessels", "file": "mask_vessels.png"}],  # 8-bit, H x W, > 0 inside
     "summary": "124 vessels ...",
     "segmenter": {"id", "label", "version", "sha256"},
     "source": {"result", "method", "burst", "mean_mtime_ns", "mean_size"},
     "made_utc": "...", "seconds": 12.3}

The Review tab draws vessels and junctions on every view (carried through
each raw frame's correction), and masks on the stabilized views.
"""
import json
import os
import re

import numpy as np

FORMAT = "limbus-overlay"
VERSION = 1
FILE = "overlay.json"


class Overlay:
    """Built by a segmenter's segment(): vessels as centrelines (with widths
    if it has them), junctions, and optional raster masks."""

    def __init__(self, shape):
        self.shape = (int(shape[0]), int(shape[1]))
        self.vessels = []
        self.junctions = []
        self.masks = []
        self.summary = None          # one line for the app; a default is made if None
        self._next_id = 1

    def add_vessel(self, points, radius=None, id=None, group=None, **props):
        """A centreline, (N, 2) points (x, y). radius: None, or one value per
        point (the lumen half-width, px). group: vessels with the same group
        share a colour. Anything else goes in props. Returns the vessel id."""
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        if len(pts) < 2 or not np.isfinite(pts).all():
            raise ValueError("a vessel needs at least 2 finite points")
        if radius is not None:
            radius = np.asarray(radius, np.float64).ravel()
            if radius.shape != (len(pts),) or not np.isfinite(radius).all():
                raise ValueError("radius needs one finite value per point")
        vid = self._next_id if id is None else int(id)
        self._next_id = max(self._next_id, vid + 1)
        self.vessels.append({"id": vid, "points": pts, "radius": radius,
                             "group": None if group is None else int(group),
                             "props": _jsonable(props)})
        return vid

    def add_junction(self, x, y, kind, vessels=()):
        """Where vessels meet. kind is free text; the app has markers for
        bifurcation, junction, crossing, crossing under, overlap, unresolved
        and endpoint, and a generic one for anything else."""
        self.junctions.append({"x": float(x), "y": float(y), "kind": str(kind),
                               "vessels": [int(v) for v in vessels]})

    def add_mask(self, name, mask):
        """A raster layer, H x W, true (or > 0) inside."""
        mask = np.asarray(mask)
        if mask.shape != self.shape:
            raise ValueError(f"mask {name!r} is {mask.shape}, the frame {self.shape}")
        self.masks.append((str(name), mask > 0))

    def default_summary(self):
        kinds = {}
        for j in self.junctions:
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
        text = f"{len(self.vessels)} vessels"
        if kinds:
            text += " · " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
        return text

    def write(self, folder, meta):
        """overlay.json (written last, via a temporary file, so a crash never
        leaves a half-written one) and a PNG per mask."""
        import cv2
        os.makedirs(folder, exist_ok=True)
        masks = []
        for name, m in self.masks:
            fname = "mask_" + (re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "layer") + ".png"
            if not cv2.imwrite(os.path.join(folder, fname), m.astype(np.uint8) * 255):
                raise OSError(f"could not write {fname}")
            masks.append({"name": name, "file": fname})
        doc = {
            "format": FORMAT, "version": VERSION, "shape": list(self.shape),
            "vessels": [{"id": v["id"], "points": np.round(v["points"], 2).tolist(),
                         "radius": None if v["radius"] is None else np.round(v["radius"], 2).tolist(),
                         "group": v["group"], "props": v["props"]} for v in self.vessels],
            "junctions": self.junctions,
            "masks": masks,
            "summary": self.summary or self.default_summary(),
            **meta,
        }
        tmp = os.path.join(folder, FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, os.path.join(folder, FILE))


def _jsonable(props):
    out = {}
    for k, v in props.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        json.dumps(v)                   # raises if it can't be stored
        out[str(k)] = v
    return out


def read(folder):
    """An overlay folder as a dict: vessels with (N, 2) float arrays, masks
    as (name, path). Raises ValueError for anything that isn't a readable
    version-1 overlay, with a reason fit to show."""
    path = os.path.join(folder, FILE)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    if doc.get("version") != VERSION:
        raise ValueError(f"{path} is overlay version {doc.get('version')}; this app reads {VERSION}")
    shape = tuple(int(v) for v in doc["shape"])
    vessels = []
    for v in doc.get("vessels", []):
        pts = np.asarray(v["points"], np.float64).reshape(-1, 2)
        r = v.get("radius")
        r = None if r is None else np.asarray(r, np.float64).ravel()
        if len(pts) < 2 or (r is not None and r.shape != (len(pts),)):
            raise ValueError(f"{path}: vessel {v.get('id')} is malformed")
        vessels.append({"id": int(v["id"]), "points": pts, "radius": r,
                        "group": v.get("group"), "props": v.get("props") or {}})
    masks = [(m["name"], os.path.join(folder, m["file"])) for m in doc.get("masks", [])]
    doc.update(shape=shape, vessels=vessels, masks=masks,
               junctions=[j for j in doc.get("junctions", [])
                          if np.isfinite([j.get("x", np.nan), j.get("y", np.nan)]).all()],
               made=os.path.getmtime(path))
    return doc


def staleness(doc, result_folder, segmenter=None):
    """Why an overlay may be out of date, as short sentences (none: current):
    made on an earlier stabilized mean, or by an earlier version of the
    segmenter's file."""
    notes = []
    src = doc.get("source") or {}
    try:
        st = os.stat(os.path.join(result_folder, "mean_stabilized.tif"))
        if (st.st_mtime_ns, st.st_size) != (src.get("mean_mtime_ns"), src.get("mean_size")):
            notes.append("found on an earlier stabilization")
    except OSError:
        pass
    if segmenter is not None:
        made_by = (doc.get("segmenter") or {}).get("sha256")
        if made_by and made_by != segmenter.source_hash():
            notes.append("made by an earlier version of this segmenter")
    return notes
