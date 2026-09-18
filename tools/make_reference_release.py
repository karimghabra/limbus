#!/usr/bin/env python3
"""Package the LIMBUS reference TIFF bursts for a GitHub release.

For each of the five reference bursts this script

  * writes <out>/<burst>.zip holding the burst folder (member names are
    "<burst>/<file>") with every file stored byte-identical (deflate, level 6,
    ZIP64 when needed),
  * re-reads every member from the finished zip and checks its SHA-256 against
    the source file, so a zip is never published with silently altered content,
  * records the zip's size and SHA-256 plus the size and SHA-256 of every file,

then writes reference_data/manifest.json (consumed by
tools/fetch_reference_data.py) and <out>/RELEASE_NOTES.md, and prints a size
table.

The source recordings are only read, never modified.  Standard library only.

Usage:
    python tools/make_reference_release.py [--recordings DIR] [--out DIR] [--manifest FILE]
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RELEASE_TAG = "reference-data-v1"
RELEASE_TITLE = "Reference TIFF bursts v1"
BASE_URL = "https://github.com/karimghabra/limbus/releases/download/" + RELEASE_TAG

CONSENT = "Published with the informed consent of the imaged participant."
LICENCE = "Licence: TO BE DECIDED by the repository owner before public reuse."
CAVEAT = ('Parallel "doubled" vessel lines are present in individual raw frames and are '
          'not a stabilization artefact; whether they are anatomy or an optical ghost is '
          'unresolved.')

# The reference set.  "expect" holds what the burst is believed to be; it is
# checked against the burst's own manifest.json and any disagreement is
# reported as a WARNING (the values written to the release manifest always
# come from the recording itself, never from this table).
BURSTS = [
    {
        "name": "burst_2026-09-16_15-50-52",
        "description": ("Clean fixation with one saccade at ~3.9 s; two distinct eye poses "
                        "(reference case for single-mode / non-rigid stabilization)."),
        "expect": {"width": 1920, "height": 1200, "pixel_format": "Mono12", "fps": 32, "frames": 140},
    },
    {
        "name": "burst_2026-09-16_15-45-12",
        "description": ("Pronounced peripheral misalignment (rotation/magnification jitter); "
                        "the 'doubling' test case."),
        "expect": {"width": 1920, "height": 1200, "pixel_format": "Mono12", "fps": 32, "frames": 84},
    },
    {
        "name": "burst_2026-09-16_15-40-55",
        "description": ("Hard case: low usable fraction (~37%) and non-converging translation "
                        "registration."),
        "expect": {"width": 1920, "height": 1200, "pixel_format": "Mono12", "fps": 32, "frames": 63},
    },
    {
        "name": "burst_2026-09-16_15-31-37",
        "description": "Short-ROI geometry, clean.",
        "expect": {"width": 1920, "height": 500, "pixel_format": "Mono12", "fps": 74, "frames": 160},
    },
    {
        "name": "burst_2026-09-16_12-57-20",
        "description": "Thin strip geometry (little vertical capture range).",
        "expect": {"width": 1920, "height": 100, "pixel_format": "Mono12p", "fps": 149, "frames": 296},
    },
]

CHUNK = 4 * 1024 * 1024
MB = 1_000_000  # sizes are reported in decimal megabytes
FPS_TOLERANCE = 1.0  # expected fps values above are nominal integers


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def sha256_zip_member(zf, name):
    h = hashlib.sha256()
    size = 0
    with zf.open(name, "r") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def list_burst_files(burst_dir):
    """All regular files below burst_dir, sorted by POSIX relative path."""
    files = [p for p in burst_dir.rglob("*") if p.is_file()]
    return sorted(files, key=lambda p: p.relative_to(burst_dir).as_posix())


def read_burst_metadata(burst_dir, files, warnings):
    """Frames / geometry / pixel format / fps as recorded, with consistency checks."""
    name = burst_dir.name
    manifest_path = burst_dir / "manifest.json"
    meta = {}
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        warnings.append(f"{name}: no manifest.json in burst folder")

    acq = meta.get("acquisition_requested", {}) or {}
    cap = meta.get("capture", {}) or {}

    tif_count = sum(1 for p in files if p.parent == burst_dir
                    and p.name.startswith("frame_") and p.suffix.lower() == ".tif")

    frames_written = cap.get("frames_written")
    if frames_written is not None and frames_written != tif_count:
        warnings.append(f"{name}: manifest.json frames_written={frames_written} but "
                        f"{tif_count} frame_*.tif files are present")

    csv_path = burst_dir / "frames.csv"
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            csv_rows = sum(1 for _ in csv.DictReader(f))
        if csv_rows != tif_count:
            warnings.append(f"{name}: frames.csv has {csv_rows} rows but "
                            f"{tif_count} frame_*.tif files are present")
    else:
        warnings.append(f"{name}: no frames.csv in burst folder")

    fps = cap.get("effective_fps")
    if fps is None:
        fps = (meta.get("camera_state_at_start", {}) or {}).get("resulting_fps")
    if fps is not None:
        fps = round(float(fps), 3)

    return {
        "frames": tif_count,
        "width": acq.get("width"),
        "height": acq.get("height"),
        "pixel_format": acq.get("pixel_format"),
        "fps": fps,
        "requested_fps": acq.get("requested_fps"),
    }


def check_expectations(entry, rec, warnings, notes):
    name = entry["name"]
    exp = entry["expect"]
    for key in ("frames", "width", "height", "pixel_format"):
        if rec.get(key) != exp[key]:
            warnings.append(f"{name}: expected {key}={exp[key]!r} but the recording says "
                            f"{rec.get(key)!r}")
    if rec.get("fps") is None or abs(rec["fps"] - exp["fps"]) > FPS_TOLERANCE:
        warnings.append(f"{name}: expected ~{exp['fps']} fps but the recording's effective "
                        f"fps is {rec.get('fps')!r}")
    req = rec.get("requested_fps")
    if req is not None and rec.get("fps") is not None and abs(req - rec["fps"]) > FPS_TOLERANCE:
        notes.append(f"{name}: requested {req} fps but the camera delivered {rec['fps']} fps "
                     f"(the release manifest records the delivered rate)")


def build_zip(burst_dir, files, zip_path):
    """Write zip_path atomically; return per-file records (path, bytes, sha256)."""
    name = burst_dir.name
    records = []
    for p in files:
        digest, size = sha256_file(p)
        arcname = name + "/" + p.relative_to(burst_dir).as_posix()
        records.append({"path": arcname, "bytes": size, "sha256": digest, "_src": p})

    tmp_path = zip_path.with_name(zip_path.name + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    total = sum(r["bytes"] for r in records)
    done = 0
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        for i, r in enumerate(records, 1):
            zf.write(r["_src"], arcname=r["path"])
            done += r["bytes"]
            if i == len(records) or i % 25 == 0:
                print(f"    zipped {i}/{len(records)} files, {done / MB:8.1f} / {total / MB:.1f} MB",
                      flush=True)

    # Independent read-back: every member must decompress to the exact source bytes,
    # and the zip must contain nothing else.
    with zipfile.ZipFile(tmp_path, "r") as zf:
        members = [zi.filename for zi in zf.infolist()]
        expected = [r["path"] for r in records]
        if sorted(members) != sorted(expected):
            raise RuntimeError(f"{zip_path.name}: member list does not match source files")
        for r in records:
            digest, size = sha256_zip_member(zf, r["path"])
            if digest != r["sha256"] or size != r["bytes"]:
                raise RuntimeError(f"{zip_path.name}: member {r['path']} read back as "
                                   f"{size} bytes sha256={digest}, source is {r['bytes']} bytes "
                                   f"sha256={r['sha256']}")
            # Source must not have changed while we were zipping it.
            if os.path.getsize(r["_src"]) != r["bytes"]:
                raise RuntimeError(f"source file changed during packaging: {r['_src']}")
    os.replace(tmp_path, zip_path)
    for r in records:
        del r["_src"]
    return records


def fmt_geometry(b):
    return f"{b['width']}x{b['height']} {b['pixel_format']} @ {b['fps']:.1f} fps"


def write_release_notes(path, manifest):
    lines = [
        f"# {RELEASE_TITLE}",
        "",
        f"Release tag: `{manifest['release_tag']}`",
        "",
        "Raw TIFF bursts from the LIMBUS conjunctival/limbal imaging camera (Basler acA1920-40um), "
        "published so that stabilization and analysis algorithms can be test-run on other computers "
        "against identical input data.",
        "",
        "| # | Burst | Description | Frames | Geometry | Zip size | SHA-256 (zip) |",
        "|---|-------|-------------|-------:|----------|---------:|---------------|",
    ]
    for i, b in enumerate(manifest["bursts"], 1):
        lines.append(f"| {i} | `{b['name']}` | {b['description']} | {b['frames']} | "
                     f"{fmt_geometry(b)} | {b['zip_bytes'] / MB:.1f} MB | `{b['zip_sha256']}` |")
    total_raw = sum(b["raw_bytes"] for b in manifest["bursts"])
    total_zip = sum(b["zip_bytes"] for b in manifest["bursts"])
    lines += [
        "",
        f"Total: {total_zip / MB:.1f} MB zipped, {total_raw / MB:.1f} MB extracted "
        "(MB = 10^6 bytes).",
        "",
        "Each zip contains one burst folder: `frame_NNNNNN.tif` (16-bit TIFF holding right-aligned "
        "12-bit sensor DN, 0-4095), `manifest.json` (camera and acquisition settings) and "
        "`frames.csv` (per-frame camera timestamp, exposure and gain). Every file is stored "
        "byte-identical to the original recording; per-file sizes and SHA-256 hashes are listed in "
        "`reference_data/manifest.json` in the repository.",
        "",
        "Fetch and verify with:",
        "",
        "```",
        "python tools/fetch_reference_data.py",
        "```",
        "",
        CONSENT,
        "",
        LICENCE,
        "",
        "Caveat: " + CAVEAT,
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recordings", type=Path, default=REPO_ROOT / "recordings",
                    help="folder holding the burst_* folders (default: <repo>/recordings)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT.parent / "release_assets",
                    help="where the zips and RELEASE_NOTES.md are written "
                         "(default: <repo parent>/release_assets)")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "reference_data" / "manifest.json",
                    help="release manifest to write (default: <repo>/reference_data/manifest.json)")
    args = ap.parse_args(argv)

    recordings = args.recordings.resolve()
    out_dir = args.out.resolve()
    manifest_path = args.manifest.resolve()

    missing = [b["name"] for b in BURSTS if not (recordings / b["name"]).is_dir()]
    if missing:
        print("ERROR: burst folder(s) not found under " + str(recordings) + ":", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 2
    if out_dir == recordings or recordings in out_dir.parents:
        print(f"ERROR: --out ({out_dir}) must not be inside --recordings ({recordings})",
              file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    warnings = []
    notes = []
    bursts_out = []
    for n, entry in enumerate(BURSTS, 1):
        burst_dir = recordings / entry["name"]
        files = list_burst_files(burst_dir)
        rec = read_burst_metadata(burst_dir, files, warnings)
        check_expectations(entry, rec, warnings, notes)

        zip_name = entry["name"] + ".zip"
        zip_path = out_dir / zip_name
        print(f"[{n}/{len(BURSTS)}] {entry['name']}: {len(files)} files -> {zip_path}", flush=True)
        records = build_zip(burst_dir, files, zip_path)
        zip_sha, zip_bytes = sha256_file(zip_path)
        print(f"    zip {zip_bytes / MB:.1f} MB  sha256 {zip_sha}", flush=True)

        bursts_out.append({
            "name": entry["name"],
            "description": entry["description"],
            "zip": zip_name,
            "zip_bytes": zip_bytes,
            "zip_sha256": zip_sha,
            "raw_bytes": sum(r["bytes"] for r in records),
            "frames": rec["frames"],
            "width": rec["width"],
            "height": rec["height"],
            "pixel_format": rec["pixel_format"],
            "fps": rec["fps"],
            "files": records,
        })

    manifest = {
        "release_tag": RELEASE_TAG,
        "base_url": BASE_URL,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "bursts": bursts_out,
    }
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    with open(tmp_manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    os.replace(tmp_manifest, manifest_path)

    notes_path = out_dir / "RELEASE_NOTES.md"
    write_release_notes(notes_path, manifest)

    print()
    print(f"{'burst':<28}{'raw MB':>10}{'zip MB':>10}{'ratio':>8}")
    for b in bursts_out:
        print(f"{b['name']:<28}{b['raw_bytes'] / MB:>10.1f}{b['zip_bytes'] / MB:>10.1f}"
              f"{b['zip_bytes'] / b['raw_bytes']:>8.3f}")
    total_raw = sum(b["raw_bytes"] for b in bursts_out)
    total_zip = sum(b["zip_bytes"] for b in bursts_out)
    print(f"{'TOTAL':<28}{total_raw / MB:>10.1f}{total_zip / MB:>10.1f}{total_zip / total_raw:>8.3f}")
    print("(MB = 10^6 bytes; ratio = zip / raw)")
    print()
    print(f"manifest:      {manifest_path}")
    print(f"release notes: {notes_path}")

    if notes:
        print()
        for m in notes:
            print("  NOTE: " + m)
    if warnings:
        print()
        print(f"{len(warnings)} WARNING(S) - recording metadata vs. expected description:")
        for w in warnings:
            print("  WARNING: " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
