#!/usr/bin/env python3
"""Fetch and verify the LIMBUS reference TIFF bursts.

Reads reference_data/manifest.json (written by tools/make_reference_release.py)
and, for every burst in it:

  1. skips the burst if <dest>/<burst>/ already holds every listed file with the
     right size and SHA-256;
  2. otherwise obtains <burst>.zip - downloaded from the manifest's base_url, or
     taken from a local folder given with --source - and checks the zip's size
     and SHA-256 against the manifest (a mismatching zip is never extracted);
  3. extracts it into a temporary folder inside --dest, checks every extracted
     file's size and SHA-256 against the manifest, and only then moves the files
     into <dest>/<burst>/.

Exits 0 when every requested burst is present and verified, 1 if any burst
failed, 2 for usage or manifest errors.  Standard library only.

Usage:
    python tools/fetch_reference_data.py                    # download all bursts
    python tools/fetch_reference_data.py --source D:/zips   # use local zips instead
    python tools/fetch_reference_data.py --only burst_2026-09-16_12-57-20
"""

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNK = 4 * 1024 * 1024
MB = 1_000_000
USER_AGENT = "limbus-fetch-reference-data/1 (+python urllib)"


class FetchError(Exception):
    """A burst could not be fetched or verified; the message says why."""


class ManifestError(Exception):
    """The manifest is missing or malformed."""


# --------------------------------------------------------------------------- manifest

def _safe_member_path(path, burst_name):
    """True if `path` is a relative POSIX path inside the burst folder."""
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        return False
    p = PurePosixPath(path)
    if p.is_absolute() or any(part in ("", ".", "..") for part in p.parts):
        return False
    return len(p.parts) >= 2 and p.parts[0] == burst_name


def load_manifest(path):
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise ManifestError(f"cannot read manifest {path}: {e}")

    bursts = manifest.get("bursts") if isinstance(manifest, dict) else None
    if not isinstance(bursts, list) or not bursts:
        raise ManifestError(f"manifest {path} has no 'bursts' list")
    seen = set()
    for i, b in enumerate(bursts):
        where = f"manifest {path}, bursts[{i}]"
        for key, typ in (("name", str), ("zip", str), ("zip_bytes", int), ("zip_sha256", str),
                         ("raw_bytes", int), ("files", list)):
            if not isinstance(b.get(key), typ):
                raise ManifestError(f"{where}: missing or invalid '{key}'")
        name = b["name"]
        if name in seen or any(c in name for c in "/\\:") or name in (".", ".."):
            raise ManifestError(f"{where}: invalid or duplicate burst name {name!r}")
        seen.add(name)
        if any(c in b["zip"] for c in "/\\:") or not b["zip"].endswith(".zip"):
            raise ManifestError(f"{where}: invalid zip file name {b['zip']!r}")
        if not b["files"]:
            raise ManifestError(f"{where}: empty 'files' list")
        paths = set()
        for f in b["files"]:
            if not (isinstance(f, dict) and isinstance(f.get("bytes"), int)
                    and isinstance(f.get("sha256"), str) and len(f["sha256"]) == 64):
                raise ManifestError(f"{where}: malformed file entry {f!r}")
            if not _safe_member_path(f.get("path"), name):
                raise ManifestError(f"{where}: unsafe or foreign file path {f.get('path')!r}")
            if f["path"] in paths:
                raise ManifestError(f"{where}: duplicate file path {f['path']!r}")
            paths.add(f["path"])
    return manifest


# --------------------------------------------------------------------------- hashing

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


def verify_files(root, burst):
    """Check root/<path> for every manifest file.  Returns a list of problem strings."""
    problems = []
    for f in burst["files"]:
        p = root.joinpath(*PurePosixPath(f["path"]).parts)
        if not p.is_file():
            problems.append(f"missing: {f['path']}")
            continue
        size = p.stat().st_size
        if size != f["bytes"]:
            problems.append(f"size mismatch: {f['path']}: expected {f['bytes']} bytes, found {size}")
    if problems:  # no point hashing when files are missing or the wrong size
        return problems
    for f in burst["files"]:
        p = root.joinpath(*PurePosixPath(f["path"]).parts)
        try:
            digest, _ = sha256_file(p)
        except OSError as e:
            problems.append(f"unreadable: {f['path']}: {e}")
            continue
        if digest != f["sha256"]:
            problems.append(f"SHA-256 mismatch: {f['path']}: expected {f['sha256']}, got {digest}")
    return problems


def extra_files(dest, burst):
    listed = {f["path"] for f in burst["files"]}
    burst_dir = dest / burst["name"]
    return sorted(p.relative_to(dest).as_posix() for p in burst_dir.rglob("*")
                  if p.is_file() and p.relative_to(dest).as_posix() not in listed)


def summarize(problems, limit=5):
    lines = ["    " + p for p in problems[:limit]]
    if len(problems) > limit:
        lines.append(f"    ... and {len(problems) - limit} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------- zip handling

def check_zip(path, burst):
    """Raise FetchError unless `path` has the manifest's size and SHA-256."""
    size = path.stat().st_size
    if size != burst["zip_bytes"]:
        raise FetchError(
            f"zip size mismatch for {burst['name']}: {path} is {size} bytes, the manifest expects "
            f"{burst['zip_bytes']} bytes (truncated, corrupt, or from a different release). "
            f"Not extracted.")
    print(f"  verifying zip SHA-256 ({size / MB:.1f} MB) ...", flush=True)
    digest, _ = sha256_file(path)
    if digest != burst["zip_sha256"]:
        raise FetchError(
            f"zip SHA-256 mismatch for {burst['name']}: {path} has sha256 {digest}, the manifest "
            f"expects {burst['zip_sha256']} (corrupt, or from a different release). Not extracted.")
    print("  zip SHA-256 OK", flush=True)


def download(url, target, burst):
    """Stream `url` to `target`, hashing on the fly; raise FetchError on any problem."""
    expected = burst["zip_bytes"]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"  downloading {url}", flush=True)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        hint = ""
        if e.code == 404:
            hint = (" - the release asset does not exist at this URL; check that the release "
                    "has been published and that base_url in the manifest is correct")
        raise FetchError(f"download of {burst['zip']} failed: HTTP {e.code} {e.reason}{hint}")
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise FetchError(f"download of {burst['zip']} failed: cannot reach {url}: {reason} "
                         f"(check the network connection, or pass --source with local zips)")

    h = hashlib.sha256()
    n = 0
    tty = sys.stdout.isatty()
    next_report = 0.0
    t0 = time.monotonic()
    try:
        with resp, open(target, "wb") as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                n += len(block)
                if n > expected:
                    raise FetchError(f"download of {burst['zip']} is larger than the "
                                     f"{expected} bytes the manifest expects; aborted")
                frac = n / expected if expected else 1.0
                rate = n / MB / max(time.monotonic() - t0, 1e-6)
                msg = (f"  {n / MB:8.1f} / {expected / MB:.1f} MB ({100 * frac:5.1f}%)"
                       f"  {rate:6.1f} MB/s")
                if tty:
                    print("\r" + msg, end="", flush=True)
                elif frac >= next_report:  # one line per 10% when not on a terminal
                    print(msg, flush=True)
                    next_report = (int(frac * 10) + 1) / 10
        if tty:
            print()
    except FetchError:
        raise
    except (OSError, http.client.HTTPException) as e:
        if tty:
            print()
        raise FetchError(f"download of {burst['zip']} interrupted after {n / MB:.1f} MB: {e}")

    if n != expected:
        raise FetchError(f"download of {burst['zip']} is incomplete: got {n} bytes, the manifest "
                         f"expects {expected} bytes. Not extracted.")
    digest = h.hexdigest()
    if digest != burst["zip_sha256"]:
        raise FetchError(f"zip SHA-256 mismatch for downloaded {burst['zip']}: got {digest}, the "
                         f"manifest expects {burst['zip_sha256']}. Not extracted.")
    print("  zip size and SHA-256 OK", flush=True)


def extract_and_install(zip_path, burst, dest):
    name = burst["name"]
    expected = {f["path"] for f in burst["files"]}
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.extracting-", dir=dest))
    try:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                infos = [zi for zi in zf.infolist() if not zi.is_dir()]
                names = [zi.filename for zi in infos]
                unexpected = sorted(set(names) - expected)
                absent = sorted(expected - set(names))
                if unexpected or absent or len(names) != len(set(names)):
                    raise FetchError(
                        f"{zip_path.name} contents do not match the manifest: "
                        f"{len(unexpected)} unexpected member(s) {unexpected[:3]}, "
                        f"{len(absent)} missing member(s) {absent[:3]}. Not installed.")
                print(f"  extracting {len(infos)} files ...", flush=True)
                for zi in infos:
                    target = staging.joinpath(*PurePosixPath(zi.filename).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(zi, "r") as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
        except zipfile.BadZipFile as e:
            raise FetchError(f"{zip_path.name} could not be extracted: {e}. Not installed.")
        except OSError as e:
            raise FetchError(f"extracting {zip_path.name} into {staging} failed: {e} "
                             f"(disk full or permission problem?). Not installed.")

        print(f"  verifying {len(expected)} extracted files (size + SHA-256) ...", flush=True)
        problems = verify_files(staging, burst)
        if problems:
            raise FetchError(f"{len(problems)} extracted file(s) of {name} failed verification; "
                             f"nothing was installed:\n" + summarize(problems))

        for f in burst["files"]:
            rel = PurePosixPath(f["path"]).parts
            src = staging.joinpath(*rel)
            dst = dest.joinpath(*rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(src, dst)
            except OSError as e:
                raise FetchError(f"could not move verified file into place: {dst}: {e} "
                                 f"(is it open in another program?)")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # A rename does not change content; confirm every verified file landed with its size.
    problems = []
    for f in burst["files"]:
        p = dest.joinpath(*PurePosixPath(f["path"]).parts)
        if not p.is_file() or p.stat().st_size != f["bytes"]:
            problems.append(f"missing or wrong size after install: {f['path']}")
    if problems:
        raise FetchError(f"{name}: installed files do not match:\n" + summarize(problems))


# --------------------------------------------------------------------------- per burst

def ensure_free_space(dest, needed, name):
    free = shutil.disk_usage(dest).free
    if free < needed:
        raise FetchError(f"not enough free disk space in {dest} for {name}: need about "
                         f"{needed / MB:.0f} MB, {free / MB:.0f} MB free")


def process_burst(burst, dest, base_url, source, keep_zips):
    name = burst["name"]
    burst_dir = dest / name

    if burst_dir.is_dir():
        print(f"  present - verifying {len(burst['files'])} files (size + SHA-256) ...", flush=True)
        problems = verify_files(dest, burst)
        if not problems:
            extras = extra_files(dest, burst)
            if extras:
                print(f"  note: {len(extras)} file(s) not in the manifest are present and were "
                      f"left untouched, e.g. {extras[0]}")
            return "already present and verified - skipped"
        print(f"  existing copy failed verification ({len(problems)} problem(s)); fetching again:")
        print(summarize(problems), flush=True)

    # Remove staging folders left behind by an interrupted earlier run of this tool.
    for leftover in dest.glob(f".{name}.extracting-*"):
        if leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)

    downloaded = None
    if source is not None:
        zip_path = source / burst["zip"]
        if not zip_path.is_file():
            raise FetchError(f"{burst['zip']} not found in --source folder {source}")
        ensure_free_space(dest, burst["raw_bytes"] + 64 * MB, name)
        check_zip(zip_path, burst)
    else:
        if not base_url:
            raise FetchError("the manifest has no base_url; pass --source with a folder of zips")
        zip_path = dest / burst["zip"]
        reuse = False
        if zip_path.is_file():  # kept from an earlier --keep-zips run
            try:
                check_zip(zip_path, burst)
                reuse = True
            except FetchError as e:
                print(f"  kept zip is not usable ({e}); downloading again", flush=True)
        if not reuse:
            ensure_free_space(dest, burst["zip_bytes"] + burst["raw_bytes"] + 64 * MB, name)
            partial = dest / (burst["zip"] + ".partial")
            url = base_url.rstrip("/") + "/" + urllib.parse.quote(burst["zip"])
            try:
                download(url, partial, burst)
                os.replace(partial, zip_path)
            finally:
                if partial.exists():
                    partial.unlink()
            downloaded = zip_path

    try:
        extract_and_install(zip_path, burst, dest)
    finally:
        if downloaded is not None and not keep_zips and downloaded.exists():
            downloaded.unlink()
    if downloaded is not None and keep_zips:
        print(f"  kept {downloaded}")
    return "fetched and verified"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download (or copy from --source), verify and extract the LIMBUS reference "
                    "TIFF bursts listed in the reference-data manifest.")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "reference_data" / "manifest.json",
                    help="release manifest (default: <repo>/reference_data/manifest.json)")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT / "reference_data",
                    help="where burst folders are placed (default: <repo>/reference_data)")
    ap.add_argument("--source", type=Path, default=None,
                    help="local folder containing the <burst>.zip files; no download is made")
    ap.add_argument("--only", metavar="NAME", action="append", default=None,
                    help="fetch only this burst (may be given more than once)")
    ap.add_argument("--keep-zips", action="store_true",
                    help="keep downloaded zips in --dest (ignored with --source)")
    args = ap.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest.resolve())
    except ManifestError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    bursts = manifest["bursts"]
    if args.only:
        known = {b["name"] for b in bursts}
        unknown = [n for n in args.only if n not in known]
        if unknown:
            print(f"ERROR: --only names not in the manifest: {', '.join(unknown)}", file=sys.stderr)
            print("available bursts:\n  " + "\n  ".join(b["name"] for b in bursts), file=sys.stderr)
            return 2
        bursts = [b for b in bursts if b["name"] in args.only]

    source = None
    if args.source is not None:
        source = args.source.resolve()
        if not source.is_dir():
            print(f"ERROR: --source folder does not exist: {source}", file=sys.stderr)
            return 2

    dest = args.dest.resolve()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: cannot create --dest folder {dest}: {e}", file=sys.stderr)
        return 2

    print(f"manifest: {args.manifest.resolve()} (release {manifest.get('release_tag', '?')})")
    print(f"source:   {source if source else manifest.get('base_url')}")
    print(f"dest:     {dest}")

    results = []
    for i, b in enumerate(bursts, 1):
        print(f"\n[{i}/{len(bursts)}] {b['name']}: {len(b['files'])} files, "
              f"zip {b['zip_bytes'] / MB:.1f} MB, extracted {b['raw_bytes'] / MB:.1f} MB", flush=True)
        try:
            status = process_burst(b, dest, manifest.get("base_url"), source, args.keep_zips)
            print(f"  OK: {status}", flush=True)
            results.append((b["name"], True, status))
        except FetchError as e:
            sys.stdout.flush()
            print(f"  ERROR: {b['name']}: {e}", file=sys.stderr, flush=True)
            results.append((b["name"], False, str(e).splitlines()[0]))

    failed = [r for r in results if not r[1]]
    print("\nsummary:")
    for name, ok, status in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {name}: {status}")
    if failed:
        sys.stdout.flush()
        print(f"\n{len(failed)} of {len(results)} burst(s) FAILED.", file=sys.stderr)
        return 1
    print(f"\nall {len(results)} burst(s) present and verified in {dest}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
