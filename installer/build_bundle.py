"""Assemble the LIMBUS app as a self-contained folder, ready for the installer.

    python installer/build_bundle.py [--out build/bundle] [--version X.Y.Z]

Runs on Windows (x64) with any Python 3.8+, and needs git and internet access.
The folder it makes holds everything the app needs on a laptop with nothing
installed:

    runtime/   the app's own Python 3.12: the relocatable build python.org
               publishes as the NuGet package "python", with every package in
               installer/requirements.txt (CPU PyTorch; ffmpeg comes with
               imageio-ffmpeg) and the Visual C++ runtime DLLs beside
               python.exe, so nothing has to be installed system-wide
    app/       the app's files as committed in git (camera_recorder.py,
               analysis/, vesselmap/, assets/, e2e/), laid out as in the
               repository, plus build_info.json (version, commit, packages),
               which also tells the app it is installed

installer/limbus.iss then packs the folder into LIMBUS-Setup-<version>.exe,
reading <out>.ini (written here) for the bundle's longest path.
Downloads are cached in build/cache, so a rebuild only reinstalls packages.
"""
import argparse
import ctypes
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYTHON_VERSION = "3.12.10"
PYTHON_NUPKG = f"https://api.nuget.org/v3-flatcontainer/python/{PYTHON_VERSION}/python.{PYTHON_VERSION}.nupkg"
# SHA-256 of that file; its SHA-512 matches the one NuGet's catalog publishes
PYTHON_NUPKG_SHA256 = "0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8"

# what goes in app/: these paths, as committed (git ls-files), so local
# experiments and data under them never reach the installer
APP_PATHS = ["camera_recorder.py", "analysis", "vesselmap", "assets", "e2e",
             "VERSION", "README.md"]

# The Visual C++ runtime, deployed app-locally beside python.exe (which
# Microsoft allows for these redistributable files): torch, OpenCV and Qt
# need it, and a new laptop may not have the system-wide redistributable.
VC_RUNTIME = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
              "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
              "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll",
              "vccorlib140.dll", "vcomp140.dll"]

# not needed to run anything, and large: C/C++ headers and link libraries
# (torch alone carries hundreds of MB of them)
PRUNE_GLOBS = ["runtime/include", "runtime/libs", "runtime/Scripts",
               "runtime/Lib/site-packages/torch/include",
               "runtime/Lib/site-packages/torch/lib/*.lib",
               "runtime/Lib/site-packages/torch/share/cmake"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, digest):
    if not (os.path.exists(dest) and sha256(dest) == digest):
        log(f"downloading {url}")
        tmp = dest + ".part"
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    got = sha256(dest)
    if got != digest:
        sys.exit(f"{dest}: SHA-256 {got}, expected {digest}")
    return dest


def clean_env():
    """The build's own environment, kept from leaking into the runtime: no
    user site-packages, no PYTHONPATH/PYTHONHOME of the machine's Python."""
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("PYTHON", "PIP_", "VIRTUAL_ENV", "CONDA"))}
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def file_version(path):
    """A DLL's file version as a tuple, (0,) when it has none."""
    ver = ctypes.WinDLL("version.dll")
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return (0,)
    buf = ctypes.create_string_buffer(size)
    ver.GetFileVersionInfoW(path, 0, size, buf)
    ptr, n = ctypes.c_void_p(), ctypes.c_uint()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(n)):
        return (0,)
    ms, ls = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))[2:4]
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def vc_runtime_dirs():
    """Folders holding the VC++ redistributable DLLs, best first: a Visual
    Studio install's Redist folder (as on GitHub's Windows runners), then
    System32 (the redistributable installed system-wide)."""
    dirs = []
    vswhere = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
    if os.path.exists(vswhere):
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"],
                             capture_output=True, text=True).stdout.strip()
        for vs in out.splitlines():
            for kind in ("CRT", "OpenMP"):
                found = glob.glob(os.path.join(vs, "VC", "Redist", "MSVC", "*", "x64",
                                               f"Microsoft.VC*.{kind}"))
                dirs += sorted(found, reverse=True)[:1]
    dirs.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    return dirs


def deploy_vc_runtime(runtime):
    """Copy the newest of each VC++ runtime DLL beside python.exe."""
    deployed = {}
    for name in VC_RUNTIME:
        best = None
        for d in vc_runtime_dirs():
            src = os.path.join(d, name)
            if os.path.exists(src) and (best is None or file_version(src) > file_version(best)):
                best = src
        if best is None:
            sys.exit(f"Visual C++ runtime file {name} not found (install Visual Studio's C++ "
                     "tools or the VC++ redistributable on the build machine)")
        dest = os.path.join(runtime, name)
        if not os.path.exists(dest) or file_version(best) > file_version(dest):
            shutil.copy2(best, dest)
        deployed[name] = {"version": ".".join(map(str, file_version(dest))), "from": os.path.dirname(best)}
    return deployed


def build_runtime(out, cache):
    nupkg = fetch(PYTHON_NUPKG, os.path.join(cache, os.path.basename(PYTHON_NUPKG)),
                  PYTHON_NUPKG_SHA256)
    runtime = os.path.join(out, "runtime")
    log(f"unpacking Python {PYTHON_VERSION} into {runtime}")
    with zipfile.ZipFile(nupkg) as z:
        for member in z.namelist():
            if member.startswith("tools/") and not member.endswith("/"):
                dest = os.path.join(runtime, *member.split("/")[1:])
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as f:
                    shutil.copyfileobj(src, f)
    py = os.path.join(runtime, "python.exe")
    env = clean_env()
    run([py, "-m", "ensurepip", "--default-pip"], env=env)
    run([py, "-m", "pip", "install", "--no-warn-script-location",
         "-r", os.path.join(ROOT, "installer", "requirements.txt")], env=env)
    run([py, "-m", "pip", "check"], env=env)
    return runtime


def copy_app(out):
    app = os.path.join(out, "app")
    files = subprocess.run(["git", "ls-files", "-z", "--", *APP_PATHS], cwd=ROOT,
                           capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    files = [f for f in files if f]
    for rel in files:
        dest = os.path.join(app, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, *rel.split("/")), dest)
    log(f"copied {len(files)} app files into {app}")
    return app


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def prune(out):
    """Drop what nothing runs: headers and link libraries, and the packages'
    own test suites (thousands of files, and the deepest paths in the bundle
    apart from torch's; torch's one tests folder is left alone)."""
    doomed = [p for pattern in PRUNE_GLOBS for p in glob.glob(os.path.join(out, *pattern.split("/")))]
    site = os.path.join(out, "runtime", "Lib", "site-packages")
    for d, dirs, _ in os.walk(site):
        if os.path.relpath(d, site).split(os.sep)[0] == "torch":
            dirs[:] = []
            continue
        for name in [n for n in dirs if n == "tests"]:
            doomed.append(os.path.join(d, name))
            dirs.remove(name)
    freed = 0
    for path in doomed:
        if os.path.isdir(path):
            freed += tree_size(path)
            shutil.rmtree(path)
        else:
            freed += os.path.getsize(path)
            os.remove(path)
    log(f"pruned {freed / 1e6:.0f} MB of headers, link libraries and test suites")


def longest_path(out):
    """The longest file path inside the bundle, relative to the install
    folder: the installer refuses a folder so deep that this file would pass
    Windows' 260-character path limit (installer/limbus.iss)."""
    return max((os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs), key=len)


def tree_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)


def main():
    if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit("the bundle is a Windows x64 build: run this on 64-bit Windows")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bundle"))
    ap.add_argument("--version", help="defaults to the VERSION file")
    ap.add_argument("--app-only", action="store_true",
                    help="refresh only app/ in an existing bundle, keeping its runtime")
    args = ap.parse_args()
    version = args.version or open(os.path.join(ROOT, "VERSION"), encoding="utf-8").read().strip()
    out = os.path.abspath(args.out)
    cache = os.path.join(ROOT, "build", "cache")
    os.makedirs(cache, exist_ok=True)
    runtime = os.path.join(out, "runtime")
    t0 = time.time()
    if args.app_only:
        with open(os.path.join(out, "app", "build_info.json"), encoding="utf-8") as f:
            vc = json.load(f)["vc_runtime"]
        shutil.rmtree(os.path.join(out, "app"))
        app = copy_app(out)
    else:
        if os.path.exists(out):
            log(f"removing the previous {out}")
            shutil.rmtree(out)
        build_runtime(out, cache)
        vc = deploy_vc_runtime(runtime)
        app = copy_app(out)
        prune(out)

    py = os.path.join(runtime, "python.exe")
    env = clean_env()
    freeze = subprocess.run([py, "-m", "pip", "freeze", "--all"], env=env, capture_output=True,
                            text=True, check=True).stdout.split()
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", *APP_PATHS,
                     "installer"))
    info = {
        "version": version,
        "commit": commit + ("-dirty" if dirty else ""),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": PYTHON_VERSION,
        "packages": freeze,
        "vc_runtime": vc,
    }
    with open(os.path.join(app, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # compile ahead, so the first launch doesn't: the app checked against its
    # sources (an edited segmenter plugin is recompiled), the standard library
    # as usual (pip already compiled the packages)
    run([py, "-m", "compileall", "-q", "-j", "0", "--invalidation-mode", "checked-hash", app], env=env)
    if not args.app_only:
        # (not checked: the standard library ships a few deliberately broken
        # files, test data for its own tests)
        subprocess.run([py, "-m", "compileall", "-q", "-j", "0", os.path.join(runtime, "Lib")],
                       env=env, stdout=subprocess.DEVNULL)

    deepest = longest_path(out)
    with open(out + ".ini", "w", encoding="utf-8") as f:
        f.write(f"[bundle]\nversion={version}\nlongest_path={len(deepest)}\n")
    log(f"longest path inside: {len(deepest)} characters ({deepest})")
    size = tree_size(out)
    log(f"bundle {version} ({info['commit'][:12]}) ready in {out}: {size / 1e9:.2f} GB, "
        f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
