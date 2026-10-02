"""LIMBUS end to end, run inside an installed copy (or a checkout).

Drives the real app window. Only the camera is stood in for, by a replay
camera that streams a synthetic 12-bit vessel scene drifting by a known
amount, through the app's own grab thread, preview, writers and analysis:

  environment  the installed copy's own Python, ffmpeg and recordings folder
               are the ones used; every package the app and its analysis
               import loads, and every DLL they load comes from the
               installation or from Windows itself — none from another
               Python, ffmpeg or the system-wide VC++ redistributable
  video        each recording preset writes a file that decodes, at the bit
               depth the preset promises (FFV1 bit-exact)
  bursts       the Capture-burst panel and the Record button save TIFF bursts
               bit-exact, with manifest.json and frames.csv; a snapshot is a
               bit-exact 16-bit PNG
  review       bursts and videos open and play in the Review tab
  stabilize    translation and non-rigid, run from the Review tab as separate
               analysis processes, recover the known drift
  vessels      each built-in segmenter finds vessels and draws its overlay
  live view    with no GPU, "Stabilize the live view" says it needs one and
               switches itself off; with one, it locks on
  camera       if a Basler camera is plugged in: it connects through pylon and
               records a short video and burst (skipped otherwise)
  closing      the window closes, leaving no process behind

usage: python e2e/app_scenarios.py [--work DIR] [--keep] [--pause]
                                   [--report FILE] [--expect-installed]

The Start-menu shortcut "LIMBUS Self-Test" runs this with --pause. Exit code
0 when every check passes; a JSON report of each check goes to --report.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--work", help="working folder (default: a new temporary one)")
ap.add_argument("--keep", action="store_true", help="keep the working folder")
ap.add_argument("--pause", action="store_true", help="wait for Enter before exiting")
ap.add_argument("--report", help="write a JSON report of every check here")
ap.add_argument("--expect-installed", action="store_true",
                help="fail, rather than skip, the checks that only apply to an installed copy")
ap.add_argument("--timeout", type=float, default=60, help="minutes before giving up (default 60)")
ARGS = ap.parse_args()

WORK = os.path.abspath(ARGS.work or tempfile.mkdtemp(prefix="limbus_e2e_"))
RECORDINGS = os.path.join(WORK, "recordings")
SHOTS = os.path.join(WORK, "screenshots")
os.makedirs(RECORDINGS, exist_ok=True)
os.makedirs(SHOTS, exist_ok=True)
# before the app is imported: its default recordings folder comes from here
os.environ["LIMBUS_RECORDINGS"] = RECORDINGS

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import tifffile  # noqa: E402

sys.path.insert(0, APP)
import camera_recorder as cr  # noqa: E402
from PyQt5 import QtCore, QtWidgets  # noqa: E402

INSTALLED = cr.INSTALLED or ARGS.expect_installed
INSTALL_ROOT = os.path.dirname(APP) if cr.INSTALLED else None
W, H, MARGIN, N, FPS = 1024, 768, 48, 60, 30.0
VESSELMAP_FIELD = (240, 320)          # (h, w) of the scene vesselmap is run on
JOB_TIMEOUT_S = 20 * 60

results = []
current = {"step": "starting"}


# ---- reporting ---------------------------------------------------------------

def check(ok, what, detail=""):
    status = "ok" if ok else "FAIL"
    results.append({"step": current["step"], "check": what, "status": status, "detail": detail})
    print(f"  {status:5s} {what}" + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    return ok


def skip(what, why):
    results.append({"step": current["step"], "check": what, "status": "skip", "detail": why})
    print(f"  skip  {what}: {why}", flush=True)


def note(text):
    print(f"        {text}", flush=True)


def step(name):
    def deco(fn):
        def run(*a, **kw):
            current["step"] = name
            print(f"\n[{time.strftime('%H:%M:%S')}] {name}", flush=True)
            t0 = time.time()
            try:
                return fn(*a, **kw)
            except Exception as exc:
                check(False, f"{name} raised {type(exc).__name__}: {exc}", traceback.format_exc())
                traceback.print_exc()
            finally:
                note(f"({time.time() - t0:.0f}s)")
        return run
    return deco


def watchdog():
    time.sleep(ARGS.timeout * 60)
    print(f"\nTIMED OUT after {ARGS.timeout:.0f} min in step {current['step']!r}", flush=True)
    for p in psutil.Process().children(recursive=True):
        p.kill()
    os._exit(3)


# ---- the synthetic scene and the replay camera ----------------------------------

def vessel_network(rng, h, w):
    """Vessels from venules to capillaries, drawn as wandering polylines
    (as benchmarks/bench_common.py draws them, scaled to this frame size)."""
    m = np.zeros((h, w), np.float32)
    for count, (t0, t1), steps, seg in ((10, (4, 8), 40, 16),
                                         (60, (2, 3), 30, 9),
                                         (300, (1, 1), 20, 5)):
        for _ in range(count):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            ang = rng.uniform(0, 2 * np.pi)
            pts = []
            for _ in range(steps):
                pts.append((x, y))
                ang += rng.normal(0, 0.35)
                x += seg * np.cos(ang)
                y += seg * np.sin(ang)
            cv2.polylines(m, [np.array(pts, np.int32)], False, float(rng.uniform(0.3, 0.7)),
                          int(rng.integers(t0, t1 + 1)), cv2.LINE_AA)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def make_scene(seed=7):
    """N frames of 12-bit tissue, MSB-aligned in uint16 as pylon delivers
    Mono12, drifting by a known sub-pixel trajectory (content moves by
    +trajectory), with red-cell flicker inside the vessels and shot noise."""
    rng = np.random.default_rng(seed)
    hs, ws = H + 2 * MARGIN, W + 2 * MARGIN
    illum = cv2.resize(cv2.GaussianBlur(rng.random((hs // 16, ws // 16)).astype(np.float32), (0, 0), 3),
                       (ws, hs), interpolation=cv2.INTER_CUBIC)
    illum = 0.6 + 0.8 * (illum - illum.min()) / (np.ptp(illum) + 1e-6)
    tex = cv2.GaussianBlur(rng.standard_normal((hs, ws)).astype(np.float32), (0, 0), 1.5)
    tex /= tex.std()
    depth = cv2.resize(vessel_network(rng, hs // 2, ws // 2), (ws, hs), interpolation=cv2.INTER_LINEAR)
    traj = np.cumsum(rng.normal(0, 0.7, (N, 2)), axis=0)
    traj -= traj.mean(axis=0)
    frames = []
    for t in range(N):
        cells = cv2.GaussianBlur(rng.random((hs, ws)).astype(np.float32), (0, 0), 3)
        cells = 0.55 + 0.45 * (cells - cells.min()) / (np.ptp(cells) + 1e-6)
        scene = 2200.0 * illum * (1 + 0.06 * tex) * (1 - depth * cells)
        m = np.float32([[1, 0, traj[t, 0]], [0, 1, traj[t, 1]]])
        moved = cv2.warpAffine(scene, m, (ws, hs), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = moved[MARGIN:MARGIN + H, MARGIN:MARGIN + W]
        f = f + rng.normal(0, 1, f.shape).astype(np.float32) * np.sqrt(np.maximum(f, 0) / 4)
        frames.append(np.clip(np.rint(f), 0, 4095).astype(np.uint16) << 4)
    return np.stack(frames), traj


class ReplayCamera:
    """A camera backend (the interface camera_recorder.make_backend returns)
    that plays the scene in a loop at its frame rate."""

    def __init__(self, frames, fps):
        self.frames, self.fps = frames, fps
        self.i = 0
        self.t_next = None
        self.frame_meta = {}

    def open(self):
        h, w = self.frames.shape[1:]
        return {"model": "Replay", "serial": "e2e", "width": w, "height": h,
                "width_max": w, "height_max": h, "is_color": False, "bit16": True,
                "pixel_format": "Mono12", "pixel_formats": ["Mono12"], "bit_depth": 12,
                "can12": True, "exposure_us": 10000, "exposure_min": 20, "exposure_max": 1e6,
                "gain": 10, "gain_min": 0, "gain_max": 36, "fps": self.fps,
                "fps_max": self.fps, "auto_exposure": False}

    def read(self):
        now = time.perf_counter()
        if self.t_next is None:
            self.t_next = now
        if now < self.t_next:
            time.sleep(self.t_next - now)
        self.t_next += 1.0 / self.fps
        k = self.i % len(self.frames)
        self.i += 1
        self.frame_meta = {"replay_index": k}
        return self.frames[k]

    def set(self, key, value):
        pass

    def status(self):
        return {"exposure_us": 10000.0, "gain": 10.0, "resulting_fps": self.fps,
                "frame_rate": self.fps, "offset_x": 0, "offset_y": 0}

    def resulting_fps(self):
        return self.fps

    def close(self):
        pass


# ---- helpers ------------------------------------------------------------------

def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait_for(cond, seconds):
    end = time.time() + seconds
    while time.time() < end and not cond():
        pump(0.05)
    return cond()


def shot(widget, label):
    widget.grab().save(os.path.join(SHOTS, f"{label}.png"))


def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_ffmpeg(*args, timeout=300):
    return subprocess.run([cr.FFMPEG, "-hide_banner", *args], capture_output=True,
                          timeout=timeout, creationflags=cr.NO_WINDOW)


def decode_gray16(path):
    out = run_ffmpeg("-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray16le", "-")
    data = np.frombuffer(out.stdout, np.uint16)
    n = data.size // (W * H)
    return data[:n * W * H].reshape(n, H, W), out.stderr.decode(errors="replace").strip()


def stream_pix_fmt(path):
    err = run_ffmpeg("-i", path).stderr.decode(errors="replace")
    for line in err.splitlines():
        if "Video:" in line:
            return line.split("Video:", 1)[1].strip()
    return ""


def best_match(img, refs_small):
    """Correlation of a decoded frame with the closest scene frame."""
    small = cv2.resize(img.astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_AREA).ravel()
    small = (small - small.mean()) / (small.std() + 1e-6)
    scores = refs_small @ small / small.size
    return float(scores.max())


def loaded_modules(pid):
    """Paths of the DLLs, .pyd and .exe files mapped into a process."""
    paths = set()
    for m in psutil.Process(pid).memory_maps(grouped=True):
        if m.path.lower().endswith((".dll", ".pyd", ".exe")):
            paths.add(os.path.normcase(os.path.abspath(m.path)))
    return sorted(paths)


VC_NAMES = ("msvcp140", "vcruntime140", "concrt140", "vccorlib140", "vcomp140")
TOLERATED = [os.path.normcase(os.path.expandvars(p)) for p in (
    r"%ProgramData%\Microsoft\Windows Defender", r"%ProgramFiles%\Windows Defender",
    r"%ProgramFiles%\Basler", r"%ProgramFiles%\NVIDIA Corporation",
    r"%ProgramFiles(x86)%\NVIDIA Corporation", r"%ProgramFiles%\Common Files")]


def check_provenance(label, paths):
    """Everything a process loaded comes from the installation, or from
    Windows itself; the VC++ runtime from the installation's own copy."""
    root = os.path.normcase(INSTALL_ROOT)
    windows = os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows"))
    foreign, vc_system, tolerated = [], [], []
    for p in paths:
        name = os.path.basename(p)
        if p.startswith(root + os.sep):
            continue
        if p.startswith(windows + os.sep):
            if name.startswith(VC_NAMES):
                vc_system.append(p)
            continue
        (tolerated if any(p.startswith(t + os.sep) for t in TOLERATED) else foreign).append(p)
    ours = sum(p.startswith(root + os.sep) for p in paths)
    check(not foreign, f"{label}: every one of {len(paths)} modules is the installation's "
          f"({ours}) or Windows' own", "; ".join(foreign[:12]))
    check(not vc_system, f"{label}: the VC++ runtime is the installation's own copy",
          "; ".join(vc_system))
    if tolerated:
        note(f"{label}: also loaded (system components, tolerated): "
             + ", ".join(os.path.basename(p) for p in tolerated))


PROBE = r"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = [os.path.join(sys.argv[1], "analysis"), sys.argv[1]]
import matplotlib
matplotlib.use("Agg")
import numpy, scipy, scipy.ndimage, skimage, skimage.morphology, networkx, pandas, cv2, tifffile
import imageio_ffmpeg, torch
import stabilize, stabilize.pipeline, stabilize.live, segmentation, segmentation.overlay
import vesselmap
info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
        "threads": torch.get_num_threads(), "segmenters": [s.id for s in segmentation.list_segmenters()],
        "pid": os.getpid()}
print(json.dumps(info), flush=True)
sys.stdin.readline()      # stay alive until the parent has listed our modules
"""


NO_CONSOLE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["LOCALAPPDATA"] = sys.argv[2]          # keep the log under the work folder
result = {"stderr_was_none": sys.stderr is None}
import camera_recorder as cr
from PyQt5 import QtCore, QtWidgets
shown = []
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: shown.append(a[2]))
cr._log_without_console()
app = QtWidgets.QApplication([])

def fail():
    raise RuntimeError("e2e: a deliberate error")

for ms in (10, 20, 30):
    QtCore.QTimer.singleShot(ms, fail)
QtCore.QTimer.singleShot(500, app.quit)
app.exec_()
with open(cr.LOG_PATH, encoding="utf-8") as f:
    result.update(shown=shown, log=f.read(), log_path=cr.LOG_PATH)
with open(sys.argv[3], "w", encoding="utf-8") as f:
    json.dump(result, f)
"""


# ---- the scenarios ----------------------------------------------------------------

@step("environment")
def environment():
    note(f"app {APP}")
    note(f"python {sys.executable} ({sys.version.split()[0]})")
    note(f"ffmpeg {cr.FFMPEG}")
    note(f"recordings {cr.DEFAULT_OUTPUT_DIR}")
    check(cr.FFMPEG is not None and os.path.isfile(cr.FFMPEG), "ffmpeg found", str(cr.FFMPEG))
    out = run_ffmpeg("-encoders").stdout.decode(errors="replace")
    for enc in ("libx264", "libx265", "ffv1"):
        check(f" {enc} " in out, f"ffmpeg has the {enc} encoder")
    check(os.path.normcase(cr.DEFAULT_OUTPUT_DIR) == os.path.normcase(RECORDINGS),
          "LIMBUS_RECORDINGS chooses the recordings folder", cr.DEFAULT_OUTPUT_DIR)
    if not INSTALLED:
        skip("installation checks", "not an installed copy (no build_info.json)")
        return
    check(cr.INSTALLED, "build_info.json is present", APP)
    if not cr.INSTALLED:
        return
    note(f"LIMBUS {cr.BUILD_INFO.get('version')} ({cr.BUILD_INFO.get('commit', '')[:12]}), "
         f"built {cr.BUILD_INFO.get('built_utc')}")
    root = os.path.normcase(INSTALL_ROOT) + os.sep
    check(os.path.normcase(sys.executable).startswith(root), "runs on the installation's own Python",
          sys.executable)
    check(os.path.normcase(cr.FFMPEG).startswith(root), "records with the installation's own ffmpeg",
          cr.FFMPEG)
    check(sys.flags.no_user_site == 1, "user site-packages are ignored (-s)")
    saved = os.environ.pop("LIMBUS_RECORDINGS")
    try:
        default = cr.default_output_dir()
    finally:
        os.environ["LIMBUS_RECORDINGS"] = saved
    docs = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DocumentsLocation)
    check(os.path.normcase(default) == os.path.normcase(os.path.join(os.path.normpath(docs), "LIMBUS",
                                                                     "recordings")),
          "installed, recordings default to Documents\\LIMBUS\\recordings", default)


@step("analysis packages")
def analysis_packages():
    """Every package the analysis processes import loads, in a process of its
    own (as the app runs them: torch can't load into the GUI's process)."""
    proc = subprocess.Popen([sys.executable, *cr.CHILD_PY_FLAGS, "-c", PROBE, APP],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, creationflags=cr.NO_WINDOW)
    try:
        line = proc.stdout.readline()
        if not check(bool(line.strip()), "numpy, scipy, scikit-image, torch, OpenCV, pandas, networkx, "
                     "matplotlib, stabilize, segmentation and vesselmap all import",
                     proc.stderr.read() if not line.strip() else ""):
            return None
        info = json.loads(line)
        note(f"torch {info['torch']}, CUDA {'yes' if info['cuda'] else 'no'}, "
             f"{info['threads']} threads; segmenters {info['segmenters']}")
        check({"stabilization_mask", "vesselmap"} <= set(info["segmenters"]),
              "both built-in segmenters are listed")
        if INSTALLED and cr.INSTALLED:
            check_provenance("analysis process", loaded_modules(proc.pid))
        return info
    finally:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
        except OSError:
            pass
        proc.wait(timeout=60)


@step("errors without a console")
def no_console():
    """Run as the Start-menu shortcut runs it (pythonw: no console), an
    uncaught error is written to the log and shown once in a dialog, and the
    app carries on, rather than vanishing without a word."""
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    if not os.path.isfile(pythonw):
        skip("pythonw", f"no {pythonw}")
        return
    out = os.path.join(WORK, "no_console.json")
    appdata = os.path.join(WORK, "localappdata")
    proc = subprocess.run([pythonw, *cr.CHILD_PY_FLAGS, "-c", NO_CONSOLE, APP, appdata, out],
                          timeout=120)
    try:
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
    except (OSError, ValueError) as exc:
        check(False, f"pythonw run finished (exit code {proc.returncode})", str(exc))
        return
    check(res["stderr_was_none"], "pythonw has no console (sys.stderr is None)")
    check(len(res["shown"]) == 1 and "deliberate error" in res["shown"][0],
          f"the error is shown once, not three times ({len(res['shown'])} dialogs)")
    check(res["log"].count("RuntimeError: e2e: a deliberate error") == 3 and cr.APP_TITLE in res["log"],
          f"all three tracebacks and the startup line are in {res['log_path']}", res["log"][-800:])
    check(proc.returncode == 0, "the app carries on after the error")


@step("connect the replay camera")
def connect():
    cr.make_backend = lambda desc: ReplayCamera(frames, FPS)
    desc = {"type": "replay", "label": "Replay (e2e)"}
    win.cameras = [desc]
    win._connect_camera(desc)
    ok = wait_for(lambda: win.cam_thread is not None and win.cam_thread.latest_frame is not None
                  and win.record_btn.isEnabled(), 20)
    check(ok, f"streams {W}x{H} Mono12 at {FPS:.0f} fps")
    pump(1.5)
    check(win.preview.pixmap() is not None and not win.preview.pixmap().isNull(), "the preview shows it")
    shot(win, "camera_tab")
    return ok


@step("video recording")
def videos():
    refs = np.stack([cv2.resize((f >> 4).astype(np.float32), (W // 4, H // 4),
                                interpolation=cv2.INTER_AREA).ravel() for f in frames])
    refs = (refs - refs.mean(1, keepdims=True)) / (refs.std(1, keepdims=True) + 1e-6)
    exact = {digest(f): k for k, f in enumerate(frames)}
    win.record_mode.setCurrentIndex(win.record_mode.findData("video"))
    pump(0.2)
    expect = {"h264": ("yuv420p10le", 10), "hevc12": ("gray12le", 12), "ffv1": ("gray16le", 16)}
    for i, (label, quality) in enumerate(cr.QUALITY_PRESETS):
        codec = quality.get("codec", "h264")
        before = set(glob.glob(os.path.join(RECORDINGS, "recording_*")))
        win.quality_combo.setCurrentIndex(i)
        win._record_pressed()
        started = win.recording
        pump(2.5)
        win._record_released()                      # a hold: the clip ends on release
        done = wait_for(lambda: getattr(win, "_finishing", None) is None and not win.recording, 120)
        new = sorted(set(glob.glob(os.path.join(RECORDINGS, "recording_*"))) - before)
        if not check(started and done and len(new) == 1,
                     f"{label}: recorded {os.path.basename(new[0]) if new else 'nothing'}",
                     win.record_info.text()):
            continue
        path = new[0]
        decoded, err = decode_gray16(path)
        fmt = stream_pix_fmt(path)
        pix, bits = expect[codec]
        check(pix in fmt, f"{label}: {bits}-bit stream ({fmt.split(',')[1].strip() if ',' in fmt else fmt})")
        check(len(decoded) >= 0.6 * 2.5 * FPS and not err,
              f"{label}: {len(decoded)} frames decode cleanly", err[:300])
        if not len(decoded):
            continue
        if codec == "ffv1":
            hits = sum(digest(d) in exact for d in decoded)
            check(hits == len(decoded), f"{label}: lossless — {hits}/{len(decoded)} frames bit-exact")
        else:
            worst = min(best_match(d, refs) for d in decoded[::max(1, len(decoded) // 8)])
            check(worst > 0.97, f"{label}: frames match the scene (correlation {worst:.3f})")
    win.quality_combo.setCurrentIndex(0)
    mp4 = sorted(glob.glob(os.path.join(RECORDINGS, "*.mp4")))
    if mp4:
        win.tabs.setCurrentWidget(win.review)
        win.review.refresh()
        win.review.load(mp4[-1])
        pump(0.5)
        src = win.review.source
        check(src is not None and len(src) > 0 and win.review.view.pixmap() is not None,
              f"the Review tab plays the H.264 recording ({len(src) if src else 0} frames)")
        shot(win, "review_video")
        win.tabs.setCurrentIndex(0)


def burst_frames(folder):
    """The scene frame each TIFF of a burst holds (None if it matches none)."""
    exact = {digest(f >> 4): k for k, f in enumerate(frames)}
    tifs = sorted(glob.glob(os.path.join(folder, "frame_*.tif")))
    return [exact.get(digest(tifffile.imread(t))) for t in tifs]


def check_burst(label, folder, min_frames):
    ks = burst_frames(folder)
    check(len(ks) >= min_frames, f"{label}: {len(ks)} frames saved")
    check(ks and all(k is not None for k in ks), f"{label}: every frame bit-exact with the camera's",
          f"{sum(k is None for k in ks)} differ")
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    check(man.get("pixel_values", {}).get("bit_depth") == 12, f"{label}: manifest says 12-bit",
          json.dumps(man.get("pixel_values")))
    check(os.path.exists(os.path.join(folder, "frames.csv")), f"{label}: frames.csv saved")
    return ks


@step("bursts and a snapshot")
def bursts():
    win.burst_unit.setCurrentIndex(0)
    win.burst_frames.setValue(N)
    before = set(glob.glob(os.path.join(RECORDINGS, "burst_*")))
    win._burst_clicked()
    check(win.bursting, "Capture burst starts")
    wait_for(lambda: not win.bursting, 60)
    new = sorted(set(glob.glob(os.path.join(RECORDINGS, "burst_*"))) - before)
    if not check(len(new) == 1, f"Capture burst of {N} frames: {win.burst_info.text()}"):
        return None
    panel = new[0]
    ks = check_burst("Capture burst", panel, N)

    win.record_mode.setCurrentIndex(win.record_mode.findData("burst"))
    pump(0.2)
    before = set(glob.glob(os.path.join(RECORDINGS, "burst_*")))
    win._record_pressed()
    pump(1.5)
    win._record_released()                          # held: ends on release
    wait_for(lambda: not win.bursting and not win.recording, 60)
    new = sorted(set(glob.glob(os.path.join(RECORDINGS, "burst_*"))) - before)
    if check(len(new) == 1, f"Record-button burst (held 1.5 s): {win.record_info.text()}"):
        check_burst("Record-button burst", new[0], int(0.8 * 1.5 * FPS))

    win._snapshot()
    pngs = glob.glob(os.path.join(RECORDINGS, "snapshot_*.png"))
    if check(len(pngs) == 1, "Snapshot saves a PNG"):
        img = cv2.imread(pngs[0], cv2.IMREAD_UNCHANGED)
        exact = {digest(f) for f in frames}
        check(img is not None and img.dtype == np.uint16 and digest(img) in exact,
              "the snapshot is the camera's frame, bit-exact, in 16 bits")
    return panel, ks


@step("review")
def review(panel):
    tab = win.review
    win.tabs.setCurrentWidget(tab)
    tab.refresh(select=panel)
    pump(0.5)
    labels = [tab.listing.item(i).text() for i in range(tab.listing.count())]
    check(sum(t.startswith("burst_") for t in labels) == 2 and any(t.endswith(".mkv") for t in labels),
          f"the listing shows the bursts and recordings ({len(labels)} items)")
    tab.load(panel)
    pump(0.3)
    src = tab.source
    check(isinstance(src, cr.BurstSource) and len(src) == N, f"the burst opens: {len(src) if src else 0} frames")
    tab._step(5)
    pump(0.2)
    check(tab.counter.text() == f"6 / {N}" and tab.view.pixmap() is not None,
          f"stepping shows frame 6 ({tab.counter.text()})")
    tab._toggle_play()
    pump(1.0)
    playing = tab.timer.isActive()
    tab._toggle_play()
    check(playing and tab.index != 5, "it plays")
    shot(win, "review_burst")


def run_job(tab, button):
    button.click()
    pump(0.5)
    started = tab.proc is not None
    end = time.time() + JOB_TIMEOUT_S
    while tab.proc is not None and time.time() < end:
        pump(0.2)
    log = tab.job_log.toPlainText().splitlines()
    if tab.proc is not None:
        tab._cancel_job()
        log.append("timed out")
    return started, log


@step("stabilize")
def stabilize(panel, ks):
    tab = win.review
    truth = np.array([traj[k] for k in ks])
    for method in ("translation", "nonrigid"):
        tab.refresh(select=panel)
        tab.method_combo.setCurrentIndex(tab.method_combo.findData(method))
        pump(0.3)
        t0 = time.time()
        started, log = run_job(tab, tab.stab_btn)
        took = time.time() - t0
        note(f"{method}: " + " | ".join(log[-3:]))
        if not check(started and log and log[-1] == "finished" and tab.result is not None and tab.result.ok,
                     f"{method}: the analysis process finishes with a result ({took:.0f}s)",
                     " | ".join(log[-6:])):
            continue
        with open(os.path.join(tab.result.folder, "transforms.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        est = np.array([[float(r["dx_px"]), float(r["dy_px"])] for r in rows])
        used = np.array([r["registered"] == "1" for r in rows])
        err = (est[used] - est[used].mean(0)) - (truth[used] - truth[used].mean(0))
        rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
        check(used.sum() >= 0.9 * len(rows), f"{method}: used {used.sum()} of {len(rows)} frames")
        check(rms < 0.35, f"{method}: recovers the known drift ({np.ptp(truth[:, 0]):.1f} x "
              f"{np.ptp(truth[:, 1]):.1f} px) to {rms:.3f} px RMS")
        device = next((line for line in log if "device" in line.lower() or "cpu" in line.lower()), "")
        if device:
            note(f"{method}: {device.strip()}")
        tab.view_combo.setCurrentIndex(tab.view_combo.findData("mean"))
        pump(0.5)
        check(tab.view.pixmap() is not None and not tab.view.pixmap().isNull(),
              f"{method}: the stabilized mean is shown")
        shot(win, f"stabilized_mean_{method}")
    tab.method_combo.setCurrentIndex(tab.method_combo.findData("translation"))
    pump(0.3)


def vessels_line(tab):
    form = tab.frame_form
    for r in range(form.rowCount()):
        label = form.itemAt(r, QtWidgets.QFormLayout.LabelRole)
        field = form.itemAt(r, QtWidgets.QFormLayout.FieldRole)
        if label and field and label.widget().text() == "Vessels":
            return field.widget().text()
    return None


def segment(tab, burst, sid):
    """Find vessels in a burst's translation result with one segmenter."""
    win.tabs.setCurrentWidget(tab)
    tab.refresh(select=burst)
    tab.method_combo.setCurrentIndex(tab.method_combo.findData("translation"))
    pump(0.3)
    if not (tab.result is not None and tab.result.ok):
        started, log = run_job(tab, tab.stab_btn)
        if not check(tab.result is not None and tab.result.ok,
                     f"{sid}: a translation result to segment", " | ".join(log[-4:])):
            return
    tab.segmenter_combo.setCurrentIndex(tab.segmenter_combo.findData(sid))
    pump(0.3)
    t0 = time.time()
    started, log = run_job(tab, tab.vessel_btn)
    took = time.time() - t0
    found = tab.vessels is not None and len(tab.vessels.vessels) > 0
    check(started and log and log[-1] == "finished" and found,
          f"{sid}: {len(tab.vessels.vessels) if found else 0} vessels found in "
          f"{os.path.basename(burst)} ({took:.0f}s)",
          " | ".join(log[-6:]))
    if found:
        tab.view_combo.setCurrentIndex(tab.view_combo.findData("mean"))
        pump(0.5)
        line = vessels_line(tab)
        check(bool(line) and "drawn" in line, f"{sid}: the overlay is drawn ({line})")
        shot(win, f"vessels_{sid}")


def use_scene(scene, label):
    """Plug the replay camera in again, playing `scene`."""
    cr.make_backend = lambda desc: ReplayCamera(scene, FPS)
    desc = {"type": "replay", "label": label}
    win.cameras = [desc]
    win.tabs.setCurrentIndex(0)
    win._connect_camera(desc)
    return wait_for(lambda: win.cam_thread is not None and win.cam_thread.latest_frame is not None
                    and win.cam_thread.latest_frame.shape == scene.shape[1:]
                    and win.burst_btn.isEnabled(), 20)


@step("find vessels")
def find_vessels(panel):
    segment(win.review, panel, "stabilization_mask")
    # vesselmap fits every vessel as a spline: thorough, and slow on a CPU
    # (most of an hour for a full frame), so it gets a small field of view of
    # the same scene, recorded and stabilized through the app like the rest
    h, w = VESSELMAP_FIELD
    y0, x0 = (H - h) // 2, (W - w) // 2
    if not check(use_scene(frames[:, y0:y0 + h, x0:x0 + w], "Replay (e2e, small field)"),
                 f"the replay camera streams a {w}x{h} field"):
        return
    win.burst_unit.setCurrentIndex(0)
    win.burst_frames.setValue(30)
    before = set(glob.glob(os.path.join(RECORDINGS, "burst_*")))
    win._burst_clicked()
    wait_for(lambda: not win.bursting, 60)
    new = sorted(set(glob.glob(os.path.join(RECORDINGS, "burst_*"))) - before)
    if check(len(new) == 1, f"a 30-frame burst of the {w}x{h} field: {win.burst_info.text()}"):
        segment(win.review, new[0], "vesselmap")
    win.tabs.setCurrentIndex(0)
    check(use_scene(frames, "Replay (e2e)"), "back to the full field")


@step("live stabilization")
def live_view(cuda):
    win.tabs.setCurrentIndex(0)
    pump(0.5)
    win.live_check.setChecked(True)
    if cuda:
        ok = wait_for(lambda: win._live_locked, 60)
        check(ok, f"locks on with the GPU: {win.live_status.text().splitlines()[0]!r}")
        win.live_check.setChecked(False)
        pump(1.0)
    else:
        off = wait_for(lambda: not win.live_check.isChecked(), 90)
        check(off and "GPU" in win.live_status.text(),
              f"without a GPU it says so and switches off: {win.live_status.text()!r}")
    shot(win, "live_view")
    check(win.live is None, "no stabilizer left attached")


@step("Basler camera")
def real_camera():
    cr.make_backend = MAKE_BACKEND
    cams = [c for c in cr.list_cameras() if c["type"] == "pylon"]
    if not cams:
        skip("a Basler camera", "none plugged in" if cr.HAVE_PYLON else "pypylon unavailable")
        return
    cam = cams[0]
    note(f"{cam['label']} (serial {cam.get('serial')}), pylon {cr._pylon_version()}")
    win.cameras = cams
    win._connect_camera(cam)
    ok = wait_for(lambda: win.cam_thread is not None and win.cam_thread.latest_frame is not None
                  and win.record_btn.isEnabled(), 30)
    if not check(ok, "connects and streams through pylon", win.preview.text()):
        return
    frame = win.cam_thread.latest_frame
    note(f"{frame.shape[1]}x{frame.shape[0]} {frame.dtype}, {win.cam_thread.measured_fps:.1f} fps")
    pump(2.0)
    shot(win, "basler_camera")
    win.record_mode.setCurrentIndex(win.record_mode.findData("video"))
    win.quality_combo.setCurrentIndex(0)
    before = set(glob.glob(os.path.join(RECORDINGS, "recording_*")))
    win._record_pressed()
    pump(2.0)
    win._record_released()
    wait_for(lambda: getattr(win, "_finishing", None) is None and not win.recording, 120)
    new = sorted(set(glob.glob(os.path.join(RECORDINGS, "recording_*"))) - before)
    if check(len(new) == 1, f"records a 2 s video: {win.record_info.text()}"):
        cap = cv2.VideoCapture(new[0])
        check(cap.isOpened() and cap.read()[0], "the video decodes")
        cap.release()
    win.burst_unit.setCurrentIndex(0)
    win.burst_frames.setValue(30)
    before = set(glob.glob(os.path.join(RECORDINGS, "burst_*")))
    win._burst_clicked()
    wait_for(lambda: not win.bursting, 60)
    new = sorted(set(glob.glob(os.path.join(RECORDINGS, "burst_*"))) - before)
    if check(len(new) == 1, f"captures a 30-frame burst: {win.burst_info.text()}"):
        tifs = glob.glob(os.path.join(new[0], "frame_*.tif"))
        check(len(tifs) == 30 and os.path.exists(os.path.join(new[0], "manifest.json")),
              f"{len(tifs)} TIFFs and a manifest saved")


@step("closing")
def closing():
    if INSTALLED and cr.INSTALLED:
        check_provenance("app process", loaded_modules(os.getpid()))
    win.close()
    pump(1.0)
    check(not win.isVisible(), "the window closes")
    left = [p for p in psutil.Process().children(recursive=True) if p.is_running()
            and p.status() != psutil.STATUS_ZOMBIE]
    if left:
        time.sleep(5)
        left = [p for p in left if p.is_running()]
    check(not left, "no camera, encoder or analysis process is left running",
          ", ".join(f"{p.name()} {p.pid}" for p in left))


def main():
    global app, win, frames, traj, MAKE_BACKEND
    threading.Thread(target=watchdog, daemon=True).start()
    print(f"LIMBUS end-to-end check — work folder {WORK}", flush=True)
    t0 = time.time()
    app = QtWidgets.QApplication(sys.argv)
    MAKE_BACKEND = cr.make_backend
    current["step"] = "scene"
    frames, traj = make_scene()
    note(f"synthetic scene: {N} frames {W}x{H}, drift {np.ptp(traj[:, 0]):.1f} x {np.ptp(traj[:, 1]):.1f} px "
         f"({time.time() - t0:.0f}s)")

    environment()
    info = analysis_packages() or {}
    no_console()
    win = cr.MainWindow()
    win.resize(1500, 950)
    win.show()
    pump(1.0)
    check(win.output_dir == RECORDINGS, "the app records into the chosen folder", win.output_dir)
    if connect():
        videos()
        got = bursts()
        if got:
            panel, ks = got
            review(panel)
            stabilize(panel, ks)
            find_vessels(panel)
        live_view(info.get("cuda", False))
    real_camera()
    closing()

    failed = [r for r in results if r["status"] == "FAIL"]
    passed = sum(r["status"] == "ok" for r in results)
    skipped = sum(r["status"] == "skip" for r in results)
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped in {time.time() - t0:.0f}s")
    for r in failed:
        print(f"  FAIL  [{r['step']}] {r['check']}")
    print("RESULT:", "FAIL" if failed else "PASS", flush=True)
    if ARGS.report:
        with open(ARGS.report, "w", encoding="utf-8") as f:
            json.dump({"result": "FAIL" if failed else "PASS", "work": WORK,
                       "build": cr.BUILD_INFO and {k: cr.BUILD_INFO.get(k) for k in
                                                   ("version", "commit", "built_utc")},
                       "checks": results}, f, indent=2)
    if not ARGS.keep and not ARGS.work:
        shutil.rmtree(WORK, ignore_errors=True)
    else:
        print(f"work folder kept: {WORK} (screenshots in {SHOTS})")
    if ARGS.pause:
        input("\nPress Enter to close this window.")
    sys.stdout.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
