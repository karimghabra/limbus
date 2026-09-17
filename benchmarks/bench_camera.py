"""Camera-side measurements for the Basler acA1920-40um.

  1. grab rate       - sustained fps with no encoding running
  2. end-to-end mp4  - the app's VideoWriter at a set fps, written vs dropped
  3. end-to-end tiff - 16-bit TIFF per frame, written vs dropped

Also reports the Windows equivalents of the Linux USB checks: negotiated USB
link speed (SuperSpeed = USB 3, the lsusb "5000M" check), the camera's
device link throughput limit, and the count of incomplete/failed grabs (the
symptom the Linux usbfs_memory_mb limit produces).

usage: python bench_camera.py [fps] [seconds] [mode]
       mode: all (default) | grab | mp4 | tiff
"""
import os
import queue
import shutil
import sys
import threading
import time

import numpy as np
import tifffile
import os as _os
_os.environ.pop("GENICAM_GENTL64_PATH", None)  # see camera_recorder.py: system GenTL producer crashes pypylon
from pypylon import pylon

from bench_common import CpuSampler, REPO, probe


def frame_mb(cam):
    """Actual frame size â€” not the full-frame constant, which would badly
    overstate data rates for a cropped ROI."""
    return cam.Width.Value * cam.Height.Value * 2 / 1e6

sys.path.insert(0, os.path.abspath(REPO))
import camera_recorder as cr  # noqa: E402

FPS = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam_out")
os.makedirs(OUTDIR, exist_ok=True)


def node(cam, *names):
    for n in names:
        if hasattr(cam, n):
            try:
                g = getattr(cam, n)
                if g.IsReadable():
                    return g
            except Exception:
                pass
    return None


def open_camera(pixel_format="Mono12", fps=FPS,
                exposure_us=float(os.environ.get("EXPOSURE_US", 5000.0))):
    # NB: exposure caps the frame rate independently of ROI â€” a 5 ms exposure
    # holds the camera to 200 fps no matter how few rows are read
    tlf = pylon.TlFactory.GetInstance()
    devs = tlf.EnumerateDevices()
    if not devs:
        raise SystemExit("No Basler camera found.")
    cam = pylon.InstantCamera(tlf.CreateDevice(devs[0]))
    cam.Open()
    try:
        cam.UserSetSelector.Value = "Default"
        cam.UserSetLoad.Execute()
    except Exception:
        pass
    cam.MaxNumBuffer.Value = 32
    cam.Width.Value = cam.Width.Max
    # ROI_HEIGHT trims sensor rows: 12-bit readout time is ~25 us/row, so a
    # shorter ROI is the only way this camera exceeds ~32 fps at 12-bit.
    want_h = int(os.environ.get("ROI_HEIGHT", 0)) or cam.Height.Max
    cam.Height.Value = min(want_h, cam.Height.Max)
    fmts = list(cam.PixelFormat.Symbolics)
    if pixel_format not in fmts:
        raise SystemExit(f"{pixel_format} unsupported; available: {fmts}")
    cam.PixelFormat.Value = pixel_format
    for name in ("ExposureTime", "ExposureTimeAbs"):
        n = node(cam, name)
        if n is not None:
            n.Value = min(max(exposure_us, n.Min), n.Max)
            break
    try:
        cam.ExposureAuto.Value = "Off"
        cam.GainAuto.Value = "Off"
    except Exception:
        pass
    try:
        cam.AcquisitionFrameRateEnable.Value = True
        n = node(cam, "AcquisitionFrameRate", "AcquisitionFrameRateAbs")
        n.Value = min(max(fps, n.Min), n.Max)
    except Exception:
        pass
    return cam


def report_link(cam):
    bits = []
    for name in ("BslUSBSpeedMode", "DeviceLinkSpeedMode", "DeviceLinkSpeed"):
        n = node(cam, name)
        if n is not None:
            try:
                bits.append(f"{name}={n.Value}")
            except Exception:
                pass
    for name in ("DeviceLinkThroughputLimit", "DeviceLinkCurrentThroughput",
                 "DeviceLinkThroughputLimitMode", "ResultingFrameRate",
                 "ResultingFrameRateAbs", "PayloadSize", "ExposureTime"):
        n = node(cam, name)
        if n is not None:
            try:
                bits.append(f"{name}={n.Value}")
            except Exception:
                pass
    print("camera: " + ", ".join(str(b) for b in bits), flush=True)


def converter():
    c = pylon.ImageFormatConverter()
    c.OutputPixelFormat = pylon.PixelType_Mono16
    c.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return c


def grab_loop(cam, conv, secs, on_frame=None):
    """Returns (frames, incomplete, elapsed)."""
    n = bad = 0
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_Return)
        if not res or not res.IsValid():
            bad += 1
            continue
        if not res.GrabSucceeded():
            bad += 1
            res.Release()
            continue
        if on_frame is not None:
            on_frame(np.ascontiguousarray(conv.Convert(res).GetArray()))
        n += 1
        res.Release()
    dt = time.perf_counter() - t0
    cam.StopGrabbing()
    return n, bad, dt


def test_grab(pixel_format, request_fps):
    cam = open_camera(pixel_format, request_fps)
    report_link(cam)
    conv = converter()
    with CpuSampler() as cpu:
        n, bad, dt = grab_loop(cam, conv, SECS, on_frame=lambda f: None)
    print(f"[grab {pixel_format} @request {request_fps}] {n} frames in {dt:.1f}s "
          f"-> {n / dt:.2f} fps ({n / dt * frame_mb(cam):.0f} MB/s as 16-bit), "
          f"incomplete/failed {bad} | {cpu.summary()}", flush=True)
    cam.Close()
    return n / dt


def test_mp4(pixel_format, fps):
    cam = open_camera(pixel_format, fps)
    conv = converter()
    path = os.path.join(OUTDIR, f"cam_{pixel_format}_{fps:.0f}fps.mp4")
    w = cr.VideoWriter(path, cam.Width.Value, cam.Height.Value, fps, False,
                       cr.QUALITY_PRESETS[0][1], bit16=True)
    last_preview = [0.0]

    def on_frame(frame):
        w.write(frame)
        now = time.perf_counter()
        if now - last_preview[0] >= 1.0 / cr.PREVIEW_MAX_FPS:
            last_preview[0] = now
            pf = (frame >> 8).astype(np.uint8)
            cr.cv2.resize(pf, (1280, int(pf.shape[0] * 1280 / pf.shape[1])),
                          interpolation=cr.cv2.INTER_AREA)

    with CpuSampler(w.proc.pid) as cpu:
        n, bad, dt = grab_loop(cam, conv, SECS, on_frame)
        err = w.close()
    p = probe(path)
    print(f"[mp4 {pixel_format} @{fps} fps] grabbed {n} in {dt:.1f}s "
          f"({n / dt:.2f} fps), incomplete {bad} -> written {w.frames_written}, "
          f"dropped {w.frames_dropped} "
          f"({100 * w.frames_dropped / max(n, 1):.1f}%) | {cpu.summary()} | "
          f"ffprobe {p.get('pix_fmt')} bits={p.get('bits_per_raw_sample')} "
          f"frames={p.get('nb_read_frames')}"
          + (f" | ffmpeg: {err}" if err else ""), flush=True)
    cam.Close()


def test_tiff(pixel_format, fps, nthreads=2):
    cam = open_camera(pixel_format, fps)
    conv = converter()
    d = os.path.join(OUTDIR, "tiff_burst")
    os.makedirs(d, exist_ok=True)
    q = queue.Queue(maxsize=max(30, int(240e6 // (frame_mb(cam) * 1e6))))
    state = {"written": 0, "dropped": 0}
    lock = threading.Lock()

    def drain():
        while True:
            item = q.get()
            if item is None:
                return
            i, frame = item
            tifffile.imwrite(os.path.join(d, f"frame_{i:06d}.tif"), frame,
                             photometric="minisblack")
            with lock:
                state["written"] += 1

    threads = [threading.Thread(target=drain, daemon=True)
               for _ in range(nthreads)]
    for t in threads:
        t.start()
    counter = [0]

    def on_frame(frame):
        try:
            q.put_nowait((counter[0], frame))
        except queue.Full:
            state["dropped"] += 1
        counter[0] += 1

    with CpuSampler() as cpu:
        n, bad, dt = grab_loop(cam, conv, SECS, on_frame)
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()
    total = time.perf_counter()
    print(f"[tiff {pixel_format} @{fps} fps] grabbed {n} in {dt:.1f}s "
          f"({n / dt:.2f} fps), incomplete {bad} -> written {state['written']}, "
          f"dropped {state['dropped']} "
          f"({100 * state['dropped'] / max(n, 1):.1f}%), "
          f"{state['written'] * frame_mb(cam) / dt:.0f} MB/s | {cpu.summary()}",
          flush=True)
    sample = sorted(os.listdir(d))[:1]
    if sample:
        a = tifffile.imread(os.path.join(d, sample[0]))
        print(f"  TIFF check: dtype={a.dtype} shape={a.shape} "
              f"min={a.min()} max={a.max()} "
              f"(MSB-aligned 12-bit: low nibble should be 0 -> "
              f"{'yes' if int(a.max()) % 16 == 0 else 'no'})", flush=True)
    shutil.rmtree(d, ignore_errors=True)
    cam.Close()


if __name__ == "__main__":
    if MODE in ("all", "grab"):
        for fmt in ("Mono8", "Mono12"):
            test_grab(fmt, 1000.0)   # ask for more than it can do: find the cap
        test_grab("Mono12", FPS)
    if MODE in ("all", "mp4"):
        test_mp4("Mono12", FPS)
    if MODE in ("all", "tiff"):
        test_tiff("Mono12", FPS)
