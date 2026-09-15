#!/usr/bin/env python3
"""
Camera Video Recorder
A simple GUI for previewing and recording video from industrial cameras.

Supports:
  - Basler cameras via pylon/pypylon (e.g. ace acA1300-200um)
  - UVC/V4L2 cameras (e.g. The Imaging Source DMK 33UX250)

Preview + one-click recording to .mp4 (H.264 via ffmpeg).
"""

import os
import re
import sys
import glob
import time
import queue
import shutil
import threading
import subprocess
from datetime import datetime

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from pypylon import pylon
    HAVE_PYLON = True
except ImportError:
    HAVE_PYLON = False

try:
    import gi
    gi.require_version("Aravis", "0.8")
    from gi.repository import Aravis
    HAVE_ARAVIS = True
except (ImportError, ValueError):
    HAVE_ARAVIS = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(APP_DIR, "recordings")

PREVIEW_MAX_FPS = 30.0  # don't push preview frames to the GUI faster than this
PREVIEW_MAX_WIDTH = 1280  # downscale bigger frames before handing to the GUI
DEFAULT_FPS = 30.0


# Raw pixel data this Pi can push through the H.264 encoder per second.
# Measured across three configurations: 1928x1208 8-bit sustained 30 fps
# (70 MB/s), the same frame in 12-bit sustained 14.5 fps (68 MB/s), and
# 2448x2048 8-bit sustained ~17 fps (85 MB/s). Easy scenes do better and
# noisy ones worse, so this is deliberately a conservative estimate.
ENCODER_THROUGHPUT_MBS = 70.0


def encoder_limit_fps(width, height, bytes_per_px=1):
    """Frame rate above which recording starts dropping frames."""
    frame_mb = width * height * bytes_per_px / 1e6
    return ENCODER_THROUGHPUT_MBS / frame_mb if frame_mb else 1e6


def default_fps_for(width, height, bytes_per_px=1):
    """Default frame rate: the usual 30 fps, unless the encoder can't
    sustain it at this resolution and bit depth."""
    return min(DEFAULT_FPS, encoder_limit_fps(width, height, bytes_per_px))


# Recording quality presets. "maxrate" caps the bitrate — noisy high-gain
# footage is incompressible, so capped presets trade visible quality for
# predictable file sizes.
QUALITY_PRESETS = (
    ("High quality", {"crf": 20, "maxrate": None}),
    ("Balanced", {"crf": 23, "maxrate": "60M"}),
    ("Compact files", {"crf": 26, "maxrate": "20M"}),
)

MIN_FREE_BYTES = 500e6  # auto-stop recording below this much free disk

# Press-and-hold recording: releasing the Record button after at least this
# long ends the clip. A quicker tap latches instead, so the recording keeps
# running until the button is tapped again.
HOLD_TO_RECORD_SECS = 0.4

# /dev/video* devices with these in their driver name are Raspberry Pi
# internals (codecs, ISP blocks), not cameras
V4L2_IGNORE = ("bcm2835", "rpivid", "pispbe", "rp1-cfe", "hevc", "codec",
               "isp", "v4l2loopback")


# ----------------------------------------------------------------------------
# camera discovery
# ----------------------------------------------------------------------------

def list_cameras():
    """Return a list of camera descriptors: {'type', 'label', ...}."""
    cams = []
    if HAVE_PYLON:
        try:
            for d in pylon.TlFactory.GetInstance().EnumerateDevices():
                cams.append({
                    "type": "pylon",
                    "serial": d.GetSerialNumber(),
                    "label": f"{d.GetModelName()} (Basler)",
                })
        except Exception:
            pass
    if HAVE_ARAVIS:
        try:
            Aravis.update_device_list()
            for i in range(Aravis.get_n_devices()):
                vendor = Aravis.get_device_vendor(i) or ""
                # Basler cameras are handled by the pylon backend above
                if HAVE_PYLON and "basler" in vendor.lower():
                    continue
                cams.append({
                    "type": "aravis",
                    "id": Aravis.get_device_id(i),
                    "label": f"{Aravis.get_device_model(i)} (USB3 Vision)",
                })
        except Exception:
            pass
    for dev in sorted(glob.glob("/dev/video*"),
                      key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        name = v4l2_device_name(dev)
        if not name or any(x in name.lower() for x in V4L2_IGNORE):
            continue
        if not v4l2_list_formats(dev):
            continue  # no capture formats -> not a usable camera
        cams.append({
            "type": "v4l2",
            "dev": dev,
            "label": f"{name} ({dev})",
        })
    return cams


def v4l2_device_name(dev):
    node = os.path.basename(dev)
    try:
        with open(f"/sys/class/video4linux/{node}/name") as f:
            return f.read().strip()
    except OSError:
        return None


def _v4l2_ctl(dev, *args):
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", dev, *args],
            capture_output=True, text=True, timeout=10)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def v4l2_list_formats(dev):
    """Parse `v4l2-ctl --list-formats-ext` into
    {fourcc: {(w, h): [fps, ...]}}."""
    out = _v4l2_ctl(dev, "--list-formats-ext")
    formats = {}
    fourcc = size = None
    for line in out.splitlines():
        m = re.search(r"\[\d+\]: '(\S+)\s?'", line)
        if m:
            fourcc = m.group(1)
            formats[fourcc] = {}
            continue
        m = re.search(r"Size: \w+ (\d+)x(\d+)", line)
        if m and fourcc:
            size = (int(m.group(1)), int(m.group(2)))
            formats[fourcc][size] = []
            continue
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and fourcc and size:
            formats[fourcc][size].append(float(m.group(1)))
    return {f: s for f, s in formats.items() if s}


def v4l2_controls(dev):
    """Parse `v4l2-ctl -l` into {name: {'min':, 'max':, 'value':, ...}}."""
    controls = {}
    for line in _v4l2_ctl(dev, "-l").splitlines():
        m = re.match(r"\s*(\w+)\s+0x\w+\s+\((int|menu|bool)\)\s*:\s*(.*)",
                     line)
        if not m:
            continue
        name, ctype, rest = m.groups()
        info = {"type": ctype}
        for k, v in re.findall(r"(\w+)=(-?\d+)", rest):
            info[k] = int(v)
        controls[name] = info
    return controls


def _first_key(d, *names):
    for n in names:
        if n in d:
            return n
    return None


# ----------------------------------------------------------------------------
# video writer: pipes raw frames into ffmpeg, which encodes H.264 mp4
# ----------------------------------------------------------------------------

class VideoWriter:
    def __init__(self, path, width, height, fps, is_color, quality=None,
                 bit16=False):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        self.frames_dropped = 0
        self.t_first = None
        self.t_last = None
        quality = quality or QUALITY_PRESETS[0][1]
        if bit16:
            # 16-bit mono in, 10-bit H.264 out. x264 has no 12-bit support
            # here (a 12-bit request silently degrades to 10), so 10-bit is
            # the most of the camera's 12 bits a video file can carry.
            pix_fmt, out_fmt = "gray16le", "yuv420p10le"
        else:
            pix_fmt = "bgr24" if is_color else "gray"
            out_fmt = "yuv420p"
        frame_bytes = width * height * (3 if is_color else (2 if bit16 else 1))
        # NOTE: no "-movflags +faststart" — it rewrites the whole file when
        # encoding finishes, which takes minutes for GB-sized recordings on
        # an SD card (and a truncated rewrite corrupts the file)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
            # leave a CPU core free for the camera grab loop and GUI
            "-threads", "3",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(quality["crf"]),
        ]
        if quality["maxrate"]:
            cmd += ["-maxrate", quality["maxrate"], "-bufsize", "60M"]
        cmd += ["-pix_fmt", out_fmt, path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        # cap buffered frames by memory, not count (~240 MB)
        self.queue = queue.Queue(
            maxsize=max(30, int(240e6 // frame_bytes)))
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            try:
                self.proc.stdin.write(frame.tobytes())
                self.frames_written += 1
            except (BrokenPipeError, ValueError):
                break

    def write(self, frame):
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.frames_dropped += 1
            return
        now = time.monotonic()
        if self.t_first is None:
            self.t_first = now
        self.t_last = now

    def close(self):
        # draining buffered frames and finalizing a large file can take a
        # while — never kill ffmpeg early, that corrupts the recording
        self.queue.put(None)
        self.thread.join(timeout=600)
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        err = b""
        if self.proc.stderr:
            try:
                err = self.proc.stderr.read()
            except Exception:
                pass
        self._fix_frame_rate()
        return err.decode(errors="replace").strip()

    def _fix_frame_rate(self):
        """If frames arrived slower than the declared rate (encoder or
        exposure limited), re-stamp the file so it plays at true speed."""
        if self.frames_written < 8 or not self.t_first:
            return
        duration = self.t_last - self.t_first
        if duration <= 0.5:
            return
        actual = (self.frames_written - 1) / duration
        if abs(actual - self.fps) / self.fps <= 0.05:
            return
        raw = self.path + ".h264"
        tmp = self.path + ".tmp.mp4"
        try:
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-y", "-i", self.path, "-c", "copy",
                            "-f", "h264", raw], check=True, timeout=600)
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-y", "-r", f"{actual:.3f}", "-i", raw,
                            "-c", "copy", tmp],
                           check=True, timeout=600)
            os.replace(tmp, self.path)
        except Exception:
            pass  # keep the original file rather than fail the recording
        finally:
            for p in (raw, tmp):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ----------------------------------------------------------------------------
# Basler (pylon) backend
# ----------------------------------------------------------------------------

def _node(cam, *names):
    for n in names:
        if hasattr(cam, n):
            node = getattr(cam, n)
            try:
                if node.IsReadable() or node.IsWritable():
                    return node
            except Exception:
                continue
    return None


def clamp_to_node(node, value):
    try:
        return min(max(value, node.Min), node.Max)
    except Exception:
        return value


class PylonCamera:
    """Basler camera via pypylon."""

    def __init__(self, serial=None, want12=False):
        self.serial = serial
        self.want12 = want12
        self.cam = None
        self.converter = None
        self.is_color = False
        self.bit16 = False

    def _n(self, *names):
        return _node(self.cam, *names)

    def _set_auto_exposure_limit(self, fps):
        node = self._n("AutoExposureTimeUpperLimit",
                       "AutoExposureTimeUpperLimitAbs")
        if node is not None:
            node.Value = clamp_to_node(node, 0.95e6 / fps)

    def open(self):
        tlf = pylon.TlFactory.GetInstance()
        device = None
        for d in tlf.EnumerateDevices():
            if self.serial in (None, d.GetSerialNumber()):
                device = d
                break
        if device is None:
            raise RuntimeError("Basler camera not found.")
        cam = self.cam = pylon.InstantCamera(tlf.CreateDevice(device))
        cam.Open()
        cam.MaxNumBuffer.Value = 32

        # Start from factory defaults so leftover settings from other tools
        # (e.g. Pylon Viewer) don't surprise us.
        try:
            cam.UserSetSelector.Value = "Default"
            cam.UserSetLoad.Execute()
        except Exception:
            pass
        try:
            cam.Width.Value = cam.Width.Max
            cam.Height.Value = cam.Height.Max
        except Exception:
            pass

        fmts = list(cam.PixelFormat.Symbolics)
        order = ("Mono8", "BGR8", "RGB8", "BayerRG8", "BayerBG8",
                 "BayerGR8", "BayerGB8", "YCbCr422_8")
        if self.want12:
            # prefer unpacked high bit depth; packed (…p) formats need
            # extra unpacking work for no benefit here
            order = ("Mono16", "Mono12", "Mono10") + order
        for fmt in order:
            if fmt in fmts:
                cam.PixelFormat.Value = fmt
                break
        pixfmt = cam.PixelFormat.Value
        self.is_color = not pixfmt.startswith("Mono")
        self.bit16 = pixfmt in ("Mono16", "Mono12", "Mono10")

        exp = self._n("ExposureTime", "ExposureTimeAbs")
        if exp is not None:
            exp.Value = clamp_to_node(exp, 10000.0)  # 10 ms

        try:
            cam.AcquisitionFrameRateEnable.Value = True
        except Exception:
            pass
        fpsn = self._n("AcquisitionFrameRate", "AcquisitionFrameRateAbs")
        if fpsn is not None:
            fpsn.Value = clamp_to_node(fpsn, default_fps_for(
                cam.Width.Value, cam.Height.Value, 2 if self.bit16 else 1))

        # Auto brightness on by default: exposure (capped so it can't lower
        # the fps) plus gain once exposure maxes out.
        auto = False
        try:
            self._set_auto_exposure_limit(fpsn.Value if fpsn else DEFAULT_FPS)
            cam.ExposureAuto.Value = "Continuous"
            auto = True
        except Exception:
            pass
        try:
            cam.GainAuto.Value = "Continuous" if auto else "Off"
        except Exception:
            pass

        self.converter = pylon.ImageFormatConverter()
        if self.is_color:
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        elif self.bit16:
            self.converter.OutputPixelFormat = pylon.PixelType_Mono16
        else:
            self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        # MSB-aligned puts the camera's 12 bits in the top of each 16-bit
        # word, so the data spans the full 16-bit range that ffmpeg, ImageJ
        # and 16-bit PNG all expect.
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        dev = cam.GetDeviceInfo()
        gain = self._n("Gain", "GainRaw")
        cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        return {
            "model": dev.GetModelName(),
            "serial": dev.GetSerialNumber(),
            "width": cam.Width.Value,
            "height": cam.Height.Value,
            "is_color": self.is_color,
            "bit16": self.bit16,
            "pixel_format": pixfmt,
            "can12": any(f in fmts for f in ("Mono16", "Mono12", "Mono10")),
            "exposure_us": exp.Value if exp else 0,
            "exposure_min": exp.Min if exp else 0,
            "exposure_max": exp.Max if exp else 0,
            "gain": gain.Value if gain else 0,
            "gain_min": gain.Min if gain else 0,
            "gain_max": gain.Max if gain else 0,
            "fps": fpsn.Value if fpsn else 0,
            "fps_max": fpsn.Max if fpsn else 0,
            "auto_exposure": auto,
        }

    def read(self):
        try:
            res = self.cam.RetrieveResult(500, pylon.TimeoutHandling_Return)
        except pylon.RuntimeException as exc:
            raise RuntimeError(str(exc))
        if not res or not res.IsValid():
            return None
        if not res.GrabSucceeded():
            res.Release()
            return None
        frame = np.ascontiguousarray(self.converter.Convert(res).GetArray())
        res.Release()
        return frame

    def set(self, key, value):
        cam = self.cam
        if key == "exposure_us":
            node = self._n("ExposureTime", "ExposureTimeAbs")
            node.Value = clamp_to_node(node, float(value))
        elif key == "gain":
            try:
                cam.GainAuto.Value = "Off"
            except Exception:
                pass
            node = self._n("Gain", "GainRaw")
            node.Value = clamp_to_node(node, value)
        elif key == "fps":
            node = self._n("AcquisitionFrameRate", "AcquisitionFrameRateAbs")
            node.Value = clamp_to_node(node, float(value))
            self._set_auto_exposure_limit(node.Value)
        elif key == "auto_exposure":
            mode = "Continuous" if value else "Off"
            cam.ExposureAuto.Value = mode
            try:
                cam.GainAuto.Value = mode
            except Exception:
                pass

    def resulting_fps(self):
        node = self._n("ResultingFrameRate", "ResultingFrameRateAbs")
        try:
            return node.Value if node else 0.0
        except Exception:
            return 0.0

    def close(self):
        if self.cam is not None:
            try:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
                self.cam.Close()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# UVC/V4L2 backend (The Imaging Source DMK etc.)
# ----------------------------------------------------------------------------

class V4L2Camera:
    """UVC camera via V4L2/OpenCV. Exposure unit on UVC is 100 µs."""

    PREFERRED_FOURCC = ("GREY", "YUYV", "UYVY", "MJPG")

    def __init__(self, dev, label=""):
        self.dev = dev
        self.label = label or dev
        self.cap = None
        self.is_color = True
        self.width = self.height = 0
        self.fps = DEFAULT_FPS
        self.fps_choices = []
        self.fourcc = None
        self.controls = {}
        self.exp_name = None
        self.auto_name = None
        self.gain_name = None

    def _set_ctrl(self, name, value):
        if name:
            _v4l2_ctl(self.dev, "--set-ctrl", f"{name}={int(value)}")

    def _open_capture(self):
        cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open {self.dev}.\n"
                "Is another program using the camera?")
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.fourcc.ljust(4)))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc == "GREY":
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        return cap

    def _closest_fps(self, want):
        if not self.fps_choices:
            return want
        at_most = [f for f in self.fps_choices if f <= want + 0.01]
        return max(at_most) if at_most else min(self.fps_choices)

    def open(self):
        formats = v4l2_list_formats(self.dev)
        if not formats:
            raise RuntimeError(f"No video formats found on {self.dev}.")
        self.fourcc = _first_key(formats, *self.PREFERRED_FOURCC) \
            or next(iter(formats))
        sizes = formats[self.fourcc]
        self.width, self.height = max(sizes, key=lambda s: s[0] * s[1])
        self.fps_choices = sorted(sizes[(self.width, self.height)])
        self.fps = self._closest_fps(
            default_fps_for(self.width, self.height))

        self.controls = v4l2_controls(self.dev)
        self.exp_name = _first_key(self.controls, "exposure_time_absolute",
                                   "exposure_absolute")
        self.auto_name = _first_key(self.controls, "auto_exposure",
                                    "exposure_auto")
        self.gain_name = _first_key(self.controls, "gain")

        # auto exposure on by default (UVC: 3 = aperture priority, 1 = manual)
        auto = False
        if self.auto_name:
            self._set_ctrl(self.auto_name, 3)
            auto = True

        self.cap = self._open_capture()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from {self.dev}.")
        frame = self._normalize(frame)
        self.height, self.width = frame.shape[:2]
        self.is_color = frame.ndim == 3

        exp = self.controls.get(self.exp_name, {})
        gain = self.controls.get(self.gain_name, {})
        return {
            "model": self.label.split(" (")[0],
            "serial": self.dev,
            "width": self.width,
            "height": self.height,
            "is_color": self.is_color,
            "bit16": False,
            "can12": False,
            # UVC exposure unit is 100 us
            "exposure_us": exp.get("value", 0) * 100,
            "exposure_min": exp.get("min", 1) * 100,
            "exposure_max": exp.get("max", 10000) * 100,
            "gain": gain.get("value", 0),
            "gain_min": gain.get("min", 0),
            "gain_max": gain.get("max", 0),
            "fps": self.fps,
            "fps_max": max(self.fps_choices) if self.fps_choices else 60,
            "auto_exposure": auto,
        }

    def _normalize(self, frame):
        frame = np.squeeze(frame)
        if frame.ndim == 1:  # raw buffer; try to reshape to the frame size
            if frame.size == self.width * self.height:
                frame = frame.reshape(self.height, self.width)
            else:
                raise RuntimeError("Unexpected frame layout from camera.")
        return np.ascontiguousarray(frame)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            return None
        return self._normalize(frame)

    def set(self, key, value):
        if key == "exposure_us":
            self._set_ctrl(self.auto_name, 1)  # manual
            self._set_ctrl(self.exp_name, max(1, round(float(value) / 100)))
        elif key == "gain":
            self._set_ctrl(self.gain_name, value)
        elif key == "auto_exposure":
            self._set_ctrl(self.auto_name, 3 if value else 1)
        elif key == "fps":
            new_fps = self._closest_fps(float(value))
            if abs(new_fps - self.fps) > 0.01:
                # V4L2 needs the stream restarted to change the frame rate
                self.fps = new_fps
                self.cap.release()
                self.cap = self._open_capture()

    def resulting_fps(self):
        return 0.0  # not reported by V4L2; the app shows measured fps instead

    def close(self):
        if self.cap is not None:
            self.cap.release()


# ----------------------------------------------------------------------------
# GenICam/USB3 Vision backend via Aravis (The Imaging Source DZK etc.)
# ----------------------------------------------------------------------------

class AravisCamera:
    """GenICam camera via Aravis. Exposure is in µs, gain in dB.

    Used for cameras the other backends can't drive — e.g. The Imaging
    Source polarization cameras (DZK/DYK), whose pixel formats the Linux
    UVC driver rejects. Raw frames are plain 8-bit mono (the polarizer
    mosaic sits on the sensor itself)."""

    N_BUFFERS = 24

    def __init__(self, device_id, label="", want12=False):
        self.device_id = device_id
        self.label = label or device_id
        self.want12 = want12
        self.bit16 = False
        self.cam = None
        self.dev = None
        self.stream = None
        self.is_color = False
        self.width = self.height = 0        # full sensor size
        self.is_polarized = False
        # For polarization sensors, "smooth" averages each 2x2 polarizer
        # block into one pixel: true light intensity, no mosaic pattern in
        # players. Raw keeps the full mosaic for polarization analysis.
        self.smooth = False

    def _set_feature(self, name, value):
        try:
            if self.dev.is_feature_available(name):
                self.dev.set_string_feature_value(name, value)
                return True
        except Exception:
            pass
        return False

    def _set_auto_exposure_limit(self, fps):
        # ExposureAutoUpperLimitAuto is a boolean on TIS cameras; turn it
        # off so our explicit limit is honored
        try:
            self.dev.set_boolean_feature_value(
                "ExposureAutoUpperLimitAuto", False)
        except Exception:
            try:
                self.dev.set_string_feature_value(
                    "ExposureAutoUpperLimitAuto", "Off")
            except Exception:
                pass
        try:
            self.dev.set_float_feature_value(
                "ExposureAutoUpperLimit", 0.95e6 / fps)
        except Exception:
            pass

    def open(self):
        Aravis.update_device_list()
        cam = self.cam = Aravis.Camera.new(self.device_id)
        self.dev = cam.get_device()

        # Queue USB transfers asynchronously. The synchronous default waits
        # on one transfer at a time, which caps a 5 MP camera at ~35 fps and
        # corrupts frames under load; async reaches the full 75 fps.
        try:
            self.dev.set_usb_mode(Aravis.UvUsbMode.ASYNC)
        except Exception:
            pass

        # Pick an unpacked pixel format (packed "…p" / "Packed" variants
        # would need bit-unpacking for no benefit here)
        fmts = cam.dup_available_pixel_formats_as_strings()
        def unpacked(f):
            return "Packed" not in f and f[-1:] != "p"
        can16 = [f for f in fmts if "16" in f and unpacked(f)]
        chosen = None
        if self.want12 and can16:
            chosen = can16[0]
        else:
            chosen = next((f for f in fmts if "8" in f and unpacked(f)), None)
        if chosen:
            cam.set_pixel_format_from_string(chosen)
        pixfmt = cam.get_pixel_format_as_string()
        self.bit16 = "16" in pixfmt
        self.is_color = not ("Mono" in pixfmt or "Gray" in pixfmt
                             or pixfmt in ("Y8", "GREY"))
        if self.is_color:
            raise RuntimeError(
                f"Unsupported pixel format {pixfmt} — this backend "
                "currently handles mono cameras only.")
        self.is_polarized = "Polarized" in pixfmt
        self.smooth = self.smooth and self.is_polarized

        _, _, self.width, self.height = cam.get_region()

        out_w, out_h = self.output_size()
        fps_min, fps_max = cam.get_frame_rate_bounds()
        fps = min(max(default_fps_for(out_w, out_h,
                                      2 if self.bit16 else 1),
                      fps_min), fps_max)
        try:
            cam.set_frame_rate(fps)
        except Exception:
            pass

        # Auto brightness on by default, capped so it can't lower the fps
        self._set_auto_exposure_limit(fps)
        auto = self._set_feature("ExposureAuto", "Continuous")
        self._set_feature("GainAuto", "Continuous" if auto else "Off")

        exp_min, exp_max = cam.get_exposure_time_bounds()
        gain_min, gain_max = cam.get_gain_bounds()

        self.stream = cam.create_stream(None, None)
        if self.stream is None:
            raise RuntimeError(
                "Could not open a stream to the camera.\n"
                "Is another program using it? Is it on a USB 3 port?")
        payload = cam.get_payload()
        for _ in range(self.N_BUFFERS):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        cam.start_acquisition()

        out_w, out_h = self.output_size()
        return {
            "model": cam.get_model_name(),
            "serial": cam.get_device_serial_number(),
            "width": out_w,
            "height": out_h,
            "is_color": False,
            "bit16": self.bit16,
            "pixel_format": pixfmt,
            "can12": bool(can16),
            "polarized": self.is_polarized,
            "smooth": self.smooth,
            "exposure_us": cam.get_exposure_time(),
            "exposure_min": exp_min,
            "exposure_max": exp_max,
            "gain": cam.get_gain(),
            "gain_min": gain_min,
            "gain_max": gain_max,
            "fps": cam.get_frame_rate(),
            "fps_max": fps_max,
            "auto_exposure": auto,
        }

    def output_size(self):
        if self.smooth:
            return self.width // 2, self.height // 2
        return self.width, self.height

    def read(self):
        buf = self.stream.timeout_pop_buffer(500000)  # µs
        if buf is None:
            return None
        try:
            if buf.get_status() != Aravis.BufferStatus.SUCCESS:
                return None
            # get_data() already returns a private copy of the buffer, so
            # the array stays valid after the buffer is requeued
            frame = np.frombuffer(
                buf.get_data(),
                dtype=np.uint16 if self.bit16 else np.uint8).reshape(
                    self.height, self.width)
        finally:
            self.stream.push_buffer(buf)
        if self.smooth:
            # INTER_AREA at exactly half size averages each 2x2 block —
            # i.e. the mean of the four polarizer angles
            frame = cv2.resize(frame,
                               (self.width // 2, self.height // 2),
                               interpolation=cv2.INTER_AREA)
        return frame

    def set(self, key, value):
        cam = self.cam
        if key == "exposure_us":
            self._set_feature("ExposureAuto", "Off")
            cam.set_exposure_time(float(value))
        elif key == "gain":
            self._set_feature("GainAuto", "Off")
            lo, hi = cam.get_gain_bounds()
            cam.set_gain(min(max(float(value), lo), hi))
        elif key == "fps":
            lo, hi = cam.get_frame_rate_bounds()
            cam.set_frame_rate(min(max(float(value), lo), hi))
            self._set_auto_exposure_limit(cam.get_frame_rate())
        elif key == "auto_exposure":
            mode = "Continuous" if value else "Off"
            self._set_feature("ExposureAuto", mode)
            self._set_feature("GainAuto", mode)
        elif key == "pol_smooth":
            self.smooth = bool(value) and self.is_polarized

    def resulting_fps(self):
        # the camera reports its nominal rate even when a long exposure
        # slows it down, so account for exposure ourselves
        try:
            fps = self.cam.get_frame_rate()
            exp_us = self.cam.get_exposure_time()
            if exp_us > 0:
                fps = min(fps, 1e6 / exp_us)
            return fps
        except Exception:
            return 0.0

    def close(self):
        # tear down in order — dropping the refs out of order can crash
        # in GObject cleanup at interpreter exit
        if self.cam is not None:
            try:
                self.cam.stop_acquisition()
            except Exception:
                pass
        self.stream = None
        self.dev = None
        self.cam = None
        import gc
        gc.collect()


def make_backend(desc):
    want12 = desc.get("want12", False)
    if desc["type"] == "pylon":
        if not HAVE_PYLON:
            raise RuntimeError("pypylon is not installed.")
        return PylonCamera(desc.get("serial"), want12)
    if desc["type"] == "aravis":
        if not HAVE_ARAVIS:
            raise RuntimeError("Aravis is not installed.")
        return AravisCamera(desc["id"], desc.get("label", ""), want12)
    return V4L2Camera(desc["dev"], desc.get("label", ""))


# ----------------------------------------------------------------------------
# camera thread: grabs frames continuously, feeds preview + recorder
# ----------------------------------------------------------------------------

class CameraThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    connected = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    info = QtCore.pyqtSignal(str)

    def __init__(self, descriptor, parent=None):
        super().__init__(parent)
        self.descriptor = descriptor
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = {}          # settings requested by the GUI
        self.writer = None          # VideoWriter while recording, else None
        self.is_color = False
        self.width = 0
        self.height = 0
        self.measured_fps = 0.0
        self.resulting_fps = 0.0
        self.latest_frame = None

    # ---- called from the GUI thread ----------------------------------------
    def apply(self, **settings):
        with self._lock:
            self._pending.update(settings)

    def start_recording(self, path, fps, quality=None):
        # size the video from the frames actually being produced (frame
        # size depends on the polarization view mode)
        frame = self.latest_frame
        if frame is None:
            raise RuntimeError("No image from the camera yet.")
        h, w = frame.shape[:2]
        writer = VideoWriter(path, w, h, fps, frame.ndim == 3, quality,
                             bit16=frame.dtype == np.uint16)
        with self._lock:
            self.writer = writer

    def detach_writer(self):
        """Take the writer out of the grab loop without closing it."""
        with self._lock:
            writer, self.writer = self.writer, None
        return writer

    def stop_recording(self):
        writer = self.detach_writer()
        if writer:
            err = writer.close()
            if err:
                self.info.emit(f"ffmpeg: {err}")
            return writer.frames_written, writer.frames_dropped
        return 0, 0

    def shutdown(self):
        self._stop.set()
        self.wait(8000)

    def _apply_pending(self, backend):
        with self._lock:
            pending, self._pending = self._pending, {}
        for key, value in pending.items():
            try:
                backend.set(key, value)
            except Exception as exc:
                self.info.emit(f"Could not set {key}: {exc}")

    # ---- main loop ----------------------------------------------------------
    def run(self):
        backend = None
        try:
            backend = make_backend(self.descriptor)
            info = backend.open()
            self.width = info["width"]
            self.height = info["height"]
            self.is_color = info["is_color"]
            self.connected.emit(info)

            last_preview = 0.0
            fps_count = 0
            fps_t0 = time.monotonic()

            while not self._stop.is_set():
                self._apply_pending(backend)
                frame = backend.read()
                if frame is None:
                    continue

                self.latest_frame = frame
                with self._lock:
                    writer = self.writer
                if writer and frame.shape[0] == writer.height \
                        and frame.shape[1] == writer.width:
                    writer.write(frame)

                now = time.monotonic()
                fps_count += 1
                if now - fps_t0 >= 1.0:
                    self.measured_fps = fps_count / (now - fps_t0)
                    fps_count = 0
                    fps_t0 = now
                    self.resulting_fps = backend.resulting_fps()

                if now - last_preview >= 1.0 / PREVIEW_MAX_FPS:
                    last_preview = now
                    pf = frame
                    if pf.dtype == np.uint16:
                        # the screen is 8-bit; keep the top 8 bits
                        pf = (pf >> 8).astype(np.uint8)
                    if pf.shape[1] > PREVIEW_MAX_WIDTH:
                        scale = PREVIEW_MAX_WIDTH / pf.shape[1]
                        pf = cv2.resize(
                            pf, (PREVIEW_MAX_WIDTH,
                                 int(pf.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
                    self.frame_ready.emit(pf)

        except Exception as exc:
            self.error.emit(f"Camera error:\n{exc}")
        finally:
            self.stop_recording()
            if backend is not None:
                backend.close()


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Video Recorder")
        # fit comfortably on whatever screen we have (leave room for
        # window decorations and taskbar)
        avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1100, int(avail.width() * 0.94)),
                    min(660, int(avail.height() * 0.88)))

        self.cam_thread = None
        self.recording = False
        self.record_started = None
        self._finishing = None
        self._press_action = None
        self._press_time = 0.0
        # default to the camera's higher bit depth for dynamic range;
        # cameras without one fall back to 8-bit automatically
        self.want12 = True
        self.output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self._build_ui()
        self._refresh_cameras(auto_connect=True)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)

    # ---- UI construction ----------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        # keep group-box titles clear of their content with large fonts
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 1.2em; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 4px; }")
        layout = QtWidgets.QHBoxLayout(central)

        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # preview area
        self.preview = QtWidgets.QLabel("Looking for cameras…")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet(
            "background-color: #202020; color: #aaaaaa; font-size: 16px;")
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.preview, stretch=1)

        # right-hand control panel, scrollable so it never forces the
        # window taller than a small screen
        panel = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(panel)
        side.setContentsMargins(4, 0, 4, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll)
        self._panel = panel
        self._panel_scroll = scroll

        # camera picker
        cam_row = QtWidgets.QHBoxLayout()
        self.camera_combo = QtWidgets.QComboBox()
        # keep long camera names from forcing the panel wide (the dropdown
        # list still shows the full name)
        self.camera_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(14)
        self.camera_combo.activated.connect(self._camera_chosen)
        refresh = QtWidgets.QPushButton("⟳")
        refresh.setToolTip("Look for cameras again")
        refresh.clicked.connect(lambda: self._refresh_cameras())
        cam_row.addWidget(self.camera_combo, stretch=1)
        cam_row.addWidget(refresh)
        side.addLayout(cam_row)

        self.camera_label = QtWidgets.QLabel("No camera")
        self.camera_label.setWordWrap(True)
        self.camera_label.setStyleSheet("font-weight: bold;")
        side.addWidget(self.camera_label)

        # record button
        self.record_btn = QtWidgets.QPushButton("●  Record")
        self.record_btn.setMinimumHeight(52)
        self.record_btn.setStyleSheet(
            "QPushButton {font-size: 18px; font-weight: bold;"
            " background-color: #c62828; color: white; border-radius: 8px;}"
            "QPushButton:disabled {background-color: #666666;}")
        self.record_btn.setToolTip(
            "Hold down to record for as long as you hold.\n"
            "Or tap once to start, and tap again to stop.")
        self.record_btn.pressed.connect(self._record_pressed)
        self.record_btn.released.connect(self._record_released)
        self.record_btn.setEnabled(False)
        side.addWidget(self.record_btn)

        hint = QtWidgets.QLabel("Hold for a short clip · tap to start/stop")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777777; font-size: 11px;")
        side.addWidget(hint)

        self.record_info = QtWidgets.QLabel(" ")
        self.record_info.setAlignment(QtCore.Qt.AlignCenter)
        self.record_info.setWordWrap(True)
        side.addWidget(self.record_info)

        self.snapshot_btn = QtWidgets.QPushButton("\U0001f4f7  Snapshot")
        self.snapshot_btn.setMinimumHeight(36)
        self.snapshot_btn.clicked.connect(self._snapshot)
        self.snapshot_btn.setEnabled(False)
        side.addWidget(self.snapshot_btn)

        # settings
        settings = QtWidgets.QGroupBox("Camera settings")
        form = QtWidgets.QFormLayout(settings)
        side.addWidget(settings)

        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 1000.0)
        self.fps_spin.setValue(DEFAULT_FPS)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.valueChanged.connect(self._fps_changed)
        form.addRow("Frame rate", self.fps_spin)

        self.rate_warning = QtWidgets.QLabel("")
        self.rate_warning.setWordWrap(True)
        self.rate_warning.setStyleSheet("color: #b35000; font-size: 11px;")
        self.rate_warning.setVisible(False)
        form.addRow("", self.rate_warning)

        self.auto_exp = QtWidgets.QCheckBox("Auto brightness")
        self.auto_exp.toggled.connect(self._auto_exposure_toggled)
        form.addRow(self.auto_exp)

        self.exp_spin = QtWidgets.QDoubleSpinBox()
        self.exp_spin.setRange(0.01, 10000.0)
        self.exp_spin.setValue(10.0)
        self.exp_spin.setDecimals(2)
        self.exp_spin.setSuffix(" ms")
        self.exp_spin.valueChanged.connect(
            lambda v: self._apply(exposure_us=v * 1000.0))
        form.addRow("Exposure", self.exp_spin)

        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_label = QtWidgets.QLabel("0")
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.gain_slider)
        gain_row.addWidget(self.gain_label)
        self.gain_slider.valueChanged.connect(self._gain_changed)
        form.addRow("Gain", gain_row)

        self.depth_combo = QtWidgets.QComboBox()
        self.depth_combo.addItems(["8-bit", "12-bit"])
        self.depth_combo.setToolTip(
            "12-bit captures 4096 grey levels instead of 256 — far more "
            "dynamic range for measurement.\nStills keep all 12 bits; video "
            "keeps 10 of them, and runs at a lower frame rate.")
        self.depth_combo.activated.connect(self._depth_changed)
        form.addRow("Bit depth", self.depth_combo)
        self._depth_row_label = form.labelForField(self.depth_combo)

        self.pol_combo = QtWidgets.QComboBox()
        self.pol_combo.addItems(["Smooth video", "Raw mosaic"])
        self.pol_combo.setToolTip(
            "Smooth video: averages the sensor's four polarizer pixels — "
            "clean, normal-looking footage.\nRaw mosaic: full-resolution "
            "polarization data for analysis (looks grainy/pixelated in "
            "video players).")
        self.pol_combo.activated.connect(self._pol_changed)
        form.addRow("Polarization", self.pol_combo)
        self._pol_row_label = form.labelForField(self.pol_combo)

        self.quality_combo = QtWidgets.QComboBox()
        for label, _ in QUALITY_PRESETS:
            self.quality_combo.addItem(label)
        self.quality_combo.setToolTip(
            "High quality: best image, biggest files (can reach ~1 GB/min "
            "on noisy scenes).\nBalanced / Compact: smaller files by "
            "limiting the bitrate, which can soften detail.")
        form.addRow("Quality", self.quality_combo)

        # output folder
        out_box = QtWidgets.QGroupBox("Save to")
        out_layout = QtWidgets.QHBoxLayout(out_box)
        self.out_edit = QtWidgets.QLineEdit(self.output_dir)
        self.out_edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("…")
        browse.clicked.connect(self._choose_folder)
        open_btn = QtWidgets.QPushButton("Open")
        open_btn.clicked.connect(self._open_folder)
        out_layout.addWidget(self.out_edit)
        out_layout.addWidget(browse)
        out_layout.addWidget(open_btn)
        side.addWidget(out_box)

        side.addStretch(1)

        self.status = QtWidgets.QLabel(" ")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #555555;")
        side.addWidget(self.status)

        self._gain_scale = 1.0  # slider ticks -> gain units
        self._gain_min = 0.0

        # size the panel to what its content actually needs with the
        # system's real font metrics, then fix the scroll area to match
        width = min(420, max(240, panel.sizeHint().width()))
        scroll.setFixedWidth(width + 20)  # room for the scrollbar

    # ---- camera hookup ------------------------------------------------------
    def _refresh_cameras(self, auto_connect=True):
        self.cameras = list_cameras()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for c in self.cameras:
            self.camera_combo.addItem(c["label"])
        self.camera_combo.blockSignals(False)
        if not self.cameras:
            self.preview.setText(
                "No camera found.\n\nPlug in a camera and press ⟳.")
            self.camera_label.setText("No camera")
            return
        if auto_connect:
            self.camera_combo.setCurrentIndex(0)
            self._connect_camera(self.cameras[0])

    def _camera_chosen(self, index):
        if 0 <= index < len(self.cameras):
            self._connect_camera(self.cameras[index])

    def _connect_camera(self, descriptor):
        if self.recording:
            self._toggle_record()
        if self.cam_thread:
            self.cam_thread.shutdown()
        self.preview.setText(f"Connecting to {descriptor['label']}…")
        self.record_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        descriptor = dict(descriptor, want12=self.want12)
        self._last_descriptor = descriptor
        self.cam_thread = CameraThread(descriptor, self)
        self.cam_thread.frame_ready.connect(self._show_frame)
        self.cam_thread.connected.connect(self._on_connected)
        self.cam_thread.error.connect(self._on_error)
        self.cam_thread.info.connect(self._on_info)
        self.cam_thread.start()

    def _apply(self, **kw):
        if self.cam_thread:
            self.cam_thread.apply(**kw)

    def _on_connected(self, d):
        self._cam_info = d
        color = "color" if d["is_color"] else "mono"
        self.camera_label.setText(
            f"{d['model']}  ({d['serial']})\n"
            f"{d['width']}×{d['height']} {color} · "
            f"{d.get('pixel_format', '8-bit')}")
        self.record_btn.setEnabled(True)
        self.snapshot_btn.setEnabled(True)

        can12 = d.get("can12", False)
        self.depth_combo.setVisible(can12)
        self._depth_row_label.setVisible(can12)
        self.depth_combo.blockSignals(True)
        self.depth_combo.setCurrentIndex(1 if d.get("bit16") else 0)
        self.depth_combo.blockSignals(False)

        polarized = d.get("polarized", False)
        self.pol_combo.setVisible(polarized)
        self._pol_row_label.setVisible(polarized)
        if polarized:
            self.pol_combo.blockSignals(True)
            self.pol_combo.setCurrentIndex(0 if d.get("smooth") else 1)
            self.pol_combo.blockSignals(False)

        for w in (self.fps_spin, self.exp_spin, self.gain_slider):
            w.blockSignals(True)
        if d["fps_max"]:
            self.fps_spin.setMaximum(d["fps_max"])
        if d["fps"]:
            self.fps_spin.setValue(d["fps"])
        if d["exposure_max"]:
            self.exp_spin.setRange(max(d["exposure_min"], 1) / 1000.0,
                                   d["exposure_max"] / 1000.0)
        self.exp_spin.setValue(d["exposure_us"] / 1000.0)
        span = (d["gain_max"] - d["gain_min"]) or 1
        self._gain_scale = span / 100.0
        self._gain_min = d["gain_min"]
        self.gain_slider.setValue(
            int(round((d["gain"] - d["gain_min"]) / self._gain_scale)))
        self.gain_label.setText(f"{d['gain']:.0f}")
        gain_available = d["gain_max"] > d["gain_min"]
        self.gain_slider.setEnabled(gain_available)
        for w in (self.fps_spin, self.exp_spin, self.gain_slider):
            w.blockSignals(False)

        self.auto_exp.blockSignals(True)
        self.auto_exp.setChecked(d["auto_exposure"])
        self.auto_exp.blockSignals(False)
        self.exp_spin.setEnabled(not d["auto_exposure"])
        self._update_rate_warning()

    def _on_error(self, msg):
        self.preview.setText(msg)
        self.record_btn.setEnabled(False)
        self.snapshot_btn.setEnabled(False)
        if self.recording:
            self._toggle_record()
        retry = QtWidgets.QMessageBox(self)
        retry.setWindowTitle("Camera problem")
        retry.setText(msg)
        retry.setStandardButtons(
            QtWidgets.QMessageBox.Retry | QtWidgets.QMessageBox.Close)
        if retry.exec_() == QtWidgets.QMessageBox.Retry:
            self._refresh_cameras()

    def _on_info(self, msg):
        self.status.setText(msg)

    # ---- preview ------------------------------------------------------------
    def _show_frame(self, frame):
        if frame.ndim == 2:
            h, w = frame.shape
            qimg = QtGui.QImage(frame.data, w, h, w,
                                QtGui.QImage.Format_Grayscale8)
        else:
            h, w, _ = frame.shape
            qimg = QtGui.QImage(frame.data, w, h, 3 * w,
                                QtGui.QImage.Format_BGR888)
        pix = QtGui.QPixmap.fromImage(qimg.copy())
        self.preview.setPixmap(pix.scaled(
            self.preview.size(), QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))

    # ---- recording ----------------------------------------------------------
    def _record_pressed(self):
        """Start recording the moment the button goes down, so a hold
        captures everything from the press onwards."""
        if self.recording:
            self._press_action = "stop"   # tapping while recording ends it
            return
        self._press_action = "started"
        self._press_time = time.monotonic()
        self._toggle_record()

    def _record_released(self):
        action, self._press_action = getattr(self, "_press_action", None), None
        if action == "stop":
            if self.recording:
                self._toggle_record()
            return
        if action == "started" and self.recording:
            held = time.monotonic() - self._press_time
            if held >= HOLD_TO_RECORD_SECS:
                self._toggle_record()   # a hold: the clip ends on release
            # a quick tap latches instead — recording runs until tapped again

    def _toggle_record(self):
        if not self.recording:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(self.output_dir, f"recording_{ts}.mp4")
            fps = self.fps_spin.value()
            # trust the measured rate over the camera's nominal one, so the
            # video plays back at true speed even when exposure (or the
            # camera) limits the frame rate
            actual = self.cam_thread.measured_fps or self.cam_thread.resulting_fps
            if actual and actual < fps * 0.95:
                fps = actual
            free = shutil.disk_usage(self.output_dir).free
            if free < 2 * MIN_FREE_BYTES:
                QtWidgets.QMessageBox.warning(
                    self, "Disk almost full",
                    f"Only {free / 1e9:.1f} GB free — make space before "
                    "recording.")
                return
            quality = QUALITY_PRESETS[self.quality_combo.currentIndex()][1]
            try:
                self.cam_thread.start_recording(path, fps, quality)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Recording failed", str(exc))
                return
            self.recording = True
            self.record_started = time.monotonic()
            self.current_file = path
            self.quality_combo.setEnabled(False)
            self.pol_combo.setEnabled(False)
            self.depth_combo.setEnabled(False)
            self.record_btn.setText("■  Stop")
            self.record_btn.setStyleSheet(
                "QPushButton {font-size: 18px; font-weight: bold;"
                " background-color: #2e7d32; color: white;"
                " border-radius: 8px;}")
            self.fps_spin.setEnabled(False)
            self.camera_combo.setEnabled(False)
        else:
            self.recording = False
            self.record_btn.setText("●  Record")
            self.record_btn.setStyleSheet(
                "QPushButton {font-size: 18px; font-weight: bold;"
                " background-color: #c62828; color: white;"
                " border-radius: 8px;}"
                "QPushButton:disabled {background-color: #666666;}")
            self._finish_recording()

    def _finish_recording(self):
        """Finalize the recording in the background — encoding the buffered
        frames and finalizing a large file can take a while, and killing
        ffmpeg early would corrupt it."""
        writer = self.cam_thread.detach_writer()
        if writer is None:
            return
        self.record_btn.setEnabled(False)
        self.record_info.setText("Finishing recording — please wait…")
        state = {"done": False, "writer": writer, "err": ""}
        self._finishing = state

        def work():
            state["err"] = writer.close() or ""
            state["done"] = True

        threading.Thread(target=work, daemon=True).start()
        timer = QtCore.QTimer(self)

        def poll():
            if not state["done"]:
                return
            timer.stop()
            timer.deleteLater()
            self._finishing = None
            self.record_btn.setEnabled(True)
            self.fps_spin.setEnabled(True)
            self.camera_combo.setEnabled(True)
            self.quality_combo.setEnabled(True)
            self.pol_combo.setEnabled(True)
            self.depth_combo.setEnabled(True)
            name = os.path.basename(writer.path)
            if writer.frames_written == 0:
                # too brief to catch a single frame — don't leave an
                # unplayable file behind
                try:
                    os.remove(writer.path)
                except OSError:
                    pass
                self.record_info.setText(
                    "Too short — no frames captured. Hold a little longer.")
                return
            msg = f"Saved {name} ({writer.frames_written} frames)"
            if writer.frames_dropped:
                total = writer.frames_written + writer.frames_dropped
                span = (writer.t_last - writer.t_first) if writer.t_first else 0
                rate = writer.frames_written / span if span else 0
                msg += (f" — kept {100 * writer.frames_written / total:.0f}%,"
                        f" effectively {rate:.1f} fps")
                self.record_info.setStyleSheet("color: #b35000;")
            else:
                self.record_info.setStyleSheet("")
            self.record_info.setText(msg)
            if state["err"]:
                self.status.setText(f"ffmpeg: {state['err']}")

        timer.timeout.connect(poll)
        timer.start(300)

    def _snapshot(self):
        frame = self.cam_thread.latest_frame if self.cam_thread else None
        if frame is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.output_dir, f"snapshot_{ts}.png")
        # a uint16 frame writes a true 16-bit PNG, keeping all 12 camera bits
        cv2.imwrite(path, frame)
        depth = "16-bit" if frame.dtype == np.uint16 else "8-bit"
        self.record_info.setText(
            f"Saved {os.path.basename(path)} ({depth})")

    # ---- settings callbacks -------------------------------------------------
    def _fps_changed(self, value):
        self._apply(fps=value)
        self._update_rate_warning()

    def _update_rate_warning(self):
        """Warn before recording if the chosen rate exceeds what the
        encoder can write, rather than letting frames vanish silently."""
        d = getattr(self, "_cam_info", None)
        if not d:
            self.rate_warning.setVisible(False)
            return
        limit = encoder_limit_fps(
            d["width"], d["height"],
            2 if d.get("bit16") else (3 if d["is_color"] else 1))
        want = self.fps_spin.value()
        if want > limit * 1.05:
            depth = "12-bit" if d.get("bit16") else "8-bit"
            self.rate_warning.setText(
                f"⚠ Recording can only keep about {limit:.0f} fps at "
                f"{depth} — above that, frames are dropped. Preview and "
                f"snapshots still run at {want:.0f}.")
            self.rate_warning.setVisible(True)
        else:
            self.rate_warning.setVisible(False)

    def _depth_changed(self, idx):
        # the pixel format can't be switched mid-stream, so reconnect
        self.want12 = idx == 1
        desc = getattr(self, "_last_descriptor", None)
        if desc:
            self._connect_camera(desc)

    def _pol_changed(self, idx):
        smooth = idx == 0
        self._apply(pol_smooth=smooth)
        d = getattr(self, "_cam_info", None)
        if d and d.get("polarized"):
            # info dims were reported for the mode active at connect time
            base_w = d["width"] * (2 if d.get("smooth") else 1)
            base_h = d["height"] * (2 if d.get("smooth") else 1)
            w = base_w // 2 if smooth else base_w
            h = base_h // 2 if smooth else base_h
            self.camera_label.setText(
                f"{d['model']}  ({d['serial']})\n{w}×{h} mono")

    def _auto_exposure_toggled(self, on):
        self.exp_spin.setEnabled(not on)
        self._apply(auto_exposure=on)
        if not on:
            self._apply(exposure_us=self.exp_spin.value() * 1000.0)

    def _gain_changed(self, ticks):
        gain = self._gain_min + ticks * self._gain_scale
        self.gain_label.setText(f"{gain:.0f}")
        self._apply(gain=gain)

    def _choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_dir)
        if d:
            self.output_dir = d
            self.out_edit.setText(d)

    def _open_folder(self):
        subprocess.Popen(["xdg-open", self.output_dir])

    # ---- status bar ---------------------------------------------------------
    def _update_status(self):
        if not self.cam_thread:
            return
        parts = [f"{self.cam_thread.measured_fps:.1f} fps"]
        if self.recording:
            free = shutil.disk_usage(self.output_dir).free
            if free < MIN_FREE_BYTES:
                self._toggle_record()
                QtWidgets.QMessageBox.warning(
                    self, "Disk full",
                    "Recording stopped automatically — the disk is "
                    "almost full.")
                return
            secs = int(time.monotonic() - self.record_started)
            writer = self.cam_thread.writer
            if writer:
                parts.append(f"REC {secs // 60:02d}:{secs % 60:02d}")
                parts.append(f"{writer.frames_written} frames")
                if writer.frames_dropped:
                    # say what is actually being written, not just that
                    # something went wrong
                    elapsed = max(time.monotonic() - self.record_started, 0.1)
                    parts.append(
                        f"dropping! saving {writer.frames_written / elapsed:.1f}"
                        f" of {self.fps_spin.value():.0f} fps")
                    self.record_info.setStyleSheet("color: #b35000;")
                else:
                    self.record_info.setStyleSheet("")
                try:
                    size = os.path.getsize(writer.path)
                    parts.append(f"{size / 1e6:.0f} MB")
                    parts.append(f"{free / 1e9:.1f} GB free")
                except OSError:
                    pass
            self.record_info.setText("  |  ".join(parts[1:]))
        self.status.setText("  |  ".join(parts))

        # warn if exposure limits the frame rate
        rf = self.cam_thread.resulting_fps
        want = self.fps_spin.value()
        if rf and not self.recording and rf < want * 0.95:
            self.status.setText(
                f"{self.cam_thread.measured_fps:.1f} fps — limited to "
                f"{rf:.1f} fps by exposure time")

    def closeEvent(self, event):
        if self.recording:
            self._toggle_record()  # starts background finalization
        fin = self._finishing
        if fin and not fin["done"]:
            dlg = QtWidgets.QProgressDialog(
                "Finishing the recording — this can take a while for "
                "large files.\nPlease don't unplug the camera or power.",
                None, 0, 0, self)
            dlg.setWindowTitle("Saving recording")
            dlg.setWindowModality(QtCore.Qt.WindowModal)
            dlg.setCancelButton(None)
            dlg.setMinimumDuration(0)
            dlg.show()
            while not fin["done"]:
                QtWidgets.QApplication.processEvents()
                time.sleep(0.05)
            dlg.close()
        if self.cam_thread:
            self.cam_thread.shutdown()
        event.accept()


def main():
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is required but was not found. "
              "Install it with: sudo apt install ffmpeg")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Camera Video Recorder")
    win = MainWindow()
    avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
    if avail.width() <= 1366 or avail.height() <= 800:
        win.showMaximized()   # small screen: use all of it
    else:
        win.show()
    rc = app.exec_()
    # skip Python finalization: GObject (Aravis) teardown at interpreter
    # exit can segfault after everything is already cleanly closed
    os._exit(rc)


if __name__ == "__main__":
    main()
