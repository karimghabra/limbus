"""Why does Mono12 cap at ~32 fps? Compare pixel formats and test whether
the device link throughput limit or the sensor readout is responsible.

usage: python bench_pixfmt.py [seconds_per_format]
"""
import sys
import time

import numpy as np
import os as _os
_os.environ.pop("GENICAM_GENTL64_PATH", None)  # see camera_recorder.py: system GenTL producer crashes pypylon
from pypylon import pylon

from bench_camera import node, open_camera

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0


def probe_format(fmt, throughput_limit_mode="On", exposure_us=5000.0):
    cam = open_camera(fmt, 1000.0, exposure_us)
    try:
        n = node(cam, "DeviceLinkThroughputLimitMode")
        if n is not None:
            try:
                cam.DeviceLinkThroughputLimitMode.Value = throughput_limit_mode
            except Exception as exc:
                print(f"  (could not set throughput limit mode: {exc})")
        try:
            cam.AcquisitionFrameRateEnable.Value = True
            fr = node(cam, "AcquisitionFrameRate", "AcquisitionFrameRateAbs")
            fr.Value = fr.Max
        except Exception:
            pass
        info = {}
        for name in ("PayloadSize", "ResultingFrameRate", "DeviceLinkThroughputLimit",
                     "DeviceLinkCurrentThroughput", "SensorReadoutTime",
                     "ReadoutTime", "AcquisitionFrameRate"):
            g = node(cam, name)
            if g is not None:
                try:
                    info[name] = g.Value
                except Exception:
                    pass
        conv = pylon.ImageFormatConverter()
        conv.OutputPixelFormat = pylon.PixelType_Mono16
        conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        n_frames = bad = 0
        first = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < SECS:
            res = cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if not res or not res.IsValid():
                bad += 1
                continue
            if res.GrabSucceeded():
                arr = conv.Convert(res).GetArray()
                if first is None:
                    first = (arr.dtype, arr.shape, int(arr.max()))
                n_frames += 1
            else:
                bad += 1
            res.Release()
        dt = time.perf_counter() - t0
        cam.StopGrabbing()
        mbs = n_frames / dt * info.get("PayloadSize", 0) / 1e6
        print(f"[{fmt:8s} limit={throughput_limit_mode:3s}] measured "
              f"{n_frames / dt:6.2f} fps, incomplete {bad}, "
              f"wire {mbs:5.0f} MB/s, payload {info.get('PayloadSize', 0) / 1e6:.2f} MB, "
              f"ResultingFrameRate={info.get('ResultingFrameRate', 0):.2f}, "
              f"linkLimit={info.get('DeviceLinkThroughputLimit', 0) / 1e6:.0f} MB/s, "
              f"converted {first}", flush=True)
    finally:
        cam.Close()


if __name__ == "__main__":
    cam = open_camera("Mono8", 1000.0)
    print("available pixel formats:", list(cam.PixelFormat.Symbolics))
    n = node(cam, "DeviceLinkThroughputLimit")
    if n is not None:
        print(f"DeviceLinkThroughputLimit: value={n.Value} min={n.Min} max={n.Max}")
    cam.Close()

    for fmt in ("Mono8", "Mono12", "Mono12p"):
        probe_format(fmt, "On")
    for fmt in ("Mono12", "Mono12p"):
        probe_format(fmt, "Off")
    # short exposure can't be the limit, but prove it
    probe_format("Mono12p", "Off", exposure_us=1000.0)
