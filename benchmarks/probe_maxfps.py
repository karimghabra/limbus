"""What is the absolute maximum frame rate in each pixel format?

Sweeps ROI height down to a single row with an exposure short enough not to
be the limit, and grabs at each to confirm the reported ceiling is real.
"""
import time

import os as _os
_os.environ.pop("GENICAM_GENTL64_PATH", None)  # see camera_recorder.py: system GenTL producer crashes pypylon
from pypylon import pylon

from bench_camera import converter, open_camera

EXPOSURE_US = 100.0
HEIGHTS = (200, 100, 60, 32, 16, 8, 4, 2, 1)


def measure(cam, conv, secs=3.0):
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    n = bad = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        r = cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if r and r.IsValid():
            if r.GrabSucceeded():
                conv.Convert(r).GetArray()
                n += 1
            else:
                bad += 1
            r.Release()
    dt = time.perf_counter() - t0
    cam.StopGrabbing()
    return n / dt, bad


for fmt in ("Mono12", "Mono8"):
    print(f"\n=== {fmt}, full width, exposure {EXPOSURE_US:.0f} us ===")
    cam = open_camera(fmt, 100000.0, EXPOSURE_US)
    conv = converter()
    for h in HEIGHTS:
        cam.OffsetY.Value = cam.OffsetY.Min
        cam.Height.Value = h
        cam.AcquisitionFrameRateEnable.Value = True
        cam.AcquisitionFrameRate.Value = cam.AcquisitionFrameRate.Max
        reported = cam.ResultingFrameRate.Value
        actual, bad = measure(cam, conv)
        print(f"  {cam.Width.Value}x{h:<4d} reported {reported:7.1f} fps | "
              f"measured {actual:7.1f} fps | incomplete {bad} | "
              f"period {1000.0 / reported:.3f} ms | "
              f"{actual * cam.PayloadSize.Value / 1e6:5.0f} MB/s")
    cam.Close()
