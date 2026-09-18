import time
import os as _os
_os.environ.pop("GENICAM_GENTL64_PATH", None)  # see camera_recorder.py: system GenTL producer crashes pypylon
from pypylon import pylon
from bench_camera import node, open_camera

cam = open_camera("Mono12", 1000.0, 2000.0)
print("Width : min=%d max=%d inc=%d" % (cam.Width.Min, cam.Width.Max, cam.Width.Inc))
print("Height: min=%d max=%d inc=%d" % (cam.Height.Min, cam.Height.Max, cam.Height.Inc))
print("OffsetX: min=%d max=%d inc=%d" % (cam.OffsetX.Min, cam.OffsetX.Max, cam.OffsetX.Inc))
print("OffsetY: min=%d max=%d inc=%d" % (cam.OffsetY.Min, cam.OffsetY.Max, cam.OffsetY.Inc))

def ceiling(w, h, oy=None):
    cam.OffsetX.Value = cam.OffsetX.Min
    cam.OffsetY.Value = cam.OffsetY.Min
    cam.Width.Value = w
    cam.Height.Value = h
    if oy is not None:
        cam.OffsetY.Value = min(oy, cam.OffsetY.Max)
    cam.AcquisitionFrameRateEnable.Value = True
    cam.AcquisitionFrameRate.Value = cam.AcquisitionFrameRate.Max
    return cam.ResultingFrameRate.Value, cam.OffsetY.Value

print("\n--- does WIDTH change the frame rate? (height fixed at 200) ---")
for w in (1920, 1440, 960, 480, 128):
    fps, _ = ceiling(w, 200)
    print("  %4dx200 -> %7.1f fps" % (w, fps))

print("\n--- HEIGHT, full width (reported ceiling) ---")
for h in (604, 400, 300, 200, 128, 100, 60, 32, 16):
    fps, _ = ceiling(1920, h)
    print("  1920x%-4d -> %7.1f fps  (period %.3f ms)" % (h, fps, 1000.0/fps))

print("\n--- can the strip be placed anywhere vertically? (1920x60) ---")
for oy in (0, 300, 600, 900, 1156):
    fps, actual = ceiling(1920, 60, oy)
    print("  OffsetY=%-5d actual=%-5d -> %7.1f fps" % (oy, actual, fps))

print("\n--- measured, not just reported: 1920x60 grab test ---")
cam.OffsetY.Value = cam.OffsetY.Min
cam.Height.Value = 60
cam.Width.Value = 1920
cam.OffsetY.Value = 600
cam.AcquisitionFrameRate.Value = cam.AcquisitionFrameRate.Max
conv = pylon.ImageFormatConverter()
conv.OutputPixelFormat = pylon.PixelType_Mono16
conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
n = bad = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < 8.0:
    r = cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if r and r.IsValid():
        if r.GrabSucceeded():
            arr = conv.Convert(r).GetArray()
            n += 1
        else:
            bad += 1
        r.Release()
dt = time.perf_counter() - t0
cam.StopGrabbing()
print("  grabbed %d frames in %.1fs = %.1f fps, incomplete %d, shape %s" % (n, dt, n/dt, bad, arr.shape))
print("  data rate: %.0f MB/s" % (n/dt*arr.nbytes/1e6))
cam.Close()
