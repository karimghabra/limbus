"""Can the camera skip or average rows in hardware (keeping full field of
view) instead of us cropping rows away — and does it actually read out
faster if it does?
"""
from bench_camera import node, open_camera

cam = open_camera("Mono12", 1000.0, 1000.0)

for name in ("DecimationVertical", "DecimationHorizontal", "BinningVertical",
             "BinningHorizontal", "BinningVerticalMode", "ScalingVertical"):
    n = node(cam, name)
    if n is None:
        print(f"{name:22s} not available")
        continue
    try:
        print(f"{name:22s} value={n.Value} min={n.Min} max={n.Max}")
    except Exception:
        print(f"{name:22s} value={n.Value} (enum)")


def report(label):
    cam.OffsetX.Value = cam.OffsetX.Min
    cam.OffsetY.Value = cam.OffsetY.Min
    cam.Width.Value = cam.Width.Max
    cam.Height.Value = cam.Height.Max
    cam.AcquisitionFrameRateEnable.Value = True
    cam.AcquisitionFrameRate.Value = cam.AcquisitionFrameRate.Max
    print(f"  {label:32s} -> {cam.ResultingFrameRate.Value:6.1f} fps, "
          f"image {cam.Width.Value}x{cam.Height.Value}, "
          f"payload {cam.PayloadSize.Value / 1e6:.2f} MB")


print("\n--- whole sensor covered, Mono12 ---")
report("no decimation/binning")

dec = node(cam, "DecimationVertical")
if dec is not None:
    for val in (2, 4):
        try:
            dec.Value = val
            report(f"DecimationVertical={val}")
        except Exception as exc:
            print(f"  DecimationVertical={val} failed: {exc}")
    dec.Value = 1

binv = node(cam, "BinningVertical")
if binv is not None:
    mode = node(cam, "BinningVerticalMode")
    for m in ("Average", "Sum"):
        try:
            if mode is not None:
                mode.Value = m
            binv.Value = 2
            report(f"BinningVertical=2 ({m})")
        except Exception as exc:
            print(f"  BinningVertical=2 ({m}) failed: {exc}")
    binv.Value = 1

cam.Close()
