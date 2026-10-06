"""Training speed and peak VRAM of candidate backbones (PLAN.md §4A, measured in iteration 0).

    python vesselnet/bench.py --encoders resnet34 tu-convnext_tiny --batches 8 12 --crop 512

A U-Net (segmentation_models_pytorch) with 1 input channel and --classes outputs, trained on random tensors
with AdamW and mixed precision (bf16 autocast); --steps timed after --warmup.  Prints one JSON row per setting.
"""
import argparse
import json
import time

import torch


def bench(encoder, batch, crop, classes, steps, warmup, amp):
    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")
    m = smp.Unet(encoder, encoder_weights=None, in_channels=1, classes=classes).to(dev)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(batch, 1, crop, crop, device=dev)
    y = torch.rand(batch, classes, crop, crop, device=dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for it in range(warmup + steps):
        if it == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = m(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    return dict(encoder=encoder, batch=batch, crop=crop, classes=classes, amp=amp, params_m=round(n_par / 1e6, 1),
                s_per_step=round(dt, 3), crops_per_s=round(batch / dt, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoders", nargs="+", default=["resnet34", "tu-convnext_tiny"])
    ap.add_argument("--batches", nargs="+", type=int, default=[8, 12])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)), flush=True)
    for enc in a.encoders:
        for b in a.batches:
            try:
                r = bench(enc, b, a.crop, a.classes, a.steps, a.warmup, not a.no_amp)
            except torch.cuda.OutOfMemoryError:
                r = dict(encoder=enc, batch=b, crop=a.crop, error="out of memory")
            print(json.dumps(r), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
