# vesselnet pilot (CPU)

The proof of concept behind `vesselnet/PLAN.md`: a small network that finds vessel junctions, trained on
vesselscene scenes. It was run on 4 CPU cores. Iteration 0 of the plan ports it to the GPU and to averaged
frames.

| file | what |
|---|---|
| `gen.py` | scenes from vesselscene: one image kind (`average` or `frame`) and its truth, saved compactly as `s<seed>.npz` |
| `net.py` | preprocessing, targets, crops with augmentation, the U-Net (1.6M parameters), losses, tiled prediction, peak detection, scoring |
| `train.py` | training; validation on seeds divisible by 10, every N iterations |
| `evaluate.py` | `val` (held-out scenes), `zoo` (vesselmap's structure zoo), `real` (overlay on a real image) |

```bash
pip install -e path/to/vesselscene      # or: export VESSELSCENE=path/to/vesselscene
export PYTHONPATH=$PWD                   # the LIMBUS checkout, for vesselmap

python vesselnet/pilot/gen.py --start 0 --stop 60 --kind average --size 512 --device cuda --out data/pilot
python vesselnet/pilot/train.py --data data/pilot --iters 3000 --run runs/pilot1 --gain 0.6,2.6
python vesselnet/pilot/evaluate.py val  --run runs/pilot1 --data data/pilot
python vesselnet/pilot/evaluate.py zoo  --run runs/pilot1 --seeds 0 10 20
python vesselnet/pilot/evaluate.py real --run runs/pilot1 --image path/to/mean_stabilized.tif --out real.png
```

**The network.**
- Input: log intensity less its Gaussian of σ 24 px, divided by the median absolute deviation.
- Outputs:
  - junction heat: a Gaussian of σ 3 px at each observable junction;
  - junction type: crossing / branch / compound;
  - vessel lumen and centreline.
- Losses: focal loss on the heat; cross-entropy on the type near junctions; BCE on lumen and centreline.
- Don't-care masks come from vesselscene's observable truth.

**Results.** These are for 512² single frames (`--kind frame`); the averages were not tried on CPU. See
`vesselnet/RESULTS.md`, entry 0, and PLAN.md §9.
