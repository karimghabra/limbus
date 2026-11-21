# synth: a synthetic conjunctiva

Synthetic bursts of the bulbar conjunctival microcirculation, with the exact answer: an anatomically constrained 3D
vessel network, blood flow that conserves volume and red cells at every junction, discrete red cells moving
through each vessel's circular cross-section, and the LIMBUS camera that records them. It exists to validate flow
detection and vessel tracing on data whose truth is known, and to generate training data.

```python
from synth import Scene, simulate, write_burst
result = simulate(Scene(seed=1), n_frames=120)        # ~4 min at 768 x 512 on 6 threads
write_burst(result, "burst_synthetic_seed1")         # the recorder's format, plus truth/
```

`notebooks/synthetic.py` walks through every stage and compares a synthetic burst with a real one. The
specification, with the source of every number, is
[docs/synthetic_conjunctiva_anatomy.md](../docs/synthetic_conjunctiva_anatomy.md); the validation is
[docs/synthetic_conjunctiva_validation.md](../docs/synthetic_conjunctiva_validation.md).

## Modules

| module | what it does |
|---|---|
| `anatomy.py` | every parameter (`Anatomy`, `Physiology`, `Optics`, `Scene`), each with its source |
| `network.py` | the 3D network: conjunctival arteriolar and venular trees grown by space colonisation, the capillary mesh, anastomoses, the episcleral and deep plexuses, episcleral arteriovenous anastomoses, communicating and perforating vessels |
| `haemodynamics.py` | Poiseuille network flow with the in vivo viscosity law, Fåhræus tube haematocrit, phase separation at bifurcations, the cardiac pulse |
| `cells.py` | discrete red cells: cell-free layer, blunted profile, clumped spacing, exact red-cell routing at junctions |
| `camera.py` | eye motion, depth-dependent absorption, blur and contrast, scleral texture, illumination flicker, motion blur, the IMX249 sensor and gamma LUT |
| `simulate.py` | one burst from a scene, its truth maps, and `write_burst` |
| `check.py` | the anatomy check: every structure and haemodynamic quantity against its cited range |
| `targets.py` | the target statistics of any burst, real or synthetic (motion, flicker, noise, blur, texture, morphology, contrast, red-cell flicker) |
| `validate.py` | a flow analysis of a synthetic burst against its truth |
| `randomize.py` | random scenes from a seed and a configuration: anatomy within cited ranges, appearance domain-randomised |
| `media.py` | side-by-side animated previews |

## The ground truth

`write_burst` writes `frame_NNNNNN.tif`, `manifest.json` and `frames.csv` exactly as the recorder does, so
`notebooks/foundation.py` and the flow analysis run on a synthetic burst unchanged, and `truth/` beside them:

| file | content |
|---|---|
| `scene.json` | every parameter of the scene |
| `graph.json` | the directed graph: nodes (3D position in µm, pressure), edges oriented along the flow (3D centreline, diameter, length, depth, kind, layer, vessel id, flow, blood and red-cell velocity, haematocrits, viscosity, wall shear rate) |
| `frames.npz` | per frame: tissue motion, rotation, focal depth, time, every edge's flow and red-cell velocity (the cardiac pulse) |
| `cells.npz` | per frame: every visible red cell's 3D position, id and segment |
| `masks.npz` | per frame: the shallowest vessel's id, depth and projected velocity at every pixel; and the frame-0 maps with the mean flow |

Coordinates and conventions are in the `synth.simulate` docstring. A 120-frame 768 × 512 burst takes about 500 MB
with its truth, most of it `cells.npz`; `simulate(..., record_cells=False)` drops it.

## Tests

```bash
python -m pytest synth/tests
```

They check the rheology laws, exact conservation of flow and red cells at every junction (also through the
pulse), that cells stay in their lumens and follow the haematocrit, that the network is layered in 3D, that the
anatomy check passes for the default and random scenes, determinism, and the export.
