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

A whole eye, recorded at any position and magnification:

```python
from synth.eye import Eye, Pose, simulate_view
eye = Eye(seed=4)                                         # every vessel of the eye, solved once, ~2 min, 0.4 GB
burst = simulate_view(eye, Pose(s_mm=3.0, clock_deg=90))  # the temporal conjunctiva, 3 mm from the limbus

from synth.eyeview import write_viewer
write_viewer(eye, "eye_seed4.html")                       # the whole eye as the camera sees it, in the browser
```

A patient imaged at the slit lamp, and their vascular map built from the live frames:

```python
from synth.gaze import patient, SlitLamp
from synth.mapping import Mapper, plan
eye = patient(7)                                          # the same seed is always the same patient
lamp = SlitLamp(eye, aim=(40.0, 0.0))                     # the camera, fixed in the head
mapper = Mapper(eye.globe, region=((60, 120), (0.5, 6.5)))
previous = None
for gaze in plan(lamp, clocks=(60, 120), s_mm=(1.0, 6.0)):
    burst = lamp.record(gaze, n_frames=16, previous=previous)   # lids, fixational movements, saccades
    mapper.add(burst.result.frames, gaze, lamp)                  # from the frames alone
    previous = gaze
```

`notebooks/synthetic.py` walks through every stage and compares a synthetic burst with a real one. The
specification, with the source of every number, is
[docs/synthetic_conjunctiva_anatomy.md](../docs/synthetic_conjunctiva_anatomy.md); the validation is
[docs/synthetic_conjunctiva_validation.md](../docs/synthetic_conjunctiva_validation.md).

## Modules

| module | what it does |
|---|---|
| `anatomy.py` | every parameter (`Anatomy`, `Physiology`, `Optics`, `Scene`), each with its source |
| `network.py` | the 3D network: conjunctival arteriolar and venular trees grown by space colonisation, companion venules beside the arterioles, the capillary mesh, anastomoses, the episcleral and deep plexuses, episcleral arteriovenous anastomoses, communicating and perforating vessels; multi-scale tortuosity (meander, bends, jitter) and calibre varying along every vessel |
| `haemodynamics.py` | Poiseuille network flow with the in vivo viscosity law, Fåhræus tube haematocrit, phase separation at bifurcations, the cardiac pulse |
| `cells.py` | discrete red cells: cell-free layer, blunted profile, clumped spacing, exact red-cell routing at junctions |
| `camera.py` | eye motion, depth-dependent absorption, blur and contrast, scleral texture, illumination flicker, motion blur, the IMX249 sensor and gamma LUT |
| `simulate.py` | one burst from a scene, its truth maps, and `write_burst` |
| `check.py` | the anatomy check: every structure and haemodynamic quantity against its cited range |
| `targets.py` | the target statistics of any burst, real or synthetic (motion, flicker, noise, blur, texture, morphology, contrast, red-cell flicker) |
| `validate.py` | a flow analysis of a synthetic burst against its truth |
| `randomize.py` | random scenes from a seed and a configuration: anatomy within cited ranges, appearance domain-randomised |
| `media.py` | side-by-side animated previews, on one shared optical-density scale |
| `view3d.py` | interactive 3D views (plotly) of a patch network on the globe and of a whole eye, and a rotating render |
| `eye.py` | the whole anterior eye grown from its arterial sources and venous sinks as one network with levels of detail - the coarse vessels of the globe, the conjunctival trees of every tile, one capillary plexus over the whole conjunctiva - solved once, and camera views at any pose and magnification that draw it with its own flows (`Eye`, `Pose`, `simulate_view`, `overview`, `perfusion`) |
| `eyeview.py` | an interactive page of a whole eye as the camera sees it: Beer-Lambert absorption of every vessel, its calibre along its length, its red cells moving at its pulsing speed, at any magnification |
| `gaze.py` | a patient from a seed; the eye in the head (Listing's law), fixational drift, microsaccades, saccades, head motion, lids that follow the gaze, blinks; the slit lamp that records bursts at a commanded gaze, with their truth |
| `mapping.py` | a proof of concept of mapping a patient's vasculature from live bursts only: fixation planning, burst stabilisation, placement on the globe by the commanded gaze refined by registration, accumulation into a surface map, and its evaluation against the truth |

## The ground truth

`write_burst` writes `frame_NNNNNN.tif`, `manifest.json` and `frames.csv` exactly as the recorder does, so
`notebooks/foundation.py` and the flow analysis run on a synthetic burst unchanged, and `truth/` beside them:

| file | content |
|---|---|
| `scene.json` | every parameter of the scene |
| `graph.json` | the directed graph: nodes (3D position in µm, pressure), edges oriented along the flow (3D centreline, diameter, length, depth, kind, layer, vessel id, flow, blood and red-cell velocity, haematocrits, viscosity, wall shear rate) |
| `frames.npz` | per frame: tissue motion, rotation, focal depth, time, every edge's flow and red-cell velocity (the cardiac pulse), positive from the edge's source to its target as in `graph.json` |
| `cells.npz` | per frame, at mid-exposure: the 3D position, id and segment of every red cell the camera draws (cells hidden inside a wider vessel at a junction are left out) |
| `masks.npz` | per frame, at mid-exposure: the shallowest vessel's id, depth and projected velocity at every pixel; and the frame-0 maps with the mean flow. Vessels are drawn at their local diameter but at least about 1.5 px wide, so thin vessels (all capillaries at coarse magnification) look wider than they are: take widths from `graph.json` |

Coordinates and conventions are in the `synth.simulate` docstring, with the camera's projection of the curved globe
that takes a tissue point to its pixel (it is not a plain division by the pixel scale: the globe curves away from
the camera and deeper vessels sit nearer the eye's centre). A 120-frame 768 × 512 burst takes about 500 MB
with its truth, most of it `cells.npz`; `simulate(..., record_cells=False)` drops it.

## Tests

```bash
python -m pytest synth/tests
```

They check the rheology laws, exact conservation of flow and red cells at every junction (through the pulse, with
the volume the distending vessels store), the damping of the pulse downstream, that cells stay in their lumens and
follow the haematocrit, that the network is layered in 3D, that the anatomy check passes for the default and random
scenes, determinism, the export, and for the whole eye: that the same seed gives the same eye, that the capillaries
form one continuous plexus, that every vessel carries its flow in every view at every magnification, the gaze
geometry, the lids, and that the mapper places bursts near their true places.
