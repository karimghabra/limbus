"""An interactive view of a whole synthetic eye as the slit-lamp camera would see it, in the browser.

    from synth.eye import Eye
    from synth.eyeview import write_viewer
    write_viewer(Eye(seed=0), "eye_seed0.html")

Every vessel of the eye, at every level of detail (coarse trunks, fine trees, the capillary plexus), is drawn as a
tube that absorbs light by the Beer-Lambert law over the scleral texture: its optical density across the lumen
follows the chord a ray crosses, its tube haematocrit and the camera's spectral bands (synth.anatomy.Optics), with
the light returned from above it (the scattering veil) and a blur that grows with depth. Its calibre varies along
its length as the generator drew it. The red cells are a density texture carried along each vessel at its own
red-cell speed, through the cardiac cycle (the speed of every vessel pulses with its own amplitude and delay from
the flow solution's harmonics): discrete cells in single file in the capillaries, clumps and plasma gaps in the
larger vessels. A vessel narrower than a screen pixel is drawn one pixel wide with its absorption scaled by its
width, so the levels of detail never switch: zooming in reveals the fine vessels exactly as a camera of finer
resolution would, each already carrying its flow.

The page loads three.js from a CDN; pass inline_three=<folder with three.module.min.js and OrbitControls.js> to
write a page that works offline.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .network import resample

KIND_ORDER = ["conjunctival artery", "conjunctival vein", "ciliary artery", "episcleral artery", "episcleral vein",
              "limbal vein", "communicating artery", "communicating vein", "perforating artery", "perforating vein",
              "arteriovenous anastomosis", "arteriole", "venule", "capillary", "deep artery", "deep vein"]
STEP_UM = {0: 25.0, 1: 15.0, 2: 30.0}           # centreline sampling of each level in the page


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def export(eye, harmonics=4):
    """The eye as flat arrays for the page: points (float32 mm, eye coordinates), their local diameter (0.25 µm
    units), and per vessel its first point, kind, level, tube haematocrit, mean red-cell speed and the pulse of
    its flow relative to the mean, harmonic by harmonic (int8 pairs, 1/32)."""
    net, flow = eye.network, eye.flow
    pts, dia, first, kind, level, ht, speed, pulse = [], [], [0], [], [], [], [], []
    K = min(harmonics, len(flow.pulse))
    for i, e in enumerate(net.edges):
        if len(e.points) < 2:
            continue
        local = np.column_stack([e.points, e.local_diameter])
        sampled = resample(local, STEP_UM.get(e.level, 20.0))
        if len(sampled) < 2:
            sampled = local[[0, -1]]
        q = float(flow.flow[i])
        ratio = flow.pulse[:K, i] / q if (K and abs(q) > 1e-9) else np.zeros(K, complex)
        if q < 0:                                    # every vessel runs along its mean flow in the page
            sampled = sampled[::-1]
        ratio = np.clip(ratio.real, -3.9, 3.9) + 1j * np.clip(ratio.imag, -3.9, 3.9)
        pts.append(sampled[:, :3] * 1e-3)
        dia.append(np.clip(np.round(sampled[:, 3] * 4), 1, 65535))
        first.append(first[-1] + len(sampled))
        kind.append(KIND_ORDER.index(e.kind) if e.kind in KIND_ORDER else 255)
        level.append(e.level)
        ht.append(float(flow.ht[i]) if flow.hd[i] > 0 else 0.0)
        speed.append(abs(float(flow.rbc_velocity[i])))
        pulse.append(np.stack([ratio.real, ratio.imag], 1).ravel() if K else np.zeros(0))
    points = np.vstack(pts).astype(np.float32)
    data = dict(
        n_edges=len(first) - 1, n_points=int(len(points)), harmonics=K, kinds=KIND_ORDER,
        R=eye.globe.R * 1e-3, limbus=eye.anatomy.limbus_radius_um * 1e-3,
        heart_rate_hz=eye.scene.physiology.heart_rate_hz,
        fissure=[eye.anatomy.fissure_half_width_mm, eye.anatomy.upper_lid_mm, eye.anatomy.lower_lid_mm],
        spectrum=[[float(w), float(mu)] for w, mu in eye.scene.optics.spectrum],
        double_pass=eye.scene.optics.double_pass,
        veil_um=eye.scene.optics.scatter_veil_length_um, blur_um_at_500=eye.scene.optics.scatter_blur_um_at_500,
        rbc_um=eye.scene.physiology.rbc_diameter_um, rbc_volume=eye.scene.physiology.rbc_volume_um3,
        seed=eye.seed,
        points=_b64(points), diameter=_b64(np.concatenate(dia).astype(np.uint16)),
        first=_b64(np.asarray(first, np.int32)), kind=_b64(np.asarray(kind, np.uint8)),
        level=_b64(np.asarray(level, np.uint8)), ht=_b64(np.round(np.asarray(ht) * 255).astype(np.uint8)),
        speed=_b64(np.asarray(speed, np.float32)),
        pulse=_b64(np.round(np.concatenate(pulse) * 32).astype(np.int8)) if K else "",
    )
    return data


def write_viewer(eye, path, inline_three=None, title=None):
    """Write the page for an eye. inline_three: a folder holding three.module.min.js and OrbitControls.js (r160) to
    embed, for a page that needs no network."""
    template = (Path(__file__).with_name("eyeview.html")).read_text()
    data = export(eye)
    if inline_three:
        folder = Path(inline_three)
        modules = {"three": folder / "three.module.min.js", "three/OrbitControls.js": folder / "OrbitControls.js"}
        imports = {name: "data:text/javascript;base64," + base64.b64encode(f.read_bytes()).decode()
                   for name, f in modules.items()}
    else:
        base = "https://cdn.jsdelivr.net/npm/three@0.160.0"
        imports = {"three": f"{base}/build/three.module.min.js",
                   "three/OrbitControls.js": f"{base}/examples/jsm/controls/OrbitControls.js"}
    html = (template.replace("__IMPORTMAP__", json.dumps({"imports": imports}))
            .replace("__TITLE__", title or f"Synthetic eye {eye.seed}")
            .replace("__DATA__", json.dumps(data)))
    Path(path).write_text(html)
    return Path(path)
