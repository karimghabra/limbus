import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # the notebook may be opened through a symlink: import the synth package from the checkout it really lives in
    REPOSITORY = Path(__file__).resolve().parents[1]
    for _path in (REPOSITORY, REPOSITORY / "notebooks"):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    return Path, REPOSITORY, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A synthetic conjunctiva

    A patch of bulbar conjunctiva built from anatomy and physiology, and the camera that records it, with the
    exact ground truth of every clip: for validating flow detection and vessel tracing on data whose answer is
    known, and for generating training data.

    | § | Step | What it gives |
    |---|---|---|
    | 0 | Setup: the scene's seed and size, the real burst to compare with | `scene` |
    | 1 | **The anatomy**: the parameters and where each comes from | the parameter table |
    | 2 | **The 3D network**: conjunctival, episcleral and deep plexuses, communicating and perforating vessels | `net` |
    | 3 | **Blood flow**: pressures, flows, haematocrit, conservation at every junction | `flow` |
    | 4 | **The anatomy check**: every structure and haemodynamic quantity against its cited range | `anatomy_rows` |
    | 5 | **Red cells** in their vessels' circular cross-sections | |
    | 6 | **The camera**: a synthetic burst, and real and synthetic clips side by side | `result` |
    | 7 | **Real against synthetic**: the target statistics through one estimator | `comparison` |
    | 8 | The ground truth, written in the recorder's format with a `truth/` folder | `burst_folder` |
    | 9 | **Flow detection against the truth**, when a flow analysis of the synthetic burst is given | `flow_rows` |

    The specification, with every source, is `docs/synthetic_conjunctiva_anatomy.md`; the package is `synth/`.
    Lengths are micrometres in the tissue and pixels in the image (3.26 µm per pixel, measured with a ruler, docs
    §1.3); velocities are mm/s in the tissue and px/frame in the image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup

    | Setting | Default | What it does |
    |---|---|---|
    | `SEED` | `1` (env `SYNTH_SEED`) | the scene's random seed: the same seed and settings give the same burst |
    | `N_FRAMES` | `120` (env `SYNTH_FRAMES`) | frames to render (about 1-2 s each on 6 threads) |
    | `WIDTH_PX`, `HEIGHT_PX` | `768`, `512` | the field, 2.5 × 1.7 mm at 3.26 µm/px |
    | `REAL_BURST`, `REAL_CROP` | env `LIMBUS_BURST`, `(240, 384, 720, 1152)` | the real burst to compare with, and a crop `(top, left, height, width)` of it; without one, §6-7 show the synthetic side alone |
    | `FLOW_OUTPUTS` | env `SYNTH_FLOW_OUTPUTS` | the `flow_outputs/<burst>` folder the flow notebook wrote for the synthetic burst, for §9 |
    | `OUTPUT_DIR` | `synthetic_outputs` | where the figures, previews and the synthetic burst go, in the working directory |
    """)
    return


@app.cell
def _(Path):
    import os
    import time

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    from synth import Scene, check, haemodynamics, media, network, simulate, targets, validate, write_burst
    from synth.anatomy import describe

    notebook_start = time.perf_counter()
    SEED = int(os.environ.get("SYNTH_SEED", 1))
    N_FRAMES = int(os.environ.get("SYNTH_FRAMES", 120))
    WIDTH_PX = int(os.environ.get("SYNTH_WIDTH", 768))
    HEIGHT_PX = int(os.environ.get("SYNTH_HEIGHT", 512))
    REAL_BURST = os.environ.get("LIMBUS_BURST")
    REAL_CROP = (240, 384, 720, 1152)   # (top, left, height, width); the crop of docs/synthetic_conjunctiva_validation.md
    FLOW_OUTPUTS = os.environ.get("SYNTH_FLOW_OUTPUTS")
    OUTPUT_DIR = Path("synthetic_outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    scene = Scene(width_px=WIDTH_PX, height_px=HEIGHT_PX, seed=SEED)
    INK_MUTED = "#898781"
    SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.titlesize": 10,
        "legend.frameon": False,
    })
    print(f"Scene seed {SEED}: {WIDTH_PX} x {HEIGHT_PX} px = {WIDTH_PX * scene.anatomy.um_per_px / 1e3:.2f} x "
          f"{HEIGHT_PX * scene.anatomy.um_per_px / 1e3:.2f} mm, {N_FRAMES} frames at {scene.optics.fps:.2f} fps")
    return (
        FLOW_OUTPUTS,
        INK_MUTED,
        N_FRAMES,
        OUTPUT_DIR,
        REAL_BURST,
        REAL_CROP,
        SERIES,
        check,
        cv2,
        describe,
        haemodynamics,
        media,
        network,
        notebook_start,
        np,
        plt,
        scene,
        simulate,
        targets,
        time,
        validate,
        write_burst,
    )


@app.cell
def _(OUTPUT_DIR, mo, plt):
    import io as _io


    def show_figure(figure, name, dpi=100):
        """Save a figure to OUTPUT_DIR/<name>.png, close it and show that PNG."""
        path = OUTPUT_DIR / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        mo.output.append(mo.image(_io.BytesIO(path.read_bytes())))

    return (show_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The anatomy

    The field is a patch of bulbar conjunctiva 3 mm from the limbus, behind the palisades. Seen from above, three
    vascular layers are stacked in depth (docs §2):

    - the **conjunctival plexus**, 50-230 µm below the surface in the substantia propria: arteriolar and venular
      trees running radially from the limbus, venules accompanying arterioles, joined by a polygonal capillary
      mesh just under the epithelium, with arcades between trees;
    - the **episcleral plexus**, 260-440 µm deep under Tenon's capsule: long, straight arteries running towards the
      limbus and veins draining away from it, joined by arteriovenous anastomoses (the episclera has no capillary
      bed);
    - the **deep (intrascleral) plexus**, 520-900 µm deep.

    Communicating vessels rise obliquely from the episcleral plexus to feed and drain the conjunctival trees, and
    perforating vessels join the episcleral to the deep plexus. Every parameter and its source:
    """)
    return


@app.cell
def _(describe, mo, scene):
    _rows = ["| group | parameter | value |", "|---|---|---|"]
    for _group, _obj in (("anatomy", scene.anatomy), ("physiology", scene.physiology), ("optics", scene.optics)):
        for _name, _value in describe(_obj):
            _shown = _value if not isinstance(_value, float) else f"{_value:g}"
            _rows.append(f"| {_group} | `{_name}` | {_shown} |")
    mo.md("The sources of every value are in `synth/anatomy.py` beside it and in docs §10.\n\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The 3D network

    `network.generate(scene)` grows the network in a domain larger than the field by `margin_um`, so that vessels
    continue across the image border; the dashed box is the camera's field. Left: seen from the camera, coloured by
    kind, line width proportional to diameter. Right: seen from the side, depth downwards; the oblique green lines
    are the communicating vessels between the conjunctival and episcleral plexuses.
    """)
    return


@app.cell
def _(network, scene, time):
    _start = time.perf_counter()
    net = network.generate(scene)
    _summary = network.summary(net)
    print(f"{_summary['nodes']} nodes, {_summary['edges']} segments, {_summary['boundary nodes']} boundary nodes, "
          f"in {time.perf_counter() - _start:.1f} s")
    for _kind, _row in _summary["by kind"].items():
        print(f"  {_kind:28s} {_row['edges']:5d} segments  {_row['vessels']:5d} vessels  {_row['length (mm)']:7.2f} mm")
    return (net,)


@app.cell
def _(net, np, plt, show_figure):
    KIND_COLORS = {"arteriole": "#e34948", "venule": "#2a78d6", "capillary": "#e87ba4", "episcleral artery": "#9b1c1c",
                   "episcleral vein": "#123f7a", "deep artery": "#eda100", "deep vein": "#1baf7a",
                   "communicating artery": "#008300", "communicating vein": "#6bd16b",
                   "perforating artery": "#000000", "perforating vein": "#6250d6", "arteriovenous anastomosis": "#c97d00"}
    _figure, (_top, _side) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for _e in sorted(net.edges, key=lambda e: -np.mean(e.points[:, 2])):
        _c = KIND_COLORS.get(_e.kind, "gray")
        _w = max(0.25, _e.diameter / 14)
        _top.plot(_e.points[:, 0] / 1e3, _e.points[:, 1] / 1e3, color=_c, lw=_w, alpha=0.85)
        _side.plot(_e.points[:, 0] / 1e3, _e.points[:, 2], color=_c, lw=_w, alpha=0.6)
    _x0, _x1, _y0, _y1 = (v / 1e3 for v in net.field_of_view)
    _top.plot([_x0, _x1, _x1, _x0, _x0], [_y0, _y0, _y1, _y1, _y0], "k--", lw=1)
    _top.set(aspect="equal", xlabel="x along the limbus (mm)", ylabel="y away from the limbus (mm)", title="from the camera")
    _top.invert_yaxis()
    _side.invert_yaxis()
    _side.set(xlabel="x (mm)", ylabel="depth below the surface (µm)", title="from the side")
    for _label, _color in KIND_COLORS.items():
        if any(_e.kind == _label for _e in net.edges):
            _side.plot([], [], color=_color, lw=2, label=_label)
    _side.legend(fontsize=7, loc="lower right", ncol=2)
    show_figure(_figure, "network")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Blood flow

    Each segment is a Poiseuille tube whose viscosity follows the in vivo law of Pries et al. (1994); boundary
    pressures come from each vessel's side of the circulation and its calibre; the discharge haematocrit splits at
    every diverging bifurcation by the phase-separation law (Pries & Secomb 2005), and flow and haematocrit are
    iterated to a fixed point (docs §6). Volume flow and red-cell flux are conserved at every interior node: the
    printed imbalances are relative to each node's throughput.

    Below: red-cell velocity against diameter for the conjunctival vessels, with the published means (Khansari 2016
    and 2017 for arterioles and venules, Koutsiaris 2015 for venules by size, Moka 2020 for capillaries).
    """)
    return


@app.cell
def _(haemodynamics, net, scene, time):
    _start = time.perf_counter()
    flow = haemodynamics.solve(net, scene.physiology)
    _flow_error, _cell_error = flow.conservation(net)
    print(f"Solved in {time.perf_counter() - _start:.1f} s, {flow.iterations} iterations of flow and haematocrit")
    print(f"Largest imbalance at an interior node: volume flow {_flow_error:.1e}, red-cell flux {_cell_error:.1e}")
    print(f"Pressures {flow.pressure.min():.1f}-{flow.pressure.max():.1f} mmHg")
    for _kind, _row in haemodynamics.summary(net, flow).items():
        if not _kind.startswith("_"):
            print(f"  {_kind:28s} d {_row['diameter (um)']:5.1f} µm  |v_rbc| {_row['|v_rbc| median (mm/s)']:5.2f} mm/s  "
                  f"|Q| {_row['|Q| median (pl/s)']:8.1f} pl/s  H_D {_row['H_D mean']:.2f}  H_T {_row['H_T mean']:.2f}")
    return (flow,)


@app.cell
def _(SERIES, flow, net, np, plt, show_figure):
    _kind = np.array([e.kind for e in net.edges])
    _figure, _axis = plt.subplots(figsize=(7.5, 4.5))
    for _k, _c in (("capillary", SERIES[4]), ("arteriole", SERIES[7]), ("venule", SERIES[0])):
        _s = _kind == _k
        _axis.scatter(flow.diameter[_s], np.abs(flow.rbc_velocity[_s]), s=6, color=_c, alpha=0.5, label=f"{_k}s (synthetic)")
    _axis.errorbar([15, 18], [0.63, 0.70], yerr=[0.17, 0.23], fmt="s", color=SERIES[7], mfc="white", label="arterioles [Khansari2016, 2017]")
    _axis.errorbar([18, 20], [0.54, 0.59], yerr=[0.13, 0.17], fmt="o", color=SERIES[0], mfc="white", label="venules [Khansari2016, 2017]")
    _axis.plot([6.6, 11.4, 16.5, 21.3], [0.48, 0.64, 0.82, 0.88], "D-", color=SERIES[0], mfc="white", label="venules by size [Koutsiaris2015]")
    _axis.plot([6.5], [0.51], "^", color=SERIES[4], mfc="white", ms=8, label="capillaries [Moka2020]")
    _axis.set(xscale="log", yscale="log", xlabel="diameter (µm)", ylabel="mean red-cell speed (mm/s)", ylim=(0.01, 10))
    _axis.legend(fontsize=7)
    show_figure(_figure, "velocity_by_diameter")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The anatomy check

    `check.anatomy_report` measures every structure and haemodynamic quantity on the ground truth and compares it
    with the range cited in the docs (the published range, or mean ± 2 SD). *outside* would mean the network is
    not anatomical; *reference* rows have a published value but no usable range.
    """)
    return


@app.cell
def _(check, flow, mo, net, scene):
    anatomy_rows = check.anatomy_report(scene, net, flow)
    _outside = [r["quantity"] for r in anatomy_rows if r["status"] == "outside"]
    mo.md(("**Every checked quantity is within its cited range.**" if not _outside else
           f"**Outside the cited range: {', '.join(_outside)}.**") + "\n\n" + check.as_markdown(anatomy_rows))
    return (anatomy_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Red cells

    Every segment holds the cells its tube haematocrit implies. Each cell sits in the circular cross-section
    inside the cell-free layer, moves with a blunted velocity profile (centre speed 1.5 times the mean), and the
    cells gather in aggregates separated by plasma gaps; at a junction the cells split exactly as the
    phase-separated red-cell fluxes do (docs §6.4). Below: the cells of the widest conjunctival venule in the field,
    seen from above along a stretch of it and in cross-section.
    """)
    return


@app.cell
def _(flow, net, np, plt, scene, show_figure):
    from synth import cells as _cells_module

    _cells = _cells_module.Cells(net, flow, scene.physiology, np.random.default_rng(0))
    _venules = [i for i, e in enumerate(net.edges) if e.kind == "venule"]
    _widest = max(_venules, key=lambda i: net.edges[i].diameter * min(net.edges[i].length, 400))
    _pick = _cells.edge == _widest
    _xyz = _cells.positions()[_pick]
    _s = _cells.s[_pick]
    _r = _cells.rfrac[_pick] * _cells.core[_widest]
    _theta = _cells.theta[_pick]
    _figure, (_a, _b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.6, 1]})
    _near = _s < 300
    _a.scatter(_s[_near], (_r * np.cos(_theta))[_near], s=8, color="#9b1c1c", alpha=0.6)
    _R = _cells.radius[_widest]
    _a.axhline(_R, color="k", lw=0.8)
    _a.axhline(-_R, color="k", lw=0.8)
    _a.set(xlabel="distance along the vessel (µm)", ylabel="across (µm)", title=f"venule {_R * 2:.0f} µm: cells along 300 µm")
    _b.scatter(_r * np.cos(_theta), _r * np.sin(_theta), s=4, color="#9b1c1c", alpha=0.4)
    _b.add_patch(plt.Circle((0, 0), _R, fill=False, color="k"))
    _b.add_patch(plt.Circle((0, 0), _cells.core[_widest], fill=False, color="k", ls=":"))
    _b.set(aspect="equal", title="cross-section (dotted: edge of the cell-free layer)", xlim=(-_R * 1.1, _R * 1.1), ylim=(-_R * 1.1, _R * 1.1))
    show_figure(_figure, "cells")
    print(f"{len(_cells.edge)} red cells in the domain ({_cells.expected.sum():.0f} expected from the tube haematocrits)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The camera

    `simulate` runs the cells through time and renders every frame: the cells absorb by their haemoglobin over a
    broad spectrum (light goes through a vessel to the sclera and back), deeper slabs are blurred by the tissue
    above them and by defocus and lose contrast, the eye moves with the measured fixational motion, the focal plane
    wanders, the illumination flickers at the measured mains alias, each frame averages sub-exposures over the
    exposure time, and the sensor adds the measured shot and read noise, gain, gamma and quantisation (docs §7).
    """)
    return


@app.cell
def _(N_FRAMES, net, scene, simulate, time):
    _start = time.perf_counter()
    _last = [0.0]


    def _progress(done, total):
        if time.perf_counter() - _last[0] > 20 or done == total:
            _last[0] = time.perf_counter()
            print(f"  frame {done}/{total}, {time.perf_counter() - _start:.0f} s")


    result = simulate(scene, n_frames=N_FRAMES, progress=_progress)
    assert len(result.network.edges) == len(net.edges)
    print("Seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in result.seconds.items()))
    return (result,)


@app.cell
def _(REAL_BURST, REAL_CROP, targets):
    real = None
    if REAL_BURST:
        real = targets.load_burst(REAL_BURST, crop=REAL_CROP)
        print(f"Real burst {real.name}: {real.frames.shape[0]} frames of {real.frames.shape[1]} x {real.frames.shape[2]} at {real.fps:.2f} fps")
    else:
        print("No real burst given (LIMBUS_BURST): the comparison shows the synthetic side alone")
    return (real,)


@app.cell
def _(OUTPUT_DIR, media, mo, real, result, scene):
    _synthetic = result.frames.astype("float32")
    if real is not None:
        _h, _w = _synthetic.shape[1:]
        _real = real.frames[:, :_h, :_w]
        _pair = media.side_by_side(_real, _synthetic, od_range=media.OD_RANGE)
    else:
        _pair = media.od_to_gray(media.optical_density(_synthetic))
    _path = media.webp(_pair, OUTPUT_DIR / f"real_and_synthetic_seed{scene.seed}.webp", scene.optics.fps)
    mo.output.append(mo.md(f"Optical density of every frame, both halves on one grey scale "
                           f"({media.OD_RANGE[0]} white to {media.OD_RANGE[1]} black), so equally dark vessels look "
                           f"equally dark; saved to `{_path}`."))
    mo.output.append(mo.image(_path.read_bytes()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Real against synthetic

    `targets.measure` runs on the real and the synthetic burst alike: the same motion estimator, the same
    registration of the longest fixation, the same vessel mask and skeleton, the same spectra. So the table compares
    like with like, including the biases of the estimators. Diameters and lengths are in µm at 3.26 µm/px for both.
    """)
    return


@app.cell
def _(mo, np, real, result, scene, targets):
    synthetic_stats = targets.measure(result.frames, scene.optics.fps, scene.anatomy.um_per_px, label="synthetic")
    real_stats = targets.measure(real.frames, real.fps, scene.anatomy.um_per_px, label="real") if real is not None else None
    comparison = []
    for _name, _value in targets.public(synthetic_stats).items():
        if isinstance(_value, float):
            _r = real_stats.get(_name, np.nan) if real_stats else np.nan
            comparison.append((_name, _r, _value))
    _rows = ["| statistic | real | synthetic | synthetic / real |", "|---|---|---|---|"]
    for _name, _r, _s in comparison:
        _ratio = _s / _r if np.isfinite(_r) and _r != 0 else np.nan
        _rows.append(f"| {_name} | {_r:.4g} | {_s:.4g} | {_ratio:.2f} |")
    mo.md("\n".join(_rows))
    return comparison, real_stats, synthetic_stats


@app.cell
def _(SERIES, np, plt, real_stats, show_figure, synthetic_stats):
    _figure, (_a, _b, _c, _d) = plt.subplots(1, 4, figsize=(19, 3.8))
    for _stats, _color, _label in ((real_stats, SERIES[1], "real"), (synthetic_stats, SERIES[0], "synthetic")):
        if _stats is None:
            continue
        _f, _pv, _pb = _stats["_flicker_power"]
        _a.semilogy(_f[1:], (_pv - _pb)[1:].clip(1e-12), color=_color, label=_label)
        _f, _amp = _stats["_flicker_spectrum"]
        _b.semilogy(_f[1:], _amp[1:] * 100, color=_color, label=_label)
        _c.hist(_stats["_diameters_um"], bins=60, range=(0, 150), histtype="step", density=True, color=_color, label=_label)
        _vessel_od = np.nan_to_num(_stats["_od"])[_stats["_mask"] & _stats["_valid"]]
        _d.hist(_vessel_od, bins=70, range=(-0.05, 0.6), histtype="step", density=True, color=_color, label=_label)
    _a.set(xlabel="frequency (Hz)", ylabel="excess power in vessels", title="red-cell flicker: vessel minus background")
    _b.set(xlabel="frequency (Hz)", ylabel="amplitude (% of the mean)", title="frame-mean brightness spectrum")
    _c.set(xlabel="diameter (µm)", ylabel="density (length-weighted)", title="vessel diameters on the skeleton")
    _d.set(xlabel="optical density", ylabel="density", title="darkness of the vessel pixels")
    for _axis in (_a, _b, _c, _d):
        _axis.legend(fontsize=8)
    show_figure(_figure, "spectra_and_diameters")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · The ground truth

    `write_burst` writes the frames exactly as the LIMBUS recorder does (`frame_NNNNNN.tif`, `manifest.json`,
    `frames.csv`), so `foundation.py` and the flow analysis run on it unchanged, and a `truth/` folder beside them:
    the directed graph with 3D centrelines, diameters, depths, flows, velocities and haematocrits (`graph.json`),
    the motion, focus and per-segment velocities of every frame (`frames.npz`), every red cell's 3D position in
    every frame (`cells.npz`), per-frame maps of the shallowest vessel's id, depth and projected velocity
    (`masks.npz`), and every parameter of the scene (`scene.json`). The `synth.simulate` docstring describes each
    field.
    """)
    return


@app.cell
def _(OUTPUT_DIR, result, scene, write_burst):
    burst_folder = write_burst(result, OUTPUT_DIR / f"burst_synthetic_seed{scene.seed}")
    for _path in sorted(burst_folder.glob("truth/*")) + [burst_folder / "manifest.json"]:
        print(f"{str(_path.relative_to(burst_folder)):24s} {_path.stat().st_size / 1e6:8.2f} MB")
    print(f"{len(list(burst_folder.glob('frame_*.tif')))} frames in {burst_folder}")
    return (burst_folder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Flow detection against the truth

    Run the flow notebook on the synthetic burst (`LIMBUS_BURST=<burst_folder> marimo export html flow.py`), then set
    `SYNTH_FLOW_OUTPUTS` to the `flow_outputs/<burst>` folder it wrote. `validate.compare_flow` registers its
    centrelines onto the truth's frame 0 and compares every measured segment's red-cell velocity with the true
    cycle-averaged velocity along that segment, in px/frame (docs and `synth/validate.py`).
    """)
    return


@app.cell
def _(FLOW_OUTPUTS, Path, SERIES, burst_folder, mo, np, plt, show_figure, validate):
    flow_rows = None
    if FLOW_OUTPUTS and Path(FLOW_OUTPUTS).exists():
        flow_rows, _registration = validate.compare_flow(burst_folder, FLOW_OUTPUTS)
        _summary = validate.summarise(flow_rows)
        mo.output.append(mo.md(f"Registration to the truth: dx {_registration['dx']:.1f}, dy {_registration['dy']:.1f} px "
                               f"(peak {_registration['response']:.2f})\n\n" +
                               "\n".join(f"- {k}: {v:.3g}" if isinstance(v, float) else f"- {k}: {v}" for k, v in _summary.items())))
        _use = [r for r in flow_rows if r["status"] in ("confirmed", "consistent") and np.isfinite(r["v_true"])]
        _figure, _axis = plt.subplots(figsize=(5, 5))
        _t = np.array([r["v_true"] for r in _use])
        _m = np.array([r["v_measured"] for r in _use])
        _axis.scatter(_t, _m, s=12, color=SERIES[0])
        _lim = max(1.0, float(np.nanmax(np.abs(np.r_[_t, _m]))) if len(_t) else 1.0)
        _axis.plot([-_lim, _lim], [-_lim, _lim], color="k", lw=0.8)
        _axis.set(xlabel="true velocity along the segment (px/frame)", ylabel="flow notebook (px/frame)", aspect="equal")
        show_figure(_figure, "flow_against_truth")
    else:
        print("SYNTH_FLOW_OUTPUTS not set: run the flow notebook on the synthetic burst first")
    return (flow_rows,)


@app.cell
def _(notebook_start, time):
    import resource as _resource

    print(f"Notebook ran in {time.perf_counter() - notebook_start:.0f} s, peak memory "
          f"{_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
    return


if __name__ == "__main__":
    app.run()
