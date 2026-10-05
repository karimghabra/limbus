"""vesselmap: every vessel as a fitted spline (vesselmap/README.md).

Runs vesselmap's discovery on the stabilized mean: each vessel is a spline
with its own width, blur and contrast, found coarse to fine and kept only if
it explains the image. The map is written to this segmenter's folder in
vesselmap's own formats too (map.json, the interactive map_digraph.html,
overlays, map_edges.csv), so it can be opened, refined or consolidated with
`python -m vesselmap` later.

Slow: about an hour for a full 1920x1200 frame on the CPU (65 min for
burst 15-50-52 on a 6-core Ryzen 5 5600; about 2 min for a 1920x100 strip).
Needs the packages in requirements-vesselmap.txt.
"""
import os
import sys
import warnings

LABEL = "vesselmap (spline network)"
DESCRIPTION = ("vesselmap's spline network, fitted to the stabilized mean: centrelines, "
               "widths, branch points and crossings. Thorough and slow: about an hour "
               "for a full frame on the CPU. Needs requirements-vesselmap.txt.")
VERSION = "1"

# vesselmap.MapConfig settings to change, e.g. {"reps_per_band": 2,
# "penalty_scale": 3.0}: about a third faster, at slightly lower recall
MAP_SETTINGS = {}
# further vesselmap steps, in order: "refine" (small and side-by-side
# vessels), "faint" (faint vessels traced along their length),
# "consolidate" (one spline per vessel instead of one per segment)
STEPS = ()
# vesselmap's own outputs (HTML viewer, overlays, graph files): minutes more
WRITE_VESSELMAP_OUTPUTS = True

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def segment(inputs):
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    # as vesselmap's own command line does: PyTorch warns about every sparse
    # tensor the renderer makes
    warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
    from vesselmap import MapConfig, build_map, load_image, prepare

    image = load_image(inputs.mean_path)          # NaN (no data) stays invalid
    prepared = prepare(image)
    cfg = MapConfig(verbose=True)
    for key, value in MAP_SETTINGS.items():
        setattr(cfg, key, value)
    inputs.log(f"vesselmap on the {inputs.method} mean, {image.shape[1]}x{image.shape[0]}")
    net = build_map(image, cfg, prepared=prepared)
    for step in STEPS:
        inputs.log(f"vesselmap {step} ...")
        if step == "refine":
            from vesselmap.refine import RefineConfig, refine_map
            refine_map(image, net, cfg, RefineConfig(verbose=True), prepared=prepared)
        elif step == "faint":
            from vesselmap import FaintConfig, add_faint_tier
            add_faint_tier(image, net, FaintConfig(verbose=True), prepared=prepared, log=inputs.log)
        elif step == "consolidate":
            from vesselmap import ConsolidateConfig, consolidate_map
            net = consolidate_map(image, net, cfg, ConsolidateConfig(verbose=True), prepared=prepared)
        else:
            raise ValueError(f"unknown vesselmap step {step!r}")
    net.meta["source_image"] = inputs.mean_path
    net.save(os.path.join(inputs.out_dir, "map.json"))
    if WRITE_VESSELMAP_OUTPUTS:
        from vesselmap.__main__ import write_outputs
        write_outputs(net, image, prepared, inputs.out_dir)
    return to_overlay(net, inputs.overlay())


def to_overlay(net, ov):
    """A vesselmap VesselNetwork as an overlay: each segment's centreline and
    width, branch points and junctions (degree 3 and 4+), and crossings."""
    for eid, e in net.edges.items():
        smp = net.sample(eid, 1.0)
        ov.add_vessel(smp["xy"], radius=smp["r"], id=eid, group=e.info.get("vessel"),
                      tier=e.info.get("tier", "mapped"), length_px=round(float(smp["s_arc"][-1]), 1),
                      diameter_px=round(float(2 * smp["r"].mean()), 2),
                      blur_px=round(float(smp["s"].mean()), 2),
                      contrast=round(float(smp["a"].mean()), 3))
    degree = net.degrees()
    for nid, node in net.nodes.items():
        kind = net.node_kind(nid, degree.get(nid, 0))
        if kind in ("bifurcation", "junction"):
            ov.add_junction(node.x, node.y, kind,
                            [eid for eid, e in net.edges.items() if nid in (e.u, e.v)])
    for c in net.crossings():
        ov.add_junction(c["x"], c["y"], "crossing", c["edges"])
    s = net.summary()
    ov.summary = (f"{s['n_edges']} vessel segments, {s['total_length_px']:.0f} px of centreline")
    return ov
