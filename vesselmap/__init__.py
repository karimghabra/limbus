"""vesselmap: a spline-network model of the vessels in a single still image.

    from vesselmap import load_image, build_map, fit_frame, VesselNetwork

    ref = build_map(load_image("frame.tif"))          # discover the network
    ref.save("map.json")
    G = ref.to_digraph()                              # networkx DiGraph

    net, report = fit_frame(load_image("frame_017.tif"), ref)   # per frame

Structure only: no temporal information, kymographs or velocities are used.
"""
from .image import load_image, prepare
from .network import VesselNetwork
from .fit import MapConfig, FrameFitConfig, build_map, fit_frame, fit_frames
from .consolidate import ConsolidateConfig, consolidate_map

__all__ = ["load_image", "prepare", "VesselNetwork", "MapConfig", "FrameFitConfig",
           "ConsolidateConfig", "build_map", "consolidate_map", "fit_frame", "fit_frames"]
