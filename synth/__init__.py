"""Synthetic conjunctiva: an anatomically constrained 3D microvessel network, its blood flow, its red cells, and
the camera that records them, with the exact ground truth exported alongside every clip.

    from synth import Scene, simulate, write_burst
    result = simulate(Scene(seed=0), n_frames=120)
    write_burst(result, "burst_synthetic_seed0")

Every parameter traces to docs/synthetic_conjunctiva_anatomy.md or to a measurement of the real bursts
(synth.targets); synth.anatomy holds them with their sources.
"""
from .anatomy import Anatomy, Optics, Physiology, Scene
from .simulate import Result, load_graph, simulate, write_burst

__all__ = ["Anatomy", "Optics", "Physiology", "Scene", "Result", "simulate", "write_burst", "load_graph"]
