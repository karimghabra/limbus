"""Vessel identification on stabilized bursts.

The averaged stabilized frame is the input: averaging over a burst removes
the flicker and noise of single frames and leaves a clean picture of the
network, with no velocity information needed. detect() returns vessels with
stable ids; labels() paints them into an image that the stabilization fields
carry into every raw frame of the burst.
"""
from .network import NetConfig, centrelines, detect, group, labels  # noqa: F401

__all__ = ["NetConfig", "detect", "group", "labels", "centrelines"]
