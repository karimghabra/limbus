"""Offline stabilization of conjunctival microcirculation TIFF bursts.

Registers each burst on its vessel-envelope mask rather than raw intensity,
and scores how stable the burst was and how well stabilization worked.
See METHODS.md for the physics and maths behind every step.

    cd analysis
    python -m stabilize ../recordings                      # every burst, translation
    python -m stabilize ../recordings/burst_... --method nonrigid
"""

__version__ = "0.2.0"
