"""Bulk stabilization of conjunctival microcirculation TIFF bursts.

Registers each burst on its vessel-envelope mask rather than raw intensity,
and scores how stable the burst was and how well stabilization worked.
See METHODS.md for the physics and maths behind every step.

    python -m stabilize "E:\\Conjunctiva Code\\limbus\\recordings" --out E:\\Analysis\\stabilization
"""

__version__ = "0.1.0"
