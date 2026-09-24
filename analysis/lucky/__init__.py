"""Lucky imaging and vessel annotation on stabilized bursts.

Builds on a finished stabilization (python -m stabilize): re-warps each raw
frame with the saved transforms, fuses them band by band so each place's
fine detail comes from the frames that were sharp there, and annotates the
vessels of the plain mean and the fused image with identical constants, so
the two can be compared. See METHODS.md.

    cd analysis
    python -m lucky ../recordings                   # every stabilized burst
"""

__version__ = "0.1.0"
