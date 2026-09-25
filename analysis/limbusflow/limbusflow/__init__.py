"""limbusflow - vessel annotation, morphometry and RBC velocimetry for limbal / scleral video.

Modules (in pipeline order)
    io           load the reference image and a camera burst
    register     SIFT + RANSAC + ECC registration of every frame to the reference
    perfusion    temporal flicker map of the registered video (functional detection of thin vessels)
    vesselness   flat-field, multi-scale Hessian (Frangi) enhancement, segmentation, skeleton
    graph        skeleton -> vessel graph (spurs, junctions, crossings, borders)
    spline       smoothing B-spline centre-lines, arc length, Frenet frame, curvature
    morphometry  diameter profiles, vessel masks, tortuosity
    velocity     kymographs; LSPIV / structure tensor / time-of-flight; multi-scale (slow flow)
    network      end-to-end assembly and the directed flow graph (+ export)
    viz          figures, vessel dashboard, interactive explorer
    validate     synthetic ground-truth phantoms and validation videos
"""
__version__ = "0.1.0"
