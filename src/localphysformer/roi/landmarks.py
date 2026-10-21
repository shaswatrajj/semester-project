"""MediaPipe Face Landmarker indices for the three rPPG regions.

Indices follow the 468-point Face Mesh topology. Left/right are the
subject's left and right (not the camera's).
"""

from localphysformer import ROI_FOREHEAD, ROI_LEFT_CHEEK, ROI_RIGHT_CHEEK

# Hairline to just above the brows (convex hull of these points).
FOREHEAD_LANDMARKS = (
    10,
    67,
    69,
    104,
    108,
    151,
    337,
    338,
    297,
    299,
)

# Subject's left cheek (viewer's right).
LEFT_CHEEK_LANDMARKS = (
    266,
    280,
    330,
    346,
    347,
    352,
    376,
    411,
    425,
    426,
)

# Subject's right cheek (viewer's left).
RIGHT_CHEEK_LANDMARKS = (
    36,
    50,
    101,
    117,
    118,
    123,
    147,
    187,
    205,
    206,
)

ROI_LANDMARKS = {
    ROI_FOREHEAD: FOREHEAD_LANDMARKS,
    ROI_LEFT_CHEEK: LEFT_CHEEK_LANDMARKS,
    ROI_RIGHT_CHEEK: RIGHT_CHEEK_LANDMARKS,
}
