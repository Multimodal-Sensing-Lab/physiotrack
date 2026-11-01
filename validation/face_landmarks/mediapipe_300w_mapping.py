"""Mapping between the 300-W 51-point markup and MediaPipe landmarks.

The 300-W 51-point protocol corresponds to landmarks 18-68 of the
standard 68-point markup, excluding the 17-point face boundary.

This mapping is used only for benchmark evaluation. It does not change
the 478 landmarks returned by PhysioTrack/MediaPipe.
"""

MEDIAPIPE_300W_51 = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 1,
    98, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270,
    291, 321, 314, 17, 84, 91,
    78, 81, 13, 311,
    308, 402, 14, 178,
]


def get_mediapipe_300w_51():
    """Return the MediaPipe indices used for the 300-W 51-point protocol."""
    return MEDIAPIPE_300W_51.copy()


if __name__ == "__main__":
    print(
        "Mapped landmarks:",
        len(MEDIAPIPE_300W_51),
    )

    print(
        "Indices:",
        MEDIAPIPE_300W_51,
    )