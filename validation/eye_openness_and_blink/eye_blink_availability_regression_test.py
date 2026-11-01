"""Validate EyeOpenness/BlinkDetector availability in the FaceAnalysis pipeline.

Uses deterministic landmarks and injected face detections; no external data,
model downloads, model weights, or benchmark recalculation are required.
"""

from types import SimpleNamespace

import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.blink import BlinkDetector
from physiotrack.face.eyes import EyeOpenness
from physiotrack.results import Instance, Result


class ControlledDetector:
    def __init__(self, person_id=7):
        self.person_id = person_id

    def predict(self, frame):
        return Result(
            orig_img=frame,
            task="face",
            instances=[Instance(
                id=self.person_id,
                box=np.array([10, 10, 90, 90], dtype=float),
                confidence=0.9,
                cls=0,
                cls_name="face",
            )],
        )


class ControlledLandmarks:
    def __init__(self, points):
        self.points = points

    def predict_face(self, frame, box):
        return self.points


class InjectedEyes:
    def __init__(self, measurements):
        self.measurements = list(measurements)

    def predict(self, landmarks, image_size):
        return dict(self.measurements.pop(0))


def eye_landmarks(left_width_zero=False, right_width_zero=False, vertical=0.1):
    """Create 478 points with controlled inter-corner and eyelid distances."""
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    positions = {
        33: (0.2, 0.5), 133: (0.4, 0.5),
        160: (0.25, 0.5 - vertical / 2),
        144: (0.25, 0.5 + vertical / 2),
        158: (0.35, 0.5 - vertical / 2),
        153: (0.35, 0.5 + vertical / 2),
        362: (0.6, 0.5), 263: (0.8, 0.5),
        385: (0.65, 0.5 - vertical / 2),
        380: (0.65, 0.5 + vertical / 2),
        387: (0.75, 0.5 - vertical / 2),
        373: (0.75, 0.5 + vertical / 2),
    }
    if left_width_zero:
        positions[263] = positions[362]
    if right_width_zero:
        positions[133] = positions[33]
    for index, (x, y) in positions.items():
        points[index].x, points[index].y = x, y
    return points


def pipeline(landmarks, eyes=None, person_id=7):
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=True, quality=False,
        eyes=True, blink=True, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False, blink_threshold=0.22, min_closed_frames=1,
    )
    return FaceAnalysis(
        detector=ControlledDetector(person_id),
        landmarks=ControlledLandmarks(landmarks),
        eyes=EyeOpenness() if eyes is None else eyes,
        blink=BlinkDetector(threshold=0.22, fps=25, min_closed_frames=1),
        config=config,
        fps=25,
        device="cpu",
    )


def features(analysis):
    result = analysis.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(result) == 1
    return result[0].face_features


def check_unavailable(f):
    assert f["landmarks"]["available"] is True
    assert f["eyes"]["available"] is False
    assert f["blink"]["available"] is False
    assert f["blink"]["eye_state"] == "unknown"
    assert f["blink"]["blink"] is False
    assert f["blink"]["blink_duration"] is None
    for key in ("left_openness", "right_openness", "mean_openness"):
        value = f["eyes"][key]
        assert value is None or np.isfinite(value), (key, value)


def run():
    # Both zero-width eyes and a single zero-width eye are unavailable.
    for left, right in ((True, True), (True, False), (False, True)):
        a = pipeline(eye_landmarks(left, right))
        check_unavailable(features(a))
        assert a.blink._states[7]["closed_frames"] == 0

    # Required fields may be missing or non-finite in an injected descriptor.
    invalid = [
        {"left_openness": 0.3, "right_openness": 0.3},
        {"left_openness": 0.3, "right_openness": 0.3, "mean_openness": None},
        {"left_openness": float("nan"), "right_openness": 0.3, "mean_openness": 0.3},
        {"left_openness": 0.3, "right_openness": float("inf"), "mean_openness": 0.3},
        {"left_openness": 0.3, "right_openness": 0.3, "mean_openness": float("-inf")},
        {"left_openness": False, "right_openness": 0.3, "mean_openness": 0.3},
    ]
    for value in invalid:
        a = pipeline(eye_landmarks(), eyes=InjectedEyes([value]))
        check_unavailable(features(a))

    # A missing sample interrupts a closed run without deleting the person ID.
    measurements = [
        {"left_openness": 0.1, "right_openness": 0.1, "mean_openness": 0.1},
        {"left_openness": None, "right_openness": 0.1, "mean_openness": None},
        {"left_openness": 0.4, "right_openness": 0.4, "mean_openness": 0.4},
        {"left_openness": 0.0, "right_openness": 0.0, "mean_openness": 0.0},
        {"left_openness": 0.4, "right_openness": 0.4, "mean_openness": 0.4},
    ]
    a = pipeline(eye_landmarks(), eyes=InjectedEyes(measurements))
    first = features(a)
    missing = features(a)
    reopened = features(a)
    closed_zero = features(a)
    completed = features(a)
    assert first["blink"]["eye_state"] == "closed"
    check_unavailable(missing)
    assert reopened["blink"]["available"] is True
    assert reopened["blink"]["blink"] is False
    assert reopened["blink"]["blink_count"] == 0
    assert closed_zero["eyes"]["available"] is True
    assert closed_zero["blink"]["eye_state"] == "closed"
    assert completed["blink"]["available"] is True
    assert completed["blink"]["blink"] is True
    assert completed["blink"]["blink_count"] == 1

    # A completed event remains in the tracked person's history even if a
    # later eye measurement is unavailable; elapsed video time still counts.
    history = [
        {"left_openness": 0.1, "right_openness": 0.1, "mean_openness": 0.1},
        {"left_openness": 0.4, "right_openness": 0.4, "mean_openness": 0.4},
        {"left_openness": None, "right_openness": 0.4, "mean_openness": None},
        {"left_openness": 0.4, "right_openness": 0.4, "mean_openness": 0.4},
    ]
    a = pipeline(eye_landmarks(), eyes=InjectedEyes(history))
    features(a)
    completed_before_gap = features(a)
    missing_after_event = features(a)
    restored = features(a)
    assert completed_before_gap["blink"]["blink_count"] == 1
    check_unavailable(missing_after_event)
    assert missing_after_event["blink"]["blink_count"] == 1
    assert restored["blink"]["blink_count"] == 1
    assert restored["blink"]["blink"] is False
    assert np.isclose(restored["blink"]["blink_rate"], 60.0 * 25 / 4)

    # Without an identity, a valid instantaneous eye result is still usable,
    # but no temporal blink history is allocated (identity separation).
    a = pipeline(eye_landmarks(), person_id=None)
    untracked = features(a)
    assert untracked["eyes"]["available"] is True
    assert np.isclose(untracked["eyes"]["mean_openness"], 0.5)
    assert untracked["blink"]["available"] is False
    assert a.blink._states == {}

    print("B07 Eye/Blink measurement availability regression: PASS")


if __name__ == "__main__":
    run()
