from types import SimpleNamespace

import numpy as np
import pytest

from physiotrack.face.blink import BlinkDetector
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.eyes import EyeOpenness
from physiotrack.face.gaze import GazeDescriptor
from physiotrack.face.mouth import MouthOpenness
from physiotrack.face.mouth_motion import MouthMovement
from physiotrack.face.quality import FaceQuality
from physiotrack.results import Instance


def make_landmarks(count=478):
    return [
        SimpleNamespace(
            x=0.5,
            y=0.5,
            z=0.0,
        )
        for _ in range(count)
    ]


def test_eye_openness_returns_expected_ratio():
    landmarks = make_landmarks()

    # Right eye
    landmarks[33].x = 0.2
    landmarks[33].y = 0.5

    landmarks[133].x = 0.4
    landmarks[133].y = 0.5

    landmarks[160].x = 0.25
    landmarks[160].y = 0.45

    landmarks[144].x = 0.25
    landmarks[144].y = 0.55

    landmarks[158].x = 0.35
    landmarks[158].y = 0.45

    landmarks[153].x = 0.35
    landmarks[153].y = 0.55

    # Left eye
    landmarks[362].x = 0.6
    landmarks[362].y = 0.5

    landmarks[263].x = 0.8
    landmarks[263].y = 0.5

    landmarks[385].x = 0.65
    landmarks[385].y = 0.45

    landmarks[380].x = 0.65
    landmarks[380].y = 0.55

    landmarks[387].x = 0.75
    landmarks[387].y = 0.45

    landmarks[373].x = 0.75
    landmarks[373].y = 0.55

    result = EyeOpenness().predict(
        landmarks,
        image_size=(100, 100),
    )

    assert np.isclose(result["right_openness"], 0.5)
    assert np.isclose(result["left_openness"], 0.5)
    assert np.isclose(result["mean_openness"], 0.5)


def test_blink_detector_counts_closed_sequence():
    detector = BlinkDetector(
        threshold=0.2,
        fps=10,
        min_closed_frames=2,
    )

    detector.update(
        openness=0.1,
        person_id=1,
    )

    detector.update(
        openness=0.1,
        person_id=1,
    )

    result = detector.update(
        openness=0.4,
        person_id=1,
    )

    assert result["blink"] is True
    assert result["blink_count"] == 1

    assert np.isclose(
        result["blink_duration"],
        0.2,
    )


def test_blink_state_is_independent_per_person():
    detector = BlinkDetector(
        threshold=0.2,
        fps=10,
        min_closed_frames=2,
    )

    detector.update(
        0.1,
        person_id=1,
    )

    detector.update(
        0.1,
        person_id=1,
    )

    person_2 = detector.update(
        0.4,
        person_id=2,
    )

    assert person_2["blink"] is False
    assert person_2["blink_count"] == 0


def test_missing_eye_value_does_not_bridge_blink():
    detector = BlinkDetector(
        threshold=0.2,
        fps=10,
        min_closed_frames=2,
    )

    detector.update(
        0.1,
        person_id=1,
    )

    detector.update(
        None,
        person_id=1,
    )

    detector.update(
        0.1,
        person_id=1,
    )

    result = detector.update(
        0.4,
        person_id=1,
    )

    assert result["blink"] is False
    assert result["blink_count"] == 0


def test_mouth_openness_ratio():
    landmarks = make_landmarks()

    landmarks[61].x = 0.3
    landmarks[61].y = 0.5

    landmarks[291].x = 0.7
    landmarks[291].y = 0.5

    landmarks[13].x = 0.5
    landmarks[13].y = 0.45

    landmarks[14].x = 0.5
    landmarks[14].y = 0.55

    result = MouthOpenness().predict(
        landmarks,
        image_size=(100, 100),
    )

    assert np.isclose(
        result["mouth_width"],
        0.4,
    )

    assert np.isclose(
        result["mouth_height"],
        0.1,
    )

    assert np.isclose(
        result["mouth_openness"],
        0.25,
    )


def test_mouth_movement_uses_frame_interval():
    movement = MouthMovement(
        fps=10,
    )

    first = movement.update(
        openness=0.1,
        person_id=1,
    )

    second = movement.update(
        openness=0.3,
        person_id=1,
    )

    assert first["mouth_movement"] == 0.0
    assert first["mouth_velocity"] == 0.0

    assert np.isclose(
        second["mouth_movement"],
        0.2,
    )

    assert np.isclose(
        second["mouth_velocity"],
        2.0,
    )


def test_missing_mouth_value_resets_previous_state():
    movement = MouthMovement(
        fps=10,
    )

    movement.update(
        openness=0.1,
        person_id=1,
    )

    movement.update(
        openness=None,
        person_id=1,
    )

    result = movement.update(
        openness=0.4,
        person_id=1,
    )

    assert result["mouth_movement"] == 0.0
    assert result["mouth_velocity"] == 0.0


def test_gaze_descriptor_uses_both_irises():
    landmarks = make_landmarks()

    # Right eye: corner 33 -> 133, iris 468
    landmarks[33].x = 0.2
    landmarks[33].y = 0.5

    landmarks[133].x = 0.4
    landmarks[133].y = 0.5

    landmarks[468].x = 0.3
    landmarks[468].y = 0.5

    # Left eye: corner 362 -> 263, iris 473
    landmarks[362].x = 0.6
    landmarks[362].y = 0.5

    landmarks[263].x = 0.8
    landmarks[263].y = 0.5

    landmarks[473].x = 0.7
    landmarks[473].y = 0.5

    result = GazeDescriptor().predict(
        landmarks,
        image_size=(100, 100),
    )

    assert result["mean_iris_x"] is not None
    assert result["mean_iris_y"] is not None

    assert np.isclose(
        result["mean_iris_x"],
        0.5,
    )


def test_eye_openness_is_invariant_to_frame_aspect_ratio():
    landmarks_square = make_landmarks()
    landmarks_wide = make_landmarks()

    points = {
        33: (20.0, 50.0),
        133: (40.0, 50.0),
        160: (25.0, 45.0),
        144: (25.0, 55.0),
        158: (35.0, 45.0),
        153: (35.0, 55.0),
        362: (60.0, 50.0),
        263: (80.0, 50.0),
        385: (65.0, 45.0),
        380: (65.0, 55.0),
        387: (75.0, 45.0),
        373: (75.0, 55.0),
    }

    for index, (x, y) in points.items():
        landmarks_square[index].x = x / 100.0
        landmarks_square[index].y = y / 100.0
        landmarks_wide[index].x = x / 200.0
        landmarks_wide[index].y = y / 100.0

    square = EyeOpenness().predict(
        landmarks_square,
        image_size=(100, 100),
    )
    wide = EyeOpenness().predict(
        landmarks_wide,
        image_size=(200, 100),
    )

    assert np.isclose(
        square["mean_openness"],
        wide["mean_openness"],
    )


def test_mouth_openness_is_invariant_to_frame_aspect_ratio():
    landmarks_square = make_landmarks()
    landmarks_wide = make_landmarks()

    points = {
        61: (30.0, 50.0),
        291: (70.0, 50.0),
        13: (50.0, 45.0),
        14: (50.0, 55.0),
    }

    for index, (x, y) in points.items():
        landmarks_square[index].x = x / 100.0
        landmarks_square[index].y = y / 100.0
        landmarks_wide[index].x = x / 200.0
        landmarks_wide[index].y = y / 100.0

    square = MouthOpenness().predict(
        landmarks_square,
        image_size=(100, 100),
    )
    wide = MouthOpenness().predict(
        landmarks_wide,
        image_size=(200, 100),
    )

    assert np.isclose(
        square["mouth_openness"],
        wide["mouth_openness"],
    )


def test_gaze_descriptor_is_invariant_to_frame_aspect_ratio():
    landmarks_square = make_landmarks()
    landmarks_wide = make_landmarks()

    points = {
        33: (20.0, 50.0),
        133: (40.0, 50.0),
        468: (30.0, 45.0),
        362: (60.0, 50.0),
        263: (80.0, 50.0),
        473: (70.0, 45.0),
    }

    for index, (x, y) in points.items():
        landmarks_square[index].x = x / 100.0
        landmarks_square[index].y = y / 100.0
        landmarks_wide[index].x = x / 200.0
        landmarks_wide[index].y = y / 100.0

    square = GazeDescriptor().predict(
        landmarks_square,
        image_size=(100, 100),
    )
    wide = GazeDescriptor().predict(
        landmarks_wide,
        image_size=(200, 100),
    )

    assert np.isclose(
        square["mean_iris_x"],
        wide["mean_iris_x"],
    )
    assert np.isclose(
        square["mean_iris_y"],
        wide["mean_iris_y"],
    )


def test_face_quality_returns_expected_fields():
    frame = np.full(
        (100, 100, 3),
        128,
        dtype=np.uint8,
    )

    face = Instance(
        id=1,
        box=np.array(
            [20, 20, 80, 80],
            dtype=float,
        ),
        confidence=0.9,
        cls=0,
    )

    result = FaceQuality().predict(
        frame,
        [face],
    )

    assert len(result) == 1

    quality = result[0]

    assert np.isclose(
        quality["confidence"],
        0.9,
    )

    assert np.isclose(
        quality["brightness"],
        128 / 255.0,
    )

    assert quality["sharpness"] >= 0.0

    assert np.isclose(
        quality["face_area_ratio"],
        0.36,
    )


def test_face_analysis_config_defaults_are_valid():
    config = FaceAnalysisConfig()

    config.validate()

    assert np.isclose(
        config.blink_threshold,
        0.22,
    )

    assert config.min_closed_frames == 3


def test_face_analysis_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        FaceAnalysisConfig(
            blink_threshold=0.0,
        ).validate()

    with pytest.raises(ValueError):
        FaceAnalysisConfig(
            min_closed_frames=0,
        ).validate()

    with pytest.raises(ValueError):
        FaceAnalysisConfig(
            temporal_window_sec=0.0,
        ).validate()

    with pytest.raises(ValueError):
        FaceAnalysisConfig(
            tracker_type="unknown",
        ).validate()