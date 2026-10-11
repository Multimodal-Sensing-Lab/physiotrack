from types import SimpleNamespace

import numpy as np
import pytest

from physiotrack.face.analysis import FaceAnalysis
from physiotrack.face.blink import BlinkDetector
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.mouth_motion import MouthMovement
from physiotrack.results import Instance, Result


class DummyDetector:
    def predict(self, frame):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=None,
                    box=np.array(
                        [10, 10, 50, 50],
                        dtype=float,
                    ),
                    confidence=0.9,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


class DummyTracker:
    def track(self, frame, detection_result):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=1,
                    box=np.array(
                        [10, 10, 50, 50],
                        dtype=float,
                    ),
                    confidence=0.9,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


class DummyOrientation:
    def predict(self, frame, boxes):
        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    box=np.asarray(
                        boxes[0],
                        dtype=float,
                    ),
                    orientation={
                        "pitch": 1.0,
                        "yaw": 2.0,
                        "roll": 3.0,
                    },
                )
            ],
            task="face",
        )


class DummyLandmarks:
    def predict_face(self, frame, box):
        return [
            SimpleNamespace(
                x=0.5,
                y=0.5,
                z=0.0,
            )
            for _ in range(478)
        ]

    def close(self):
        pass


class DummyQuality:
    def predict(self, frame, faces):
        return [
            {
                "confidence": 0.9,
                "brightness": 0.5,
                "sharpness": 100.0,
                "face_area_ratio": 0.16,
            }
        ]


class DummyEyes:
    def predict(self, landmarks, image_size):
        return {
            "left_openness": 0.3,
            "right_openness": 0.3,
            "mean_openness": 0.3,
        }


class DummyBlink:
    def update(self, openness, person_id=0):
        return {
            "eye_state": "open",
            "blink": False,
            "blink_count": 0,
            "blink_duration": None,
            "blink_rate": 0.0,
        }

    def reset(self):
        pass


class DummyGaze:
    def predict(self, landmarks, image_size):
        return {
            "right_iris_x": 0.5,
            "right_iris_y": 0.0,
            "left_iris_x": 0.5,
            "left_iris_y": 0.0,
            "mean_iris_x": 0.5,
            "mean_iris_y": 0.0,
        }


class DummyGazeEstimator:
    def __init__(self):
        self.initialized = True
        self.was_closed = False
        self.last_boxes = None
        self.last_min_iou = None

    def predict_faces(
        self,
        image,
        boxes,
        min_iou=0.10,
    ):
        self.last_boxes = np.asarray(
            boxes,
            dtype=float,
        ).copy()

        self.last_min_iou = float(
            min_iou
        )

        return [
            {
                "available": True,
                "gaze_vector": [
                    -0.1,
                    0.2,
                    -0.97,
                ],
                "pitch": 11.5,
                "yaw": -5.9,
                "association_iou": 0.85,
            }
            for _ in self.last_boxes
        ]

    def close(self):
        self.was_closed = True
        self.initialized = False


class DummyMouth:
    def predict(self, landmarks, image_size):
        return {
            "mouth_openness": 0.2,
            "mouth_width": 0.4,
            "mouth_height": 0.08,
        }


class DummyMouthMotion:
    def update(self, openness, person_id=0):
        return {
            "mouth_movement": 0.0,
            "mouth_velocity": 0.0,
        }

    def reset(self):
        pass


class DummyEmotion:
    def predict(self, crop):
        return {
            "emotion": "Neutral",
            "confidence": 0.8,
            "scores": {
                "Neutral": 0.8,
            },
        }


class DummyRegions:
    def predict(self, frame, boxes=None):
        mask = np.ones(
            (40, 40),
            dtype=bool,
        )

        return {
            "result": None,
            "faces": [
                {
                    "box": np.array(
                        [10, 10, 50, 50],
                        dtype=int,
                    ),
                    "regions": {
                        "skin": mask,
                    },
                }
            ],
        }


class DummyTemporal:
    def update(self, instance):
        return {
            "person_id": instance.id,
            "window_frames": 1,
            "window_sec": 0.04,
            "head_pose": {
                "yaw": {
                    "mean": 2.0,
                    "std": 0.0,
                    "min": 2.0,
                    "max": 2.0,
                },
                "pitch": {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                },
                "roll": {
                    "mean": 3.0,
                    "std": 0.0,
                    "min": 3.0,
                    "max": 3.0,
                },
            },
            "eyes": {},
            "gaze": {},
            "mouth": {},
            "quality": {},
            "blink": {
                "events": 0,
            },
            "emotion": {
                "dominant": "Neutral",
            },
        }

    def reset(self):
        pass


def make_pipeline(
    config=None,
    gaze_estimation=None,
):
    if gaze_estimation is None:
        gaze_estimation = (
            DummyGazeEstimator()
        )

    return FaceAnalysis(
        detector=DummyDetector(),
        tracker=DummyTracker(),
        orientation=DummyOrientation(),
        landmarks=DummyLandmarks(),
        quality=DummyQuality(),
        eyes=DummyEyes(),
        blink=DummyBlink(),
        gaze=DummyGaze(),
        gaze_estimation=gaze_estimation,
        mouth=DummyMouth(),
        mouth_motion=DummyMouthMotion(),
        emotion=DummyEmotion(),
        regions=DummyRegions(),
        temporal=DummyTemporal(),
        config=config,
        fps=25,
        device="cpu",
    )


def test_analysis_returns_structured_face_instance():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    instance = result[0]

    assert instance.id == 1

    assert instance.orientation == {
        "pitch": 1.0,
        "yaw": 2.0,
        "roll": 3.0,
    }

    assert instance.face_features is not None

    assert set(
        instance.face_features.keys()
    ) == {
        "landmarks",
        "quality",
        "eyes",
        "blink",
        "gaze",
        "mouth",
        "mouth_motion",
        "emotion",
        "regions",
        "temporal",
    }


def test_analysis_preserves_feature_values():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    features = result[0].face_features

    assert features["landmarks"]["available"] is True
    assert features["landmarks"]["count"] == 478

    assert features["quality"]["available"] is True

    assert features["eyes"]["available"] is True
    assert features["eyes"]["mean_openness"] == 0.3

    assert features["blink"]["available"] is True

    assert features["gaze"]["available"] is True
    assert features["gaze"]["mean_iris_x"] == 0.5

    assert features["mouth"]["available"] is True
    assert features["mouth"]["mouth_openness"] == 0.2

    assert features["mouth_motion"]["available"] is True

    assert features["emotion"]["available"] is True
    assert features["emotion"]["emotion"] == "Neutral"

    assert features["regions"]["available"] is True
    assert features["regions"]["skin_pixel_count"] == 1600

    assert features["temporal"]["available"] is True

    assert (
        features["temporal"]["summary"]["person_id"]
        == 1
    )


def test_analysis_returns_empty_result_when_tracker_has_no_faces():
    class EmptyTracker:
        def track(self, frame, detection_result):
            return Result(
                orig_img=frame,
                instances=[],
                task="face",
            )

    pipeline = make_pipeline()

    pipeline.tracker = EmptyTracker()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 0
    assert result.task == "face"


def test_reset_temporal_state_calls_temporal_components():
    class FlagBlink(DummyBlink):
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    class FlagMouth(DummyMouthMotion):
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    class FlagTemporal(DummyTemporal):
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    pipeline = make_pipeline()

    pipeline.blink = FlagBlink()
    pipeline.mouth_motion = FlagMouth()
    pipeline.temporal = FlagTemporal()

    pipeline.reset_temporal_state()

    assert pipeline.blink.was_reset is True
    assert pipeline.mouth_motion.was_reset is True
    assert pipeline.temporal.was_reset is True


def test_config_can_disable_optional_modules():
    config = FaceAnalysisConfig(
        emotion=False,
        regions=False,
    )

    pipeline = make_pipeline(
        config=config
    )

    assert pipeline.emotion is None
    assert pipeline.regions is None

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    features = result[0].face_features

    assert features["emotion"]["available"] is False
    assert features["emotion"]["emotion"] is None

    assert features["regions"]["available"] is False
    assert features["regions"]["classes"] == []


def test_config_can_disable_landmark_dependent_modules():
    config = FaceAnalysisConfig(
        landmarks=False,
        eyes=False,
        blink=False,
        gaze=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    pipeline = make_pipeline(
        config=config
    )

    assert pipeline.landmarks is None
    assert pipeline.eyes is None
    assert pipeline.blink is None
    assert pipeline.gaze is None
    assert pipeline.mouth is None
    assert pipeline.mouth_motion is None

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(frame)

    assert len(result) == 1

    features = result[0].face_features

    assert features["landmarks"]["available"] is False
    assert features["landmarks"]["count"] == 0

    assert features["eyes"]["available"] is False
    assert features["blink"]["available"] is False
    assert features["gaze"]["available"] is False
    assert features["mouth"]["available"] is False
    assert features["mouth_motion"]["available"] is False
    assert features["emotion"]["available"] is False
    assert features["regions"]["available"] is False
    assert features["temporal"]["available"] is False


def test_gaze_estimation_is_absent_when_disabled():
    pipeline = make_pipeline()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    features = result[
        0
    ].face_features

    assert (
        "gaze_estimation"
        not in features
    )

    assert (
        pipeline.gaze_estimation
        is None
    )


def test_gaze_estimation_adds_structured_features_when_enabled():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    assert len(result) == 1

    features = result[
        0
    ].face_features

    assert (
        "gaze_estimation"
        in features
    )

    gaze_estimation = features[
        "gaze_estimation"
    ]

    assert (
        gaze_estimation[
            "available"
        ]
        is True
    )

    assert gaze_estimation[
        "gaze_vector"
    ] == [
        -0.1,
        0.2,
        -0.97,
    ]

    assert (
        gaze_estimation[
            "pitch"
        ]
        == 11.5
    )

    assert (
        gaze_estimation[
            "yaw"
        ]
        == -5.9
    )

    assert (
        gaze_estimation[
            "association_iou"
        ]
        == 0.85
    )


def test_gaze_estimation_receives_pipeline_face_boxes():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    pipeline.predict(
        frame
    )

    np.testing.assert_allclose(
        dummy_gaze_estimator.last_boxes,
        np.array(
            [
                [
                    10.0,
                    10.0,
                    50.0,
                    50.0,
                ]
            ]
        ),
    )

    assert (
        dummy_gaze_estimator.last_min_iou
        == pytest.approx(
            config.gaze_estimation_min_iou
        )
    )


def test_gaze_estimation_does_not_require_landmarks():
    config = FaceAnalysisConfig(
        landmarks=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=True,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    assert pipeline.landmarks is None
    assert pipeline.gaze is None

    assert (
        pipeline.gaze_estimation
        is dummy_gaze_estimator
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = pipeline.predict(
        frame
    )

    features = result[
        0
    ].face_features

    assert (
        features[
            "landmarks"
        ][
            "available"
        ]
        is False
    )

    assert (
        features[
            "gaze"
        ][
            "available"
        ]
        is False
    )

    assert (
        features[
            "gaze_estimation"
        ][
            "available"
        ]
        is True
    )


def test_close_releases_gaze_estimator():
    config = FaceAnalysisConfig(
        gaze_estimation=True,
    )

    dummy_gaze_estimator = (
        DummyGazeEstimator()
    )

    pipeline = make_pipeline(
        config=config,
        gaze_estimation=(
            dummy_gaze_estimator
        ),
    )

    assert (
        dummy_gaze_estimator.was_closed
        is False
    )

    pipeline.close()

    assert (
        dummy_gaze_estimator.was_closed
        is True
    )

    assert (
        dummy_gaze_estimator.initialized
        is False
    )


def test_analysis_rejects_invalid_config_type():
    with pytest.raises(
        TypeError,
        match="config must be a FaceAnalysisConfig instance",
    ):
        FaceAnalysis(
            detector=DummyDetector(),
            config={},
            fps=25,
        )

def test_analysis_default_blink_parameters_match_config():
    pipeline = FaceAnalysis(
        detector=DummyDetector(),
        tracker=DummyTracker(),
        orientation=DummyOrientation(),
        landmarks=DummyLandmarks(),
        quality=DummyQuality(),
        eyes=DummyEyes(),
        blink=None,
        gaze=DummyGaze(),
        mouth=DummyMouth(),
        mouth_motion=DummyMouthMotion(),
        emotion=DummyEmotion(),
        regions=DummyRegions(),
        temporal=DummyTemporal(),
        fps=25,
        device="cpu",
    )

    assert pipeline.config.blink_threshold == pytest.approx(
        0.22
    )
    assert pipeline.config.min_closed_frames == 3

    assert pipeline.blink.threshold == pytest.approx(
        0.22
    )
    assert pipeline.blink.min_closed_frames == 3

def test_missing_tracked_frame_breaks_temporal_continuity():
    class GapTracker:
        def __init__(self):
            self.calls = 0

        def track(self, frame, detection_result):
            self.calls += 1

            if self.calls == 2:
                return Result(
                    orig_img=frame,
                    instances=[],
                    task="face",
                )

            return Result(
                orig_img=frame,
                instances=[
                    Instance(
                        id=1,
                        box=np.array(
                            [10, 10, 50, 50],
                            dtype=float,
                        ),
                        confidence=0.9,
                        cls=0,
                        cls_name="face",
                    )
                ],
                task="face",
            )

    class SequenceMouth(DummyMouth):
        def __init__(self):
            self.values = iter(
                [
                    0.1,
                    0.4,
                ]
            )

        def predict(self, landmarks, image_size=None):
            openness = next(
                self.values
            )

            return {
                "mouth_openness": openness,
                "mouth_width": 0.4,
                "mouth_height": (
                    openness * 0.4
                ),
            }

    class RecordingTemporal(DummyTemporal):
        def __init__(self):
            self.reset_person_ids = []

        def reset(self, person_id=None):
            self.reset_person_ids.append(
                person_id
            )

    blink = BlinkDetector(
        threshold=0.22,
        fps=25,
        min_closed_frames=1,
    )

    # Establish one completed blink before the tracked-frame gap.
    blink.update(
        0.1,
        person_id=1,
    )
    completed = blink.update(
        0.3,
        person_id=1,
    )

    assert completed["blink_count"] == 1

    mouth_motion = MouthMovement(
        fps=25
    )

    temporal = RecordingTemporal()

    pipeline = make_pipeline()

    pipeline.tracker = GapTracker()
    pipeline.blink = blink
    pipeline.mouth = SequenceMouth()
    pipeline.mouth_motion = mouth_motion
    pipeline.temporal = temporal

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    first = pipeline.predict(
        frame
    )

    missing = pipeline.predict(
        frame
    )

    returned = pipeline.predict(
        frame
    )

    assert len(first) == 1
    assert len(missing) == 0
    assert len(returned) == 1

    returned_features = (
        returned[0].face_features
    )

    # The gap must not create a blink or erase the prior cumulative count.
    assert returned_features[
        "blink"
    ][
        "blink"
    ] is False

    assert returned_features[
        "blink"
    ][
        "blink_count"
    ] == 1

    # Mouth movement after the gap must start a new contiguous segment.
    assert returned_features[
        "mouth_motion"
    ][
        "mouth_movement"
    ] == 0.0

    assert returned_features[
        "mouth_motion"
    ][
        "mouth_velocity"
    ] == 0.0

    # Temporal aggregation must not bridge the unobserved frame.
    assert temporal.reset_person_ids == [
        1
    ]

