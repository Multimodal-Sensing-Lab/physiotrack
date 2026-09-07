from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

DATASET_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "MPEBlink2"
    / "mpeblink2.0"
)

RESULTS_DIR = SCRIPT_DIR / "results"
COMPONENT_RESULTS_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_FILENAME = (
    "eye_openness_blink_component_results.csv"
)

RESULTS_PART_PREFIX = (
    "eye_openness_blink_component_results_part"
)

GIT_SAFE_MAX_FILE_SIZE_MIB = 90.0
GIT_SAFE_MAX_FILE_SIZE_BYTES = int(
    GIT_SAFE_MAX_FILE_SIZE_MIB
    * 1024
    * 1024
)

SUMMARY_FILENAME = (
    "eye_openness_blink_component_summary.json"
)

TEST_SPLIT = "test"

EXPECTED_TEST_VIDEOS = 212
EXPECTED_TEST_ANNOTATION_FRAMES = 219706
EXPECTED_TEST_PERSON_SEQUENCES = 687
EXPECTED_TEST_PERSON_FRAME_SAMPLES = 596209
EXPECTED_TEST_VALID_BBOX = 495858
EXPECTED_TEST_MISSING_BBOX = 100341
EXPECTED_TEST_INVALID_BBOX = 10

BLINK_THRESHOLD = 0.22
MIN_CLOSED_FRAMES = 3

CSV_FIELDS = [
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "person_id",
    "fps",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "landmarks_available",
    "landmark_count",
    "eyes_available",
    "left_openness",
    "right_openness",
    "mean_openness",
    "blink_available",
    "eye_state",
    "blink",
    "blink_count",
    "blink_duration",
    "blink_rate",
    "status",
    "failure_reason",
]


if not (
    REPO_ROOT
    / "validation"
).is_dir():
    raise RuntimeError(
        "Could not resolve the PhysioTrack repository root from "
        f"the validation script location: {SCRIPT_DIR}"
    )

if not (
    SRC_DIR
    / "physiotrack"
).is_dir():
    raise RuntimeError(
        "Could not resolve the PhysioTrack source package from "
        f"the repository root: {REPO_ROOT}"
    )

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.blink import BlinkDetector
from physiotrack.results import Instance, Result


class ControlledFaceDetector:
    """Return MPEBlink ground-truth face boxes to FaceAnalysis."""

    def __init__(self) -> None:
        self.instances: list[Instance] = []

    def set_faces(
        self,
        faces: list[
            tuple[
                str,
                np.ndarray,
            ]
        ],
    ) -> None:
        """Set controlled face instances for the next frame."""
        self.instances = [
            Instance(
                id=person_id,
                box=np.asarray(
                    box,
                    dtype=float,
                ),
                confidence=1.0,
                cls=0,
                cls_name="face",
            )
            for person_id, box in faces
        ]

    def predict(
        self,
        frame: np.ndarray,
    ) -> Result:
        """Return the currently configured face boxes."""
        return Result(
            orig_img=frame,
            instances=list(
                self.instances
            ),
            task="face",
        )


def finite_numeric(
    value: Any,
) -> bool:
    """Return True for finite real numerical values."""
    if value is None or isinstance(
        value,
        bool,
    ):
        return False

    try:
        return bool(
            np.isfinite(
                float(
                    value
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def sorted_video_dirs() -> list[Path]:
    """Return MPEBlink test video directories in numeric order."""
    split_root = (
        DATASET_ROOT
        / TEST_SPLIT
    )

    if not split_root.is_dir():
        raise FileNotFoundError(
            "MPEBlink 2.0 test split was not found:\n"
            f"{split_root}"
        )

    video_dirs = sorted(
        [
            path
            for path in split_root.iterdir()
            if path.is_dir()
        ],
        key=lambda path: int(
            path.name
        ),
    )

    return video_dirs


def load_annotation(
    video_dir: Path,
) -> tuple[
    Path,
    dict[str, Any],
    int,
    list[str],
]:
    """Load one MPEBlink video and annotation package."""
    annotation_path = (
        video_dir
        / "annotation_WFLW.json"
    )

    video_path = (
        video_dir
        / "video.mp4"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        annotation = json.load(
            file
        )

    if "length" not in annotation:
        raise RuntimeError(
            f"Missing length in: {annotation_path}"
        )

    expected_frames = int(
        annotation[
            "length"
        ]
    )

    person_keys = sorted(
        [
            key
            for key, value in annotation.items()
            if (
                key.startswith(
                    "person"
                )
                and isinstance(
                    value,
                    dict,
                )
            )
        ]
    )

    if not person_keys:
        raise RuntimeError(
            f"No person annotations found in: {annotation_path}"
        )

    for person_key in person_keys:
        person = annotation[
            person_key
        ]

        if "bbox" not in person:
            raise RuntimeError(
                f"Missing bbox for {person_key} in: {annotation_path}"
            )

        if len(
            person[
                "bbox"
            ]
        ) != expected_frames:
            raise RuntimeError(
                "Bounding-box length mismatch for "
                f"{person_key} in: {annotation_path}"
            )

    return (
        video_path,
        annotation,
        expected_frames,
        person_keys,
    )


def normalize_box(
    bbox: Any,
    image_width: int,
    image_height: int,
) -> tuple[
    np.ndarray | None,
    str | None,
    str | None,
]:
    """Convert one MPEBlink xywh box to clipped xyxy coordinates."""
    if bbox is None:
        return (
            None,
            "MISSING_BOX",
            "Ground-truth bounding box is missing.",
        )

    if (
        not isinstance(
            bbox,
            (list, tuple),
        )
        or len(
            bbox
        ) != 4
    ):
        return (
            None,
            "INVALID_BOX",
            "Ground-truth bounding box is not a four-value sequence.",
        )

    try:
        x, y, width, height = [
            float(
                value
            )
            for value in bbox
        ]
    except (
        TypeError,
        ValueError,
    ):
        return (
            None,
            "INVALID_BOX",
            "Ground-truth bounding box contains non-numeric values.",
        )

    if (
        width <= 0
        or height <= 0
    ):
        return (
            None,
            "INVALID_BOX",
            "Ground-truth bounding box has non-positive size.",
        )

    x1 = max(
        0.0,
        x,
    )
    y1 = max(
        0.0,
        y,
    )
    x2 = min(
        float(
            image_width
        ),
        x + width,
    )
    y2 = min(
        float(
            image_height
        ),
        y + height,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return (
            None,
            "INVALID_BOX",
            "Clipped ground-truth bounding box has no positive area.",
        )

    return (
        np.asarray(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=float,
        ),
        None,
        None,
    )


def dataset_preflight() -> dict[str, int]:
    """Validate the exact accepted MPEBlink test population."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "MPEBlink 2.0 dataset root was not found:\n"
            f"{DATASET_ROOT}"
        )

    video_dirs = sorted_video_dirs()

    if len(
        video_dirs
    ) != EXPECTED_TEST_VIDEOS:
        raise RuntimeError(
            "Unexpected MPEBlink test video count: "
            f"{len(video_dirs)} != {EXPECTED_TEST_VIDEOS}"
        )

    annotation_frames = 0
    person_sequences = 0
    person_frame_samples = 0
    valid_boxes = 0
    missing_boxes = 0
    invalid_boxes = 0

    for video_dir in video_dirs:
        (
            video_path,
            annotation,
            expected_frames,
            person_keys,
        ) = load_annotation(
            video_dir
        )

        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open benchmark video during preflight: "
                f"{video_path}"
            )

        image_width = int(
            round(
                capture.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )
        )
        image_height = int(
            round(
                capture.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )
        )

        capture.release()

        if (
            image_width <= 0
            or image_height <= 0
        ):
            raise RuntimeError(
                f"Invalid video dimensions during preflight: {video_path}"
            )

        annotation_frames += (
            expected_frames
        )

        person_sequences += len(
            person_keys
        )

        person_frame_samples += (
            expected_frames
            * len(
                person_keys
            )
        )

        for person_key in person_keys:
            for bbox in annotation[
                person_key
            ][
                "bbox"
            ]:
                (
                    box,
                    status,
                    _,
                ) = normalize_box(
                    bbox,
                    image_width,
                    image_height,
                )

                if box is not None:
                    valid_boxes += 1
                elif status == "MISSING_BOX":
                    missing_boxes += 1
                elif status == "INVALID_BOX":
                    invalid_boxes += 1
                else:
                    raise RuntimeError(
                        "Unexpected preflight bounding-box state."
                    )

    if (
        valid_boxes
        + missing_boxes
        + invalid_boxes
        != person_frame_samples
    ):
        raise RuntimeError(
            "Preflight person-frame accounting is inconsistent."
        )

    observed = {
        "videos":
            len(
                video_dirs
            ),
        "annotation_frames":
            annotation_frames,
        "person_sequences":
            person_sequences,
        "person_frame_samples":
            person_frame_samples,
        "valid_bbox_annotations":
            valid_boxes,
        "missing_bbox_annotations":
            missing_boxes,
        "invalid_bbox_annotations":
            invalid_boxes,
    }

    expected = {
        "videos":
            EXPECTED_TEST_VIDEOS,
        "annotation_frames":
            EXPECTED_TEST_ANNOTATION_FRAMES,
        "person_sequences":
            EXPECTED_TEST_PERSON_SEQUENCES,
        "person_frame_samples":
            EXPECTED_TEST_PERSON_FRAME_SAMPLES,
        "valid_bbox_annotations":
            EXPECTED_TEST_VALID_BBOX,
        "missing_bbox_annotations":
            EXPECTED_TEST_MISSING_BBOX,
        "invalid_bbox_annotations":
            EXPECTED_TEST_INVALID_BBOX,
    }

    for key, expected_value in expected.items():
        if observed[
            key
        ] != expected_value:
            raise RuntimeError(
                "MPEBlink test preflight count mismatch for "
                f"{key}: {observed[key]} != {expected_value}"
            )

    return observed


def make_config() -> FaceAnalysisConfig:
    """Build the isolated target-component configuration."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=True,
        quality=False,
        eyes=True,
        blink=True,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
        blink_threshold=BLINK_THRESHOLD,
        min_closed_frames=MIN_CLOSED_FRAMES,
    )

    config.validate()

    return config


def validate_pipeline_configuration(
    pipeline: FaceAnalysis,
) -> None:
    """Verify that only EyeOpenness, Blink, and required landmarks are active."""
    if pipeline.tracker is not None:
        raise RuntimeError(
            "Tracking must be disabled."
        )

    if pipeline.orientation is not None:
        raise RuntimeError(
            "Head pose must be disabled."
        )

    if pipeline.landmarks is None:
        raise RuntimeError(
            "FaceLandmarks is required."
        )

    if pipeline.quality is not None:
        raise RuntimeError(
            "FaceQuality must be disabled."
        )

    if pipeline.eyes is None:
        raise RuntimeError(
            "EyeOpenness is required."
        )

    if pipeline.blink is None:
        raise RuntimeError(
            "BlinkDetector is required."
        )

    if pipeline.gaze is not None:
        raise RuntimeError(
            "GazeDescriptor must be disabled."
        )

    if pipeline.gaze_estimation is not None:
        raise RuntimeError(
            "GazeEstimator must be disabled."
        )

    if pipeline.mouth is not None:
        raise RuntimeError(
            "MouthOpenness must be disabled."
        )

    if pipeline.mouth_motion is not None:
        raise RuntimeError(
            "MouthMovement must be disabled."
        )

    if pipeline.emotion is not None:
        raise RuntimeError(
            "FaceEmotion must be disabled."
        )

    if pipeline.regions is not None:
        raise RuntimeError(
            "FaceRegions must be disabled."
        )

    if pipeline.temporal is not None:
        raise RuntimeError(
            "FaceTemporalAggregator must be disabled."
        )

    if not math.isclose(
        float(
            pipeline.blink.threshold
        ),
        BLINK_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Unexpected BlinkDetector threshold."
        )

    if int(
        pipeline.blink.min_closed_frames
    ) != MIN_CLOSED_FRAMES:
        raise RuntimeError(
            "Unexpected BlinkDetector min_closed_frames."
        )

    if not math.isclose(
        float(
            pipeline.blink.fps
        ),
        float(
            pipeline.fps
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "BlinkDetector FPS does not match FaceAnalysis FPS."
        )


def make_pipeline(
    detector: ControlledFaceDetector,
    fps: float,
) -> FaceAnalysis:
    """Build the real current PhysioTrack FaceAnalysis pipeline."""
    config = make_config()

    pipeline = FaceAnalysis(
        detector=detector,
        config=config,
        fps=float(
            fps
        ),
        device="cpu",
        verbose=False,
    )

    validate_pipeline_configuration(
        pipeline
    )

    return pipeline


def configure_video_state(
    pipeline: FaceAnalysis,
    fps: float,
) -> None:
    """Reset temporal state and bind BlinkDetector to this video's FPS."""
    if fps <= 0:
        raise ValueError(
            "fps must be greater than zero"
        )

    pipeline.reset_temporal_state()

    pipeline.fps = float(
        fps
    )

    pipeline.blink = BlinkDetector(
        threshold=BLINK_THRESHOLD,
        fps=float(
            fps
        ),
        min_closed_frames=MIN_CLOSED_FRAMES,
    )

    validate_pipeline_configuration(
        pipeline
    )


def base_row(
    video_id: int,
    frame_index: int,
    fps: float,
    person_id: str,
    box: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create a stable per-person/per-frame output row."""
    row = {
        "video_id":
            int(
                video_id
            ),
        "frame_index":
            int(
                frame_index
            ),
        "timestamp_seconds":
            float(
                frame_index
            )
            / float(
                fps
            ),
        "person_id":
            str(
                person_id
            ),
        "fps":
            float(
                fps
            ),
        "bbox_x1":
            None,
        "bbox_y1":
            None,
        "bbox_x2":
            None,
        "bbox_y2":
            None,
        "landmarks_available":
            False,
        "landmark_count":
            0,
        "eyes_available":
            False,
        "left_openness":
            None,
        "right_openness":
            None,
        "mean_openness":
            None,
        "blink_available":
            False,
        "eye_state":
            "unknown",
        "blink":
            False,
        "blink_count":
            0,
        "blink_duration":
            None,
        "blink_rate":
            None,
        "status":
            None,
        "failure_reason":
            None,
    }

    if box is not None:
        row[
            "bbox_x1"
        ] = float(
            box[
                0
            ]
        )
        row[
            "bbox_y1"
        ] = float(
            box[
                1
            ]
        )
        row[
            "bbox_x2"
        ] = float(
            box[
                2
            ]
        )
        row[
            "bbox_y2"
        ] = float(
            box[
                3
            ]
        )

    return row


def populate_from_instance(
    row: dict[str, Any],
    instance: Instance,
) -> dict[str, Any]:
    """Read the exact current FaceAnalysis feature schema."""
    features = (
        instance.face_features
        if isinstance(
            instance.face_features,
            dict,
        )
        else {}
    )

    landmarks = features.get(
        "landmarks",
        {},
    )

    eyes = features.get(
        "eyes",
        {},
    )

    blink = features.get(
        "blink",
        {},
    )

    if not isinstance(
        landmarks,
        dict,
    ):
        landmarks = {}

    if not isinstance(
        eyes,
        dict,
    ):
        eyes = {}

    if not isinstance(
        blink,
        dict,
    ):
        blink = {}

    row[
        "landmarks_available"
    ] = bool(
        landmarks.get(
            "available",
            False,
        )
    )

    row[
        "landmark_count"
    ] = int(
        landmarks.get(
            "count",
            0,
        )
        or 0
    )

    row[
        "eyes_available"
    ] = bool(
        eyes.get(
            "available",
            False,
        )
    )

    row[
        "left_openness"
    ] = eyes.get(
        "left_openness"
    )

    row[
        "right_openness"
    ] = eyes.get(
        "right_openness"
    )

    row[
        "mean_openness"
    ] = eyes.get(
        "mean_openness"
    )

    row[
        "blink_available"
    ] = bool(
        blink.get(
            "available",
            False,
        )
    )

    row[
        "eye_state"
    ] = blink.get(
        "eye_state",
        "unknown",
    )

    row[
        "blink"
    ] = bool(
        blink.get(
            "blink",
            False,
        )
    )

    row[
        "blink_count"
    ] = int(
        blink.get(
            "blink_count",
            0,
        )
        or 0
    )

    row[
        "blink_duration"
    ] = blink.get(
        "blink_duration"
    )

    row[
        "blink_rate"
    ] = blink.get(
        "blink_rate"
    )

    if row[
        "eyes_available"
    ]:
        eye_values = [
            row[
                "left_openness"
            ],
            row[
                "right_openness"
            ],
            row[
                "mean_openness"
            ],
        ]

        if not all(
            finite_numeric(
                value
            )
            for value in eye_values
        ):
            raise RuntimeError(
                "Available EyeOpenness output contains non-finite values."
            )

        expected_mean = (
            float(
                row[
                    "left_openness"
                ]
            )
            + float(
                row[
                    "right_openness"
                ]
            )
        ) / 2.0

        if not math.isclose(
            float(
                row[
                    "mean_openness"
                ]
            ),
            expected_mean,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "EyeOpenness mean is inconsistent with left/right openness."
            )

    if row[
        "blink_available"
    ]:
        if not row[
            "eyes_available"
        ]:
            raise RuntimeError(
                "Blink output is available while EyeOpenness is unavailable."
            )

        if str(
            row[
                "eye_state"
            ]
        ) not in {
            "open",
            "closed",
        }:
            raise RuntimeError(
                "Available BlinkDetector output has an invalid eye_state."
            )

        if row[
            "blink_count"
        ] < 0:
            raise RuntimeError(
                "BlinkDetector produced a negative blink count."
            )

        if (
            row[
                "blink_duration"
            ] is not None
            and (
                not finite_numeric(
                    row[
                        "blink_duration"
                    ]
                )
                or float(
                    row[
                        "blink_duration"
                    ]
                )
                < 0.0
            )
        ):
            raise RuntimeError(
                "BlinkDetector produced an invalid blink duration."
            )

        if (
            row[
                "blink_rate"
            ] is not None
            and (
                not finite_numeric(
                    row[
                        "blink_rate"
                    ]
                )
                or float(
                    row[
                        "blink_rate"
                    ]
                )
                < 0.0
            )
        ):
            raise RuntimeError(
                "BlinkDetector produced an invalid blink rate."
            )

    if row[
        "eyes_available"
    ]:
        row[
            "status"
        ] = "OK"
    else:
        row[
            "status"
        ] = "NO_EYE_OUTPUT"
        row[
            "failure_reason"
        ] = (
            "FaceAnalysis produced no available EyeOpenness output."
        )

    return row


def run_smoke_test(
    smoke_count: int,
) -> None:
    """Require several genuine EyeOpenness + Blink outputs without saving files."""
    if smoke_count < 1:
        raise ValueError(
            "smoke_count must be at least 1."
        )

    detector = ControlledFaceDetector()
    pipeline: FaceAnalysis | None = None
    successful_samples = 0

    try:
        for video_dir in sorted_video_dirs():
            (
                video_path,
                annotation,
                expected_frames,
                person_keys,
            ) = load_annotation(
                video_dir
            )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open smoke-test video: {video_path}"
                )

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if fps <= 0:
                capture.release()
                raise RuntimeError(
                    f"Invalid FPS in: {video_path}"
                )

            if pipeline is None:
                pipeline = make_pipeline(
                    detector,
                    fps,
                )
            else:
                configure_video_state(
                    pipeline,
                    fps,
                )

            for frame_index in range(
                expected_frames
            ):
                ok, frame = capture.read()

                if not ok:
                    capture.release()
                    raise RuntimeError(
                        "Smoke-test video ended before annotation length."
                    )

                height, width = frame.shape[
                    :2
                ]

                controlled_faces = []

                for person_key in person_keys:
                    box, _, _ = normalize_box(
                        annotation[
                            person_key
                        ][
                            "bbox"
                        ][
                            frame_index
                        ],
                        width,
                        height,
                    )

                    if box is not None:
                        controlled_faces.append(
                            (
                                person_key,
                                box,
                            )
                        )

                detector.set_faces(
                    controlled_faces
                )

                result = pipeline.predict(
                    frame
                )

                result_by_id = {
                    str(
                        instance.id
                    ):
                        instance
                    for instance in result
                }

                expected_ids = {
                    str(
                        person_id
                    )
                    for person_id, _
                    in controlled_faces
                }

                if set(
                    result_by_id
                ) != expected_ids:
                    capture.release()
                    raise RuntimeError(
                        "Smoke-test FaceAnalysis output IDs do not match "
                        "the controlled input face IDs."
                    )

                for person_id, box in controlled_faces:
                    instance = result_by_id[
                        str(
                            person_id
                        )
                    ]

                    row = base_row(
                        int(
                            video_dir.name
                        ),
                        frame_index,
                        fps,
                        person_id,
                        box,
                    )

                    populate_from_instance(
                        row,
                        instance,
                    )

                    if not (
                        row[
                            "eyes_available"
                        ]
                        and row[
                            "blink_available"
                        ]
                    ):
                        continue

                    successful_samples += 1

                    print(
                        "Smoke sample "
                        f"{successful_samples}: "
                        f"video={video_dir.name}, "
                        f"frame={frame_index}, "
                        f"person={person_id}, "
                        f"left_openness={row['left_openness']}, "
                        f"right_openness={row['right_openness']}, "
                        f"mean_openness={row['mean_openness']}, "
                        f"eye_state={row['eye_state']}, "
                        f"blink={row['blink']}, "
                        f"blink_count={row['blink_count']}, "
                        f"blink_duration={row['blink_duration']}, "
                        f"blink_rate={row['blink_rate']}"
                    )

                    if successful_samples >= smoke_count:
                        capture.release()
                        print(
                            "Smoke test confirmed real EyeOpenness + "
                            "BlinkDetector outputs."
                        )
                        print(
                            "Smoke test: PASS"
                        )
                        return

            capture.release()

        raise RuntimeError(
            "Smoke test did not observe enough successful "
            "EyeOpenness + BlinkDetector samples."
        )

    finally:
        if pipeline is not None:
            pipeline.close()


def create_staging_directory() -> tuple[
    Path,
    Path,
]:
    """Create staging before any full generative work begins."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".eye_openness_blink_component_",
            dir=RESULTS_DIR,
        )
    )

    staged_component_dir = (
        staging_dir
        / "component_execution"
    )

    staged_component_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        staging_dir,
        staged_component_dir,
    )


def write_csv(
    output_path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    """Write per-person/per-frame isolated outputs."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )



def result_part_name(
    part_number: int,
) -> str:
    """Return the deterministic filename for one split result part."""
    return (
        f"{RESULTS_PART_PREFIX}"
        f"{part_number:03d}.csv"
    )


def csv_row_bytes(
    row: dict[str, Any],
) -> int:
    """Return the encoded CSV byte size of one result row."""
    buffer = io.StringIO(
        newline="",
    )

    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_FIELDS,
    )

    writer.writerow(
        row
    )

    return len(
        buffer.getvalue().encode(
            "utf-8"
        )
    )


def csv_header_bytes() -> int:
    """Return the encoded CSV byte size of the repeated part header."""
    buffer = io.StringIO(
        newline="",
    )

    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_FIELDS,
    )

    writer.writeheader()

    return len(
        buffer.getvalue().encode(
            "utf-8"
        )
    )


def group_rows_by_video(
    rows: list[
        dict[str, Any]
    ],
) -> list[
    tuple[
        int,
        list[
            dict[str, Any]
        ],
        int,
    ]
]:
    """Group ordered rows by complete video without changing row order."""
    groups: list[
        tuple[
            int,
            list[
                dict[str, Any]
            ],
            int,
        ]
    ] = []

    current_video_id: int | None = None
    current_rows: list[
        dict[str, Any]
    ] = []
    current_bytes = 0

    for row in rows:
        video_id = int(
            row[
                "video_id"
            ]
        )

        if (
            current_video_id is not None
            and video_id
            != current_video_id
        ):
            groups.append(
                (
                    current_video_id,
                    current_rows,
                    current_bytes,
                )
            )

            current_rows = []
            current_bytes = 0

        current_video_id = video_id

        current_rows.append(
            row
        )

        current_bytes += csv_row_bytes(
            row
        )

    if current_video_id is not None:
        groups.append(
            (
                current_video_id,
                current_rows,
                current_bytes,
            )
        )

    if not groups:
        raise RuntimeError(
            "No result rows were available for output grouping."
        )

    observed_video_ids = [
        video_id
        for video_id, _, _
        in groups
    ]

    if observed_video_ids != sorted(
        observed_video_ids
    ):
        raise RuntimeError(
            "Result rows are not ordered by video_id."
        )

    if len(
        observed_video_ids
    ) != len(
        set(
            observed_video_ids
        )
    ):
        raise RuntimeError(
            "A video_id appears in more than one result group."
        )

    return groups


def partition_video_groups(
    groups: list[
        tuple[
            int,
            list[
                dict[str, Any]
            ],
            int,
        ]
    ],
) -> list[
    list[
        tuple[
            int,
            list[
                dict[str, Any]
            ],
            int,
        ]
    ]
]:
    """Split oversized results only at complete-video boundaries."""
    header_bytes = csv_header_bytes()

    total_data_bytes = sum(
        group_bytes
        for _, _, group_bytes
        in groups
    )

    single_file_bytes = (
        header_bytes
        + total_data_bytes
    )

    if (
        single_file_bytes
        <= GIT_SAFE_MAX_FILE_SIZE_BYTES
    ):
        return [
            groups
        ]

    largest_group_bytes = max(
        group_bytes
        for _, _, group_bytes
        in groups
    )

    if (
        header_bytes
        + largest_group_bytes
        > GIT_SAFE_MAX_FILE_SIZE_BYTES
    ):
        raise RuntimeError(
            "At least one complete video exceeds the configured Git-safe "
            "maximum file size and cannot be split without dividing a video."
        )

    minimum_parts = int(
        math.ceil(
            single_file_bytes
            / GIT_SAFE_MAX_FILE_SIZE_BYTES
        )
    )

    part_count = max(
        2,
        minimum_parts,
    )

    while part_count <= len(
        groups
    ):
        target_data_bytes = (
            total_data_bytes
            / part_count
        )

        partitions: list[
            list[
                tuple[
                    int,
                    list[
                        dict[str, Any]
                    ],
                    int,
                ]
            ]
        ] = []

        current_part: list[
            tuple[
                int,
                list[
                    dict[str, Any]
                ],
                int,
            ]
        ] = []

        current_bytes = 0
        remaining_parts = part_count

        for group_index, group in enumerate(
            groups
        ):
            group_bytes = group[
                2
            ]

            groups_remaining_after = (
                len(
                    groups
                )
                - group_index
                - 1
            )

            must_leave_for_remaining_parts = (
                groups_remaining_after
                >= remaining_parts
            )

            if current_part:
                bytes_if_added = (
                    header_bytes
                    + current_bytes
                    + group_bytes
                )

                distance_without = abs(
                    current_bytes
                    - target_data_bytes
                )

                distance_with = abs(
                    current_bytes
                    + group_bytes
                    - target_data_bytes
                )

                close_here = (
                    distance_without
                    <= distance_with
                    and must_leave_for_remaining_parts
                )

                exceeds_limit = (
                    bytes_if_added
                    > GIT_SAFE_MAX_FILE_SIZE_BYTES
                )

                if (
                    close_here
                    or exceeds_limit
                ):
                    partitions.append(
                        current_part
                    )

                    current_part = []
                    current_bytes = 0
                    remaining_parts -= 1

            current_part.append(
                group
            )

            current_bytes += group_bytes

        if current_part:
            partitions.append(
                current_part
            )

        if len(
            partitions
        ) == part_count:
            valid = True

            for partition in partitions:
                partition_bytes = (
                    header_bytes
                    + sum(
                        group[
                            2
                        ]
                        for group
                        in partition
                    )
                )

                if (
                    partition_bytes
                    > GIT_SAFE_MAX_FILE_SIZE_BYTES
                ):
                    valid = False
                    break

            if valid:
                return partitions

        part_count += 1

    raise RuntimeError(
        "Could not create Git-safe result parts without splitting a video."
    )


def write_result_outputs(
    output_dir: Path,
    rows: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """Write one CSV or Git-safe split CSV parts at video boundaries."""
    groups = group_rows_by_video(
        rows
    )

    partitions = partition_video_groups(
        groups
    )

    output_manifest: list[
        dict[str, Any]
    ] = []

    if len(
        partitions
    ) == 1:
        output_path = (
            output_dir
            / RESULTS_FILENAME
        )

        write_csv(
            output_path,
            rows,
        )

        output_manifest.append(
            {
                "filename":
                    output_path.name,
                "part_number":
                    1,
                "row_count":
                    len(
                        rows
                    ),
                "first_video_id":
                    int(
                        rows[
                            0
                        ][
                            "video_id"
                        ]
                    ),
                "last_video_id":
                    int(
                        rows[
                            -1
                        ][
                            "video_id"
                        ]
                    ),
                "size_bytes":
                    output_path.stat().st_size,
                "size_mib":
                    (
                        output_path.stat().st_size
                        / 1024
                        / 1024
                    ),
            }
        )

        return output_manifest

    for part_number, partition in enumerate(
        partitions,
        start=1,
    ):
        part_rows = [
            row
            for _, video_rows, _
            in partition
            for row in video_rows
        ]

        output_path = (
            output_dir
            / result_part_name(
                part_number
            )
        )

        write_csv(
            output_path,
            part_rows,
        )

        size_bytes = (
            output_path.stat().st_size
        )

        if (
            size_bytes
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                "Generated result part exceeds the configured Git-safe "
                f"maximum size: {output_path.name}"
            )

        output_manifest.append(
            {
                "filename":
                    output_path.name,
                "part_number":
                    part_number,
                "row_count":
                    len(
                        part_rows
                    ),
                "first_video_id":
                    int(
                        partition[
                            0
                        ][
                            0
                        ]
                    ),
                "last_video_id":
                    int(
                        partition[
                            -1
                        ][
                            0
                        ]
                    ),
                "size_bytes":
                    size_bytes,
                "size_mib":
                    (
                        size_bytes
                        / 1024
                        / 1024
                    ),
            }
        )

    return output_manifest


def staged_result_paths(
    staged_component_dir: Path,
    output_manifest: list[
        dict[str, Any]
    ],
) -> list[Path]:
    """Resolve result paths from the staged output manifest."""
    return [
        staged_component_dir
        / item[
            "filename"
        ]
        for item in output_manifest
    ]


def validate_result_parts(
    result_paths: list[Path],
    expected_rows: list[
        dict[str, Any]
    ],
    output_manifest: list[
        dict[str, Any]
    ],
) -> None:
    """Verify that ordered parts reconstruct the generated rows exactly."""
    if not result_paths:
        raise RuntimeError(
            "No staged component-result CSV files were generated."
        )

    if len(
        result_paths
    ) != len(
        output_manifest
    ):
        raise RuntimeError(
            "Result manifest and result-file counts differ."
        )

    expected_index = 0
    seen_video_ids: set[int] = set()
    previous_last_video_id: int | None = None

    for result_path, manifest_item in zip(
        result_paths,
        output_manifest,
    ):
        if not result_path.is_file():
            raise RuntimeError(
                f"Missing staged result file: {result_path}"
            )

        if (
            len(
                result_paths
            )
            > 1
            and result_path.stat().st_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                "A staged result part exceeds the configured Git-safe size."
            )

        with result_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(
                file
            )

            if reader.fieldnames != CSV_FIELDS:
                raise RuntimeError(
                    "Unexpected component-result CSV schema."
                )

            part_rows = list(
                reader
            )

        if len(
            part_rows
        ) != int(
            manifest_item[
                "row_count"
            ]
        ):
            raise RuntimeError(
                "Result-part row count does not match its manifest entry."
            )

        if not part_rows:
            raise RuntimeError(
                "A generated result part is empty."
            )

        first_video_id = int(
            part_rows[
                0
            ][
                "video_id"
            ]
        )

        last_video_id = int(
            part_rows[
                -1
            ][
                "video_id"
            ]
        )

        if (
            first_video_id
            != int(
                manifest_item[
                    "first_video_id"
                ]
            )
            or last_video_id
            != int(
                manifest_item[
                    "last_video_id"
                ]
            )
        ):
            raise RuntimeError(
                "Result-part video boundaries do not match the manifest."
            )

        if (
            previous_last_video_id is not None
            and first_video_id
            <= previous_last_video_id
        ):
            raise RuntimeError(
                "Result parts overlap or are out of video order."
            )

        part_video_ids = {
            int(
                row[
                    "video_id"
                ]
            )
            for row in part_rows
        }

        if (
            seen_video_ids
            & part_video_ids
        ):
            raise RuntimeError(
                "A video was split across more than one result part."
            )

        seen_video_ids.update(
            part_video_ids
        )

        previous_last_video_id = (
            last_video_id
        )

        for stored_row in part_rows:
            if expected_index >= len(
                expected_rows
            ):
                raise RuntimeError(
                    "Result parts contain more rows than expected."
                )

            expected_row = expected_rows[
                expected_index
            ]

            serialized_expected = {
                field:
                    (
                        ""
                        if expected_row.get(
                            field
                        ) is None
                        else str(
                            expected_row.get(
                                field
                            )
                        )
                    )
                for field in CSV_FIELDS
            }

            if stored_row != serialized_expected:
                raise RuntimeError(
                    "A split result row differs from the generated "
                    f"in-memory result at row {expected_index}."
                )

            expected_index += 1

    if expected_index != len(
        expected_rows
    ):
        raise RuntimeError(
            "Result parts do not reconstruct the complete generated result set."
        )


def validate_rows(
    rows: list[
        dict[str, Any]
    ],
    preflight: dict[str, int],
) -> None:
    """Validate complete isolated numerical outputs before commit."""
    if len(
        rows
    ) != preflight[
        "person_frame_samples"
    ]:
        raise RuntimeError(
            "Unexpected component-result row count: "
            f"{len(rows)} != {preflight['person_frame_samples']}"
        )

    keys = [
        (
            int(
                row[
                    "video_id"
                ]
            ),
            int(
                row[
                    "frame_index"
                ]
            ),
            str(
                row[
                    "person_id"
                ]
            ),
        )
        for row in rows
    ]

    if len(
        keys
    ) != len(
        set(
            keys
        )
    ):
        raise RuntimeError(
            "Duplicate video/frame/person rows were found."
        )

    valid_statuses = {
        "OK",
        "NO_EYE_OUTPUT",
        "MISSING_BOX",
        "INVALID_BOX",
        "VIDEO_READ_FAILED",
        "FRAME_READ_FAILED",
        "EXECUTION_FAILED",
    }

    for row in rows:
        if row[
            "status"
        ] not in valid_statuses:
            raise RuntimeError(
                f"Unexpected status: {row['status']}"
            )

        if row[
            "eyes_available"
        ]:
            eye_values = [
                row[
                    "left_openness"
                ],
                row[
                    "right_openness"
                ],
                row[
                    "mean_openness"
                ],
            ]

            if not all(
                finite_numeric(
                    value
                )
                for value in eye_values
            ):
                raise RuntimeError(
                    "Available EyeOpenness row contains non-finite values."
                )

            expected_mean = (
                float(
                    row[
                        "left_openness"
                    ]
                )
                + float(
                    row[
                        "right_openness"
                    ]
                )
            ) / 2.0

            if not math.isclose(
                float(
                    row[
                        "mean_openness"
                    ]
                ),
                expected_mean,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Stored EyeOpenness mean is inconsistent."
                )

        if row[
            "blink_available"
        ]:
            if not row[
                "eyes_available"
            ]:
                raise RuntimeError(
                    "Stored Blink output is available without EyeOpenness."
                )

            if row[
                "eye_state"
            ] not in {
                "open",
                "closed",
            }:
                raise RuntimeError(
                    "Stored Blink eye_state is invalid."
                )

            if int(
                row[
                    "blink_count"
                ]
            ) < 0:
                raise RuntimeError(
                    "Stored blink_count is negative."
                )

            if (
                row[
                    "blink_duration"
                ] is not None
                and (
                    not finite_numeric(
                        row[
                            "blink_duration"
                        ]
                    )
                    or float(
                        row[
                            "blink_duration"
                        ]
                    )
                    < 0.0
                )
            ):
                raise RuntimeError(
                    "Stored blink_duration is invalid."
                )

            if (
                row[
                    "blink_rate"
                ] is not None
                and (
                    not finite_numeric(
                        row[
                            "blink_rate"
                        ]
                    )
                    or float(
                        row[
                            "blink_rate"
                        ]
                    )
                    < 0.0
                )
            ):
                raise RuntimeError(
                    "Stored blink_rate is invalid."
                )


def build_summary(
    rows: list[
        dict[str, Any]
    ],
    preflight: dict[str, int],
    runtime_seconds: float,
    output_manifest: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """Create isolated execution summary without accuracy metrics."""
    status_counts: dict[str, int] = {}

    for row in rows:
        status = str(
            row[
                "status"
            ]
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    eyes_available = sum(
        1
        for row in rows
        if row[
            "eyes_available"
        ]
    )

    blink_available = sum(
        1
        for row in rows
        if row[
            "blink_available"
        ]
    )

    blink_pulses = sum(
        1
        for row in rows
        if row[
            "blink"
        ]
    )

    execution_failures = (
        status_counts.get(
            "VIDEO_READ_FAILED",
            0,
        )
        + status_counts.get(
            "FRAME_READ_FAILED",
            0,
        )
        + status_counts.get(
            "EXECUTION_FAILED",
            0,
        )
    )

    overall_status = (
        "PASS"
        if execution_failures == 0
        else "FAIL"
    )

    return {
        "component":
            "PhysioTrack EyeOpenness + BlinkDetector",
        "execution_type":
            "isolated component execution; not an accuracy benchmark",
        "dataset":
            "MPEBlink 2.0 test",
        "pipeline":
            "PhysioTrack FaceAnalysis",
        "device":
            "cpu",
        "controlled_input":
            "MPEBlink per-person ground-truth face bounding boxes",
        "required_prerequisite":
            "PhysioTrack FaceLandmarks",
        "enabled_components": [
            "landmarks",
            "eyes",
            "blink",
        ],
        "disabled_components": [
            "tracking",
            "head_pose",
            "quality",
            "gaze",
            "gaze_estimation",
            "mouth",
            "mouth_motion",
            "emotion",
            "regions",
            "temporal",
        ],
        "geometry_note":
            (
                "EyeOpenness is produced by the current PhysioTrack "
                "image-dimension-aware implementation."
            ),
        "temporal_note":
            (
                "Blink state is isolated per benchmark video and no "
                "accuracy metric is recomputed here."
            ),
        "blink_threshold":
            BLINK_THRESHOLD,
        "min_closed_frames":
            MIN_CLOSED_FRAMES,
        "videos":
            preflight[
                "videos"
            ],
        "annotation_frames":
            preflight[
                "annotation_frames"
            ],
        "person_sequences":
            preflight[
                "person_sequences"
            ],
        "person_frame_samples":
            preflight[
                "person_frame_samples"
            ],
        "valid_bbox_annotations":
            preflight[
                "valid_bbox_annotations"
            ],
        "missing_bbox_annotations":
            preflight[
                "missing_bbox_annotations"
            ],
        "invalid_bbox_annotations":
            preflight[
                "invalid_bbox_annotations"
            ],
        "eye_openness_available_rows":
            eyes_available,
        "blink_available_rows":
            blink_available,
        "blink_event_pulses":
            blink_pulses,
        "status_counts":
            status_counts,
        "execution_failures":
            execution_failures,
        "runtime_seconds":
            float(
                runtime_seconds
            ),
        "result_output_policy": {
            "git_safe_max_file_size_mib":
                GIT_SAFE_MAX_FILE_SIZE_MIB,
            "split_only_at_complete_video_boundaries":
                True,
            "result_file_count":
                len(
                    output_manifest
                ),
            "files":
                output_manifest,
        },
        "overall_status":
            overall_status,
    }


def validate_staged_outputs(
    result_paths: list[Path],
    summary_path: Path,
    preflight: dict[str, int],
    expected_rows: list[
        dict[str, Any]
    ],
    output_manifest: list[
        dict[str, Any]
    ],
) -> None:
    """Re-read staged outputs and verify them before final replacement."""
    if not summary_path.is_file():
        raise RuntimeError(
            f"Missing staged summary: {summary_path}"
        )

    validate_result_parts(
        result_paths,
        expected_rows,
        output_manifest,
    )

    total_staged_rows = sum(
        int(
            item[
                "row_count"
            ]
        )
        for item in output_manifest
    )

    if (
        total_staged_rows
        != preflight[
            "person_frame_samples"
        ]
    ):
        raise RuntimeError(
            "Staged result-part row count does not match preflight population."
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if summary.get(
        "overall_status"
    ) != "PASS":
        raise RuntimeError(
            "Staged isolated component summary did not pass."
        )

    if int(
        summary.get(
            "execution_failures",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Staged isolated component output contains execution failures."
        )

    expected_counts = {
        "videos":
            EXPECTED_TEST_VIDEOS,
        "annotation_frames":
            EXPECTED_TEST_ANNOTATION_FRAMES,
        "person_sequences":
            EXPECTED_TEST_PERSON_SEQUENCES,
        "person_frame_samples":
            EXPECTED_TEST_PERSON_FRAME_SAMPLES,
        "valid_bbox_annotations":
            EXPECTED_TEST_VALID_BBOX,
        "missing_bbox_annotations":
            EXPECTED_TEST_MISSING_BBOX,
        "invalid_bbox_annotations":
            EXPECTED_TEST_INVALID_BBOX,
    }

    for key, expected_value in expected_counts.items():
        if int(
            summary.get(
                key,
                -1,
            )
        ) != expected_value:
            raise RuntimeError(
                "Staged isolated summary count mismatch for "
                f"{key}."
            )

    summary_output_policy = summary.get(
        "result_output_policy",
        {},
    )

    if (
        int(
            summary_output_policy.get(
                "result_file_count",
                -1,
            )
        )
        != len(
            output_manifest
        )
    ):
        raise RuntimeError(
            "Staged summary result-file count does not match generated files."
        )

    if (
        summary_output_policy.get(
            "files"
        )
        != output_manifest
    ):
        raise RuntimeError(
            "Staged summary result manifest does not match generated files."
        )


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Atomically install one validated output file."""
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )

    os.close(
        descriptor
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        shutil.copy2(
            source_path,
            temporary_path,
        )

        os.replace(
            temporary_path,
            destination_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def owned_result_paths() -> list[Path]:
    """Return all final files owned by this isolated execution script."""
    paths = [
        COMPONENT_RESULTS_DIR
        / RESULTS_FILENAME,
        COMPONENT_RESULTS_DIR
        / SUMMARY_FILENAME,
    ]

    if COMPONENT_RESULTS_DIR.is_dir():
        paths.extend(
            sorted(
                COMPONENT_RESULTS_DIR.glob(
                    f"{RESULTS_PART_PREFIX}*.csv"
                )
            )
        )

    unique_paths = []

    seen = set()

    for path in paths:
        key = str(
            path
        )

        if key not in seen:
            seen.add(
                key
            )
            unique_paths.append(
                path
            )

    return unique_paths


def commit_outputs(
    staged_result_paths: list[Path],
    staged_summary: Path,
    staging_dir: Path,
) -> None:
    """Replace all script-owned outputs transactionally with rollback."""
    COMPONENT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_paths = (
        staged_result_paths
        + [
            staged_summary
        ]
    )

    final_paths = [
        COMPONENT_RESULTS_DIR
        / path.name
        for path in staged_paths
    ]

    desired_names = {
        path.name
        for path in final_paths
    }

    previous_paths = [
        path
        for path in owned_result_paths()
        if path.is_file()
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for previous_path in previous_paths:
        shutil.copy2(
            previous_path,
            backup_dir
            / previous_path.name,
        )

    installed_paths: list[Path] = []

    try:
        for staged_path, final_path in zip(
            staged_paths,
            final_paths,
        ):
            atomic_copy_file(
                staged_path,
                final_path,
            )

            installed_paths.append(
                final_path
            )

        for previous_path in previous_paths:
            if (
                previous_path.name
                not in desired_names
                and previous_path.exists()
            ):
                previous_path.unlink()

    except Exception:
        for installed_path in installed_paths:
            if installed_path.exists():
                installed_path.unlink()

        for backup_path in backup_dir.iterdir():
            atomic_copy_file(
                backup_path,
                COMPONENT_RESULTS_DIR
                / backup_path.name,
            )

        raise



def run_full(
    preflight: dict[str, int],
) -> None:
    """Run complete isolated execution on the accepted MPEBlink test set."""
    (
        staging_dir,
        staged_component_dir,
    ) = create_staging_directory()

    print(
        f"Staging directory: {staging_dir}"
    )

    staged_summary = (
        staged_component_dir
        / SUMMARY_FILENAME
    )

    rows: list[
        dict[str, Any]
    ] = []

    detector = ControlledFaceDetector()
    pipeline: FaceAnalysis | None = None

    start_time = time.perf_counter()

    try:
        video_dirs = sorted_video_dirs()

        for video_number, video_dir in enumerate(
            video_dirs,
            start=1,
        ):
            (
                video_path,
                annotation,
                expected_frames,
                person_keys,
            ) = load_annotation(
                video_dir
            )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                for frame_index in range(
                    expected_frames
                ):
                    for person_key in person_keys:
                        row = base_row(
                            int(
                                video_dir.name
                            ),
                            frame_index,
                            1.0,
                            person_key,
                        )

                        row[
                            "status"
                        ] = "VIDEO_READ_FAILED"

                        row[
                            "failure_reason"
                        ] = (
                            "Could not open benchmark video."
                        )

                        rows.append(
                            row
                        )

                print(
                    f"Processed test video {video_number}/"
                    f"{len(video_dirs)} (VIDEO_READ_FAILED)"
                )

                continue

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if fps <= 0:
                capture.release()
                raise RuntimeError(
                    f"Invalid FPS in: {video_path}"
                )

            if pipeline is None:
                pipeline = make_pipeline(
                    detector,
                    fps,
                )

            else:
                configure_video_state(
                    pipeline,
                    fps,
                )

            frame_index = 0

            while frame_index < expected_frames:
                ok, frame = capture.read()

                if not ok:
                    for remaining_frame in range(
                        frame_index,
                        expected_frames,
                    ):
                        for person_key in person_keys:
                            row = base_row(
                                int(
                                    video_dir.name
                                ),
                                remaining_frame,
                                fps,
                                person_key,
                            )

                            row[
                                "status"
                            ] = "FRAME_READ_FAILED"

                            row[
                                "failure_reason"
                            ] = (
                                "Video ended before the annotated frame count."
                            )

                            rows.append(
                                row
                            )

                    frame_index = expected_frames
                    break

                height, width = frame.shape[
                    :2
                ]

                frame_rows: dict[
                    str,
                    dict[str, Any],
                ] = {}

                controlled_faces: list[
                    tuple[
                        str,
                        np.ndarray,
                    ]
                ] = []

                for person_key in person_keys:
                    raw_bbox = annotation[
                        person_key
                    ][
                        "bbox"
                    ][
                        frame_index
                    ]

                    (
                        box,
                        box_status,
                        box_reason,
                    ) = normalize_box(
                        raw_bbox,
                        width,
                        height,
                    )

                    row = base_row(
                        int(
                            video_dir.name
                        ),
                        frame_index,
                        fps,
                        person_key,
                        box,
                    )

                    if box is None:
                        row[
                            "status"
                        ] = box_status

                        row[
                            "failure_reason"
                        ] = box_reason

                    else:
                        controlled_faces.append(
                            (
                                person_key,
                                box,
                            )
                        )

                    frame_rows[
                        person_key
                    ] = row

                detector.set_faces(
                    controlled_faces
                )

                try:
                    result = pipeline.predict(
                        frame
                    )

                    result_by_id = {
                        str(
                            instance.id
                        ):
                            instance
                        for instance in result
                    }

                    expected_ids = {
                        str(
                            person_id
                        )
                        for person_id, _
                        in controlled_faces
                    }

                    if set(
                        result_by_id
                    ) != expected_ids:
                        raise RuntimeError(
                            "FaceAnalysis output IDs do not match the "
                            "controlled valid-box inputs."
                        )

                    for person_id, _ in controlled_faces:
                        populate_from_instance(
                            frame_rows[
                                str(
                                    person_id
                                )
                            ],
                            result_by_id[
                                str(
                                    person_id
                                )
                            ],
                        )

                except Exception as error:
                    for person_id, _ in controlled_faces:
                        row = frame_rows[
                            str(
                                person_id
                            )
                        ]

                        row[
                            "status"
                        ] = "EXECUTION_FAILED"

                        row[
                            "failure_reason"
                        ] = (
                            f"{type(error).__name__}: {error}"
                        )

                for person_key in person_keys:
                    rows.append(
                        frame_rows[
                            person_key
                        ]
                    )

                frame_index += 1

            capture.release()

            print(
                f"Processed test video {video_number}/{len(video_dirs)}"
            )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        validate_rows(
            rows,
            preflight,
        )

        preliminary_summary = build_summary(
            rows,
            preflight,
            runtime_seconds,
            [],
        )

        if preliminary_summary[
            "execution_failures"
        ] != 0:
            raise RuntimeError(
                "Isolated component execution contained execution/read "
                "failures. Prior accepted outputs were preserved."
            )

        output_manifest = write_result_outputs(
            staged_component_dir,
            rows,
        )

        result_paths = staged_result_paths(
            staged_component_dir,
            output_manifest,
        )

        summary = build_summary(
            rows,
            preflight,
            runtime_seconds,
            output_manifest,
        )

        with staged_summary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write(
                "\n"
            )

        print(
            "Validating staged isolated component outputs..."
        )

        validate_staged_outputs(
            result_paths,
            staged_summary,
            preflight,
            rows,
            output_manifest,
        )

        commit_outputs(
            result_paths,
            staged_summary,
            staging_dir,
        )

        print(
            "Committed final isolated component outputs."
        )

        print()
        print(
            "=== Eye Openness + Blink Isolated Component Results ==="
        )

        print(
            f"Videos: {summary['videos']}"
        )

        print(
            f"Annotation frames: {summary['annotation_frames']}"
        )

        print(
            f"Person sequences: {summary['person_sequences']}"
        )

        print(
            f"Person-frame rows: {summary['person_frame_samples']}"
        )

        print(
            "EyeOpenness available rows: "
            f"{summary['eye_openness_available_rows']}"
        )

        print(
            "Blink available rows: "
            f"{summary['blink_available_rows']}"
        )

        print(
            "Blink event pulses: "
            f"{summary['blink_event_pulses']}"
        )

        print(
            "Execution failures: "
            f"{summary['execution_failures']}"
        )

        print(
            f"Runtime: {summary['runtime_seconds'] / 60.0:.2f} minutes"
        )

        print(
            f"Overall status: {summary['overall_status']}"
        )

        print()
        print(
            "Saved:"
        )

        for manifest_item in output_manifest:
            print(
                COMPONENT_RESULTS_DIR
                / manifest_item[
                    "filename"
                ]
            )

            print(
                "  "
                f"videos={manifest_item['first_video_id']}-"
                f"{manifest_item['last_video_id']} | "
                f"rows={manifest_item['row_count']} | "
                f"size={manifest_item['size_mib']:.2f} MiB"
            )

        print(
            COMPONENT_RESULTS_DIR
            / SUMMARY_FILENAME
        )

    finally:
        if pipeline is not None:
            pipeline.close()

        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


def parse_args() -> argparse.Namespace:
    """Parse isolated component execution mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated PhysioTrack EyeOpenness + BlinkDetector "
            "component execution on MPEBlink 2.0 test."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate project paths, dataset structure, and accepted "
            "population counts without model inference."
        ),
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small real FaceAnalysis inference test without "
            "writing final outputs."
        ),
    )

    parser.add_argument(
        "--smoke-count",
        type=int,
        default=3,
        help=(
            "Number of successful EyeOpenness + Blink samples required "
            "during --smoke-test."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the requested isolated component verification mode."""
    args = parse_args()

    preflight = dataset_preflight()

    print(
        "MPEBlink 2.0 isolated component preflight: PASS"
    )

    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    print(
        f"Test videos: {preflight['videos']}"
    )

    print(
        f"Annotation frames: {preflight['annotation_frames']}"
    )

    print(
        f"Person sequences: {preflight['person_sequences']}"
    )

    print(
        f"Person-frame samples: {preflight['person_frame_samples']}"
    )

    print(
        f"Valid bbox annotations: {preflight['valid_bbox_annotations']}"
    )

    print(
        f"Missing bbox annotations: {preflight['missing_bbox_annotations']}"
    )

    print(
        f"Invalid bbox annotations: {preflight['invalid_bbox_annotations']}"
    )

    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )

    print(
        "Target components: EyeOpenness + BlinkDetector"
    )

    print(
        "Required prerequisite: FaceLandmarks"
    )

    print(
        "Controlled input: MPEBlink per-person ground-truth face boxes"
    )

    print(
        "Blink threshold: 0.22"
    )

    print(
        "Minimum closed frames: 3"
    )

    print(
        "Unrelated components: disabled"
    )

    print(
        "Accuracy metrics: not computed"
    )

    if args.preflight_only:
        print(
            "Preflight-only mode: no model inference was run."
        )

        return

    if args.smoke_test:
        run_smoke_test(
            args.smoke_count
        )

        return

    run_full(
        preflight
    )


if __name__ == "__main__":
    main()
