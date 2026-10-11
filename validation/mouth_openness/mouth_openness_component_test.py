from __future__ import annotations

import argparse
import csv
import hashlib
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
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

FELT_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "FELT"
    / "raw_motion_speech"
)

RAVDESS_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "RAVDESS"
    / "Video_Speech"
)

RESULTS_DIR = SCRIPT_DIR / "results"
COMPONENT_RESULTS_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_FILENAME = (
    "mouth_openness_component_results.csv"
)

RESULTS_PART_PREFIX = (
    "mouth_openness_component_results_part"
)

SUMMARY_FILENAME = (
    "mouth_openness_component_summary.json"
)

GIT_SAFE_MAX_FILE_SIZE_MIB = 90.0
GIT_SAFE_MAX_FILE_SIZE_BYTES = int(
    GIT_SAFE_MAX_FILE_SIZE_MIB
    * 1024
    * 1024
)

DEVICE = "cpu"

EXPECTED_ACTORS = [
    f"Actor_{index:02d}"
    for index in range(
        1,
        25,
    )
]

EXPECTED_TRIALS_PER_ACTOR = 60
EXPECTED_TOTAL_TRIALS = 1440
EXPECTED_RAW_ANNOTATION_ROWS = 158288
EXPECTED_UNIQUE_ANNOTATED_FRAMES = 158286
EXPECTED_DUPLICATE_ROWS_RESOLVED = 2

EXPECTED_FPS = (
    30000.0
    / 1001.0
)

FPS_TOLERANCE = 1e-4

EXPECTED_LANDMARK_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db134"
    "1ff2bb731874b0bcc2fe6544e0bc9ff"
)

REQUIRED_COLUMNS = {
    "frame",
    "FaceRectX",
    "FaceRectY",
    "FaceRectWidth",
    "FaceRectHeight",
    "FaceScore",
}

CSV_FIELDS = [
    "actor",
    "trial",
    "frame",
    "timestamp_seconds",
    "fps",
    "image_width",
    "image_height",
    "duplicate_candidates",
    "FaceScore",
    "face_rect_x1",
    "face_rect_y1",
    "face_rect_x2",
    "face_rect_y2",
    "landmarks_available",
    "landmark_count",
    "mouth_available",
    "mouth_openness",
    "mouth_width",
    "mouth_height",
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

if str(
    SRC_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC_DIR
        ),
    )


from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.models import Models
from physiotrack.results import Instance, Result


class ControlledFaceDetector:
    """Return the accepted FELT FaceRect to the real FaceAnalysis pipeline."""

    def __init__(
        self,
    ) -> None:
        self.instance: Instance | None = None

    def set_face(
        self,
        face_id: str,
        box: np.ndarray,
    ) -> None:
        """Set one controlled face for the next frame."""
        self.instance = Instance(
            id=str(
                face_id
            ),
            box=np.asarray(
                box,
                dtype=float,
            ),
            confidence=1.0,
            cls=0,
            cls_name="face",
        )

    def clear(
        self,
    ) -> None:
        """Clear the controlled face."""
        self.instance = None

    def predict(
        self,
        frame: np.ndarray,
    ) -> Result:
        """Return the currently configured controlled face."""
        instances = (
            [
                self.instance
            ]
            if self.instance is not None
            else []
        )

        return Result(
            orig_img=frame,
            instances=instances,
            task="face",
        )


def finite_numeric(
    value: Any,
) -> bool:
    """Return True for a finite non-boolean real numerical value."""
    if (
        value is None
        or isinstance(
            value,
            bool,
        )
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


def sha256_file(
    path: Path,
) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024
                * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def dataset_inventory(
    root: Path,
) -> dict[
    str,
    tuple[
        int,
        int,
    ],
]:
    """Record dataset file size and modification time without writing data."""
    inventory = {}

    for path in sorted(
        root.rglob(
            "*"
        )
    ):
        if not path.is_file():
            continue

        stat = path.stat()

        inventory[
            path.relative_to(
                root
            ).as_posix()
        ] = (
            int(
                stat.st_size
            ),
            int(
                stat.st_mtime_ns
            ),
        )

    return inventory


def validate_trial_filename(
    actor: str,
    csv_path: Path,
) -> None:
    """Validate the locked FELT/RAVDESS speech-trial naming convention."""
    parts = csv_path.stem.split(
        "-"
    )

    if len(
        parts
    ) != 7:
        raise RuntimeError(
            "Unexpected FELT/RAVDESS trial filename: "
            f"{csv_path.name}"
        )

    if parts[
        0
    ] != "01":
        raise RuntimeError(
            "Mouth-openness isolated execution expects audio-video "
            f"speech trials beginning with '01-': {csv_path.name}"
        )

    expected_actor_token = actor.split(
        "_"
    )[
        -1
    ]

    if parts[
        -1
    ] != expected_actor_token:
        raise RuntimeError(
            "Trial actor token does not match its actor directory: "
            f"{actor}/{csv_path.name}"
        )


def resolve_duplicate_frames(
    frame_table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """Resolve duplicate FELT frame rows using the accepted benchmark rule."""
    table = frame_table.copy()

    table[
        "_face_area"
    ] = (
        table[
            "FaceRectWidth"
        ]
        * table[
            "FaceRectHeight"
        ]
    )

    unique_frame_count = int(
        table[
            "frame"
        ].nunique()
    )

    duplicate_rows_resolved = int(
        len(
            table
        )
        - unique_frame_count
    )

    table = (
        table.sort_values(
            [
                "frame",
                "_face_area",
                "FaceScore",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "frame"
            ],
            keep="first",
        )
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        table,
        duplicate_rows_resolved,
    )


def face_rect_box(
    row: pd.Series,
) -> np.ndarray:
    """Convert one accepted FELT FaceRect from xywh to xyxy."""
    x = float(
        row[
            "FaceRectX"
        ]
    )

    y = float(
        row[
            "FaceRectY"
        ]
    )

    width = float(
        row[
            "FaceRectWidth"
        ]
    )

    height = float(
        row[
            "FaceRectHeight"
        ]
    )

    values = np.asarray(
        [
            x,
            y,
            width,
            height,
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "FELT FaceRect contains non-finite values."
        )

    if (
        width <= 0.0
        or height <= 0.0
    ):
        raise ValueError(
            "FELT FaceRect has non-positive dimensions."
        )

    return np.asarray(
        [
            x,
            y,
            x + width,
            y + height,
        ],
        dtype=float,
    )


def resolve_landmark_model() -> tuple[
    Path,
    str,
]:
    """Resolve and verify the accepted FaceLandmarks model."""
    model_path = Path(
        Models.resolve(
            Models.Face.MediaPipe.Landmarks.face_landmarker
        )
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            "PhysioTrack FaceLandmarks model could not be resolved."
        )

    model_sha256 = sha256_file(
        model_path
    )

    if (
        model_sha256
        != EXPECTED_LANDMARK_MODEL_SHA256
    ):
        raise RuntimeError(
            "Unexpected FaceLandmarks model SHA256: "
            f"{model_sha256}"
        )

    return (
        model_path,
        model_sha256,
    )


def dataset_preflight() -> dict[
    str,
    Any,
]:
    """Validate the exact accepted FELT/RAVDESS isolated population."""
    if not FELT_ROOT.is_dir():
        raise FileNotFoundError(
            "FELT speech annotations were not found at:\n"
            "datasets/FELT/raw_motion_speech"
        )

    if not RAVDESS_ROOT.is_dir():
        raise FileNotFoundError(
            "RAVDESS speech videos were not found at:\n"
            "datasets/RAVDESS/Video_Speech"
        )

    felt_actor_dirs = sorted(
        path.name
        for path in FELT_ROOT.iterdir()
        if path.is_dir()
    )

    ravdess_actor_dirs = sorted(
        path.name
        for path in RAVDESS_ROOT.iterdir()
        if path.is_dir()
    )

    if (
        felt_actor_dirs
        != EXPECTED_ACTORS
    ):
        raise RuntimeError(
            "Unexpected FELT actor structure."
        )

    if (
        ravdess_actor_dirs
        != EXPECTED_ACTORS
    ):
        raise RuntimeError(
            "Unexpected RAVDESS actor structure."
        )

    trials = []
    total_raw_rows = 0
    total_unique_frames = 0
    total_duplicate_rows_resolved = 0

    for actor in EXPECTED_ACTORS:
        felt_actor_dir = (
            FELT_ROOT
            / actor
        )

        ravdess_actor_dir = (
            RAVDESS_ROOT
            / actor
        )

        csv_paths = sorted(
            felt_actor_dir.glob(
                "*.csv"
            )
        )

        if (
            len(
                csv_paths
            )
            != EXPECTED_TRIALS_PER_ACTOR
        ):
            raise RuntimeError(
                f"Unexpected FELT trial count for {actor}: "
                f"{len(csv_paths)}"
            )

        expected_video_names = {
            f"{path.stem}.mp4"
            for path in csv_paths
        }

        matching_video_names = {
            path.name
            for path in ravdess_actor_dir.glob(
                "01-*.mp4"
            )
        }

        if (
            matching_video_names
            != expected_video_names
        ):
            raise RuntimeError(
                "FELT/RAVDESS pairing mismatch for "
                f"{actor}."
            )

        for csv_path in csv_paths:
            validate_trial_filename(
                actor,
                csv_path,
            )

            video_path = (
                ravdess_actor_dir
                / f"{csv_path.stem}.mp4"
            )

            frame_table = pd.read_csv(
                csv_path
            )

            missing_columns = sorted(
                REQUIRED_COLUMNS
                - set(
                    frame_table.columns
                )
            )

            if missing_columns:
                raise RuntimeError(
                    "FELT annotation is missing required columns in "
                    f"{actor}/{csv_path.name}: {missing_columns}"
                )

            frame_values = pd.to_numeric(
                frame_table[
                    "frame"
                ],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    frame_values
                )
            ):
                raise RuntimeError(
                    "FELT frame IDs contain non-finite values in "
                    f"{actor}/{csv_path.name}."
                )

            if not np.all(
                frame_values
                == np.floor(
                    frame_values
                )
            ):
                raise RuntimeError(
                    "FELT frame IDs must be integers in "
                    f"{actor}/{csv_path.name}."
                )

            frame_table[
                "frame"
            ] = frame_values.astype(
                np.int64
            )

            for column in (
                "FaceRectX",
                "FaceRectY",
                "FaceRectWidth",
                "FaceRectHeight",
                "FaceScore",
            ):
                frame_table[
                    column
                ] = pd.to_numeric(
                    frame_table[
                        column
                    ],
                    errors="raise",
                )

            face_values = frame_table[
                [
                    "FaceRectX",
                    "FaceRectY",
                    "FaceRectWidth",
                    "FaceRectHeight",
                    "FaceScore",
                ]
            ].to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    face_values
                )
            ):
                raise RuntimeError(
                    "FELT FaceRect/FaceScore contains non-finite values in "
                    f"{actor}/{csv_path.name}."
                )

            if (
                frame_table[
                    "FaceRectWidth"
                ]
                <= 0
            ).any() or (
                frame_table[
                    "FaceRectHeight"
                ]
                <= 0
            ).any():
                raise RuntimeError(
                    "FELT contains a non-positive FaceRect in "
                    f"{actor}/{csv_path.name}."
                )

            (
                unique_table,
                duplicate_rows_resolved,
            ) = resolve_duplicate_frames(
                frame_table
            )

            frame_ids = unique_table[
                "frame"
            ].to_numpy(
                dtype=np.int64
            )

            expected_frame_ids = np.arange(
                len(
                    unique_table
                ),
                dtype=np.int64,
            )

            if not np.array_equal(
                frame_ids,
                expected_frame_ids,
            ):
                raise RuntimeError(
                    "FELT frame IDs must be contiguous and zero-based "
                    "after duplicate resolution in "
                    f"{actor}/{csv_path.name}."
                )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                capture.release()

                raise RuntimeError(
                    "RAVDESS video could not be opened during preflight: "
                    f"{actor}/{video_path.name}"
                )

            video_frame_count = int(
                round(
                    capture.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )
            )

            video_fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
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
                video_frame_count
                < len(
                    unique_table
                )
            ):
                raise RuntimeError(
                    "RAVDESS video has fewer frames than FELT annotations "
                    f"for {actor}/{video_path.name}."
                )

            if (
                not math.isfinite(
                    video_fps
                )
                or abs(
                    video_fps
                    - EXPECTED_FPS
                )
                > FPS_TOLERANCE
            ):
                raise RuntimeError(
                    "Unexpected RAVDESS FPS for "
                    f"{actor}/{video_path.name}: {video_fps}"
                )

            if (
                image_width <= 0
                or image_height <= 0
            ):
                raise RuntimeError(
                    "Invalid RAVDESS video dimensions for "
                    f"{actor}/{video_path.name}."
                )

            trials.append(
                {
                    "actor":
                        actor,
                    "csv_path":
                        csv_path,
                    "video_path":
                        video_path,
                    "annotated_frames":
                        len(
                            unique_table
                        ),
                    "raw_annotation_rows":
                        len(
                            frame_table
                        ),
                    "duplicate_rows_resolved":
                        duplicate_rows_resolved,
                    "video_frames":
                        video_frame_count,
                    "video_fps":
                        video_fps,
                    "image_width":
                        image_width,
                    "image_height":
                        image_height,
                }
            )

            total_raw_rows += len(
                frame_table
            )

            total_unique_frames += len(
                unique_table
            )

            total_duplicate_rows_resolved += (
                duplicate_rows_resolved
            )

    if (
        len(
            trials
        )
        != EXPECTED_TOTAL_TRIALS
    ):
        raise RuntimeError(
            "Unexpected paired trial count: "
            f"{len(trials)}"
        )

    if (
        total_raw_rows
        != EXPECTED_RAW_ANNOTATION_ROWS
    ):
        raise RuntimeError(
            "Unexpected raw FELT annotation-row count: "
            f"{total_raw_rows}"
        )

    if (
        total_unique_frames
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Unexpected unique annotated-frame count: "
            f"{total_unique_frames}"
        )

    if (
        total_duplicate_rows_resolved
        != EXPECTED_DUPLICATE_ROWS_RESOLVED
    ):
        raise RuntimeError(
            "Unexpected duplicate-row resolution count: "
            f"{total_duplicate_rows_resolved}"
        )

    (
        model_path,
        model_sha256,
    ) = resolve_landmark_model()

    return {
        "actors":
            len(
                EXPECTED_ACTORS
            ),
        "trials":
            len(
                trials
            ),
        "raw_annotation_rows":
            total_raw_rows,
        "unique_annotated_frames":
            total_unique_frames,
        "duplicate_rows_resolved":
            total_duplicate_rows_resolved,
        "trials_data":
            trials,
        "landmark_model_path":
            model_path,
        "landmark_model_sha256":
            model_sha256,
    }


def make_config(
) -> FaceAnalysisConfig:
    """Build the isolated MouthOpenness configuration."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=True,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=True,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    config.validate()

    return config


def validate_pipeline_configuration(
    pipeline: FaceAnalysis,
    detector: ControlledFaceDetector,
) -> None:
    """Verify that only MouthOpenness and required FaceLandmarks are active."""
    if (
        pipeline.detector
        is not detector
    ):
        raise RuntimeError(
            "FaceAnalysis is not using the controlled FELT FaceRect detector."
        )

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

    if pipeline.eyes is not None:
        raise RuntimeError(
            "EyeOpenness must be disabled."
        )

    if pipeline.blink is not None:
        raise RuntimeError(
            "BlinkDetector must be disabled."
        )

    if pipeline.gaze is not None:
        raise RuntimeError(
            "GazeDescriptor must be disabled."
        )

    if pipeline.gaze_estimation is not None:
        raise RuntimeError(
            "GazeEstimator must be disabled."
        )

    if pipeline.mouth is None:
        raise RuntimeError(
            "MouthOpenness is required."
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


def make_pipeline(
    detector: ControlledFaceDetector,
    model_path: Path,
) -> FaceAnalysis:
    """Build the real current PhysioTrack FaceAnalysis pipeline."""
    pipeline = FaceAnalysis(
        detector=detector,
        config=make_config(),
        fps=EXPECTED_FPS,
        landmark_model_path=model_path,
        device=DEVICE,
        verbose=False,
    )

    validate_pipeline_configuration(
        pipeline,
        detector,
    )

    return pipeline


def base_row(
    actor: str,
    trial: str,
    frame_id: int,
    fps: float,
    image_width: int,
    image_height: int,
    duplicate_candidates: int,
    face_score: float,
    box: np.ndarray,
) -> dict[
    str,
    Any,
]:
    """Create one stable isolated component result row."""
    return {
        "actor":
            actor,
        "trial":
            trial,
        "frame":
            int(
                frame_id
            ),
        "timestamp_seconds":
            float(
                frame_id
            )
            / float(
                fps
            ),
        "fps":
            float(
                fps
            ),
        "image_width":
            int(
                image_width
            ),
        "image_height":
            int(
                image_height
            ),
        "duplicate_candidates":
            int(
                duplicate_candidates
            ),
        "FaceScore":
            float(
                face_score
            ),
        "face_rect_x1":
            float(
                box[
                    0
                ]
            ),
        "face_rect_y1":
            float(
                box[
                    1
                ]
            ),
        "face_rect_x2":
            float(
                box[
                    2
                ]
            ),
        "face_rect_y2":
            float(
                box[
                    3
                ]
            ),
        "landmarks_available":
            False,
        "landmark_count":
            0,
        "mouth_available":
            False,
        "mouth_openness":
            None,
        "mouth_width":
            None,
        "mouth_height":
            None,
        "status":
            None,
        "failure_reason":
            None,
    }


def populate_from_instance(
    row: dict[
        str,
        Any,
    ],
    instance: Instance,
) -> None:
    """Read the exact current FaceAnalysis landmark and mouth feature schema."""
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

    mouth = features.get(
        "mouth",
        {},
    )

    if not isinstance(
        landmarks,
        dict,
    ):
        landmarks = {}

    if not isinstance(
        mouth,
        dict,
    ):
        mouth = {}

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
        "mouth_available"
    ] = bool(
        mouth.get(
            "available",
            False,
        )
    )

    row[
        "mouth_openness"
    ] = mouth.get(
        "mouth_openness"
    )

    row[
        "mouth_width"
    ] = mouth.get(
        "mouth_width"
    )

    row[
        "mouth_height"
    ] = mouth.get(
        "mouth_height"
    )

    if row[
        "landmarks_available"
    ]:
        if int(
            row[
                "landmark_count"
            ]
        ) != 478:
            raise RuntimeError(
                "Available FaceLandmarks output does not contain 478 landmarks."
            )

    if row[
        "mouth_available"
    ]:
        if not row[
            "landmarks_available"
        ]:
            raise RuntimeError(
                "MouthOpenness is available while FaceLandmarks is unavailable."
            )

        values = [
            row[
                "mouth_openness"
            ],
            row[
                "mouth_width"
            ],
            row[
                "mouth_height"
            ],
        ]

        if not all(
            finite_numeric(
                value
            )
            for value in values
        ):
            raise RuntimeError(
                "Available MouthOpenness output contains non-finite values."
            )

        mouth_openness = float(
            row[
                "mouth_openness"
            ]
        )

        mouth_width = float(
            row[
                "mouth_width"
            ]
        )

        mouth_height = float(
            row[
                "mouth_height"
            ]
        )

        if (
            mouth_openness < 0.0
            or mouth_width <= 0.0
            or mouth_height < 0.0
        ):
            raise RuntimeError(
                "Available MouthOpenness output contains invalid geometry."
            )

        expected_openness = (
            mouth_height
            / mouth_width
        )

        if not math.isclose(
            mouth_openness,
            expected_openness,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "MouthOpenness is inconsistent with mouth_height/mouth_width."
            )

        row[
            "status"
        ] = "OK"

    else:
        row[
            "status"
        ] = "NO_MOUTH_OUTPUT"

        row[
            "failure_reason"
        ] = (
            "FaceAnalysis produced no available MouthOpenness output."
        )


def run_smoke_test(
    preflight: dict[
        str,
        Any,
    ],
    smoke_count: int,
) -> None:
    """Require several genuine MouthOpenness outputs without saving files."""
    if smoke_count < 1:
        raise ValueError(
            "smoke_count must be at least 1."
        )

    detector = ControlledFaceDetector()
    pipeline: FaceAnalysis | None = None
    successful_samples = 0

    try:
        pipeline = make_pipeline(
            detector,
            Path(
                preflight[
                    "landmark_model_path"
                ]
            ),
        )

        for trial in preflight[
            "trials_data"
        ]:
            actor = str(
                trial[
                    "actor"
                ]
            )

            csv_path = Path(
                trial[
                    "csv_path"
                ]
            )

            video_path = Path(
                trial[
                    "video_path"
                ]
            )

            frame_table = pd.read_csv(
                csv_path
            )

            frame_table[
                "frame"
            ] = pd.to_numeric(
                frame_table[
                    "frame"
                ],
                errors="raise",
            ).astype(
                np.int64
            )

            raw_frame_counts = (
                frame_table.groupby(
                    "frame"
                )
                .size()
                .to_dict()
            )

            (
                frame_table,
                _,
            ) = resolve_duplicate_frames(
                frame_table
            )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                capture.release()

                raise RuntimeError(
                    f"Could not open smoke-test video: {video_path}"
                )

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            for _, annotation_row in frame_table.iterrows():
                frame_id = int(
                    annotation_row[
                        "frame"
                    ]
                )

                ok, frame = capture.read()

                if (
                    not ok
                    or frame is None
                ):
                    capture.release()

                    raise RuntimeError(
                        "Smoke-test video ended before annotation length."
                    )

                box = face_rect_box(
                    annotation_row
                )

                face_id = (
                    f"{actor}/"
                    f"{csv_path.stem}"
                )

                detector.set_face(
                    face_id,
                    box,
                )

                result = pipeline.predict(
                    frame
                )

                if len(
                    result
                ) != 1:
                    capture.release()

                    raise RuntimeError(
                        "Smoke-test FaceAnalysis did not return exactly one "
                        "controlled face."
                    )

                instance = result[
                    0
                ]

                if str(
                    instance.id
                ) != face_id:
                    capture.release()

                    raise RuntimeError(
                        "Smoke-test FaceAnalysis did not preserve the "
                        "controlled face identifier."
                    )

                output_row = base_row(
                    actor=actor,
                    trial=csv_path.stem,
                    frame_id=frame_id,
                    fps=fps,
                    image_width=frame.shape[
                        1
                    ],
                    image_height=frame.shape[
                        0
                    ],
                    duplicate_candidates=int(
                        raw_frame_counts.get(
                            frame_id,
                            1,
                        )
                    ),
                    face_score=float(
                        annotation_row[
                            "FaceScore"
                        ]
                    ),
                    box=box,
                )

                populate_from_instance(
                    output_row,
                    instance,
                )

                if not output_row[
                    "mouth_available"
                ]:
                    continue

                successful_samples += 1

                print(
                    "Smoke sample "
                    f"{successful_samples}: "
                    f"actor={actor}, "
                    f"trial={csv_path.stem}, "
                    f"frame={frame_id}, "
                    f"mouth_openness={output_row['mouth_openness']}, "
                    f"mouth_width={output_row['mouth_width']}, "
                    f"mouth_height={output_row['mouth_height']}, "
                    f"landmark_count={output_row['landmark_count']}"
                )

                if (
                    successful_samples
                    >= smoke_count
                ):
                    capture.release()

                    print(
                        "Smoke test confirmed real MouthOpenness output "
                        "through FaceAnalysis."
                    )

                    print(
                        "Smoke test: PASS"
                    )

                    return

            capture.release()

        raise RuntimeError(
            "Smoke test did not observe enough successful "
            "MouthOpenness samples."
        )

    finally:
        detector.clear()

        if pipeline is not None:
            pipeline.close()


def create_staging_directory(
) -> tuple[
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
            prefix=".mouth_openness_component_",
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
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Write isolated per-frame MouthOpenness outputs."""
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
    """Return a deterministic Git-safe result-part filename."""
    return (
        f"{RESULTS_PART_PREFIX}"
        f"{part_number:03d}.csv"
    )


def actor_groups(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    tuple[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ]
]:
    """Group ordered rows by complete actor without changing row order."""
    groups = []

    current_actor = None
    current_rows = []

    for row in rows:
        actor = str(
            row[
                "actor"
            ]
        )

        if (
            current_actor is not None
            and actor
            != current_actor
        ):
            groups.append(
                (
                    current_actor,
                    current_rows,
                )
            )

            current_rows = []

        current_actor = actor

        current_rows.append(
            row
        )

    if current_actor is not None:
        groups.append(
            (
                current_actor,
                current_rows,
            )
        )

    observed_actors = [
        actor
        for actor, _
        in groups
    ]

    if (
        observed_actors
        != EXPECTED_ACTORS
    ):
        raise RuntimeError(
            "Component-result rows are not grouped in the expected actor order."
        )

    return groups


def write_result_outputs(
    output_dir: Path,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """Write one CSV or split oversized output only between complete actors."""
    single_path = (
        output_dir
        / RESULTS_FILENAME
    )

    write_csv(
        single_path,
        rows,
    )

    single_size = (
        single_path.stat().st_size
    )

    if (
        single_size
        <= GIT_SAFE_MAX_FILE_SIZE_BYTES
    ):
        return [
            {
                "filename":
                    single_path.name,
                "part_number":
                    1,
                "row_count":
                    len(
                        rows
                    ),
                "first_actor":
                    str(
                        rows[
                            0
                        ][
                            "actor"
                        ]
                    ),
                "last_actor":
                    str(
                        rows[
                            -1
                        ][
                            "actor"
                        ]
                    ),
                "size_bytes":
                    single_size,
                "size_mib":
                    (
                        single_size
                        / 1024
                        / 1024
                    ),
            }
        ]

    single_path.unlink()

    groups = actor_groups(
        rows
    )

    output_manifest = []
    current_groups = []
    part_number = 1

    def write_current_part(
    ) -> None:
        nonlocal part_number
        nonlocal current_groups

        if not current_groups:
            return

        part_rows = [
            row
            for _, actor_rows
            in current_groups
            for row in actor_rows
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
                "A complete-actor result part exceeds the configured "
                "Git-safe file-size limit."
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
                "first_actor":
                    current_groups[
                        0
                    ][
                        0
                    ],
                "last_actor":
                    current_groups[
                        -1
                    ][
                        0
                    ],
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

        part_number += 1
        current_groups = []

    for actor, actor_rows in groups:
        trial_groups = (
            current_groups
            + [
                (
                    actor,
                    actor_rows,
                )
            ]
        )

        trial_rows = [
            row
            for _, grouped_rows
            in trial_groups
            for row in grouped_rows
        ]

        probe_path = (
            output_dir
            / ".mouth_openness_component_size_probe.csv"
        )

        write_csv(
            probe_path,
            trial_rows,
        )

        probe_size = (
            probe_path.stat().st_size
        )

        probe_path.unlink()

        if (
            probe_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
            and current_groups
        ):
            write_current_part()

        current_groups.append(
            (
                actor,
                actor_rows,
            )
        )

    write_current_part()

    if not output_manifest:
        raise RuntimeError(
            "No Git-safe component-result parts were generated."
        )

    return output_manifest


def validate_rows(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    preflight: dict[
        str,
        Any,
    ],
) -> None:
    """Validate complete real MouthOpenness numerical outputs before writing."""
    if len(
        rows
    ) != preflight[
        "unique_annotated_frames"
    ]:
        raise RuntimeError(
            "Unexpected isolated component row count: "
            f"{len(rows)}"
        )

    keys = [
        (
            str(
                row[
                    "actor"
                ]
            ),
            str(
                row[
                    "trial"
                ]
            ),
            int(
                row[
                    "frame"
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
            "Duplicate actor/trial/frame rows were found."
        )

    valid_statuses = {
        "OK",
        "NO_MOUTH_OUTPUT",
        "VIDEO_OPEN_FAILED",
        "FRAME_READ_FAILED",
        "EXECUTION_FAILED",
    }

    for row in rows:
        if row[
            "status"
        ] not in valid_statuses:
            raise RuntimeError(
                f"Unexpected isolated component status: {row['status']}"
            )

        if row[
            "mouth_available"
        ]:
            if not row[
                "landmarks_available"
            ]:
                raise RuntimeError(
                    "Stored MouthOpenness is available without FaceLandmarks."
                )

            if int(
                row[
                    "landmark_count"
                ]
            ) != 478:
                raise RuntimeError(
                    "Stored available FaceLandmarks count is not 478."
                )

            for field in (
                "mouth_openness",
                "mouth_width",
                "mouth_height",
            ):
                if not finite_numeric(
                    row[
                        field
                    ]
                ):
                    raise RuntimeError(
                        f"Stored available MouthOpenness field is invalid: "
                        f"{field}"
                    )

            openness = float(
                row[
                    "mouth_openness"
                ]
            )

            width = float(
                row[
                    "mouth_width"
                ]
            )

            height = float(
                row[
                    "mouth_height"
                ]
            )

            if (
                openness < 0.0
                or width <= 0.0
                or height < 0.0
            ):
                raise RuntimeError(
                    "Stored MouthOpenness geometry is invalid."
                )

            if not math.isclose(
                openness,
                height
                / width,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Stored MouthOpenness ratio is inconsistent."
                )


def build_summary(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    preflight: dict[
        str,
        Any,
    ],
    runtime_seconds: float,
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    """Create isolated execution summary without benchmark accuracy metrics."""
    status_counts = {}

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

    landmarks_available = sum(
        1
        for row in rows
        if row[
            "landmarks_available"
        ]
    )

    mouth_available = sum(
        1
        for row in rows
        if row[
            "mouth_available"
        ]
    )

    execution_failures = (
        status_counts.get(
            "VIDEO_OPEN_FAILED",
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

    component_unavailable = (
        status_counts.get(
            "NO_MOUTH_OUTPUT",
            0,
        )
    )

    overall_status = (
        "PASS"
        if (
            execution_failures == 0
            and component_unavailable == 0
            and landmarks_available
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
            and mouth_available
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
        )
        else "FAIL"
    )

    return {
        "component":
            "PhysioTrack MouthOpenness",
        "execution_type":
            "isolated component execution; not an accuracy benchmark",
        "dataset":
            "FELT/RAVDESS speech subset",
        "pipeline":
            "PhysioTrack FaceAnalysis",
        "physiotrack_source":
            "src/physiotrack/face/mouth.py",
        "device":
            DEVICE,
        "controlled_input":
            "Accepted FELT FaceRect bounding boxes",
        "required_prerequisite":
            "PhysioTrack FaceLandmarks",
        "enabled_components": [
            "landmarks",
            "mouth",
        ],
        "disabled_components": [
            "tracking",
            "head_pose",
            "quality",
            "eyes",
            "blink",
            "gaze",
            "gaze_estimation",
            "mouth_motion",
            "emotion",
            "regions",
            "temporal",
        ],
        "geometry_note":
            (
                "MouthOpenness is produced by the current PhysioTrack "
                "image-dimension-aware pixel-consistent implementation."
            ),
        "accuracy_metrics_computed":
            False,
        "actors":
            preflight[
                "actors"
            ],
        "paired_speech_trials":
            preflight[
                "trials"
            ],
        "raw_annotation_rows":
            preflight[
                "raw_annotation_rows"
            ],
        "unique_annotated_frames":
            preflight[
                "unique_annotated_frames"
            ],
        "duplicate_rows_resolved":
            preflight[
                "duplicate_rows_resolved"
            ],
        "landmark_model_file":
            Path(
                preflight[
                    "landmark_model_path"
                ]
            ).name,
        "landmark_model_sha256":
            preflight[
                "landmark_model_sha256"
            ],
        "landmarks_available_rows":
            landmarks_available,
        "mouth_openness_available_rows":
            mouth_available,
        "status_counts":
            status_counts,
        "execution_failures":
            execution_failures,
        "component_unavailable_rows":
            component_unavailable,
        "runtime_seconds":
            float(
                runtime_seconds
            ),
        "result_output_policy": {
            "git_safe_max_file_size_mib":
                GIT_SAFE_MAX_FILE_SIZE_MIB,
            "split_only_at_complete_actor_boundaries":
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


def staged_result_paths(
    staged_component_dir: Path,
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    Path
]:
    """Resolve staged result paths from the output manifest."""
    return [
        staged_component_dir
        / item[
            "filename"
        ]
        for item in output_manifest
    ]


def validate_result_files(
    result_paths: list[
        Path
    ],
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
    preflight: dict[
        str,
        Any,
    ],
) -> None:
    """Re-read staged CSV file(s) and verify full population coverage."""
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
            "Result-file count does not match output manifest."
        )

    all_rows = []
    seen_actors = set()

    for path, manifest_item in zip(
        result_paths,
        output_manifest,
    ):
        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Missing or empty staged result file: {path}"
            )

        if (
            len(
                result_paths
            ) > 1
            and path.stat().st_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                "A split isolated component result exceeds the "
                "Git-safe file-size limit."
            )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(
                file
            )

            if (
                reader.fieldnames
                != CSV_FIELDS
            ):
                raise RuntimeError(
                    "Unexpected MouthOpenness component-result CSV schema."
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
                "Staged result-part row count does not match manifest."
            )

        part_actors = {
            row[
                "actor"
            ]
            for row in part_rows
        }

        if (
            seen_actors
            & part_actors
        ):
            raise RuntimeError(
                "An actor was split across multiple component-result files."
            )

        seen_actors.update(
            part_actors
        )

        all_rows.extend(
            part_rows
        )

    if len(
        all_rows
    ) != preflight[
        "unique_annotated_frames"
    ]:
        raise RuntimeError(
            "Staged result files do not contain the complete frame population."
        )

    keys = {
        (
            row[
                "actor"
            ],
            row[
                "trial"
            ],
            int(
                row[
                    "frame"
                ]
            ),
        )
        for row in all_rows
    }

    if len(
        keys
    ) != preflight[
        "unique_annotated_frames"
    ]:
        raise RuntimeError(
            "Staged result files do not preserve unique actor/trial/frame "
            "coverage."
        )

    mouth_available = 0
    landmarks_available = 0

    for row in all_rows:
        status = row[
            "status"
        ]

        if status == "OK":
            mouth_available += 1

        if str(
            row[
                "landmarks_available"
            ]
        ).strip().lower() == "true":
            landmarks_available += 1

        if status == "OK":
            for field in (
                "mouth_openness",
                "mouth_width",
                "mouth_height",
            ):
                if not finite_numeric(
                    row[
                        field
                    ]
                ):
                    raise RuntimeError(
                        "Staged available MouthOpenness row contains "
                        f"non-finite {field}."
                    )

            openness = float(
                row[
                    "mouth_openness"
                ]
            )

            width = float(
                row[
                    "mouth_width"
                ]
            )

            height = float(
                row[
                    "mouth_height"
                ]
            )

            if not math.isclose(
                openness,
                height
                / width,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Staged MouthOpenness ratio is inconsistent."
                )

    if (
        mouth_available
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Staged isolated output does not contain complete "
            "MouthOpenness availability."
        )

    if (
        landmarks_available
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Staged isolated output does not contain complete "
            "FaceLandmarks availability."
        )


def validate_staged_outputs(
    result_paths: list[
        Path
    ],
    summary_path: Path,
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
    preflight: dict[
        str,
        Any,
    ],
) -> None:
    """Validate staged CSV(s) and summary before final replacement."""
    if (
        not summary_path.is_file()
        or summary_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"Missing or empty staged summary: {summary_path}"
        )

    validate_result_files(
        result_paths,
        output_manifest,
        preflight,
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if (
        summary.get(
            "overall_status"
        )
        != "PASS"
    ):
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

    if int(
        summary.get(
            "component_unavailable_rows",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Staged isolated component output contains unavailable "
            "MouthOpenness rows."
        )

    expected_counts = {
        "actors":
            len(
                EXPECTED_ACTORS
            ),
        "paired_speech_trials":
            EXPECTED_TOTAL_TRIALS,
        "raw_annotation_rows":
            EXPECTED_RAW_ANNOTATION_ROWS,
        "unique_annotated_frames":
            EXPECTED_UNIQUE_ANNOTATED_FRAMES,
        "duplicate_rows_resolved":
            EXPECTED_DUPLICATE_ROWS_RESOLVED,
        "landmarks_available_rows":
            EXPECTED_UNIQUE_ANNOTATED_FRAMES,
        "mouth_openness_available_rows":
            EXPECTED_UNIQUE_ANNOTATED_FRAMES,
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

    if bool(
        summary.get(
            "accuracy_metrics_computed",
            True,
        )
    ):
        raise RuntimeError(
            "Isolated summary incorrectly claims benchmark accuracy metrics."
        )

    output_policy = summary.get(
        "result_output_policy",
        {},
    )

    if (
        int(
            output_policy.get(
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
        output_policy.get(
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
    """Atomically install one validated file."""
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


def owned_result_paths(
) -> list[
    Path
]:
    """Return all final files owned by this isolated component script."""
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
    staged_result_paths: list[
        Path
    ],
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

    installed_paths = []

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
    preflight: dict[
        str,
        Any,
    ],
) -> None:
    """Run full isolated MouthOpenness execution on the accepted population."""
    (
        staging_dir,
        staged_component_dir,
    ) = create_staging_directory()

    staged_summary = (
        staged_component_dir
        / SUMMARY_FILENAME
    )

    detector = ControlledFaceDetector()
    pipeline: FaceAnalysis | None = None

    felt_before = dataset_inventory(
        FELT_ROOT
    )

    ravdess_before = dataset_inventory(
        RAVDESS_ROOT
    )

    rows = []

    try:
        pipeline = make_pipeline(
            detector,
            Path(
                preflight[
                    "landmark_model_path"
                ]
            ),
        )

        start_time = time.perf_counter()

        for trial_index, trial in enumerate(
            preflight[
                "trials_data"
            ],
            start=1,
        ):
            actor = str(
                trial[
                    "actor"
                ]
            )

            csv_path = Path(
                trial[
                    "csv_path"
                ]
            )

            video_path = Path(
                trial[
                    "video_path"
                ]
            )

            frame_table = pd.read_csv(
                csv_path
            )

            frame_table[
                "frame"
            ] = pd.to_numeric(
                frame_table[
                    "frame"
                ],
                errors="raise",
            ).astype(
                np.int64
            )

            raw_frame_counts = (
                frame_table.groupby(
                    "frame"
                )
                .size()
                .to_dict()
            )

            (
                frame_table,
                _,
            ) = resolve_duplicate_frames(
                frame_table
            )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                capture.release()

                for _, annotation_row in frame_table.iterrows():
                    frame_id = int(
                        annotation_row[
                            "frame"
                        ]
                    )

                    box = face_rect_box(
                        annotation_row
                    )

                    row = base_row(
                        actor=actor,
                        trial=csv_path.stem,
                        frame_id=frame_id,
                        fps=float(
                            trial[
                                "video_fps"
                            ]
                        ),
                        image_width=int(
                            trial[
                                "image_width"
                            ]
                        ),
                        image_height=int(
                            trial[
                                "image_height"
                            ]
                        ),
                        duplicate_candidates=int(
                            raw_frame_counts.get(
                                frame_id,
                                1,
                            )
                        ),
                        face_score=float(
                            annotation_row[
                                "FaceScore"
                            ]
                        ),
                        box=box,
                    )

                    row[
                        "status"
                    ] = "VIDEO_OPEN_FAILED"

                    row[
                        "failure_reason"
                    ] = (
                        "cv2.VideoCapture could not open video."
                    )

                    rows.append(
                        row
                    )

                print(
                    f"Processed trial {trial_index}/{EXPECTED_TOTAL_TRIALS}: "
                    f"{actor}/{csv_path.stem} | VIDEO_OPEN_FAILED"
                )

                continue

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
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

            frame_rows = list(
                frame_table.iterrows()
            )

            frame_position = 0

            while (
                frame_position
                < len(
                    frame_rows
                )
            ):
                _, annotation_row = frame_rows[
                    frame_position
                ]

                frame_id = int(
                    annotation_row[
                        "frame"
                    ]
                )

                box = face_rect_box(
                    annotation_row
                )

                row = base_row(
                    actor=actor,
                    trial=csv_path.stem,
                    frame_id=frame_id,
                    fps=fps,
                    image_width=image_width,
                    image_height=image_height,
                    duplicate_candidates=int(
                        raw_frame_counts.get(
                            frame_id,
                            1,
                        )
                    ),
                    face_score=float(
                        annotation_row[
                            "FaceScore"
                        ]
                    ),
                    box=box,
                )

                ok, frame = capture.read()

                if (
                    not ok
                    or frame is None
                ):
                    row[
                        "status"
                    ] = "FRAME_READ_FAILED"

                    row[
                        "failure_reason"
                    ] = (
                        "RAVDESS video ended before the complete accepted "
                        "annotation range."
                    )

                    rows.append(
                        row
                    )

                    for remaining_position in range(
                        frame_position
                        + 1,
                        len(
                            frame_rows
                        ),
                    ):
                        _, remaining_annotation = frame_rows[
                            remaining_position
                        ]

                        remaining_frame_id = int(
                            remaining_annotation[
                                "frame"
                            ]
                        )

                        remaining_box = face_rect_box(
                            remaining_annotation
                        )

                        remaining_row = base_row(
                            actor=actor,
                            trial=csv_path.stem,
                            frame_id=remaining_frame_id,
                            fps=fps,
                            image_width=image_width,
                            image_height=image_height,
                            duplicate_candidates=int(
                                raw_frame_counts.get(
                                    remaining_frame_id,
                                    1,
                                )
                            ),
                            face_score=float(
                                remaining_annotation[
                                    "FaceScore"
                                ]
                            ),
                            box=remaining_box,
                        )

                        remaining_row[
                            "status"
                        ] = "FRAME_READ_FAILED"

                        remaining_row[
                            "failure_reason"
                        ] = (
                            "RAVDESS video ended before the complete "
                            "accepted annotation range."
                        )

                        rows.append(
                            remaining_row
                        )

                    break

                face_id = (
                    f"{actor}/"
                    f"{csv_path.stem}"
                )

                detector.set_face(
                    face_id,
                    box,
                )

                try:
                    result = pipeline.predict(
                        frame
                    )

                    if len(
                        result
                    ) != 1:
                        raise RuntimeError(
                            "FaceAnalysis did not return exactly one "
                            "controlled FELT face."
                        )

                    instance = result[
                        0
                    ]

                    if str(
                        instance.id
                    ) != face_id:
                        raise RuntimeError(
                            "FaceAnalysis did not preserve the controlled "
                            "FELT face identifier."
                        )

                    populate_from_instance(
                        row,
                        instance,
                    )

                except Exception as error:
                    row[
                        "status"
                    ] = "EXECUTION_FAILED"

                    row[
                        "failure_reason"
                    ] = (
                        f"{type(error).__name__}: {error}"
                    )

                rows.append(
                    row
                )

                frame_position += 1

            capture.release()

            if (
                trial_index
                % 20
                == 0
                or trial_index
                == EXPECTED_TOTAL_TRIALS
            ):
                mouth_available = sum(
                    1
                    for item in rows
                    if item[
                        "mouth_available"
                    ]
                )

                print(
                    f"Trials: {trial_index}/{EXPECTED_TOTAL_TRIALS} | "
                    f"rows={len(rows)} | "
                    f"mouth_available={mouth_available}"
                )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        detector.clear()

        felt_after = dataset_inventory(
            FELT_ROOT
        )

        ravdess_after = dataset_inventory(
            RAVDESS_ROOT
        )

        if (
            felt_before
            != felt_after
        ):
            raise RuntimeError(
                "FELT dataset integrity check failed: dataset contents "
                "changed during isolated execution."
            )

        if (
            ravdess_before
            != ravdess_after
        ):
            raise RuntimeError(
                "RAVDESS dataset integrity check failed: dataset contents "
                "changed during isolated execution."
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

        if (
            preliminary_summary[
                "overall_status"
            ]
            != "PASS"
        ):
            raise RuntimeError(
                "Isolated MouthOpenness execution did not satisfy PASS "
                "invariants. Prior accepted outputs were preserved."
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
            "Validating staged isolated MouthOpenness outputs..."
        )

        validate_staged_outputs(
            result_paths,
            staged_summary,
            output_manifest,
            preflight,
        )

        print(
            "Staged isolated output validation: PASS"
        )

        commit_outputs(
            result_paths,
            staged_summary,
            staging_dir,
        )

        print(
            "Committed final isolated MouthOpenness outputs."
        )

        print()
        print(
            "=== Mouth Openness Isolated Component Results ==="
        )

        print(
            f"Actors: {summary['actors']}"
        )

        print(
            "Paired speech trials: "
            f"{summary['paired_speech_trials']}"
        )

        print(
            "Unique annotated frames: "
            f"{summary['unique_annotated_frames']}"
        )

        print(
            "Landmarks available rows: "
            f"{summary['landmarks_available_rows']}"
        )

        print(
            "MouthOpenness available rows: "
            f"{summary['mouth_openness_available_rows']}"
        )

        print(
            "Execution failures: "
            f"{summary['execution_failures']}"
        )

        print(
            "Component unavailable rows: "
            f"{summary['component_unavailable_rows']}"
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
                f"actors={manifest_item['first_actor']}-"
                f"{manifest_item['last_actor']} | "
                f"rows={manifest_item['row_count']} | "
                f"size={manifest_item['size_mib']:.2f} MiB"
            )

        print(
            COMPONENT_RESULTS_DIR
            / SUMMARY_FILENAME
        )

    finally:
        detector.clear()

        if pipeline is not None:
            pipeline.close()

        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


def parse_args(
) -> argparse.Namespace:
    """Parse isolated component execution mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated PhysioTrack MouthOpenness component execution "
            "on the complete FELT/RAVDESS speech population."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate project paths, both datasets, paired trials, "
            "population counts, and model provenance without inference."
        ),
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small real FaceAnalysis MouthOpenness test without "
            "writing final outputs."
        ),
    )

    parser.add_argument(
        "--smoke-count",
        type=int,
        default=3,
        help=(
            "Number of successful MouthOpenness samples required during "
            "--smoke-test."
        ),
    )

    return parser.parse_args()


def main(
) -> None:
    """Run the requested isolated MouthOpenness verification mode."""
    args = parse_args()

    preflight = dataset_preflight()

    print(
        "FELT/RAVDESS isolated MouthOpenness preflight: PASS"
    )

    print(
        f"FELT dataset root: {FELT_ROOT}"
    )

    print(
        f"RAVDESS dataset root: {RAVDESS_ROOT}"
    )

    print(
        f"Actors: {preflight['actors']}"
    )

    print(
        f"Paired speech trials: {preflight['trials']}"
    )

    print(
        "Raw FELT annotation rows: "
        f"{preflight['raw_annotation_rows']}"
    )

    print(
        "Unique annotated frames: "
        f"{preflight['unique_annotated_frames']}"
    )

    print(
        "Duplicate annotation rows resolved: "
        f"{preflight['duplicate_rows_resolved']}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )

    print(
        "Target component: MouthOpenness"
    )

    print(
        "Required prerequisite: FaceLandmarks"
    )

    print(
        "Controlled input: accepted FELT FaceRect bounding boxes"
    )

    print(
        "Mouth geometry: current image-dimension-aware "
        "pixel-consistent PhysioTrack implementation"
    )

    print(
        "Unrelated optional components: disabled"
    )

    print(
        "Accuracy metrics: not computed"
    )

    print(
        "Git-safe result policy: split only between complete actors "
        f"if a CSV would exceed {GIT_SAFE_MAX_FILE_SIZE_MIB:.0f} MiB"
    )

    if args.preflight_only:
        print(
            "Preflight-only mode: no landmark or MouthOpenness "
            "inference was run."
        )

        return

    if args.smoke_test:
        run_smoke_test(
            preflight,
            args.smoke_count,
        )

        return

    run_full(
        preflight
    )


if __name__ == "__main__":
    main()
