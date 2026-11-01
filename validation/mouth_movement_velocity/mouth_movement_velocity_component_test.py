from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
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
    "mouth_movement_velocity_component_results.csv"
)

SUMMARY_FILENAME = (
    "mouth_movement_velocity_component_summary.json"
)

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

EXPECTED_FPS = 30000.0 / 1001.0
FPS_TOLERANCE = 1e-4

EXPECTED_LANDMARK_COUNT = 478

EXPECTED_LANDMARK_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)

GIT_SAFE_MAX_FILE_SIZE_MIB = 90.0
GIT_SAFE_MAX_FILE_SIZE_BYTES = int(
    GIT_SAFE_MAX_FILE_SIZE_MIB
    * 1024
    * 1024
)

NUMERIC_TOLERANCE = 1e-12
VELOCITY_TOLERANCE = 1e-10

DEVICE = "cpu"

REQUIRED_FELT_COLUMNS = {
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
    "mouth_motion_available",
    "mouth_movement",
    "mouth_velocity",
    "temporal_segment_start",
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
    """Return one accepted FELT FaceRect to FaceAnalysis."""

    def __init__(
        self,
    ) -> None:
        self.instance: Instance | None = None

    def set_face(
        self,
        person_id: str,
        box: np.ndarray,
        confidence: float,
    ) -> None:
        """Set the controlled face instance for the next frame."""
        self.instance = Instance(
            id=person_id,
            box=np.asarray(
                box,
                dtype=float,
            ),
            confidence=float(
                confidence
            ),
            cls=0,
            cls_name="face",
        )

    def clear(
        self,
    ) -> None:
        """Return no face on the next frame."""
        self.instance = None

    def predict(
        self,
        frame: np.ndarray,
    ) -> Result:
        """Return the currently configured controlled face."""
        instances = (
            []
            if self.instance is None
            else [
                self.instance
            ]
        )

        return Result(
            orig_img=frame,
            instances=instances,
            task="face",
        )


def finite_numeric(
    value: Any,
) -> bool:
    """Return True for a finite non-boolean real value."""
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


def file_sha256(
    path: Path,
) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as file:
        while True:
            block = file.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def dataset_inventory(
    root: Path,
) -> list[
    tuple[
        str,
        int,
        int,
    ]
]:
    """Record relative path, size, and modification time for all dataset files."""
    return [
        (
            path.relative_to(
                root
            ).as_posix(),
            int(
                path.stat().st_size
            ),
            int(
                path.stat().st_mtime_ns
            ),
        )
        for path in sorted(
            root.rglob(
                "*"
            )
        )
        if path.is_file()
    ]


def resolve_duplicate_frames(
    table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """Resolve duplicate FELT frame IDs with the accepted deterministic rule."""
    resolved = table.copy()

    resolved[
        "_face_area"
    ] = (
        resolved[
            "FaceRectWidth"
        ]
        * resolved[
            "FaceRectHeight"
        ]
    )

    duplicate_count_by_frame = (
        resolved.groupby(
            "frame"
        )
        .size()
        .to_dict()
    )

    duplicate_rows_resolved = int(
        len(
            resolved
        )
        - resolved[
            "frame"
        ].nunique()
    )

    resolved = (
        resolved.sort_values(
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

    resolved[
        "duplicate_candidates"
    ] = resolved[
        "frame"
    ].map(
        duplicate_count_by_frame
    ).astype(
        int
    )

    resolved = resolved.drop(
        columns=[
            "_face_area"
        ]
    )

    return (
        resolved,
        duplicate_rows_resolved,
    )


def validate_and_resolve_annotation(
    csv_path: Path,
) -> tuple[
    pd.DataFrame,
    int,
]:
    """Load one FELT annotation and return accepted unique frame rows."""
    table = pd.read_csv(
        csv_path
    )

    missing = sorted(
        REQUIRED_FELT_COLUMNS
        - set(
            table.columns
        )
    )

    if missing:
        raise RuntimeError(
            "FELT annotation is missing required columns in "
            f"{csv_path}: {missing}"
        )

    numeric_columns = [
        "frame",
        "FaceRectX",
        "FaceRectY",
        "FaceRectWidth",
        "FaceRectHeight",
        "FaceScore",
    ]

    for column in numeric_columns:
        table[
            column
        ] = pd.to_numeric(
            table[
                column
            ],
            errors="coerce",
        )

        values = table[
            column
        ].to_numpy(
            dtype=np.float64
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                "FELT annotation contains non-finite values in "
                f"{column}: {csv_path}"
            )

    frame_values = table[
        "frame"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        frame_values
        == np.floor(
            frame_values
        )
    ):
        raise RuntimeError(
            f"FELT frame IDs must be integers: {csv_path}"
        )

    table[
        "frame"
    ] = frame_values.astype(
        np.int64
    )

    if np.any(
        table[
            "FaceRectWidth"
        ].to_numpy(
            dtype=np.float64
        )
        <= 0
    ):
        raise RuntimeError(
            f"FELT FaceRectWidth must be positive: {csv_path}"
        )

    if np.any(
        table[
            "FaceRectHeight"
        ].to_numpy(
            dtype=np.float64
        )
        <= 0
    ):
        raise RuntimeError(
            f"FELT FaceRectHeight must be positive: {csv_path}"
        )

    (
        table,
        duplicate_rows_resolved,
    ) = resolve_duplicate_frames(
        table
    )

    frame_ids = table[
        "frame"
    ].to_numpy(
        dtype=np.int64
    )

    expected_ids = np.arange(
        len(
            table
        ),
        dtype=np.int64,
    )

    if not np.array_equal(
        frame_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "FELT frame IDs must be contiguous and zero-based after "
            f"duplicate resolution: {csv_path}"
        )

    return (
        table,
        duplicate_rows_resolved,
    )


def resolve_landmark_model() -> Path:
    """Resolve and verify the accepted PhysioTrack FaceLandmarks model."""
    model_path = Path(
        Models.resolve(
            Models.Face.MediaPipe.Landmarks.face_landmarker
        )
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face landmarker model was not found: {model_path}"
        )

    model_hash = file_sha256(
        model_path
    )

    if (
        model_hash
        != EXPECTED_LANDMARK_MODEL_SHA256
    ):
        raise RuntimeError(
            "Face landmarker model hash changed. "
            f"Expected {EXPECTED_LANDMARK_MODEL_SHA256}, found {model_hash}."
        )

    return model_path


def make_config(
    ) -> FaceAnalysisConfig:
    """Build the isolated MouthMovement/Velocity FaceAnalysis configuration."""
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
        mouth_motion=True,
        emotion=False,
        regions=False,
        temporal=False,
    )

    config.validate()

    return config


def make_pipeline(
    detector: ControlledFaceDetector,
    model_path: Path,
) -> FaceAnalysis:
    """Build the real FaceAnalysis path with only required components enabled."""
    pipeline = FaceAnalysis(
        detector=detector,
        config=make_config(),
        fps=EXPECTED_FPS,
        landmark_model_path=model_path,
        device=DEVICE,
        verbose=False,
    )

    validate_pipeline_configuration(
        pipeline
    )

    return pipeline


def validate_pipeline_configuration(
    pipeline: FaceAnalysis,
) -> None:
    """Require MouthMovement and only its unavoidable upstream dependencies."""
    if not isinstance(
        pipeline.detector,
        ControlledFaceDetector,
    ):
        raise RuntimeError(
            "The isolated execution must use the controlled FELT FaceRect "
            "detector."
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
            "FaceLandmarks is a required prerequisite."
        )

    if pipeline.quality is not None:
        raise RuntimeError(
            "Face quality must be disabled."
        )

    if pipeline.eyes is not None:
        raise RuntimeError(
            "EyeOpenness must be disabled."
        )

    if pipeline.blink is not None:
        raise RuntimeError(
            "Blink must be disabled."
        )

    if pipeline.gaze is not None:
        raise RuntimeError(
            "Legacy gaze must be disabled."
        )

    if pipeline.gaze_estimation is not None:
        raise RuntimeError(
            "Learned gaze estimation must be disabled."
        )

    if pipeline.mouth is None:
        raise RuntimeError(
            "MouthOpenness is a required prerequisite."
        )

    if pipeline.mouth_motion is None:
        raise RuntimeError(
            "MouthMovement must be enabled."
        )

    if pipeline.emotion is not None:
        raise RuntimeError(
            "Emotion must be disabled."
        )

    if pipeline.regions is not None:
        raise RuntimeError(
            "Face regions must be disabled."
        )

    if pipeline.temporal is not None:
        raise RuntimeError(
            "FaceTemporalAggregator must be disabled."
        )

    if not math.isclose(
        float(
            pipeline.fps
        ),
        EXPECTED_FPS,
        rel_tol=0.0,
        abs_tol=FPS_TOLERANCE,
    ):
        raise RuntimeError(
            "FaceAnalysis FPS differs from the locked RAVDESS FPS."
        )


def preflight_dataset(
    ) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    int,
]:
    """Validate the complete FELT/RAVDESS speech population without inference."""
    if not FELT_ROOT.is_dir():
        raise FileNotFoundError(
            "FELT speech root not found:\n"
            f"{FELT_ROOT}"
        )

    if not RAVDESS_ROOT.is_dir():
        raise FileNotFoundError(
            "RAVDESS speech root not found:\n"
            f"{RAVDESS_ROOT}"
        )

    actor_dirs = sorted(
        path.name
        for path in FELT_ROOT.iterdir()
        if path.is_dir()
    )

    if actor_dirs != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected FELT actor structure."
        )

    ravdess_actor_dirs = sorted(
        path.name
        for path in RAVDESS_ROOT.iterdir()
        if path.is_dir()
    )

    if ravdess_actor_dirs != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected RAVDESS actor structure."
        )

    trials = []
    raw_rows = 0
    unique_rows = 0
    duplicate_rows_resolved = 0

    for actor in EXPECTED_ACTORS:
        annotation_paths = sorted(
            (
                FELT_ROOT
                / actor
            ).glob(
                "*.csv"
            )
        )

        if (
            len(
                annotation_paths
            )
            != EXPECTED_TRIALS_PER_ACTOR
        ):
            raise RuntimeError(
                f"Unexpected FELT trial count for {actor}."
            )

        for csv_path in annotation_paths:
            raw_table = pd.read_csv(
                csv_path,
                usecols=lambda column: (
                    column
                    in REQUIRED_FELT_COLUMNS
                ),
            )

            raw_rows += len(
                raw_table
            )

            (
                table,
                duplicates,
            ) = validate_and_resolve_annotation(
                csv_path
            )

            duplicate_rows_resolved += duplicates
            unique_rows += len(
                table
            )

            trial = csv_path.stem

            video_path = (
                RAVDESS_ROOT
                / actor
                / f"{trial}.mp4"
            )

            if not video_path.is_file():
                raise FileNotFoundError(
                    "Matching RAVDESS video was not found: "
                    f"{video_path}"
                )

            capture = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not capture.isOpened():
                raise RuntimeError(
                    "RAVDESS video could not be opened during preflight: "
                    f"{video_path}"
                )

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            frame_count = int(
                round(
                    capture.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
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

            if not math.isclose(
                fps,
                EXPECTED_FPS,
                rel_tol=0.0,
                abs_tol=FPS_TOLERANCE,
            ):
                raise RuntimeError(
                    "Unexpected RAVDESS FPS in "
                    f"{video_path}: {fps}"
                )

            if frame_count < len(
                table
            ):
                raise RuntimeError(
                    "RAVDESS frame count is shorter than FELT annotations: "
                    f"{video_path}"
                )

            if (
                image_width <= 0
                or image_height <= 0
            ):
                raise RuntimeError(
                    f"Invalid RAVDESS video dimensions: {video_path}"
                )

            trials.append(
                {
                    "actor":
                        actor,
                    "trial":
                        trial,
                    "csv_path":
                        csv_path,
                    "video_path":
                        video_path,
                    "annotated_frames":
                        len(
                            table
                        ),
                    "raw_rows":
                        len(
                            raw_table
                        ),
                    "duplicate_rows_resolved":
                        duplicates,
                    "image_width":
                        image_width,
                    "image_height":
                        image_height,
                }
            )

    if len(
        trials
    ) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(
            "Unexpected paired trial count: "
            f"{len(trials)} != {EXPECTED_TOTAL_TRIALS}"
        )

    if (
        raw_rows
        != EXPECTED_RAW_ANNOTATION_ROWS
    ):
        raise RuntimeError(
            "Unexpected raw FELT annotation row count: "
            f"{raw_rows} != {EXPECTED_RAW_ANNOTATION_ROWS}"
        )

    if (
        unique_rows
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Unexpected unique annotated-frame count: "
            f"{unique_rows} != {EXPECTED_UNIQUE_ANNOTATED_FRAMES}"
        )

    if (
        duplicate_rows_resolved
        != EXPECTED_DUPLICATE_ROWS_RESOLVED
    ):
        raise RuntimeError(
            "Unexpected duplicate-row count: "
            f"{duplicate_rows_resolved} != "
            f"{EXPECTED_DUPLICATE_ROWS_RESOLVED}"
        )

    return (
        trials,
        duplicate_rows_resolved,
    )


def base_row(
    actor: str,
    trial: str,
    annotation_row: pd.Series,
    fps: float,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[
    str,
    Any,
]:
    """Create one structured isolated-execution result row."""
    frame_index = int(
        annotation_row[
            "frame"
        ]
    )

    x1 = float(
        annotation_row[
            "FaceRectX"
        ]
    )

    y1 = float(
        annotation_row[
            "FaceRectY"
        ]
    )

    x2 = (
        x1
        + float(
            annotation_row[
                "FaceRectWidth"
            ]
        )
    )

    y2 = (
        y1
        + float(
            annotation_row[
                "FaceRectHeight"
            ]
        )
    )

    return {
        "actor":
            actor,
        "trial":
            trial,
        "frame":
            frame_index,
        "timestamp_seconds":
            (
                float(
                    frame_index
                )
                / float(
                    fps
                )
            ),
        "fps":
            float(
                fps
            ),
        "image_width":
            image_width,
        "image_height":
            image_height,
        "duplicate_candidates":
            int(
                annotation_row[
                    "duplicate_candidates"
                ]
            ),
        "FaceScore":
            float(
                annotation_row[
                    "FaceScore"
                ]
            ),
        "face_rect_x1":
            x1,
        "face_rect_y1":
            y1,
        "face_rect_x2":
            x2,
        "face_rect_y2":
            y2,
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
        "mouth_motion_available":
            False,
        "mouth_movement":
            None,
        "mouth_velocity":
            None,
        "temporal_segment_start":
            False,
        "status":
            "UNPROCESSED",
        "failure_reason":
            "",
    }


def populate_from_instance(
    row: dict[
        str,
        Any,
    ],
    instance: Instance,
) -> None:
    """Copy real FaceAnalysis MouthOpenness and MouthMovement values."""
    features = (
        instance.face_features
        if isinstance(
            instance.face_features,
            dict,
        )
        else {}
    )

    landmarks = features.get(
        "landmarks"
    )

    mouth = features.get(
        "mouth"
    )

    motion = features.get(
        "mouth_motion"
    )

    if not isinstance(
        landmarks,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis landmarks feature is missing."
        )

    if not isinstance(
        mouth,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis mouth feature is missing."
        )

    if not isinstance(
        motion,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis mouth_motion feature is missing."
        )

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
        "mouth_motion_available"
    ] = bool(
        motion.get(
            "available",
            False,
        )
    )

    if row[
        "landmarks_available"
    ]:
        if (
            row[
                "landmark_count"
            ]
            != EXPECTED_LANDMARK_COUNT
        ):
            raise RuntimeError(
                "Unexpected FaceLandmarks count: "
                f"{row['landmark_count']}"
            )

    for source, key in (
        (
            mouth,
            "mouth_openness",
        ),
        (
            mouth,
            "mouth_width",
        ),
        (
            mouth,
            "mouth_height",
        ),
        (
            motion,
            "mouth_movement",
        ),
        (
            motion,
            "mouth_velocity",
        ),
    ):
        value = source.get(
            key
        )

        row[
            key
        ] = (
            float(
                value
            )
            if value is not None
            else None
        )

    if row[
        "mouth_available"
    ]:
        for key in (
            "mouth_openness",
            "mouth_width",
            "mouth_height",
        ):
            if not finite_numeric(
                row[
                    key
                ]
            ):
                raise RuntimeError(
                    f"Available mouth output contains invalid {key}."
                )

        if float(
            row[
                "mouth_width"
            ]
        ) <= 0:
            raise RuntimeError(
                "Mouth width must be positive."
            )

        if float(
            row[
                "mouth_height"
            ]
        ) < 0:
            raise RuntimeError(
                "Mouth height must not be negative."
            )

        expected_openness = (
            float(
                row[
                    "mouth_height"
                ]
            )
            / float(
                row[
                    "mouth_width"
                ]
            )
        )

        if not math.isclose(
            float(
                row[
                    "mouth_openness"
                ]
            ),
            expected_openness,
            rel_tol=0.0,
            abs_tol=NUMERIC_TOLERANCE,
        ):
            raise RuntimeError(
                "MouthOpenness is inconsistent with mouth_height / "
                "mouth_width."
            )

    if row[
        "mouth_motion_available"
    ]:
        if not row[
            "mouth_available"
        ]:
            raise RuntimeError(
                "MouthMovement cannot be available when MouthOpenness is "
                "unavailable."
            )

        for key in (
            "mouth_movement",
            "mouth_velocity",
        ):
            if not finite_numeric(
                row[
                    key
                ]
            ):
                raise RuntimeError(
                    f"Available MouthMovement output contains invalid {key}."
                )

        if float(
            row[
                "mouth_movement"
            ]
        ) < 0:
            raise RuntimeError(
                "Mouth movement must not be negative."
            )

        if float(
            row[
                "mouth_velocity"
            ]
        ) < 0:
            raise RuntimeError(
                "Mouth velocity must not be negative."
            )

        row[
            "status"
        ] = "OK"

    elif not row[
        "mouth_available"
    ]:
        row[
            "status"
        ] = "NO_MOUTH_OUTPUT"

        row[
            "failure_reason"
        ] = (
            "FaceAnalysis produced no available MouthOpenness output; "
            "MouthMovement continuity is reset by the real pipeline."
        )

    else:
        row[
            "status"
        ] = "NO_MOUTH_MOTION_OUTPUT"

        row[
            "failure_reason"
        ] = (
            "FaceAnalysis produced MouthOpenness but no available "
            "MouthMovement output."
        )


def validate_trial_temporal_semantics(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Verify initialization, gap reset, movement, and velocity semantics."""
    previous_openness: float | None = None

    for row in rows:
        if not (
            row[
                "mouth_available"
            ]
            and row[
                "mouth_motion_available"
            ]
        ):
            previous_openness = None
            continue

        openness = float(
            row[
                "mouth_openness"
            ]
        )

        movement = float(
            row[
                "mouth_movement"
            ]
        )

        velocity = float(
            row[
                "mouth_velocity"
            ]
        )

        segment_start = (
            previous_openness
            is None
        )

        row[
            "temporal_segment_start"
        ] = bool(
            segment_start
        )

        if segment_start:
            if not math.isclose(
                movement,
                0.0,
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            ):
                raise RuntimeError(
                    "The first valid sample of a temporal segment must have "
                    "mouth_movement=0."
                )

            if not math.isclose(
                velocity,
                0.0,
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            ):
                raise RuntimeError(
                    "The first valid sample of a temporal segment must have "
                    "mouth_velocity=0."
                )

        else:
            expected_movement = abs(
                openness
                - previous_openness
            )

            expected_velocity = (
                expected_movement
                * EXPECTED_FPS
            )

            if not math.isclose(
                movement,
                expected_movement,
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            ):
                raise RuntimeError(
                    "MouthMovement output is inconsistent with consecutive "
                    "real FaceAnalysis MouthOpenness outputs."
                )

            if not math.isclose(
                velocity,
                expected_velocity,
                rel_tol=0.0,
                abs_tol=VELOCITY_TOLERANCE,
            ):
                raise RuntimeError(
                    "Mouth velocity is inconsistent with movement multiplied "
                    "by the locked FPS."
                )

        previous_openness = openness


def run_smoke_test(
    smoke_count: int,
) -> None:
    """Run a small real FaceAnalysis test including gap-reset semantics."""
    if smoke_count < 2:
        raise ValueError(
            "smoke_count must be at least 2."
        )

    (
        trials,
        _,
    ) = preflight_dataset()

    model_path = resolve_landmark_model()

    trial = trials[
        0
    ]

    (
        annotation,
        _,
    ) = validate_and_resolve_annotation(
        trial[
            "csv_path"
        ]
    )

    required_frames = (
        smoke_count
        + 1
    )

    if len(
        annotation
    ) < required_frames:
        raise RuntimeError(
            "The first trial does not contain enough frames for the requested "
            "smoke test."
        )

    detector = ControlledFaceDetector()

    pipeline = make_pipeline(
        detector,
        model_path,
    )

    capture = cv2.VideoCapture(
        str(
            trial[
                "video_path"
            ]
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open smoke-test video: "
            f"{trial['video_path']}"
        )

    rows = []

    try:
        for sample_index in range(
            smoke_count
        ):
            ok, frame = capture.read()

            if not ok:
                raise RuntimeError(
                    "Smoke-test video ended before required frames."
                )

            annotation_row = annotation.iloc[
                sample_index
            ]

            x1 = float(
                annotation_row[
                    "FaceRectX"
                ]
            )

            y1 = float(
                annotation_row[
                    "FaceRectY"
                ]
            )

            box = np.asarray(
                [
                    x1,
                    y1,
                    (
                        x1
                        + float(
                            annotation_row[
                                "FaceRectWidth"
                            ]
                        )
                    ),
                    (
                        y1
                        + float(
                            annotation_row[
                                "FaceRectHeight"
                            ]
                        )
                    ),
                ],
                dtype=float,
            )

            detector.set_face(
                trial[
                    "actor"
                ],
                box,
                float(
                    annotation_row[
                        "FaceScore"
                    ]
                ),
            )

            result = pipeline.predict(
                frame
            )

            if len(
                result
            ) != 1:
                raise RuntimeError(
                    "Smoke-test FaceAnalysis must return exactly one face."
                )

            row = base_row(
                trial[
                    "actor"
                ],
                trial[
                    "trial"
                ],
                annotation_row,
                EXPECTED_FPS,
                frame.shape[
                    1
                ],
                frame.shape[
                    0
                ],
            )

            populate_from_instance(
                row,
                result[
                    0
                ],
            )

            rows.append(
                row
            )

        validate_trial_temporal_semantics(
            rows
        )

        for index, row in enumerate(
            rows,
            start=1,
        ):
            print(
                "Smoke sample "
                f"{index}: "
                f"actor={row['actor']}, "
                f"trial={row['trial']}, "
                f"frame={row['frame']}, "
                f"mouth_openness={row['mouth_openness']}, "
                f"mouth_movement={row['mouth_movement']}, "
                f"mouth_velocity={row['mouth_velocity']}, "
                f"landmark_count={row['landmark_count']}"
            )

        detector.clear()

        gap_frame = frame.copy()

        gap_result = pipeline.predict(
            gap_frame
        )

        if len(
            gap_result
        ) != 0:
            raise RuntimeError(
                "Smoke-test missing-face gap did not produce an empty "
                "FaceAnalysis result."
            )

        ok, next_frame = capture.read()

        if not ok:
            raise RuntimeError(
                "Smoke-test video ended before the post-gap frame."
            )

        annotation_row = annotation.iloc[
            smoke_count
        ]

        x1 = float(
            annotation_row[
                "FaceRectX"
            ]
        )

        y1 = float(
            annotation_row[
                "FaceRectY"
            ]
        )

        box = np.asarray(
            [
                x1,
                y1,
                (
                    x1
                    + float(
                        annotation_row[
                            "FaceRectWidth"
                        ]
                    )
                ),
                (
                    y1
                    + float(
                        annotation_row[
                            "FaceRectHeight"
                        ]
                    )
                ),
            ],
            dtype=float,
        )

        detector.set_face(
            trial[
                "actor"
            ],
            box,
            float(
                annotation_row[
                    "FaceScore"
                ]
            ),
        )

        result = pipeline.predict(
            next_frame
        )

        if len(
            result
        ) != 1:
            raise RuntimeError(
                "Smoke-test post-gap FaceAnalysis must return exactly one "
                "face."
            )

        post_gap_row = base_row(
            trial[
                "actor"
            ],
            trial[
                "trial"
            ],
            annotation_row,
            EXPECTED_FPS,
            next_frame.shape[
                1
            ],
            next_frame.shape[
                0
            ],
        )

        populate_from_instance(
            post_gap_row,
            result[
                0
            ],
        )

        if not (
            post_gap_row[
                "mouth_motion_available"
            ]
            and math.isclose(
                float(
                    post_gap_row[
                        "mouth_movement"
                    ]
                ),
                0.0,
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            )
            and math.isclose(
                float(
                    post_gap_row[
                        "mouth_velocity"
                    ]
                ),
                0.0,
                rel_tol=0.0,
                abs_tol=NUMERIC_TOLERANCE,
            )
        ):
            raise RuntimeError(
                "A missing face did not break MouthMovement continuity. "
                "The first valid post-gap sample must restart at 0/0."
            )

        print(
            "Smoke gap-reset sample: "
            f"frame={post_gap_row['frame']}, "
            f"mouth_movement={post_gap_row['mouth_movement']}, "
            f"mouth_velocity={post_gap_row['mouth_velocity']}"
        )

        print(
            "Smoke test confirmed real MouthOpenness + MouthMovement outputs "
            "through FaceAnalysis."
        )

        print(
            "Smoke test confirmed missing-sample temporal continuity reset."
        )

        print(
            "Smoke test: PASS"
        )

    finally:
        capture.release()

        pipeline.reset_temporal_state()


def write_csv(
    path: Path,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Write one isolated component CSV."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def serialized_csv_size(
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
) -> int:
    """Return UTF-8 byte size for a result CSV without writing it."""
    buffer = io.StringIO(
        newline=""
    )

    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )

    return len(
        buffer.getvalue().encode(
            "utf-8"
        )
    )


def write_git_safe_results(
    staging_dir: Path,
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
    """Write one CSV or split only between complete actors if needed."""
    single_path = (
        staging_dir
        / RESULTS_FILENAME
    )

    write_csv(
        single_path,
        rows,
    )

    if (
        single_path.stat().st_size
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
                    rows[
                        0
                    ][
                        "actor"
                    ],
                "last_actor":
                    rows[
                        -1
                    ][
                        "actor"
                    ],
                "size_bytes":
                    int(
                        single_path.stat().st_size
                    ),
                "size_mib":
                    float(
                        single_path.stat().st_size
                        / 1024
                        / 1024
                    ),
            }
        ]

    single_path.unlink()

    rows_by_actor = {
        actor: [
            row
            for row in rows
            if row[
                "actor"
            ] == actor
        ]
        for actor in EXPECTED_ACTORS
    }

    actor_groups = []
    current_actors = []
    current_rows = []

    for actor in EXPECTED_ACTORS:
        candidate_rows = (
            current_rows
            + rows_by_actor[
                actor
            ]
        )

        candidate_size = serialized_csv_size(
            candidate_rows
        )

        if (
            current_rows
            and candidate_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            actor_groups.append(
                (
                    list(
                        current_actors
                    ),
                    list(
                        current_rows
                    ),
                )
            )

            current_actors = [
                actor
            ]

            current_rows = list(
                rows_by_actor[
                    actor
                ]
            )

        else:
            current_actors.append(
                actor
            )

            current_rows = candidate_rows

        if (
            serialized_csv_size(
                current_rows
            )
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                "A single complete actor result exceeds the configured "
                "Git-safe file-size threshold and cannot be split without "
                "violating actor-boundary preservation."
            )

    if current_rows:
        actor_groups.append(
            (
                current_actors,
                current_rows,
            )
        )

    manifest = []

    for part_number, (
        actors,
        part_rows,
    ) in enumerate(
        actor_groups,
        start=1,
    ):
        path = (
            staging_dir
            / (
                "mouth_movement_velocity_component_results_"
                f"part{part_number:03d}.csv"
            )
        )

        write_csv(
            path,
            part_rows,
        )

        if (
            path.stat().st_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                f"Generated Git-safe part still exceeds threshold: {path}"
            )

        manifest.append(
            {
                "filename":
                    path.name,
                "part_number":
                    part_number,
                "row_count":
                    len(
                        part_rows
                    ),
                "first_actor":
                    actors[
                        0
                    ],
                "last_actor":
                    actors[
                        -1
                    ],
                "size_bytes":
                    int(
                        path.stat().st_size
                    ),
                "size_mib":
                    float(
                        path.stat().st_size
                        / 1024
                        / 1024
                    ),
            }
        )

    return manifest


def load_staged_result_rows(
    staging_dir: Path,
    manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> pd.DataFrame:
    """Read the staged isolated outputs with round-trip float parsing."""
    tables = []

    for entry in manifest:
        path = (
            staging_dir
            / entry[
                "filename"
            ]
        )

        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Missing or empty staged component result: {path}"
            )

        table = pd.read_csv(
            path,
            float_precision="round_trip",
            keep_default_na=False,
        )

        if list(
            table.columns
        ) != CSV_FIELDS:
            raise RuntimeError(
                f"Staged component result schema changed: {path}"
            )

        if len(
            table
        ) != int(
            entry[
                "row_count"
            ]
        ):
            raise RuntimeError(
                f"Staged component result row count changed: {path}"
            )

        tables.append(
            table
        )

    if not tables:
        raise RuntimeError(
            "No staged component result files were generated."
        )

    return pd.concat(
        tables,
        ignore_index=True,
    )


def validate_staged_results(
    staging_dir: Path,
    manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> pd.DataFrame:
    """Validate coverage, schema, numerical semantics, and actor boundaries."""
    table = load_staged_result_rows(
        staging_dir,
        manifest,
    )

    if len(
        table
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Unexpected total staged row count."
        )

    if table.duplicated(
        [
            "actor",
            "trial",
            "frame",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in staged output."
        )

    actor_values = table[
        "actor"
    ].astype(
        str
    )

    if sorted(
        actor_values.unique().tolist()
    ) != EXPECTED_ACTORS:
        raise RuntimeError(
            "Staged output actor coverage changed."
        )

    trial_counts = (
        table[
            [
                "actor",
                "trial",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "actor"
        )
        .size()
        .to_dict()
    )

    for actor in EXPECTED_ACTORS:
        if (
            int(
                trial_counts.get(
                    actor,
                    0,
                )
            )
            != EXPECTED_TRIALS_PER_ACTOR
        ):
            raise RuntimeError(
                f"Unexpected staged trial count for {actor}."
            )

    status_counts = Counter(
        table[
            "status"
        ].astype(
            str
        ).tolist()
    )

    if (
        status_counts
        != Counter(
            {
                "OK":
                    EXPECTED_UNIQUE_ANNOTATED_FRAMES
            }
        )
    ):
        raise RuntimeError(
            "The accepted isolated execution requires every annotated frame "
            "to produce an available MouthMovement output. "
            f"Observed status counts: {dict(status_counts)}"
        )

    boolean_columns = [
        "landmarks_available",
        "mouth_available",
        "mouth_motion_available",
    ]

    for column in boolean_columns:
        normalized = table[
            column
        ].astype(
            str
        ).str.lower()

        if not (
            normalized
            == "true"
        ).all():
            raise RuntimeError(
                f"All staged rows must have {column}=True."
            )

    landmark_counts = pd.to_numeric(
        table[
            "landmark_count"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    if not np.all(
        landmark_counts
        == EXPECTED_LANDMARK_COUNT
    ):
        raise RuntimeError(
            "Staged landmark counts changed."
        )

    numeric_columns = [
        "fps",
        "image_width",
        "image_height",
        "FaceScore",
        "face_rect_x1",
        "face_rect_y1",
        "face_rect_x2",
        "face_rect_y2",
        "mouth_openness",
        "mouth_width",
        "mouth_height",
        "mouth_movement",
        "mouth_velocity",
    ]

    numeric_table = table[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if not np.all(
        np.isfinite(
            numeric_table.to_numpy(
                dtype=np.float64
            )
        )
    ):
        raise RuntimeError(
            "Staged component outputs contain non-finite numeric values."
        )

    if np.any(
        numeric_table[
            "mouth_width"
        ].to_numpy(
            dtype=np.float64
        )
        <= 0
    ):
        raise RuntimeError(
            "Staged mouth widths must be positive."
        )

    if np.any(
        numeric_table[
            "mouth_height"
        ].to_numpy(
            dtype=np.float64
        )
        < 0
    ):
        raise RuntimeError(
            "Staged mouth heights must not be negative."
        )

    openness = numeric_table[
        "mouth_openness"
    ].to_numpy(
        dtype=np.float64
    )

    expected_openness = (
        numeric_table[
            "mouth_height"
        ].to_numpy(
            dtype=np.float64
        )
        / numeric_table[
            "mouth_width"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if not np.allclose(
        openness,
        expected_openness,
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
    ):
        raise RuntimeError(
            "Staged MouthOpenness values are inconsistent with "
            "mouth_height / mouth_width."
        )

    for (
        _,
        trial_table,
    ) in table.groupby(
        [
            "actor",
            "trial",
        ],
        sort=False,
    ):
        trial_table = trial_table.sort_values(
            "frame"
        )

        expected_frames = np.arange(
            len(
                trial_table
            ),
            dtype=np.int64,
        )

        observed_frames = pd.to_numeric(
            trial_table[
                "frame"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=np.int64
        )

        if not np.array_equal(
            observed_frames,
            expected_frames,
        ):
            raise RuntimeError(
                "Staged trial frame coverage is not contiguous and "
                "zero-based."
            )

        trial_rows = trial_table.to_dict(
            orient="records"
        )

        for row in trial_rows:
            for key in (
                "landmarks_available",
                "mouth_available",
                "mouth_motion_available",
                "temporal_segment_start",
            ):
                value = str(
                    row[
                        key
                    ]
                ).strip().lower()

                row[
                    key
                ] = (
                    value
                    == "true"
                )

            for key in (
                "mouth_openness",
                "mouth_movement",
                "mouth_velocity",
            ):
                row[
                    key
                ] = float(
                    row[
                        key
                    ]
                )

        validate_trial_temporal_semantics(
            trial_rows
        )

        segment_start_values = [
            bool(
                row[
                    "temporal_segment_start"
                ]
            )
            for row in trial_rows
        ]

        if (
            sum(
                segment_start_values
            )
            != 1
            or not segment_start_values[
                0
            ]
        ):
            raise RuntimeError(
                "With complete accepted FELT/RAVDESS execution, each trial "
                "must contain exactly one temporal segment beginning at its "
                "first frame."
            )

    if len(
        manifest
    ) > 1:
        actor_to_part = {}

        for entry in manifest:
            part_path = (
                staging_dir
                / entry[
                    "filename"
                ]
            )

            part_table = pd.read_csv(
                part_path,
                usecols=[
                    "actor"
                ],
            )

            for actor in part_table[
                "actor"
            ].astype(
                str
            ).unique():
                if actor in actor_to_part:
                    raise RuntimeError(
                        "A complete actor was split across multiple Git-safe "
                        "result files."
                    )

                actor_to_part[
                    actor
                ] = int(
                    entry[
                        "part_number"
                    ]
                )

    return table


def build_summary(
    table: pd.DataFrame,
    manifest: list[
        dict[
            str,
            Any,
        ]
    ],
    runtime_seconds: float,
    model_path: Path,
) -> dict[
    str,
    Any,
]:
    """Build the isolated software-execution summary without accuracy metrics."""
    status_counts = Counter(
        table[
            "status"
        ].astype(
            str
        ).tolist()
    )

    landmarks_available = int(
        (
            table[
                "landmarks_available"
            ].astype(
                str
            ).str.lower()
            == "true"
        ).sum()
    )

    mouth_available = int(
        (
            table[
                "mouth_available"
            ].astype(
                str
            ).str.lower()
            == "true"
        ).sum()
    )

    motion_available = int(
        (
            table[
                "mouth_motion_available"
            ].astype(
                str
            ).str.lower()
            == "true"
        ).sum()
    )

    segment_starts = int(
        (
            table[
                "temporal_segment_start"
            ].astype(
                str
            ).str.lower()
            == "true"
        ).sum()
    )

    execution_failures = int(
        sum(
            count
            for status, count
            in status_counts.items()
            if status
            in {
                "VIDEO_OPEN_FAILED",
                "FRAME_READ_FAILED",
                "EXECUTION_FAILED",
            }
        )
    )

    component_unavailable_rows = int(
        len(
            table
        )
        - motion_available
    )

    overall_status = (
        "PASS"
        if (
            len(
                table
            )
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
            and landmarks_available
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
            and mouth_available
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
            and motion_available
            == EXPECTED_UNIQUE_ANNOTATED_FRAMES
            and segment_starts
            == EXPECTED_TOTAL_TRIALS
            and execution_failures
            == 0
            and component_unavailable_rows
            == 0
        )
        else "FAIL"
    )

    return {
        "component":
            "PhysioTrack MouthMovement / Mouth Velocity",
        "execution_type":
            "isolated component execution; not an accuracy benchmark",
        "dataset":
            "FELT/RAVDESS speech subset",
        "pipeline":
            "PhysioTrack FaceAnalysis",
        "physiotrack_source":
            "src/physiotrack/face/mouth_motion.py",
        "device":
            DEVICE,
        "controlled_input":
            "Accepted FELT FaceRect bounding boxes",
        "required_prerequisites":
            [
                "PhysioTrack FaceLandmarks",
                "PhysioTrack MouthOpenness",
            ],
        "enabled_components":
            [
                "landmarks",
                "mouth",
                "mouth_motion",
            ],
        "disabled_components":
            [
                "tracking",
                "head_pose",
                "quality",
                "eyes",
                "blink",
                "gaze",
                "gaze_estimation",
                "emotion",
                "regions",
                "temporal",
            ],
        "temporal_semantics":
            (
                "First valid sample of each temporal segment has "
                "mouth_movement=0 and mouth_velocity=0; missing mouth/face "
                "input breaks continuity; later valid samples use absolute "
                "openness change and elapsed frame time."
            ),
        "accuracy_metrics_computed":
            False,
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
        "locked_fps":
            EXPECTED_FPS,
        "landmark_model_file":
            model_path.name,
        "landmark_model_sha256":
            file_sha256(
                model_path
            ),
        "landmarks_available_rows":
            landmarks_available,
        "mouth_openness_available_rows":
            mouth_available,
        "mouth_movement_available_rows":
            motion_available,
        "temporal_segment_start_rows":
            segment_starts,
        "expected_temporal_segment_start_rows":
            EXPECTED_TOTAL_TRIALS,
        "status_counts":
            dict(
                sorted(
                    status_counts.items()
                )
            ),
        "execution_failures":
            execution_failures,
        "component_unavailable_rows":
            component_unavailable_rows,
        "runtime_seconds":
            float(
                runtime_seconds
            ),
        "result_output_policy":
            {
                "git_safe_max_file_size_mib":
                    GIT_SAFE_MAX_FILE_SIZE_MIB,
                "split_only_at_complete_actor_boundaries":
                    True,
                "result_file_count":
                    len(
                        manifest
                    ),
                "files":
                    manifest,
            },
        "overall_status":
            overall_status,
    }


def validate_summary(
    summary_path: Path,
    table: pd.DataFrame,
    manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:
    """Re-read and validate the staged summary JSON."""
    if (
        not summary_path.is_file()
        or summary_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Staged component summary is missing or empty."
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
            "Staged isolated component summary did not reach PASS."
        )

    if summary.get(
        "accuracy_metrics_computed"
    ) is not False:
        raise RuntimeError(
            "Isolated component execution must not compute accuracy metrics."
        )

    if int(
        summary.get(
            "unique_annotated_frames",
            -1,
        )
    ) != len(
        table
    ):
        raise RuntimeError(
            "Staged summary frame count does not match staged CSV rows."
        )

    if int(
        summary.get(
            "mouth_movement_available_rows",
            -1,
        )
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Staged summary MouthMovement availability count changed."
        )

    if int(
        summary.get(
            "temporal_segment_start_rows",
            -1,
        )
    ) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(
            "Staged summary initialization-segment count changed."
        )

    files = (
        summary.get(
            "result_output_policy",
            {}
        ).get(
            "files",
            []
        )
    )

    if files != manifest:
        raise RuntimeError(
            "Staged summary result manifest does not match generated files."
        )

    return summary


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Copy one file into place through an atomic same-directory rename."""
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


def commit_component_outputs(
    staging_dir: Path,
    manifest: list[
        dict[
            str,
            Any,
        ]
    ],
    summary_path: Path,
) -> None:
    """Transactionally replace only this script's component-execution files."""
    COMPONENT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    owned_patterns = [
        RESULTS_FILENAME,
        "mouth_movement_velocity_component_results_part*.csv",
        SUMMARY_FILENAME,
    ]

    old_files = []

    for pattern in owned_patterns:
        old_files.extend(
            COMPONENT_RESULTS_DIR.glob(
                pattern
            )
        )

    old_files = sorted(
        set(
            old_files
        )
    )

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_map = {}

    for old_path in old_files:
        backup_path = (
            backup_dir
            / old_path.name
        )

        shutil.copy2(
            old_path,
            backup_path,
        )

        backup_map[
            old_path.name
        ] = backup_path

    new_paths = [
        (
            staging_dir
            / entry[
                "filename"
            ],
            COMPONENT_RESULTS_DIR
            / entry[
                "filename"
            ],
        )
        for entry in manifest
    ]

    new_paths.append(
        (
            summary_path,
            COMPONENT_RESULTS_DIR
            / SUMMARY_FILENAME,
        )
    )

    installed_names = []

    try:
        for source_path, destination_path in new_paths:
            atomic_copy_file(
                source_path,
                destination_path,
            )

            installed_names.append(
                destination_path.name
            )

        final_names = set(
            installed_names
        )

        for old_path in old_files:
            if (
                old_path.name
                not in final_names
                and old_path.exists()
            ):
                old_path.unlink()

    except Exception:
        for name in installed_names:
            installed_path = (
                COMPONENT_RESULTS_DIR
                / name
            )

            backup_path = backup_map.get(
                name
            )

            if backup_path is not None:
                shutil.copy2(
                    backup_path,
                    installed_path,
                )

            elif installed_path.exists():
                installed_path.unlink()

        for old_name, backup_path in backup_map.items():
            destination_path = (
                COMPONENT_RESULTS_DIR
                / old_name
            )

            if not destination_path.exists():
                shutil.copy2(
                    backup_path,
                    destination_path,
                )

        raise


def run_full_execution(
    trials: list[
        dict[
            str,
            Any,
        ]
    ],
    duplicate_rows_resolved: int,
) -> None:
    """Run isolated MouthMovement/Velocity through real FaceAnalysis."""
    model_path = resolve_landmark_model()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_context = tempfile.TemporaryDirectory(
        prefix=".mouth_movement_velocity_component_staging_",
        dir=RESULTS_DIR,
    )

    staging_dir = Path(
        staging_context.name
    )

    print(
        f"Staging directory created: {staging_dir}"
    )

    print(
        "Previous accepted isolated component outputs are preserved until "
        "staged validation passes."
    )

    felt_inventory_before = dataset_inventory(
        FELT_ROOT
    )

    ravdess_inventory_before = dataset_inventory(
        RAVDESS_ROOT
    )

    detector = ControlledFaceDetector()

    pipeline = make_pipeline(
        detector,
        model_path,
    )

    rows = []

    start_time = time.perf_counter()

    try:
        for trial_number, trial in enumerate(
            trials,
            start=1,
        ):
            (
                annotation,
                trial_duplicates,
            ) = validate_and_resolve_annotation(
                trial[
                    "csv_path"
                ]
            )

            if (
                trial_duplicates
                != int(
                    trial[
                        "duplicate_rows_resolved"
                    ]
                )
            ):
                raise RuntimeError(
                    "Duplicate-resolution count changed between preflight and "
                    f"execution for {trial['actor']}/{trial['trial']}."
                )

            pipeline.reset_temporal_state()

            capture = cv2.VideoCapture(
                str(
                    trial[
                        "video_path"
                    ]
                )
            )

            trial_rows = []

            if not capture.isOpened():
                for _, annotation_row in annotation.iterrows():
                    row = base_row(
                        trial[
                            "actor"
                        ],
                        trial[
                            "trial"
                        ],
                        annotation_row,
                        EXPECTED_FPS,
                        trial[
                            "image_width"
                        ],
                        trial[
                            "image_height"
                        ],
                    )

                    row[
                        "status"
                    ] = "VIDEO_OPEN_FAILED"

                    row[
                        "failure_reason"
                    ] = (
                        "RAVDESS video could not be opened during isolated "
                        "component execution."
                    )

                    trial_rows.append(
                        row
                    )

                rows.extend(
                    trial_rows
                )

                pipeline.reset_temporal_state()
                continue

            try:
                for frame_index, annotation_row in annotation.iterrows():
                    ok, frame = capture.read()

                    row = base_row(
                        trial[
                            "actor"
                        ],
                        trial[
                            "trial"
                        ],
                        annotation_row,
                        EXPECTED_FPS,
                        (
                            frame.shape[
                                1
                            ]
                            if ok
                            else trial[
                                "image_width"
                            ]
                        ),
                        (
                            frame.shape[
                                0
                            ]
                            if ok
                            else trial[
                                "image_height"
                            ]
                        ),
                    )

                    if not ok:
                        row[
                            "status"
                        ] = "FRAME_READ_FAILED"

                        row[
                            "failure_reason"
                        ] = (
                            "RAVDESS video ended before the accepted FELT "
                            "annotation sequence."
                        )

                        trial_rows.append(
                            row
                        )

                        for remaining_index in range(
                            frame_index
                            + 1,
                            len(
                                annotation
                            ),
                        ):
                            remaining_row = base_row(
                                trial[
                                    "actor"
                                ],
                                trial[
                                    "trial"
                                ],
                                annotation.iloc[
                                    remaining_index
                                ],
                                EXPECTED_FPS,
                                trial[
                                    "image_width"
                                ],
                                trial[
                                    "image_height"
                                ],
                            )

                            remaining_row[
                                "status"
                            ] = "FRAME_READ_FAILED"

                            remaining_row[
                                "failure_reason"
                            ] = (
                                "RAVDESS video ended before the accepted FELT "
                                "annotation sequence."
                            )

                            trial_rows.append(
                                remaining_row
                            )

                        pipeline.reset_temporal_state()
                        break

                    x1 = float(
                        annotation_row[
                            "FaceRectX"
                        ]
                    )

                    y1 = float(
                        annotation_row[
                            "FaceRectY"
                        ]
                    )

                    box = np.asarray(
                        [
                            x1,
                            y1,
                            (
                                x1
                                + float(
                                    annotation_row[
                                        "FaceRectWidth"
                                    ]
                                )
                            ),
                            (
                                y1
                                + float(
                                    annotation_row[
                                        "FaceRectHeight"
                                    ]
                                )
                            ),
                        ],
                        dtype=float,
                    )

                    detector.set_face(
                        trial[
                            "actor"
                        ],
                        box,
                        float(
                            annotation_row[
                                "FaceScore"
                            ]
                        ),
                    )

                    try:
                        result = pipeline.predict(
                            frame
                        )

                        if len(
                            result
                        ) != 1:
                            raise RuntimeError(
                                "FaceAnalysis output count does not match the "
                                "single controlled FELT FaceRect."
                            )

                        instance = result[
                            0
                        ]

                        if str(
                            instance.id
                        ) != trial[
                            "actor"
                        ]:
                            raise RuntimeError(
                                "FaceAnalysis output ID changed from the "
                                "controlled actor ID."
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

                        pipeline.reset_temporal_state()

                    trial_rows.append(
                        row
                    )

            finally:
                capture.release()

            validate_trial_temporal_semantics(
                trial_rows
            )

            rows.extend(
                trial_rows
            )

            pipeline.reset_temporal_state()

            if (
                trial_number % 20
                == 0
                or trial_number
                == EXPECTED_TOTAL_TRIALS
            ):
                available = sum(
                    1
                    for row in rows
                    if row[
                        "mouth_motion_available"
                    ]
                )

                print(
                    f"Trials: {trial_number}/{EXPECTED_TOTAL_TRIALS} "
                    f"| rows={len(rows)} "
                    f"| mouth_motion_available={available}"
                )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        if len(
            rows
        ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
            raise RuntimeError(
                "Isolated execution row accounting changed before staging."
            )

        if (
            duplicate_rows_resolved
            != EXPECTED_DUPLICATE_ROWS_RESOLVED
        ):
            raise RuntimeError(
                "Preflight duplicate-resolution count changed unexpectedly."
            )

        felt_inventory_after = dataset_inventory(
            FELT_ROOT
        )

        ravdess_inventory_after = dataset_inventory(
            RAVDESS_ROOT
        )

        if (
            felt_inventory_before
            != felt_inventory_after
        ):
            raise RuntimeError(
                "FELT dataset inventory changed during isolated execution."
            )

        if (
            ravdess_inventory_before
            != ravdess_inventory_after
        ):
            raise RuntimeError(
                "RAVDESS dataset inventory changed during isolated execution."
            )

        manifest = write_git_safe_results(
            staging_dir,
            rows,
        )

        print(
            "Validating staged isolated MouthMovement/Velocity outputs..."
        )

        staged_table = validate_staged_results(
            staging_dir,
            manifest,
        )

        summary = build_summary(
            staged_table,
            manifest,
            runtime_seconds,
            model_path,
        )

        if summary[
            "overall_status"
        ] != "PASS":
            raise RuntimeError(
                "Isolated MouthMovement/Velocity execution did not reach PASS. "
                "Prior accepted outputs were preserved."
            )

        summary_path = (
            staging_dir
            / SUMMARY_FILENAME
        )

        with summary_path.open(
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

        validate_summary(
            summary_path,
            staged_table,
            manifest,
        )

        print(
            "Staged isolated output validation: PASS"
        )

        commit_component_outputs(
            staging_dir,
            manifest,
            summary_path,
        )

        print(
            "Committed final isolated MouthMovement/Velocity outputs."
        )

        print()
        print(
            "=== Mouth Movement / Velocity Isolated Component Results ==="
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
            "MouthMovement available rows: "
            f"{summary['mouth_movement_available_rows']}"
        )

        print(
            "Temporal segment starts: "
            f"{summary['temporal_segment_start_rows']}"
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
            f"Runtime: {runtime_seconds / 60.0:.2f} minutes"
        )

        print(
            f"Overall status: {summary['overall_status']}"
        )

        print()
        print(
            "Saved:"
        )

        for entry in manifest:
            path = (
                COMPONENT_RESULTS_DIR
                / entry[
                    "filename"
                ]
            )

            print(
                f"{path}"
            )

            print(
                "  actors="
                f"{entry['first_actor']}-{entry['last_actor']} "
                f"| rows={entry['row_count']} "
                f"| size={entry['size_mib']:.2f} MiB"
            )

        print(
            COMPONENT_RESULTS_DIR
            / SUMMARY_FILENAME
        )

    finally:
        pipeline.reset_temporal_state()

        staging_context.cleanup()


def print_preflight(
    trials: list[
        dict[
            str,
            Any,
        ]
    ],
    duplicate_rows_resolved: int,
) -> None:
    """Print the locked isolated-execution preflight state."""
    model_path = resolve_landmark_model()

    print(
        "FELT/RAVDESS isolated MouthMovement/Velocity preflight: PASS"
    )

    print(
        f"FELT root: {FELT_ROOT}"
    )

    print(
        f"RAVDESS root: {RAVDESS_ROOT}"
    )

    print(
        f"Actors: {len(EXPECTED_ACTORS)}"
    )

    print(
        f"Paired speech trials: {len(trials)}"
    )

    print(
        "Raw FELT annotation rows: "
        f"{EXPECTED_RAW_ANNOTATION_ROWS}"
    )

    print(
        "Unique annotated frames: "
        f"{EXPECTED_UNIQUE_ANNOTATED_FRAMES}"
    )

    print(
        "Duplicate annotation rows resolved: "
        f"{duplicate_rows_resolved}"
    )

    print(
        f"FPS: {EXPECTED_FPS:.12f}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )

    print(
        "Target component: MouthMovement / mouth velocity"
    )

    print(
        "Required prerequisites: FaceLandmarks + MouthOpenness"
    )

    print(
        "Controlled input: accepted FELT FaceRect bounding boxes"
    )

    print(
        "Temporal rule: first valid segment sample -> movement=0, "
        "velocity=0; missing input breaks continuity"
    )

    print(
        "Unrelated optional components: disabled"
    )

    print(
        "Accuracy metrics: not computed"
    )

    print(
        "Git-safe result policy: split only between complete actors if a CSV "
        f"would exceed {GIT_SAFE_MAX_FILE_SIZE_MIB:.0f} MiB"
    )

    print(
        f"Face landmarker model: {model_path.name}"
    )

    print(
        "Face landmarker SHA256: "
        f"{EXPECTED_LANDMARK_MODEL_SHA256}"
    )


def main(
    ) -> None:
    """Run preflight, smoke, or the full isolated component execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated PhysioTrack MouthMovement and mouth-velocity "
            "execution on the complete paired FELT/RAVDESS speech population."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate dataset/model/configuration prerequisites without "
            "running FaceLandmarks/MouthOpenness/MouthMovement inference."
        ),
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small real FaceAnalysis MouthMovement test before the full "
            "isolated execution."
        ),
    )

    parser.add_argument(
        "--smoke-count",
        type=int,
        default=3,
        help=(
            "Number of consecutive real output samples used by --smoke-test "
            "before the explicit gap-reset check."
        ),
    )

    args = parser.parse_args()

    (
        trials,
        duplicate_rows_resolved,
    ) = preflight_dataset()

    print_preflight(
        trials,
        duplicate_rows_resolved,
    )

    if args.preflight_only:
        print()

        print(
            "Preflight-only: no FaceLandmarks, MouthOpenness, or "
            "MouthMovement inference was run."
        )

        return

    if args.smoke_test:
        print()

        run_smoke_test(
            args.smoke_count
        )

        return

    print()

    run_full_execution(
        trials,
        duplicate_rows_resolved,
    )


if __name__ == "__main__":
    main()
