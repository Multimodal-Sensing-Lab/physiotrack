from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

DATASET_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "MPIIFaceGaze"
    / "Data"
)

RESULTS_DIR = SCRIPT_DIR / "results"
COMPONENT_RESULTS_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_FILENAME = (
    "gaze_estimation_component_results.csv"
)

RESULTS_PART_PREFIX = (
    "gaze_estimation_component_results_part"
)

GIT_SAFE_MAX_FILE_SIZE_MIB = 90.0
GIT_SAFE_MAX_FILE_SIZE_BYTES = int(
    GIT_SAFE_MAX_FILE_SIZE_MIB
    * 1024
    * 1024
)

SUMMARY_FILENAME = (
    "gaze_estimation_component_summary.json"
)

EXPECTED_PARTICIPANTS = [
    f"p{index:02d}"
    for index in range(15)
]

EXPECTED_TOTAL_ANNOTATIONS = 37667

EXPECTED_PTGAZE_VERSION = "0.3.0"

MODEL_MODE = "eth-xgaze"
DEVICE = "cpu"
MIN_IOU = 0.10

EXPECTED_CHECKPOINT_FILENAME = (
    "model.safetensors"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc"
)

CSV_FIELDS = [
    "participant",
    "image_relative_path",
    "image_width",
    "image_height",
    "face_count",
    "face_index",
    "face_id",
    "detector_confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "gaze_estimation_available",
    "gaze_x",
    "gaze_y",
    "gaze_z",
    "gaze_vector_norm",
    "pitch_deg",
    "yaw_deg",
    "association_iou",
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


def sha256_file(
    path: Path,
) -> str:
    """Return SHA256 for one file."""
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        for chunk in iter(
            lambda: file.read(
                1024
                * 1024
            ),
            b"",
        ):
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
    """Return a read-only dataset file inventory."""
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


def parse_annotation_image_path(
    line: str,
) -> str:
    """Validate one MPIIFaceGaze row and return its image path."""
    parts = line.strip().split()

    if len(
        parts
    ) != 28:
        raise ValueError(
            "Unexpected annotation field count: "
            f"{len(parts)}"
        )

    values = np.asarray(
        [
            float(
                value
            )
            for value in parts[
                1:27
            ]
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "Annotation contains non-finite numeric values."
        )

    return str(
        parts[
            0
        ]
    )


def load_annotation_lines(
    person_dir: Path,
) -> list[str]:
    """Load one participant annotation file."""
    annotation_path = (
        person_dir
        / f"{person_dir.name}.txt"
    )

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation file: {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def dataset_preflight() -> dict[
    str,
    Any,
]:
    """Validate the exact accepted MPIIFaceGaze population."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "MPIIFaceGaze dataset was not found at the expected "
            "project-relative location:\n"
            f"{DATASET_ROOT}"
        )

    generated_camera_files = list(
        DATASET_ROOT.rglob(
            "ptgaze_camera.yaml"
        )
    )

    if generated_camera_files:
        raise RuntimeError(
            "Dataset cleanliness check failed: generated "
            "ptgaze_camera.yaml files were found inside the dataset."
        )

    person_dirs = sorted(
        [
            path
            for path in DATASET_ROOT.glob(
                "p*"
            )
            if path.is_dir()
        ],
        key=lambda path: path.name,
    )

    participant_names = [
        path.name
        for path in person_dirs
    ]

    if (
        participant_names
        != EXPECTED_PARTICIPANTS
    ):
        raise RuntimeError(
            "Unexpected participant structure. "
            f"Expected {EXPECTED_PARTICIPANTS}, "
            f"found {participant_names}."
        )

    annotations_by_person: dict[
        str,
        list[str],
    ] = {}

    total_annotations = 0

    required_calibration_files = (
        "Camera.mat",
        "monitorPose.mat",
        "screenSize.mat",
    )

    for person_dir in person_dirs:
        calibration_dir = (
            person_dir
            / "Calibration"
        )

        for filename in (
            required_calibration_files
        ):
            calibration_path = (
                calibration_dir
                / filename
            )

            if not calibration_path.is_file():
                raise FileNotFoundError(
                    "Missing calibration file for "
                    f"{person_dir.name}: {filename}"
                )

        camera_data = loadmat(
            calibration_dir
            / "Camera.mat"
        )

        for key in (
            "cameraMatrix",
            "distCoeffs",
        ):
            if key not in camera_data:
                raise RuntimeError(
                    "Camera.mat is missing required field "
                    f"'{key}' for {person_dir.name}."
                )

        lines = load_annotation_lines(
            person_dir
        )

        if not lines:
            raise RuntimeError(
                f"No annotations found for {person_dir.name}."
            )

        seen_paths = set()

        for line_number, line in enumerate(
            lines,
            start=1,
        ):
            try:
                image_relative_path = (
                    parse_annotation_image_path(
                        line
                    )
                )

            except Exception as error:
                raise RuntimeError(
                    "Invalid MPIIFaceGaze annotation in "
                    f"{person_dir.name}.txt line {line_number}: "
                    f"{error}"
                ) from error

            if (
                image_relative_path
                in seen_paths
            ):
                raise RuntimeError(
                    "Duplicate image path in annotation file "
                    f"{person_dir.name}.txt: {image_relative_path}"
                )

            seen_paths.add(
                image_relative_path
            )

            image_path = (
                person_dir
                / image_relative_path
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    "Annotation references a missing image: "
                    f"{person_dir.name}/{image_relative_path}"
                )

        annotations_by_person[
            person_dir.name
        ] = lines

        total_annotations += len(
            lines
        )

    if (
        total_annotations
        != EXPECTED_TOTAL_ANNOTATIONS
    ):
        raise RuntimeError(
            "Unexpected MPIIFaceGaze annotation count. "
            f"Expected {EXPECTED_TOTAL_ANNOTATIONS}, "
            f"found {total_annotations}."
        )

    return {
        "participants":
            len(
                participant_names
            ),
        "annotations":
            total_annotations,
        "annotations_by_person":
            annotations_by_person,
    }


def create_camera_yaml(
    person_dir: Path,
    image_width: int,
    image_height: int,
    runtime_dir: Path,
) -> Path:
    """Create one participant runtime camera file outside the dataset."""
    calibration_dir = (
        person_dir
        / "Calibration"
    )

    camera_data = loadmat(
        calibration_dir
        / "Camera.mat"
    )

    camera_matrix = np.asarray(
        camera_data[
            "cameraMatrix"
        ],
        dtype=np.float64,
    )

    distortion = np.asarray(
        camera_data[
            "distCoeffs"
        ],
        dtype=np.float64,
    ).reshape(
        -1
    )

    if (
        camera_matrix.shape
        != (
            3,
            3,
        )
    ):
        raise RuntimeError(
            "Unexpected camera matrix shape for "
            f"{person_dir.name}: {camera_matrix.shape}"
        )

    if (
        not np.all(
            np.isfinite(
                camera_matrix
            )
        )
        or distortion.size == 0
        or not np.all(
            np.isfinite(
                distortion
            )
        )
    ):
        raise RuntimeError(
            f"Invalid camera calibration for {person_dir.name}."
        )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        runtime_dir
        / f"{person_dir.name}_camera.yaml"
    )

    data = {
        "image_width":
            int(
                image_width
            ),
        "image_height":
            int(
                image_height
            ),
        "camera_matrix": {
            "rows":
                3,
            "cols":
                3,
            "data":
                (
                    camera_matrix
                    .reshape(
                        -1
                    )
                    .tolist()
                ),
        },
        "distortion_coefficients": {
            "rows":
                1,
            "cols":
                int(
                    len(
                        distortion
                    )
                ),
            "data":
                distortion.tolist(),
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    return output_path


def first_readable_image(
    person_dir: Path,
    lines: list[str],
) -> tuple[
    Path,
    np.ndarray,
]:
    """Return the first readable participant image."""
    for line in lines:
        image_relative_path = (
            parse_annotation_image_path(
                line
            )
        )

        image_path = (
            person_dir
            / image_relative_path
        )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is not None:
            return (
                image_path,
                image,
            )

    raise RuntimeError(
        "No readable reference image for "
        f"{person_dir.name}."
    )


def make_config() -> FaceAnalysisConfig:
    """Build the isolated current GazeEstimator configuration."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=True,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
        gaze_estimation_mode=MODEL_MODE,
        gaze_estimation_min_iou=MIN_IOU,
    )

    config.validate()

    return config


def validate_pipeline_configuration(
    pipeline: FaceAnalysis,
) -> None:
    """Verify only GazeEstimator and the core face detector are active."""
    if pipeline.detector is None:
        raise RuntimeError(
            "PhysioTrack face detection is required as controlled "
            "upstream pipeline input for GazeEstimator association."
        )

    if pipeline.tracker is not None:
        raise RuntimeError(
            "Tracking must be disabled."
        )

    if pipeline.orientation is not None:
        raise RuntimeError(
            "Head pose must be disabled."
        )

    if pipeline.landmarks is not None:
        raise RuntimeError(
            "FaceLandmarks must be disabled."
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
            "Legacy GazeDescriptor must be disabled."
        )

    if pipeline.gaze_estimation is None:
        raise RuntimeError(
            "GazeEstimator must be enabled."
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

    if (
        pipeline.config.gaze_estimation_mode
        != MODEL_MODE
    ):
        raise RuntimeError(
            "Unexpected GazeEstimator model mode."
        )

    if not math.isclose(
        float(
            pipeline.config.gaze_estimation_min_iou
        ),
        MIN_IOU,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Unexpected GazeEstimator minimum IoU."
        )


def make_pipeline(
    camera_yaml_path: Path,
) -> FaceAnalysis:
    """Build the real current PhysioTrack FaceAnalysis pipeline."""
    pipeline = FaceAnalysis(
        config=make_config(),
        gaze_estimation_camera_path=(
            camera_yaml_path
        ),
        device=DEVICE,
        verbose=False,
    )

    validate_pipeline_configuration(
        pipeline
    )

    return pipeline


def checkpoint_provenance(
    pipeline: FaceAnalysis,
) -> dict[
    str,
    str,
]:
    """Verify the accepted pretrained ETH-XGaze checkpoint."""
    estimator = (
        pipeline.gaze_estimation
    )

    checkpoint_path = getattr(
        estimator,
        "checkpoint_path",
        None,
    )

    if (
        checkpoint_path is None
        or not Path(
            checkpoint_path
        ).is_file()
    ):
        raise RuntimeError(
            "GazeEstimator did not resolve a valid pretrained checkpoint."
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_hash = sha256_file(
        checkpoint_path
    )

    if (
        checkpoint_path.name
        != EXPECTED_CHECKPOINT_FILENAME
    ):
        raise RuntimeError(
            "Unexpected GazeEstimator checkpoint filename: "
            f"{checkpoint_path.name}"
        )

    if (
        checkpoint_hash
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Unexpected GazeEstimator checkpoint SHA256."
        )

    return {
        "checkpoint_file":
            checkpoint_path.name,
        "checkpoint_sha256":
            checkpoint_hash,
    }


def validate_gaze_feature(
    feature: Any,
) -> dict[
    str,
    Any,
]:
    """Validate one real FaceAnalysis GazeEstimator feature output."""
    if not isinstance(
        feature,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis gaze_estimation feature is not a dictionary."
        )

    available = bool(
        feature.get(
            "available",
            False,
        )
    )

    if not available:
        return {
            "available":
                False,
            "gaze_x":
                None,
            "gaze_y":
                None,
            "gaze_z":
                None,
            "gaze_vector_norm":
                None,
            "pitch_deg":
                None,
            "yaw_deg":
                None,
            "association_iou":
                feature.get(
                    "association_iou"
                ),
        }

    vector = np.asarray(
        feature.get(
            "gaze_vector"
        ),
        dtype=np.float64,
    ).reshape(
        -1
    )

    if (
        vector.size != 3
        or not np.all(
            np.isfinite(
                vector
            )
        )
    ):
        raise RuntimeError(
            "Available GazeEstimator output contains an invalid "
            "3D gaze vector."
        )

    vector_norm = float(
        np.linalg.norm(
            vector
        )
    )

    if not math.isclose(
        vector_norm,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "Available GazeEstimator gaze vector is not unit-normalized."
        )

    pitch = feature.get(
        "pitch"
    )

    yaw = feature.get(
        "yaw"
    )

    association_iou = feature.get(
        "association_iou"
    )

    if not finite_numeric(
        pitch
    ):
        raise RuntimeError(
            "Available GazeEstimator output contains invalid pitch."
        )

    if not finite_numeric(
        yaw
    ):
        raise RuntimeError(
            "Available GazeEstimator output contains invalid yaw."
        )

    if not finite_numeric(
        association_iou
    ):
        raise RuntimeError(
            "Available GazeEstimator output contains invalid association IoU."
        )

    association_iou = float(
        association_iou
    )

    if (
        association_iou < MIN_IOU - 1e-12
        or association_iou > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "Available GazeEstimator output violates the configured "
            "minimum-IoU association requirement."
        )

    return {
        "available":
            True,
        "gaze_x":
            float(
                vector[
                    0
                ]
            ),
        "gaze_y":
            float(
                vector[
                    1
                ]
            ),
        "gaze_z":
            float(
                vector[
                    2
                ]
            ),
        "gaze_vector_norm":
            vector_norm,
        "pitch_deg":
            float(
                pitch
            ),
        "yaw_deg":
            float(
                yaw
            ),
        "association_iou":
            association_iou,
    }


def base_row(
    participant: str,
    image_relative_path: str,
    image_width: int,
    image_height: int,
    face_count: int,
    face_index: int,
) -> dict[
    str,
    Any,
]:
    """Create one stable component-execution row."""
    return {
        "participant":
            participant,
        "image_relative_path":
            image_relative_path,
        "image_width":
            int(
                image_width
            ),
        "image_height":
            int(
                image_height
            ),
        "face_count":
            int(
                face_count
            ),
        "face_index":
            int(
                face_index
            ),
        "face_id":
            None,
        "detector_confidence":
            None,
        "bbox_x1":
            None,
        "bbox_y1":
            None,
        "bbox_x2":
            None,
        "bbox_y2":
            None,
        "gaze_estimation_available":
            False,
        "gaze_x":
            None,
        "gaze_y":
            None,
        "gaze_z":
            None,
        "gaze_vector_norm":
            None,
        "pitch_deg":
            None,
        "yaw_deg":
            None,
        "association_iou":
            None,
        "status":
            None,
        "failure_reason":
            None,
    }


def populate_face_row(
    row: dict[
        str,
        Any,
    ],
    instance: Any,
) -> dict[
    str,
    Any,
]:
    """Read the exact current FaceAnalysis face and GazeEstimator schema."""
    box = np.asarray(
        instance.box,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if (
        box.size != 4
        or not np.all(
            np.isfinite(
                box
            )
        )
    ):
        raise RuntimeError(
            "PhysioTrack face detector produced an invalid face box."
        )

    if (
        box[
            2
        ]
        <= box[
            0
        ]
        or box[
            3
        ]
        <= box[
            1
        ]
    ):
        raise RuntimeError(
            "PhysioTrack face detector produced a non-positive face box."
        )

    row[
        "face_id"
    ] = (
        None
        if instance.id is None
        else str(
            instance.id
        )
    )

    if finite_numeric(
        instance.confidence
    ):
        row[
            "detector_confidence"
        ] = float(
            instance.confidence
        )

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

    features = (
        instance.face_features
        if isinstance(
            instance.face_features,
            dict,
        )
        else {}
    )

    if (
        "gaze_estimation"
        not in features
    ):
        raise RuntimeError(
            "Enabled GazeEstimator output is missing from face_features."
        )

    legacy_gaze = features.get(
        "gaze"
    )

    if (
        isinstance(
            legacy_gaze,
            dict,
        )
        and legacy_gaze.get(
            "available",
            False,
        )
    ):
        raise RuntimeError(
            "Legacy GazeDescriptor produced an available output while "
            "gaze=False in the isolated component configuration."
        )

    gaze = validate_gaze_feature(
        features[
            "gaze_estimation"
        ]
    )

    row[
        "gaze_estimation_available"
    ] = gaze[
        "available"
    ]

    for key in (
        "gaze_x",
        "gaze_y",
        "gaze_z",
        "gaze_vector_norm",
        "pitch_deg",
        "yaw_deg",
        "association_iou",
    ):
        row[
            key
        ] = gaze[
            key
        ]

    if gaze[
        "available"
    ]:
        row[
            "status"
        ] = "OK"

    else:
        row[
            "status"
        ] = "NO_GAZE_OUTPUT"

        row[
            "failure_reason"
        ] = (
            "FaceAnalysis GazeEstimator returned available=False."
        )

    return row


def run_smoke_test(
    preflight: dict[
        str,
        Any,
    ],
    smoke_count: int,
) -> None:
    """Require several genuine FaceAnalysis GazeEstimator outputs."""
    if smoke_count < 1:
        raise ValueError(
            "smoke_count must be at least 1."
        )

    successful_samples = 0

    smoke_runtime_dir = Path(
        tempfile.mkdtemp(
            prefix=".gaze_estimation_smoke_",
            dir=RESULTS_DIR,
        )
    )

    try:
        for participant in EXPECTED_PARTICIPANTS:
            person_dir = (
                DATASET_ROOT
                / participant
            )

            lines = preflight[
                "annotations_by_person"
            ][
                participant
            ]

            (
                _,
                reference_image,
            ) = first_readable_image(
                person_dir,
                lines,
            )

            image_height, image_width = (
                reference_image.shape[
                    :2
                ]
            )

            camera_yaml_path = (
                create_camera_yaml(
                    person_dir,
                    image_width,
                    image_height,
                    smoke_runtime_dir,
                )
            )

            pipeline = make_pipeline(
                camera_yaml_path
            )

            try:
                checkpoint_provenance(
                    pipeline
                )

                for line in lines:
                    image_relative_path = (
                        parse_annotation_image_path(
                            line
                        )
                    )

                    image_path = (
                        person_dir
                        / image_relative_path
                    )

                    image = cv2.imread(
                        str(
                            image_path
                        )
                    )

                    if image is None:
                        raise RuntimeError(
                            f"Smoke-test image could not be read: {image_path}"
                        )

                    result = pipeline.predict(
                        image
                    )

                    for face_index, instance in enumerate(
                        result
                    ):
                        row = base_row(
                            participant,
                            image_relative_path,
                            image.shape[
                                1
                            ],
                            image.shape[
                                0
                            ],
                            len(
                                result
                            ),
                            face_index,
                        )

                        populate_face_row(
                            row,
                            instance,
                        )

                        if not row[
                            "gaze_estimation_available"
                        ]:
                            continue

                        successful_samples += 1

                        print(
                            "Smoke sample "
                            f"{successful_samples}: "
                            f"participant={participant}, "
                            f"image={image_relative_path}, "
                            f"face_index={face_index}, "
                            f"gaze=("
                            f"{row['gaze_x']}, "
                            f"{row['gaze_y']}, "
                            f"{row['gaze_z']}), "
                            f"pitch={row['pitch_deg']}, "
                            f"yaw={row['yaw_deg']}, "
                            f"association_iou={row['association_iou']}"
                        )

                        if (
                            successful_samples
                            >= smoke_count
                        ):
                            print(
                                "Smoke test confirmed real GazeEstimator "
                                "outputs through FaceAnalysis."
                            )

                            print(
                                "Smoke test: PASS"
                            )

                            return

            finally:
                pipeline.close()

        raise RuntimeError(
            "Smoke test did not observe enough successful "
            "FaceAnalysis GazeEstimator samples."
        )

    finally:
        if smoke_runtime_dir.exists():
            shutil.rmtree(
                smoke_runtime_dir,
                ignore_errors=True,
            )


def create_staging_directory() -> tuple[
    Path,
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
            prefix=".gaze_estimation_component_",
            dir=RESULTS_DIR,
        )
    )

    staged_component_dir = (
        staging_dir
        / "component_execution"
    )

    runtime_dir = (
        staging_dir
        / "runtime"
    )

    staged_component_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        staging_dir,
        staged_component_dir,
        runtime_dir,
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
    """Write isolated per-image/per-face numerical outputs."""
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


def participant_groups(
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
    """Group ordered rows by complete participant without changing row order."""
    groups: list[
        tuple[
            str,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ]
    ] = []

    current_participant: str | None = None
    current_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for row in rows:
        participant = str(
            row[
                "participant"
            ]
        )

        if (
            current_participant is not None
            and participant
            != current_participant
        ):
            groups.append(
                (
                    current_participant,
                    current_rows,
                )
            )

            current_rows = []

        current_participant = (
            participant
        )

        current_rows.append(
            row
        )

    if current_participant is not None:
        groups.append(
            (
                current_participant,
                current_rows,
            )
        )

    observed = [
        participant
        for participant, _
        in groups
    ]

    if observed != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            "Result rows are not grouped in the expected participant order."
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
    """Write one CSV or split oversized output only between participants."""
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
                "first_participant":
                    str(
                        rows[
                            0
                        ][
                            "participant"
                        ]
                    ),
                "last_participant":
                    str(
                        rows[
                            -1
                        ][
                            "participant"
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

    groups = participant_groups(
        rows
    )

    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ] = []

    current_groups: list[
        tuple[
            str,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ]
    ] = []

    part_number = 1

    def write_current_part() -> None:
        nonlocal part_number
        nonlocal current_groups

        if not current_groups:
            return

        part_rows = [
            row
            for _, participant_rows
            in current_groups
            for row in participant_rows
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
                "A complete-participant result part exceeds the configured "
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
                "first_participant":
                    current_groups[
                        0
                    ][
                        0
                    ],
                "last_participant":
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

    for participant, participant_rows in groups:
        trial_groups = (
            current_groups
            + [
                (
                    participant,
                    participant_rows,
                )
            ]
        )

        trial_rows = [
            row
            for _, grouped_rows
            in trial_groups
            for row in grouped_rows
        ]

        trial_path = (
            output_dir
            / ".gaze_estimation_component_size_probe.csv"
        )

        write_csv(
            trial_path,
            trial_rows,
        )

        trial_size = (
            trial_path.stat().st_size
        )

        trial_path.unlink()

        if (
            trial_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
            and current_groups
        ):
            write_current_part()

        current_groups.append(
            (
                participant,
                participant_rows,
            )
        )

    write_current_part()

    if not output_manifest:
        raise RuntimeError(
            "No Git-safe component-result parts were generated."
        )

    return output_manifest


def staged_result_paths(
    staged_component_dir: Path,
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[Path]:
    """Resolve staged result files from the output manifest."""
    return [
        staged_component_dir
        / item[
            "filename"
        ]
        for item in output_manifest
    ]


def validate_result_files(
    result_paths: list[Path],
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
    """Validate complete result coverage across one or more CSV files."""
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

    all_rows: list[
        dict[
            str,
            str,
        ]
    ] = []

    seen_participants: set[str] = set()

    for path, manifest_item in zip(
        result_paths,
        output_manifest,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing staged component-result file: {path}"
            )

        if (
            len(
                result_paths
            ) > 1
            and path.stat().st_size
            > GIT_SAFE_MAX_FILE_SIZE_BYTES
        ):
            raise RuntimeError(
                "A split component-result CSV exceeds the Git-safe limit."
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
                    "Unexpected GazeEstimator component-result CSV schema."
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

        part_participants = {
            row[
                "participant"
            ]
            for row in part_rows
        }

        if (
            seen_participants
            & part_participants
        ):
            raise RuntimeError(
                "A participant was split across multiple result files."
            )

        seen_participants.update(
            part_participants
        )

        all_rows.extend(
            part_rows
        )

    image_keys = {
        (
            row[
                "participant"
            ],
            row[
                "image_relative_path"
            ],
        )
        for row in all_rows
    }

    if (
        len(
            image_keys
        )
        != preflight[
            "annotations"
        ]
    ):
        raise RuntimeError(
            "Staged result files do not cover the complete accepted "
            "annotation-image population."
        )

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
    """Validate complete isolated outputs before commit."""
    image_keys = {
        (
            str(
                row[
                    "participant"
                ]
            ),
            str(
                row[
                    "image_relative_path"
                ]
            ),
        )
        for row in rows
    }

    if (
        len(
            image_keys
        )
        != preflight[
            "annotations"
        ]
    ):
        raise RuntimeError(
            "Component results do not cover the complete accepted "
            "MPIIFaceGaze image population."
        )

    row_keys = [
        (
            str(
                row[
                    "participant"
                ]
            ),
            str(
                row[
                    "image_relative_path"
                ]
            ),
            int(
                row[
                    "face_index"
                ]
            ),
        )
        for row in rows
    ]

    if len(
        row_keys
    ) != len(
        set(
            row_keys
        )
    ):
        raise RuntimeError(
            "Duplicate participant/image/face-index rows were found."
        )

    valid_statuses = {
        "OK",
        "NO_FACE",
        "NO_GAZE_OUTPUT",
        "IMAGE_READ_FAILED",
        "EXECUTION_FAILED",
    }

    for row in rows:
        if row[
            "status"
        ] not in valid_statuses:
            raise RuntimeError(
                f"Unexpected component status: {row['status']}"
            )

        available = bool(
            row[
                "gaze_estimation_available"
            ]
        )

        if available:
            required_values = [
                row[
                    "gaze_x"
                ],
                row[
                    "gaze_y"
                ],
                row[
                    "gaze_z"
                ],
                row[
                    "gaze_vector_norm"
                ],
                row[
                    "pitch_deg"
                ],
                row[
                    "yaw_deg"
                ],
                row[
                    "association_iou"
                ],
            ]

            if not all(
                finite_numeric(
                    value
                )
                for value in required_values
            ):
                raise RuntimeError(
                    "Available component row contains non-finite "
                    "GazeEstimator values."
                )

            if not math.isclose(
                float(
                    row[
                        "gaze_vector_norm"
                    ]
                ),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Stored gaze vector norm is not one."
                )

            association_iou = float(
                row[
                    "association_iou"
                ]
            )

            if (
                association_iou < MIN_IOU - 1e-12
                or association_iou > 1.0 + 1e-12
            ):
                raise RuntimeError(
                    "Stored association IoU violates the configured range."
                )

            if row[
                "status"
            ] != "OK":
                raise RuntimeError(
                    "Available GazeEstimator row is not marked OK."
                )

        elif (
            row[
                "status"
            ]
            == "OK"
        ):
            raise RuntimeError(
                "Unavailable GazeEstimator row is marked OK."
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
    provenance: dict[
        str,
        str,
    ],
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
    """Create isolated component summary without accuracy metrics."""
    status_counts: dict[
        str,
        int,
    ] = {}

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

    image_keys = {
        (
            str(
                row[
                    "participant"
                ]
            ),
            str(
                row[
                    "image_relative_path"
                ]
            ),
        )
        for row in rows
    }

    detected_face_rows = sum(
        1
        for row in rows
        if int(
            row[
                "face_index"
            ]
        )
        >= 0
    )

    gaze_available_rows = sum(
        1
        for row in rows
        if bool(
            row[
                "gaze_estimation_available"
            ]
        )
    )

    images_without_faces = sum(
        1
        for row in rows
        if row[
            "status"
        ]
        == "NO_FACE"
    )

    images_with_faces = (
        len(
            image_keys
        )
        - images_without_faces
    )

    execution_failures = (
        status_counts.get(
            "IMAGE_READ_FAILED",
            0,
        )
        + status_counts.get(
            "EXECUTION_FAILED",
            0,
        )
    )

    overall_status = (
        "PASS"
        if (
            execution_failures == 0
            and len(
                image_keys
            )
            == preflight[
                "annotations"
            ]
            and gaze_available_rows > 0
        )
        else "FAIL"
    )

    source_path = Path(
        inspect.getfile(
            pipeline_gaze_estimator_class()
        )
    ).resolve()

    source_display = (
        source_path.relative_to(
            REPO_ROOT
        ).as_posix()
        if REPO_ROOT in source_path.parents
        else source_path.name
    )

    return {
        "component":
            "PhysioTrack GazeEstimator",
        "execution_type":
            "isolated component execution; not an accuracy benchmark",
        "dataset":
            "MPIIFaceGaze full annotated image population",
        "pipeline":
            "PhysioTrack FaceAnalysis",
        "physiotrack_source":
            source_display,
        "device":
            DEVICE,
        "model_mode":
            MODEL_MODE,
        "ptgaze_version":
            package_version(
                "ptgaze"
            ),
        "checkpoint_file":
            provenance[
                "checkpoint_file"
            ],
        "checkpoint_sha256":
            provenance[
                "checkpoint_sha256"
            ],
        "required_upstream_input":
            (
                "Current PhysioTrack face detector boxes passed by "
                "FaceAnalysis to GazeEstimator.predict_faces()."
            ),
        "association_note":
            (
                "The current GazeEstimator face-conditioned path uses "
                "one-to-one greedy IoU association and the configured "
                "minimum-IoU requirement; target-conditioned crop fallback "
                "is handled inside the current source implementation."
            ),
        "legacy_gaze_note":
            (
                "Legacy geometric GazeDescriptor is disabled and remains "
                "separate from learned gaze_estimation."
            ),
        "public_api_note":
            (
                "The current GazeEstimator implementation retains the "
                "PredictorMixin-compatible NumPy/path/batch public predict "
                "contract; this isolated run exercises the FaceAnalysis "
                "predict_faces integration path."
            ),
        "enabled_components": [
            "gaze_estimation",
        ],
        "disabled_optional_components": [
            "tracking",
            "head_pose",
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
        ],
        "gaze_estimation_min_iou":
            MIN_IOU,
        "participants":
            preflight[
                "participants"
            ],
        "annotation_images":
            preflight[
                "annotations"
            ],
        "covered_annotation_images":
            len(
                image_keys
            ),
        "images_with_faces":
            images_with_faces,
        "images_without_faces":
            images_without_faces,
        "detected_face_rows":
            detected_face_rows,
        "gaze_estimation_available_rows":
            gaze_available_rows,
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
            "split_only_at_complete_participant_boundaries":
                True,
            "result_file_count":
                len(
                    output_manifest
                ),
            "files":
                output_manifest,
        },
        "accuracy_metrics_computed":
            False,
        "overall_status":
            overall_status,
    }


def pipeline_gaze_estimator_class():
    """Return the concrete current GazeEstimator class used by FaceAnalysis."""
    from physiotrack.face.gaze_estimation import GazeEstimator

    return GazeEstimator


def validate_staged_outputs(
    result_paths: list[Path],
    summary_path: Path,
    preflight: dict[
        str,
        Any,
    ],
    output_manifest: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """Re-read staged outputs and verify them before final replacement."""
    if not summary_path.is_file():
        raise RuntimeError(
            f"Missing staged summary: {summary_path}"
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

    if int(
        summary.get(
            "participants",
            -1,
        )
    ) != len(
        EXPECTED_PARTICIPANTS
    ):
        raise RuntimeError(
            "Staged isolated summary participant count mismatch."
        )

    if int(
        summary.get(
            "annotation_images",
            -1,
        )
    ) != EXPECTED_TOTAL_ANNOTATIONS:
        raise RuntimeError(
            "Staged isolated summary annotation count mismatch."
        )

    if int(
        summary.get(
            "covered_annotation_images",
            -1,
        )
    ) != EXPECTED_TOTAL_ANNOTATIONS:
        raise RuntimeError(
            "Staged isolated summary coverage count mismatch."
        )

    if bool(
        summary.get(
            "accuracy_metrics_computed",
            True,
        )
    ):
        raise RuntimeError(
            "Isolated component summary incorrectly claims accuracy metrics."
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
    preflight: dict[
        str,
        Any,
    ],
) -> None:
    """Run isolated GazeEstimator execution on the full accepted population."""
    (
        staging_dir,
        staged_component_dir,
        runtime_dir,
    ) = create_staging_directory()

    print(
        f"Staging directory: {staging_dir}"
    )

    staged_summary = (
        staged_component_dir
        / SUMMARY_FILENAME
    )

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    dataset_before = dataset_inventory(
        DATASET_ROOT
    )

    provenance: dict[
        str,
        str,
    ] | None = None

    start_time = time.perf_counter()

    try:
        for participant_number, participant in enumerate(
            EXPECTED_PARTICIPANTS,
            start=1,
        ):
            person_dir = (
                DATASET_ROOT
                / participant
            )

            lines = preflight[
                "annotations_by_person"
            ][
                participant
            ]

            (
                _,
                reference_image,
            ) = first_readable_image(
                person_dir,
                lines,
            )

            image_height, image_width = (
                reference_image.shape[
                    :2
                ]
            )

            camera_yaml_path = (
                create_camera_yaml(
                    person_dir,
                    image_width,
                    image_height,
                    runtime_dir,
                )
            )

            pipeline = make_pipeline(
                camera_yaml_path
            )

            try:
                participant_provenance = (
                    checkpoint_provenance(
                        pipeline
                    )
                )

                if provenance is None:
                    provenance = (
                        participant_provenance
                    )

                elif (
                    provenance
                    != participant_provenance
                ):
                    raise RuntimeError(
                        "GazeEstimator checkpoint provenance changed "
                        "between participants."
                    )

                for image_number, line in enumerate(
                    lines,
                    start=1,
                ):
                    image_relative_path = (
                        parse_annotation_image_path(
                            line
                        )
                    )

                    image_path = (
                        person_dir
                        / image_relative_path
                    )

                    image = cv2.imread(
                        str(
                            image_path
                        )
                    )

                    if image is None:
                        row = base_row(
                            participant,
                            image_relative_path,
                            0,
                            0,
                            0,
                            -1,
                        )

                        row[
                            "status"
                        ] = "IMAGE_READ_FAILED"

                        row[
                            "failure_reason"
                        ] = (
                            "cv2.imread returned None."
                        )

                        rows.append(
                            row
                        )

                        continue

                    try:
                        result = pipeline.predict(
                            image
                        )

                        face_count = len(
                            result
                        )

                        if face_count == 0:
                            row = base_row(
                                participant,
                                image_relative_path,
                                image.shape[
                                    1
                                ],
                                image.shape[
                                    0
                                ],
                                0,
                                -1,
                            )

                            row[
                                "status"
                            ] = "NO_FACE"

                            row[
                                "failure_reason"
                            ] = (
                                "PhysioTrack face detector returned no faces."
                            )

                            rows.append(
                                row
                            )

                        else:
                            for face_index, instance in enumerate(
                                result
                            ):
                                row = base_row(
                                    participant,
                                    image_relative_path,
                                    image.shape[
                                        1
                                    ],
                                    image.shape[
                                        0
                                    ],
                                    face_count,
                                    face_index,
                                )

                                populate_face_row(
                                    row,
                                    instance,
                                )

                                rows.append(
                                    row
                                )

                    except Exception as error:
                        row = base_row(
                            participant,
                            image_relative_path,
                            image.shape[
                                1
                            ],
                            image.shape[
                                0
                            ],
                            0,
                            -1,
                        )

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

                    if (
                        image_number % 500
                        == 0
                        or image_number
                        == len(
                            lines
                        )
                    ):
                        print(
                            f"{participant}: "
                            f"{image_number}/{len(lines)}"
                        )

            finally:
                pipeline.close()

            print(
                "Processed participant "
                f"{participant_number}/{len(EXPECTED_PARTICIPANTS)}: "
                f"{participant}"
            )

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        dataset_after = dataset_inventory(
            DATASET_ROOT
        )

        if (
            dataset_after
            != dataset_before
        ):
            raise RuntimeError(
                "Dataset read-only invariant failed: MPIIFaceGaze "
                "contents or metadata changed during isolated execution."
            )

        if provenance is None:
            raise RuntimeError(
                "No GazeEstimator checkpoint provenance was captured."
            )

        validate_rows(
            rows,
            preflight,
        )

        preliminary_summary = build_summary(
            rows,
            preflight,
            runtime_seconds,
            provenance,
            [],
        )

        if preliminary_summary[
            "execution_failures"
        ] != 0:
            raise RuntimeError(
                "Isolated component execution contained execution/read "
                "failures. Prior accepted outputs were preserved."
            )

        if preliminary_summary[
            "overall_status"
        ] != "PASS":
            raise RuntimeError(
                "Isolated component execution did not satisfy PASS "
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
            provenance,
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
            "Validating staged isolated GazeEstimator outputs..."
        )

        validate_staged_outputs(
            result_paths,
            staged_summary,
            preflight,
            output_manifest,
        )

        commit_outputs(
            result_paths,
            staged_summary,
            staging_dir,
        )

        print(
            "Committed final isolated GazeEstimator outputs."
        )

        print()
        print(
            "=== Gaze Estimation Isolated Component Results ==="
        )

        print(
            f"Participants: {summary['participants']}"
        )

        print(
            f"Annotation images: {summary['annotation_images']}"
        )

        print(
            "Covered annotation images: "
            f"{summary['covered_annotation_images']}"
        )

        print(
            f"Images with faces: {summary['images_with_faces']}"
        )

        print(
            f"Images without faces: {summary['images_without_faces']}"
        )

        print(
            f"Detected face rows: {summary['detected_face_rows']}"
        )

        print(
            "GazeEstimator available rows: "
            f"{summary['gaze_estimation_available_rows']}"
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
                f"participants={manifest_item['first_participant']}-"
                f"{manifest_item['last_participant']} | "
                f"rows={manifest_item['row_count']} | "
                f"size={manifest_item['size_mib']:.2f} MiB"
            )

        print(
            COMPONENT_RESULTS_DIR
            / SUMMARY_FILENAME
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


def parse_args() -> argparse.Namespace:
    """Parse isolated component execution mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated PhysioTrack GazeEstimator component execution "
            "through FaceAnalysis on MPIIFaceGaze."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate project paths, environment, dataset structure, "
            "and accepted population counts without inference."
        ),
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small real FaceAnalysis GazeEstimator inference test "
            "without writing final outputs."
        ),
    )

    parser.add_argument(
        "--smoke-count",
        type=int,
        default=3,
        help=(
            "Number of successful GazeEstimator face outputs required "
            "during --smoke-test."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the requested isolated GazeEstimator verification mode."""
    args = parse_args()

    ptgaze_version = package_version(
        "ptgaze"
    )

    if (
        ptgaze_version
        != EXPECTED_PTGAZE_VERSION
    ):
        raise RuntimeError(
            "Reproducibility check failed: "
            f"ptgaze=={EXPECTED_PTGAZE_VERSION} is required, "
            f"but {ptgaze_version} is installed."
        )

    preflight = dataset_preflight()

    print(
        "MPIIFaceGaze isolated GazeEstimator preflight: PASS"
    )

    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    print(
        f"Participants: {preflight['participants']}"
    )

    print(
        f"Annotation images: {preflight['annotations']}"
    )

    print(
        f"Model mode: {MODEL_MODE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"ptgaze: {ptgaze_version}"
    )

    print(
        f"GazeEstimator minimum IoU: {MIN_IOU}"
    )

    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )

    print(
        "Target component: learned GazeEstimator"
    )

    print(
        "Required upstream input: current PhysioTrack face detector boxes"
    )

    print(
        "Legacy GazeDescriptor: disabled"
    )

    print(
        "Unrelated optional components: disabled"
    )

    print(
        "Accuracy metrics: not computed"
    )

    if args.preflight_only:
        print(
            "Preflight-only mode: no face or gaze inference was run."
        )

        return

    if args.smoke_test:
        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

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
