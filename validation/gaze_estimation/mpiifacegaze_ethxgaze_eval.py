from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import math
import os
import shutil
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.io import loadmat


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.is_dir():
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from physiotrack.face.gaze_estimation import GazeEstimator


DATASET_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "MPIIFaceGaze"
    / "Data"
)

OUTPUT_DIR = SCRIPT_DIR / "results"

PER_PERSON_FILENAME = "mpiifacegaze_ethxgaze_per_person.csv"
PER_SAMPLE_FILENAME = "mpiifacegaze_ethxgaze_per_sample.csv"
SUMMARY_FILENAME = "mpiifacegaze_ethxgaze_summary.txt"

MODEL_MODE = "eth-xgaze"
DEVICE = "cpu"

EXPECTED_PARTICIPANTS = [
    f"p{index:02d}"
    for index in range(15)
]

EXPECTED_TOTAL_ANNOTATIONS = 37667
EXPECTED_PTGAZE_VERSION = "0.3.0"


def package_version(
    package_name: str,
) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def dataset_inventory(
    root: Path,
) -> dict[str, tuple[int, int]]:
    inventory = {}

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        stat = path.stat()

        inventory[
            path.relative_to(root).as_posix()
        ] = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    return inventory


def angular_error_degrees(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float64,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(-1)

    if (
        ground_truth.size != 3
        or prediction.size != 3
    ):
        raise ValueError(
            "Angular error requires two 3D vectors."
        )

    if (
        not np.all(
            np.isfinite(
                ground_truth
            )
        )
        or not np.all(
            np.isfinite(
                prediction
            )
        )
    ):
        raise ValueError(
            "Angular error vectors must be finite."
        )

    gt_norm = float(
        np.linalg.norm(
            ground_truth
        )
    )

    pred_norm = float(
        np.linalg.norm(
            prediction
        )
    )

    if (
        gt_norm <= 0
        or pred_norm <= 0
    ):
        raise ValueError(
            "Cannot compute angular error for a zero-length vector."
        )

    ground_truth = (
        ground_truth
        / gt_norm
    )

    prediction = (
        prediction
        / pred_norm
    )

    dot_product = float(
        np.clip(
            np.dot(
                ground_truth,
                prediction,
            ),
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(
            dot_product
        )
    )


def parse_annotation_line(
    line: str,
) -> tuple[
    str,
    np.ndarray,
]:
    parts = line.strip().split()

    if len(parts) != 28:
        raise ValueError(
            "Unexpected annotation field count: "
            f"{len(parts)}"
        )

    image_relative_path = parts[0]

    values = np.asarray(
        [
            float(value)
            for value in parts[1:27]
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

    face_center = values[
        20:23
    ]

    gaze_target = values[
        23:26
    ]

    gaze_vector = (
        gaze_target
        - face_center
    )

    norm = float(
        np.linalg.norm(
            gaze_vector
        )
    )

    if norm <= 0:
        raise ValueError(
            "Invalid ground-truth gaze vector."
        )

    gaze_vector = (
        gaze_vector
        / norm
    )

    return (
        image_relative_path,
        gaze_vector,
    )


def load_annotation_lines(
    person_dir: Path,
) -> list[str]:
    annotation_path = (
        person_dir
        / f"{person_dir.name}.txt"
    )

    with open(
        annotation_path,
        "r",
        encoding="utf-8",
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def preflight_dataset() -> dict[str, list[str]]:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "MPIIFaceGaze dataset was not found at the expected "
            "project-relative location:\n"
            "datasets/MPIIFaceGaze/Data"
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
        path
        for path in DATASET_ROOT.glob("p*")
        if path.is_dir()
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

    annotations_by_person = {}
    total_annotations = 0

    required_calibration_files = (
        "Camera.mat",
        "monitorPose.mat",
        "screenSize.mat",
    )

    for person_dir in person_dirs:
        annotation_path = (
            person_dir
            / f"{person_dir.name}.txt"
        )

        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation file: {annotation_path.name}"
            )

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
                (
                    image_relative_path,
                    _,
                ) = parse_annotation_line(
                    line
                )
            except Exception as error:
                raise RuntimeError(
                    "Invalid MPIIFaceGaze annotation in "
                    f"{person_dir.name}.txt line {line_number}: "
                    f"{error}"
                ) from error

            if image_relative_path in seen_paths:
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

    return annotations_by_person


def create_camera_yaml(
    person_dir: Path,
    image_width: int,
    image_height: int,
    runtime_dir: Path,
) -> Path:
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
    ).reshape(-1)

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
        distortion.size == 0
        or not np.all(
            np.isfinite(
                distortion
            )
        )
    ):
        raise RuntimeError(
            "Invalid distortion coefficients for "
            f"{person_dir.name}."
        )

    if not np.all(
        np.isfinite(
            camera_matrix
        )
    ):
        raise RuntimeError(
            "Invalid camera matrix for "
            f"{person_dir.name}."
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
        "image_width": int(
            image_width
        ),
        "image_height": int(
            image_height
        ),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": (
                camera_matrix
                .reshape(-1)
                .tolist()
            ),
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(
                len(
                    distortion
                )
            ),
            "data": (
                distortion
                .tolist()
            ),
        },
    }

    with open(
        output_path,
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
) -> tuple[Path, np.ndarray]:
    for line in lines:
        (
            image_relative_path,
            _,
        ) = parse_annotation_line(
            line
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


def read_summary(
    summary_path: Path,
) -> dict[str, str]:
    """Read colon-delimited summary values."""
    values = {}

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if (
                not line
                or ":" not in line
            ):
                continue

            key, value = line.split(
                ":",
                1,
            )

            values[
                key.strip()
            ] = value.strip()

    return values


def create_staging_directory() -> tuple[Path, Path]:
    """Create staging before the full benchmark performs generative work."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".mpiifacegaze_ethxgaze_eval_",
            dir=OUTPUT_DIR,
        )
    )

    staged_output_dir = (
        staging_dir
        / "outputs"
    )

    staged_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        staging_dir,
        staged_output_dir,
    )


def validate_staged_outputs(
    per_person_csv_path: Path,
    per_sample_csv_path: Path,
    summary_path: Path,
) -> None:
    """Independently verify complete staged benchmark outputs."""
    required_paths = [
        per_person_csv_path,
        per_sample_csv_path,
        summary_path,
    ]

    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(
                f"Missing staged evaluator output: {path.name}"
            )

    expected_person_fields = [
        "participant",
        "annotations",
        "successful_predictions",
        "image_read_failures",
        "face_detection_failures",
        "prediction_failures",
        "invalid_annotation_rows",
        "mean_angular_error_deg",
        "median_angular_error_deg",
        "std_angular_error_deg",
        "runtime_seconds",
    ]

    expected_sample_fields = [
        "participant",
        "image_relative_path",
        "status",
        "failure_reason",
        "gt_x",
        "gt_y",
        "gt_z",
        "pred_x",
        "pred_y",
        "pred_z",
        "angular_error_deg",
    ]

    with open(
        per_person_csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != expected_person_fields:
            raise RuntimeError(
                "Unexpected staged per-person CSV schema."
            )

        person_rows = list(
            reader
        )

    with open(
        per_sample_csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != expected_sample_fields:
            raise RuntimeError(
                "Unexpected staged per-sample CSV schema."
            )

        sample_rows = list(
            reader
        )

    if len(
        person_rows
    ) != len(
        EXPECTED_PARTICIPANTS
    ):
        raise RuntimeError(
            "Unexpected staged per-person row count."
        )

    if [
        row[
            "participant"
        ]
        for row in person_rows
    ] != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            "Staged per-person participant order or set is invalid."
        )

    if len(
        sample_rows
    ) != EXPECTED_TOTAL_ANNOTATIONS:
        raise RuntimeError(
            "Staged per-sample row count does not match the accepted "
            "MPIIFaceGaze population."
        )

    keys = [
        (
            row[
                "participant"
            ],
            row[
                "image_relative_path"
            ],
        )
        for row in sample_rows
        if row[
            "image_relative_path"
        ]
    ]

    if len(
        keys
    ) != len(
        set(
            keys
        )
    ):
        raise RuntimeError(
            "Duplicate participant/image rows were found in staged results."
        )

    allowed_statuses = {
        "success",
        "image_read_failure",
        "face_detection_failure",
        "prediction_failure",
        "invalid_annotation",
    }

    status_counts = {
        status: 0
        for status in allowed_statuses
    }

    errors = []

    for row in sample_rows:
        status = row[
            "status"
        ]

        if status not in allowed_statuses:
            raise RuntimeError(
                f"Unexpected staged per-sample status: {status}"
            )

        status_counts[
            status
        ] += 1

        if status != "success":
            continue

        ground_truth = np.asarray(
            [
                float(
                    row[
                        "gt_x"
                    ]
                ),
                float(
                    row[
                        "gt_y"
                    ]
                ),
                float(
                    row[
                        "gt_z"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        prediction = np.asarray(
            [
                float(
                    row[
                        "pred_x"
                    ]
                ),
                float(
                    row[
                        "pred_y"
                    ]
                ),
                float(
                    row[
                        "pred_z"
                    ]
                ),
            ],
            dtype=np.float64,
        )

        stored_error = float(
            row[
                "angular_error_deg"
            ]
        )

        if not (
            np.all(
                np.isfinite(
                    ground_truth
                )
            )
            and np.all(
                np.isfinite(
                    prediction
                )
            )
            and math.isfinite(
                stored_error
            )
        ):
            raise RuntimeError(
                "Successful staged sample contains non-finite numerical values."
            )

        recomputed_error = angular_error_degrees(
            ground_truth,
            prediction,
        )

        if not math.isclose(
            recomputed_error,
            stored_error,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "Stored staged angular error is inconsistent with its gaze vectors."
            )

        errors.append(
            stored_error
        )

    person_count_columns = {
        "success": "successful_predictions",
        "image_read_failure": "image_read_failures",
        "face_detection_failure": "face_detection_failures",
        "prediction_failure": "prediction_failures",
        "invalid_annotation": "invalid_annotation_rows",
    }

    for person_row in person_rows:
        participant = person_row[
            "participant"
        ]

        participant_samples = [
            row
            for row in sample_rows
            if row[
                "participant"
            ] == participant
        ]

        if len(
            participant_samples
        ) != int(
            person_row[
                "annotations"
            ]
        ):
            raise RuntimeError(
                f"{participant}: staged annotation count mismatch."
            )

        for status, column in person_count_columns.items():
            sample_count = sum(
                1
                for row in participant_samples
                if row[
                    "status"
                ] == status
            )

            if sample_count != int(
                person_row[
                    column
                ]
            ):
                raise RuntimeError(
                    f"{participant}: staged {column} mismatch."
                )

        person_errors = np.asarray(
            [
                float(
                    row[
                        "angular_error_deg"
                    ]
                )
                for row in participant_samples
                if row[
                    "status"
                ] == "success"
            ],
            dtype=np.float64,
        )

        if len(
            person_errors
        ) == 0:
            raise RuntimeError(
                f"{participant}: no successful staged predictions."
            )

        expected_statistics = {
            "mean_angular_error_deg": float(
                person_errors.mean()
            ),
            "median_angular_error_deg": float(
                np.median(
                    person_errors
                )
            ),
            "std_angular_error_deg": float(
                person_errors.std()
            ),
        }

        for column, expected_value in expected_statistics.items():
            if not math.isclose(
                float(
                    person_row[
                        column
                    ]
                ),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"{participant}: staged {column} mismatch."
                )

    summary = read_summary(
        summary_path
    )

    if summary.get(
        "Dataset integrity after evaluation"
    ) != "PASS":
        raise RuntimeError(
            "Staged summary does not report dataset-integrity PASS."
        )

    if int(
        summary.get(
            "Participants",
            "-1",
        )
    ) != len(
        EXPECTED_PARTICIPANTS
    ):
        raise RuntimeError(
            "Staged summary participant count is invalid."
        )

    if summary.get(
        "Model mode"
    ) != MODEL_MODE:
        raise RuntimeError(
            "Staged summary model mode is invalid."
        )

    if summary.get(
        "Device"
    ) != DEVICE:
        raise RuntimeError(
            "Staged summary device is invalid."
        )

    if summary.get(
        "ptgaze version"
    ) != EXPECTED_PTGAZE_VERSION:
        raise RuntimeError(
            "Staged summary ptgaze version is invalid."
        )

    summary_counts = {
        "Total annotations": EXPECTED_TOTAL_ANNOTATIONS,
        "Successful predictions": status_counts[
            "success"
        ],
        "Image read failures": status_counts[
            "image_read_failure"
        ],
        "Face detection failures": status_counts[
            "face_detection_failure"
        ],
        "Prediction failures": status_counts[
            "prediction_failure"
        ],
        "Invalid annotation rows": status_counts[
            "invalid_annotation"
        ],
    }

    for key, expected_value in summary_counts.items():
        if int(
            summary.get(
                key,
                "-1",
            )
        ) != expected_value:
            raise RuntimeError(
                f"Staged summary count mismatch for {key}."
            )

    if int(
        summary.get(
            "Accounted annotations",
            "-1",
        )
    ) != EXPECTED_TOTAL_ANNOTATIONS:
        raise RuntimeError(
            "Staged summary accounted-annotation count is invalid."
        )

    error_array = np.asarray(
        errors,
        dtype=np.float64,
    )

    summary_statistics = {
        "Mean angular error": float(
            error_array.mean()
        ),
        "Median angular error": float(
            np.median(
                error_array
            )
        ),
        "Std angular error": float(
            error_array.std()
        ),
        "Minimum angular error": float(
            error_array.min()
        ),
        "Maximum angular error": float(
            error_array.max()
        ),
        "90th percentile angular error": float(
            np.percentile(
                error_array,
                90,
            )
        ),
        "95th percentile angular error": float(
            np.percentile(
                error_array,
                95,
            )
        ),
    }

    for key, expected_value in summary_statistics.items():
        if key not in summary:
            raise RuntimeError(
                f"Missing staged summary statistic: {key}"
            )

        stored_value = float(
            summary[
                key
            ]
            .replace(
                "deg",
                "",
            )
            .strip()
        )

        if not math.isclose(
            stored_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=5e-6,
        ):
            raise RuntimeError(
                f"Staged summary statistic mismatch for {key}."
            )


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Atomically install one validated evaluator-owned file."""
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


def commit_outputs(
    staged_paths: list[Path],
    staging_dir: Path,
) -> None:
    """Replace only evaluator-owned outputs with rollback protection."""
    final_paths = [
        OUTPUT_DIR
        / path.name
        for path in staged_paths
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for final_path in final_paths:
        if final_path.is_file():
            shutil.copy2(
                final_path,
                backup_dir
                / final_path.name,
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

    except Exception:
        for installed_path in installed_paths:
            if installed_path.exists():
                installed_path.unlink()

        for backup_path in backup_dir.iterdir():
            atomic_copy_file(
                backup_path,
                OUTPUT_DIR
                / backup_path.name,
            )

        raise


def parse_args() -> argparse.Namespace:
    """Parse quantitative evaluator execution mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PhysioTrack GazeEstimator on MPIIFaceGaze."
        )
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate environment, dataset structure, annotations, and "
            "dataset cleanliness without gaze inference or output changes."
        ),
    )

    return parser.parse_args()


def main() -> None:
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

    print(
        "=== MPIIFaceGaze Cross-Dataset "
        "Gaze Estimation Evaluation ==="
    )

    print(
        "Validating PhysioTrack GazeEstimator"
    )

    print(
        f"Model: {MODEL_MODE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Dataset: datasets/MPIIFaceGaze/Data"
    )

    print(
        f"ptgaze: {ptgaze_version}"
    )

    print()

    print(
        "Running dataset preflight..."
    )

    annotations_by_person = (
        preflight_dataset()
    )

    print(
        "Dataset preflight: PASS"
    )

    print(
        f"Participants: {len(annotations_by_person)}"
    )

    print(
        "Total annotations: "
        f"{sum(len(lines) for lines in annotations_by_person.values())}"
    )

    print()

    if args.preflight_only:
        print(
            "Preflight-only mode: no gaze inference or output "
            "changes were performed."
        )

        return

    (
        staging_dir,
        staged_output_dir,
    ) = create_staging_directory()

    runtime_dir = (
        staging_dir
        / "runtime"
    )

    print(
        f"Staging directory: {staging_dir}"
    )


    try:
        dataset_before = (
            dataset_inventory(
                DATASET_ROOT
            )
        )

        per_person_csv_path = (
            staged_output_dir
            / PER_PERSON_FILENAME
        )

        per_sample_csv_path = (
            staged_output_dir
            / PER_SAMPLE_FILENAME
        )

        summary_path = (
            staged_output_dir
            / SUMMARY_FILENAME
        )

        all_errors = []

        total_annotations = 0
        successful_predictions = 0
        image_read_failures = 0
        face_detection_failures = 0
        prediction_failures = 0
        invalid_annotation_rows = 0

        person_results = []
        sample_results = []

        checkpoint_name = None
        checkpoint_sha256 = None

        start_time = time.time()

        try:
            for participant in (
                EXPECTED_PARTICIPANTS
            ):
                person_dir = (
                    DATASET_ROOT
                    / participant
                )

                lines = (
                    annotations_by_person[
                        participant
                    ]
                )

                total_annotations += len(
                    lines
                )

                (
                    reference_image_path,
                    reference_image,
                ) = first_readable_image(
                    person_dir,
                    lines,
                )

                image_height, image_width = (
                    reference_image.shape[:2]
                )

                camera_yaml_path = (
                    create_camera_yaml(
                        person_dir,
                        image_width,
                        image_height,
                        runtime_dir,
                    )
                )

                estimator = GazeEstimator(
                    mode=MODEL_MODE,
                    device=DEVICE,
                    camera_path=(
                        camera_yaml_path
                    ),
                )

                person_errors = []
                person_image_failures = 0
                person_detection_failures = 0
                person_prediction_failures = 0
                person_invalid_annotations = 0

                person_start = time.time()

                try:
                    estimator.initialize()

                    if checkpoint_name is None:
                        checkpoint_path = (
                            estimator.checkpoint_path
                        )

                        if (
                            checkpoint_path is None
                            or not checkpoint_path.is_file()
                        ):
                            raise RuntimeError(
                                "PhysioTrack GazeEstimator did not resolve "
                                "a valid pretrained checkpoint."
                            )

                        checkpoint_name = (
                            checkpoint_path.name
                        )

                        checkpoint_sha256 = (
                            sha256_file(
                                checkpoint_path
                            )
                        )

                    for index, line in enumerate(
                        lines,
                        start=1,
                    ):
                        status = "success"
                        failure_reason = ""
                        angular_error = math.nan
                        prediction = None

                        try:
                            (
                                image_relative_path,
                                ground_truth,
                            ) = parse_annotation_line(
                                line
                            )
                        except Exception as error:
                            invalid_annotation_rows += 1
                            person_invalid_annotations += 1

                            sample_results.append(
                                {
                                    "participant": participant,
                                    "image_relative_path": "",
                                    "status": "invalid_annotation",
                                    "failure_reason": str(error),
                                    "gt_x": math.nan,
                                    "gt_y": math.nan,
                                    "gt_z": math.nan,
                                    "pred_x": math.nan,
                                    "pred_y": math.nan,
                                    "pred_z": math.nan,
                                    "angular_error_deg": math.nan,
                                }
                            )

                            continue

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
                            image_read_failures += 1
                            person_image_failures += 1
                            status = "image_read_failure"
                            failure_reason = "cv2.imread returned None"

                        else:
                            try:
                                result = (
                                    estimator
                                    .predict_image(
                                        image
                                    )
                                )

                                if not result[
                                    "available"
                                ]:
                                    face_detection_failures += 1
                                    person_detection_failures += 1
                                    status = "face_detection_failure"
                                    failure_reason = (
                                        "PhysioTrack GazeEstimator "
                                        "returned available=False"
                                    )

                                else:
                                    prediction = np.asarray(
                                        result[
                                            "gaze_vector"
                                        ],
                                        dtype=np.float64,
                                    )

                                    angular_error = (
                                        angular_error_degrees(
                                            ground_truth,
                                            prediction,
                                        )
                                    )

                                    person_errors.append(
                                        angular_error
                                    )

                                    all_errors.append(
                                        angular_error
                                    )

                                    successful_predictions += 1

                            except Exception as error:
                                prediction_failures += 1
                                person_prediction_failures += 1
                                status = "prediction_failure"
                                failure_reason = (
                                    f"{type(error).__name__}: {error}"
                                )

                        if prediction is None:
                            pred_values = (
                                math.nan,
                                math.nan,
                                math.nan,
                            )
                        else:
                            pred_values = tuple(
                                float(value)
                                for value in (
                                    prediction
                                    .reshape(-1)
                                )
                            )

                        sample_results.append(
                            {
                                "participant": participant,
                                "image_relative_path": image_relative_path,
                                "status": status,
                                "failure_reason": failure_reason,
                                "gt_x": float(ground_truth[0]),
                                "gt_y": float(ground_truth[1]),
                                "gt_z": float(ground_truth[2]),
                                "pred_x": pred_values[0],
                                "pred_y": pred_values[1],
                                "pred_z": pred_values[2],
                                "angular_error_deg": angular_error,
                            }
                        )

                        if (
                            index % 500 == 0
                            or index == len(lines)
                        ):
                            print(
                                f"{participant}: "
                                f"{index}/{len(lines)}"
                            )

                finally:
                    estimator.close()

                person_runtime = (
                    time.time()
                    - person_start
                )

                if person_errors:
                    person_array = np.asarray(
                        person_errors,
                        dtype=np.float64,
                    )

                    person_mean = float(
                        person_array.mean()
                    )

                    person_median = float(
                        np.median(
                            person_array
                        )
                    )

                    person_std = float(
                        person_array.std()
                    )

                else:
                    person_mean = math.nan
                    person_median = math.nan
                    person_std = math.nan

                person_results.append(
                    {
                        "participant": participant,
                        "annotations": len(lines),
                        "successful_predictions": len(person_errors),
                        "image_read_failures": person_image_failures,
                        "face_detection_failures": person_detection_failures,
                        "prediction_failures": person_prediction_failures,
                        "invalid_annotation_rows": person_invalid_annotations,
                        "mean_angular_error_deg": person_mean,
                        "median_angular_error_deg": person_median,
                        "std_angular_error_deg": person_std,
                        "runtime_seconds": person_runtime,
                    }
                )

                print(
                    f"{participant}: "
                    f"success={len(person_errors)}, "
                    f"mean={person_mean:.4f} deg, "
                    f"median={person_median:.4f} deg"
                )

                print()

        finally:
            if runtime_dir.exists():
                shutil.rmtree(
                    runtime_dir
                )

        total_runtime = (
            time.time()
            - start_time
        )

        dataset_after = (
            dataset_inventory(
                DATASET_ROOT
            )
        )

        if (
            dataset_before
            != dataset_after
        ):
            before_keys = set(
                dataset_before
            )

            after_keys = set(
                dataset_after
            )

            added = sorted(
                after_keys
                - before_keys
            )

            removed = sorted(
                before_keys
                - after_keys
            )

            changed = sorted(
                key
                for key in (
                    before_keys
                    & after_keys
                )
                if (
                    dataset_before[key]
                    != dataset_after[key]
                )
            )

            raise RuntimeError(
                "Dataset integrity check failed. "
                f"Added={added[:5]}, "
                f"removed={removed[:5]}, "
                f"changed={changed[:5]}."
            )

        accounted_annotations = (
            successful_predictions
            + image_read_failures
            + face_detection_failures
            + prediction_failures
            + invalid_annotation_rows
        )

        if (
            accounted_annotations
            != total_annotations
        ):
            raise RuntimeError(
                "Failure accounting invariant failed: "
                f"total={total_annotations}, "
                f"accounted={accounted_annotations}."
            )

        if (
            len(sample_results)
            != total_annotations
        ):
            raise RuntimeError(
                "Per-sample accounting invariant failed: "
                f"expected {total_annotations} rows, "
                f"found {len(sample_results)}."
            )

        error_array = np.asarray(
            all_errors,
            dtype=np.float64,
        )

        if len(
            error_array
        ):
            mean_error = float(
                error_array.mean()
            )

            median_error = float(
                np.median(
                    error_array
                )
            )

            std_error = float(
                error_array.std()
            )

            min_error = float(
                error_array.min()
            )

            max_error = float(
                error_array.max()
            )

            p90_error = float(
                np.percentile(
                    error_array,
                    90,
                )
            )

            p95_error = float(
                np.percentile(
                    error_array,
                    95,
                )
            )

        else:
            raise RuntimeError(
                "No successful gaze predictions were produced."
            )

        per_person_fieldnames = [
            "participant",
            "annotations",
            "successful_predictions",
            "image_read_failures",
            "face_detection_failures",
            "prediction_failures",
            "invalid_annotation_rows",
            "mean_angular_error_deg",
            "median_angular_error_deg",
            "std_angular_error_deg",
            "runtime_seconds",
        ]

        with open(
            per_person_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    per_person_fieldnames
                ),
            )

            writer.writeheader()
            writer.writerows(
                person_results
            )

        per_sample_fieldnames = [
            "participant",
            "image_relative_path",
            "status",
            "failure_reason",
            "gt_x",
            "gt_y",
            "gt_z",
            "pred_x",
            "pred_y",
            "pred_z",
            "angular_error_deg",
        ]

        with open(
            per_sample_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    per_sample_fieldnames
                ),
            )

            writer.writeheader()
            writer.writerows(
                sample_results
            )

        source_path = Path(
            inspect.getfile(
                GazeEstimator
            )
        ).resolve()

        source_display = (
            source_path
            .relative_to(
                REPO_ROOT
            )
            .as_posix()
            if (
                REPO_ROOT
                in source_path.parents
            )
            else source_path.name
        )

        summary_lines = [
            "MPIIFaceGaze Cross-Dataset Gaze Estimation Evaluation",
            "",
            "Validation target: PhysioTrack GazeEstimator",
            f"PhysioTrack source: {source_display}",
            f"Dataset: datasets/MPIIFaceGaze/Data",
            f"Dataset expected annotations: {EXPECTED_TOTAL_ANNOTATIONS}",
            "Dataset integrity after evaluation: PASS",
            f"Model mode: {MODEL_MODE}",
            f"Device: {DEVICE}",
            f"ptgaze version: {ptgaze_version}",
            f"OpenCV version: {cv2.__version__}",
            f"NumPy version: {np.__version__}",
            f"Checkpoint file: {checkpoint_name}",
            f"Checkpoint SHA256: {checkpoint_sha256}",
            f"Participants: {len(EXPECTED_PARTICIPANTS)}",
            f"Total annotations: {total_annotations}",
            f"Successful predictions: {successful_predictions}",
            f"Image read failures: {image_read_failures}",
            f"Face detection failures: {face_detection_failures}",
            f"Prediction failures: {prediction_failures}",
            f"Invalid annotation rows: {invalid_annotation_rows}",
            f"Accounted annotations: {accounted_annotations}",
            "",
            f"Mean angular error: {mean_error:.6f} deg",
            f"Median angular error: {median_error:.6f} deg",
            f"Std angular error: {std_error:.6f} deg",
            f"Minimum angular error: {min_error:.6f} deg",
            f"Maximum angular error: {max_error:.6f} deg",
            f"90th percentile angular error: {p90_error:.6f} deg",
            f"95th percentile angular error: {p95_error:.6f} deg",
            "",
            f"Runtime: {total_runtime:.2f} seconds",
        ]

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "\n".join(
                    summary_lines
                )
            )

        print(
            "Validating staged evaluator outputs..."
        )

        validate_staged_outputs(
            per_person_csv_path,
            per_sample_csv_path,
            summary_path,
        )

        commit_outputs(
            [
                per_person_csv_path,
                per_sample_csv_path,
                summary_path,
            ],
            staging_dir,
        )

        print(
            "Committed final evaluator outputs."
        )

        print(
            "=== Final Summary ==="
        )

        for line in summary_lines:
            print(
                line
            )

        print()

        print(
            f"Saved: {OUTPUT_DIR / SUMMARY_FILENAME}"
        )

        print(
            f"Saved: {OUTPUT_DIR / PER_PERSON_FILENAME}"
        )

        print(
            f"Saved: {OUTPUT_DIR / PER_SAMPLE_FILENAME}"
        )


    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
