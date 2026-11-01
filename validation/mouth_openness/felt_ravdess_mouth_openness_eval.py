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
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.is_dir():
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.face.mouth import MouthOpenness
from physiotrack.models import Models


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

OUTPUT_DIR = SCRIPT_DIR / "results"

DEVICE = "cpu"

SUMMARY_METRIC_TOLERANCE = 5e-7
PER_ACTOR_METRIC_TOLERANCE = 5e-12
RESULT_CSV_FLOAT_PRECISION = "round_trip"
EXPECTED_ACTORS = [
    f"Actor_{index:02d}"
    for index in range(1, 25)
]
EXPECTED_TRIALS_PER_ACTOR = 60
EXPECTED_TOTAL_TRIALS = 1440
EXPECTED_UNIQUE_ANNOTATED_FRAMES = 158286
EXPECTED_DUPLICATE_ROWS_RESOLVED = 2
EXPECTED_FPS = 30000.0 / 1001.0
FPS_TOLERANCE = 1e-4

REQUIRED_COLUMNS = {
    "frame",
    "FaceRectX",
    "FaceRectY",
    "FaceRectWidth",
    "FaceRectHeight",
    "FaceScore",
    "x_48",
    "y_48",
    "x_54",
    "y_54",
    "x_61",
    "y_61",
    "x_62",
    "y_62",
    "x_63",
    "y_63",
    "x_65",
    "y_65",
    "x_66",
    "y_66",
    "x_67",
    "y_67",
}

REFERENCE_NUMERIC_COLUMNS = sorted(
    REQUIRED_COLUMNS
    - {"frame"}
)


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


def distance_from_row(
    row: pd.Series,
    point_a: int,
    point_b: int,
) -> float:
    dx = float(
        row[f"x_{point_a}"]
        - row[f"x_{point_b}"]
    )

    dy = float(
        row[f"y_{point_a}"]
        - row[f"y_{point_b}"]
    )

    return math.hypot(
        dx,
        dy,
    )


def resolve_duplicate_frames(
    frame_table: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    table = frame_table.copy()

    table["_face_area"] = (
        table["FaceRectWidth"]
        * table["FaceRectHeight"]
    )

    unique_frame_count = int(
        table["frame"].nunique()
    )

    duplicate_rows_resolved = int(
        len(table)
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
            subset=["frame"],
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


def reference_values(
    row: pd.Series,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    values = np.asarray(
        [
            row[column]
            for column in REFERENCE_NUMERIC_COLUMNS
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "FELT annotation contains non-finite required values."
        )

    face_width = float(
        row["FaceRectWidth"]
    )

    face_height = float(
        row["FaceRectHeight"]
    )

    if (
        face_width <= 0
        or face_height <= 0
    ):
        raise ValueError(
            "FELT FaceRect has non-positive dimensions."
        )

    mouth_width = distance_from_row(
        row,
        48,
        54,
    )

    central_height = distance_from_row(
        row,
        62,
        66,
    )

    three_pair_height = float(
        np.mean(
            [
                distance_from_row(
                    row,
                    61,
                    67,
                ),
                central_height,
                distance_from_row(
                    row,
                    63,
                    65,
                ),
            ]
        )
    )

    if (
        not math.isfinite(
            mouth_width
        )
        or mouth_width <= 0
    ):
        raise ValueError(
            "FELT mouth-corner width is invalid."
        )

    primary_reference = (
        central_height
        / mouth_width
    )

    secondary_reference = (
        three_pair_height
        / mouth_width
    )

    if (
        not math.isfinite(
            primary_reference
        )
        or not math.isfinite(
            secondary_reference
        )
    ):
        raise ValueError(
            "FELT mouth-openness reference is non-finite."
        )

    return (
        mouth_width,
        central_height,
        primary_reference,
        secondary_reference,
    )


def pearson_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if (
        reference.size < 2
        or prediction.size < 2
    ):
        return math.nan

    if (
        np.std(reference) <= 0
        or np.std(prediction) <= 0
    ):
        return math.nan

    return float(
        np.corrcoef(
            reference,
            prediction,
        )[0, 1]
    )


def spearman_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_ranks = (
        pd.Series(
            reference,
            dtype="float64",
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    prediction_ranks = (
        pd.Series(
            prediction,
            dtype="float64",
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    return pearson_correlation(
        reference_ranks,
        prediction_ranks,
    )


def concordance_correlation_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if (
        reference.size < 2
        or prediction.size < 2
    ):
        return math.nan

    reference_mean = float(
        reference.mean()
    )

    prediction_mean = float(
        prediction.mean()
    )

    reference_variance = float(
        reference.var()
    )

    prediction_variance = float(
        prediction.var()
    )

    covariance = float(
        np.mean(
            (
                reference
                - reference_mean
            )
            * (
                prediction
                - prediction_mean
            )
        )
    )

    denominator = (
        reference_variance
        + prediction_variance
        + (
            reference_mean
            - prediction_mean
        ) ** 2
    )

    if denominator <= 0:
        return math.nan

    return float(
        2.0
        * covariance
        / denominator
    )


def regression_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if (
        reference.size == 0
        or prediction.size == 0
        or reference.size != prediction.size
    ):
        raise ValueError(
            "Regression metrics require non-empty paired arrays."
        )

    if (
        not np.all(
            np.isfinite(
                reference
            )
        )
        or not np.all(
            np.isfinite(
                prediction
            )
        )
    ):
        raise ValueError(
            "Regression metrics require finite paired arrays."
        )

    signed_error = (
        prediction
        - reference
    )

    absolute_error = np.abs(
        signed_error
    )

    squared_error = (
        signed_error
        ** 2
    )

    return {
        "mae": float(
            absolute_error.mean()
        ),
        "rmse": float(
            math.sqrt(
                squared_error.mean()
            )
        ),
        "median_absolute_error": float(
            np.median(
                absolute_error
            )
        ),
        "std_absolute_error": float(
            absolute_error.std()
        ),
        "p90_absolute_error": float(
            np.percentile(
                absolute_error,
                90,
            )
        ),
        "p95_absolute_error": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),
        "mean_signed_error": float(
            signed_error.mean()
        ),
        "pearson_r": pearson_correlation(
            reference,
            prediction,
        ),
        "spearman_rho": spearman_correlation(
            reference,
            prediction,
        ),
        "ccc": concordance_correlation_coefficient(
            reference,
            prediction,
        ),
    }


def validate_trial_filename(
    actor: str,
    csv_path: Path,
) -> None:
    parts = csv_path.stem.split("-")

    if len(parts) != 7:
        raise RuntimeError(
            "Unexpected RAVDESS/FELT trial filename: "
            f"{csv_path.name}"
        )

    if parts[0] != "01":
        raise RuntimeError(
            "Mouth-openness validation expects audio-video speech "
            f"trials beginning with '01-': {csv_path.name}"
        )

    expected_actor_token = actor.split("_")[-1]

    if parts[-1] != expected_actor_token:
        raise RuntimeError(
            "Trial actor token does not match its actor directory: "
            f"{actor}/{csv_path.name}"
        )


def preflight_datasets() -> tuple[
    list[dict[str, object]],
    int,
]:
    if not FELT_ROOT.is_dir():
        raise FileNotFoundError(
            "FELT speech annotations were not found at the expected "
            "project-relative location:\n"
            "datasets/FELT/raw_motion_speech"
        )

    if not RAVDESS_ROOT.is_dir():
        raise FileNotFoundError(
            "RAVDESS speech videos were not found at the expected "
            "project-relative location:\n"
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

    if felt_actor_dirs != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected FELT actor structure. "
            f"Expected {EXPECTED_ACTORS}, found {felt_actor_dirs}."
        )

    if ravdess_actor_dirs != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected RAVDESS actor structure. "
            f"Expected {EXPECTED_ACTORS}, found {ravdess_actor_dirs}."
        )

    trials = []
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
            len(csv_paths)
            != EXPECTED_TRIALS_PER_ACTOR
        ):
            raise RuntimeError(
                f"Unexpected FELT trial count for {actor}. "
                f"Expected {EXPECTED_TRIALS_PER_ACTOR}, "
                f"found {len(csv_paths)}."
            )

        expected_video_names = {
            f"{path.stem}.mp4"
            for path in csv_paths
        }

        matching_video_paths = sorted(
            ravdess_actor_dir.glob(
                "01-*.mp4"
            )
        )

        matching_video_names = {
            path.name
            for path in matching_video_paths
        }

        if (
            matching_video_names
            != expected_video_names
        ):
            missing = sorted(
                expected_video_names
                - matching_video_names
            )

            extra = sorted(
                matching_video_names
                - expected_video_names
            )

            raise RuntimeError(
                f"FELT/RAVDESS pairing mismatch for {actor}. "
                f"Missing videos={missing[:5]}, "
                f"extra videos={extra[:5]}."
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
                frame_table["frame"],
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

            frame_table["frame"] = (
                frame_values.astype(
                    np.int64
                )
            )

            (
                unique_table,
                duplicate_rows_resolved,
            ) = resolve_duplicate_frames(
                frame_table
            )

            frame_ids = (
                unique_table["frame"]
                .to_numpy(
                    dtype=np.int64
                )
            )

            expected_frame_ids = np.arange(
                len(unique_table),
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

            cap = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            if not cap.isOpened():
                cap.release()

                raise RuntimeError(
                    "RAVDESS video could not be opened during preflight: "
                    f"{actor}/{video_path.name}"
                )

            video_frame_count = int(
                round(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )
            )

            video_fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            cap.release()

            if (
                video_frame_count
                < len(unique_table)
            ):
                raise RuntimeError(
                    "RAVDESS video has fewer frames than the FELT "
                    "annotation range for "
                    f"{actor}/{video_path.name}: "
                    f"video_frames={video_frame_count}, "
                    f"annotated_frames={len(unique_table)}."
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
                    f"{actor}/{video_path.name}: {video_fps}."
                )

            trials.append(
                {
                    "actor": actor,
                    "csv_path": csv_path,
                    "video_path": video_path,
                    "annotated_frames": len(
                        unique_table
                    ),
                    "raw_annotation_rows": len(
                        frame_table
                    ),
                    "duplicate_rows_resolved": duplicate_rows_resolved,
                    "video_frames": video_frame_count,
                    "video_fps": video_fps,
                }
            )

            total_unique_frames += len(
                unique_table
            )

            total_duplicate_rows_resolved += (
                duplicate_rows_resolved
            )

    if len(trials) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(
            "Unexpected paired trial count. "
            f"Expected {EXPECTED_TOTAL_TRIALS}, found {len(trials)}."
        )

    if (
        total_unique_frames
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Unexpected unique annotated-frame count. "
            f"Expected {EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found {total_unique_frames}."
        )

    if (
        total_duplicate_rows_resolved
        != EXPECTED_DUPLICATE_ROWS_RESOLVED
    ):
        raise RuntimeError(
            "Unexpected duplicate-row count. "
            f"Expected {EXPECTED_DUPLICATE_ROWS_RESOLVED}, "
            f"found {total_duplicate_rows_resolved}."
        )

    return (
        trials,
        total_duplicate_rows_resolved,
    )


def source_display_path(
    source_object,
) -> str:
    source_path = Path(
        inspect.getfile(
            source_object
        )
    ).resolve()

    if REPO_ROOT in source_path.parents:
        return (
            source_path
            .relative_to(
                REPO_ROOT
            )
            .as_posix()
        )

    return source_path.name


def format_metric(
    value: float,
) -> str:
    if math.isfinite(
        float(value)
    ):
        return f"{float(value):.6f}"

    return "nan"



def parse_summary_scalar(
    summary_path: Path,
    label: str,
) -> str:
    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith(
                f"{label}:"
            ):
                return line.split(
                    ":",
                    1,
                )[1].strip()

    raise RuntimeError(
        f"Required staged summary value was not found: {label}"
    )


def parse_summary_metric_sections(
    summary_path: Path,
) -> dict[
    str,
    dict[
        str,
        float,
    ],
]:
    section_titles = {
        "Primary Reference Metrics":
            "primary",
        "Secondary Three-Pair Reference Metrics":
            "secondary",
        "Face-Box-Normalized Sensitivity Metrics":
            "facebox",
    }

    metric_labels = {
        "MAE":
            "mae",
        "RMSE":
            "rmse",
        "Median absolute error":
            "median_absolute_error",
        "Std absolute error":
            "std_absolute_error",
        "90th percentile absolute error":
            "p90_absolute_error",
        "95th percentile absolute error":
            "p95_absolute_error",
        "Mean signed error (prediction - reference)":
            "mean_signed_error",
        "Pearson r":
            "pearson_r",
        "Spearman rho":
            "spearman_rho",
        "Lin CCC":
            "ccc",
    }

    parsed = {
        section: {}
        for section in (
            "primary",
            "secondary",
            "facebox",
        )
    }

    active_section = None

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line in section_titles:
                active_section = (
                    section_titles[
                        line
                    ]
                )
                continue

            if (
                active_section is None
                or ":" not in line
            ):
                continue

            label, value_text = line.split(
                ":",
                1,
            )

            label = label.strip()

            if label not in metric_labels:
                continue

            parsed[
                active_section
            ][
                metric_labels[
                    label
                ]
            ] = float(
                value_text.strip()
            )

    for section, metrics in parsed.items():
        missing = sorted(
            set(
                metric_labels.values()
            )
            - set(
                metrics
            )
        )

        if missing:
            raise RuntimeError(
                "Staged summary is missing required "
                f"{section} metrics: {missing}"
            )

    return parsed


def validate_staged_quantitative_outputs(
    per_frame_csv_path: Path,
    per_actor_csv_path: Path,
    summary_path: Path,
) -> None:
    for path in (
        per_frame_csv_path,
        per_actor_csv_path,
        summary_path,
    ):
        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Missing or empty staged quantitative output: {path}"
            )

    frame_table = pd.read_csv(
        per_frame_csv_path,
        float_precision=RESULT_CSV_FLOAT_PRECISION,
    )

    actor_table = pd.read_csv(
        per_actor_csv_path,
        float_precision=RESULT_CSV_FLOAT_PRECISION,
    )

    required_frame_columns = {
        "actor",
        "trial",
        "frame",
        "status",
        "failure_reason",
        "duplicate_candidates",
        "felt_reference",
        "felt_reference_three_pair",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "felt_reference_facebox",
        "physiotrack_openness_facebox",
        "facebox_signed_error",
        "facebox_absolute_error",
    }

    missing_frame_columns = sorted(
        required_frame_columns
        - set(
            frame_table.columns
        )
    )

    if missing_frame_columns:
        raise RuntimeError(
            "Staged per-frame output is missing required columns: "
            f"{missing_frame_columns}"
        )

    if len(
        frame_table
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Unexpected staged per-frame row count: "
            f"expected={EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found={len(frame_table)}."
        )

    if frame_table.duplicated(
        [
            "actor",
            "trial",
            "frame",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in staged output."
        )

    expected_statuses = {
        "success",
        "video_read_failure",
        "landmark_failure",
        "invalid_reference",
        "prediction_failure",
    }

    observed_statuses = set(
        frame_table[
            "status"
        ].astype(
            str
        )
    )

    unexpected_statuses = sorted(
        observed_statuses
        - expected_statuses
    )

    if unexpected_statuses:
        raise RuntimeError(
            "Unexpected staged per-frame statuses: "
            f"{unexpected_statuses}"
        )

    if len(
        actor_table
    ) != len(
        EXPECTED_ACTORS
    ):
        raise RuntimeError(
            "Unexpected staged per-actor row count: "
            f"expected={len(EXPECTED_ACTORS)}, "
            f"found={len(actor_table)}."
        )

    if actor_table[
        "actor"
    ].tolist() != EXPECTED_ACTORS:
        raise RuntimeError(
            "Staged per-actor rows do not preserve the expected actor order."
        )

    if int(
        actor_table[
            "annotations"
        ].sum()
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Staged per-actor annotation total is inconsistent."
        )

    status_counts = frame_table[
        "status"
    ].value_counts()

    expected_counts = {
        "success": int(
            parse_summary_scalar(
                summary_path,
                "Successful predictions",
            )
        ),
        "video_read_failure": int(
            parse_summary_scalar(
                summary_path,
                "Video read failures",
            )
        ),
        "landmark_failure": int(
            parse_summary_scalar(
                summary_path,
                "Landmark failures",
            )
        ),
        "invalid_reference": int(
            parse_summary_scalar(
                summary_path,
                "Invalid references",
            )
        ),
        "prediction_failure": int(
            parse_summary_scalar(
                summary_path,
                "Prediction failures",
            )
        ),
    }

    for status, expected_count in expected_counts.items():
        observed_count = int(
            status_counts.get(
                status,
                0,
            )
        )

        if observed_count != expected_count:
            raise RuntimeError(
                "Staged status/summary count mismatch for "
                f"{status}: observed={observed_count}, "
                f"summary={expected_count}."
            )

    if sum(
        expected_counts.values()
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Staged summary failure accounting is incomplete."
        )

    if int(
        parse_summary_scalar(
            summary_path,
            "Accounted annotated frames",
        )
    ) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Staged summary accounted-frame count is inconsistent."
        )

    summary_text = summary_path.read_text(
        encoding="utf-8"
    )

    for required_line in (
        "FELT dataset integrity after evaluation: PASS",
        "RAVDESS dataset integrity after evaluation: PASS",
    ):
        if required_line not in summary_text:
            raise RuntimeError(
                f"Staged summary is missing required integrity evidence: "
                f"{required_line}"
            )

    success_table = frame_table[
        frame_table[
            "status"
        ]
        == "success"
    ].copy()

    if success_table.empty:
        raise RuntimeError(
            "Staged output contains no successful predictions."
        )

    numeric_columns = [
        "felt_reference",
        "felt_reference_three_pair",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "felt_reference_facebox",
        "physiotrack_openness_facebox",
        "facebox_signed_error",
        "facebox_absolute_error",
    ]

    numeric_values = success_table[
        numeric_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            numeric_values
        )
    ):
        raise RuntimeError(
            "Successful staged rows contain non-finite quantitative values."
        )

    primary_reference = success_table[
        "felt_reference"
    ].to_numpy(
        dtype=np.float64
    )

    secondary_reference = success_table[
        "felt_reference_three_pair"
    ].to_numpy(
        dtype=np.float64
    )

    prediction = success_table[
        "physiotrack_openness"
    ].to_numpy(
        dtype=np.float64
    )

    facebox_reference = success_table[
        "felt_reference_facebox"
    ].to_numpy(
        dtype=np.float64
    )

    facebox_prediction = success_table[
        "physiotrack_openness_facebox"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.allclose(
        success_table[
            "signed_error"
        ].to_numpy(
            dtype=np.float64
        ),
        prediction
        - primary_reference,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Staged primary signed-error values are inconsistent."
        )

    if not np.allclose(
        success_table[
            "absolute_error"
        ].to_numpy(
            dtype=np.float64
        ),
        np.abs(
            prediction
            - primary_reference
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Staged primary absolute-error values are inconsistent."
        )

    if not np.allclose(
        success_table[
            "facebox_signed_error"
        ].to_numpy(
            dtype=np.float64
        ),
        facebox_prediction
        - facebox_reference,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Staged face-box signed-error values are inconsistent."
        )

    if not np.allclose(
        success_table[
            "facebox_absolute_error"
        ].to_numpy(
            dtype=np.float64
        ),
        np.abs(
            facebox_prediction
            - facebox_reference
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Staged face-box absolute-error values are inconsistent."
        )

    recomputed_metrics = {
        "primary": regression_metrics(
            primary_reference,
            prediction,
        ),
        "secondary": regression_metrics(
            secondary_reference,
            prediction,
        ),
        "facebox": regression_metrics(
            facebox_reference,
            facebox_prediction,
        ),
    }

    summary_metrics = (
        parse_summary_metric_sections(
            summary_path
        )
    )

    for section, metrics in recomputed_metrics.items():
        for metric_name, observed_value in metrics.items():
            summary_value = (
                summary_metrics[
                    section
                ][
                    metric_name
                ]
            )

            if not math.isclose(
                observed_value,
                summary_value,
                rel_tol=0.0,
                abs_tol=SUMMARY_METRIC_TOLERANCE,
            ):
                raise RuntimeError(
                    "Staged summary/per-frame metric mismatch for "
                    f"{section}/{metric_name}: "
                    f"recomputed={observed_value}, "
                    f"summary={summary_value}."
                )

    actor_lookup = actor_table.set_index(
        "actor"
    )

    for actor, group in success_table.groupby(
        "actor",
        sort=True,
    ):
        if actor not in actor_lookup.index:
            raise RuntimeError(
                f"Actor missing from staged per-actor output: {actor}"
            )

        actor_primary = regression_metrics(
            group[
                "felt_reference"
            ].to_numpy(
                dtype=np.float64
            ),
            group[
                "physiotrack_openness"
            ].to_numpy(
                dtype=np.float64
            ),
        )

        actor_secondary = regression_metrics(
            group[
                "felt_reference_three_pair"
            ].to_numpy(
                dtype=np.float64
            ),
            group[
                "physiotrack_openness"
            ].to_numpy(
                dtype=np.float64
            ),
        )

        actor_facebox = regression_metrics(
            group[
                "felt_reference_facebox"
            ].to_numpy(
                dtype=np.float64
            ),
            group[
                "physiotrack_openness_facebox"
            ].to_numpy(
                dtype=np.float64
            ),
        )

        expected_actor = actor_lookup.loc[
            actor
        ]

        for prefix, metrics in (
            (
                "primary",
                actor_primary,
            ),
            (
                "secondary",
                actor_secondary,
            ),
            (
                "facebox",
                actor_facebox,
            ),
        ):
            for metric_name, observed_value in metrics.items():
                column = (
                    f"{prefix}_"
                    f"{metric_name}"
                )

                if column not in actor_table.columns:
                    raise RuntimeError(
                        "Staged per-actor output is missing required "
                        f"metric column: {column}"
                    )

                expected_value = float(
                    expected_actor[
                        column
                    ]
                )

                if (
                    math.isnan(
                        expected_value
                    )
                    and math.isnan(
                        observed_value
                    )
                ):
                    continue

                if not math.isclose(
                    expected_value,
                    observed_value,
                    rel_tol=0.0,
                    abs_tol=PER_ACTOR_METRIC_TOLERANCE,
                ):
                    raise RuntimeError(
                        "Staged per-actor metric mismatch for "
                        f"{actor}/{column}: "
                        f"recomputed={observed_value}, "
                        f"stored={expected_value}."
                    )



def commit_owned_files(
    staged_to_final: list[
        tuple[
            Path,
            Path,
        ]
    ],
    staging_dir: Path,
) -> None:
    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prior_files = {}

    for _, final_path in staged_to_final:
        if final_path.is_file():
            backup_path = (
                backup_dir
                / final_path.name
            )

            shutil.copy2(
                final_path,
                backup_path,
            )

            prior_files[
                final_path
            ] = backup_path

    installed = []

    try:
        for staged_path, final_path in staged_to_final:
            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_final = (
                final_path.parent
                / (
                    f".{final_path.name}."
                    f"{uuid.uuid4().hex}.tmp"
                )
            )

            shutil.copy2(
                staged_path,
                temporary_final,
            )

            os.replace(
                temporary_final,
                final_path,
            )

            installed.append(
                final_path
            )

    except Exception:
        for final_path in installed:
            backup_path = prior_files.get(
                final_path
            )

            if backup_path is not None:
                shutil.copy2(
                    backup_path,
                    final_path,
                )

            elif final_path.exists():
                final_path.unlink()

        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PhysioTrack mouth openness on the complete "
            "FELT/RAVDESS speech benchmark."
        )
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the locked dataset/protocol without running "
            "landmark or mouth-openness inference."
        ),
    )

    mode_group.add_argument(
        "--validate-existing-results-only",
        action="store_true",
        help=(
            "Run the complete serialized-result validator on the current "
            "accepted quantitative outputs without rerunning inference."
        ),
    )

    args = parser.parse_args()

    print(
        "=== FELT/RAVDESS Mouth Openness Evaluation ==="
    )

    print(
        "Validation target: PhysioTrack FaceLandmarks + MouthOpenness"
    )

    print(
        "Primary FELT reference: d(62,66) / d(48,54)"
    )

    print(
        "Secondary FELT reference: mean[d(61,67), d(62,66), "
        "d(63,65)] / d(48,54)"
    )

    print(
        "Face initialization: FELT FaceRect"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "FELT dataset: datasets/FELT/raw_motion_speech"
    )

    print(
        "RAVDESS dataset: datasets/RAVDESS/Video_Speech"
    )

    print()

    print(
        "Running dataset preflight..."
    )

    (
        trials,
        duplicate_rows_resolved,
    ) = preflight_datasets()

    print(
        "Dataset preflight: PASS"
    )

    print(
        f"Actors: {len(EXPECTED_ACTORS)}"
    )

    print(
        f"Paired speech trials: {len(trials)}"
    )

    print(
        "Unique annotated frames: "
        f"{sum(int(trial['annotated_frames']) for trial in trials)}"
    )

    print(
        "Duplicate annotation rows resolved: "
        f"{duplicate_rows_resolved}"
    )

    if args.preflight_only:
        print()
        print(
            "Preflight-only mode: no landmark or mouth-openness "
            "inference was run."
        )
        return

    if args.validate_existing_results_only:
        print()
        print(
            "Validating current serialized quantitative outputs..."
        )

        validate_staged_quantitative_outputs(
            OUTPUT_DIR
            / "felt_ravdess_mouth_openness_per_frame.csv",
            OUTPUT_DIR
            / "felt_ravdess_mouth_openness_per_actor.csv",
            OUTPUT_DIR
            / "felt_ravdess_mouth_openness_summary.txt",
        )

        print(
            "Existing quantitative output validation: PASS"
        )

        print(
            "Validation-only mode: no landmark or mouth-openness "
            "inference was run."
        )
        return

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_context = tempfile.TemporaryDirectory(
        prefix=".mouth_openness_eval_staging_",
        dir=OUTPUT_DIR,
    )

    staging_dir = Path(
        staging_context.name
    )

    print(
        f"Staging directory created: {staging_dir}"
    )

    felt_before = dataset_inventory(
        FELT_ROOT
    )

    ravdess_before = dataset_inventory(
        RAVDESS_ROOT
    )

    final_per_frame_csv_path = (
        OUTPUT_DIR
        / "felt_ravdess_mouth_openness_per_frame.csv"
    )

    final_per_actor_csv_path = (
        OUTPUT_DIR
        / "felt_ravdess_mouth_openness_per_actor.csv"
    )

    final_summary_path = (
        OUTPUT_DIR
        / "felt_ravdess_mouth_openness_summary.txt"
    )

    per_frame_csv_path = (
        staging_dir
        / final_per_frame_csv_path.name
    )

    per_actor_csv_path = (
        staging_dir
        / final_per_actor_csv_path.name
    )

    summary_path = (
        staging_dir
        / final_summary_path.name
    )

    print(
        "Previous accepted quantitative outputs are preserved until "
        "the staged rerun passes validation."
    )

    print()

    mediapipe_version = package_version(
        "mediapipe"
    )

    pandas_version = package_version(
        "pandas"
    )

    model_path = Models.resolve(
        Models.Face.MediaPipe.Landmarks.face_landmarker
    )

    model_path = Path(
        model_path
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            "PhysioTrack MediaPipe face-landmarker model could not be resolved."
        )

    model_sha256 = sha256_file(
        model_path
    )

    landmarker = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    mouth = MouthOpenness()

    frame_results = []
    actor_accumulators = {
        actor: {
            "annotations": 0,
            "successful_predictions": 0,
            "video_read_failures": 0,
            "landmark_failures": 0,
            "invalid_references": 0,
            "prediction_failures": 0,
            "primary_reference": [],
            "secondary_reference": [],
            "prediction": [],
            "facebox_reference": [],
            "facebox_prediction": [],
        }
        for actor in EXPECTED_ACTORS
    }

    total_annotations = 0
    successful_predictions = 0
    video_read_failures = 0
    landmark_failures = 0
    invalid_references = 0
    prediction_failures = 0

    primary_reference_values = []
    secondary_reference_values = []
    prediction_values = []
    facebox_reference_values = []
    facebox_prediction_values = []

    start_time = time.time()

    try:
        for trial_index, trial in enumerate(
            trials,
            start=1,
        ):
            actor = str(
                trial["actor"]
            )

            csv_path = Path(
                trial["csv_path"]
            )

            video_path = Path(
                trial["video_path"]
            )

            frame_table = pd.read_csv(
                csv_path
            )

            frame_table["frame"] = pd.to_numeric(
                frame_table["frame"],
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

            cap = cv2.VideoCapture(
                str(
                    video_path
                )
            )

            video_opened = cap.isOpened()

            trial_start = time.time()

            for _, row in frame_table.iterrows():
                frame_id = int(
                    row["frame"]
                )

                total_annotations += 1
                actor_accumulators[
                    actor
                ]["annotations"] += 1

                status = "success"
                failure_reason = ""

                felt_mouth_width = math.nan
                felt_mouth_height = math.nan
                primary_reference = math.nan
                secondary_reference = math.nan
                physiotrack_openness = math.nan
                physiotrack_mouth_width = math.nan
                physiotrack_mouth_height = math.nan
                signed_error = math.nan
                absolute_error = math.nan
                felt_reference_facebox = math.nan
                physiotrack_openness_facebox = math.nan
                facebox_signed_error = math.nan
                facebox_absolute_error = math.nan

                try:
                    (
                        felt_mouth_width,
                        felt_mouth_height,
                        primary_reference,
                        secondary_reference,
                    ) = reference_values(
                        row
                    )
                except Exception as error:
                    invalid_references += 1
                    actor_accumulators[
                        actor
                    ]["invalid_references"] += 1
                    status = "invalid_reference"
                    failure_reason = (
                        f"{type(error).__name__}: {error}"
                    )

                frame = None

                if not video_opened:
                    if status == "success":
                        video_read_failures += 1
                        actor_accumulators[
                            actor
                        ]["video_read_failures"] += 1
                        status = "video_read_failure"
                        failure_reason = "cv2.VideoCapture could not open video"

                else:
                    read_ok, frame = cap.read()

                    if (
                        not read_ok
                        or frame is None
                    ):
                        if status == "success":
                            video_read_failures += 1
                            actor_accumulators[
                                actor
                            ]["video_read_failures"] += 1
                            status = "video_read_failure"
                            failure_reason = "cv2.VideoCapture.read returned failure"

                if (
                    status == "success"
                    and frame is not None
                ):
                    box = (
                        float(
                            row["FaceRectX"]
                        ),
                        float(
                            row["FaceRectY"]
                        ),
                        float(
                            row["FaceRectX"]
                            + row["FaceRectWidth"]
                        ),
                        float(
                            row["FaceRectY"]
                            + row["FaceRectHeight"]
                        ),
                    )

                    try:
                        landmarks = (
                            landmarker.predict_face(
                                frame,
                                box,
                            )
                        )

                        if landmarks is None:
                            landmark_failures += 1
                            actor_accumulators[
                                actor
                            ]["landmark_failures"] += 1
                            status = "landmark_failure"
                            failure_reason = (
                                "PhysioTrack FaceLandmarks.predict_face "
                                "returned None"
                            )

                        else:
                            mouth_result = mouth.predict(
                                landmarks,
                                image_size=(
                                    frame.shape[1],
                                    frame.shape[0],
                                ),
                            )

                            prediction = mouth_result[
                                "mouth_openness"
                            ]

                            mouth_width = mouth_result[
                                "mouth_width"
                            ]

                            mouth_height = mouth_result[
                                "mouth_height"
                            ]

                            if (
                                prediction is None
                                or mouth_width is None
                                or mouth_height is None
                                or not math.isfinite(
                                    float(prediction)
                                )
                                or not math.isfinite(
                                    float(mouth_width)
                                )
                                or not math.isfinite(
                                    float(mouth_height)
                                )
                            ):
                                prediction_failures += 1
                                actor_accumulators[
                                    actor
                                ]["prediction_failures"] += 1
                                status = "prediction_failure"
                                failure_reason = (
                                    "PhysioTrack MouthOpenness returned "
                                    "missing or non-finite output"
                                )

                            else:
                                physiotrack_openness = float(
                                    prediction
                                )

                                physiotrack_mouth_width = float(
                                    mouth_width
                                )

                                physiotrack_mouth_height = float(
                                    mouth_height
                                )

                                frame_width = float(
                                    frame.shape[1]
                                )

                                face_rect_height = float(
                                    row["FaceRectHeight"]
                                )

                                felt_reference_facebox = (
                                    felt_mouth_height
                                    / face_rect_height
                                )

                                physiotrack_openness_facebox = (
                                    physiotrack_mouth_height
                                    * frame_width
                                    / face_rect_height
                                )

                                facebox_signed_error = (
                                    physiotrack_openness_facebox
                                    - felt_reference_facebox
                                )

                                facebox_absolute_error = abs(
                                    facebox_signed_error
                                )

                                signed_error = (
                                    physiotrack_openness
                                    - primary_reference
                                )

                                absolute_error = abs(
                                    signed_error
                                )

                                successful_predictions += 1
                                actor_accumulators[
                                    actor
                                ]["successful_predictions"] += 1

                                primary_reference_values.append(
                                    primary_reference
                                )

                                secondary_reference_values.append(
                                    secondary_reference
                                )

                                prediction_values.append(
                                    physiotrack_openness
                                )

                                facebox_reference_values.append(
                                    felt_reference_facebox
                                )

                                facebox_prediction_values.append(
                                    physiotrack_openness_facebox
                                )

                                actor_accumulators[
                                    actor
                                ]["primary_reference"].append(
                                    primary_reference
                                )

                                actor_accumulators[
                                    actor
                                ]["secondary_reference"].append(
                                    secondary_reference
                                )

                                actor_accumulators[
                                    actor
                                ]["prediction"].append(
                                    physiotrack_openness
                                )

                                actor_accumulators[
                                    actor
                                ]["facebox_reference"].append(
                                    felt_reference_facebox
                                )

                                actor_accumulators[
                                    actor
                                ]["facebox_prediction"].append(
                                    physiotrack_openness_facebox
                                )

                    except Exception as error:
                        prediction_failures += 1
                        actor_accumulators[
                            actor
                        ]["prediction_failures"] += 1
                        status = "prediction_failure"
                        failure_reason = (
                            f"{type(error).__name__}: {error}"
                        )

                frame_results.append(
                    {
                        "actor": actor,
                        "trial": csv_path.stem,
                        "frame": frame_id,
                        "status": status,
                        "failure_reason": failure_reason,
                        "duplicate_candidates": int(
                            raw_frame_counts.get(
                                frame_id,
                                1,
                            )
                        ),
                        "felt_reference": primary_reference,
                        "felt_reference_three_pair": secondary_reference,
                        "physiotrack_openness": physiotrack_openness,
                        "signed_error": signed_error,
                        "absolute_error": absolute_error,
                        "felt_reference_facebox": felt_reference_facebox,
                        "physiotrack_openness_facebox": physiotrack_openness_facebox,
                        "facebox_signed_error": facebox_signed_error,
                        "facebox_absolute_error": facebox_absolute_error,
                        "felt_mouth_width": felt_mouth_width,
                        "felt_mouth_height": felt_mouth_height,
                        "physiotrack_mouth_width": physiotrack_mouth_width,
                        "physiotrack_mouth_height": physiotrack_mouth_height,
                        "FaceScore": float(
                            row["FaceScore"]
                        ),
                        "FaceRectX": float(
                            row["FaceRectX"]
                        ),
                        "FaceRectY": float(
                            row["FaceRectY"]
                        ),
                        "FaceRectWidth": float(
                            row["FaceRectWidth"]
                        ),
                        "FaceRectHeight": float(
                            row["FaceRectHeight"]
                        ),
                    }
                )

            cap.release()

            trial_runtime = (
                time.time()
                - trial_start
            )

            if (
                trial_index % 20 == 0
                or trial_index == len(trials)
            ):
                print(
                    f"Trials: {trial_index}/{len(trials)} | "
                    f"frames={total_annotations} | "
                    f"success={successful_predictions} | "
                    f"last_trial={trial_runtime:.2f}s"
                )

    finally:
        landmarker.close()

    total_runtime = (
        time.time()
        - start_time
    )

    felt_after = dataset_inventory(
        FELT_ROOT
    )

    ravdess_after = dataset_inventory(
        RAVDESS_ROOT
    )

    if felt_before != felt_after:
        raise RuntimeError(
            "FELT dataset integrity check failed: dataset contents "
            "changed during evaluation."
        )

    if ravdess_before != ravdess_after:
        raise RuntimeError(
            "RAVDESS dataset integrity check failed: dataset contents "
            "changed during evaluation."
        )

    accounted_annotations = (
        successful_predictions
        + video_read_failures
        + landmark_failures
        + invalid_references
        + prediction_failures
    )

    if accounted_annotations != total_annotations:
        raise RuntimeError(
            "Failure accounting invariant failed: "
            f"total={total_annotations}, "
            f"accounted={accounted_annotations}."
        )

    if (
        total_annotations
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Final annotated-frame count does not match the locked "
            "protocol: "
            f"expected={EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found={total_annotations}."
        )

    if len(frame_results) != total_annotations:
        raise RuntimeError(
            "Per-frame accounting invariant failed: "
            f"expected {total_annotations} rows, "
            f"found {len(frame_results)}."
        )

    if successful_predictions == 0:
        raise RuntimeError(
            "No successful mouth-openness predictions were produced."
        )

    primary_reference_array = np.asarray(
        primary_reference_values,
        dtype=np.float64,
    )

    secondary_reference_array = np.asarray(
        secondary_reference_values,
        dtype=np.float64,
    )

    prediction_array = np.asarray(
        prediction_values,
        dtype=np.float64,
    )

    facebox_reference_array = np.asarray(
        facebox_reference_values,
        dtype=np.float64,
    )

    facebox_prediction_array = np.asarray(
        facebox_prediction_values,
        dtype=np.float64,
    )

    primary_metrics = regression_metrics(
        primary_reference_array,
        prediction_array,
    )

    secondary_metrics = regression_metrics(
        secondary_reference_array,
        prediction_array,
    )

    facebox_metrics = regression_metrics(
        facebox_reference_array,
        facebox_prediction_array,
    )

    actor_results = []

    for actor in EXPECTED_ACTORS:
        accumulator = actor_accumulators[
            actor
        ]

        actor_annotations = int(
            accumulator[
                "annotations"
            ]
        )

        actor_successes = int(
            accumulator[
                "successful_predictions"
            ]
        )

        actor_accounted = (
            actor_successes
            + int(
                accumulator[
                    "video_read_failures"
                ]
            )
            + int(
                accumulator[
                    "landmark_failures"
                ]
            )
            + int(
                accumulator[
                    "invalid_references"
                ]
            )
            + int(
                accumulator[
                    "prediction_failures"
                ]
            )
        )

        if actor_accounted != actor_annotations:
            raise RuntimeError(
                "Per-actor failure accounting invariant failed for "
                f"{actor}: annotations={actor_annotations}, "
                f"accounted={actor_accounted}."
            )

        if actor_successes > 0:
            actor_primary_metrics = regression_metrics(
                np.asarray(
                    accumulator[
                        "primary_reference"
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    accumulator[
                        "prediction"
                    ],
                    dtype=np.float64,
                ),
            )

            actor_secondary_metrics = regression_metrics(
                np.asarray(
                    accumulator[
                        "secondary_reference"
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    accumulator[
                        "prediction"
                    ],
                    dtype=np.float64,
                ),
            )

            actor_facebox_metrics = regression_metrics(
                np.asarray(
                    accumulator[
                        "facebox_reference"
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    accumulator[
                        "facebox_prediction"
                    ],
                    dtype=np.float64,
                ),
            )

        else:
            actor_primary_metrics = {
                key: math.nan
                for key in primary_metrics
            }

            actor_secondary_metrics = {
                key: math.nan
                for key in secondary_metrics
            }

            actor_facebox_metrics = {
                key: math.nan
                for key in facebox_metrics
            }

        actor_results.append(
            {
                "actor": actor,
                "annotations": actor_annotations,
                "successful_predictions": actor_successes,
                "video_read_failures": int(
                    accumulator[
                        "video_read_failures"
                    ]
                ),
                "landmark_failures": int(
                    accumulator[
                        "landmark_failures"
                    ]
                ),
                "invalid_references": int(
                    accumulator[
                        "invalid_references"
                    ]
                ),
                "prediction_failures": int(
                    accumulator[
                        "prediction_failures"
                    ]
                ),
                "availability": (
                    actor_successes
                    / actor_annotations
                    if actor_annotations > 0
                    else math.nan
                ),
                **{
                    f"primary_{key}": value
                    for key, value in actor_primary_metrics.items()
                },
                **{
                    f"secondary_{key}": value
                    for key, value in actor_secondary_metrics.items()
                },
                **{
                    f"facebox_{key}": value
                    for key, value in actor_facebox_metrics.items()
                },
            }
        )

    per_frame_fieldnames = [
        "actor",
        "trial",
        "frame",
        "status",
        "failure_reason",
        "duplicate_candidates",
        "felt_reference",
        "felt_reference_three_pair",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "felt_reference_facebox",
        "physiotrack_openness_facebox",
        "facebox_signed_error",
        "facebox_absolute_error",
        "felt_mouth_width",
        "felt_mouth_height",
        "physiotrack_mouth_width",
        "physiotrack_mouth_height",
        "FaceScore",
        "FaceRectX",
        "FaceRectY",
        "FaceRectWidth",
        "FaceRectHeight",
    ]

    with open(
        per_frame_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                per_frame_fieldnames
            ),
        )

        writer.writeheader()
        writer.writerows(
            frame_results
        )

    per_actor_fieldnames = list(
        actor_results[0].keys()
    )

    with open(
        per_actor_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                per_actor_fieldnames
            ),
        )

        writer.writeheader()
        writer.writerows(
            actor_results
        )

    landmarks_source = source_display_path(
        FaceLandmarks
    )

    mouth_source = source_display_path(
        MouthOpenness
    )

    availability = (
        successful_predictions
        / total_annotations
    )

    summary_lines = [
        "FELT/RAVDESS Mouth Openness Evaluation",
        "",
        "Validation target: PhysioTrack FaceLandmarks + MouthOpenness",
        f"FaceLandmarks source: {landmarks_source}",
        f"MouthOpenness source: {mouth_source}",
        "FELT dataset: datasets/FELT/raw_motion_speech",
        "RAVDESS dataset: datasets/RAVDESS/Video_Speech",
        "Dataset scope: speech subset only",
        "FELT dataset integrity after evaluation: PASS",
        "RAVDESS dataset integrity after evaluation: PASS",
        "Face initialization: FELT FaceRect",
        "FELT FaceRect conversion: [x, y, x + width, y + height]",
        "Frame synchronization: exact FELT frame index, zero temporal lag",
        "Trailing RAVDESS frames without FELT annotation: ignored",
        "Duplicate-frame rule: largest FaceRect area, FaceScore tie-breaker",
        "Primary reference: d(62,66) / d(48,54)",
        "Secondary reference: mean[d(61,67), d(62,66), d(63,65)] / d(48,54)",
        "Face-box sensitivity reference: d(62,66) / FaceRectHeight",
        "Face-box sensitivity prediction: PhysioTrack mouth height in pixels / FaceRectHeight",
        "Quality exclusions: none based on FaceScore, mouth width, openness, or head pose",
        f"Device: {DEVICE}",
        f"OpenCV version: {cv2.__version__}",
        f"NumPy version: {np.__version__}",
        f"pandas version: {pandas_version}",
        f"MediaPipe version: {mediapipe_version}",
        f"Face landmarker model: {model_path.name}",
        f"Face landmarker SHA256: {model_sha256}",
        f"Actors: {len(EXPECTED_ACTORS)}",
        f"Paired speech trials: {len(trials)}",
        f"Total annotated frames: {total_annotations}",
        f"Duplicate annotation rows resolved: {duplicate_rows_resolved}",
        f"Successful predictions: {successful_predictions}",
        f"Video read failures: {video_read_failures}",
        f"Landmark failures: {landmark_failures}",
        f"Invalid references: {invalid_references}",
        f"Prediction failures: {prediction_failures}",
        f"Accounted annotated frames: {accounted_annotations}",
        f"Availability: {availability:.6f}",
        "",
        "Primary Reference Metrics",
        f"MAE: {format_metric(primary_metrics['mae'])}",
        f"RMSE: {format_metric(primary_metrics['rmse'])}",
        "Median absolute error: "
        f"{format_metric(primary_metrics['median_absolute_error'])}",
        "Std absolute error: "
        f"{format_metric(primary_metrics['std_absolute_error'])}",
        "90th percentile absolute error: "
        f"{format_metric(primary_metrics['p90_absolute_error'])}",
        "95th percentile absolute error: "
        f"{format_metric(primary_metrics['p95_absolute_error'])}",
        "Mean signed error (prediction - reference): "
        f"{format_metric(primary_metrics['mean_signed_error'])}",
        f"Pearson r: {format_metric(primary_metrics['pearson_r'])}",
        f"Spearman rho: {format_metric(primary_metrics['spearman_rho'])}",
        f"Lin CCC: {format_metric(primary_metrics['ccc'])}",
        "",
        "Secondary Three-Pair Reference Metrics",
        f"MAE: {format_metric(secondary_metrics['mae'])}",
        f"RMSE: {format_metric(secondary_metrics['rmse'])}",
        "Median absolute error: "
        f"{format_metric(secondary_metrics['median_absolute_error'])}",
        "Std absolute error: "
        f"{format_metric(secondary_metrics['std_absolute_error'])}",
        "90th percentile absolute error: "
        f"{format_metric(secondary_metrics['p90_absolute_error'])}",
        "95th percentile absolute error: "
        f"{format_metric(secondary_metrics['p95_absolute_error'])}",
        "Mean signed error (prediction - reference): "
        f"{format_metric(secondary_metrics['mean_signed_error'])}",
        f"Pearson r: {format_metric(secondary_metrics['pearson_r'])}",
        f"Spearman rho: {format_metric(secondary_metrics['spearman_rho'])}",
        f"Lin CCC: {format_metric(secondary_metrics['ccc'])}",
        "",
        "Face-Box-Normalized Sensitivity Metrics",
        f"MAE: {format_metric(facebox_metrics['mae'])}",
        f"RMSE: {format_metric(facebox_metrics['rmse'])}",
        "Median absolute error: "
        f"{format_metric(facebox_metrics['median_absolute_error'])}",
        "Std absolute error: "
        f"{format_metric(facebox_metrics['std_absolute_error'])}",
        "90th percentile absolute error: "
        f"{format_metric(facebox_metrics['p90_absolute_error'])}",
        "95th percentile absolute error: "
        f"{format_metric(facebox_metrics['p95_absolute_error'])}",
        "Mean signed error (prediction - reference): "
        f"{format_metric(facebox_metrics['mean_signed_error'])}",
        f"Pearson r: {format_metric(facebox_metrics['pearson_r'])}",
        f"Spearman rho: {format_metric(facebox_metrics['spearman_rho'])}",
        f"Lin CCC: {format_metric(facebox_metrics['ccc'])}",
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

    print()
    print(
        "Validating staged quantitative outputs..."
    )

    validate_staged_quantitative_outputs(
        per_frame_csv_path,
        per_actor_csv_path,
        summary_path,
    )

    print(
        "Staged quantitative output validation: PASS"
    )

    commit_owned_files(
        [
            (
                per_frame_csv_path,
                final_per_frame_csv_path,
            ),
            (
                per_actor_csv_path,
                final_per_actor_csv_path,
            ),
            (
                summary_path,
                final_summary_path,
            ),
        ],
        staging_dir,
    )

    staging_context.cleanup()

    print(
        "Committed final quantitative outputs."
    )

    print()
    print(
        "=== Final Summary ==="
    )

    for line in summary_lines:
        print(
            line
        )

    print()
    print(
        f"Saved: {final_summary_path}"
    )

    print(
        f"Saved: {final_per_actor_csv_path}"
    )

    print(
        f"Saved: {final_per_frame_csv_path}"
    )


if __name__ == "__main__":
    main()
