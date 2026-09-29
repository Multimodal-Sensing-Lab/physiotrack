from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_CSV = RESULTS_DIR / "robustness_results.csv"
SUMMARY_JSON = RESULTS_DIR / "robustness_summary.json"

OUTPUT_TABLE_CSV = RESULTS_DIR / "robustness_metrics.csv"
OUTPUT_TABLE_MD = RESULTS_DIR / "robustness_metrics.md"

FIGURE_AVAILABILITY = FIGURES_DIR / "robustness_availability.png"
FIGURE_CONFIDENCE = FIGURES_DIR / "robustness_detector_confidence.png"
FIGURE_SHARPNESS = FIGURES_DIR / "robustness_quality_sharpness.png"

EXPECTED_CASES = [
    "baseline",
    "dim_lighting",
    "overexposed_lighting",
    "gaussian_blur",
    "motion_blur",
    "small_face",
    "partial_occlusion",
]

MODULE_FIELDS = [
    ("Detection", "detection_available"),
    ("Landmarks", "landmarks_available"),
    ("Quality", "quality_available"),
    ("Head Pose", "head_pose_available"),
    ("Eyes", "eyes_available"),
    ("Gaze", "gaze_available"),
    ("Gaze Estimation", "gaze_estimation_available"),
    ("Mouth", "mouth_available"),
    ("Emotion", "emotion_available"),
    ("Regions", "regions_available"),
]

OUTPUT_FIELDS = [
    "case_id",
    "condition",
    "status",
    "detected_faces",
    "available_static_modules",
    "total_static_modules",
    "detector_confidence",
    "quality_brightness",
    "quality_sharpness",
    "quality_face_area_ratio",
    "head_pose_yaw_abs_delta_vs_baseline",
    "head_pose_pitch_abs_delta_vs_baseline",
    "gaze_estimation_yaw_abs_delta_vs_baseline",
    "gaze_estimation_pitch_abs_delta_vs_baseline",
    "eye_mean_openness_abs_delta_vs_baseline",
    "mouth_openness_abs_delta_vs_baseline",
    "emotion_label",
    "emotion_confidence",
    "emotion_confidence_abs_delta_vs_baseline",
    "regions_skin_fraction",
    "regions_skin_fraction_abs_delta_vs_baseline",
    "all_available_numeric_values_finite",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, np.integer)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no", ""}:
            return False

    raise ValueError(
        f"Cannot interpret boolean value: {value!r}"
    )


def finite_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all()


def read_and_validate() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            f"Accepted robustness result CSV not found: {RESULTS_CSV}"
        )

    if not SUMMARY_JSON.is_file():
        raise FileNotFoundError(
            f"Accepted robustness summary JSON not found: {SUMMARY_JSON}"
        )

    dataframe = pd.read_csv(
        RESULTS_CSV
    )

    with SUMMARY_JSON.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if len(dataframe) != len(EXPECTED_CASES):
        raise RuntimeError(
            "Robustness result row count does not match the accepted protocol"
        )

    if dataframe[
        "case_id"
    ].tolist() != EXPECTED_CASES:
        raise RuntimeError(
            "Robustness result case ordering does not match the accepted protocol"
        )

    if dataframe[
        "case_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate robustness case_id values found"
        )

    expected_statuses = {
        "DETECTED",
        "NO_FACE",
        "EXECUTION_FAILED",
        "INVALID_NUMERIC_OUTPUT",
    }

    if not set(
        dataframe[
            "status"
        ].astype(str)
    ).issubset(
        expected_statuses
    ):
        raise RuntimeError(
            "Unexpected robustness result status found"
        )

    for label, field in MODULE_FIELDS:
        if field not in dataframe.columns:
            raise RuntimeError(
                f"Required module availability field missing: {field}"
            )

        dataframe[
            field
        ] = dataframe[
            field
        ].map(
            parse_bool
        )

    dataframe[
        "all_available_numeric_values_finite"
    ] = dataframe[
        "all_available_numeric_values_finite"
    ].map(
        parse_bool
    )

    detected = dataframe[
        dataframe[
            "status"
        ] == "DETECTED"
    ]

    if not detected.empty:
        numeric_fields = [
            "detector_confidence",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
            "box_width",
            "box_height",
            "box_area",
            "quality_brightness",
            "quality_sharpness",
            "quality_face_area_ratio",
            "head_pose_pitch",
            "head_pose_yaw",
            "head_pose_roll",
            "eye_mean_openness",
            "gaze_mean_iris_x",
            "gaze_mean_iris_y",
            "gaze_estimation_pitch",
            "gaze_estimation_yaw",
            "gaze_association_iou",
            "mouth_openness",
            "emotion_confidence",
            "regions_skin_fraction",
            "regions_association_iou",
        ]

        for field in numeric_fields:
            if field not in dataframe.columns:
                raise RuntimeError(
                    f"Required robustness numeric field missing: {field}"
                )

            if not finite_series(
                detected[
                    field
                ]
            ):
                raise RuntimeError(
                    f"Non-finite values found in detected robustness field: {field}"
                )

        if (
            detected[
                "box_width"
            ] <= 0
        ).any():
            raise RuntimeError(
                "Detected robustness result contains non-positive box width"
            )

        if (
            detected[
                "box_height"
            ] <= 0
        ).any():
            raise RuntimeError(
                "Detected robustness result contains non-positive box height"
            )

        if not detected[
            "all_available_numeric_values_finite"
        ].all():
            raise RuntimeError(
                "Detected robustness result contains an invalid numeric-output flag"
            )

    baseline = dataframe[
        dataframe[
            "case_id"
        ] == "baseline"
    ]

    if len(baseline) != 1:
        raise RuntimeError(
            "Exactly one baseline robustness row is required"
        )

    if baseline.iloc[0][
        "status"
    ] != "DETECTED":
        raise RuntimeError(
            "Baseline robustness row must be DETECTED"
        )

    recomputed_available = dataframe.apply(
        lambda row: sum(
            bool(
                row[field]
            )
            for _, field in MODULE_FIELDS
        ),
        axis=1,
    )

    summary_cases = summary.get(
        "cases",
        []
    )

    if len(summary_cases) != len(
        dataframe
    ):
        raise RuntimeError(
            "Robustness summary case count does not match detailed results"
        )

    summary_by_case = {
        item[
            "case_id"
        ]: item
        for item in summary_cases
    }

    for index, row in dataframe.iterrows():
        case_id = row[
            "case_id"
        ]

        if case_id not in summary_by_case:
            raise RuntimeError(
                f"Case missing from robustness summary: {case_id}"
            )

        summary_case = summary_by_case[
            case_id
        ]

        if summary_case[
            "status"
        ] != row[
            "status"
        ]:
            raise RuntimeError(
                f"Status mismatch between robustness CSV and summary for {case_id}"
            )

        if int(
            summary_case[
                "detected_faces"
            ]
        ) != int(
            row[
                "detected_faces"
            ]
        ):
            raise RuntimeError(
                f"Detected-face mismatch for robustness case: {case_id}"
            )

        if int(
            summary_case[
                "available_static_modules"
            ]
        ) != int(
            recomputed_available.iloc[
                index
            ]
        ):
            raise RuntimeError(
                f"Module availability mismatch for robustness case: {case_id}"
            )

    execution = summary.get(
        "execution",
        {}
    )

    if int(
        execution.get(
            "cases",
            -1,
        )
    ) != len(
        dataframe
    ):
        raise RuntimeError(
            "Robustness summary execution case count mismatch"
        )

    if int(
        execution.get(
            "detected_cases",
            -1,
        )
    ) != int(
        (
            dataframe[
                "status"
            ] == "DETECTED"
        ).sum()
    ):
        raise RuntimeError(
            "Robustness summary detected-case count mismatch"
        )

    if int(
        execution.get(
            "no_face_cases",
            -1,
        )
    ) != int(
        (
            dataframe[
                "status"
            ] == "NO_FACE"
        ).sum()
    ):
        raise RuntimeError(
            "Robustness summary NO_FACE count mismatch"
        )

    if int(
        execution.get(
            "execution_failed_cases",
            -1,
        )
    ) != int(
        (
            dataframe[
                "status"
            ] == "EXECUTION_FAILED"
        ).sum()
    ):
        raise RuntimeError(
            "Robustness summary execution-failure count mismatch"
        )

    if int(
        execution.get(
            "invalid_numeric_cases",
            -1,
        )
    ) != int(
        (
            dataframe[
                "status"
            ] == "INVALID_NUMERIC_OUTPUT"
        ).sum()
    ):
        raise RuntimeError(
            "Robustness summary invalid-numeric count mismatch"
        )

    if not bool(
        execution.get(
            "input_read_only_verified",
            False,
        )
    ):
        raise RuntimeError(
            "Robustness summary does not confirm read-only input verification"
        )

    return (
        dataframe,
        summary,
    )


def build_metrics_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame()

    table[
        "case_id"
    ] = dataframe[
        "case_id"
    ]
    table[
        "condition"
    ] = dataframe[
        "condition"
    ]
    table[
        "status"
    ] = dataframe[
        "status"
    ]
    table[
        "detected_faces"
    ] = dataframe[
        "detected_faces"
    ].astype(int)

    table[
        "available_static_modules"
    ] = dataframe.apply(
        lambda row: sum(
            bool(
                row[field]
            )
            for _, field in MODULE_FIELDS
        ),
        axis=1,
    )
    table[
        "total_static_modules"
    ] = len(
        MODULE_FIELDS
    )

    for field in OUTPUT_FIELDS:
        if field in {
            "case_id",
            "condition",
            "status",
            "detected_faces",
            "available_static_modules",
            "total_static_modules",
        }:
            continue

        if field not in dataframe.columns:
            raise RuntimeError(
                f"Required output field missing from robustness results: {field}"
            )

        table[
            field
        ] = dataframe[
            field
        ]

    return table[
        OUTPUT_FIELDS
    ]


def write_markdown_table(
    path: Path,
    dataframe: pd.DataFrame,
) -> None:
    display_columns = [
        "condition",
        "status",
        "available_static_modules",
        "detector_confidence",
        "quality_sharpness",
        "quality_face_area_ratio",
        "head_pose_yaw_abs_delta_vs_baseline",
        "gaze_estimation_yaw_abs_delta_vs_baseline",
        "mouth_openness_abs_delta_vs_baseline",
        "emotion_label",
    ]

    markdown = dataframe[
        display_columns
    ].copy()

    rename = {
        "condition":
            "Condition",
        "status":
            "Status",
        "available_static_modules":
            "Available modules",
        "detector_confidence":
            "Detector confidence",
        "quality_sharpness":
            "Sharpness",
        "quality_face_area_ratio":
            "Face area ratio",
        "head_pose_yaw_abs_delta_vs_baseline":
            "|Δ Head yaw|",
        "gaze_estimation_yaw_abs_delta_vs_baseline":
            "|Δ Gaze yaw|",
        "mouth_openness_abs_delta_vs_baseline":
            "|Δ Mouth openness|",
        "emotion_label":
            "Emotion",
    }

    markdown = markdown.rename(
        columns=rename
    )

    for column in markdown.columns:
        if pd.api.types.is_float_dtype(
            markdown[
                column
            ]
        ):
            markdown[
                column
            ] = markdown[
                column
            ].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{value:.6f}"
                )
            )

    path.write_text(
        markdown.to_markdown(
            index=False
        )
        + "\n",
        encoding="utf-8",
    )


def plot_availability(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    matrix = np.asarray(
        [
            [
                1.0
                if bool(
                    row[field]
                )
                else 0.0
                for _, field in MODULE_FIELDS
            ]
            for _, row in dataframe.iterrows()
        ],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(11.0, 4.6)
    )

    image = axis.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(
            len(
                MODULE_FIELDS
            )
        )
    )
    axis.set_xticklabels(
        [
            label
            for label, _ in MODULE_FIELDS
        ],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        np.arange(
            len(
                dataframe
            )
        )
    )
    axis.set_yticklabels(
        dataframe[
            "condition"
        ].tolist()
    )
    axis.set_title(
        "Static Component Availability Across Controlled Robustness Conditions"
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                "1"
                if matrix[
                    row_index,
                    column_index,
                ] == 1.0
                else "0",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_ticks(
        [
            0.0,
            1.0,
        ]
    )
    colorbar.set_ticklabels(
        [
            "Unavailable",
            "Available",
        ]
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def plot_detector_confidence(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9.4, 4.8)
    )

    x = np.arange(
        len(
            dataframe
        )
    )
    values = dataframe[
        "detector_confidence"
    ].to_numpy(
        dtype=float
    )

    axis.bar(
        x,
        values,
    )
    axis.axhline(
        float(
            dataframe.loc[
                dataframe[
                    "case_id"
                ] == "baseline",
                "detector_confidence",
            ].iloc[0]
        ),
        linestyle="--",
        linewidth=1.0,
        label="Baseline",
    )
    axis.set_xticks(
        x
    )
    axis.set_xticklabels(
        dataframe[
            "condition"
        ].tolist(),
        rotation=30,
        ha="right",
    )
    axis.set_ylabel(
        "Detector confidence"
    )
    axis.set_ylim(
        0.0,
        1.0,
    )
    axis.set_title(
        "Face Detection Confidence Under Controlled Perturbations"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def plot_quality_sharpness(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9.4, 4.8)
    )

    x = np.arange(
        len(
            dataframe
        )
    )
    values = dataframe[
        "quality_sharpness"
    ].to_numpy(
        dtype=float
    )

    axis.bar(
        x,
        values,
    )
    axis.axhline(
        float(
            dataframe.loc[
                dataframe[
                    "case_id"
                ] == "baseline",
                "quality_sharpness",
            ].iloc[0]
        ),
        linestyle="--",
        linewidth=1.0,
        label="Baseline",
    )
    axis.set_xticks(
        x
    )
    axis.set_xticklabels(
        dataframe[
            "condition"
        ].tolist(),
        rotation=30,
        ha="right",
    )
    axis.set_ylabel(
        "Face quality sharpness"
    )
    axis.set_title(
        "Face Quality Sharpness Under Controlled Perturbations"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def validate_staged_outputs(
    staging_results_dir: Path,
) -> None:
    expected = [
        staging_results_dir
        / OUTPUT_TABLE_CSV.name,
        staging_results_dir
        / OUTPUT_TABLE_MD.name,
        staging_results_dir
        / "figures"
        / FIGURE_AVAILABILITY.name,
        staging_results_dir
        / "figures"
        / FIGURE_CONFIDENCE.name,
        staging_results_dir
        / "figures"
        / FIGURE_SHARPNESS.name,
    ]

    for path in expected:
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected staged robustness plot output missing: {path}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged robustness plot output is empty: {path}"
            )


def promote_outputs(
    staging_results_dir: Path,
) -> None:
    final_files = [
        OUTPUT_TABLE_CSV,
        OUTPUT_TABLE_MD,
        FIGURE_AVAILABILITY,
        FIGURE_CONFIDENCE,
        FIGURE_SHARPNESS,
    ]

    staged_files = [
        staging_results_dir
        / OUTPUT_TABLE_CSV.name,
        staging_results_dir
        / OUTPUT_TABLE_MD.name,
        staging_results_dir
        / "figures"
        / FIGURE_AVAILABILITY.name,
        staging_results_dir
        / "figures"
        / FIGURE_CONFIDENCE.name,
        staging_results_dir
        / "figures"
        / FIGURE_SHARPNESS.name,
    ]

    backups: dict[Path, Path] = {}
    promoted: list[Path] = []

    backup_dir = staging_results_dir / ".backups"
    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for final_path, staged_path in zip(
            final_files,
            staged_files,
        ):
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )
                os.replace(
                    final_path,
                    backup_path,
                )
                backups[
                    final_path
                ] = backup_path

            os.replace(
                staged_path,
                final_path,
            )
            promoted.append(
                final_path
            )

    except Exception:
        for final_path in promoted:
            if final_path.exists():
                final_path.unlink()

        for final_path, backup_path in backups.items():
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main() -> None:
    dataframe, summary = read_and_validate()

    metrics = build_metrics_table(
        dataframe
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".robustness_plot_",
            dir=RESULTS_DIR,
        )
    )
    staging_figures_dir = (
        staging_dir
        / "figures"
    )
    staging_figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        metrics.to_csv(
            staging_dir
            / OUTPUT_TABLE_CSV.name,
            index=False,
        )

        write_markdown_table(
            staging_dir
            / OUTPUT_TABLE_MD.name,
            metrics,
        )

        plot_availability(
            dataframe,
            staging_figures_dir
            / FIGURE_AVAILABILITY.name,
        )
        plot_detector_confidence(
            dataframe,
            staging_figures_dir
            / FIGURE_CONFIDENCE.name,
        )
        plot_quality_sharpness(
            dataframe,
            staging_figures_dir
            / FIGURE_SHARPNESS.name,
        )

        validate_staged_outputs(
            staging_dir
        )
        promote_outputs(
            staging_dir
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    execution = summary[
        "execution"
    ]

    print(
        "Controlled Robustness Plot/Table Verification"
    )
    print(
        "=" * 45
    )
    print(
        f"Cases: {len(dataframe)}"
    )
    print(
        "Detected cases: "
        f"{execution['detected_cases']}"
    )
    print(
        "NO_FACE cases: "
        f"{execution['no_face_cases']}"
    )
    print(
        "Execution failures: "
        f"{execution['execution_failed_cases']}"
    )
    print(
        "Invalid numerical cases: "
        f"{execution['invalid_numeric_cases']}"
    )
    print(
        "Detailed-result integrity checks: PASS"
    )
    print(
        "Evaluator-summary consistency: PASS"
    )
    print(
        f"Metrics CSV: {OUTPUT_TABLE_CSV}"
    )
    print(
        f"Metrics Markdown: {OUTPUT_TABLE_MD}"
    )
    print(
        f"Figures: {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()
