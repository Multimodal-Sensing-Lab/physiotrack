from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

PER_PERSON_CSV_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_per_person.csv"
)

PER_SAMPLE_CSV_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_per_sample.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_summary.txt"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "mpiifacegaze_ethxgaze_per_person_mae.png"
)

ERROR_DISTRIBUTION_FIGURE_PATH = (
    FIGURES_DIR
    / "mpiifacegaze_ethxgaze_angular_error_distribution.png"
)

TABLE_CSV_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_thesis_table.csv"
)

TABLE_MD_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_thesis_table.md"
)

AUDIT_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_audit.txt"
)

EXPECTED_PARTICIPANTS = [
    f"p{index:02d}"
    for index in range(15)
]

EXPECTED_TOTAL_ANNOTATIONS = 37667

COUNT_COLUMNS = [
    "annotations",
    "successful_predictions",
    "image_read_failures",
    "face_detection_failures",
    "prediction_failures",
    "invalid_annotation_rows",
]

ERROR_COLUMNS = [
    "mean_angular_error_deg",
    "median_angular_error_deg",
    "std_angular_error_deg",
]


def read_summary(
    summary_path: Path,
) -> dict[str, str]:
    metrics = {}

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = (
                raw_line
                .strip()
            )

            if (
                not line
                or ":" not in line
            ):
                continue

            key, value = (
                line.split(
                    ":",
                    1,
                )
            )

            metrics[
                key.strip()
            ] = value.strip()

    return metrics


def summary_int(
    summary: dict[str, str],
    key: str,
) -> int:
    if key not in summary:
        raise ValueError(
            f"Missing summary metric: {key}"
        )

    return int(
        summary[
            key
        ]
    )


def summary_float_deg(
    summary: dict[str, str],
    key: str,
) -> float:
    if key not in summary:
        raise ValueError(
            f"Missing summary metric: {key}"
        )

    return float(
        summary[
            key
        ]
        .replace(
            "deg",
            "",
        )
        .strip()
    )


def assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = 5e-6,
) -> None:
    if (
        not math.isfinite(
            actual
        )
        or not math.isfinite(
            expected
        )
        or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    ):
        raise AssertionError(
            f"{label} mismatch: "
            f"actual={actual}, expected={expected}"
        )


def create_error_distribution_figure(
    errors: np.ndarray,
    output_path: Path,
) -> None:
    """Create an empirical CDF of successful-sample angular errors."""
    if (
        errors.ndim != 1
        or len(errors) == 0
    ):
        raise ValueError(
            "Angular-error distribution requires a non-empty 1D array."
        )

    if not np.all(
        np.isfinite(
            errors
        )
    ):
        raise ValueError(
            "Angular-error distribution contains non-finite values."
        )

    if np.any(
        errors < 0.0
    ):
        raise ValueError(
            "Angular-error distribution contains negative values."
        )

    sorted_errors = np.sort(
        errors.astype(
            float
        )
    )

    cumulative_percent = (
        np.arange(
            1,
            len(sorted_errors) + 1,
            dtype=float,
        )
        / len(sorted_errors)
        * 100.0
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.plot(
        sorted_errors,
        cumulative_percent,
        linewidth=2.0,
    )

    axis.set_xlabel(
        "Angular Error (degrees)"
    )

    axis.set_ylabel(
        "Cumulative Successful Samples (%)"
    )

    axis.set_title(
        "MPIIFaceGaze Angular-Error Distribution"
    )

    axis.set_xlim(
        left=0.0
    )

    axis.set_ylim(
        0.0,
        100.0,
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_staging_directory() -> tuple[Path, Path, Path]:
    """Create staging before plot, table, and audit generation."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".mpiifacegaze_gaze_plot_",
            dir=RESULTS_DIR,
        )
    )

    staged_results_dir = (
        staging_dir
        / "results"
    )

    staged_figures_dir = (
        staged_results_dir
        / "figures"
    )

    staged_figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        staging_dir,
        staged_results_dir,
        staged_figures_dir,
    )


def validate_staged_outputs(
    staged_table_csv: Path,
    staged_table_md: Path,
    staged_audit: Path,
    staged_figure: Path,
    staged_distribution_figure: Path,
    expected_table: pd.DataFrame,
) -> None:
    """Verify all plot-owned staged outputs before final replacement."""
    required_paths = [
        staged_table_csv,
        staged_table_md,
        staged_audit,
        staged_figure,
        staged_distribution_figure,
    ]

    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(
                f"Missing staged plot output: {path.name}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged plot output is empty: {path.name}"
            )

    stored_table = pd.read_csv(
        staged_table_csv
    )

    expected_for_csv = expected_table.copy()

    if list(
        stored_table.columns
    ) != list(
        expected_for_csv.columns
    ):
        raise RuntimeError(
            "Staged thesis-table CSV schema is invalid."
        )

    if len(
        stored_table
    ) != len(
        expected_for_csv
    ):
        raise RuntimeError(
            "Staged thesis-table CSV row count is invalid."
        )

    for column in expected_for_csv.columns:
        stored_column = stored_table[
            column
        ]

        expected_column = expected_for_csv[
            column
        ]

        if pd.api.types.is_numeric_dtype(
            expected_column
        ):
            if not np.allclose(
                stored_column.astype(float).to_numpy(),
                expected_column.astype(float).to_numpy(),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise RuntimeError(
                    f"Staged thesis-table values differ in column {column}."
                )
        else:
            if (
                stored_column.astype(str).tolist()
                != expected_column.astype(str).tolist()
            ):
                raise RuntimeError(
                    f"Staged thesis-table values differ in column {column}."
                )

    with open(
        staged_audit,
        "r",
        encoding="utf-8",
    ) as file:
        audit_text = file.read()

    if "Audit status: PASS" not in audit_text:
        raise RuntimeError(
            "Staged result audit did not report PASS."
        )

    for figure_path in [
        staged_figure,
        staged_distribution_figure,
    ]:
        image = plt.imread(
            figure_path
        )

        if (
            image.ndim not in {
                2,
                3,
            }
            or image.size == 0
        ):
            raise RuntimeError(
                f"Staged figure is not readable: {figure_path.name}"
            )


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Atomically install one validated plot-owned file."""
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
    output_pairs: list[tuple[Path, Path]],
    staging_dir: Path,
) -> None:
    """Replace only plot-owned outputs with rollback protection."""
    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backups = []

    for _, final_path in output_pairs:
        if final_path.is_file():
            backup_path = (
                backup_dir
                / final_path.name
            )

            shutil.copy2(
                final_path,
                backup_path,
            )

            backups.append(
                (
                    backup_path,
                    final_path,
                )
            )

    installed_paths = []

    try:
        for staged_path, final_path in output_pairs:
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

        for backup_path, final_path in backups:
            atomic_copy_file(
                backup_path,
                final_path,
            )

        raise


def main() -> None:
    required_files = [
        PER_PERSON_CSV_PATH,
        PER_SAMPLE_CSV_PATH,
        SUMMARY_PATH,
    ]

    for path in (
        required_files
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required result file: {path.name}"
            )

    per_person = pd.read_csv(
        PER_PERSON_CSV_PATH
    )

    per_sample = pd.read_csv(
        PER_SAMPLE_CSV_PATH
    )

    summary = read_summary(
        SUMMARY_PATH
    )

    required_person_columns = [
        "participant",
        *COUNT_COLUMNS,
        *ERROR_COLUMNS,
    ]

    missing_person_columns = [
        column
        for column in required_person_columns
        if column not in per_person.columns
    ]

    if missing_person_columns:
        raise ValueError(
            "Missing per-person columns: "
            + ", ".join(
                missing_person_columns
            )
        )

    required_sample_columns = [
        "participant",
        "image_relative_path",
        "status",
        "angular_error_deg",
    ]

    missing_sample_columns = [
        column
        for column in required_sample_columns
        if column not in per_sample.columns
    ]

    if missing_sample_columns:
        raise ValueError(
            "Missing per-sample columns: "
            + ", ".join(
                missing_sample_columns
            )
        )

    participants = (
        per_person[
            "participant"
        ]
        .astype(str)
        .tolist()
    )

    if (
        participants
        != EXPECTED_PARTICIPANTS
    ):
        raise AssertionError(
            "Unexpected participant order or set: "
            f"{participants}"
        )

    if (
        len(per_sample)
        != EXPECTED_TOTAL_ANNOTATIONS
    ):
        raise AssertionError(
            "Unexpected per-sample row count: "
            f"{len(per_sample)}"
        )

    if (
        int(
            per_person[
                "annotations"
            ].sum()
        )
        != EXPECTED_TOTAL_ANNOTATIONS
    ):
        raise AssertionError(
            "Per-person annotation total does not equal "
            f"{EXPECTED_TOTAL_ANNOTATIONS}."
        )

    summary_counts = {
        "annotations": summary_int(
            summary,
            "Total annotations",
        ),
        "successful_predictions": summary_int(
            summary,
            "Successful predictions",
        ),
        "image_read_failures": summary_int(
            summary,
            "Image read failures",
        ),
        "face_detection_failures": summary_int(
            summary,
            "Face detection failures",
        ),
        "prediction_failures": summary_int(
            summary,
            "Prediction failures",
        ),
        "invalid_annotation_rows": summary_int(
            summary,
            "Invalid annotation rows",
        ),
    }

    for column in (
        COUNT_COLUMNS
    ):
        csv_total = int(
            per_person[
                column
            ].sum()
        )

        if (
            csv_total
            != summary_counts[
                column
            ]
        ):
            raise AssertionError(
                f"{column} total mismatch: "
                f"CSV={csv_total}, "
                f"summary={summary_counts[column]}"
            )

    accounted = (
        summary_counts[
            "successful_predictions"
        ]
        + summary_counts[
            "image_read_failures"
        ]
        + summary_counts[
            "face_detection_failures"
        ]
        + summary_counts[
            "prediction_failures"
        ]
        + summary_counts[
            "invalid_annotation_rows"
        ]
    )

    if (
        accounted
        != summary_counts[
            "annotations"
        ]
    ):
        raise AssertionError(
            "Summary failure accounting invariant failed."
        )

    expected_status_counts = {
        "success": (
            summary_counts[
                "successful_predictions"
            ]
        ),
        "image_read_failure": (
            summary_counts[
                "image_read_failures"
            ]
        ),
        "face_detection_failure": (
            summary_counts[
                "face_detection_failures"
            ]
        ),
        "prediction_failure": (
            summary_counts[
                "prediction_failures"
            ]
        ),
        "invalid_annotation": (
            summary_counts[
                "invalid_annotation_rows"
            ]
        ),
    }

    actual_status_counts = (
        per_sample[
            "status"
        ]
        .value_counts()
        .to_dict()
    )

    for (
        status,
        expected_count,
    ) in expected_status_counts.items():
        actual_count = int(
            actual_status_counts.get(
                status,
                0,
            )
        )

        if (
            actual_count
            != expected_count
        ):
            raise AssertionError(
                f"Status count mismatch for {status}: "
                f"sample CSV={actual_count}, "
                f"summary={expected_count}"
            )

    unexpected_statuses = (
        set(
            actual_status_counts
        )
        - set(
            expected_status_counts
        )
    )

    if unexpected_statuses:
        raise AssertionError(
            "Unexpected sample statuses: "
            + ", ".join(
                sorted(
                    unexpected_statuses
                )
            )
        )

    successful_samples = (
        per_sample[
            per_sample[
                "status"
            ]
            == "success"
        ]
        .copy()
    )

    errors = (
        successful_samples[
            "angular_error_deg"
        ]
        .astype(float)
        .to_numpy()
    )

    if (
        len(errors)
        != summary_counts[
            "successful_predictions"
        ]
    ):
        raise AssertionError(
            "Successful sample count does not match summary."
        )

    if not np.all(
        np.isfinite(
            errors
        )
    ):
        raise AssertionError(
            "Successful samples contain non-finite angular errors."
        )

    recomputed_summary = {
        "Mean angular error": float(
            np.mean(
                errors
            )
        ),
        "Median angular error": float(
            np.median(
                errors
            )
        ),
        "Std angular error": float(
            np.std(
                errors
            )
        ),
        "Minimum angular error": float(
            np.min(
                errors
            )
        ),
        "Maximum angular error": float(
            np.max(
                errors
            )
        ),
        "90th percentile angular error": float(
            np.percentile(
                errors,
                90,
            )
        ),
        "95th percentile angular error": float(
            np.percentile(
                errors,
                95,
            )
        ),
    }

    for (
        key,
        recomputed_value,
    ) in recomputed_summary.items():
        summary_value = (
            summary_float_deg(
                summary,
                key,
            )
        )

        assert_close(
            recomputed_value,
            summary_value,
            key,
        )

    for participant in (
        EXPECTED_PARTICIPANTS
    ):
        person_row = (
            per_person[
                per_person[
                    "participant"
                ]
                == participant
            ]
            .iloc[0]
        )

        person_samples = (
            per_sample[
                per_sample[
                    "participant"
                ]
                == participant
            ]
        )

        if (
            len(person_samples)
            != int(
                person_row[
                    "annotations"
                ]
            )
        ):
            raise AssertionError(
                f"{participant}: annotation count mismatch."
            )

        status_to_column = {
            "success": "successful_predictions",
            "image_read_failure": "image_read_failures",
            "face_detection_failure": "face_detection_failures",
            "prediction_failure": "prediction_failures",
            "invalid_annotation": "invalid_annotation_rows",
        }

        for (
            status,
            column,
        ) in status_to_column.items():
            sample_count = int(
                (
                    person_samples[
                        "status"
                    ]
                    == status
                ).sum()
            )

            csv_count = int(
                person_row[
                    column
                ]
            )

            if (
                sample_count
                != csv_count
            ):
                raise AssertionError(
                    f"{participant}: {column} mismatch."
                )

        person_success = (
            person_samples[
                person_samples[
                    "status"
                ]
                == "success"
            ][
                "angular_error_deg"
            ]
            .astype(float)
            .to_numpy()
        )

        if len(
            person_success
        ):
            recomputed_person = {
                "mean_angular_error_deg": float(
                    np.mean(
                        person_success
                    )
                ),
                "median_angular_error_deg": float(
                    np.median(
                        person_success
                    )
                ),
                "std_angular_error_deg": float(
                    np.std(
                        person_success
                    )
                ),
            }

            for (
                column,
                value,
            ) in recomputed_person.items():
                assert_close(
                    value,
                    float(
                        person_row[
                            column
                        ]
                    ),
                    f"{participant} {column}",
                    tolerance=1e-9,
                )

    (
        staging_dir,
        staged_results_dir,
        staged_figures_dir,
    ) = create_staging_directory()


    staged_figure_path = (
        staged_figures_dir
        / FIGURE_PATH.name
    )

    staged_distribution_figure_path = (
        staged_figures_dir
        / ERROR_DISTRIBUTION_FIGURE_PATH.name
    )

    staged_table_csv_path = (
        staged_results_dir
        / TABLE_CSV_PATH.name
    )

    staged_table_md_path = (
        staged_results_dir
        / TABLE_MD_PATH.name
    )

    staged_audit_path = (
        staged_results_dir
        / AUDIT_PATH.name
    )

    print(
        f"Staging directory: {staging_dir}"
    )

    try:
        mean_errors = (
            per_person[
                "mean_angular_error_deg"
            ]
            .astype(float)
        )

        overall_mean = (
            recomputed_summary[
                "Mean angular error"
            ]
        )

        plt.figure(
            figsize=(
                12,
                6,
            )
        )

        plt.bar(
            per_person[
                "participant"
            ],
            mean_errors,
        )

        plt.axhline(
            overall_mean,
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Overall mean = "
                f"{overall_mean:.2f}°"
            ),
        )

        plt.xlabel(
            "Participant"
        )

        plt.ylabel(
            "Mean Angular Error (degrees)"
        )

        plt.title(
            "ETH-XGaze Cross-Dataset Evaluation "
            "on MPIIFaceGaze"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            staged_figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        create_error_distribution_figure(
            errors,
            staged_distribution_figure_path,
        )

        table = per_person[
            [
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
            ]
        ].copy()

        table.columns = [
            "Participant",
            "Annotations",
            "Successful Predictions",
            "Image Read Failures",
            "Face Detection Failures",
            "Prediction Failures",
            "Invalid Annotation Rows",
            "Mean Angular Error (deg)",
            "Median Angular Error (deg)",
            "Std Angular Error (deg)",
        ]

        numeric_error_columns = [
            "Mean Angular Error (deg)",
            "Median Angular Error (deg)",
            "Std Angular Error (deg)",
        ]

        table[
            numeric_error_columns
        ] = (
            table[
                numeric_error_columns
            ]
            .round(
                4
            )
        )

        overall_row = pd.DataFrame(
            [
                {
                    "Participant": "Overall",
                    "Annotations": (
                        summary_counts[
                            "annotations"
                        ]
                    ),
                    "Successful Predictions": (
                        summary_counts[
                            "successful_predictions"
                        ]
                    ),
                    "Image Read Failures": (
                        summary_counts[
                            "image_read_failures"
                        ]
                    ),
                    "Face Detection Failures": (
                        summary_counts[
                            "face_detection_failures"
                        ]
                    ),
                    "Prediction Failures": (
                        summary_counts[
                            "prediction_failures"
                        ]
                    ),
                    "Invalid Annotation Rows": (
                        summary_counts[
                            "invalid_annotation_rows"
                        ]
                    ),
                    "Mean Angular Error (deg)": round(
                        recomputed_summary[
                            "Mean angular error"
                        ],
                        4,
                    ),
                    "Median Angular Error (deg)": round(
                        recomputed_summary[
                            "Median angular error"
                        ],
                        4,
                    ),
                    "Std Angular Error (deg)": round(
                        recomputed_summary[
                            "Std angular error"
                        ],
                        4,
                    ),
                }
            ]
        )

        table = pd.concat(
            [
                table,
                overall_row,
            ],
            ignore_index=True,
        )

        table.to_csv(
            staged_table_csv_path,
            index=False,
        )

        table.to_markdown(
            staged_table_md_path,
            index=False,
        )

        audit_lines = [
            "MPIIFaceGaze Gaze Estimation Result Audit",
            "",
            "Audit status: PASS",
            f"Participants checked: {len(EXPECTED_PARTICIPANTS)}",
            f"Per-sample rows checked: {len(per_sample)}",
            f"Successful predictions checked: {len(errors)}",
            "Per-person count consistency: PASS",
            "Summary count consistency: PASS",
            "Failure accounting: PASS",
            "Per-person angular statistics recomputation: PASS",
            "Overall angular statistics recomputation: PASS",
        ]

        with open(
            staged_audit_path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "\n".join(
                    audit_lines
                )
            )

        print(
            "Validating staged plot, table, and audit outputs..."
        )

        validate_staged_outputs(
            staged_table_csv_path,
            staged_table_md_path,
            staged_audit_path,
            staged_figure_path,
            staged_distribution_figure_path,
            table,
        )

        commit_outputs(
            [
                (
                    staged_table_csv_path,
                    TABLE_CSV_PATH,
                ),
                (
                    staged_table_md_path,
                    TABLE_MD_PATH,
                ),
                (
                    staged_audit_path,
                    AUDIT_PATH,
                ),
                (
                    staged_figure_path,
                    FIGURE_PATH,
                ),
                (
                    staged_distribution_figure_path,
                    ERROR_DISTRIBUTION_FIGURE_PATH,
                ),
            ],
            staging_dir,
        )

        print(
            "Committed final plot-owned outputs."
        )

        print(
            "Independent result audit: PASS"
        )

        print(
            f"Saved figure: {FIGURE_PATH}"
        )

        print(
            "Saved angular-error distribution figure: "
            f"{ERROR_DISTRIBUTION_FIGURE_PATH}"
        )

        print(
            f"Saved table CSV: {TABLE_CSV_PATH}"
        )

        print(
            f"Saved table Markdown: {TABLE_MD_PATH}"
        )

        print(
            f"Saved audit: {AUDIT_PATH}"
        )

        print()

        print(
            "Overall row:"
        )

        print(
            table.tail(
                1
            ).to_string(
                index=False
            )
        )


    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
