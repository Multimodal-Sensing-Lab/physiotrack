from __future__ import annotations

import math
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

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

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
        TABLE_CSV_PATH,
        index=False,
    )

    table.to_markdown(
        TABLE_MD_PATH,
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
        AUDIT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                audit_lines
            )
        )

    print(
        "Independent result audit: PASS"
    )

    print(
        f"Saved figure: {FIGURE_PATH}"
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


if __name__ == "__main__":
    main()
