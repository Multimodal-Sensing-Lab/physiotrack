from pathlib import Path
import os
import re
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_PATH = RESULTS_DIR / "mpeblink_test_summary.txt"
SEQUENCE_RESULTS_PATH = RESULTS_DIR / "mpeblink_test_sequence_results.csv"
CALIBRATION_CSV_PATH = RESULTS_DIR / "mpeblink_val_calibration.csv"

TABLE_CSV_PATH = RESULTS_DIR / "mpeblink_test_thesis_table.csv"
TABLE_MD_PATH = RESULTS_DIR / "mpeblink_test_thesis_table.md"

FIGURE_PATH = FIGURES_DIR / "mpeblink_eye_blink_metrics.png"
CALIBRATION_FIGURE_PATH = (
    FIGURES_DIR
    / "mpeblink_validation_f1_calibration.png"
)

SELECTED_THRESHOLD = 0.22
SELECTED_MIN_CLOSED_FRAMES = 3


REQUIRED_SEQUENCE_COLUMNS = {
    "video_id",
    "person_id",
    "fps",
    "frames",
    "gt_blinks",
    "predicted_blinks",
    "true_positive",
    "false_positive",
    "false_negative",
    "gt_blink_rate_per_min",
    "predicted_blink_rate_per_min",
    "absolute_count_error",
    "absolute_rate_error_per_min",
}


def make_plot_staging_dir():
    """Create a plotting-owned staging directory under results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".mpeblink_blink_plot_", dir=RESULTS_DIR))


def validate_staged_plot_outputs(table_csv, table_md, metric_figure, calibration_figure):
    """Validate staged tables and figures before final replacement."""
    for path in [table_csv, table_md, metric_figure, calibration_figure]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Expected staged plot output is missing or empty: {path}")
    table = pd.read_csv(table_csv)
    expected_metrics = ["Test videos", "Person sequences", "Availability", "ROC AUC", "Blink-frame median openness", "Non-blink median openness", "Precision", "Recall", "F1-score", "Mean matched temporal IoU", "Ground-truth blinks", "Predicted blinks", "Blink-count MAE per sequence", "Blink-rate MAE", "Mean blink-duration error", "Processing time"]
    if list(table["Metric"]) != expected_metrics:
        raise RuntimeError("Staged thesis-table metric order is incorrect.")
    markdown = table_md.read_text(encoding="utf-8")
    for metric in expected_metrics:
        if metric not in markdown:
            raise RuntimeError(f"Staged Markdown table is missing metric: {metric}")


def replace_plot_outputs(staged_to_final, staging_dir):
    """Replace only plot-owned outputs with rollback protection."""
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups=[]; installed=[]
    try:
        for staged_path, final_path in staged_to_final:
            if not staged_path.is_file():
                raise RuntimeError(f"Missing staged plot output: {staged_path}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path=backup_dir/final_path.name
                os.replace(final_path,backup_path); backups.append((backup_path,final_path))
        for staged_path, final_path in staged_to_final:
            os.replace(staged_path,final_path); installed.append(final_path)
    except Exception:
        for final_path in installed:
            if final_path.exists(): final_path.unlink()
        for backup_path, final_path in reversed(backups):
            if backup_path.exists(): os.replace(backup_path,final_path)
        raise


def remove_file_if_exists(path):
    """Remove one plot-derived output if it exists."""
    if path.is_file():
        path.unlink()


def clean_plot_outputs():
    """Remove only outputs owned by this plotting/table script."""
    paths = [
        TABLE_CSV_PATH,
        TABLE_MD_PATH,
        FIGURE_PATH,
        CALIBRATION_FIGURE_PATH,
    ]

    for path in paths:
        remove_file_if_exists(
            path
        )


def read_summary(path):
    """Read evaluator-level metrics from the accepted test summary."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation summary was not found: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    patterns = {
        "videos": r"Videos:\s+(\d+)",
        "person_sequences": (
            r"Person sequences:\s+(\d+)"
        ),
        "annotation_frames": (
            r"Annotation frames:\s+(\d+)"
        ),
        "video_frames_read": (
            r"Video frames read:\s+(\d+)"
        ),
        "valid_face_boxes": (
            r"Valid face-box samples:\s+(\d+)"
        ),
        "successful_eye_samples": (
            r"Successful eye-openness samples:\s+(\d+)"
        ),
        "eye_availability": (
            r"Eye-openness availability:\s+([\d.]+)%"
        ),
        "missing_bbox": (
            r"Missing bounding boxes:\s+(\d+)"
        ),
        "invalid_bbox": (
            r"Invalid bounding boxes:\s+(\d+)"
        ),
        "landmark_failures": (
            r"Landmark failures:\s+(\d+)"
        ),
        "video_frame_mismatches": (
            r"Video frame mismatches:\s+(\d+)"
        ),
        "video_read_failures": (
            r"Video read failures:\s+(\d+)"
        ),
        "ground_truth_blinks": (
            r"Ground-truth blinks:\s+(\d+)"
        ),
        "predicted_blinks": (
            r"Predicted blinks:\s+(\d+)"
        ),
        "true_positive": (
            r"True positives:\s+(\d+)"
        ),
        "false_positive": (
            r"False positives:\s+(\d+)"
        ),
        "false_negative": (
            r"False negatives:\s+(\d+)"
        ),
        "precision": (
            r"Precision:\s+([\d.]+)"
        ),
        "recall": (
            r"Recall:\s+([\d.]+)"
        ),
        "f1": (
            r"F1:\s+([\d.]+)"
        ),
        "mean_tiou": (
            r"Mean matched temporal IoU:\s+([\d.]+)"
        ),
        "median_tiou": (
            r"Median matched temporal IoU:\s+([\d.]+)"
        ),
        "onset_error": (
            r"Mean onset error:\s+([\d.]+)"
        ),
        "offset_error": (
            r"Mean offset error:\s+([\d.]+)"
        ),
        "count_mae": (
            r"Blink-count MAE per sequence:\s+([\d.]+)"
        ),
        "rate_mae": (
            r"Blink-rate MAE:\s+([\d.]+)"
        ),
        "duration_error": (
            r"Mean blink-duration error:\s+([\d.]+)"
        ),
        "eye_samples": (
            r"Eye-openness samples:\s+(\d+)"
        ),
        "blink_eye_samples": (
            r"Blink-frame eye samples:\s+(\d+)"
        ),
        "nonblink_eye_samples": (
            r"Non-blink eye samples:\s+(\d+)"
        ),
        "eye_auc": (
            r"Blink-vs-non-blink ROC AUC "
            r"using negative openness:\s+([\d.]+)"
        ),
        "blink_mean": (
            r"Blink-frame openness mean:\s+([\d.]+)"
        ),
        "blink_median": (
            r"Blink-frame openness median:\s+([\d.]+)"
        ),
        "nonblink_mean": (
            r"Non-blink openness mean:\s+([\d.]+)"
        ),
        "nonblink_median": (
            r"Non-blink openness median:\s+([\d.]+)"
        ),
        "runtime": (
            r"Runtime:\s+([\d.]+) minutes"
        ),
    }

    values = {}

    integer_keys = {
        "videos",
        "person_sequences",
        "annotation_frames",
        "video_frames_read",
        "valid_face_boxes",
        "successful_eye_samples",
        "missing_bbox",
        "invalid_bbox",
        "landmark_failures",
        "video_frame_mismatches",
        "video_read_failures",
        "ground_truth_blinks",
        "predicted_blinks",
        "true_positive",
        "false_positive",
        "false_negative",
        "eye_samples",
        "blink_eye_samples",
        "nonblink_eye_samples",
    }

    for key, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match is None:
            raise RuntimeError(
                f"Could not find evaluator metric: {key}"
            )

        value = match.group(
            1
        )

        if key in integer_keys:
            values[
                key
            ] = int(
                value
            )
        else:
            values[
                key
            ] = float(
                value
            )

    return values


def load_sequence_results():
    """Load and validate the evaluator's per-person test results."""
    if not SEQUENCE_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "Sequence result CSV was not found: "
            f"{SEQUENCE_RESULTS_PATH}"
        )

    data = pd.read_csv(
        SEQUENCE_RESULTS_PATH
    )

    missing_columns = sorted(
        REQUIRED_SEQUENCE_COLUMNS
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Sequence result CSV is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    if data.empty:
        raise RuntimeError(
            "Sequence result CSV contains no rows."
        )

    numeric_columns = [
        "fps",
        "frames",
        "gt_blinks",
        "predicted_blinks",
        "true_positive",
        "false_positive",
        "false_negative",
        "gt_blink_rate_per_min",
        "predicted_blink_rate_per_min",
        "absolute_count_error",
        "absolute_rate_error_per_min",
    ]

    numeric = data[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            numeric
        )
    ):
        raise RuntimeError(
            "Sequence result CSV contains non-finite numeric values."
        )

    if np.any(
        numeric < 0.0
    ):
        raise RuntimeError(
            "Sequence result CSV contains negative count or error values."
        )

    return data


def verify_results(
    metrics,
    sequence_results,
):
    """Independently verify summary metrics against per-sequence results."""
    if len(
        sequence_results
    ) != metrics[
        "person_sequences"
    ]:
        raise RuntimeError(
            "Person-sequence count does not match the evaluator summary."
        )

    sums = sequence_results[
        [
            "gt_blinks",
            "predicted_blinks",
            "true_positive",
            "false_positive",
            "false_negative",
        ]
    ].sum()

    expected_counts = {
        "ground_truth_blinks": int(
            sums[
                "gt_blinks"
            ]
        ),
        "predicted_blinks": int(
            sums[
                "predicted_blinks"
            ]
        ),
        "true_positive": int(
            sums[
                "true_positive"
            ]
        ),
        "false_positive": int(
            sums[
                "false_positive"
            ]
        ),
        "false_negative": int(
            sums[
                "false_negative"
            ]
        ),
    }

    for key, value in expected_counts.items():
        if value != metrics[
            key
        ]:
            raise RuntimeError(
                f"Summary mismatch for {key}: "
                f"{metrics[key]} != {value}"
            )

    true_positive = metrics[
        "true_positive"
    ]

    false_positive = metrics[
        "false_positive"
    ]

    false_negative = metrics[
        "false_negative"
    ]

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        )
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        )
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    checks = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_mae": float(
            sequence_results[
                "absolute_count_error"
            ].mean()
        ),
        "rate_mae": float(
            sequence_results[
                "absolute_rate_error_per_min"
            ].mean()
        ),
    }

    for key, calculated in checks.items():
        reported = metrics[
            key
        ]

        if not np.isclose(
            reported,
            calculated,
            atol=5e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Summary mismatch for {key}: "
                f"reported={reported:.8f}, "
                f"calculated={calculated:.8f}"
            )

    if metrics[
        "successful_eye_samples"
    ] != metrics[
        "eye_samples"
    ]:
        raise RuntimeError(
            "Successful eye-openness sample count does not match the "
            "ROC-AUC population."
        )

    if (
        metrics[
            "blink_eye_samples"
        ]
        + metrics[
            "nonblink_eye_samples"
        ]
        != metrics[
            "eye_samples"
        ]
    ):
        raise RuntimeError(
            "Blink and non-blink eye-sample counts do not sum to the "
            "reported ROC-AUC population."
        )

    expected_availability = (
        metrics[
            "successful_eye_samples"
        ]
        / metrics[
            "valid_face_boxes"
        ]
        * 100.0
        if metrics[
            "valid_face_boxes"
        ] > 0
        else 0.0
    )

    if not np.isclose(
        metrics[
            "eye_availability"
        ],
        expected_availability,
        atol=0.01,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Eye-openness availability does not match the evaluator counts."
        )


def save_thesis_table(metrics):
    """Save the thesis-ready quantitative summary table."""
    rows = [
        {
            "Evaluation area": "Dataset",
            "Metric": "Test videos",
            "Value": metrics[
                "videos"
            ],
            "Unit": "videos",
        },
        {
            "Evaluation area": "Dataset",
            "Metric": "Person sequences",
            "Value": metrics[
                "person_sequences"
            ],
            "Unit": "sequences",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Availability",
            "Value": round(
                metrics[
                    "eye_availability"
                ],
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "ROC AUC",
            "Value": round(
                metrics[
                    "eye_auc"
                ],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Blink-frame median openness",
            "Value": round(
                metrics[
                    "blink_median"
                ],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Non-blink median openness",
            "Value": round(
                metrics[
                    "nonblink_median"
                ],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Precision",
            "Value": round(
                metrics[
                    "precision"
                ]
                * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Recall",
            "Value": round(
                metrics[
                    "recall"
                ]
                * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "F1-score",
            "Value": round(
                metrics[
                    "f1"
                ]
                * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Mean matched temporal IoU",
            "Value": round(
                metrics[
                    "mean_tiou"
                ],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Ground-truth blinks",
            "Value": metrics[
                "ground_truth_blinks"
            ],
            "Unit": "events",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Predicted blinks",
            "Value": metrics[
                "predicted_blinks"
            ],
            "Unit": "events",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Blink-count MAE per sequence",
            "Value": round(
                metrics[
                    "count_mae"
                ],
                4,
            ),
            "Unit": "blinks",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Blink-rate MAE",
            "Value": round(
                metrics[
                    "rate_mae"
                ],
                4,
            ),
            "Unit": "blinks/min",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Mean blink-duration error",
            "Value": round(
                metrics[
                    "duration_error"
                ],
                4,
            ),
            "Unit": "s",
        },
        {
            "Evaluation area": "Runtime",
            "Metric": "Processing time",
            "Value": round(
                metrics[
                    "runtime"
                ],
                2,
            ),
            "Unit": "min",
        },
    ]

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        TABLE_CSV_PATH,
        index=False,
    )

    with TABLE_MD_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            table.to_markdown(
                index=False
            )
        )

    return table


def save_figure(metrics):
    """Create the thesis-ready quantitative performance figure."""
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    categories = [
        "Eye Openness\nROC AUC",
        "Blink\nPrecision",
        "Blink\nRecall",
        "Blink\nF1-score",
        "Matched\nTemporal IoU",
    ]

    values = [
        metrics[
            "eye_auc"
        ],
        metrics[
            "precision"
        ],
        metrics[
            "recall"
        ],
        metrics[
            "f1"
        ],
        metrics[
            "mean_tiou"
        ],
    ]

    figure, axis = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    bars = axis.bar(
        categories,
        values,
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_title(
        "MPEBlink 2.0 Test Performance"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_calibration_figure():
    """Create the validation-split F1 calibration figure."""
    if not CALIBRATION_CSV_PATH.is_file():
        raise FileNotFoundError(
            "Validation calibration CSV was not found: "
            f"{CALIBRATION_CSV_PATH}"
        )

    calibration = pd.read_csv(
        CALIBRATION_CSV_PATH
    )

    required_columns = {
        "threshold",
        "min_closed_frames",
        "f1",
    }

    missing_columns = sorted(
        required_columns
        - set(
            calibration.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Calibration CSV is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    calibration = calibration.copy()

    for column in [
        "threshold",
        "min_closed_frames",
        "f1",
    ]:
        calibration[
            column
        ] = pd.to_numeric(
            calibration[
                column
            ],
            errors="raise",
        )

    if not np.all(
        np.isfinite(
            calibration[
                [
                    "threshold",
                    "min_closed_frames",
                    "f1",
                ]
            ].to_numpy(
                dtype=float
            )
        )
    ):
        raise RuntimeError(
            "Calibration CSV contains non-finite threshold, "
            "min_closed_frames, or F1 values."
        )

    selected = calibration[
        np.isclose(
            calibration[
                "threshold"
            ].to_numpy(
                dtype=float
            ),
            SELECTED_THRESHOLD,
            atol=1e-12,
            rtol=0.0,
        )
        & (
            calibration[
                "min_closed_frames"
            ].astype(
                int
            )
            == SELECTED_MIN_CLOSED_FRAMES
        )
    ]

    if len(
        selected
    ) != 1:
        raise RuntimeError(
            "Expected exactly one selected calibration row for "
            f"threshold={SELECTED_THRESHOLD:.2f} and "
            f"min_closed_frames={SELECTED_MIN_CLOSED_FRAMES}."
        )

    selected_f1 = float(
        selected.iloc[
            0
        ][
            "f1"
        ]
    )

    best_f1 = float(
        calibration[
            "f1"
        ].max()
    )

    if not np.isclose(
        selected_f1,
        best_f1,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen selected blink parameters are not the maximum-F1 "
            "configuration in the calibration CSV."
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    min_frame_values = sorted(
        calibration[
            "min_closed_frames"
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    for min_closed_frames in min_frame_values:
        subset = calibration[
            calibration[
                "min_closed_frames"
            ].astype(
                int
            )
            == min_closed_frames
        ].sort_values(
            "threshold"
        )

        axis.plot(
            subset[
                "threshold"
            ],
            subset[
                "f1"
            ],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=(
                f"min_closed_frames = "
                f"{min_closed_frames}"
            ),
        )

    axis.scatter(
        [
            SELECTED_THRESHOLD
        ],
        [
            selected_f1
        ],
        s=80,
        zorder=5,
        label=(
            "Selected: "
            f"threshold={SELECTED_THRESHOLD:.2f}, "
            f"min_frames={SELECTED_MIN_CLOSED_FRAMES}, "
            f"F1={selected_f1:.4f}"
        ),
    )

    axis.axvline(
        SELECTED_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

    axis.set_xlabel(
        "EyeOpenness Threshold"
    )

    axis.set_ylabel(
        "Validation Blink-Event F1"
    )

    axis.set_title(
        "MPEBlink 2.0 Validation Calibration"
    )

    axis.set_xlim(
        float(
            calibration[
                "threshold"
            ].min()
        ),
        float(
            calibration[
                "threshold"
            ].max()
        ),
    )

    axis.set_ylim(
        0.0,
        min(
            1.0,
            max(
                0.35,
                float(
                    calibration[
                        "f1"
                    ].max()
                )
                + 0.05,
            ),
        ),
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        fontsize=9
    )

    figure.tight_layout()

    figure.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return selected_f1


def print_sequence_summary(
    sequence_results,
):
    """Print independent sequence-level diagnostics."""
    print(
        "\nSequence-level summary"
    )

    print(
        "----------------------"
    )

    print(
        "Rows:",
        len(
            sequence_results
        ),
    )

    print(
        "Mean absolute blink-count error:",
        f"{sequence_results['absolute_count_error'].mean():.4f}",
    )

    print(
        "Median absolute blink-count error:",
        f"{sequence_results['absolute_count_error'].median():.4f}",
    )

    print(
        "Mean absolute blink-rate error:",
        f"{sequence_results['absolute_rate_error_per_min'].mean():.4f}",
    )

    print(
        "Median absolute blink-rate error:",
        f"{sequence_results['absolute_rate_error_per_min'].median():.4f}",
    )


def main():
    global TABLE_CSV_PATH, TABLE_MD_PATH, FIGURE_PATH, CALIBRATION_FIGURE_PATH, FIGURES_DIR
    metrics = read_summary(SUMMARY_PATH)
    sequence_results = load_sequence_results()
    verify_results(metrics, sequence_results)
    staging_dir = make_plot_staging_dir()
    final_table_csv=TABLE_CSV_PATH; final_table_md=TABLE_MD_PATH; final_figure=FIGURE_PATH; final_calibration=CALIBRATION_FIGURE_PATH; final_figures_dir=FIGURES_DIR
    staged_figures=staging_dir/"figures"
    staged_table_csv=staging_dir/final_table_csv.name
    staged_table_md=staging_dir/final_table_md.name
    staged_figure=staged_figures/final_figure.name
    staged_calibration=staged_figures/final_calibration.name
    try:
        print("Staging directory:", staging_dir)
        TABLE_CSV_PATH=staged_table_csv; TABLE_MD_PATH=staged_table_md; FIGURES_DIR=staged_figures; FIGURE_PATH=staged_figure; CALIBRATION_FIGURE_PATH=staged_calibration
        table=save_thesis_table(metrics)
        save_figure(metrics)
        selected_validation_f1=save_calibration_figure()
        print("Validating staged plot outputs...")
        validate_staged_plot_outputs(staged_table_csv, staged_table_md, staged_figure, staged_calibration)
        replace_plot_outputs([(staged_table_csv,final_table_csv),(staged_table_md,final_table_md),(staged_figure,final_figure),(staged_calibration,final_calibration)], staging_dir)
        print("Committed final plot and table outputs.")
        print("Independent result verification: PASS")
        print("Selected validation calibration F1:", f"{selected_validation_f1:.6f}")
        print("\nThesis table")
        print("------------")
        print(table.to_string(index=False))
        print_sequence_summary(sequence_results)
        print("\nSaved:")
        print(final_table_csv); print(final_table_md); print(final_figure); print(final_calibration)
    finally:
        TABLE_CSV_PATH=final_table_csv; TABLE_MD_PATH=final_table_md; FIGURES_DIR=final_figures_dir; FIGURE_PATH=final_figure; CALIBRATION_FIGURE_PATH=final_calibration
        if staging_dir.exists(): shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
