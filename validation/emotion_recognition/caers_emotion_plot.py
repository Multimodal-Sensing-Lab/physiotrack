from pathlib import Path
import csv
import math
import os
import re
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parent

RESULTS_DIR = VALIDATION_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_PATH = RESULTS_DIR / "caers_emotion_results.csv"
SUMMARY_PATH = RESULTS_DIR / "caers_emotion_summary.txt"

METRICS_PATH = RESULTS_DIR / "caers_emotion_metrics.csv"
THESIS_TABLE_CSV_PATH = RESULTS_DIR / "caers_emotion_thesis_table.csv"
THESIS_TABLE_MD_PATH = RESULTS_DIR / "caers_emotion_thesis_table.md"
PER_CLASS_PATH = RESULTS_DIR / "caers_emotion_per_class_metrics.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "caers_emotion_confusion_matrix.csv"
PREDICTION_DISTRIBUTION_PATH = (
    RESULTS_DIR
    / "caers_emotion_prediction_distribution.csv"
)

CONFUSION_MATRIX_FIGURE_PATH = (
    FIGURES_DIR
    / "caers_emotion_confusion_matrix.png"
)
PER_CLASS_FIGURE_PATH = (
    FIGURES_DIR
    / "caers_emotion_per_class_metrics.png"
)
PREDICTION_DISTRIBUTION_FIGURE_PATH = (
    FIGURES_DIR
    / "caers_emotion_prediction_distribution.png"
)

GT_CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

MODEL_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

SCORE_COLUMNS = {
    "Anger": "score_anger",
    "Contempt": "score_contempt",
    "Disgust": "score_disgust",
    "Fear": "score_fear",
    "Happiness": "score_happiness",
    "Neutral": "score_neutral",
    "Sadness": "score_sadness",
    "Surprise": "score_surprise",
}

EXPECTED_RESULT_COLUMNS = [
    "relative_path",
    "dataset_label",
    "expected_model_label",
    "class_id",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
    "clipped_x1",
    "clipped_y1",
    "clipped_x2",
    "clipped_y2",
    "box_was_clipped",
    "crop_width",
    "crop_height",
    "status",
    "predicted_label",
    "confidence",
    "score_anger",
    "score_contempt",
    "score_disgust",
    "score_fear",
    "score_happiness",
    "score_neutral",
    "score_sadness",
    "score_surprise",
    "score_sum",
    "correct",
    "failure_reason",
]

FLOAT_TOLERANCE = 1e-6
SCORE_SUM_TOLERANCE = 1e-5


def parse_bool(value):
    """Parse a CSV boolean value."""
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise RuntimeError(
        f"Unexpected boolean value: {value}"
    )


def load_results():
    """Load and validate the detailed evaluator output."""
    if not RESULTS_PATH.is_file():
        raise RuntimeError(
            f"Detailed result file not found: {RESULTS_PATH}"
        )

    if not SUMMARY_PATH.is_file():
        raise RuntimeError(
            f"Evaluator summary not found: {SUMMARY_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != EXPECTED_RESULT_COLUMNS:
            raise RuntimeError(
                "Detailed result schema does not match the expected schema."
            )

        rows = list(
            reader
        )

    if not rows:
        raise RuntimeError(
            "Detailed result file contains no samples."
        )

    seen_paths = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        relative_path = row[
            "relative_path"
        ]

        if relative_path in seen_paths:
            raise RuntimeError(
                f"Duplicate result path at CSV row {row_number}: "
                f"{relative_path}"
            )

        seen_paths.add(
            relative_path
        )

        expected_label = row[
            "expected_model_label"
        ]
        predicted_label = row[
            "predicted_label"
        ]
        status = row[
            "status"
        ]

        if expected_label not in GT_CLASSES:
            raise RuntimeError(
                f"Unexpected ground-truth label at CSV row {row_number}: "
                f"{expected_label}"
            )

        if status != "ok":
            raise RuntimeError(
                "Plot/table generation requires the accepted zero-failure "
                f"quantitative result package; found status={status!r} "
                f"at CSV row {row_number}."
            )

        if predicted_label not in MODEL_CLASSES:
            raise RuntimeError(
                f"Unexpected predicted label at CSV row {row_number}: "
                f"{predicted_label}"
            )

        scores = np.asarray(
            [
                float(
                    row[
                        SCORE_COLUMNS[
                            class_name
                        ]
                    ]
                )
                for class_name in MODEL_CLASSES
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                scores
            )
        ):
            raise RuntimeError(
                f"Non-finite model score at CSV row {row_number}."
            )

        if np.any(
            scores < 0.0
        ) or np.any(
            scores > 1.0
        ):
            raise RuntimeError(
                f"Model score outside [0, 1] at CSV row {row_number}."
            )

        calculated_score_sum = float(
            np.sum(
                scores
            )
        )

        recorded_score_sum = float(
            row[
                "score_sum"
            ]
        )

        if not math.isclose(
            calculated_score_sum,
            recorded_score_sum,
            rel_tol=0.0,
            abs_tol=SCORE_SUM_TOLERANCE,
        ):
            raise RuntimeError(
                f"Recorded score sum mismatch at CSV row {row_number}."
            )

        if not math.isclose(
            calculated_score_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=SCORE_SUM_TOLERANCE,
        ):
            raise RuntimeError(
                f"Model probabilities do not sum to one at CSV row "
                f"{row_number}."
            )

        predicted_index = int(
            np.argmax(
                scores
            )
        )
        argmax_label = MODEL_CLASSES[
            predicted_index
        ]

        if argmax_label != predicted_label:
            raise RuntimeError(
                f"Predicted label is not the score argmax at CSV row "
                f"{row_number}."
            )

        confidence = float(
            row[
                "confidence"
            ]
        )

        if not math.isfinite(
            confidence
        ):
            raise RuntimeError(
                f"Non-finite confidence at CSV row {row_number}."
            )

        expected_confidence = float(
            row[
                SCORE_COLUMNS[
                    predicted_label
                ]
            ]
        )

        if not math.isclose(
            confidence,
            expected_confidence,
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError(
                f"Confidence does not match the predicted-class score at "
                f"CSV row {row_number}."
            )

        recorded_correct = parse_bool(
            row[
                "correct"
            ]
        )

        calculated_correct = (
            predicted_label
            == expected_label
        )

        if recorded_correct != calculated_correct:
            raise RuntimeError(
                f"Correctness flag mismatch at CSV row {row_number}."
            )

    return rows


def compute_confusion_matrix(
    rows,
):
    """Compute the 7-by-8 CAER-S ground-truth/model prediction matrix."""
    matrix = np.zeros(
        (
            len(
                GT_CLASSES
            ),
            len(
                MODEL_CLASSES
            ),
        ),
        dtype=np.int64,
    )

    gt_index = {
        class_name: index
        for index, class_name in enumerate(
            GT_CLASSES
        )
    }

    prediction_index = {
        class_name: index
        for index, class_name in enumerate(
            MODEL_CLASSES
        )
    }

    for row in rows:
        matrix[
            gt_index[
                row[
                    "expected_model_label"
                ]
            ],
            prediction_index[
                row[
                    "predicted_label"
                ]
            ],
        ] += 1

    return matrix


def compute_metrics(
    rows,
    matrix,
):
    """Independently recompute aggregate and per-class metrics."""
    total = len(
        rows
    )

    correct = sum(
        1
        for row in rows
        if row[
            "predicted_label"
        ]
        == row[
            "expected_model_label"
        ]
    )

    accuracy = (
        correct
        / total
    )

    per_class = []

    for gt_index, class_name in enumerate(
        GT_CLASSES
    ):
        prediction_index = MODEL_CLASSES.index(
            class_name
        )

        tp = int(
            matrix[
                gt_index,
                prediction_index,
            ]
        )

        support = int(
            matrix[
                gt_index,
                :,
            ].sum()
        )

        predicted_count = int(
            matrix[
                :,
                prediction_index,
            ].sum()
        )

        fp = (
            predicted_count
            - tp
        )

        fn = (
            support
            - tp
        )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            ) > 0
            else 0.0
        )

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            ) > 0
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
            ) > 0
            else 0.0
        )

        per_class.append(
            {
                "class": class_name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_precision = float(
        np.mean(
            [
                row[
                    "precision"
                ]
                for row in per_class
            ]
        )
    )

    macro_recall = float(
        np.mean(
            [
                row[
                    "recall"
                ]
                for row in per_class
            ]
        )
    )

    macro_f1 = float(
        np.mean(
            [
                row[
                    "f1"
                ]
                for row in per_class
            ]
        )
    )

    weighted_f1 = float(
        sum(
            row[
                "f1"
            ]
            * row[
                "support"
            ]
            for row in per_class
        )
        / total
    )

    contempt_count = int(
        matrix[
            :,
            MODEL_CLASSES.index(
                "Contempt"
            ),
        ].sum()
    )

    prediction_counts = {
        class_name: int(
            matrix[
                :,
                class_index,
            ].sum()
        )
        for class_index, class_name in enumerate(
            MODEL_CLASSES
        )
    }

    return {
        "total": total,
        "correct": correct,
        "availability": 1.0,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "contempt_count": contempt_count,
        "contempt_rate": (
            contempt_count
            / total
        ),
        "per_class": per_class,
        "prediction_counts": prediction_counts,
    }


def parse_summary_metrics():
    """Read accepted evaluator metrics from the textual summary."""
    text = SUMMARY_PATH.read_text(
        encoding="utf-8",
    )

    patterns = {
        "total": (
            r"Annotated samples:\s*(\d+)",
            int,
        ),
        "correct": (
            r"Correct predictions:\s*(\d+)",
            int,
        ),
        "accuracy": (
            r"Accuracy:\s*([0-9.]+)",
            float,
        ),
        "macro_f1": (
            r"Macro F1:\s*([0-9.]+)",
            float,
        ),
        "weighted_f1": (
            r"Weighted F1:\s*([0-9.]+)",
            float,
        ),
        "contempt_count": (
            r"Contempt predictions:\s*(\d+)",
            int,
        ),
        "contempt_rate": (
            r"Contempt prediction rate:\s*([0-9.]+)",
            float,
        ),
    }

    values = {}

    for name, (
        pattern,
        converter,
    ) in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match is None:
            raise RuntimeError(
                f"Required metric not found in evaluator summary: {name}"
            )

        values[
            name
        ] = converter(
            match.group(
                1
            )
        )

    return values


def validate_summary_consistency(
    metrics,
):
    """Verify independent metric recomputation against the evaluator summary."""
    summary = parse_summary_metrics()

    for name in [
        "total",
        "correct",
        "contempt_count",
    ]:
        if metrics[
            name
        ] != summary[
            name
        ]:
            raise RuntimeError(
                f"Evaluator summary mismatch for {name}."
            )

    for name in [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "contempt_rate",
    ]:
        if not math.isclose(
            metrics[
                name
            ],
            summary[
                name
            ],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"Evaluator summary mismatch for {name}."
            )


def write_metrics(
    metrics,
    output_path,
):
    """Write the compact aggregate metric table."""
    fieldnames = [
        "samples",
        "correct_predictions",
        "availability",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "contempt_predictions",
        "contempt_prediction_rate",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "samples": metrics[
                    "total"
                ],
                "correct_predictions": metrics[
                    "correct"
                ],
                "availability": metrics[
                    "availability"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "macro_precision": metrics[
                    "macro_precision"
                ],
                "macro_recall": metrics[
                    "macro_recall"
                ],
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "weighted_f1": metrics[
                    "weighted_f1"
                ],
                "contempt_predictions": metrics[
                    "contempt_count"
                ],
                "contempt_prediction_rate": metrics[
                    "contempt_rate"
                ],
            }
        )


def write_per_class_metrics(
    metrics,
    output_path,
):
    """Write independently recomputed per-class classification metrics."""
    fieldnames = [
        "class",
        "support",
        "predicted_count",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in metrics[
            "per_class"
        ]:
            writer.writerow(
                row
            )


def write_confusion_matrix(
    matrix,
    output_path,
):
    """Write the raw 7-by-8 confusion matrix as integer counts."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "ground_truth",
                *MODEL_CLASSES,
            ]
        )

        for row_index, class_name in enumerate(
            GT_CLASSES
        ):
            writer.writerow(
                [
                    class_name,
                    *matrix[
                        row_index,
                        :,
                    ].tolist(),
                ]
            )


def write_prediction_distribution(
    metrics,
    output_path,
):
    """Write prediction counts and rates for all eight model classes."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "predicted_class",
                "count",
                "rate",
            ],
        )

        writer.writeheader()

        for class_name in MODEL_CLASSES:
            count = metrics[
                "prediction_counts"
            ][
                class_name
            ]

            writer.writerow(
                {
                    "predicted_class": class_name,
                    "count": count,
                    "rate": (
                        count
                        / metrics[
                            "total"
                        ]
                    ),
                }
            )


def write_thesis_tables(
    metrics,
    csv_path,
    md_path,
):
    """Write compact CSV and Markdown thesis tables."""
    row = {
        "Dataset": "CAER-S annotated target-face subset",
        "Samples": metrics[
            "total"
        ],
        "Availability (%)": (
            metrics[
                "availability"
            ]
            * 100.0
        ),
        "Accuracy (%)": (
            metrics[
                "accuracy"
            ]
            * 100.0
        ),
        "Macro F1 (%)": (
            metrics[
                "macro_f1"
            ]
            * 100.0
        ),
        "Weighted F1 (%)": (
            metrics[
                "weighted_f1"
            ]
            * 100.0
        ),
        "Contempt Predictions": metrics[
            "contempt_count"
        ],
        "Contempt Rate (%)": (
            metrics[
                "contempt_rate"
            ]
            * 100.0
        ),
    }

    fieldnames = list(
        row.keys()
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerow(
            row
        )

    header = (
        "| Dataset | Samples | Availability (%) | Accuracy (%) | "
        "Macro F1 (%) | Weighted F1 (%) | Contempt Predictions | "
        "Contempt Rate (%) |"
    )

    separator = (
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    values = (
        "| "
        f"{row['Dataset']} | "
        f"{row['Samples']} | "
        f"{row['Availability (%)']:.2f} | "
        f"{row['Accuracy (%)']:.2f} | "
        f"{row['Macro F1 (%)']:.2f} | "
        f"{row['Weighted F1 (%)']:.2f} | "
        f"{row['Contempt Predictions']} | "
        f"{row['Contempt Rate (%)']:.2f} |"
    )

    md_path.write_text(
        "\n".join(
            [
                header,
                separator,
                values,
                "",
            ]
        ),
        encoding="utf-8",
    )


def create_confusion_matrix_figure(
    matrix,
    output_path,
):
    """Create a row-normalized 7-by-8 confusion-matrix figure."""
    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=(
            row_totals
            != 0
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            11.0,
            8.0,
        )
    )

    image = axis.imshow(
        normalized,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_xticks(
        np.arange(
            len(
                MODEL_CLASSES
            )
        )
    )
    axis.set_xticklabels(
        MODEL_CLASSES,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(
                GT_CLASSES
            )
        )
    )
    axis.set_yticklabels(
        GT_CLASSES
    )

    axis.set_xlabel(
        "Predicted class"
    )
    axis.set_ylabel(
        "Ground-truth class"
    )
    axis.set_title(
        "CAER-S Emotion Recognition Confusion Matrix"
    )

    for row_index in range(
        normalized.shape[
            0
        ]
    ):
        for column_index in range(
            normalized.shape[
                1
            ]
        ):
            percentage = (
                normalized[
                    row_index,
                    column_index,
                ]
                * 100.0
            )

            axis.text(
                column_index,
                row_index,
                f"{percentage:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(
        "Row-normalized proportion"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_per_class_figure(
    metrics,
    output_path,
):
    """Create the per-class precision, recall, and F1 figure."""
    classes = [
        row[
            "class"
        ]
        for row in metrics[
            "per_class"
        ]
    ]

    precision = [
        row[
            "precision"
        ]
        for row in metrics[
            "per_class"
        ]
    ]

    recall = [
        row[
            "recall"
        ]
        for row in metrics[
            "per_class"
        ]
    ]

    f1 = [
        row[
            "f1"
        ]
        for row in metrics[
            "per_class"
        ]
    ]

    x_positions = np.arange(
        len(
            classes
        )
    )

    width = 0.25

    figure, axis = plt.subplots(
        figsize=(
            11.0,
            6.0,
        )
    )

    axis.bar(
        x_positions
        - width,
        precision,
        width=width,
        label="Precision",
    )

    axis.bar(
        x_positions,
        recall,
        width=width,
        label="Recall",
    )

    axis.bar(
        x_positions
        + width,
        f1,
        width=width,
        label="F1",
    )

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels(
        classes,
        rotation=30,
        ha="right",
    )

    axis.set_ylim(
        0.0,
        1.0,
    )
    axis.set_xlabel(
        "Ground-truth class"
    )
    axis.set_ylabel(
        "Score"
    )
    axis.set_title(
        "CAER-S Emotion Recognition Per-Class Metrics"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_prediction_distribution_figure(
    metrics,
    output_path,
):
    """Create a figure showing the eight-class prediction distribution."""
    counts = [
        metrics[
            "prediction_counts"
        ][
            class_name
        ]
        for class_name in MODEL_CLASSES
    ]

    x_positions = np.arange(
        len(
            MODEL_CLASSES
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            11.0,
            6.0,
        )
    )

    axis.bar(
        x_positions,
        counts,
    )

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels(
        MODEL_CLASSES,
        rotation=30,
        ha="right",
    )

    axis.set_xlabel(
        "Predicted class"
    )
    axis.set_ylabel(
        "Prediction count"
    )
    axis.set_title(
        "CAER-S Emotion Recognition Prediction Distribution"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def validate_staged_outputs(
    staging_dir,
):
    """Validate all plot-script-owned staged outputs."""
    required_files = [
        METRICS_PATH.name,
        THESIS_TABLE_CSV_PATH.name,
        THESIS_TABLE_MD_PATH.name,
        PER_CLASS_PATH.name,
        CONFUSION_MATRIX_PATH.name,
        PREDICTION_DISTRIBUTION_PATH.name,
        CONFUSION_MATRIX_FIGURE_PATH.name,
        PER_CLASS_FIGURE_PATH.name,
        PREDICTION_DISTRIBUTION_FIGURE_PATH.name,
    ]

    for filename in required_files:
        path = (
            staging_dir
            / filename
        )

        if not path.is_file():
            raise RuntimeError(
                f"Expected staged output was not created: {filename}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Staged output is empty: {filename}"
            )

    with (
        staging_dir
        / METRICS_PATH.name
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        metrics_rows = list(
            csv.DictReader(
                handle
            )
        )

    if len(
        metrics_rows
    ) != 1:
        raise RuntimeError(
            "Aggregate metrics CSV must contain exactly one data row."
        )

    with (
        staging_dir
        / PER_CLASS_PATH.name
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        per_class_rows = list(
            csv.DictReader(
                handle
            )
        )

    if len(
        per_class_rows
    ) != len(
        GT_CLASSES
    ):
        raise RuntimeError(
            "Per-class metrics CSV does not contain seven class rows."
        )

    with (
        staging_dir
        / CONFUSION_MATRIX_PATH.name
    ).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        confusion_rows = list(
            csv.reader(
                handle
            )
        )

    if len(
        confusion_rows
    ) != (
        len(
            GT_CLASSES
        )
        + 1
    ):
        raise RuntimeError(
            "Confusion-matrix CSV row count is invalid."
        )


def replace_owned_outputs(
    staging_dir,
):
    """Replace only plot-script-owned outputs after staged validation."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_outputs = [
        METRICS_PATH,
        THESIS_TABLE_CSV_PATH,
        THESIS_TABLE_MD_PATH,
        PER_CLASS_PATH,
        CONFUSION_MATRIX_PATH,
        PREDICTION_DISTRIBUTION_PATH,
    ]

    figure_outputs = [
        CONFUSION_MATRIX_FIGURE_PATH,
        PER_CLASS_FIGURE_PATH,
        PREDICTION_DISTRIBUTION_FIGURE_PATH,
    ]

    for final_path in root_outputs:
        os.replace(
            staging_dir
            / final_path.name,
            final_path,
        )

    for final_path in figure_outputs:
        os.replace(
            staging_dir
            / final_path.name,
            final_path,
        )


def main():
    """Generate independently verified CAER-S tables and figures."""
    print(
        "CAER-S Emotion Recognition Plot and Table Generation"
    )
    print(
        "=" * 51
    )

    rows = load_results()

    matrix = compute_confusion_matrix(
        rows
    )

    metrics = compute_metrics(
        rows,
        matrix,
    )

    validate_summary_consistency(
        metrics
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".caers_emotion_plot_",
            dir=RESULTS_DIR,
        )
    )

    try:
        write_metrics(
            metrics,
            staging_dir
            / METRICS_PATH.name,
        )

        write_per_class_metrics(
            metrics,
            staging_dir
            / PER_CLASS_PATH.name,
        )

        write_confusion_matrix(
            matrix,
            staging_dir
            / CONFUSION_MATRIX_PATH.name,
        )

        write_prediction_distribution(
            metrics,
            staging_dir
            / PREDICTION_DISTRIBUTION_PATH.name,
        )

        write_thesis_tables(
            metrics,
            staging_dir
            / THESIS_TABLE_CSV_PATH.name,
            staging_dir
            / THESIS_TABLE_MD_PATH.name,
        )

        create_confusion_matrix_figure(
            matrix,
            staging_dir
            / CONFUSION_MATRIX_FIGURE_PATH.name,
        )

        create_per_class_figure(
            metrics,
            staging_dir
            / PER_CLASS_FIGURE_PATH.name,
        )

        create_prediction_distribution_figure(
            metrics,
            staging_dir
            / PREDICTION_DISTRIBUTION_FIGURE_PATH.name,
        )

        validate_staged_outputs(
            staging_dir
        )

        replace_owned_outputs(
            staging_dir
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    print(
        f"Samples: {metrics['total']:,}"
    )
    print(
        f"Correct predictions: {metrics['correct']:,}"
    )
    print(
        f"Accuracy: {metrics['accuracy']:.6f}"
    )
    print(
        f"Macro F1: {metrics['macro_f1']:.6f}"
    )
    print(
        f"Weighted F1: {metrics['weighted_f1']:.6f}"
    )
    print(
        f"Contempt predictions: {metrics['contempt_count']:,}"
    )
    print(
        "Evaluator-summary consistency: PASS"
    )
    print(
        "Detailed-result integrity checks: PASS"
    )
    print()
    print(
        "Generated plot/table outputs:"
    )
    print(
        METRICS_PATH
    )
    print(
        THESIS_TABLE_CSV_PATH
    )
    print(
        THESIS_TABLE_MD_PATH
    )
    print(
        PER_CLASS_PATH
    )
    print(
        CONFUSION_MATRIX_PATH
    )
    print(
        PREDICTION_DISTRIBUTION_PATH
    )
    print(
        CONFUSION_MATRIX_FIGURE_PATH
    )
    print(
        PER_CLASS_FIGURE_PATH
    )
    print(
        PREDICTION_DISTRIBUTION_FIGURE_PATH
    )


if __name__ == "__main__":
    main()
