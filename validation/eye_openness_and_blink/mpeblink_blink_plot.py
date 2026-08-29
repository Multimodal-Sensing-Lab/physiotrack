from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_PATH = RESULTS_DIR / "mpeblink_test_summary.txt"
SEQUENCE_RESULTS_PATH = RESULTS_DIR / "mpeblink_test_sequence_results.csv"

TABLE_CSV_PATH = RESULTS_DIR / "mpeblink_test_thesis_table.csv"
TABLE_MD_PATH = RESULTS_DIR / "mpeblink_test_thesis_table.md"

FIGURE_PATH = FIGURES_DIR / "mpeblink_eye_blink_metrics.png"


def read_summary(path):
    text = path.read_text(
        encoding="utf-8"
    )

    patterns = {
        "videos": r"Videos:\s+(\d+)",
        "person_sequences": r"Person sequences:\s+(\d+)",
        "eye_availability": (
            r"Eye-openness availability:\s+([\d.]+)%"
        ),
        "ground_truth_blinks": (
            r"Ground-truth blinks:\s+(\d+)"
        ),
        "predicted_blinks": (
            r"Predicted blinks:\s+(\d+)"
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
        "count_mae": (
            r"Blink-count MAE per sequence:\s+([\d.]+)"
        ),
        "rate_mae": (
            r"Blink-rate MAE:\s+([\d.]+)"
        ),
        "duration_error": (
            r"Mean blink-duration error:\s+([\d.]+)"
        ),
        "eye_auc": (
            r"Blink-vs-non-blink ROC AUC "
            r"using negative openness:\s+([\d.]+)"
        ),
        "blink_median": (
            r"Blink-frame openness median:\s+([\d.]+)"
        ),
        "nonblink_median": (
            r"Non-blink openness median:\s+([\d.]+)"
        ),
        "runtime": (
            r"Runtime:\s+([\d.]+) minutes"
        ),
    }

    values = {}

    for key, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match is None:
            raise ValueError(
                f"Could not find metric: {key}"
            )

        values[key] = float(
            match.group(1)
        )

    return values


def save_thesis_table(metrics):
    rows = [
        {
            "Evaluation area": "Dataset",
            "Metric": "Test videos",
            "Value": int(metrics["videos"]),
            "Unit": "videos",
        },
        {
            "Evaluation area": "Dataset",
            "Metric": "Person sequences",
            "Value": int(
                metrics["person_sequences"]
            ),
            "Unit": "sequences",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Availability",
            "Value": round(
                metrics["eye_availability"],
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "ROC AUC",
            "Value": round(
                metrics["eye_auc"],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Blink-frame median openness",
            "Value": round(
                metrics["blink_median"],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Eye Openness",
            "Metric": "Non-blink median openness",
            "Value": round(
                metrics["nonblink_median"],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Precision",
            "Value": round(
                metrics["precision"] * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Recall",
            "Value": round(
                metrics["recall"] * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "F1-score",
            "Value": round(
                metrics["f1"] * 100.0,
                2,
            ),
            "Unit": "%",
        },
        {
            "Evaluation area": "Blink Detection",
            "Metric": "Mean matched temporal IoU",
            "Value": round(
                metrics["mean_tiou"],
                4,
            ),
            "Unit": "",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Ground-truth blinks",
            "Value": int(
                metrics["ground_truth_blinks"]
            ),
            "Unit": "events",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Predicted blinks",
            "Value": int(
                metrics["predicted_blinks"]
            ),
            "Unit": "events",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Blink-count MAE per sequence",
            "Value": round(
                metrics["count_mae"],
                4,
            ),
            "Unit": "blinks",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Blink-rate MAE",
            "Value": round(
                metrics["rate_mae"],
                4,
            ),
            "Unit": "blinks/min",
        },
        {
            "Evaluation area": "Blink Events",
            "Metric": "Mean blink-duration error",
            "Value": round(
                metrics["duration_error"],
                4,
            ),
            "Unit": "s",
        },
        {
            "Evaluation area": "Runtime",
            "Metric": "Processing time",
            "Value": round(
                metrics["runtime"],
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

    with open(
        TABLE_MD_PATH,
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
        metrics["eye_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["mean_tiou"],
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        categories,
        values,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "MPEBlink 2.0 Test Performance"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def print_sequence_summary():
    sequence_results = pd.read_csv(
        SEQUENCE_RESULTS_PATH
    )

    print(
        "\nSequence-level summary"
    )
    print(
        "----------------------"
    )

    print(
        "Rows:",
        len(sequence_results),
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
    metrics = read_summary(
        SUMMARY_PATH
    )

    table = save_thesis_table(
        metrics
    )

    save_figure(
        metrics
    )

    print(
        "Thesis table"
    )
    print(
        "------------"
    )

    print(
        table.to_string(
            index=False
        )
    )

    print_sequence_summary()

    print(
        "\nSaved:"
    )
    print(
        TABLE_CSV_PATH
    )
    print(
        TABLE_MD_PATH
    )
    print(
        FIGURE_PATH
    )


if __name__ == "__main__":
    main()