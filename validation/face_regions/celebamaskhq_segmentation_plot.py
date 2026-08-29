from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_CSV = (
    RESULTS_DIR
    / "celebamaskhq_class_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "celebamaskhq_segmentation_summary.txt"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "celebamaskhq_per_class_metrics.png"
)

TABLE_CSV_PATH = (
    RESULTS_DIR
    / "celebamaskhq_thesis_table.csv"
)

TABLE_MD_PATH = (
    RESULTS_DIR
    / "celebamaskhq_thesis_table.md"
)

SUMMARY_TABLE_CSV_PATH = (
    RESULTS_DIR
    / "celebamaskhq_summary_table.csv"
)

SUMMARY_TABLE_MD_PATH = (
    RESULTS_DIR
    / "celebamaskhq_summary_table.md"
)


def load_metrics():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {INPUT_CSV}"
        )

    dataframe = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "class_id",
        "class_name",
        "gt_pixels",
        "pred_pixels",
        "iou",
        "dice",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    expected_classes = [
        "background",
        "neck",
        "skin",
        "cloth",
        "l_ear",
        "r_ear",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "nose",
        "mouth",
        "l_lip",
        "u_lip",
        "hair",
        "eye_g",
        "hat",
        "ear_r",
        "neck_l",
    ]

    if dataframe["class_name"].tolist() != expected_classes:
        raise ValueError(
            "Unexpected CelebAMask-HQ class order in metrics CSV."
        )

    return dataframe


def load_summary_metrics():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary file not found: {SUMMARY_PATH}"
        )

    metrics = {}

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith(
                "Pixel accuracy:"
            ):
                metrics[
                    "Pixel Accuracy (%)"
                ] = (
                    float(
                        line.split(
                            ":",
                            1,
                        )[1].strip()
                    )
                    * 100.0
                )

            elif line.startswith(
                "All-class mIoU:"
            ):
                metrics[
                    "All-class mIoU (%)"
                ] = (
                    float(
                        line.split(
                            ":",
                            1,
                        )[1].strip()
                    )
                    * 100.0
                )

            elif line.startswith(
                "Foreground mIoU:"
            ):
                metrics[
                    "Foreground mIoU (%)"
                ] = (
                    float(
                        line.split(
                            ":",
                            1,
                        )[1].strip()
                    )
                    * 100.0
                )

            elif line.startswith(
                "All-class mean Dice:"
            ):
                metrics[
                    "All-class Mean Dice (%)"
                ] = (
                    float(
                        line.split(
                            ":",
                            1,
                        )[1].strip()
                    )
                    * 100.0
                )

            elif line.startswith(
                "Foreground mean Dice:"
            ):
                metrics[
                    "Foreground Mean Dice (%)"
                ] = (
                    float(
                        line.split(
                            ":",
                            1,
                        )[1].strip()
                    )
                    * 100.0
                )

    required_metrics = {
        "Pixel Accuracy (%)",
        "All-class mIoU (%)",
        "Foreground mIoU (%)",
        "All-class Mean Dice (%)",
        "Foreground Mean Dice (%)",
    }

    missing_metrics = (
        required_metrics
        - set(metrics)
    )

    if missing_metrics:
        raise ValueError(
            "Missing summary metrics: "
            + ", ".join(
                sorted(missing_metrics)
            )
        )

    return metrics


def create_figure(dataframe):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = dataframe[
        "class_name"
    ].tolist()

    iou = (
        dataframe["iou"]
        .astype(float)
        .to_numpy()
        * 100.0
    )

    dice = (
        dataframe["dice"]
        .astype(float)
        .to_numpy()
        * 100.0
    )

    x_positions = list(
        range(len(dataframe))
    )

    bar_width = 0.38

    figure, axis = plt.subplots(
        figsize=(13, 6.5)
    )

    axis.bar(
        [
            position
            - bar_width / 2
            for position in x_positions
        ],
        iou,
        width=bar_width,
        label="IoU",
    )

    axis.bar(
        [
            position
            + bar_width / 2
            for position in x_positions
        ],
        dice,
        width=bar_width,
        label="Dice",
    )

    axis.set_title(
        "CelebAMask-HQ Face Region Segmentation"
    )

    axis.set_xlabel(
        "Semantic class"
    )

    axis.set_ylabel(
        "Score (%)"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_ylim(
        0,
        100,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_markdown_table(table):
    markdown_lines = [
        "| Class | IoU (%) | Dice (%) |",
        "| --- | ---: | ---: |",
    ]

    for _, row in table.iterrows():
        markdown_lines.append(
            f"| {row['Class']} | "
            f"{row['IoU (%)']:.2f} | "
            f"{row['Dice (%)']:.2f} |"
        )

    TABLE_MD_PATH.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def create_thesis_table(dataframe):
    table = dataframe[
        [
            "class_name",
            "iou",
            "dice",
        ]
    ].copy()

    table.columns = [
        "Class",
        "IoU (%)",
        "Dice (%)",
    ]

    table["IoU (%)"] = (
        table["IoU (%)"]
        .astype(float)
        * 100.0
    )

    table["Dice (%)"] = (
        table["Dice (%)"]
        .astype(float)
        * 100.0
    )

    table["IoU (%)"] = (
        table["IoU (%)"]
        .round(2)
    )

    table["Dice (%)"] = (
        table["Dice (%)"]
        .round(2)
    )

    table.to_csv(
        TABLE_CSV_PATH,
        index=False,
    )

    create_markdown_table(
        table
    )

    return table


def create_summary_table(
    summary_metrics,
):
    table = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Value (%)": round(
                    value,
                    4,
                ),
            }
            for metric, value
            in summary_metrics.items()
        ]
    )

    table.to_csv(
        SUMMARY_TABLE_CSV_PATH,
        index=False,
    )

    markdown_lines = [
        "| Metric | Value (%) |",
        "| --- | ---: |",
    ]

    for _, row in table.iterrows():
        markdown_lines.append(
            f"| {row['Metric']} | "
            f"{row['Value (%)']:.4f} |"
        )

    SUMMARY_TABLE_MD_PATH.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    return table


def main():
    dataframe = load_metrics()

    summary_metrics = (
        load_summary_metrics()
    )

    create_figure(
        dataframe
    )

    class_table = (
        create_thesis_table(
            dataframe
        )
    )

    summary_table = (
        create_summary_table(
            summary_metrics
        )
    )

    print(
        f"Classes: {len(dataframe)}"
    )

    print()
    print(
        "=== Per-Class Thesis Table ==="
    )

    print(
        class_table.to_string(
            index=False
        )
    )

    print()
    print(
        "=== Overall Summary Table ==="
    )

    print(
        summary_table.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved figure: {FIGURE_PATH}"
    )

    print(
        f"Saved per-class CSV table: "
        f"{TABLE_CSV_PATH}"
    )

    print(
        f"Saved per-class Markdown table: "
        f"{TABLE_MD_PATH}"
    )

    print(
        f"Saved summary CSV table: "
        f"{SUMMARY_TABLE_CSV_PATH}"
    )

    print(
        f"Saved summary Markdown table: "
        f"{SUMMARY_TABLE_MD_PATH}"
    )


if __name__ == "__main__":
    main()