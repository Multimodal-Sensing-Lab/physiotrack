from pathlib import Path
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_CSV = (
    RESULTS_DIR
    / "celebamaskhq_class_metrics.csv"
)

CONFUSION_MATRIX_CSV = (
    RESULTS_DIR
    / "celebamaskhq_confusion_matrix.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "celebamaskhq_segmentation_summary.txt"
)

FIGURE_PATH = (
    FIGURES_DIR
    / "celebamaskhq_per_class_metrics.png"
)

CONFUSION_MATRIX_FIGURE_PATH = (
    FIGURES_DIR
    / "celebamaskhq_normalized_confusion_matrix.png"
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


def load_confusion_matrix():
    if not CONFUSION_MATRIX_CSV.exists():
        raise FileNotFoundError(
            "Confusion matrix file not found: "
            f"{CONFUSION_MATRIX_CSV}"
        )

    dataframe = pd.read_csv(
        CONFUSION_MATRIX_CSV,
        index_col=0,
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

    if dataframe.index.tolist() != expected_classes:
        raise ValueError(
            "Unexpected CelebAMask-HQ ground-truth class order "
            "in confusion matrix CSV."
        )

    if dataframe.columns.tolist() != expected_classes:
        raise ValueError(
            "Unexpected CelebAMask-HQ predicted class order "
            "in confusion matrix CSV."
        )

    dataframe = dataframe.apply(
        pd.to_numeric,
        errors="raise",
    )

    if (dataframe.to_numpy() < 0).any():
        raise ValueError(
            "Confusion matrix contains negative pixel counts."
        )

    row_sums = dataframe.sum(
        axis=1
    )

    if (row_sums <= 0).any():
        raise ValueError(
            "Confusion matrix contains an empty ground-truth class row."
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



def verify_quantitative_inputs(class_metrics, confusion_matrix, summary_metrics):
    """Reject inconsistent evaluation inputs before preparing any plot-owned outputs.

    This does not change the scientific metrics or recompute the benchmark; it
    independently checks that all three accepted input files describe the same
    complete evaluation before their values are reused in thesis artifacts.
    """
    expected_classes = confusion_matrix.index.tolist()
    if len(class_metrics) != len(expected_classes) or class_metrics["class_name"].tolist() != expected_classes:
        raise ValueError("Class metrics and confusion matrix class order disagree.")

    class_ids = pd.to_numeric(class_metrics["class_id"], errors="raise").to_numpy(dtype=float)
    if not np.array_equal(class_ids, np.arange(len(expected_classes), dtype=float)):
        raise ValueError("Class metric IDs must be exactly 0 through 18 in order.")

    matrix = confusion_matrix.to_numpy(dtype=float)
    if (matrix.shape != (19, 19) or not np.isfinite(matrix).all()
            or (matrix < 0).any() or not np.array_equal(matrix, np.floor(matrix))):
        raise ValueError("Confusion matrix must contain finite nonnegative integer counts.")

    expected_pixels = 2824 * 512 * 512
    if int(matrix.sum()) != expected_pixels:
        raise ValueError("Confusion matrix does not cover the complete 2,824-image test split.")

    gt = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    union = gt + predicted - true_positive
    dice_denominator = gt + predicted
    if (union <= 0).any() or (dice_denominator <= 0).any():
        raise ValueError("One or more semantic classes have undefined IoU/Dice.")
    computed_iou = true_positive / union
    computed_dice = 2 * true_positive / dice_denominator

    for field, reference in (("gt_pixels", gt), ("pred_pixels", predicted),
                             ("iou", computed_iou), ("dice", computed_dice)):
        actual = pd.to_numeric(class_metrics[field], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError(f"Non-finite class metrics: {field}")
        if field.endswith("pixels"):
            valid = np.array_equal(actual, reference)
        else:
            valid = np.allclose(actual, reference, rtol=0.0, atol=1e-10)
        if not valid:
            raise ValueError(f"Class-metrics {field} disagrees with the confusion matrix.")

    expected_summary = {
        "Pixel Accuracy (%)": float(true_positive.sum() / matrix.sum() * 100),
        "All-class mIoU (%)": float(computed_iou.mean() * 100),
        "Foreground mIoU (%)": float(computed_iou[1:].mean() * 100),
        "All-class Mean Dice (%)": float(computed_dice.mean() * 100),
        "Foreground Mean Dice (%)": float(computed_dice[1:].mean() * 100),
    }
    if list(summary_metrics) != list(expected_summary):
        raise ValueError("Unexpected quantitative summary metrics or order.")
    for metric, expected in expected_summary.items():
        actual = summary_metrics[metric]
        # Benchmark summary stores proportions rounded to six decimals.
        if not np.isfinite(actual) or abs(actual - expected) > 0.000051:
            raise ValueError(f"Summary {metric} disagrees with the confusion matrix.")

    recorded_counts = {}
    for line in SUMMARY_PATH.read_text(encoding="utf-8").splitlines():
        for field in ("Official test split size", "Evaluated images", "Successful images", "Failed images"):
            if line.startswith(field + ":"):
                if field in recorded_counts:
                    raise ValueError(f"Duplicate quantitative summary field: {field}")
                recorded_counts[field] = int(line.split(":", 1)[1].strip())
    if recorded_counts != {
        "Official test split size": 2824,
        "Evaluated images": 2824,
        "Successful images": 2824,
        "Failed images": 0,
    }:
        raise ValueError("Quantitative summary does not represent a complete successful test-split run.")


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


def create_confusion_matrix_figure(dataframe):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_sums = dataframe.sum(
        axis=1
    )

    normalized = dataframe.div(
        row_sums,
        axis=0,
    ) * 100.0

    labels = normalized.columns.tolist()

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    image = axis.imshow(
        normalized.to_numpy(),
        aspect="auto",
        vmin=0.0,
        vmax=100.0,
    )

    axis.set_title(
        "CelebAMask-HQ Normalized Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "Ground-truth class"
    )

    axis.set_xticks(
        range(len(labels))
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        range(len(labels))
    )

    axis.set_yticklabels(
        labels
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Row-normalized pixels (%)"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FIGURE_PATH,
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


def run_plot_generation():
    dataframe = load_metrics()

    confusion_matrix = (
        load_confusion_matrix()
    )

    summary_metrics = (
        load_summary_metrics()
    )

    verify_quantitative_inputs(dataframe, confusion_matrix, summary_metrics)

    create_figure(
        dataframe
    )

    create_confusion_matrix_figure(
        confusion_matrix
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
        f"Staged figure: {FIGURE_PATH}"
    )

    print(
        "Staged normalized confusion matrix figure: "
        f"{CONFUSION_MATRIX_FIGURE_PATH}"
    )

    print(
        f"Staged per-class CSV table: "
        f"{TABLE_CSV_PATH}"
    )

    print(
        f"Staged per-class Markdown table: "
        f"{TABLE_MD_PATH}"
    )

    print(
        f"Staged summary CSV table: "
        f"{SUMMARY_TABLE_CSV_PATH}"
    )

    print(
        f"Staged summary Markdown table: "
        f"{SUMMARY_TABLE_MD_PATH}"
    )



def validate_plot_outputs(paths):
    """Verify the complete owned artifact set, before and after publication."""
    if len(paths) != 6:
        raise RuntimeError("Unexpected number of plot-owned output paths.")
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing, unsafe, or empty plot output: {path.name}")

    class_table = pd.read_csv(paths[2])
    if list(class_table.columns) != ["Class", "IoU (%)", "Dice (%)"] or len(class_table) != 19:
        raise RuntimeError("Per-class thesis table is invalid.")

    expected_classes = [
        "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow", "r_brow",
        "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip", "hair", "eye_g", "hat",
        "ear_r", "neck_l",
    ]
    if class_table["Class"].tolist() != expected_classes:
        raise RuntimeError("Per-class thesis table class order is incorrect.")

    summary_table = pd.read_csv(paths[4])
    expected_metrics = [
        "Pixel Accuracy (%)",
        "All-class mIoU (%)",
        "Foreground mIoU (%)",
        "All-class Mean Dice (%)",
        "Foreground Mean Dice (%)",
    ]
    if list(summary_table.columns) != ["Metric", "Value (%)"] or summary_table["Metric"].tolist() != expected_metrics:
        raise RuntimeError("Summary table metric schema or order is incorrect.")

    for path in (paths[3], paths[5]):
        if not path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"Markdown table is empty: {path.name}")

    for path in (paths[0], paths[1]):
        image = plt.imread(path)
        if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise RuntimeError(f"Figure is unreadable: {path.name}")


def replace_owned_plot_outputs(staged_paths, final_paths, staging_dir):
    """Publish only these six artifacts, verify and restore prior files on failure."""
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    installed = []
    try:
        for final_path in final_paths:
            if final_path.is_symlink() or final_path.parent.is_symlink():
                raise RuntimeError(f"Unsafe plot output path: {final_path}")
            if final_path.exists():
                if not final_path.is_file():
                    raise RuntimeError(f"Plot-owned output is not a regular file: {final_path}")
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
        for staged_path, final_path in zip(staged_paths, final_paths):
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, final_path)
            installed.append(final_path)
        validate_plot_outputs(final_paths)
    except BaseException:
        try:
            for final_path in installed:
                if final_path.exists():
                    final_path.unlink()
            for backup_path, final_path in reversed(backups):
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if backup_path.exists():
                    os.replace(backup_path, final_path)
        except BaseException as restore_error:
            # Keep any remaining backups until manually recovered.
            raise RuntimeError(
                f"Plot publication failed and rollback was incomplete; "
                f"inspect the preserved backup directory: {backup_dir}"
            ) from restore_error
        raise
    shutil.rmtree(backup_dir)


def main():
    """Generate quantitative tables and figures transactionally."""
    global FIGURES_DIR, FIGURE_PATH, CONFUSION_MATRIX_FIGURE_PATH
    global TABLE_CSV_PATH, TABLE_MD_PATH, SUMMARY_TABLE_CSV_PATH, SUMMARY_TABLE_MD_PATH

    final_figures_dir = FIGURES_DIR
    final_paths = [
        FIGURE_PATH,
        CONFUSION_MATRIX_FIGURE_PATH,
        TABLE_CSV_PATH,
        TABLE_MD_PATH,
        SUMMARY_TABLE_CSV_PATH,
        SUMMARY_TABLE_MD_PATH,
    ]
    staging_dir = Path(tempfile.mkdtemp(prefix=".celebamaskhq_segmentation_plot_", dir=RESULTS_DIR))
    FIGURES_DIR = staging_dir / "figures"
    FIGURE_PATH = FIGURES_DIR / final_paths[0].name
    CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / final_paths[1].name
    TABLE_CSV_PATH = staging_dir / final_paths[2].name
    TABLE_MD_PATH = staging_dir / final_paths[3].name
    SUMMARY_TABLE_CSV_PATH = staging_dir / final_paths[4].name
    SUMMARY_TABLE_MD_PATH = staging_dir / final_paths[5].name
    staged_paths = [
        FIGURE_PATH,
        CONFUSION_MATRIX_FIGURE_PATH,
        TABLE_CSV_PATH,
        TABLE_MD_PATH,
        SUMMARY_TABLE_CSV_PATH,
        SUMMARY_TABLE_MD_PATH,
    ]
    try:
        run_plot_generation()
        print()
        print("Validating staged plot outputs...")
        validate_plot_outputs(staged_paths)
        replace_owned_plot_outputs(staged_paths, final_paths, staging_dir)
    finally:
        FIGURES_DIR = final_figures_dir
        FIGURE_PATH, CONFUSION_MATRIX_FIGURE_PATH, TABLE_CSV_PATH, TABLE_MD_PATH, SUMMARY_TABLE_CSV_PATH, SUMMARY_TABLE_MD_PATH = final_paths
        backup_dir = staging_dir / "backup"
        if backup_dir.exists() and any(backup_dir.iterdir()):
            print("WARNING: plot recovery backups were preserved at", backup_dir)
        elif staging_dir.exists():
            shutil.rmtree(staging_dir)
    print()
    print("Committed final plot and table outputs.")
    for published_path in final_paths:
        print("Published:", published_path.relative_to(SCRIPT_DIR).as_posix())


if __name__ == "__main__":
    main()