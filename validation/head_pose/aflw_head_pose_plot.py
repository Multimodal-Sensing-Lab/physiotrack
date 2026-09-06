from pathlib import Path
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_CSV = RESULTS_DIR / "aflw_head_pose_results.csv"

THESIS_TABLE_CSV = RESULTS_DIR / "aflw_head_pose_thesis_table.csv"
THESIS_TABLE_MD = RESULTS_DIR / "aflw_head_pose_thesis_table.md"
FIGURE_PATH = FIGURES_DIR / "aflw_head_pose_error_metrics.png"
DISTRIBUTION_FIGURE_PATH = (
    FIGURES_DIR / "aflw_head_pose_error_distribution.png"
)

REQUIRED_COLUMNS = {
    "status",
    "yaw_error",
    "pitch_error",
    "roll_error",
}


def load_successful_results():
    """Load and validate successful AFLW head-pose evaluation results."""
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            f"Evaluation results were not found: {RESULTS_CSV}"
        )

    data = pd.read_csv(RESULTS_CSV)

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise RuntimeError(
            "AFLW result CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    successful = data[data["status"] == "ok"].copy()

    if successful.empty:
        raise RuntimeError(
            "No successful AFLW head-pose results were found."
        )

    error_columns = ["yaw_error", "pitch_error", "roll_error"]
    error_values = successful[error_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(error_values)):
        raise RuntimeError(
            "Successful AFLW rows contain non-finite angular errors."
        )

    if np.any(error_values < 0.0) or np.any(error_values > 180.0):
        raise RuntimeError(
            "Successful AFLW rows contain angular errors outside [0, 180]."
        )

    return data, successful


def calculate_metrics(data):
    """Calculate final thesis-level head-pose metrics."""
    axis_columns = {
        "Yaw": "yaw_error",
        "Pitch": "pitch_error",
        "Roll": "roll_error",
    }

    rows = []

    for axis, column in axis_columns.items():
        errors = data[column].to_numpy(dtype=float)

        rows.append(
            {
                "Axis": axis,
                "MAE (degrees)": float(np.mean(errors)),
                "Median absolute error (degrees)": float(
                    np.median(errors)
                ),
                "Std. absolute error (degrees)": float(np.std(errors)),
            }
        )

    table = pd.DataFrame(rows)

    all_errors = np.concatenate(
        [
            data["yaw_error"].to_numpy(dtype=float),
            data["pitch_error"].to_numpy(dtype=float),
            data["roll_error"].to_numpy(dtype=float),
        ]
    )

    overall_metrics = {
        "MAE (degrees)": float(np.mean(all_errors)),
        "Median absolute error (degrees)": float(np.median(all_errors)),
        "Std. absolute error (degrees)": float(np.std(all_errors)),
        "Total axis errors": int(len(all_errors)),
    }

    return table, overall_metrics


def save_thesis_table(
    table,
    overall_metrics,
    csv_path=THESIS_TABLE_CSV,
    md_path=THESIS_TABLE_MD,
):
    """Save thesis-ready head-pose summary tables."""
    output = table.copy()

    numeric_columns = [
        "MAE (degrees)",
        "Median absolute error (degrees)",
        "Std. absolute error (degrees)",
    ]

    output[numeric_columns] = output[numeric_columns].round(4)

    overall_row = pd.DataFrame(
        [
            {
                "Axis": "Overall",
                "MAE (degrees)": round(
                    overall_metrics["MAE (degrees)"],
                    4,
                ),
                "Median absolute error (degrees)": round(
                    overall_metrics["Median absolute error (degrees)"],
                    4,
                ),
                "Std. absolute error (degrees)": round(
                    overall_metrics["Std. absolute error (degrees)"],
                    4,
                ),
            }
        ]
    )

    output = pd.concat([output, overall_row], ignore_index=True)

    output.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as file:
        file.write(
            "| Axis | MAE (degrees) | "
            "Median absolute error (degrees) | "
            "Std. absolute error (degrees) |\n"
        )
        file.write("|---|---:|---:|---:|\n")

        for _, row in output.iterrows():
            file.write(
                f"| {row['Axis']} | "
                f"{row['MAE (degrees)']:.4f} | "
                f"{row['Median absolute error (degrees)']:.4f} | "
                f"{row['Std. absolute error (degrees)']:.4f} |\n"
            )

    return output


def create_figure(table, output_path=FIGURE_PATH):
    """Create a thesis-ready head-pose error figure."""
    axes = table["Axis"].tolist()
    mae = table["MAE (degrees)"].to_numpy(dtype=float)
    median = table["Median absolute error (degrees)"].to_numpy(dtype=float)

    positions = np.arange(len(axes))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.bar(
        positions - width / 2,
        mae,
        width,
        label="MAE",
    )
    axis.bar(
        positions + width / 2,
        median,
        width,
        label="Median absolute error",
    )

    axis.set_xlabel("Head-pose axis")
    axis.set_ylabel("Absolute angular error (degrees)")
    axis.set_title("AFLW Head Pose Validation")
    axis.set_xticks(positions)
    axis.set_xticklabels(axes)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)



def create_distribution_figure(
    data,
    output_path=DISTRIBUTION_FIGURE_PATH,
):
    """Create an empirical CDF of absolute angular errors by pose axis."""
    axis_columns = {
        "Yaw": "yaw_error",
        "Pitch": "pitch_error",
        "Roll": "roll_error",
    }

    figure, axis = plt.subplots(figsize=(8, 5))

    for label, column in axis_columns.items():
        errors = np.sort(
            data[column].to_numpy(dtype=float)
        )

        cumulative = (
            np.arange(1, len(errors) + 1, dtype=float)
            / len(errors)
            * 100.0
        )

        axis.plot(
            errors,
            cumulative,
            label=label,
        )

    axis.set_xlabel("Absolute angular error (degrees)")
    axis.set_ylabel("Cumulative successful samples (%)")
    axis.set_title("AFLW Head Pose Absolute-Error Distribution")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 100.0)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def validate_staged_plot_outputs(
    table_csv,
    table_md,
    figure_path,
    distribution_path,
    successful,
):
    """Validate staged tables and figures before replacing final outputs."""
    expected_table, expected_overall = calculate_metrics(successful)

    staged_table = pd.read_csv(table_csv)

    expected_columns = [
        "Axis",
        "MAE (degrees)",
        "Median absolute error (degrees)",
        "Std. absolute error (degrees)",
    ]

    if staged_table.columns.tolist() != expected_columns:
        raise RuntimeError(
            "Staged thesis table columns do not match the expected schema."
        )

    if staged_table["Axis"].tolist() != [
        "Yaw",
        "Pitch",
        "Roll",
        "Overall",
    ]:
        raise RuntimeError(
            "Staged thesis table axis order is incorrect."
        )

    expected_output = expected_table.copy()
    numeric_columns = expected_columns[1:]
    expected_output[numeric_columns] = expected_output[numeric_columns].round(4)
    expected_output = pd.concat(
        [
            expected_output,
            pd.DataFrame(
                [
                    {
                        "Axis": "Overall",
                        "MAE (degrees)": round(
                            expected_overall["MAE (degrees)"],
                            4,
                        ),
                        "Median absolute error (degrees)": round(
                            expected_overall[
                                "Median absolute error (degrees)"
                            ],
                            4,
                        ),
                        "Std. absolute error (degrees)": round(
                            expected_overall[
                                "Std. absolute error (degrees)"
                            ],
                            4,
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    if staged_table["Axis"].tolist() != expected_output["Axis"].tolist():
        raise RuntimeError(
            "Staged thesis table labels are inconsistent with recomputed metrics."
        )

    if not np.allclose(
        staged_table[numeric_columns].to_numpy(dtype=float),
        expected_output[numeric_columns].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-5,
    ):
        raise RuntimeError(
            "Staged thesis table values are inconsistent with the evaluator CSV."
        )

    markdown = table_md.read_text(encoding="utf-8")
    for _, row in expected_output.iterrows():
        expected_line = (
            f"| {row['Axis']} | "
            f"{row['MAE (degrees)']:.4f} | "
            f"{row['Median absolute error (degrees)']:.4f} | "
            f"{row['Std. absolute error (degrees)']:.4f} |"
        )

        if expected_line not in markdown:
            raise RuntimeError(
                "Staged Markdown thesis table is inconsistent with the CSV."
            )

    for image_path in [figure_path, distribution_path]:
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged figure is missing or empty: {image_path}"
            )

        image = plt.imread(image_path)
        if image.size == 0 or not np.all(np.isfinite(image)):
            raise RuntimeError(
                f"Staged figure could not be validated: {image_path}"
            )


def replace_owned_outputs(staging_dir):
    """Replace only plot-script-owned outputs with rollback protection."""
    final_paths = [
        THESIS_TABLE_CSV,
        THESIS_TABLE_MD,
        FIGURE_PATH,
        DISTRIBUTION_FIGURE_PATH,
    ]
    staged_paths = [
        staging_dir / THESIS_TABLE_CSV.name,
        staging_dir / THESIS_TABLE_MD.name,
        staging_dir / FIGURE_PATH.name,
        staging_dir / DISTRIBUTION_FIGURE_PATH.name,
    ]

    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backups = []
    installed = []

    try:
        for final_path in final_paths:
            final_path.parent.mkdir(parents=True, exist_ok=True)

            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))

        for staged_path, final_path in zip(staged_paths, final_paths):
            os.replace(staged_path, final_path)
            installed.append(final_path)

    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()

        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, final_path)

        raise


def main():
    all_rows, successful = load_successful_results()
    table, overall_metrics = calculate_metrics(successful)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".aflw_head_pose_plot_",
            dir=RESULTS_DIR,
        )
    )

    staged_table_csv = staging_dir / THESIS_TABLE_CSV.name
    staged_table_md = staging_dir / THESIS_TABLE_MD.name
    staged_figure = staging_dir / FIGURE_PATH.name
    staged_distribution = staging_dir / DISTRIBUTION_FIGURE_PATH.name

    print("Staging directory:", staging_dir)

    try:
        output_table = save_thesis_table(
            table,
            overall_metrics,
            staged_table_csv,
            staged_table_md,
        )
        create_figure(table, staged_figure)
        create_distribution_figure(successful, staged_distribution)

        print("Validating staged plot outputs...")
        validate_staged_plot_outputs(
            staged_table_csv,
            staged_table_md,
            staged_figure,
            staged_distribution,
            successful,
        )

        replace_owned_outputs(staging_dir)
        print("Committed final plot and table outputs.")

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("Evaluation rows:", len(all_rows))
    print("Successful samples:", len(successful))
    print(
        "Total pooled axis errors:",
        overall_metrics["Total axis errors"],
    )
    print()
    print("=== AFLW Head Pose Thesis Table ===")
    print(output_table.to_string(index=False))
    print()
    print("Overall pooled metrics:")
    print(
        f"MAE: {overall_metrics['MAE (degrees)']:.4f} degrees"
    )
    print(
        "Median absolute error: "
        f"{overall_metrics['Median absolute error (degrees)']:.4f} degrees"
    )
    print(
        "Std. absolute error: "
        f"{overall_metrics['Std. absolute error (degrees)']:.4f} degrees"
    )
    print()
    print("Saved:")
    print(THESIS_TABLE_CSV)
    print(THESIS_TABLE_MD)
    print(FIGURE_PATH)
    print(DISTRIBUTION_FIGURE_PATH)


if __name__ == "__main__":
    main()
