from pathlib import Path

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


def save_thesis_table(table, overall_metrics):
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

    output.to_csv(THESIS_TABLE_CSV, index=False)

    with open(THESIS_TABLE_MD, "w", encoding="utf-8") as file:
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


def create_figure(table):
    """Create a thesis-ready head-pose error figure."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    figure.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    all_rows, successful = load_successful_results()
    table, overall_metrics = calculate_metrics(successful)
    output_table = save_thesis_table(table, overall_metrics)
    create_figure(table)

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


if __name__ == "__main__":
    main()
