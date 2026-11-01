from pathlib import Path
import atexit
import csv
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

INPUT_CSV = (
    RESULTS_DIR
    / "300w_landmark_results.csv"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FIGURE = (
    FIGURES_DIR
    / "300w_landmark_ced.png"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "300w_landmark_ced.csv"
)

MAX_NME_PERCENT = 10.0
NUM_THRESHOLDS = 1001


def compute_ced(
    df,
    thresholds,
):
    """Compute CED over all images, including failed predictions."""
    total_images = len(df)

    valid_errors = df.loc[
        df["status"] == "ok",
        "nme_percent",
    ].to_numpy(
        dtype=float
    )

    ced = np.asarray(
        [
            np.sum(
                valid_errors
                <= threshold
            )
            / total_images
            for threshold
            in thresholds
        ],
        dtype=float,
    )

    return ced



def validate_input(
    df,
):
    """Validate accepted evaluator results before generating CED outputs."""
    required_columns = {
        "split",
        "status",
        "nme_percent",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(
            "Evaluator result is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if len(df) != 600:
        raise RuntimeError(
            f"Expected 600 evaluator rows, found {len(df)}."
        )


def validate_staged_outputs(
    output_csv,
    output_figure,
):
    """Validate newly generated CED outputs before final replacement."""
    if not output_csv.is_file():
        raise RuntimeError("Staged CED CSV was not created.")
    if not output_figure.is_file():
        raise RuntimeError("Staged CED figure was not created.")
    ced = pd.read_csv(output_csv)
    expected_columns = [
        "NME_threshold_percent",
        "Indoor_fraction",
        "Outdoor_fraction",
        "Overall_fraction",
    ]
    if list(ced.columns) != expected_columns:
        raise RuntimeError("Staged CED CSV schema is incorrect.")
    if len(ced) != NUM_THRESHOLDS:
        raise RuntimeError("Staged CED CSV has an incorrect threshold count.")
    expected_thresholds = np.linspace(0.0, MAX_NME_PERCENT, NUM_THRESHOLDS)
    thresholds = ced["NME_threshold_percent"].to_numpy(dtype=float)
    if not np.allclose(thresholds, expected_thresholds, rtol=0.0, atol=1e-12):
        raise RuntimeError("Staged CED threshold grid is incorrect.")
    for column in ("Indoor_fraction", "Outdoor_fraction", "Overall_fraction"):
        values = ced[column].to_numpy(dtype=float)
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(f"Staged CED values are outside [0, 1] in {column}.")
        if np.any(np.diff(values) < -1e-12):
            raise RuntimeError(f"Staged CED curve is not monotonic in {column}.")
    figure = plt.imread(output_figure)
    if figure.ndim < 2 or figure.shape[0] <= 0 or figure.shape[1] <= 0:
        raise RuntimeError("Staged CED figure is unreadable.")


def replace_owned_outputs(
    staged_csv, staged_figure, final_csv, final_figure, staging_dir,
):
    """Replace only plot-owned outputs with rollback on commit failure."""
    pairs = [(staged_csv, final_csv), (staged_figure, final_figure)]
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    installed = []
    try:
        for _, final_path in pairs:
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
        for staged_path, final_path in pairs:
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
    global OUTPUT_CSV
    global OUTPUT_FIGURE

    df = pd.read_csv(
        INPUT_CSV
    )

    validate_input(
        df
    )

    final_output_csv = OUTPUT_CSV
    final_output_figure = OUTPUT_FIGURE

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".300w_landmark_plot_",
            dir=RESULTS_DIR,
        )
    )

    atexit.register(
        shutil.rmtree,
        staging_dir,
        ignore_errors=True,
    )

    OUTPUT_CSV = staging_dir / final_output_csv.name
    OUTPUT_FIGURE = staging_dir / final_output_figure.name

    thresholds = np.linspace(
        0.0,
        MAX_NME_PERCENT,
        NUM_THRESHOLDS,
    )

    indoor_df = df[
        df["split"] == "Indoor"
    ]

    outdoor_df = df[
        df["split"] == "Outdoor"
    ]

    indoor_ced = compute_ced(
        indoor_df,
        thresholds,
    )

    outdoor_ced = compute_ced(
        outdoor_df,
        thresholds,
    )

    overall_ced = compute_ced(
        df,
        thresholds,
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "NME_threshold_percent",
                "Indoor_fraction",
                "Outdoor_fraction",
                "Overall_fraction",
            ]
        )

        for (
            threshold,
            indoor,
            outdoor,
            overall,
        ) in zip(
            thresholds,
            indoor_ced,
            outdoor_ced,
            overall_ced,
        ):
            writer.writerow(
                [
                    f"{threshold:.4f}",
                    f"{indoor:.6f}",
                    f"{outdoor:.6f}",
                    f"{overall:.6f}",
                ]
            )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        thresholds,
        indoor_ced * 100.0,
        label="Indoor",
        linewidth=2,
    )

    ax.plot(
        thresholds,
        outdoor_ced * 100.0,
        label="Outdoor",
        linewidth=2,
    )

    ax.plot(
        thresholds,
        overall_ced * 100.0,
        label="Overall",
        linewidth=2,
    )

    ax.set_title(
        "300-W 51-Point Facial "
        "Landmark Localization"
    )

    ax.set_xlabel(
        "Normalized Mean Error "
        "(NME, %)"
    )

    ax.set_ylabel(
        "Images with Error "
        "Below Threshold (%)"
    )

    ax.set_xlim(
        0,
        MAX_NME_PERCENT,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("CED summary:")

    for threshold in [
        5.0,
        6.0,
        8.0,
        10.0,
    ]:
        index = int(
            np.argmin(
                np.abs(
                    thresholds
                    - threshold
                )
            )
        )

        print(
            f"\nNME <= "
            f"{threshold:.1f}%"
        )

        print(
            f"Indoor:  "
            f"{indoor_ced[index] * 100:.2f}%"
        )

        print(
            f"Outdoor: "
            f"{outdoor_ced[index] * 100:.2f}%"
        )

        print(
            f"Overall: "
            f"{overall_ced[index] * 100:.2f}%"
        )

    print("\nSaved:")
    print("\nValidating staged outputs...")

    try:
        validate_staged_outputs(OUTPUT_CSV, OUTPUT_FIGURE)
        replace_owned_outputs(
            OUTPUT_CSV, OUTPUT_FIGURE,
            final_output_csv, final_output_figure, staging_dir,
        )
    finally:
        OUTPUT_CSV = final_output_csv
        OUTPUT_FIGURE = final_output_figure
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nCommitted final plot outputs:")
    print(OUTPUT_FIGURE)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
