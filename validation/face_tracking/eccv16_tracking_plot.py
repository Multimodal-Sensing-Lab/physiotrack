from pathlib import Path
import os
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

CSV_PATH = (
    RESULTS_DIR
    / "eccv16_tracking_results.csv"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

OUTPUT_PATH = (
    FIGURES_DIR
    / "eccv16_tracking_metrics.png"
)


def main():
    """Create a summary figure for the ECCV 2016 tracking results."""
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    required_columns = [
        "Video",
        "F1_percent",
        "MOTA_percent",
        "MOTP_IoU_percent",
        "IDF1_percent",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Tracking result table is missing required columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise RuntimeError(
            "Tracking result table is empty."
        )

    videos = df["Video"]

    metrics = {
        "F1": df["F1_percent"],
        "MOTA": df["MOTA_percent"],
        "MOTP": df[
            "MOTP_IoU_percent"
        ],
        "IDF1": df["IDF1_percent"],
    }

    x = range(len(videos))
    width = 0.2

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    offsets = [
        -1.5,
        -0.5,
        0.5,
        1.5,
    ]

    for (
        offset,
        (label, values),
    ) in zip(
        offsets,
        metrics.items(),
    ):
        positions = [
            i + offset * width
            for i in x
        ]

        ax.bar(
            positions,
            values,
            width,
            label=label,
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_title(
        "PhysioTrack Face Tracking "
        "Performance on ECCV 2016 "
        "Music Videos"
    )

    ax.set_xlabel("Video")

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        videos,
        rotation=35,
        ha="right",
    )

    ax.legend(
        title="Metric",
        ncols=4,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".eccv16_tracking_plot_",
            dir=FIGURES_DIR,
        )
    )

    staged_output = (
        staging_dir
        / OUTPUT_PATH.name
    )

    try:
        try:
            fig.savefig(
                staged_output,
                dpi=300,
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)

        image = cv2.imread(
            str(staged_output)
        )

        if (
            image is None
            or image.size == 0
        ):
            raise RuntimeError(
                "Staged tracking figure validation failed."
            )

        backup_path = (
            staging_dir
            / (
                OUTPUT_PATH.name
                + ".backup"
            )
        )

        if OUTPUT_PATH.exists():
            os.replace(
                OUTPUT_PATH,
                backup_path,
            )

        try:
            os.replace(
                staged_output,
                OUTPUT_PATH,
            )
        except Exception:
            if backup_path.exists():
                os.replace(
                    backup_path,
                    OUTPUT_PATH,
                )
            raise

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

    print("Loaded results:")
    print(CSV_PATH)

    print("\nSaved figure:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
