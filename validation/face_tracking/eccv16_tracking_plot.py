from pathlib import Path

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

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    FIGURES_DIR
    / "eccv16_tracking_metrics.png"
)


def main():
    """Create a summary figure for the ECCV 2016 tracking results."""
    df = pd.read_csv(
        CSV_PATH
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

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Loaded results:")
    print(CSV_PATH)

    print("\nSaved figure:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()