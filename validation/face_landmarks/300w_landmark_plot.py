from pathlib import Path
import csv

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


def main():
    df = pd.read_csv(
        INPUT_CSV
    )

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
    print(OUTPUT_FIGURE)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()