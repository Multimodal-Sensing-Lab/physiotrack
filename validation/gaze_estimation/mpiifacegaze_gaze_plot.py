from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

PER_PERSON_CSV_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_per_person.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_summary.txt"
)

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_PATH = (
    FIGURES_DIR
    / "mpiifacegaze_ethxgaze_per_person_mae.png"
)

TABLE_CSV_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_thesis_table.csv"
)

TABLE_MD_PATH = (
    RESULTS_DIR
    / "mpiifacegaze_ethxgaze_thesis_table.md"
)


def read_summary_metrics(
    summary_path: Path,
) -> dict[str, float | int]:
    metrics = {}

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("Total annotations:"):
                metrics["total_annotations"] = int(
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith(
                "Successful predictions:"
            ):
                metrics[
                    "successful_predictions"
                ] = int(
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith(
                "Face detection failures:"
            ):
                metrics[
                    "face_detection_failures"
                ] = int(
                    line.split(":", 1)[1].strip()
                )

            elif line.startswith(
                "Mean angular error:"
            ):
                metrics[
                    "mean_angular_error_deg"
                ] = float(
                    line.split(":", 1)[1]
                    .replace("deg", "")
                    .strip()
                )

            elif line.startswith(
                "Median angular error:"
            ):
                metrics[
                    "median_angular_error_deg"
                ] = float(
                    line.split(":", 1)[1]
                    .replace("deg", "")
                    .strip()
                )

            elif line.startswith(
                "Std angular error:"
            ):
                metrics[
                    "std_angular_error_deg"
                ] = float(
                    line.split(":", 1)[1]
                    .replace("deg", "")
                    .strip()
                )

    required_metrics = [
        "total_annotations",
        "successful_predictions",
        "face_detection_failures",
        "mean_angular_error_deg",
        "median_angular_error_deg",
        "std_angular_error_deg",
    ]

    missing_metrics = [
        metric
        for metric in required_metrics
        if metric not in metrics
    ]

    if missing_metrics:
        raise ValueError(
            "Missing summary metrics: "
            + ", ".join(missing_metrics)
        )

    return metrics


data = pd.read_csv(
    PER_PERSON_CSV_PATH
)

summary = read_summary_metrics(
    SUMMARY_PATH
)


required_columns = [
    "participant",
    "annotations",
    "successful_predictions",
    "face_detection_failures",
    "mean_angular_error_deg",
    "median_angular_error_deg",
    "std_angular_error_deg",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


participants = data["participant"]
mean_errors = data[
    "mean_angular_error_deg"
]

overall_mean = float(
    summary["mean_angular_error_deg"]
)


plt.figure(
    figsize=(12, 6)
)

plt.bar(
    participants,
    mean_errors,
)

plt.axhline(
    overall_mean,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Overall mean = "
        f"{overall_mean:.2f}°"
    ),
)

plt.xlabel(
    "Participant"
)

plt.ylabel(
    "Mean Angular Error (degrees)"
)

plt.title(
    "ETH-XGaze Cross-Dataset Evaluation "
    "on MPIIFaceGaze"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


table = data[
    [
        "participant",
        "annotations",
        "successful_predictions",
        "face_detection_failures",
        "mean_angular_error_deg",
        "median_angular_error_deg",
        "std_angular_error_deg",
    ]
].copy()


table.columns = [
    "Participant",
    "Annotations",
    "Successful Predictions",
    "Face Detection Failures",
    "Mean Angular Error (deg)",
    "Median Angular Error (deg)",
    "Std Angular Error (deg)",
]


numeric_error_columns = [
    "Mean Angular Error (deg)",
    "Median Angular Error (deg)",
    "Std Angular Error (deg)",
]

table[
    numeric_error_columns
] = table[
    numeric_error_columns
].round(4)


overall_row = pd.DataFrame(
    [
        {
            "Participant": "Overall",
            "Annotations": int(
                summary[
                    "total_annotations"
                ]
            ),
            "Successful Predictions": int(
                summary[
                    "successful_predictions"
                ]
            ),
            "Face Detection Failures": int(
                summary[
                    "face_detection_failures"
                ]
            ),
            "Mean Angular Error (deg)": round(
                float(
                    summary[
                        "mean_angular_error_deg"
                    ]
                ),
                4,
            ),
            "Median Angular Error (deg)": round(
                float(
                    summary[
                        "median_angular_error_deg"
                    ]
                ),
                4,
            ),
            "Std Angular Error (deg)": round(
                float(
                    summary[
                        "std_angular_error_deg"
                    ]
                ),
                4,
            ),
        }
    ]
)


table = pd.concat(
    [
        table,
        overall_row,
    ],
    ignore_index=True,
)


table.to_csv(
    TABLE_CSV_PATH,
    index=False,
)


table.to_markdown(
    TABLE_MD_PATH,
    index=False,
)


print(
    f"Saved figure: {FIGURE_PATH}"
)

print(
    f"Saved table CSV: {TABLE_CSV_PATH}"
)

print(
    f"Saved table Markdown: {TABLE_MD_PATH}"
)

print()

print("Overall row:")

print(
    table.tail(1).to_string(
        index=False
    )
)