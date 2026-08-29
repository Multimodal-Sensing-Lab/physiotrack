from pathlib import Path

import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

RESULTS_CSV = (
    RESULTS_DIR
    / "300w_landmark_results.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "300w_landmark_thesis_table.csv"
)

OUTPUT_MD = (
    RESULTS_DIR
    / "300w_landmark_thesis_table.md"
)


def summarize(
    df,
    split_name,
):
    """Summarize one 300-W evaluation split."""
    if split_name == "OVERALL":
        subset = df
    else:
        subset = df[
            df["split"]
            == split_name
        ]

    successful = subset[
        subset["status"]
        == "ok"
    ]

    images = len(subset)
    successful_count = len(
        successful
    )

    failed = (
        images
        - successful_count
    )

    detection_rate = (
        successful_count
        / images
        * 100.0
        if images > 0
        else 0.0
    )

    return {
        "Split": split_name,
        "Images": images,
        "Successful": successful_count,
        "Failed": failed,
        "Detection Rate (%)": (
            detection_rate
        ),
        "Mean NME (%)": successful[
            "nme_percent"
        ].mean(),
        "Median NME (%)": successful[
            "nme_percent"
        ].median(),
        "Std NME (%)": successful[
            "nme_percent"
        ].std(
            ddof=0
        ),
    }


def main():
    df = pd.read_csv(
        RESULTS_CSV
    )

    rows = [
        summarize(
            df,
            "Indoor",
        ),
        summarize(
            df,
            "Outdoor",
        ),
        summarize(
            df,
            "OVERALL",
        ),
    ]

    table = pd.DataFrame(
        rows
    )

    numeric_columns = [
        "Detection Rate (%)",
        "Mean NME (%)",
        "Median NME (%)",
        "Std NME (%)",
    ]

    for column in numeric_columns:
        table[column] = (
            table[column].map(
                lambda value: (
                    f"{float(value):.2f}"
                )
            )
        )

    table.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# 300-W Facial Landmark "
            "Validation Results\n\n"
        )

        columns = list(
            table.columns
        )

        file.write(
            "| "
            + " | ".join(columns)
            + " |\n"
        )

        file.write(
            "| "
            + " | ".join(
                ["---"]
                * len(columns)
            )
            + " |\n"
        )

        for _, row in table.iterrows():
            values = [
                str(row[column])
                for column in columns
            ]

            file.write(
                "| "
                + " | ".join(values)
                + " |\n"
            )

    print("Thesis table:\n")

    print(
        table.to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()