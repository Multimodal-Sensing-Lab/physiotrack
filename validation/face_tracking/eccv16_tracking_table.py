from pathlib import Path

import pandas as pd


RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

INPUT_CSV = (
    RESULTS_DIR
    / "eccv16_tracking_results.csv"
)

OUTPUT_CSV = (
    RESULTS_DIR
    / "eccv16_tracking_thesis_table.csv"
)

OUTPUT_MD = (
    RESULTS_DIR
    / "eccv16_tracking_thesis_table.md"
)


def main():
    """Create a compact thesis-ready table from the tracking results."""
    df = pd.read_csv(
        INPUT_CSV
    )

    table = df[
        [
            "Video",
            "Recall_percent",
            "Precision_percent",
            "F1_percent",
            "FAF",
            "IDS",
            "Frag",
            "MOTA_percent",
            "MOTP_IoU_percent",
            "IDF1_percent",
        ]
    ].copy()

    table.columns = [
        "Video",
        "Recall (%)",
        "Precision (%)",
        "F1 (%)",
        "FAF",
        "IDS",
        "Frag",
        "MOTA (%)",
        "MOTP (%)",
        "IDF1 (%)",
    ]

    percentage_columns = [
        "Recall (%)",
        "Precision (%)",
        "F1 (%)",
        "MOTA (%)",
        "MOTP (%)",
        "IDF1 (%)",
    ]

    for column in percentage_columns:
        table[column] = (
            table[column].map(
                lambda value: (
                    f"{float(value):.2f}"
                )
            )
        )

    table["FAF"] = (
        table["FAF"].map(
            lambda value: (
                f"{float(value):.4f}"
            )
        )
    )

    table["IDS"] = (
        table["IDS"].astype(int)
    )

    table["Frag"] = (
        table["Frag"].astype(int)
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
            "# ECCV 2016 Face Tracking "
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