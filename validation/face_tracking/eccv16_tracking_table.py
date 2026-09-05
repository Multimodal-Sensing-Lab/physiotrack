from pathlib import Path
import os
import shutil
import tempfile

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


def validate_input():
    """Validate the quantitative tracking results before generating tables."""
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    required_columns = [
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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Input result table is missing required columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise RuntimeError(
            "Input result table is empty."
        )

    if "OVERALL" not in set(
        df["Video"].astype(str)
    ):
        raise RuntimeError(
            "Input result table does not contain the OVERALL row."
        )

    return df


def replace_owned_outputs(
    staged_csv,
    staged_md,
    staging_dir,
):
    """Replace only table-owned outputs with rollback on failure."""
    outputs = [
        (staged_csv, OUTPUT_CSV),
        (staged_md, OUTPUT_MD),
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backups = []
    installed = []

    try:
        for _, final_path in outputs:
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                backups.append(
                    (
                        backup_path,
                        final_path,
                    )
                )

        for staged_path, final_path in outputs:
            os.replace(
                staged_path,
                final_path,
            )

            installed.append(
                final_path
            )

    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()

        for backup_path, final_path in reversed(
            backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main():
    """Create a compact thesis-ready table from the tracking results."""
    df = validate_input()

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

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".eccv16_tracking_table_",
            dir=RESULTS_DIR,
        )
    )

    staged_csv = (
        staging_dir
        / OUTPUT_CSV.name
    )

    staged_md = (
        staging_dir
        / OUTPUT_MD.name
    )

    try:
        table.to_csv(
            staged_csv,
            index=False,
            encoding="utf-8",
        )

        with open(
            staged_md,
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

        staged_table = pd.read_csv(
            staged_csv
        )

        if (
            len(staged_table)
            != len(table)
            or list(staged_table.columns)
            != list(table.columns)
        ):
            raise RuntimeError(
                "Staged thesis CSV validation failed."
            )

        markdown_text = (
            staged_md.read_text(
                encoding="utf-8"
            )
        )

        if (
            not markdown_text.strip()
            or "OVERALL"
            not in markdown_text
        ):
            raise RuntimeError(
                "Staged thesis Markdown validation failed."
            )

        replace_owned_outputs(
            staged_csv,
            staged_md,
            staging_dir,
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
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
