from pathlib import Path
import atexit
import os
import shutil
import tempfile

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



def validate_input(
    df,
):
    """Validate accepted evaluator results before generating thesis tables."""
    required_columns = {
        "split",
        "image",
        "status",
        "nme_percent",
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Evaluator result is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(df) != 600:
        raise RuntimeError(
            f"Expected 600 evaluator rows, found {len(df)}."
        )

    if set(
        df[
            "split"
        ].unique()
    ) != {
        "Indoor",
        "Outdoor",
    }:
        raise RuntimeError(
            "Unexpected evaluator split set."
        )


def validate_staged_outputs(
    output_csv,
    output_md,
):
    """Validate newly generated thesis tables before final replacement."""
    if not output_csv.is_file():
        raise RuntimeError(
            "Staged thesis-table CSV was not created."
        )

    if not output_md.is_file():
        raise RuntimeError(
            "Staged thesis-table Markdown was not created."
        )

    table = pd.read_csv(
        output_csv
    )

    expected_columns = [
        "Split",
        "Images",
        "Successful",
        "Failed",
        "Detection Rate (%)",
        "Mean NME (%)",
        "Median NME (%)",
        "Std NME (%)",
    ]

    if list(
        table.columns
    ) != expected_columns:
        raise RuntimeError(
            "Staged thesis-table CSV schema is incorrect."
        )

    if list(
        table[
            "Split"
        ]
    ) != [
        "Indoor",
        "Outdoor",
        "OVERALL",
    ]:
        raise RuntimeError(
            "Staged thesis-table split order is incorrect."
        )

    overall = table.iloc[
        2
    ]

    if int(
        overall[
            "Images"
        ]
    ) != 600:
        raise RuntimeError(
            "Staged thesis table has an incorrect overall image count."
        )

    markdown = output_md.read_text(
        encoding="utf-8"
    )

    if (
        "# 300-W Facial Landmark Validation Results"
        not in markdown
    ):
        raise RuntimeError(
            "Staged thesis-table Markdown is incomplete."
        )


def replace_owned_outputs(
    staged_csv,
    staged_md,
    final_csv,
    final_md,
    staging_dir,
):
    """Replace only table-owned outputs with rollback on commit failure."""
    pairs = [
        (staged_csv, final_csv),
        (staged_md, final_md),
    ]
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
    global OUTPUT_MD

    df = pd.read_csv(
        RESULTS_CSV
    )

    validate_input(
        df
    )

    final_output_csv = OUTPUT_CSV
    final_output_md = OUTPUT_MD

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".300w_landmark_table_",
            dir=RESULTS_DIR,
        )
    )

    atexit.register(
        shutil.rmtree,
        staging_dir,
        ignore_errors=True,
    )

    OUTPUT_CSV = (
        staging_dir
        / final_output_csv.name
    )

    OUTPUT_MD = (
        staging_dir
        / final_output_md.name
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

    print("\nValidating staged outputs...")

    try:
        validate_staged_outputs(
            OUTPUT_CSV,
            OUTPUT_MD,
        )

        replace_owned_outputs(
            OUTPUT_CSV,
            OUTPUT_MD,
            final_output_csv,
            final_output_md,
            staging_dir,
        )

    finally:
        OUTPUT_CSV = final_output_csv
        OUTPUT_MD = final_output_md

        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nCommitted final table outputs:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
