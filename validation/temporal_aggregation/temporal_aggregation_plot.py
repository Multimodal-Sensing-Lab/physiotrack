from pathlib import Path
import csv
import math
import os
import shutil
import tempfile

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_RESULTS_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_results.csv"
)
INPUT_METRICS_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_metrics.csv"
)

TABLE_CSV_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_thesis_table.csv"
)
TABLE_MD_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_thesis_table.md"
)
COVERAGE_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_aggregation_validation_coverage.png"
)
ERROR_FIGURE_PATH = (
    FIGURES_DIR
    / "temporal_aggregation_numeric_error.png"
)

EXPECTED_RESULT_FIELDS = (
    "case",
    "check",
    "expected",
    "observed",
    "absolute_error",
    "status",
    "notes",
)

TABLE_FIELDS = (
    "validation_case",
    "total_checks",
    "passed_checks",
    "failed_checks",
    "pass_rate",
    "numeric_checks",
    "max_numeric_absolute_error",
)

CASE_LABELS = {
    "availability_and_non_finite": (
        "Availability and\nnon-finite handling"
    ),
    "constructor_and_window": (
        "Constructor and\nwindow derivation"
    ),
    "empty_numeric_feature": (
        "Empty numeric\nfeatures"
    ),
    "full_numeric_summary": (
        "Full numeric\nsummary"
    ),
    "person_isolation": (
        "Per-person\nisolation"
    ),
    "rejected_updates": (
        "Rejected\nupdates"
    ),
    "reset_behavior": (
        "Reset\nbehavior"
    ),
    "sliding_window": (
        "Sliding\nwindow"
    ),
}


def read_results():
    """Load and validate the accepted evaluator result table."""
    if not INPUT_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Required evaluator result file not found: "
            f"{INPUT_RESULTS_PATH}"
        )

    if not INPUT_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Required evaluator metrics file not found: "
            f"{INPUT_METRICS_PATH}"
        )

    with INPUT_RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if tuple(
            reader.fieldnames
            or ()
        ) != EXPECTED_RESULT_FIELDS:
            raise RuntimeError(
                "Evaluator result CSV schema does not match the "
                "accepted temporal-aggregation format"
            )

        rows = list(
            reader
        )

    if not rows:
        raise RuntimeError(
            "Evaluator result CSV contains no data rows"
        )

    duplicate_keys = set()
    seen = set()

    for row in rows:
        key = (
            row[
                "case"
            ],
            row[
                "check"
            ],
        )

        if key in seen:
            duplicate_keys.add(
                key
            )

        seen.add(
            key
        )

    if duplicate_keys:
        raise RuntimeError(
            "Duplicate evaluator check keys found: "
            f"{sorted(duplicate_keys)[:10]}"
        )

    invalid_statuses = sorted(
        {
            row[
                "status"
            ]
            for row in rows
            if row[
                "status"
            ]
            not in {
                "PASS",
                "FAIL",
            }
        }
    )

    if invalid_statuses:
        raise RuntimeError(
            "Unexpected evaluator statuses: "
            f"{invalid_statuses}"
        )

    failed_rows = [
        row
        for row in rows
        if row[
            "status"
        ]
        != "PASS"
    ]

    if failed_rows:
        raise RuntimeError(
            "The accepted evaluator table contains failed checks; "
            "quantitative plots will not be generated"
        )

    return rows


def parse_absolute_error(
    value,
):
    """Parse one optional finite absolute-error value."""
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    try:
        numeric = float(
            value
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid absolute-error value: {value}"
        ) from exc

    if not math.isfinite(
        numeric
    ):
        raise RuntimeError(
            f"Non-finite absolute-error value: {value}"
        )

    if numeric < 0:
        raise RuntimeError(
            f"Negative absolute-error value: {value}"
        )

    return numeric


def build_table(
    rows,
):
    """Create compact aggregate correctness statistics by validation case."""
    case_order = []

    for row in rows:
        case = row[
            "case"
        ]

        if case not in case_order:
            case_order.append(
                case
            )

    table_rows = []

    for case in case_order:
        case_rows = [
            row
            for row in rows
            if row[
                "case"
            ]
            == case
        ]

        total_checks = len(
            case_rows
        )
        passed_checks = sum(
            1
            for row in case_rows
            if row[
                "status"
            ]
            == "PASS"
        )
        failed_checks = (
            total_checks
            - passed_checks
        )

        numeric_errors = [
            error
            for error in (
                parse_absolute_error(
                    row[
                        "absolute_error"
                    ]
                )
                for row in case_rows
            )
            if error is not None
        ]

        table_rows.append(
            {
                "validation_case": case,
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "pass_rate": (
                    passed_checks
                    / total_checks
                ),
                "numeric_checks": len(
                    numeric_errors
                ),
                "max_numeric_absolute_error": (
                    max(
                        numeric_errors
                    )
                    if numeric_errors
                    else 0.0
                ),
            }
        )

    return table_rows


def write_csv(
    path,
    rows,
):
    """Write the thesis-ready aggregate table."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TABLE_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def write_markdown(
    path,
    rows,
):
    """Write an optional Markdown representation of the aggregate table."""
    lines = [
        (
            "| Validation case | Total checks | Passed | Failed | "
            "Pass rate | Numeric checks | Max numeric absolute error |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['validation_case']} | "
            f"{row['total_checks']} | "
            f"{row['passed_checks']} | "
            f"{row['failed_checks']} | "
            f"{row['pass_rate']:.6f} | "
            f"{row['numeric_checks']} | "
            f"{row['max_numeric_absolute_error']:.16g} |"
        )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def plot_coverage(
    path,
    rows,
):
    """Plot passed checks against total checks for every validation case."""
    labels = [
        CASE_LABELS.get(
            row[
                "validation_case"
            ],
            row[
                "validation_case"
            ],
        )
        for row in rows
    ]
    totals = [
        row[
            "total_checks"
        ]
        for row in rows
    ]
    passed = [
        row[
            "passed_checks"
        ]
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    positions = list(
        range(
            len(
                rows
            )
        )
    )

    axis.bar(
        positions,
        totals,
        label="Total checks",
        alpha=0.35,
    )
    axis.bar(
        positions,
        passed,
        label="Passed checks",
        alpha=0.85,
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )
    axis.set_ylabel(
        "Number of checks"
    )
    axis.set_title(
        "Temporal Aggregation Validation Coverage"
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def plot_numeric_error(
    path,
    rows,
):
    """Plot numerical-check coverage together with the maximum observed error."""
    labels = [
        CASE_LABELS.get(
            row[
                "validation_case"
            ],
            row[
                "validation_case"
            ],
        )
        for row in rows
    ]
    numeric_checks = [
        row[
            "numeric_checks"
        ]
        for row in rows
    ]
    max_errors = [
        row[
            "max_numeric_absolute_error"
        ]
        for row in rows
    ]

    figure, axis = plt.subplots(
        figsize=(
            12,
            6,
        )
    )

    positions = list(
        range(
            len(
                rows
            )
        )
    )

    bars = axis.bar(
        positions,
        numeric_checks,
    )

    axis.set_xticks(
        positions
    )
    axis.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )
    axis.set_ylabel(
        "Independently checked numerical values"
    )
    axis.set_title(
        "Temporal Aggregation Numerical Correctness"
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    maximum_count = max(
        numeric_checks
        or [
            0
        ]
    )
    vertical_offset = max(
        0.5,
        maximum_count
        * 0.02,
    )

    for bar, count, error in zip(
        bars,
        numeric_checks,
        max_errors,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width()
            / 2.0,
            bar.get_height()
            + vertical_offset,
            (
                f"n={count}\n"
                f"max error={error:.3g}"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    total_numeric_checks = sum(
        numeric_checks
    )
    global_max_error = max(
        max_errors
        or [
            0.0
        ]
    )

    axis.text(
        0.5,
        0.96,
        (
            f"Independent numerical comparisons: {total_numeric_checks}    "
            f"Global maximum absolute error: {global_max_error:.3g}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "alpha": 0.12,
        },
    )

    upper_limit = max(
        1.0,
        maximum_count
        + max(
            4.0,
            maximum_count
            * 0.18,
        ),
    )

    axis.set_ylim(
        0.0,
        upper_limit,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def validate_staged_outputs(
    staging_root,
):
    """Validate all staged table and figure outputs before promotion."""
    staged_table_csv = (
        staging_root
        / TABLE_CSV_PATH.name
    )
    staged_table_md = (
        staging_root
        / TABLE_MD_PATH.name
    )
    staged_figures = (
        staging_root
        / "figures"
    )
    staged_coverage = (
        staged_figures
        / COVERAGE_FIGURE_PATH.name
    )
    staged_error = (
        staged_figures
        / ERROR_FIGURE_PATH.name
    )

    for path in (
        staged_table_csv,
        staged_table_md,
        staged_coverage,
        staged_error,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing staged output: {path.name}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Empty staged output: {path.name}"
            )

    with staged_table_csv.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if tuple(
            reader.fieldnames
            or ()
        ) != TABLE_FIELDS:
            raise RuntimeError(
                "Staged thesis table schema mismatch"
            )

        table_rows = list(
            reader
        )

    if not table_rows:
        raise RuntimeError(
            "Staged thesis table contains no data rows"
        )

    for row in table_rows:
        if int(
            row[
                "failed_checks"
            ]
        ) != 0:
            raise RuntimeError(
                "Staged thesis table contains failed checks"
            )

        if float(
            row[
                "pass_rate"
            ]
        ) != 1.0:
            raise RuntimeError(
                "Staged thesis table contains a pass rate below 1.0"
            )

    for image_path in (
        staged_coverage,
        staged_error,
    ):
        image = plt.imread(
            image_path
        )

        if image.size == 0:
            raise RuntimeError(
                f"Unable to validate staged figure: {image_path.name}"
            )


def promote_outputs(
    staging_root,
):
    """Transactionally replace only outputs owned by this plotting script."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_paths = (
        TABLE_CSV_PATH,
        TABLE_MD_PATH,
        COVERAGE_FIGURE_PATH,
        ERROR_FIGURE_PATH,
    )

    staged_paths = (
        staging_root
        / TABLE_CSV_PATH.name,
        staging_root
        / TABLE_MD_PATH.name,
        staging_root
        / "figures"
        / COVERAGE_FIGURE_PATH.name,
        staging_root
        / "figures"
        / ERROR_FIGURE_PATH.name,
    )

    rollback_root = Path(
        tempfile.mkdtemp(
            prefix=".temporal_aggregation_plot_rollback_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    replaced = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                backup_path = (
                    rollback_root
                    / final_path.relative_to(
                        RESULTS_DIR
                    )
                )
                backup_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                shutil.copy2(
                    final_path,
                    backup_path,
                )

        for staged_path, final_path in zip(
            staged_paths,
            final_paths,
        ):
            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            os.replace(
                staged_path,
                final_path,
            )
            replaced.append(
                final_path
            )

    except Exception:
        for final_path in replaced:
            if final_path.exists():
                final_path.unlink()

        for backup_path in rollback_root.rglob(
            "*"
        ):
            if not backup_path.is_file():
                continue

            relative_backup = backup_path.relative_to(
                rollback_root
            )
            restore_path = (
                RESULTS_DIR
                / relative_backup
            )
            restore_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            os.replace(
                backup_path,
                restore_path,
            )

        raise

    finally:
        shutil.rmtree(
            rollback_root,
            ignore_errors=True,
        )


def main():
    """Generate thesis-ready quantitative tables and figures safely."""
    print(
        "PhysioTrack Temporal Aggregation Plot and Table Generation"
    )
    print(
        "========================================================"
    )

    rows = read_results()

    print(
        "Preflight: PASS"
    )
    print(
        f"Evaluator checks loaded: {len(rows)}"
    )

    table_rows = build_table(
        rows
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".temporal_aggregation_plot_staging_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    staged_figures = (
        staging_root
        / "figures"
    )
    staged_figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        write_csv(
            staging_root
            / TABLE_CSV_PATH.name,
            table_rows,
        )
        write_markdown(
            staging_root
            / TABLE_MD_PATH.name,
            table_rows,
        )
        plot_coverage(
            staged_figures
            / COVERAGE_FIGURE_PATH.name,
            table_rows,
        )
        plot_numeric_error(
            staged_figures
            / ERROR_FIGURE_PATH.name,
            table_rows,
        )

        validate_staged_outputs(
            staging_root
        )

        total_checks = sum(
            row[
                "total_checks"
            ]
            for row in table_rows
        )
        passed_checks = sum(
            row[
                "passed_checks"
            ]
            for row in table_rows
        )
        max_error = max(
            row[
                "max_numeric_absolute_error"
            ]
            for row in table_rows
        )

        print(
            f"Validation cases: {len(table_rows)}"
        )
        print(
            f"Total checks: {total_checks}"
        )
        print(
            f"Passed checks: {passed_checks}"
        )
        print(
            f"Maximum numerical absolute error: {max_error:.16g}"
        )
        print(
            "Staged outputs: PASS"
        )

        promote_outputs(
            staging_root
        )

        print(
            "Final output promotion: PASS"
        )
        print(
            "Overall status: PASS"
        )

    finally:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
