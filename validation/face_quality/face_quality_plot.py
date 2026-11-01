from pathlib import Path
import csv
import math
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_RESULTS_PATH = RESULTS_DIR / "face_quality_results.csv"

METRICS_PATH = RESULTS_DIR / "face_quality_metrics.csv"

BRIGHTNESS_FIGURE_PATH = (
    FIGURES_DIR
    / "face_quality_brightness_sensitivity.png"
)

SHARPNESS_FIGURE_PATH = (
    FIGURES_DIR
    / "face_quality_sharpness_sensitivity.png"
)

AREA_FIGURE_PATH = (
    FIGURES_DIR
    / "face_quality_face_area_ratio_sensitivity.png"
)

EXPECTED_IMAGES = 600
EXPECTED_FULLY_INSIDE_IMAGES = 532

BRIGHTNESS_FACTORS = (
    0.40,
    0.60,
    0.80,
    1.00,
    1.20,
    1.40,
)

BLUR_KERNEL_SIZES = (
    1,
    3,
    5,
    9,
    15,
    25,
)

AREA_SCALE_FACTORS = (
    0.50,
    0.75,
    1.00,
)

METRIC_FIELDS = (
    "experiment",
    "descriptor",
    "parameter_name",
    "parameter_value",
    "sample_count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "expected_direction",
    "monotonic_images",
    "total_images",
    "monotonic_rate",
)


def read_results(path):
    """Read and parse the accepted Face Quality evaluator output."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required evaluator result file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        raw_rows = list(
            csv.DictReader(
                handle
            )
        )

    required_columns = {
        "experiment",
        "split",
        "image",
        "parameter_name",
        "parameter_value",
        "brightness",
        "sharpness",
        "face_area_ratio",
        "expected_face_area_ratio",
        "face_area_ratio_absolute_error",
    }

    if not raw_rows:
        raise RuntimeError(
            "Evaluator result table is empty"
        )

    missing_columns = (
        required_columns
        - set(
            raw_rows[0].keys()
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Evaluator result table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for raw in raw_rows:
        row = {
            **raw,
            "parameter_value": float(
                raw[
                    "parameter_value"
                ]
            ),
            "brightness": float(
                raw[
                    "brightness"
                ]
            ),
            "sharpness": float(
                raw[
                    "sharpness"
                ]
            ),
            "face_area_ratio": float(
                raw[
                    "face_area_ratio"
                ]
            ),
            "expected_face_area_ratio": float(
                raw[
                    "expected_face_area_ratio"
                ]
            ),
            "face_area_ratio_absolute_error": float(
                raw[
                    "face_area_ratio_absolute_error"
                ]
            ),
        }

        for key in (
            "parameter_value",
            "brightness",
            "sharpness",
            "face_area_ratio",
            "expected_face_area_ratio",
            "face_area_ratio_absolute_error",
        ):
            if not math.isfinite(
                row[
                    key
                ]
            ):
                raise RuntimeError(
                    f"Non-finite evaluator value in column {key}"
                )

        rows.append(
            row
        )

    return rows


def validate_input_results(rows):
    """Independently verify the accepted evaluator table before plotting."""
    expected_counts = {
        "baseline": EXPECTED_IMAGES,
        "brightness": (
            EXPECTED_IMAGES
            * len(
                BRIGHTNESS_FACTORS
            )
        ),
        "sharpness": (
            EXPECTED_IMAGES
            * len(
                BLUR_KERNEL_SIZES
            )
        ),
        "face_area_ratio": (
            EXPECTED_FULLY_INSIDE_IMAGES
            * len(
                AREA_SCALE_FACTORS
            )
        ),
    }

    actual_counts = {
        experiment: 0
        for experiment in expected_counts
    }

    for row in rows:
        experiment = row[
            "experiment"
        ]

        if experiment not in actual_counts:
            raise RuntimeError(
                f"Unexpected experiment: {experiment}"
            )

        actual_counts[
            experiment
        ] += 1

        if row[
            "face_area_ratio_absolute_error"
        ] > 1e-12:
            raise RuntimeError(
                "Face-area-ratio formula error exceeds tolerance"
            )

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Evaluator experiment counts do not match the accepted protocol: "
            f"expected {expected_counts}, got {actual_counts}"
        )

    baseline_rows = [
        row
        for row in rows
        if row[
            "experiment"
        ] == "baseline"
    ]

    baseline_keys = {
        (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )
        for row in baseline_rows
    }

    if len(
        baseline_keys
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Baseline image identifiers are not unique and complete"
        )

    brightness_count = monotonic_image_count(
        rows=rows,
        experiment="brightness",
        descriptor="brightness",
        parameters=BRIGHTNESS_FACTORS,
        direction="increasing",
    )

    sharpness_count = monotonic_image_count(
        rows=rows,
        experiment="sharpness",
        descriptor="sharpness",
        parameters=BLUR_KERNEL_SIZES,
        direction="decreasing",
    )

    area_count = monotonic_image_count(
        rows=rows,
        experiment="face_area_ratio",
        descriptor="face_area_ratio",
        parameters=AREA_SCALE_FACTORS,
        direction="increasing",
    )

    if brightness_count != EXPECTED_IMAGES:
        raise RuntimeError(
            "Brightness monotonic-image count mismatch"
        )

    if sharpness_count != EXPECTED_IMAGES:
        raise RuntimeError(
            "Sharpness monotonic-image count mismatch"
        )

    if area_count != EXPECTED_FULLY_INSIDE_IMAGES:
        raise RuntimeError(
            "Face-area monotonic-image count mismatch"
        )

    return {
        "brightness_monotonic_images": brightness_count,
        "sharpness_monotonic_images": sharpness_count,
        "face_area_monotonic_images": area_count,
    }


def monotonic_image_count(
    rows,
    experiment,
    descriptor,
    parameters,
    direction,
):
    """Count images satisfying the complete ordered sensitivity response."""
    grouped = {}

    for row in rows:
        if row[
            "experiment"
        ] != experiment:
            continue

        key = (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )

        grouped.setdefault(
            key,
            []
        ).append(
            row
        )

    count = 0

    for key, values in grouped.items():
        values.sort(
            key=lambda item: item[
                "parameter_value"
            ]
        )

        actual_parameters = [
            row[
                "parameter_value"
            ]
            for row in values
        ]

        expected_parameters = [
            float(
                value
            )
            for value in parameters
        ]

        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"Parameter sequence mismatch for {experiment} {key}"
            )

        descriptor_values = [
            row[
                descriptor
            ]
            for row in values
        ]

        if direction == "increasing":
            passed = all(
                later > earlier
                for earlier, later in zip(
                    descriptor_values,
                    descriptor_values[
                        1:
                    ],
                )
            )
        elif direction == "decreasing":
            passed = all(
                later < earlier
                for earlier, later in zip(
                    descriptor_values,
                    descriptor_values[
                        1:
                    ],
                )
            )
        else:
            raise ValueError(
                f"Unsupported monotonic direction: {direction}"
            )

        if passed:
            count += 1

    return count


def describe(values):
    """Return compact descriptive statistics."""
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        raise RuntimeError(
            "Cannot summarize an empty value set"
        )

    return {
        "sample_count": int(
            array.size
        ),
        "mean": float(
            array.mean()
        ),
        "median": float(
            np.median(
                array
            )
        ),
        "std": float(
            array.std(
                ddof=1
            )
        )
        if array.size > 1
        else 0.0,
        "min": float(
            array.min()
        ),
        "max": float(
            array.max()
        ),
    }


def build_metrics(
    rows,
    validation,
):
    """Build the compact aggregate metrics table."""
    metric_rows = []

    baseline_descriptors = (
        "brightness",
        "sharpness",
        "face_area_ratio",
    )

    baseline_rows = [
        row
        for row in rows
        if row[
            "experiment"
        ] == "baseline"
    ]

    for descriptor in baseline_descriptors:
        stats = describe(
            [
                row[
                    descriptor
                ]
                for row in baseline_rows
            ]
        )

        metric_rows.append(
            {
                "experiment": "baseline",
                "descriptor": descriptor,
                "parameter_name": "none",
                "parameter_value": 1.0,
                **stats,
                "expected_direction": "not_applicable",
                "monotonic_images": "",
                "total_images": EXPECTED_IMAGES,
                "monotonic_rate": "",
            }
        )

    sensitivity_specs = (
        (
            "brightness",
            "brightness",
            "brightness_factor",
            BRIGHTNESS_FACTORS,
            "increasing",
            validation[
                "brightness_monotonic_images"
            ],
            EXPECTED_IMAGES,
        ),
        (
            "sharpness",
            "sharpness",
            "gaussian_kernel_size",
            BLUR_KERNEL_SIZES,
            "decreasing",
            validation[
                "sharpness_monotonic_images"
            ],
            EXPECTED_IMAGES,
        ),
        (
            "face_area_ratio",
            "face_area_ratio",
            "box_scale_factor",
            AREA_SCALE_FACTORS,
            "increasing",
            validation[
                "face_area_monotonic_images"
            ],
            EXPECTED_FULLY_INSIDE_IMAGES,
        ),
    )

    for (
        experiment,
        descriptor,
        parameter_name,
        parameters,
        direction,
        monotonic_images,
        total_images,
    ) in sensitivity_specs:
        for parameter in parameters:
            values = [
                row[
                    descriptor
                ]
                for row in rows
                if (
                    row[
                        "experiment"
                    ] == experiment
                    and math.isclose(
                        row[
                            "parameter_value"
                        ],
                        float(
                            parameter
                        ),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                )
            ]

            stats = describe(
                values
            )

            metric_rows.append(
                {
                    "experiment": experiment,
                    "descriptor": descriptor,
                    "parameter_name": parameter_name,
                    "parameter_value": parameter,
                    **stats,
                    "expected_direction": direction,
                    "monotonic_images": monotonic_images,
                    "total_images": total_images,
                    "monotonic_rate": (
                        monotonic_images
                        / total_images
                    ),
                }
            )

    return metric_rows


def write_metrics(
    path,
    metric_rows,
):
    """Write the aggregate numerical metrics table."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=METRIC_FIELDS,
        )

        writer.writeheader()

        for row in metric_rows:
            writer.writerow(
                row
            )


def validate_metrics(path):
    """Validate the staged aggregate metrics table."""
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    expected_rows = (
        3
        + len(
            BRIGHTNESS_FACTORS
        )
        + len(
            BLUR_KERNEL_SIZES
        )
        + len(
            AREA_SCALE_FACTORS
        )
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Metrics row count mismatch: "
            f"expected {expected_rows}, got {len(rows)}"
        )

    for row in rows:
        for key in (
            "sample_count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ):
            value = float(
                row[
                    key
                ]
            )

            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    f"Non-finite metrics value in column {key}"
                )

    return rows


def aggregate_series(
    rows,
    experiment,
    descriptor,
    parameters,
):
    """Build mean and standard-deviation series for one experiment."""
    means = []
    stds = []

    for parameter in parameters:
        values = [
            row[
                descriptor
            ]
            for row in rows
            if (
                row[
                    "experiment"
                ] == experiment
                and math.isclose(
                    row[
                        "parameter_value"
                    ],
                    float(
                        parameter
                    ),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        ]

        stats = describe(
            values
        )

        means.append(
            stats[
                "mean"
            ]
        )
        stds.append(
            stats[
                "std"
            ]
        )

    return (
        np.asarray(
            means,
            dtype=np.float64,
        ),
        np.asarray(
            stds,
            dtype=np.float64,
        ),
    )


def save_brightness_figure(
    path,
    rows,
):
    """Plot aggregate brightness response."""
    means, stds = aggregate_series(
        rows=rows,
        experiment="brightness",
        descriptor="brightness",
        parameters=BRIGHTNESS_FACTORS,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            4.8,
        )
    )

    axis.errorbar(
        BRIGHTNESS_FACTORS,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
    )

    axis.set_xlabel(
        "Intensity scale factor"
    )
    axis.set_ylabel(
        "Brightness"
    )
    axis.set_title(
        "FaceQuality Brightness Sensitivity"
    )
    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def save_sharpness_figure(
    path,
    rows,
):
    """Plot aggregate sharpness response."""
    means, stds = aggregate_series(
        rows=rows,
        experiment="sharpness",
        descriptor="sharpness",
        parameters=BLUR_KERNEL_SIZES,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            4.8,
        )
    )

    axis.errorbar(
        BLUR_KERNEL_SIZES,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
    )

    axis.set_xlabel(
        "Gaussian blur kernel size"
    )
    axis.set_ylabel(
        "Laplacian variance"
    )
    axis.set_title(
        "FaceQuality Sharpness Sensitivity"
    )
    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def save_area_figure(
    path,
    rows,
):
    """Plot aggregate face-area-ratio response."""
    means, stds = aggregate_series(
        rows=rows,
        experiment="face_area_ratio",
        descriptor="face_area_ratio",
        parameters=AREA_SCALE_FACTORS,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.0,
            4.8,
        )
    )

    axis.errorbar(
        AREA_SCALE_FACTORS,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
    )

    axis.set_xlabel(
        "Controlled face-box scale factor"
    )
    axis.set_ylabel(
        "Face area ratio"
    )
    axis.set_title(
        "FaceQuality Face-Area-Ratio Sensitivity"
    )
    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def validate_figure(path):
    """Check that a staged figure is present and non-empty."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected figure was not created: {path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Generated figure is empty: {path}"
        )


def commit_outputs(
    staging_dir,
):
    """Commit plot-owned outputs with rollback protection."""
    replacements = (
        (
            staging_dir
            / METRICS_PATH.name,
            METRICS_PATH,
        ),
        (
            staging_dir
            / "figures"
            / BRIGHTNESS_FIGURE_PATH.name,
            BRIGHTNESS_FIGURE_PATH,
        ),
        (
            staging_dir
            / "figures"
            / SHARPNESS_FIGURE_PATH.name,
            SHARPNESS_FIGURE_PATH,
        ),
        (
            staging_dir
            / "figures"
            / AREA_FIGURE_PATH.name,
            AREA_FIGURE_PATH,
        ),
    )

    backup_dir = (
        staging_dir
        / "_rollback"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backups = {}
    promoted = []

    try:
        for staged_path, final_path in replacements:
            if not staged_path.is_file():
                raise FileNotFoundError(
                    f"Missing staged output: {staged_path}"
                )

            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                backups[
                    final_path
                ] = backup_path

            os.replace(
                staged_path,
                final_path,
            )

            promoted.append(
                final_path
            )

    except Exception:
        for final_path in promoted:
            if final_path.exists():
                final_path.unlink()

        for final_path, backup_path in backups.items():
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise

    for backup_path in backups.values():
        if backup_path.exists():
            backup_path.unlink()


def main():
    """Create aggregate Face Quality metrics and quantitative figures."""
    print(
        "PhysioTrack Face Quality Plot Generation"
    )
    print(
        "========================================"
    )
    print()
    print(
        "Validating accepted evaluator results..."
    )

    rows = read_results(
        INPUT_RESULTS_PATH
    )

    validation = validate_input_results(
        rows
    )

    print(
        "Input validation: PASS"
    )
    print(
        f"Detailed result rows: {len(rows)}"
    )
    print(
        "Brightness monotonic images:",
        validation[
            "brightness_monotonic_images"
        ],
    )
    print(
        "Sharpness monotonic images:",
        validation[
            "sharpness_monotonic_images"
        ],
    )
    print(
        "Face-area monotonic images:",
        validation[
            "face_area_monotonic_images"
        ],
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_quality_plot_",
            dir=RESULTS_DIR,
        )
    )

    try:
        staged_figures_dir = (
            staging_dir
            / "figures"
        )

        staged_figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        staged_metrics_path = (
            staging_dir
            / METRICS_PATH.name
        )

        staged_brightness_path = (
            staged_figures_dir
            / BRIGHTNESS_FIGURE_PATH.name
        )

        staged_sharpness_path = (
            staged_figures_dir
            / SHARPNESS_FIGURE_PATH.name
        )

        staged_area_path = (
            staged_figures_dir
            / AREA_FIGURE_PATH.name
        )

        metric_rows = build_metrics(
            rows,
            validation,
        )

        write_metrics(
            staged_metrics_path,
            metric_rows,
        )

        save_brightness_figure(
            staged_brightness_path,
            rows,
        )

        save_sharpness_figure(
            staged_sharpness_path,
            rows,
        )

        save_area_figure(
            staged_area_path,
            rows,
        )

        print()
        print(
            "Validating staged plot outputs..."
        )

        metrics = validate_metrics(
            staged_metrics_path
        )

        validate_figure(
            staged_brightness_path
        )
        validate_figure(
            staged_sharpness_path
        )
        validate_figure(
            staged_area_path
        )

        print(
            f"Aggregate metric rows: {len(metrics)}"
        )
        print(
            "Figures: 3"
        )

        commit_outputs(
            staging_dir
        )

        print(
            "Committed final plot outputs."
        )
        print()
        print(
            f"Metrics: {METRICS_PATH}"
        )
        print(
            f"Figures: {FIGURES_DIR}"
        )
        print(
            "Plot generation status: PASS"
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
