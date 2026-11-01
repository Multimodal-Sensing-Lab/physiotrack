from dataclasses import dataclass
from pathlib import Path
import csv
import math
import os
import shutil
import sys
import tempfile

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "300W"

RESULTS_DIR = SCRIPT_DIR / "results"
QUALITATIVE_DIR = RESULTS_DIR / "qualitative"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUT_RESULTS_PATH = RESULTS_DIR / "face_quality_results.csv"

QUALITATIVE_CSV_PATH = (
    QUALITATIVE_DIR
    / "face_quality_qualitative_results.csv"
)

COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "face_quality_qualitative_examples.png"
)

EXPECTED_IMAGES = 600
EXPECTED_LANDMARKS = 68
FACE_BOX_PADDING = 0.20
CONTROLLED_CONFIDENCE = 1.0

BRIGHTNESS_FACTORS = (
    0.40,
    1.00,
    1.40,
)

BLUR_KERNEL_SIZES = (
    1,
    9,
    25,
)

AREA_SCALE_FACTORS = (
    0.50,
    0.75,
    1.00,
)

SELECTED_SAMPLE_COUNT = 3

QUALITATIVE_FIELDS = (
    "sample_index",
    "split",
    "image",
    "experiment",
    "parameter_name",
    "parameter_value",
    "brightness",
    "sharpness",
    "face_area_ratio",
    "output_image",
)


@dataclass
class ControlledFace:
    """Minimal controlled face record accepted by FaceQuality."""

    box: list
    confidence: float = CONTROLLED_CONFIDENCE


def resolve_face_quality():
    """Import the real PhysioTrack FaceQuality implementation."""
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(
            f"PhysioTrack source directory not found: {SRC_ROOT}"
        )

    src_root_string = str(
        SRC_ROOT
    )

    if src_root_string not in sys.path:
        sys.path.insert(
            0,
            src_root_string,
        )

    from physiotrack.face.quality import FaceQuality

    return FaceQuality


def load_pts(path):
    """Load one 68-point 300-W PTS annotation."""
    lines = path.read_text(
        encoding="utf-8",
        errors="strict",
    ).splitlines()

    points = []
    inside_points = False

    for line in lines:
        line = line.strip()

        if line == "{":
            inside_points = True
            continue

        if line == "}":
            break

        if not inside_points:
            continue

        parts = line.split()

        if len(parts) != 2:
            raise ValueError(
                f"Malformed landmark row in {path}: {line}"
            )

        points.append(
            (
                float(parts[0]),
                float(parts[1]),
            )
        )

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.shape != (
        EXPECTED_LANDMARKS,
        2,
    ):
        raise ValueError(
            f"Expected {EXPECTED_LANDMARKS} landmarks in {path}, "
            f"got {points.shape}"
        )

    if not np.isfinite(
        points
    ).all():
        raise ValueError(
            f"Non-finite landmark coordinates in {path}"
        )

    return points


def derive_controlled_box(
    points,
    image_width,
    image_height,
):
    """Derive the documented 20-percent padded landmark box."""
    x_min = float(
        points[:, 0].min()
    )
    y_min = float(
        points[:, 1].min()
    )
    x_max = float(
        points[:, 0].max()
    )
    y_max = float(
        points[:, 1].max()
    )

    box_width = (
        x_max
        - x_min
    )
    box_height = (
        y_max
        - y_min
    )

    pad_x = (
        FACE_BOX_PADDING
        * box_width
    )
    pad_y = (
        FACE_BOX_PADDING
        * box_height
    )

    x1 = max(
        0.0,
        x_min - pad_x,
    )
    y1 = max(
        0.0,
        y_min - pad_y,
    )
    x2 = min(
        float(
            image_width
        ),
        x_max + pad_x,
    )
    y2 = min(
        float(
            image_height
        ),
        y_max + pad_y,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            "Controlled face box has non-positive size"
        )

    touches_boundary = (
        x1 <= 0.0
        or y1 <= 0.0
        or x2 >= float(
            image_width
        )
        or y2 >= float(
            image_height
        )
    )

    return (
        [
            x1,
            y1,
            x2,
            y2,
        ],
        touches_boundary,
    )


def run_quality(
    quality,
    image,
    box,
):
    """Run the real FaceQuality component on one controlled face."""
    result = quality.predict(
        image,
        [
            ControlledFace(
                box=list(
                    box
                )
            ),
        ],
    )

    if len(
        result
    ) != 1:
        raise RuntimeError(
            "Expected exactly one FaceQuality result, "
            f"got {len(result)}"
        )

    output = result[
        0
    ]

    for key in (
        "brightness",
        "sharpness",
        "face_area_ratio",
    ):
        value = float(
            output[
                key
            ]
        )

        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                f"Non-finite FaceQuality output: {key}"
            )

    return {
        "brightness": float(
            output[
                "brightness"
            ]
        ),
        "sharpness": float(
            output[
                "sharpness"
            ]
        ),
        "face_area_ratio": float(
            output[
                "face_area_ratio"
            ]
        ),
    }


def read_accepted_results():
    """Read accepted evaluator results used for deterministic sampling."""
    if not INPUT_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Accepted evaluator result file not found: {INPUT_RESULTS_PATH}"
        )

    with INPUT_RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    if not rows:
        raise RuntimeError(
            "Accepted evaluator result table is empty"
        )

    baseline_rows = [
        row
        for row in rows
        if row[
            "experiment"
        ] == "baseline"
    ]

    if len(
        baseline_rows
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} baseline rows, "
            f"found {len(baseline_rows)}"
        )

    for row in baseline_rows:
        row[
            "brightness"
        ] = float(
            row[
                "brightness"
            ]
        )
        row[
            "sharpness"
        ] = float(
            row[
                "sharpness"
            ]
        )
        row[
            "face_area_ratio"
        ] = float(
            row[
                "face_area_ratio"
            ]
        )
        row[
            "box_touches_boundary"
        ] = (
            str(
                row[
                    "box_touches_boundary"
                ]
            ).strip().lower()
            == "true"
        )

    return baseline_rows


def select_samples(
    baseline_rows,
):
    """Select deterministic representative samples from accepted results."""
    eligible = [
        row
        for row in baseline_rows
        if not row[
            "box_touches_boundary"
        ]
    ]

    if len(
        eligible
    ) < SELECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "Not enough fully-inside baseline samples"
        )

    eligible.sort(
        key=lambda row: (
            row[
                "brightness"
            ],
            row[
                "sharpness"
            ],
            row[
                "face_area_ratio"
            ],
            row[
                "split"
            ],
            row[
                "image"
            ],
        )
    )

    indices = (
        int(
            round(
                0.25
                * (
                    len(
                        eligible
                    )
                    - 1
                )
            )
        ),
        int(
            round(
                0.50
                * (
                    len(
                        eligible
                    )
                    - 1
                )
            )
        ),
        int(
            round(
                0.75
                * (
                    len(
                        eligible
                    )
                    - 1
                )
            )
        ),
    )

    selected = [
        eligible[
            index
        ]
        for index in indices
    ]

    keys = {
        (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )
        for row in selected
    }

    if len(
        keys
    ) != SELECTED_SAMPLE_COUNT:
        raise RuntimeError(
            "Deterministic qualitative sample selection produced duplicates"
        )

    return selected


def image_path_from_row(
    row,
):
    """Resolve a selected 300-W image path."""
    split_dir = (
        "01_Indoor"
        if row[
            "split"
        ] == "Indoor"
        else "02_Outdoor"
    )

    return (
        DATASET_ROOT
        / split_dir
        / row[
            "image"
        ]
    )


def draw_box(
    image,
    box,
    label,
):
    """Draw one controlled face box and label."""
    output = (
        image.copy()
    )

    x1, y1, x2, y2 = [
        int(
            value
        )
        for value in box
    ]

    cv2.rectangle(
        output,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            255,
            255,
            255,
        ),
        2,
    )

    cv2.putText(
        output,
        label,
        (
            max(
                0,
                x1,
            ),
            max(
                20,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    return output


def save_image(
    path,
    image,
):
    """Write one qualitative image and verify success."""
    success = cv2.imwrite(
        str(
            path
        ),
        image,
    )

    if not success:
        raise RuntimeError(
            f"OpenCV could not write qualitative image: {path}"
        )


def make_scaled_box(
    box,
    scale,
):
    """Scale one fully-inside box around its center."""
    center_x = (
        box[0]
        + box[2]
    ) / 2.0
    center_y = (
        box[1]
        + box[3]
    ) / 2.0

    width = (
        box[2]
        - box[0]
    )
    height = (
        box[3]
        - box[1]
    )

    scaled_width = (
        width
        * scale
    )
    scaled_height = (
        height
        * scale
    )

    return [
        center_x
        - scaled_width / 2.0,
        center_y
        - scaled_height / 2.0,
        center_x
        + scaled_width / 2.0,
        center_y
        + scaled_height / 2.0,
    ]


def create_qualitative_outputs(
    quality,
    selected_rows,
    staging_qualitative_dir,
):
    """Create deterministic qualitative evidence and numerical records."""
    records = []
    panel_data = []

    for sample_index, row in enumerate(
        selected_rows,
        start=1,
    ):
        image_path = image_path_from_row(
            row
        )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read image: {image_path}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        points = load_pts(
            image_path.with_suffix(
                ".pts"
            )
        )

        box, touches_boundary = (
            derive_controlled_box(
                points,
                image_width,
                image_height,
            )
        )

        if touches_boundary:
            raise RuntimeError(
                f"Selected qualitative box unexpectedly touches boundary: "
                f"{row['split']}/{row['image']}"
            )

        sample_panels = []

        for factor in BRIGHTNESS_FACTORS:
            adjusted = np.clip(
                image.astype(
                    np.float32
                )
                * factor,
                0,
                255,
            ).astype(
                np.uint8
            )

            result = run_quality(
                quality,
                adjusted,
                box,
            )

            label = (
                f"Brightness x{factor:.2f} | "
                f"B={result['brightness']:.3f}"
            )

            visual = draw_box(
                adjusted,
                box,
                label,
            )

            filename = (
                f"sample_{sample_index:02d}_brightness_"
                f"{factor:.2f}.png"
            )

            output_path = (
                staging_qualitative_dir
                / filename
            )

            save_image(
                output_path,
                visual,
            )

            records.append(
                {
                    "sample_index": sample_index,
                    "split": row[
                        "split"
                    ],
                    "image": row[
                        "image"
                    ],
                    "experiment": "brightness",
                    "parameter_name": "brightness_factor",
                    "parameter_value": factor,
                    "brightness": result[
                        "brightness"
                    ],
                    "sharpness": result[
                        "sharpness"
                    ],
                    "face_area_ratio": result[
                        "face_area_ratio"
                    ],
                    "output_image": filename,
                }
            )

            sample_panels.append(
                (
                    "Brightness",
                    factor,
                    visual,
                    result,
                )
            )

        for kernel_size in BLUR_KERNEL_SIZES:
            if kernel_size == 1:
                blurred = (
                    image.copy()
                )
            else:
                blurred = cv2.GaussianBlur(
                    image,
                    (
                        kernel_size,
                        kernel_size,
                    ),
                    0,
                )

            result = run_quality(
                quality,
                blurred,
                box,
            )

            label = (
                f"Blur {kernel_size} | "
                f"S={result['sharpness']:.1f}"
            )

            visual = draw_box(
                blurred,
                box,
                label,
            )

            filename = (
                f"sample_{sample_index:02d}_sharpness_"
                f"{kernel_size}.png"
            )

            output_path = (
                staging_qualitative_dir
                / filename
            )

            save_image(
                output_path,
                visual,
            )

            records.append(
                {
                    "sample_index": sample_index,
                    "split": row[
                        "split"
                    ],
                    "image": row[
                        "image"
                    ],
                    "experiment": "sharpness",
                    "parameter_name": "gaussian_kernel_size",
                    "parameter_value": kernel_size,
                    "brightness": result[
                        "brightness"
                    ],
                    "sharpness": result[
                        "sharpness"
                    ],
                    "face_area_ratio": result[
                        "face_area_ratio"
                    ],
                    "output_image": filename,
                }
            )

            sample_panels.append(
                (
                    "Sharpness",
                    kernel_size,
                    visual,
                    result,
                )
            )

        for scale in AREA_SCALE_FACTORS:
            scaled_box = make_scaled_box(
                box,
                scale,
            )

            result = run_quality(
                quality,
                image,
                scaled_box,
            )

            label = (
                f"Scale {scale:.2f} | "
                f"A={result['face_area_ratio']:.3f}"
            )

            visual = draw_box(
                image,
                scaled_box,
                label,
            )

            filename = (
                f"sample_{sample_index:02d}_face_area_ratio_"
                f"{scale:.2f}.png"
            )

            output_path = (
                staging_qualitative_dir
                / filename
            )

            save_image(
                output_path,
                visual,
            )

            records.append(
                {
                    "sample_index": sample_index,
                    "split": row[
                        "split"
                    ],
                    "image": row[
                        "image"
                    ],
                    "experiment": "face_area_ratio",
                    "parameter_name": "box_scale_factor",
                    "parameter_value": scale,
                    "brightness": result[
                        "brightness"
                    ],
                    "sharpness": result[
                        "sharpness"
                    ],
                    "face_area_ratio": result[
                        "face_area_ratio"
                    ],
                    "output_image": filename,
                }
            )

            sample_panels.append(
                (
                    "Face area",
                    scale,
                    visual,
                    result,
                )
            )

        panel_data.append(
            {
                "sample_index": sample_index,
                "split": row[
                    "split"
                ],
                "image": row[
                    "image"
                ],
                "panels": sample_panels,
            }
        )

    return (
        records,
        panel_data,
    )


def write_qualitative_csv(
    path,
    records,
):
    """Write numerical qualitative evidence."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=QUALITATIVE_FIELDS,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                record
            )


def create_combined_figure(
    path,
    panel_data,
):
    """Create one compact deterministic qualitative figure."""
    columns = 9
    rows = len(
        panel_data
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            18.0,
            3.7
            * rows,
        ),
    )

    if rows == 1:
        axes = np.asarray(
            [
                axes,
            ]
        )

    column_titles = (
        "Dark\nx0.40",
        "Original\nx1.00",
        "Bright\nx1.40",
        "No blur\nk=1",
        "Blur\nk=9",
        "Strong blur\nk=25",
        "Small box\nx0.50",
        "Medium box\nx0.75",
        "Original box\nx1.00",
    )

    for column_index, title in enumerate(
        column_titles
    ):
        axes[
            0,
            column_index,
        ].set_title(
            title,
            fontsize=11,
        )

    for row_index, sample in enumerate(
        panel_data
    ):
        panels = sample[
            "panels"
        ]

        if len(
            panels
        ) != columns:
            raise RuntimeError(
                "Unexpected qualitative panel count"
            )

        for column_index, (
            _,
            _,
            visual,
            result,
        ) in enumerate(
            panels
        ):
            rgb = cv2.cvtColor(
                visual,
                cv2.COLOR_BGR2RGB,
            )

            axis = axes[
                row_index,
                column_index,
            ]

            axis.imshow(
                rgb
            )
            axis.axis(
                "off"
            )

            if column_index <= 2:
                value_text = (
                    f"B={result['brightness']:.3f}"
                )
            elif column_index <= 5:
                value_text = (
                    f"S={result['sharpness']:.1f}"
                )
            else:
                value_text = (
                    f"A={result['face_area_ratio']:.3f}"
                )

            axis.text(
                0.5,
                -0.05,
                value_text,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

        axes[
            row_index,
            0,
        ].text(
            -0.10,
            0.5,
            (
                f"Sample {sample['sample_index']}\n"
                f"{sample['split']}\n"
                f"{sample['image']}"
            ),
            transform=axes[
                row_index,
                0,
            ].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )

    figure.suptitle(
        "PhysioTrack FaceQuality Controlled Qualitative Examples",
        fontsize=16,
        y=0.995,
    )

    figure.tight_layout(
        rect=(
            0.04,
            0.02,
            1.0,
            0.97,
        )
    )

    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def validate_staged_outputs(
    csv_path,
    qualitative_dir,
    combined_figure_path,
):
    """Validate all staged qualitative outputs before commit."""
    with csv_path.open(
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
        SELECTED_SAMPLE_COUNT
        * (
            len(
                BRIGHTNESS_FACTORS
            )
            + len(
                BLUR_KERNEL_SIZES
            )
            + len(
                AREA_SCALE_FACTORS
            )
        )
    )

    if len(
        rows
    ) != expected_rows:
        raise RuntimeError(
            f"Qualitative CSV row count mismatch: "
            f"expected {expected_rows}, got {len(rows)}"
        )

    expected_image_count = (
        expected_rows
    )

    images = sorted(
        qualitative_dir.glob(
            "*.png"
        )
    )

    if len(
        images
    ) != expected_image_count:
        raise RuntimeError(
            f"Qualitative image count mismatch: "
            f"expected {expected_image_count}, got {len(images)}"
        )

    for image_path in images:
        if image_path.stat().st_size <= 0:
            raise RuntimeError(
                f"Empty qualitative image: {image_path}"
            )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            raise RuntimeError(
                f"Unreadable qualitative image: {image_path}"
            )

    for row in rows:
        for key in (
            "parameter_value",
            "brightness",
            "sharpness",
            "face_area_ratio",
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
                    f"Non-finite qualitative value in {key}"
                )

        output_image = (
            qualitative_dir
            / row[
                "output_image"
            ]
        )

        if not output_image.is_file():
            raise RuntimeError(
                f"Missing qualitative image referenced by CSV: {output_image}"
            )

    if not combined_figure_path.is_file():
        raise FileNotFoundError(
            f"Combined qualitative figure not found: {combined_figure_path}"
        )

    if combined_figure_path.stat().st_size <= 0:
        raise RuntimeError(
            "Combined qualitative figure is empty"
        )

    combined_image = cv2.imread(
        str(
            combined_figure_path
        )
    )

    if combined_image is None:
        raise RuntimeError(
            "Combined qualitative figure is unreadable"
        )

    return {
        "csv_rows": len(
            rows
        ),
        "qualitative_images": len(
            images
        ),
    }


def commit_outputs(
    staging_dir,
):
    """Commit qualitative-owned outputs with rollback protection."""
    staged_qualitative_dir = (
        staging_dir
        / "qualitative"
    )

    staged_figure_path = (
        staging_dir
        / "figures"
        / COMBINED_FIGURE_PATH.name
    )

    backup_dir = (
        staging_dir
        / "_rollback"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_qualitative_backup = (
        backup_dir
        / "qualitative"
    )

    final_figure_backup = (
        backup_dir
        / COMBINED_FIGURE_PATH.name
    )

    qualitative_backed_up = False
    figure_backed_up = False
    qualitative_promoted = False
    figure_promoted = False

    try:
        if QUALITATIVE_DIR.exists():
            os.replace(
                QUALITATIVE_DIR,
                final_qualitative_backup,
            )
            qualitative_backed_up = True

        QUALITATIVE_DIR.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.replace(
            staged_qualitative_dir,
            QUALITATIVE_DIR,
        )
        qualitative_promoted = True

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if COMBINED_FIGURE_PATH.exists():
            os.replace(
                COMBINED_FIGURE_PATH,
                final_figure_backup,
            )
            figure_backed_up = True

        os.replace(
            staged_figure_path,
            COMBINED_FIGURE_PATH,
        )
        figure_promoted = True

    except Exception:
        if figure_promoted and COMBINED_FIGURE_PATH.exists():
            COMBINED_FIGURE_PATH.unlink()

        if figure_backed_up and final_figure_backup.exists():
            os.replace(
                final_figure_backup,
                COMBINED_FIGURE_PATH,
            )

        if qualitative_promoted and QUALITATIVE_DIR.exists():
            shutil.rmtree(
                QUALITATIVE_DIR,
                ignore_errors=True,
            )

        if qualitative_backed_up and final_qualitative_backup.exists():
            os.replace(
                final_qualitative_backup,
                QUALITATIVE_DIR,
            )

        raise

    if final_qualitative_backup.exists():
        shutil.rmtree(
            final_qualitative_backup,
            ignore_errors=True,
        )

    if final_figure_backup.exists():
        final_figure_backup.unlink()


def main():
    """Create deterministic Face Quality qualitative evidence."""
    print(
        "PhysioTrack Face Quality Qualitative Evaluation"
    )
    print(
        "==============================================="
    )
    print()
    print(
        "Validating accepted evaluator results..."
    )

    baseline_rows = read_accepted_results()
    selected_rows = select_samples(
        baseline_rows
    )

    FaceQuality = resolve_face_quality()
    quality = FaceQuality()

    print(
        "Input validation: PASS"
    )
    print(
        "Selected deterministic samples:"
    )

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        print(
            f"  {index}: {row['split']}/{row['image']}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_quality_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    try:
        staged_qualitative_dir = (
            staging_dir
            / "qualitative"
        )

        staged_figures_dir = (
            staging_dir
            / "figures"
        )

        staged_qualitative_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        staged_figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        staged_csv_path = (
            staged_qualitative_dir
            / QUALITATIVE_CSV_PATH.name
        )

        staged_combined_figure_path = (
            staged_figures_dir
            / COMBINED_FIGURE_PATH.name
        )

        records, panel_data = (
            create_qualitative_outputs(
                quality,
                selected_rows,
                staged_qualitative_dir,
            )
        )

        write_qualitative_csv(
            staged_csv_path,
            records,
        )

        create_combined_figure(
            staged_combined_figure_path,
            panel_data,
        )

        print()
        print(
            "Validating staged qualitative outputs..."
        )

        validation = validate_staged_outputs(
            staged_csv_path,
            staged_qualitative_dir,
            staged_combined_figure_path,
        )

        print(
            "Qualitative CSV rows:",
            validation[
                "csv_rows"
            ],
        )
        print(
            "Qualitative images:",
            validation[
                "qualitative_images"
            ],
        )
        print(
            "Combined figure: 1"
        )

        commit_outputs(
            staging_dir
        )

        print(
            "Committed final qualitative outputs."
        )
        print()
        print(
            f"Qualitative directory: {QUALITATIVE_DIR}"
        )
        print(
            f"Combined figure: {COMBINED_FIGURE_PATH}"
        )
        print(
            "Qualitative evaluation status: PASS"
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
