from dataclasses import dataclass
from pathlib import Path
import csv
import math
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "300W"
RESULTS_DIR = SCRIPT_DIR / "results"

RESULTS_PATH = RESULTS_DIR / "face_quality_results.csv"
SUMMARY_PATH = RESULTS_DIR / "face_quality_summary.txt"

EXPECTED_IMAGES = 600
EXPECTED_SPLIT_COUNTS = {
    "Indoor": 300,
    "Outdoor": 300,
}
EXPECTED_LANDMARKS = 68

FACE_BOX_PADDING = 0.20
CONTROLLED_CONFIDENCE = 1.0

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

RESULT_FIELDS = (
    "experiment",
    "split",
    "image",
    "image_width",
    "image_height",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_touches_boundary",
    "parameter_name",
    "parameter_value",
    "confidence",
    "brightness",
    "sharpness",
    "face_area_ratio",
    "expected_face_area_ratio",
    "face_area_ratio_absolute_error",
)


@dataclass
class ControlledFace:
    """Minimal face record accepted by FaceQuality."""

    box: list
    confidence: float = CONTROLLED_CONFIDENCE


def resolve_face_quality():
    """Import the real PhysioTrack FaceQuality implementation."""
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(
            f"PhysioTrack source directory not found: {SRC_ROOT}"
        )

    src_root_string = str(SRC_ROOT)

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

    if not np.isfinite(points).all():
        raise ValueError(
            f"Non-finite landmark coordinates in {path}"
        )

    return points


def discover_dataset():
    """Preflight the complete 300-W evaluation set."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"300-W dataset root not found: {DATASET_ROOT}"
        )

    split_directories = {
        "Indoor": DATASET_ROOT / "01_Indoor",
        "Outdoor": DATASET_ROOT / "02_Outdoor",
    }

    records = []

    for split, split_dir in split_directories.items():
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Required split directory not found: {split_dir}"
            )

        image_paths = sorted(
            split_dir.glob("*.png")
        )

        expected_count = EXPECTED_SPLIT_COUNTS[
            split
        ]

        if len(image_paths) != expected_count:
            raise RuntimeError(
                f"{split} image count mismatch: "
                f"expected {expected_count}, found {len(image_paths)}"
            )

        for image_path in image_paths:
            pts_path = image_path.with_suffix(
                ".pts"
            )

            if not pts_path.is_file():
                raise FileNotFoundError(
                    f"Missing PTS annotation: {pts_path}"
                )

            points = load_pts(
                pts_path
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                raise RuntimeError(
                    f"OpenCV could not read image: {image_path}"
                )

            image_height, image_width = (
                image.shape[:2]
            )

            records.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "pts_path": pts_path,
                    "points": points,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )

    if len(records) != EXPECTED_IMAGES:
        raise RuntimeError(
            "300-W image count mismatch: "
            f"expected {EXPECTED_IMAGES}, found {len(records)}"
        )

    return records


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

    box_width = x_max - x_min
    box_height = y_max - y_min

    if (
        box_width <= 0
        or box_height <= 0
    ):
        raise ValueError(
            "GT-derived face box has non-positive size"
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
        float(image_width),
        x_max + pad_x,
    )
    y2 = min(
        float(image_height),
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
        or x2 >= float(image_width)
        or y2 >= float(image_height)
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


def expected_area_ratio(
    box,
    image_width,
    image_height,
):
    """Reproduce FaceQuality integer conversion and clipping."""
    x1, y1, x2, y2 = [
        int(value)
        for value in box
    ]

    x1 = max(
        0,
        x1,
    )
    y1 = max(
        0,
        y1,
    )
    x2 = min(
        image_width,
        x2,
    )
    y2 = min(
        image_height,
        y2,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            "Integer-converted face box has non-positive size"
        )

    face_area = (
        (x2 - x1)
        * (y2 - y1)
    )

    return float(
        face_area
        / float(
            image_width
            * image_height
        )
    )


def run_quality(
    quality,
    image,
    box,
):
    """Run the real FaceQuality component on one controlled face."""
    face = ControlledFace(
        box=list(
            box
        ),
    )

    outputs = quality.predict(
        image,
        [
            face,
        ],
    )

    if len(outputs) != 1:
        raise RuntimeError(
            "Expected exactly one FaceQuality result, "
            f"got {len(outputs)}"
        )

    output = outputs[
        0
    ]

    required_keys = {
        "confidence",
        "brightness",
        "sharpness",
        "face_area_ratio",
    }

    if set(
        output.keys()
    ) != required_keys:
        raise RuntimeError(
            "Unexpected FaceQuality output keys: "
            f"{sorted(output.keys())}"
        )

    confidence = output[
        "confidence"
    ]

    if confidence is None:
        raise RuntimeError(
            "Controlled confidence was not preserved"
        )

    confidence = float(
        confidence
    )

    if not math.isclose(
        confidence,
        CONTROLLED_CONFIDENCE,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError(
            "Controlled confidence changed unexpectedly"
        )

    result = {
        "confidence": confidence,
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

    for key, value in result.items():
        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                f"Non-finite FaceQuality output: {key}"
            )

    if not (
        0.0
        <= result[
            "brightness"
        ]
        <= 1.0
    ):
        raise RuntimeError(
            "Brightness outside [0, 1]"
        )

    if result[
        "sharpness"
    ] < 0.0:
        raise RuntimeError(
            "Sharpness is negative"
        )

    if not (
        0.0
        < result[
            "face_area_ratio"
        ]
        <= 1.0
    ):
        raise RuntimeError(
            "Face area ratio outside (0, 1]"
        )

    return result


def make_result_row(
    experiment,
    split,
    image_name,
    image_width,
    image_height,
    box,
    touches_boundary,
    parameter_name,
    parameter_value,
    result,
    expected_ratio,
):
    """Create one structured evaluation row."""
    absolute_error = abs(
        result[
            "face_area_ratio"
        ]
        - expected_ratio
    )

    return {
        "experiment": experiment,
        "split": split,
        "image": image_name,
        "image_width": image_width,
        "image_height": image_height,
        "box_x1": box[0],
        "box_y1": box[1],
        "box_x2": box[2],
        "box_y2": box[3],
        "box_touches_boundary": touches_boundary,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "confidence": result[
            "confidence"
        ],
        "brightness": result[
            "brightness"
        ],
        "sharpness": result[
            "sharpness"
        ],
        "face_area_ratio": result[
            "face_area_ratio"
        ],
        "expected_face_area_ratio": expected_ratio,
        "face_area_ratio_absolute_error": (
            absolute_error
        ),
    }


def process_dataset(
    quality,
    dataset_records,
):
    """Run baseline and controlled sensitivity experiments."""
    rows = []
    fully_inside_count = 0

    for image_index, item in enumerate(
        dataset_records,
        start=1,
    ):
        split = item[
            "split"
        ]
        image_path = item[
            "image_path"
        ]
        image_name = (
            image_path.name
        )
        image_width = item[
            "image_width"
        ]
        image_height = item[
            "image_height"
        ]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"OpenCV could not read image: {image_path}"
            )

        box, touches_boundary = (
            derive_controlled_box(
                item[
                    "points"
                ],
                image_width,
                image_height,
            )
        )

        baseline_expected_ratio = (
            expected_area_ratio(
                box,
                image_width,
                image_height,
            )
        )

        baseline_result = run_quality(
            quality,
            image,
            box,
        )

        rows.append(
            make_result_row(
                experiment="baseline",
                split=split,
                image_name=image_name,
                image_width=image_width,
                image_height=image_height,
                box=box,
                touches_boundary=touches_boundary,
                parameter_name="none",
                parameter_value=1.0,
                result=baseline_result,
                expected_ratio=baseline_expected_ratio,
            )
        )

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

            rows.append(
                make_result_row(
                    experiment="brightness",
                    split=split,
                    image_name=image_name,
                    image_width=image_width,
                    image_height=image_height,
                    box=box,
                    touches_boundary=touches_boundary,
                    parameter_name="brightness_factor",
                    parameter_value=factor,
                    result=result,
                    expected_ratio=baseline_expected_ratio,
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

            rows.append(
                make_result_row(
                    experiment="sharpness",
                    split=split,
                    image_name=image_name,
                    image_width=image_width,
                    image_height=image_height,
                    box=box,
                    touches_boundary=touches_boundary,
                    parameter_name="gaussian_kernel_size",
                    parameter_value=kernel_size,
                    result=result,
                    expected_ratio=baseline_expected_ratio,
                )
            )

        if not touches_boundary:
            fully_inside_count += 1

            center_x = (
                box[0]
                + box[2]
            ) / 2.0
            center_y = (
                box[1]
                + box[3]
            ) / 2.0

            base_width = (
                box[2]
                - box[0]
            )
            base_height = (
                box[3]
                - box[1]
            )

            for scale in AREA_SCALE_FACTORS:
                scaled_width = (
                    base_width
                    * scale
                )
                scaled_height = (
                    base_height
                    * scale
                )

                scaled_box = [
                    center_x
                    - scaled_width / 2.0,
                    center_y
                    - scaled_height / 2.0,
                    center_x
                    + scaled_width / 2.0,
                    center_y
                    + scaled_height / 2.0,
                ]

                expected_ratio = (
                    expected_area_ratio(
                        scaled_box,
                        image_width,
                        image_height,
                    )
                )

                result = run_quality(
                    quality,
                    image,
                    scaled_box,
                )

                rows.append(
                    make_result_row(
                        experiment="face_area_ratio",
                        split=split,
                        image_name=image_name,
                        image_width=image_width,
                        image_height=image_height,
                        box=scaled_box,
                        touches_boundary=False,
                        parameter_name="box_scale_factor",
                        parameter_value=scale,
                        result=result,
                        expected_ratio=expected_ratio,
                    )
                )

        if (
            image_index % 50 == 0
            or image_index
            == EXPECTED_IMAGES
        ):
            print(
                f"Processed {image_index}/{EXPECTED_IMAGES} images"
            )

    return (
        rows,
        fully_inside_count,
    )


def write_results(
    path,
    rows,
):
    """Write the detailed result table."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def read_results(path):
    """Read the generated CSV for staged validation."""
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def parse_result_rows(path):
    """Parse numerical columns from the staged CSV."""
    rows = read_results(
        path
    )

    parsed = []

    for row in rows:
        parsed.append(
            {
                **row,
                "image_width": int(
                    row[
                        "image_width"
                    ]
                ),
                "image_height": int(
                    row[
                        "image_height"
                    ]
                ),
                "parameter_value": float(
                    row[
                        "parameter_value"
                    ]
                ),
                "confidence": float(
                    row[
                        "confidence"
                    ]
                ),
                "brightness": float(
                    row[
                        "brightness"
                    ]
                ),
                "sharpness": float(
                    row[
                        "sharpness"
                    ]
                ),
                "face_area_ratio": float(
                    row[
                        "face_area_ratio"
                    ]
                ),
                "expected_face_area_ratio": float(
                    row[
                        "expected_face_area_ratio"
                    ]
                ),
                "face_area_ratio_absolute_error": float(
                    row[
                        "face_area_ratio_absolute_error"
                    ]
                ),
            }
        )

    return parsed


def validate_staged_results(
    path,
    fully_inside_count,
):
    """Validate all staged evaluator rows before final replacement."""
    rows = parse_result_rows(
        path
    )

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
            fully_inside_count
            * len(
                AREA_SCALE_FACTORS
            )
        ),
    }

    actual_counts = {
        key: 0
        for key in expected_counts
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

        if not math.isclose(
            row[
                "confidence"
            ],
            CONTROLLED_CONFIDENCE,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Controlled confidence mismatch"
            )

        for key in (
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
                    f"Non-finite staged value in {key}"
                )

        if row[
            "face_area_ratio_absolute_error"
        ] > 1e-12:
            raise RuntimeError(
                "Face-area-ratio formula mismatch"
            )

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Experiment row-count mismatch: "
            f"expected {expected_counts}, got {actual_counts}"
        )

    expected_total = sum(
        expected_counts.values()
    )

    if len(rows) != expected_total:
        raise RuntimeError(
            f"Total row-count mismatch: "
            f"expected {expected_total}, got {len(rows)}"
        )

    baseline_rows = [
        row
        for row in rows
        if row[
            "experiment"
        ] == "baseline"
    ]

    baseline_lookup = {
        (
            row[
                "split"
            ],
            row[
                "image"
            ],
        ): row
        for row in baseline_rows
    }

    if len(
        baseline_lookup
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Baseline image identifiers are not unique and complete"
        )

    split_counts = {
        split: 0
        for split in EXPECTED_SPLIT_COUNTS
    }

    for row in baseline_rows:
        split = row[
            "split"
        ]

        if split not in split_counts:
            raise RuntimeError(
                f"Unexpected split: {split}"
            )

        split_counts[
            split
        ] += 1

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            "Baseline split-count mismatch: "
            f"expected {EXPECTED_SPLIT_COUNTS}, got {split_counts}"
        )

    grouped = {}

    for row in rows:
        if row[
            "experiment"
        ] not in {
            "brightness",
            "sharpness",
            "face_area_ratio",
        }:
            continue

        key = (
            row[
                "experiment"
            ],
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

    brightness_monotonic = 0
    sharpness_monotonic = 0
    area_monotonic = 0

    for (
        experiment,
        split,
        image,
    ), values in grouped.items():
        values.sort(
            key=lambda item: item[
                "parameter_value"
            ]
        )

        if experiment == "brightness":
            parameters = [
                row[
                    "parameter_value"
                ]
                for row in values
            ]

            if parameters != list(
                BRIGHTNESS_FACTORS
            ):
                raise RuntimeError(
                    f"Brightness parameter mismatch for {split}/{image}"
                )

            descriptor_values = [
                row[
                    "brightness"
                ]
                for row in values
            ]

            if not all(
                later > earlier
                for earlier, later in zip(
                    descriptor_values,
                    descriptor_values[
                        1:
                    ],
                )
            ):
                raise RuntimeError(
                    f"Brightness is not strictly increasing for {split}/{image}"
                )

            brightness_monotonic += 1

        elif experiment == "sharpness":
            parameters = [
                int(
                    row[
                        "parameter_value"
                    ]
                )
                for row in values
            ]

            if parameters != list(
                BLUR_KERNEL_SIZES
            ):
                raise RuntimeError(
                    f"Blur parameter mismatch for {split}/{image}"
                )

            descriptor_values = [
                row[
                    "sharpness"
                ]
                for row in values
            ]

            if not all(
                later < earlier
                for earlier, later in zip(
                    descriptor_values,
                    descriptor_values[
                        1:
                    ],
                )
            ):
                raise RuntimeError(
                    f"Sharpness is not strictly decreasing for {split}/{image}"
                )

            sharpness_monotonic += 1

        elif experiment == "face_area_ratio":
            parameters = [
                row[
                    "parameter_value"
                ]
                for row in values
            ]

            if parameters != list(
                AREA_SCALE_FACTORS
            ):
                raise RuntimeError(
                    f"Face-area parameter mismatch for {split}/{image}"
                )

            descriptor_values = [
                row[
                    "face_area_ratio"
                ]
                for row in values
            ]

            if not all(
                later > earlier
                for earlier, later in zip(
                    descriptor_values,
                    descriptor_values[
                        1:
                    ],
                )
            ):
                raise RuntimeError(
                    f"Face area ratio is not strictly increasing for "
                    f"{split}/{image}"
                )

            area_monotonic += 1

    if brightness_monotonic != EXPECTED_IMAGES:
        raise RuntimeError(
            "Brightness monotonic-image count mismatch"
        )

    if sharpness_monotonic != EXPECTED_IMAGES:
        raise RuntimeError(
            "Sharpness monotonic-image count mismatch"
        )

    if area_monotonic != fully_inside_count:
        raise RuntimeError(
            "Face-area monotonic-image count mismatch"
        )

    for row in rows:
        key = (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )

        if key not in baseline_lookup:
            raise RuntimeError(
                f"Missing baseline row for {key}"
            )

        baseline_row = baseline_lookup[
            key
        ]

        if (
            row[
                "experiment"
            ] == "brightness"
            and math.isclose(
                row[
                    "parameter_value"
                ],
                1.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            if not math.isclose(
                row[
                    "brightness"
                ],
                baseline_row[
                    "brightness"
                ],
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"Brightness baseline mismatch for {key}"
                )

        if (
            row[
                "experiment"
            ] == "sharpness"
            and int(
                row[
                    "parameter_value"
                ]
            ) == 1
        ):
            if not math.isclose(
                row[
                    "sharpness"
                ],
                baseline_row[
                    "sharpness"
                ],
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"Sharpness baseline mismatch for {key}"
                )

        if (
            (
                row[
                    "experiment"
                ] == "brightness"
                and math.isclose(
                    row[
                        "parameter_value"
                    ],
                    1.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
            or (
                row[
                    "experiment"
                ] == "sharpness"
                and int(
                    row[
                        "parameter_value"
                    ]
                ) == 1
            )
            or (
                row[
                    "experiment"
                ] == "face_area_ratio"
                and math.isclose(
                    row[
                        "parameter_value"
                    ],
                    1.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        ):
            if not math.isclose(
                row[
                    "face_area_ratio"
                ],
                baseline_row[
                    "face_area_ratio"
                ],
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    f"Face-area baseline mismatch for {key}"
                )

    return {
        "rows": len(rows),
        "experiment_counts": actual_counts,
        "brightness_monotonic_images": (
            brightness_monotonic
        ),
        "sharpness_monotonic_images": (
            sharpness_monotonic
        ),
        "face_area_monotonic_images": (
            area_monotonic
        ),
    }


def write_summary(
    path,
    results_path,
    fully_inside_count,
    elapsed_seconds,
):
    """Write the scientific summary from the staged detailed result table."""
    rows = parse_result_rows(
        results_path
    )

    baseline_rows = [
        row
        for row in rows
        if row[
            "experiment"
        ] == "baseline"
    ]

    baseline_brightness = np.asarray(
        [
            row[
                "brightness"
            ]
            for row in baseline_rows
        ],
        dtype=np.float64,
    )

    baseline_sharpness = np.asarray(
        [
            row[
                "sharpness"
            ]
            for row in baseline_rows
        ],
        dtype=np.float64,
    )

    baseline_area = np.asarray(
        [
            row[
                "face_area_ratio"
            ]
            for row in baseline_rows
        ],
        dtype=np.float64,
    )

    brightness_means = {}

    for factor in BRIGHTNESS_FACTORS:
        values = [
            row[
                "brightness"
            ]
            for row in rows
            if (
                row[
                    "experiment"
                ] == "brightness"
                and math.isclose(
                    row[
                        "parameter_value"
                    ],
                    factor,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        ]

        brightness_means[
            factor
        ] = float(
            np.mean(
                values
            )
        )

    sharpness_means = {}

    for kernel_size in BLUR_KERNEL_SIZES:
        values = [
            row[
                "sharpness"
            ]
            for row in rows
            if (
                row[
                    "experiment"
                ] == "sharpness"
                and int(
                    row[
                        "parameter_value"
                    ]
                ) == kernel_size
            )
        ]

        sharpness_means[
            kernel_size
        ] = float(
            np.mean(
                values
            )
        )

    area_means = {}

    for scale in AREA_SCALE_FACTORS:
        values = [
            row[
                "face_area_ratio"
            ]
            for row in rows
            if (
                row[
                    "experiment"
                ] == "face_area_ratio"
                and math.isclose(
                    row[
                        "parameter_value"
                    ],
                    scale,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            )
        ]

        area_means[
            scale
        ] = float(
            np.mean(
                values
            )
        )

    max_area_error = max(
        row[
            "face_area_ratio_absolute_error"
        ]
        for row in rows
    )

    lines = [
        "PhysioTrack Face Quality Validation",
        "====================================",
        "",
        "Scientific scope",
        "----------------",
        "This controlled evaluation validates the current FaceQuality",
        "descriptors on the complete 300-W evaluation set.",
        "",
        "Validated descriptors:",
        "- brightness",
        "- sharpness",
        "- face_area_ratio",
        "",
        "Detector confidence is preserved as an auxiliary field only.",
        "It is not treated as an independently validated FaceQuality descriptor",
        "because the value originates from the face detector.",
        "",
        "Dataset",
        "-------",
        f"Images: {EXPECTED_IMAGES}",
        f"Indoor images: {EXPECTED_SPLIT_COUNTS['Indoor']}",
        f"Outdoor images: {EXPECTED_SPLIT_COUNTS['Outdoor']}",
        "Annotation format: 68-point PTS",
        "Dataset handling: read-only",
        "",
        "Controlled face initialization",
        "------------------------------",
        "Face box: tight 68-point GT landmark box with 20% padding",
        f"Boxes fully inside image: {fully_inside_count}",
        f"Boxes touching image boundary: {EXPECTED_IMAGES - fully_inside_count}",
        "",
        "Baseline descriptor distribution",
        "--------------------------------",
        f"Brightness mean: {baseline_brightness.mean():.6f}",
        f"Brightness median: {np.median(baseline_brightness):.6f}",
        f"Brightness std: {baseline_brightness.std(ddof=1):.6f}",
        f"Brightness min: {baseline_brightness.min():.6f}",
        f"Brightness max: {baseline_brightness.max():.6f}",
        "",
        f"Sharpness mean: {baseline_sharpness.mean():.6f}",
        f"Sharpness median: {np.median(baseline_sharpness):.6f}",
        f"Sharpness std: {baseline_sharpness.std(ddof=1):.6f}",
        f"Sharpness min: {baseline_sharpness.min():.6f}",
        f"Sharpness max: {baseline_sharpness.max():.6f}",
        "",
        f"Face-area-ratio mean: {baseline_area.mean():.6f}",
        f"Face-area-ratio median: {np.median(baseline_area):.6f}",
        f"Face-area-ratio std: {baseline_area.std(ddof=1):.6f}",
        f"Face-area-ratio min: {baseline_area.min():.6f}",
        f"Face-area-ratio max: {baseline_area.max():.6f}",
        "",
        "Brightness sensitivity",
        "----------------------",
    ]

    for factor, mean_value in brightness_means.items():
        lines.append(
            f"Factor {factor:.2f}: mean brightness {mean_value:.6f}"
        )

    lines.extend(
        [
            f"Strictly increasing images: {EXPECTED_IMAGES}/{EXPECTED_IMAGES}",
            "",
            "Sharpness sensitivity",
            "---------------------",
        ]
    )

    for kernel_size, mean_value in sharpness_means.items():
        lines.append(
            f"Kernel {kernel_size}: mean sharpness {mean_value:.6f}"
        )

    lines.extend(
        [
            f"Strictly decreasing images: {EXPECTED_IMAGES}/{EXPECTED_IMAGES}",
            "",
            "Face-area-ratio validation",
            "--------------------------",
        ]
    )

    for scale, mean_value in area_means.items():
        lines.append(
            f"Scale {scale:.2f}: mean face-area ratio {mean_value:.6f}"
        )

    lines.extend(
        [
            f"Strictly increasing images: {fully_inside_count}/{fully_inside_count}",
            f"Maximum formula absolute error: {max_area_error:.12g}",
            "",
            "Interpretation",
            "--------------",
            "Brightness responds monotonically to controlled intensity scaling.",
            "Exact proportionality is not required at high factors because",
            "8-bit image values are clipped at 255.",
            "",
            "Sharpness is the variance of the grayscale Laplacian and decreases",
            "monotonically as controlled Gaussian blur increases.",
            "",
            "Face area ratio is the integer-clipped face-box area divided by",
            "full frame area. The controlled scale experiment uses only boxes",
            "fully inside the image to avoid clipping as a confounding factor.",
            "",
            f"Detailed result rows: {len(rows)}",
            f"Runtime seconds: {elapsed_seconds:.3f}",
            "",
            "Overall scientific status: PASS",
        ]
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def validate_summary(path):
    """Validate the staged summary before commit."""
    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    )

    required_fragments = (
        "PhysioTrack Face Quality Validation",
        f"Images: {EXPECTED_IMAGES}",
        "Strictly increasing images: 600/600",
        "Strictly decreasing images: 600/600",
        "Overall scientific status: PASS",
    )

    for fragment in required_fragments:
        if fragment not in text:
            raise RuntimeError(
                f"Missing required summary content: {fragment}"
            )


def commit_outputs(
    staging_dir,
):
    """Commit evaluator-owned outputs with rollback protection."""
    replacements = (
        (
            staging_dir
            / RESULTS_PATH.name,
            RESULTS_PATH,
        ),
        (
            staging_dir
            / SUMMARY_PATH.name,
            SUMMARY_PATH,
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
    """Run the complete Face Quality scientific evaluation."""
    print(
        "PhysioTrack Face Quality Validation"
    )
    print(
        "===================================="
    )
    print()
    print(
        "Running complete preflight..."
    )

    FaceQuality = resolve_face_quality()
    dataset_records = discover_dataset()
    quality = FaceQuality()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Preflight: PASS"
    )
    print(
        f"Dataset: {DATASET_ROOT}"
    )
    print(
        f"Images: {len(dataset_records)}"
    )
    print()
    print(
        "Running controlled FaceQuality evaluation..."
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_quality_eval_",
            dir=RESULTS_DIR,
        )
    )

    start_time = time.perf_counter()

    try:
        rows, fully_inside_count = (
            process_dataset(
                quality,
                dataset_records,
            )
        )

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        staged_results_path = (
            staging_dir
            / RESULTS_PATH.name
        )

        staged_summary_path = (
            staging_dir
            / SUMMARY_PATH.name
        )

        write_results(
            staged_results_path,
            rows,
        )

        print()
        print(
            "Validating staged evaluator outputs..."
        )

        validation = (
            validate_staged_results(
                staged_results_path,
                fully_inside_count,
            )
        )

        write_summary(
            staged_summary_path,
            staged_results_path,
            fully_inside_count,
            elapsed_seconds,
        )

        validate_summary(
            staged_summary_path
        )

        print(
            "Staged result rows:",
            validation[
                "rows"
            ],
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

        commit_outputs(
            staging_dir
        )

        print(
            "Committed final evaluator outputs."
        )
        print()
        print(
            f"Results: {RESULTS_PATH}"
        )
        print(
            f"Summary: {SUMMARY_PATH}"
        )
        print(
            "Overall scientific status: PASS"
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
