from pathlib import Path
import argparse
import csv
import json
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
COMPONENT_DIR = RESULTS_DIR / "component_execution"

RESULTS_PATH = (
    COMPONENT_DIR
    / "face_quality_component_results.csv"
)

SUMMARY_PATH = (
    COMPONENT_DIR
    / "face_quality_component_summary.json"
)

EXPECTED_IMAGES = 600
EXPECTED_SPLIT_COUNTS = {
    "Indoor": 300,
    "Outdoor": 300,
}

SMOKE_IMAGES = 8

RESULT_FIELDS = (
    "split",
    "image",
    "image_width",
    "image_height",
    "face_index",
    "status",
    "failure_reason",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "detector_confidence",
    "quality_available",
    "quality_confidence",
    "brightness",
    "sharpness",
    "face_area_ratio",
    "expected_brightness",
    "expected_sharpness",
    "expected_face_area_ratio",
    "brightness_absolute_error",
    "sharpness_absolute_error",
    "face_area_ratio_absolute_error",
    "confidence_absolute_error",
)

NUMERIC_TOLERANCE = 1e-12


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated execution of the real PhysioTrack "
            "Face Quality component through FaceAnalysis."
        )
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, dataset, and component configuration only.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            f"Run a deterministic {SMOKE_IMAGES}-image execution test "
            "without replacing final outputs."
        ),
    )

    return parser.parse_args()


def import_physio_track():
    """Import current PhysioTrack FaceAnalysis classes."""
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(
            f"PhysioTrack source directory not found: {SRC_ROOT}"
        )

    src_string = str(
        SRC_ROOT
    )

    if src_string not in sys.path:
        sys.path.insert(
            0,
            src_string,
        )

    from physiotrack.face.analysis import FaceAnalysis
    from physiotrack.face.config import FaceAnalysisConfig

    return (
        FaceAnalysis,
        FaceAnalysisConfig,
    )


def discover_dataset():
    """Discover and validate the complete 300-W image set."""
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
                f"Required 300-W split directory not found: {split_dir}"
            )

        image_paths = sorted(
            split_dir.glob(
                "*.png"
            )
        )

        expected_count = EXPECTED_SPLIT_COUNTS[
            split
        ]

        if len(
            image_paths
        ) != expected_count:
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
                    f"Missing paired PTS annotation: {pts_path}"
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

            records.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "pts_path": pts_path,
                    "image_width": int(
                        image_width
                    ),
                    "image_height": int(
                        image_height
                    ),
                }
            )

    if len(
        records
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "300-W total image count mismatch: "
            f"expected {EXPECTED_IMAGES}, found {len(records)}"
        )

    return records


def dataset_inventory():
    """Create a non-destructive inventory for dataset read-only verification."""
    inventory = []

    for path in sorted(
        item
        for item in DATASET_ROOT.rglob(
            "*"
        )
        if item.is_file()
    ):
        stat = path.stat()

        inventory.append(
            (
                str(
                    path.relative_to(
                        DATASET_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),
                int(
                    stat.st_size
                ),
                int(
                    stat.st_mtime_ns
                ),
            )
        )

    return inventory


def build_config(
    FaceAnalysisConfig,
):
    """Build the isolated Face Quality configuration."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=False,
        quality=True,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    config.validate()

    return config


def validate_component_isolation(
    analysis,
):
    """Verify that only detector plus Face Quality are active."""
    required_active = (
        "detector",
        "quality",
    )

    required_disabled = (
        "tracker",
        "orientation",
        "landmarks",
        "eyes",
        "blink",
        "gaze",
        "gaze_estimation",
        "mouth",
        "mouth_motion",
        "emotion",
        "regions",
        "temporal",
    )

    for attribute in required_active:
        if getattr(
            analysis,
            attribute,
        ) is None:
            raise RuntimeError(
                f"Required component is inactive: {attribute}"
            )

    for attribute in required_disabled:
        if getattr(
            analysis,
            attribute,
        ) is not None:
            raise RuntimeError(
                f"Unrelated component is unexpectedly active: {attribute}"
            )


def independent_quality_values(
    image,
    box,
):
    """Independently recompute the FaceQuality descriptor formulas."""
    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = [
        int(
            value
        )
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
        return None

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        gray.mean()
        / 255.0
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    face_area_ratio = float(
        (
            (x2 - x1)
            * (y2 - y1)
        )
        / float(
            image_width
            * image_height
        )
    )

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "face_area_ratio": face_area_ratio,
    }


def empty_measurements():
    """Return blank numerical fields for unavailable component output."""
    return {
        "box_x1": "",
        "box_y1": "",
        "box_x2": "",
        "box_y2": "",
        "detector_confidence": "",
        "quality_available": False,
        "quality_confidence": "",
        "brightness": "",
        "sharpness": "",
        "face_area_ratio": "",
        "expected_brightness": "",
        "expected_sharpness": "",
        "expected_face_area_ratio": "",
        "brightness_absolute_error": "",
        "sharpness_absolute_error": "",
        "face_area_ratio_absolute_error": "",
        "confidence_absolute_error": "",
    }


def process_image(
    analysis,
    item,
):
    """Run one image through the real FaceAnalysis Face Quality path."""
    image_path = item[
        "image_path"
    ]

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read image: {image_path}"
        )

    result = analysis.predict(
        image
    )

    instances = list(
        result.instances
    )

    if not instances:
        return [
            {
                "split": item[
                    "split"
                ],
                "image": image_path.name,
                "image_width": item[
                    "image_width"
                ],
                "image_height": item[
                    "image_height"
                ],
                "face_index": "",
                "status": "NO_FACE",
                "failure_reason": "",
                **empty_measurements(),
            }
        ]

    rows = []

    for face_index, instance in enumerate(
        instances
    ):
        box = instance.box

        if box is None:
            rows.append(
                {
                    "split": item[
                        "split"
                    ],
                    "image": image_path.name,
                    "image_width": item[
                        "image_width"
                    ],
                    "image_height": item[
                        "image_height"
                    ],
                    "face_index": face_index,
                    "status": "INVALID_FACE_BOX",
                    "failure_reason": "Detected face instance has no bounding box",
                    **empty_measurements(),
                }
            )
            continue

        box_values = [
            float(
                value
            )
            for value in box
        ]

        if (
            len(
                box_values
            )
            != 4
            or not all(
                math.isfinite(
                    value
                )
                for value in box_values
            )
        ):
            rows.append(
                {
                    "split": item[
                        "split"
                    ],
                    "image": image_path.name,
                    "image_width": item[
                        "image_width"
                    ],
                    "image_height": item[
                        "image_height"
                    ],
                    "face_index": face_index,
                    "status": "INVALID_FACE_BOX",
                    "failure_reason": "Detected face box is non-finite or malformed",
                    **empty_measurements(),
                }
            )
            continue

        detector_confidence = (
            None
            if instance.confidence is None
            else float(
                instance.confidence
            )
        )

        quality_features = (
            instance.face_features.get(
                "quality",
                {},
            )
            if instance.face_features
            is not None
            else {}
        )

        quality_available = bool(
            quality_features.get(
                "available",
                False,
            )
        )

        independent = independent_quality_values(
            image,
            box_values,
        )

        base_row = {
            "split": item[
                "split"
            ],
            "image": image_path.name,
            "image_width": item[
                "image_width"
            ],
            "image_height": item[
                "image_height"
            ],
            "face_index": face_index,
            "box_x1": box_values[
                0
            ],
            "box_y1": box_values[
                1
            ],
            "box_x2": box_values[
                2
            ],
            "box_y2": box_values[
                3
            ],
            "detector_confidence": (
                ""
                if detector_confidence is None
                else detector_confidence
            ),
            "quality_available": quality_available,
        }

        if (
            not quality_available
            or independent is None
        ):
            rows.append(
                {
                    **base_row,
                    "status": "QUALITY_UNAVAILABLE",
                    "failure_reason": (
                        "FaceQuality output unavailable or clipped face crop invalid"
                    ),
                    "quality_confidence": "",
                    "brightness": "",
                    "sharpness": "",
                    "face_area_ratio": "",
                    "expected_brightness": "",
                    "expected_sharpness": "",
                    "expected_face_area_ratio": "",
                    "brightness_absolute_error": "",
                    "sharpness_absolute_error": "",
                    "face_area_ratio_absolute_error": "",
                    "confidence_absolute_error": "",
                }
            )
            continue

        required_quality_keys = (
            "confidence",
            "brightness",
            "sharpness",
            "face_area_ratio",
        )

        if any(
            key not in quality_features
            for key in required_quality_keys
        ):
            raise RuntimeError(
                "Available quality output is missing required numerical fields"
            )

        quality_confidence = quality_features[
            "confidence"
        ]

        quality_confidence = (
            None
            if quality_confidence is None
            else float(
                quality_confidence
            )
        )

        brightness = float(
            quality_features[
                "brightness"
            ]
        )
        sharpness = float(
            quality_features[
                "sharpness"
            ]
        )
        face_area_ratio = float(
            quality_features[
                "face_area_ratio"
            ]
        )

        for key, value in (
            (
                "brightness",
                brightness,
            ),
            (
                "sharpness",
                sharpness,
            ),
            (
                "face_area_ratio",
                face_area_ratio,
            ),
        ):
            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    f"Non-finite FaceQuality value: {key}"
                )

        confidence_error = ""

        if (
            detector_confidence is not None
            and quality_confidence is not None
        ):
            confidence_error = abs(
                quality_confidence
                - detector_confidence
            )

        rows.append(
            {
                **base_row,
                "status": "QUALITY_AVAILABLE",
                "failure_reason": "",
                "quality_confidence": (
                    ""
                    if quality_confidence is None
                    else quality_confidence
                ),
                "brightness": brightness,
                "sharpness": sharpness,
                "face_area_ratio": face_area_ratio,
                "expected_brightness": independent[
                    "brightness"
                ],
                "expected_sharpness": independent[
                    "sharpness"
                ],
                "expected_face_area_ratio": independent[
                    "face_area_ratio"
                ],
                "brightness_absolute_error": abs(
                    brightness
                    - independent[
                        "brightness"
                    ]
                ),
                "sharpness_absolute_error": abs(
                    sharpness
                    - independent[
                        "sharpness"
                    ]
                ),
                "face_area_ratio_absolute_error": abs(
                    face_area_ratio
                    - independent[
                        "face_area_ratio"
                    ]
                ),
                "confidence_absolute_error": confidence_error,
            }
        )

    return rows


def execution_failure_row(
    item,
    error,
):
    """Create one explicit row for an image-level execution failure."""
    return {
        "split": item[
            "split"
        ],
        "image": item[
            "image_path"
        ].name,
        "image_width": item[
            "image_width"
        ],
        "image_height": item[
            "image_height"
        ],
        "face_index": "",
        "status": "EXECUTION_FAILED",
        "failure_reason": str(
            error
        ),
        **empty_measurements(),
    }


def run_execution(
    analysis,
    records,
):
    """Execute all requested images while preserving image-level failures."""
    rows = []

    for image_index, item in enumerate(
        records,
        start=1,
    ):
        try:
            image_rows = process_image(
                analysis,
                item,
            )

        except Exception as error:
            image_rows = [
                execution_failure_row(
                    item,
                    error,
                )
            ]

        rows.extend(
            image_rows
        )

        if (
            image_index % 50 == 0
            or image_index
            == len(
                records
            )
        ):
            print(
                f"Processed {image_index}/{len(records)} images"
            )

    return rows


def validate_execution_rows(
    rows,
    expected_records,
):
    """Validate coverage, schema, status accounting, and numerical outputs."""
    allowed_statuses = {
        "QUALITY_AVAILABLE",
        "QUALITY_UNAVAILABLE",
        "INVALID_FACE_BOX",
        "NO_FACE",
        "EXECUTION_FAILED",
    }

    image_keys = {
        (
            item[
                "split"
            ],
            item[
                "image_path"
            ].name,
        )
        for item in expected_records
    }

    observed_image_keys = {
        (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )
        for row in rows
    }

    if observed_image_keys != image_keys:
        missing = sorted(
            image_keys
            - observed_image_keys
        )
        unexpected = sorted(
            observed_image_keys
            - image_keys
        )

        raise RuntimeError(
            "Image coverage mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    row_keys = set()
    status_counts = {
        status: 0
        for status in allowed_statuses
    }

    available_rows = []

    for row_index, row in enumerate(
        rows
    ):
        status = row[
            "status"
        ]

        if status not in allowed_statuses:
            raise RuntimeError(
                f"Unexpected component status: {status}"
            )

        status_counts[
            status
        ] += 1

        face_index = row[
            "face_index"
        ]

        uniqueness_key = (
            row[
                "split"
            ],
            row[
                "image"
            ],
            str(
                face_index
            ),
        )

        if uniqueness_key in row_keys:
            raise RuntimeError(
                f"Duplicate component row key: {uniqueness_key}"
            )

        row_keys.add(
            uniqueness_key
        )

        if status == "QUALITY_AVAILABLE":
            available_rows.append(
                row
            )

            if not bool(
                row[
                    "quality_available"
                ]
            ):
                raise RuntimeError(
                    "QUALITY_AVAILABLE row has quality_available=False"
                )

            for key in (
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "brightness",
                "sharpness",
                "face_area_ratio",
                "expected_brightness",
                "expected_sharpness",
                "expected_face_area_ratio",
                "brightness_absolute_error",
                "sharpness_absolute_error",
                "face_area_ratio_absolute_error",
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
                        f"Non-finite available-row value in {key}"
                    )

            if not (
                0.0
                <= float(
                    row[
                        "brightness"
                    ]
                )
                <= 1.0
            ):
                raise RuntimeError(
                    "Brightness outside [0, 1]"
                )

            if float(
                row[
                    "sharpness"
                ]
            ) < 0.0:
                raise RuntimeError(
                    "Sharpness is negative"
                )

            if not (
                0.0
                < float(
                    row[
                        "face_area_ratio"
                    ]
                )
                <= 1.0
            ):
                raise RuntimeError(
                    "Face area ratio outside (0, 1]"
                )

            for error_key in (
                "brightness_absolute_error",
                "sharpness_absolute_error",
                "face_area_ratio_absolute_error",
            ):
                if float(
                    row[
                        error_key
                    ]
                ) > NUMERIC_TOLERANCE:
                    raise RuntimeError(
                        f"FaceQuality numerical mismatch in {error_key}"
                    )

            if row[
                "confidence_absolute_error"
            ] != "":
                if float(
                    row[
                        "confidence_absolute_error"
                    ]
                ) > NUMERIC_TOLERANCE:
                    raise RuntimeError(
                        "Detector confidence was not preserved by FaceQuality"
                    )

    if not available_rows:
        raise RuntimeError(
            "No numerical FaceQuality output was produced"
        )

    split_image_counts = {
        split: len(
            {
                row[
                    "image"
                ]
                for row in rows
                if row[
                    "split"
                ] == split
            }
        )
        for split in EXPECTED_SPLIT_COUNTS
    }

    expected_split_counts = {
        split: sum(
            1
            for item in expected_records
            if item[
                "split"
            ] == split
        )
        for split in EXPECTED_SPLIT_COUNTS
    }

    if split_image_counts != expected_split_counts:
        raise RuntimeError(
            "Split coverage mismatch: "
            f"expected {expected_split_counts}, got {split_image_counts}"
        )

    image_status_sets = {}

    for row in rows:
        key = (
            row[
                "split"
            ],
            row[
                "image"
            ],
        )

        image_status_sets.setdefault(
            key,
            set(),
        ).add(
            row[
                "status"
            ]
        )

    for key, statuses in image_status_sets.items():
        if (
            "NO_FACE" in statuses
            and len(
                statuses
            ) != 1
        ):
            raise RuntimeError(
                f"NO_FACE mixed with face rows for image {key}"
            )

        if (
            "EXECUTION_FAILED" in statuses
            and len(
                statuses
            ) != 1
        ):
            raise RuntimeError(
                f"EXECUTION_FAILED mixed with successful rows for image {key}"
            )

    return {
        "total_rows": len(
            rows
        ),
        "total_images": len(
            observed_image_keys
        ),
        "status_counts": status_counts,
        "quality_available_rows": len(
            available_rows
        ),
        "images_with_quality": len(
            {
                (
                    row[
                        "split"
                    ],
                    row[
                        "image"
                    ],
                )
                for row in available_rows
            }
        ),
        "detected_face_rows": sum(
            status_counts[
                status
            ]
            for status in (
                "QUALITY_AVAILABLE",
                "QUALITY_UNAVAILABLE",
                "INVALID_FACE_BOX",
            )
        ),
        "max_brightness_absolute_error": max(
            float(
                row[
                    "brightness_absolute_error"
                ]
            )
            for row in available_rows
        ),
        "max_sharpness_absolute_error": max(
            float(
                row[
                    "sharpness_absolute_error"
                ]
            )
            for row in available_rows
        ),
        "max_face_area_ratio_absolute_error": max(
            float(
                row[
                    "face_area_ratio_absolute_error"
                ]
            )
            for row in available_rows
        ),
        "max_confidence_absolute_error": max(
            [
                float(
                    row[
                        "confidence_absolute_error"
                    ]
                )
                for row in available_rows
                if row[
                    "confidence_absolute_error"
                ]
                != ""
            ],
            default=0.0,
        ),
    }


def write_results(
    path,
    rows,
):
    """Write detailed isolated component execution results."""
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


def read_results(
    path,
):
    """Read component execution results from CSV."""
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


def validate_staged_csv(
    path,
    expected_records,
):
    """Revalidate the serialized staged CSV before final replacement."""
    raw_rows = read_results(
        path
    )

    parsed_rows = []

    for raw in raw_rows:
        row = dict(
            raw
        )

        row[
            "quality_available"
        ] = (
            str(
                raw[
                    "quality_available"
                ]
            ).strip().lower()
            == "true"
        )

        parsed_rows.append(
            row
        )

    return validate_execution_rows(
        parsed_rows,
        expected_records,
    )


def build_summary(
    validation,
    elapsed_seconds,
    pre_inventory,
    post_inventory,
):
    """Build a software-execution summary without benchmark accuracy claims."""
    return {
        "component": "FaceQuality",
        "evidence_type": "isolated_component_execution",
        "scientific_accuracy_evidence": False,
        "execution_path": "FaceAnalysis",
        "active_components": [
            "Face detector",
            "FaceQuality",
        ],
        "disabled_optional_components": [
            "tracking",
            "head_pose",
            "landmarks",
            "eyes",
            "blink",
            "gaze",
            "gaze_estimation",
            "mouth",
            "mouth_motion",
            "emotion",
            "regions",
            "temporal",
        ],
        "dataset": {
            "name": "300-W",
            "images": EXPECTED_IMAGES,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "read_only_verified": (
                pre_inventory
                == post_inventory
            ),
        },
        "execution": {
            "total_images": validation[
                "total_images"
            ],
            "total_rows": validation[
                "total_rows"
            ],
            "detected_face_rows": validation[
                "detected_face_rows"
            ],
            "quality_available_rows": validation[
                "quality_available_rows"
            ],
            "images_with_quality": validation[
                "images_with_quality"
            ],
            "status_counts": validation[
                "status_counts"
            ],
            "runtime_seconds": elapsed_seconds,
        },
        "numerical_consistency": {
            "maximum_brightness_absolute_error": validation[
                "max_brightness_absolute_error"
            ],
            "maximum_sharpness_absolute_error": validation[
                "max_sharpness_absolute_error"
            ],
            "maximum_face_area_ratio_absolute_error": validation[
                "max_face_area_ratio_absolute_error"
            ],
            "maximum_confidence_absolute_error": validation[
                "max_confidence_absolute_error"
            ],
            "tolerance": NUMERIC_TOLERANCE,
        },
        "interpretation": (
            "This test verifies real numerical execution and export of "
            "FaceQuality through the current FaceAnalysis path. "
            "It is software execution evidence and does not replace the "
            "controlled scientific Face Quality validation."
        ),
        "status": "PASS",
    }


def write_summary(
    path,
    summary,
):
    """Write component execution summary JSON."""
    path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def validate_summary(
    path,
):
    """Validate the staged summary JSON."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(
            handle
        )

    if summary.get(
        "component"
    ) != "FaceQuality":
        raise RuntimeError(
            "Unexpected component name in summary"
        )

    if summary.get(
        "evidence_type"
    ) != "isolated_component_execution":
        raise RuntimeError(
            "Unexpected evidence type in summary"
        )

    if summary.get(
        "scientific_accuracy_evidence"
    ) is not False:
        raise RuntimeError(
            "Component execution summary must not claim scientific accuracy"
        )

    if summary.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Component execution summary status is not PASS"
        )

    dataset = summary.get(
        "dataset",
        {},
    )

    if not dataset.get(
        "read_only_verified",
        False,
    ):
        raise RuntimeError(
            "Dataset read-only verification failed"
        )


def commit_outputs(
    staging_dir,
):
    """Commit component-owned outputs with transactional rollback."""
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


def run_preflight():
    """Run package-level component execution preflight."""
    (
        FaceAnalysis,
        FaceAnalysisConfig,
    ) = import_physio_track()

    dataset_records = discover_dataset()

    config = build_config(
        FaceAnalysisConfig
    )

    analysis = FaceAnalysis(
        config=config,
        device="cpu",
        verbose=False,
    )

    try:
        validate_component_isolation(
            analysis
        )

    finally:
        analysis.close()

    return (
        FaceAnalysis,
        FaceAnalysisConfig,
        dataset_records,
    )


def run_smoke_test(
    FaceAnalysis,
    FaceAnalysisConfig,
    dataset_records,
):
    """Run deterministic real FaceAnalysis execution without final outputs."""
    smoke_records = (
        dataset_records[
            : SMOKE_IMAGES // 2
        ]
        + dataset_records[
            EXPECTED_SPLIT_COUNTS[
                "Indoor"
            ]:
            EXPECTED_SPLIT_COUNTS[
                "Indoor"
            ]
            + SMOKE_IMAGES // 2
        ]
    )

    config = build_config(
        FaceAnalysisConfig
    )

    analysis = FaceAnalysis(
        config=config,
        device="cpu",
        verbose=False,
    )

    try:
        validate_component_isolation(
            analysis
        )

        rows = run_execution(
            analysis,
            smoke_records,
        )

    finally:
        analysis.close()

    validation = validate_execution_rows(
        rows,
        smoke_records,
    )

    print()
    print(
        "Smoke test status: PASS"
    )
    print(
        f"Smoke images: {validation['total_images']}"
    )
    print(
        f"Smoke rows: {validation['total_rows']}"
    )
    print(
        "Quality-available rows:",
        validation[
            "quality_available_rows"
        ],
    )
    print(
        "Status counts:",
        validation[
            "status_counts"
        ],
    )


def run_full(
    FaceAnalysis,
    FaceAnalysisConfig,
    dataset_records,
):
    """Run full isolated component execution with safe staged replacement."""
    COMPONENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_quality_component_",
            dir=COMPONENT_DIR,
        )
    )

    pre_inventory = dataset_inventory()

    config = build_config(
        FaceAnalysisConfig
    )

    analysis = FaceAnalysis(
        config=config,
        device="cpu",
        verbose=False,
    )

    start_time = time.perf_counter()

    try:
        validate_component_isolation(
            analysis
        )

        rows = run_execution(
            analysis,
            dataset_records,
        )

    finally:
        analysis.close()

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    post_inventory = dataset_inventory()

    if pre_inventory != post_inventory:
        raise RuntimeError(
            "300-W dataset inventory changed during component execution"
        )

    staged_results_path = (
        staging_dir
        / RESULTS_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_PATH.name
    )

    try:
        validation = validate_execution_rows(
            rows,
            dataset_records,
        )

        write_results(
            staged_results_path,
            rows,
        )

        print()
        print(
            "Validating staged component outputs..."
        )

        staged_validation = validate_staged_csv(
            staged_results_path,
            dataset_records,
        )

        if staged_validation != validation:
            raise RuntimeError(
                "Serialized staged validation differs from in-memory validation"
            )

        summary = build_summary(
            validation,
            elapsed_seconds,
            pre_inventory,
            post_inventory,
        )

        write_summary(
            staged_summary_path,
            summary,
        )

        validate_summary(
            staged_summary_path
        )

        print(
            "Images:",
            validation[
                "total_images"
            ],
        )
        print(
            "Rows:",
            validation[
                "total_rows"
            ],
        )
        print(
            "Detected face rows:",
            validation[
                "detected_face_rows"
            ],
        )
        print(
            "Quality-available rows:",
            validation[
                "quality_available_rows"
            ],
        )
        print(
            "Images with quality:",
            validation[
                "images_with_quality"
            ],
        )
        print(
            "Status counts:",
            validation[
                "status_counts"
            ],
        )
        print(
            "Maximum brightness error:",
            validation[
                "max_brightness_absolute_error"
            ],
        )
        print(
            "Maximum sharpness error:",
            validation[
                "max_sharpness_absolute_error"
            ],
        )
        print(
            "Maximum face-area-ratio error:",
            validation[
                "max_face_area_ratio_absolute_error"
            ],
        )
        print(
            "Maximum confidence error:",
            validation[
                "max_confidence_absolute_error"
            ],
        )
        print(
            "Dataset read-only verification: PASS"
        )

        commit_outputs(
            staging_dir
        )

        print(
            "Committed final component outputs."
        )
        print()
        print(
            f"Results: {RESULTS_PATH}"
        )
        print(
            f"Summary: {SUMMARY_PATH}"
        )
        print(
            "Isolated component execution status: PASS"
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


def main():
    """Run preflight, smoke, or full isolated Face Quality execution."""
    args = parse_args()

    print(
        "PhysioTrack Face Quality Isolated Component Execution"
    )
    print(
        "====================================================="
    )
    print()
    print(
        "Running complete preflight..."
    )

    (
        FaceAnalysis,
        FaceAnalysisConfig,
        dataset_records,
    ) = run_preflight()

    print(
        "Preflight: PASS"
    )
    print(
        f"Dataset: {DATASET_ROOT}"
    )
    print(
        f"Images: {len(dataset_records)}"
    )
    print(
        "Active path: Face detector -> FaceQuality"
    )
    print(
        "All unrelated optional FaceAnalysis components: disabled"
    )

    if args.preflight_only:
        print()
        print(
            "Preflight-only status: PASS"
        )
        return

    if args.smoke_test:
        print()
        print(
            f"Running deterministic {SMOKE_IMAGES}-image smoke test..."
        )

        run_smoke_test(
            FaceAnalysis,
            FaceAnalysisConfig,
            dataset_records,
        )
        return

    print()
    print(
        "Running full isolated component execution..."
    )

    run_full(
        FaceAnalysis,
        FaceAnalysisConfig,
        dataset_records,
    )


if __name__ == "__main__":
    main()
