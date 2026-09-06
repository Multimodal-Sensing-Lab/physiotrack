from pathlib import Path
import csv
import json
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.results import Instance, Result


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "CelebAMask-HQ"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "CelebA-HQ-img"
)

MAPPING_PATH = (
    DATASET_ROOT
    / "CelebA-HQ-to-CelebA-mapping.txt"
)

PARTITION_PATH = (
    SCRIPT_DIR
    / "list_eval_partition.txt"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "component_execution"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "face_regions_component_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "face_regions_component_summary.json"
)

TEST_PARTITION = 2
EXPECTED_TEST_IMAGES = 2824
INPUT_SIZE = 512

EXPECTED_FOREGROUND_CLASSES = [
    "neck",
    "skin",
    "cloth",
    "l_ear",
    "r_ear",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "l_lip",
    "u_lip",
    "hair",
    "eye_g",
    "hat",
    "ear_r",
    "neck_l",
]


class ControlledFullImageDetector:
    """Provide the controlled full-image face box used by the benchmark."""

    def predict(
        self,
        frame,
    ):
        height, width = (
            frame.shape[:2]
        )

        instance = Instance(
            box=np.array(
                [
                    0.0,
                    0.0,
                    float(width),
                    float(height),
                ],
                dtype=float,
            ),
            confidence=1.0,
            cls=0,
            cls_name="face",
        )

        return Result(
            orig_img=frame,
            instances=[
                instance
            ],
            task="face",
        )


class RegionCapture:
    """Capture the real FaceRegions output used by FaceAnalysis."""

    def __init__(
        self,
        regions,
    ):
        self.regions = regions
        self.last_output = None

    def predict(
        self,
        frame,
        boxes=None,
    ):
        output = (
            self.regions.predict(
                frame,
                boxes=boxes,
            )
        )

        self.last_output = output

        return output

    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.regions,
            name,
        )


def validate_required_paths():
    """Validate the dataset and static validation metadata."""
    required_paths = {
        "image directory": IMAGE_DIR,
        "HQ-to-CelebA mapping": MAPPING_PATH,
        "CelebA partition file": PARTITION_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required Face Regions component-test inputs:\n"
            + "\n".join(
                missing
            )
        )


def load_partition_file():
    """Load the official CelebA partition metadata."""
    partitions = {}

    with PARTITION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid partition line {line_number}: {line}"
                )

            filename, partition = parts
            partition = int(
                partition
            )

            if partition not in {
                0,
                1,
                2,
            }:
                raise ValueError(
                    f"Invalid partition value at line "
                    f"{line_number}: {partition}"
                )

            if filename in partitions:
                raise ValueError(
                    f"Duplicate partition entry for {filename}"
                )

            partitions[
                filename
            ] = partition

    if len(
        partitions
    ) != 202599:
        raise RuntimeError(
            f"Expected 202599 CelebA partition entries, "
            f"found {len(partitions)}."
        )

    return partitions


def load_test_hq_indices():
    """Derive the same official CelebAMask-HQ test subset as the benchmark."""
    partitions = (
        load_partition_file()
    )

    test_indices = []

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        file.readline()

        for line_number, line in enumerate(
            file,
            start=2,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid mapping line {line_number}: {line}"
                )

            hq_index = int(
                parts[
                    0
                ]
            )

            original_filename = (
                parts[
                    2
                ]
            )

            if original_filename not in partitions:
                raise KeyError(
                    f"Missing partition entry for "
                    f"{original_filename}"
                )

            if (
                partitions[
                    original_filename
                ]
                == TEST_PARTITION
            ):
                test_indices.append(
                    hq_index
                )

    if len(
        set(
            test_indices
        )
    ) != len(
        test_indices
    ):
        raise RuntimeError(
            "Duplicate CelebAMask-HQ indices were found in the test split."
        )

    if len(
        test_indices
    ) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} test images, "
            f"found {len(test_indices)}."
        )

    return test_indices


def preflight_images(
    test_indices,
):
    """Verify that all required test images exist before execution."""
    missing = []

    for image_id in test_indices:
        image_path = (
            IMAGE_DIR
            / f"{image_id}.jpg"
        )

        if not image_path.is_file():
            missing.append(
                str(
                    image_path
                )
            )

    if missing:
        preview = "\n".join(
            missing[
                :20
            ]
        )

        raise FileNotFoundError(
            f"Missing {len(missing)} required test images.\n"
            f"{preview}"
        )


def load_image(
    image_id,
):
    """Load and resize one benchmark image exactly as the scientific evaluator."""
    image_path = (
        IMAGE_DIR
        / f"{image_id}.jpg"
    )

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    image = cv2.resize(
        image,
        (
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return (
        image_path,
        image,
    )


def make_config():
    """Enable only Face Regions in the modular PhysioTrack pipeline."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=True,
        temporal=False,
    )

    config.validate()

    return config


def get_raw_region_face(
    capture,
):
    """Return the single native FaceRegions face output captured from FaceAnalysis."""
    output = (
        capture.last_output
    )

    if not isinstance(
        output,
        dict,
    ):
        raise RuntimeError(
            "Captured FaceRegions output is not a dictionary."
        )

    faces = output.get(
        "faces"
    )

    if not isinstance(
        faces,
        list,
    ):
        raise RuntimeError(
            "Captured FaceRegions output does not contain a face list."
        )

    if len(
        faces
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one captured region face, found {len(faces)}."
        )

    region_face = (
        faces[
            0
        ]
    )

    if not isinstance(
        region_face,
        dict,
    ):
        raise RuntimeError(
            "Captured region face is not a dictionary."
        )

    return region_face


def collect_region_rows(
    image_id,
    image_path,
    frame,
    prediction,
    capture,
):
    """Collect and cross-check real FaceAnalysis and native FaceRegions outputs."""
    if len(
        prediction
    ) != 1:
        raise RuntimeError(
            f"Expected exactly one FaceAnalysis instance, found {len(prediction)}."
        )

    face = prediction[
        0
    ]

    features = (
        face.face_features
        if isinstance(
            face.face_features,
            dict,
        )
        else {}
    )

    regions = features.get(
        "regions",
        {}
    )

    if not isinstance(
        regions,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis regions feature is not a dictionary."
        )

    raw_face = (
        get_raw_region_face(
            capture
        )
    )

    raw_regions = raw_face.get(
        "regions"
    )

    if not isinstance(
        raw_regions,
        dict,
    ):
        raise RuntimeError(
            "Captured FaceRegions face does not contain a regions dictionary."
        )

    pipeline_available = bool(
        regions.get(
            "available",
            False,
        )
    )

    pipeline_counts = regions.get(
        "pixel_counts",
        {}
    )

    if not isinstance(
        pipeline_counts,
        dict,
    ):
        raise RuntimeError(
            "FaceAnalysis regions pixel_counts is not a dictionary."
        )

    raw_counts = {}

    for (
        region_name,
        mask,
    ) in raw_regions.items():
        mask_array = np.asarray(
            mask
        )

        if mask_array.ndim != 2:
            raise RuntimeError(
                f"Region mask {region_name!r} has unexpected shape "
                f"{mask_array.shape}."
            )

        if not np.all(
            np.isfinite(
                mask_array
            )
        ):
            raise RuntimeError(
                f"Region mask {region_name!r} contains non-finite values."
            )

        raw_counts[
            str(
                region_name
            )
        ] = int(
            mask_array.sum()
        )

    normalized_pipeline_counts = {
        str(
            name
        ): int(
            count
        )
        for (
            name,
            count,
        ) in pipeline_counts.items()
    }

    if (
        normalized_pipeline_counts
        != raw_counts
    ):
        raise RuntimeError(
            "FaceAnalysis region pixel counts do not match the native "
            "FaceRegions mask sums."
        )

    pipeline_classes = [
        str(
            name
        )
        for name in regions.get(
            "classes",
            []
        )
    ]

    if pipeline_classes != list(
        normalized_pipeline_counts.keys()
    ):
        raise RuntimeError(
            "FaceAnalysis region class list does not match pixel-count keys."
        )

    if pipeline_available != bool(
        raw_counts
    ):
        raise RuntimeError(
            "FaceAnalysis regions availability is inconsistent with "
            "the captured FaceRegions output."
        )

    raw_box = np.asarray(
        raw_face.get(
            "box"
        ),
        dtype=float,
    ).reshape(
        -1
    )

    if raw_box.size != 4:
        raise RuntimeError(
            "Captured FaceRegions box does not contain four coordinates."
        )

    if not np.all(
        np.isfinite(
            raw_box
        )
    ):
        raise RuntimeError(
            "Captured FaceRegions box contains non-finite values."
        )

    face_box = np.asarray(
        face.box,
        dtype=float,
    ).reshape(
        -1
    )

    if face_box.size != 4:
        raise RuntimeError(
            "FaceAnalysis face box does not contain four coordinates."
        )

    association_iou = regions.get(
        "association_iou"
    )

    if association_iou is None:
        raise RuntimeError(
            "FaceAnalysis did not associate the captured FaceRegions output."
        )

    association_iou = float(
        association_iou
    )

    if not np.isfinite(
        association_iou
    ):
        raise RuntimeError(
            "FaceAnalysis region association IoU is non-finite."
        )

    if not (
        0.0
        < association_iou
        <= 1.0
    ):
        raise RuntimeError(
            f"Invalid FaceAnalysis region association IoU: {association_iou}"
        )

    height, width = (
        frame.shape[:2]
    )

    expected_box = np.array(
        [
            0.0,
            0.0,
            float(
                width
            ),
            float(
                height
            ),
        ],
        dtype=float,
    )

    if not np.allclose(
        face_box,
        expected_box,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            f"Controlled FaceAnalysis box changed unexpectedly: {face_box}"
        )

    skin_pixel_count = int(
        regions.get(
            "skin_pixel_count",
            0,
        )
    )

    if skin_pixel_count != normalized_pipeline_counts.get(
        "skin",
        0,
    ):
        raise RuntimeError(
            "FaceAnalysis skin_pixel_count does not match pixel_counts['skin']."
        )

    region_width = max(
        0,
        int(
            raw_box[
                2
            ]
            - raw_box[
                0
            ]
        ),
    )

    region_height = max(
        0,
        int(
            raw_box[
                3
            ]
            - raw_box[
                1
            ]
        ),
    )

    region_area = (
        region_width
        * region_height
    )

    expected_skin_fraction = (
        float(
            skin_pixel_count
            / region_area
        )
        if region_area > 0
        else None
    )

    skin_fraction = regions.get(
        "skin_fraction"
    )

    if expected_skin_fraction is None:
        if skin_fraction is not None:
            raise RuntimeError(
                "FaceAnalysis skin_fraction should be None for zero region area."
            )
    else:
        if skin_fraction is None:
            raise RuntimeError(
                "FaceAnalysis skin_fraction is unexpectedly None."
            )

        if not np.isclose(
            float(
                skin_fraction
            ),
            expected_skin_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "FaceAnalysis skin_fraction does not match the captured "
                "FaceRegions pixel counts and region area."
            )

    if raw_counts:
        rows = []

        for region_name in sorted(
            raw_counts
        ):
            rows.append(
                {
                    "image_id": image_id,
                    "source_image": image_path.name,
                    "image_width": width,
                    "image_height": height,
                    "input_box_source": "full_aligned_image",
                    "input_box_x1": face_box[0],
                    "input_box_y1": face_box[1],
                    "input_box_x2": face_box[2],
                    "input_box_y2": face_box[3],
                    "region_box_x1": raw_box[0],
                    "region_box_y1": raw_box[1],
                    "region_box_x2": raw_box[2],
                    "region_box_y2": raw_box[3],
                    "association_iou": association_iou,
                    "pipeline_regions_available": pipeline_available,
                    "region_name": region_name,
                    "native_pixel_count": raw_counts[
                        region_name
                    ],
                    "pipeline_pixel_count": normalized_pipeline_counts[
                        region_name
                    ],
                    "pixel_count_match": True,
                    "skin_pixel_count": skin_pixel_count,
                    "skin_fraction": (
                        ""
                        if skin_fraction is None
                        else float(
                            skin_fraction
                        )
                    ),
                    "status": "OK",
                    "failure_reason": "",
                }
            )

        return rows

    return [
        {
            "image_id": image_id,
            "source_image": image_path.name,
            "image_width": width,
            "image_height": height,
            "input_box_source": "full_aligned_image",
            "input_box_x1": face_box[0],
            "input_box_y1": face_box[1],
            "input_box_x2": face_box[2],
            "input_box_y2": face_box[3],
            "region_box_x1": raw_box[0],
            "region_box_y1": raw_box[1],
            "region_box_x2": raw_box[2],
            "region_box_y2": raw_box[3],
            "association_iou": association_iou,
            "pipeline_regions_available": False,
            "region_name": "",
            "native_pixel_count": "",
            "pipeline_pixel_count": "",
            "pixel_count_match": "",
            "skin_pixel_count": 0,
            "skin_fraction": "",
            "status": "NO_REGIONS",
            "failure_reason": "",
        }
    ]


def failure_row(
    image_id,
    error,
):
    """Create one structured row for a genuine execution failure."""
    return {
        "image_id": image_id,
        "source_image": f"{image_id}.jpg",
        "image_width": "",
        "image_height": "",
        "input_box_source": "full_aligned_image",
        "input_box_x1": "",
        "input_box_y1": "",
        "input_box_x2": "",
        "input_box_y2": "",
        "region_box_x1": "",
        "region_box_y1": "",
        "region_box_x2": "",
        "region_box_y2": "",
        "association_iou": "",
        "pipeline_regions_available": "",
        "region_name": "",
        "native_pixel_count": "",
        "pipeline_pixel_count": "",
        "pixel_count_match": "",
        "skin_pixel_count": "",
        "skin_fraction": "",
        "status": "EXECUTION_FAILED",
        "failure_reason": str(
            error
        ),
    }


def write_results_csv(
    output_path,
    rows,
):
    """Write the structured component-execution table."""
    fieldnames = [
        "image_id",
        "source_image",
        "image_width",
        "image_height",
        "input_box_source",
        "input_box_x1",
        "input_box_y1",
        "input_box_x2",
        "input_box_y2",
        "region_box_x1",
        "region_box_y1",
        "region_box_x2",
        "region_box_y2",
        "association_iou",
        "pipeline_regions_available",
        "region_name",
        "native_pixel_count",
        "pipeline_pixel_count",
        "pixel_count_match",
        "skin_pixel_count",
        "skin_fraction",
        "status",
        "failure_reason",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def validate_staged_results(
    results_path,
    summary_path,
):
    """Independently validate staged component-execution artifacts."""
    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:
        raise RuntimeError(
            "Staged component-execution CSV is empty."
        )

    by_image = {}

    for row in rows:
        image_id = int(
            row[
                "image_id"
            ]
        )

        by_image.setdefault(
            image_id,
            []
        ).append(
            row
        )

    if len(
        by_image
    ) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} unique images in the staged CSV, "
            f"found {len(by_image)}."
        )

    execution_failed = 0
    images_with_regions = 0
    images_without_regions = 0
    region_rows = 0
    total_native_pixels = 0
    observed_classes = set()

    for (
        image_id,
        image_rows,
    ) in by_image.items():
        statuses = {
            row[
                "status"
            ]
            for row in image_rows
        }

        if statuses == {
            "EXECUTION_FAILED"
        }:
            if len(
                image_rows
            ) != 1:
                raise RuntimeError(
                    f"HQ {image_id} has multiple execution-failure rows."
                )

            execution_failed += 1
            continue

        if statuses == {
            "NO_REGIONS"
        }:
            if len(
                image_rows
            ) != 1:
                raise RuntimeError(
                    f"HQ {image_id} has multiple NO_REGIONS rows."
                )

            images_without_regions += 1
            continue

        if statuses != {
            "OK"
        }:
            raise RuntimeError(
                f"HQ {image_id} has mixed or invalid statuses: {statuses}"
            )

        images_with_regions += 1

        region_names = [
            row[
                "region_name"
            ]
            for row in image_rows
        ]

        if len(
            region_names
        ) != len(
            set(
                region_names
            )
        ):
            raise RuntimeError(
                f"HQ {image_id} contains duplicate region rows."
            )

        for row in image_rows:
            width = int(
                row[
                    "image_width"
                ]
            )

            height = int(
                row[
                    "image_height"
                ]
            )

            if (
                width != INPUT_SIZE
                or height != INPUT_SIZE
            ):
                raise RuntimeError(
                    f"HQ {image_id} has unexpected image size "
                    f"{width} x {height}."
                )

            input_box = np.array(
                [
                    float(
                        row[
                            "input_box_x1"
                        ]
                    ),
                    float(
                        row[
                            "input_box_y1"
                        ]
                    ),
                    float(
                        row[
                            "input_box_x2"
                        ]
                    ),
                    float(
                        row[
                            "input_box_y2"
                        ]
                    ),
                ],
                dtype=float,
            )

            expected_box = np.array(
                [
                    0.0,
                    0.0,
                    float(
                        INPUT_SIZE
                    ),
                    float(
                        INPUT_SIZE
                    ),
                ],
                dtype=float,
            )

            if not np.allclose(
                input_box,
                expected_box,
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"HQ {image_id} has an incorrect controlled input box."
                )

            association_iou = float(
                row[
                    "association_iou"
                ]
            )

            if not (
                0.0
                < association_iou
                <= 1.0
            ):
                raise RuntimeError(
                    f"HQ {image_id} has invalid association IoU."
                )

            native_count = int(
                row[
                    "native_pixel_count"
                ]
            )

            pipeline_count = int(
                row[
                    "pipeline_pixel_count"
                ]
            )

            if native_count < 0:
                raise RuntimeError(
                    f"HQ {image_id} has a negative region pixel count."
                )

            if native_count != pipeline_count:
                raise RuntimeError(
                    f"HQ {image_id} has inconsistent native and "
                    "FaceAnalysis pixel counts."
                )

            if row[
                "pixel_count_match"
            ].strip().lower() not in {
                "true",
                "1",
            }:
                raise RuntimeError(
                    f"HQ {image_id} has an unsuccessful pixel-count match."
                )

            skin_count = int(
                row[
                    "skin_pixel_count"
                ]
            )

            if skin_count < 0:
                raise RuntimeError(
                    f"HQ {image_id} has a negative skin pixel count."
                )

            skin_fraction_text = row[
                "skin_fraction"
            ].strip()

            if skin_fraction_text:
                skin_fraction = float(
                    skin_fraction_text
                )

                if not (
                    0.0
                    <= skin_fraction
                    <= 1.0
                ):
                    raise RuntimeError(
                        f"HQ {image_id} has invalid skin fraction."
                    )

            observed_classes.add(
                row[
                    "region_name"
                ]
            )

            total_native_pixels += (
                native_count
            )

            region_rows += 1

    if (
        images_with_regions
        + images_without_regions
        + execution_failed
        != EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Staged component-execution image accounting is inconsistent."
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    expected_summary_values = {
        "expected_total_images": EXPECTED_TEST_IMAGES,
        "processed_images": EXPECTED_TEST_IMAGES,
        "images_with_regions": images_with_regions,
        "images_without_regions": images_without_regions,
        "execution_failed_images": execution_failed,
        "region_rows": region_rows,
        "result_rows": len(
            rows
        ),
        "total_native_foreground_pixels": total_native_pixels,
    }

    for (
        key,
        expected_value,
    ) in expected_summary_values.items():
        if summary.get(
            key
        ) != expected_value:
            raise RuntimeError(
                f"Summary value {key!r} is inconsistent with the staged CSV."
            )

    if summary.get(
        "observed_region_classes"
    ) != sorted(
        observed_classes
    ):
        raise RuntimeError(
            "Summary observed-region class list is inconsistent with the CSV."
        )

    if execution_failed == 0:
        if summary.get(
            "status"
        ) != "PASS":
            raise RuntimeError(
                "Summary status must be PASS when there are no execution failures."
            )
    else:
        if summary.get(
            "status"
        ) != "FAIL":
            raise RuntimeError(
                "Summary status must be FAIL when execution failures exist."
            )


def replace_owned_outputs(
    staging_dir,
):
    """Replace only component-test-owned final outputs with rollback protection."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_results = (
        staging_dir
        / RESULTS_PATH.name
    )

    staged_summary = (
        staging_dir
        / SUMMARY_PATH.name
    )

    final_paths = [
        RESULTS_PATH,
        SUMMARY_PATH,
    ]

    staged_paths = [
        staged_results,
        staged_summary,
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
        for final_path in final_paths:
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

        for (
            staged_path,
            final_path,
        ) in zip(
            staged_paths,
            final_paths,
        ):
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

        for (
            backup_path,
            final_path,
        ) in reversed(
            backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main():
    """Run the real PhysioTrack Face Regions component in isolation."""
    validate_required_paths()

    test_indices = (
        load_test_hq_indices()
    )

    preflight_images(
        test_indices
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_regions_component_test_",
            dir=OUTPUT_DIR,
        )
    )

    staged_results_path = (
        staging_dir
        / RESULTS_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_PATH.name
    )

    print("=" * 78)
    print(
        "PhysioTrack Face Regions Isolated Component Execution"
    )
    print("=" * 78)
    print(
        f"Dataset: {DATASET_ROOT}"
    )
    print(
        f"Images: {len(test_indices)}"
    )
    print(
        "Input resolution: 512 x 512"
    )
    print(
        "Input face box: full aligned image"
    )
    print(
        "Target component: FaceRegions"
    )
    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )
    print(
        "Device: CPU"
    )
    print(
        "Tracking: disabled"
    )
    print(
        "Unrelated face-analysis components: disabled"
    )
    print(
        "Ground-truth masks: not used"
    )
    print(
        "Accuracy metrics: not computed by this component-execution test"
    )
    print()

    config = (
        make_config()
    )

    pipeline = FaceAnalysis(
        detector=ControlledFullImageDetector(),
        config=config,
        device="cpu",
        verbose=False,
    )

    if pipeline.regions is None:
        raise RuntimeError(
            "FaceAnalysis did not initialize FaceRegions."
        )

    capture = RegionCapture(
        pipeline.regions
    )

    pipeline.regions = capture

    rows = []
    images_with_regions = 0
    images_without_regions = 0
    execution_failed = 0
    region_rows = 0
    total_native_foreground_pixels = 0
    observed_classes = set()

    start_time = (
        time.perf_counter()
    )

    try:
        for (
            position,
            image_id,
        ) in enumerate(
            test_indices,
            start=1,
        ):
            capture.last_output = None

            try:
                (
                    image_path,
                    frame,
                ) = load_image(
                    image_id
                )

                prediction = pipeline.predict(
                    frame
                )

                image_rows = (
                    collect_region_rows(
                        image_id=image_id,
                        image_path=image_path,
                        frame=frame,
                        prediction=prediction,
                        capture=capture,
                    )
                )

                rows.extend(
                    image_rows
                )

                if image_rows[
                    0
                ][
                    "status"
                ] == "NO_REGIONS":
                    images_without_regions += 1
                else:
                    images_with_regions += 1

                    for row in image_rows:
                        observed_classes.add(
                            row[
                                "region_name"
                            ]
                        )

                        total_native_foreground_pixels += int(
                            row[
                                "native_pixel_count"
                            ]
                        )

                        region_rows += 1

                print(
                    f"[{position:04d}/"
                    f"{len(test_indices):04d}] "
                    f"HQ {image_id}: "
                    f"{image_rows[0]['status']}"
                )

            except Exception as error:
                execution_failed += 1

                rows.append(
                    failure_row(
                        image_id,
                        error,
                    )
                )

                print(
                    f"[{position:04d}/"
                    f"{len(test_indices):04d}] "
                    f"HQ {image_id}: "
                    f"EXECUTION_FAILED - {error}"
                )

    finally:
        if hasattr(
            pipeline,
            "close",
        ):
            pipeline.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    status = (
        "PASS"
        if execution_failed == 0
        else "FAIL"
    )

    summary = {
        "component": "Face Regions",
        "execution_type": "isolated_component_execution",
        "dataset": "CelebAMask-HQ official CelebA test partition",
        "dataset_root": "datasets/CelebAMask-HQ",
        "input_resolution": "512 x 512",
        "input_face_box": "full aligned image",
        "ground_truth_use": (
            "Ground-truth segmentation masks are not used. "
            "Official partition metadata and HQ-to-CelebA mapping are used "
            "only to select the same 2,824-image test subset as the accepted "
            "scientific benchmark."
        ),
        "pipeline": "PhysioTrack FaceAnalysis",
        "target_component": "FaceRegions",
        "backend": "SegFace",
        "device": "cpu",
        "tracking_enabled": False,
        "unrelated_components_disabled": True,
        "expected_total_images": EXPECTED_TEST_IMAGES,
        "processed_images": len(
            test_indices
        ),
        "images_with_regions": images_with_regions,
        "images_without_regions": images_without_regions,
        "execution_failed_images": execution_failed,
        "region_rows": region_rows,
        "result_rows": len(
            rows
        ),
        "observed_region_classes": sorted(
            observed_classes
        ),
        "expected_foreground_classes": EXPECTED_FOREGROUND_CLASSES,
        "total_native_foreground_pixels": total_native_foreground_pixels,
        "runtime_seconds": elapsed,
        "runtime_minutes": (
            elapsed
            / 60.0
        ),
        "images_per_second": (
            len(
                test_indices
            )
            / elapsed
            if elapsed > 0
            else None
        ),
        "status": status,
        "interpretation": (
            "This output is software execution evidence for the real "
            "PhysioTrack Face Regions component. It is not a replacement "
            "for the accepted CelebAMask-HQ IoU, Dice, pixel-accuracy, or "
            "confusion-matrix benchmark."
        ),
    }

    write_results_csv(
        staged_results_path,
        rows,
    )

    staged_summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Validating staged component-execution outputs..."
    )

    validate_staged_results(
        staged_results_path,
        staged_summary_path,
    )

    replace_owned_outputs(
        staging_dir
    )

    if staging_dir.exists():
        shutil.rmtree(
            staging_dir
        )

    print()
    print(
        "Committed final component-execution outputs."
    )
    print()
    print(
        "Execution summary:"
    )
    print(
        f"Images processed: {len(test_indices)}"
    )
    print(
        f"Images with regions: {images_with_regions}"
    )
    print(
        f"Images without regions: {images_without_regions}"
    )
    print(
        f"Execution failed images: {execution_failed}"
    )
    print(
        f"Region rows: {region_rows}"
    )
    print(
        f"Result rows: {len(rows)}"
    )
    print(
        f"Observed region classes: {len(observed_classes)}"
    )
    print(
        f"Runtime: {elapsed / 60.0:.2f} minutes"
    )
    print(
        f"Overall status: {status}"
    )
    print()
    print(
        f"Saved results: {RESULTS_PATH}"
    )
    print(
        f"Saved summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
