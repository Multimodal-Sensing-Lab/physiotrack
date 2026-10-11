from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

WIDER_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "WIDER_FACE"
)

IMAGES_DIR = (
    WIDER_ROOT
    / "WIDER_val"
    / "images"
)

RESULTS_DIR = (
    SCRIPT_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_CSV_PATH = (
    OUTPUT_DIR
    / "face_detection_component_results.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "face_detection_component_summary.json"
)

EXPECTED_IMAGE_COUNT = 3226

FIELDNAMES = [
    "input_identifier",
    "input_type",
    "image_index",
    "detection_index",
    "detections_in_image",
    "image_width",
    "image_height",
    "person_id",
    "class_id",
    "class_name",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_width",
    "box_height",
    "box_area",
    "confidence",
    "status",
    "failure_reason",
]

UNRELATED_PIPELINE_ATTRIBUTES = (
    "tracker",
    "orientation",
    "landmarks",
    "quality",
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


def image_label(
    image_path: Path,
) -> str:
    return str(
        image_path.relative_to(
            IMAGES_DIR
        )
    ).replace("\\", "/")


def make_config() -> FaceAnalysisConfig:
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
        regions=False,
        temporal=False,
    )

    config.validate()

    return config


def preflight() -> list[Path]:
    if not WIDER_ROOT.exists():
        raise FileNotFoundError(
            f"WIDER FACE dataset root not found: {WIDER_ROOT}"
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"WIDER FACE validation image directory not found: {IMAGES_DIR}"
        )

    image_paths = sorted(
        IMAGES_DIR.rglob("*.jpg")
    )

    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            "Unexpected WIDER FACE validation image count: "
            f"{len(image_paths)} "
            f"(expected {EXPECTED_IMAGE_COUNT})"
        )

    unreadable_paths = []

    for image_path in image_paths:
        if not image_path.is_file():
            unreadable_paths.append(
                image_path
            )
            continue

        try:
            with image_path.open(
                "rb"
            ) as file:
                if not file.read(1):
                    unreadable_paths.append(
                        image_path
                    )
        except OSError:
            unreadable_paths.append(
                image_path
            )

    if unreadable_paths:
        raise RuntimeError(
            "WIDER FACE preflight found unreadable image files. "
            f"Count: {len(unreadable_paths)}. "
            f"First: {unreadable_paths[0]}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_path = (
        OUTPUT_DIR
        / ".write_probe"
    )

    try:
        with probe_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "preflight"
            )
    finally:
        if probe_path.exists():
            probe_path.unlink()

    return image_paths


def assert_detection_only_pipeline(
    pipeline: FaceAnalysis,
) -> None:
    if pipeline.detector is None:
        raise RuntimeError(
            "Face detection component is not initialized."
        )

    unexpected_enabled = [
        name
        for name in UNRELATED_PIPELINE_ATTRIBUTES
        if getattr(
            pipeline,
            name,
        ) is not None
    ]

    if unexpected_enabled:
        raise RuntimeError(
            "Unrelated FaceAnalysis components are enabled: "
            + ", ".join(
                unexpected_enabled
            )
        )


def finite_numeric(
    value: Any,
) -> bool:
    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def scalar_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (
            TypeError,
            ValueError,
        ):
            pass

    return value


def normalize_box(
    box: Any,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if isinstance(
        box,
        np.ndarray,
    ):
        values = box.tolist()
    else:
        values = list(
            box
        )

    if len(values) != 4:
        raise RuntimeError(
            "Detection box does not contain four coordinates."
        )

    if not all(
        finite_numeric(value)
        for value in values
    ):
        raise RuntimeError(
            "Detection box contains a non-finite coordinate."
        )

    return tuple(
        float(value)
        for value in values
    )


def detection_row(
    image_path: Path,
    image_index: int,
    detection_index: int,
    detections_in_image: int,
    image_width: int,
    image_height: int,
    face: Any,
) -> tuple[
    dict[str, Any],
    bool,
]:
    x1, y1, x2, y2 = (
        normalize_box(
            face.box
        )
    )

    confidence = scalar_value(
        face.confidence
    )

    if not finite_numeric(
        confidence
    ):
        raise RuntimeError(
            "Detection confidence is not finite."
        )

    confidence = float(
        confidence
    )

    if not (
        0.0
        <= confidence
        <= 1.0
    ):
        raise RuntimeError(
            "Detection confidence is outside [0, 1]."
        )

    person_id = scalar_value(
        face.id
    )

    if person_id is not None:
        raise RuntimeError(
            "Tracking identifier was produced while tracking is disabled."
        )

    width = x2 - x1
    height = y2 - y1

    valid_box = (
        width > 0.0
        and height > 0.0
    )

    if valid_box:
        box_width = width
        box_height = height
        box_area = width * height
        status = "DETECTED"
        failure_reason = ""
    else:
        box_width = None
        box_height = None
        box_area = None
        status = "DETECTED_INVALID_BOX"
        failure_reason = (
            "Raw detector box has non-positive width or height; "
            "raw coordinates are preserved without correction."
        )

    row = {
        "input_identifier":
            image_label(
                image_path
            ),
        "input_type":
            "image",
        "image_index":
            image_index,
        "detection_index":
            detection_index,
        "detections_in_image":
            detections_in_image,
        "image_width":
            image_width,
        "image_height":
            image_height,
        "person_id":
            None,
        "class_id":
            scalar_value(
                face.cls
            ),
        "class_name":
            scalar_value(
                face.cls_name
            ),
        "box_x1":
            x1,
        "box_y1":
            y1,
        "box_x2":
            x2,
        "box_y2":
            y2,
        "box_width":
            box_width,
        "box_height":
            box_height,
        "box_area":
            box_area,
        "confidence":
            confidence,
        "status":
            status,
        "failure_reason":
            failure_reason,
    }

    return (
        row,
        valid_box,
    )


def empty_detection_row(
    image_path: Path,
    image_index: int,
    image_width: int | None,
    image_height: int | None,
    status: str,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "input_identifier":
            image_label(
                image_path
            ),
        "input_type":
            "image",
        "image_index":
            image_index,
        "detection_index":
            None,
        "detections_in_image":
            0,
        "image_width":
            image_width,
        "image_height":
            image_height,
        "person_id":
            None,
        "class_id":
            None,
        "class_name":
            None,
        "box_x1":
            None,
        "box_y1":
            None,
        "box_x2":
            None,
        "box_y2":
            None,
        "box_width":
            None,
        "box_height":
            None,
        "box_area":
            None,
        "confidence":
            None,
        "status":
            status,
        "failure_reason":
            failure_reason,
    }


def generate_outputs(
    image_paths: list[Path],
    staging_dir: Path,
) -> dict[str, Any]:
    staged_csv_path = (
        staging_dir
        / RESULTS_CSV_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_JSON_PATH.name
    )

    config = make_config()

    detector = Face(
        device="cpu",
        verbose=False,
        conf=0.001,
        max_det=10000,
    )

    pipeline = FaceAnalysis(
        detector=detector,
        config=config,
        device="cpu",
        verbose=False,
    )

    assert_detection_only_pipeline(
        pipeline
    )

    total_detections = 0
    invalid_box_detections = 0
    images_with_detections = 0
    images_without_detections = 0
    read_failures = 0
    prediction_failures = 0
    rows_written = 0

    start_time = time.perf_counter()

    try:
        with staged_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )

            writer.writeheader()

            for (
                image_index,
                image_path,
            ) in enumerate(
                image_paths,
                start=1,
            ):
                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    read_failures += 1

                    writer.writerow(
                        empty_detection_row(
                            image_path=image_path,
                            image_index=image_index,
                            image_width=None,
                            image_height=None,
                            status="READ_FAILURE",
                            failure_reason=(
                                "OpenCV could not decode the image."
                            ),
                        )
                    )

                    rows_written += 1
                    continue

                image_height, image_width = (
                    image.shape[:2]
                )

                try:
                    result = pipeline.predict(
                        image
                    )
                except Exception as exc:
                    prediction_failures += 1

                    writer.writerow(
                        empty_detection_row(
                            image_path=image_path,
                            image_index=image_index,
                            image_width=image_width,
                            image_height=image_height,
                            status="PREDICTION_FAILURE",
                            failure_reason=str(
                                exc
                            ),
                        )
                    )

                    rows_written += 1
                    continue

                faces = list(
                    result
                )

                detections_in_image = len(
                    faces
                )

                if detections_in_image == 0:
                    images_without_detections += 1

                    writer.writerow(
                        empty_detection_row(
                            image_path=image_path,
                            image_index=image_index,
                            image_width=image_width,
                            image_height=image_height,
                            status="NO_DETECTIONS",
                        )
                    )

                    rows_written += 1
                else:
                    images_with_detections += 1
                    total_detections += (
                        detections_in_image
                    )

                    for (
                        detection_index,
                        face,
                    ) in enumerate(
                        faces,
                        start=1,
                    ):
                        (
                            row,
                            valid_box,
                        ) = detection_row(
                            image_path=image_path,
                            image_index=image_index,
                            detection_index=detection_index,
                            detections_in_image=detections_in_image,
                            image_width=image_width,
                            image_height=image_height,
                            face=face,
                        )

                        writer.writerow(
                            row
                        )

                        if not valid_box:
                            invalid_box_detections += 1

                        rows_written += 1

                if (
                    image_index % 100 == 0
                    or image_index
                    == len(
                        image_paths
                    )
                ):
                    print(
                        "Processed "
                        f"{image_index}/{len(image_paths)}"
                    )

    finally:
        pipeline.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    failed_images = (
        read_failures
        + prediction_failures
    )

    processed_images = (
        len(image_paths)
        - failed_images
    )

    summary = {
        "test_type":
            "isolated_physiotrack_face_detection_component_execution",
        "purpose":
            (
                "Software execution evidence for the real PhysioTrack "
                "FaceAnalysis detection path. This is not an accuracy benchmark."
            ),
        "dataset":
            "WIDER FACE validation images",
        "validation_images":
            len(image_paths),
        "processed_images":
            processed_images,
        "failed_images":
            failed_images,
        "read_failures":
            read_failures,
        "prediction_failures":
            prediction_failures,
        "images_with_detections":
            images_with_detections,
        "images_without_detections":
            images_without_detections,
        "total_detections":
            total_detections,
        "invalid_box_detections":
            invalid_box_detections,
        "rows_written":
            rows_written,
        "device":
            "CPU",
        "confidence_threshold":
            0.001,
        "max_det":
            10000,
        "enabled_component":
            "face_detection",
        "disabled_components": [
            "tracking",
            "head_pose",
            "landmarks",
            "quality",
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
        "runtime_seconds":
            elapsed,
        "runtime_minutes":
            elapsed / 60.0,
        "overall_status":
            (
                "PASS"
                if failed_images == 0
                else "FAIL"
            ),
    }

    with staged_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


def validate_staged_outputs(
    staging_dir: Path,
    expected_images: int,
) -> dict[str, Any]:
    staged_csv_path = (
        staging_dir
        / RESULTS_CSV_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_JSON_PATH.name
    )

    if not staged_csv_path.is_file():
        raise RuntimeError(
            "Staged component result CSV was not created."
        )

    if not staged_summary_path.is_file():
        raise RuntimeError(
            "Staged component summary JSON was not created."
        )

    with staged_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if summary.get(
        "validation_images"
    ) != expected_images:
        raise RuntimeError(
            "Staged summary validation image count is incorrect."
        )

    if summary.get(
        "processed_images"
    ) + summary.get(
        "failed_images"
    ) != expected_images:
        raise RuntimeError(
            "Staged summary image accounting is inconsistent."
        )

    if summary.get(
        "failed_images"
    ) != 0:
        raise RuntimeError(
            "Component execution completed with image failures; "
            "existing accepted outputs will not be replaced."
        )

    if summary.get(
        "overall_status"
    ) != "PASS":
        raise RuntimeError(
            "Staged component execution did not produce PASS status."
        )

    input_identifiers = set()
    detected_rows = 0
    invalid_box_rows = 0
    non_detection_rows = 0
    csv_rows = 0

    with staged_csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(
                "Staged component result CSV schema is incorrect."
            )

        for row in reader:
            csv_rows += 1

            input_identifier = row[
                "input_identifier"
            ].strip()

            if not input_identifier:
                raise RuntimeError(
                    "Component result row has no input identifier."
                )

            input_identifiers.add(
                input_identifier
            )

            status = row[
                "status"
            ]

            if status in {
                "DETECTED",
                "DETECTED_INVALID_BOX",
            }:
                detected_rows += 1

                for field in (
                    "box_x1",
                    "box_y1",
                    "box_x2",
                    "box_y2",
                    "confidence",
                ):
                    if not finite_numeric(
                        row[field]
                    ):
                        raise RuntimeError(
                            "Detected row contains a non-finite "
                            f"{field} value."
                        )

                confidence = float(
                    row[
                        "confidence"
                    ]
                )

                if not (
                    0.0
                    <= confidence
                    <= 1.0
                ):
                    raise RuntimeError(
                        "Detected row confidence is outside [0, 1]."
                    )

                x1 = float(
                    row[
                        "box_x1"
                    ]
                )
                y1 = float(
                    row[
                        "box_y1"
                    ]
                )
                x2 = float(
                    row[
                        "box_x2"
                    ]
                )
                y2 = float(
                    row[
                        "box_y2"
                    ]
                )

                if row[
                    "person_id"
                ].strip():
                    raise RuntimeError(
                        "Detected row contains a tracking identifier."
                    )

                if status == "DETECTED":
                    for field in (
                        "box_width",
                        "box_height",
                        "box_area",
                    ):
                        if not finite_numeric(
                            row[field]
                        ):
                            raise RuntimeError(
                                "Valid detected row contains a non-finite "
                                f"{field} value."
                            )

                    width = float(
                        row[
                            "box_width"
                        ]
                    )
                    height = float(
                        row[
                            "box_height"
                        ]
                    )
                    area = float(
                        row[
                            "box_area"
                        ]
                    )

                    if (
                        width <= 0.0
                        or height <= 0.0
                    ):
                        raise RuntimeError(
                            "Valid detected row has non-positive dimensions."
                        )

                    if not math.isclose(
                        width,
                        x2 - x1,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Detected row box width is inconsistent."
                        )

                    if not math.isclose(
                        height,
                        y2 - y1,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Detected row box height is inconsistent."
                        )

                    if not math.isclose(
                        area,
                        width * height,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Detected row box area is inconsistent."
                        )

                    if row[
                        "failure_reason"
                    ].strip():
                        raise RuntimeError(
                            "Valid detected row has an unexpected failure reason."
                        )

                else:
                    invalid_box_rows += 1

                    if (
                        x2 > x1
                        and y2 > y1
                    ):
                        raise RuntimeError(
                            "Invalid-box row contains a geometrically valid box."
                        )

                    for field in (
                        "box_width",
                        "box_height",
                        "box_area",
                    ):
                        if row[
                            field
                        ].strip():
                            raise RuntimeError(
                                "Invalid-box row contains derived box dimensions."
                            )

                    if not row[
                        "failure_reason"
                    ].strip():
                        raise RuntimeError(
                            "Invalid-box row has no diagnostic reason."
                        )

            elif status == "NO_DETECTIONS":
                non_detection_rows += 1

                numerical_fields = (
                    "box_x1",
                    "box_y1",
                    "box_x2",
                    "box_y2",
                    "box_width",
                    "box_height",
                    "box_area",
                    "confidence",
                )

                if any(
                    row[field].strip()
                    for field in numerical_fields
                ):
                    raise RuntimeError(
                        "NO_DETECTIONS row contains fabricated "
                        "detection values."
                    )

            else:
                raise RuntimeError(
                    "Unexpected status in staged component CSV: "
                    f"{status}"
                )

    if csv_rows != summary.get(
        "rows_written"
    ):
        raise RuntimeError(
            "Staged CSV row count does not match the summary."
        )

    if detected_rows != summary.get(
        "total_detections"
    ):
        raise RuntimeError(
            "Staged detected-row count does not match total detections."
        )

    if invalid_box_rows != summary.get(
        "invalid_box_detections"
    ):
        raise RuntimeError(
            "Staged invalid-box row count does not match the summary."
        )

    if non_detection_rows != summary.get(
        "images_without_detections"
    ):
        raise RuntimeError(
            "Staged NO_DETECTIONS row count is inconsistent."
        )

    if len(input_identifiers) != expected_images:
        raise RuntimeError(
            "Staged CSV does not account for every validation image."
        )

    return summary


def replace_owned_outputs(
    staging_dir: Path,
) -> None:
    staged_paths = [
        (
            staging_dir
            / RESULTS_CSV_PATH.name,
            RESULTS_CSV_PATH,
        ),
        (
            staging_dir
            / SUMMARY_JSON_PATH.name,
            SUMMARY_JSON_PATH,
        ),
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    moved_backups = []
    installed_outputs = []

    try:
        for _, final_path in staged_paths:
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                moved_backups.append(
                    (
                        backup_path,
                        final_path,
                    )
                )

        for (
            staged_path,
            final_path,
        ) in staged_paths:
            os.replace(
                staged_path,
                final_path,
            )

            installed_outputs.append(
                final_path
            )

    except Exception:
        for final_path in installed_outputs:
            if final_path.exists():
                final_path.unlink()

        for (
            backup_path,
            final_path,
        ) in reversed(
            moved_backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main() -> None:
    print(
        "PhysioTrack isolated Face Detection component execution"
    )

    print(
        "Preflight..."
    )

    image_paths = preflight()

    print(
        "Preflight: PASS"
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_detection_component_",
            dir=OUTPUT_DIR,
        )
    )

    try:
        summary = generate_outputs(
            image_paths=image_paths,
            staging_dir=staging_dir,
        )

        print()
        print(
            "Validating staged outputs..."
        )

        validated_summary = (
            validate_staged_outputs(
                staging_dir=staging_dir,
                expected_images=len(
                    image_paths
                ),
            )
        )

        if validated_summary != summary:
            raise RuntimeError(
                "In-memory and staged component summaries differ."
            )

        replace_owned_outputs(
            staging_dir
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

    print()
    print(
        "Finished."
    )

    print(
        "Images processed:",
        summary[
            "processed_images"
        ],
    )

    print(
        "Failed images:",
        summary[
            "failed_images"
        ],
    )

    print(
        "Images with detections:",
        summary[
            "images_with_detections"
        ],
    )

    print(
        "Images without detections:",
        summary[
            "images_without_detections"
        ],
    )

    print(
        "Total detections:",
        summary[
            "total_detections"
        ],
    )

    print(
        "Invalid-box detections:",
        summary[
            "invalid_box_detections"
        ],
    )

    print(
        "Result rows:",
        summary[
            "rows_written"
        ],
    )

    print(
        "Runtime minutes:",
        f"{summary['runtime_minutes']:.2f}",
    )

    print(
        "Overall status:",
        summary[
            "overall_status"
        ],
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULTS_CSV_PATH
    )

    print(
        SUMMARY_JSON_PATH
    )


if __name__ == "__main__":
    main()
