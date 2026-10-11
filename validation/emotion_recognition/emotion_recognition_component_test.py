from pathlib import Path
import argparse
import csv
import json
import math
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face.analysis import FaceAnalysis
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.emotion import FaceEmotion


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "CAER-S"
TEST_ROOT = DATASET_ROOT / "test"
ANNOTATION_PATH = DATASET_ROOT / "test.txt"

RESULTS_DIR = VALIDATION_DIR / "results"
COMPONENT_DIR = RESULTS_DIR / "component_execution"

RESULTS_PATH = (
    COMPONENT_DIR
    / "emotion_recognition_component_results.csv"
)

SUMMARY_PATH = (
    COMPONENT_DIR
    / "emotion_recognition_component_summary.json"
)

MODEL_NAME = "enet_b0_8_best_afew"
ENGINE = "onnx"

MODEL_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

CLASS_ID_TO_DATASET_LABEL = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

TOLERANCE = 1e-6

RESULT_FIELDS = [
    "relative_path",
    "dataset_label",
    "class_id",
    "face_index",
    "status",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "detector_confidence",
    "emotion",
    "emotion_confidence",
    "score_anger",
    "score_contempt",
    "score_disgust",
    "score_fear",
    "score_happiness",
    "score_neutral",
    "score_sadness",
    "score_surprise",
    "score_sum",
    "direct_emotion",
    "direct_confidence",
    "direct_score_anger",
    "direct_score_contempt",
    "direct_score_disgust",
    "direct_score_fear",
    "direct_score_happiness",
    "direct_score_neutral",
    "direct_score_sadness",
    "direct_score_surprise",
    "label_match",
    "confidence_absolute_error",
    "maximum_score_absolute_error",
    "failure_reason",
]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated real PhysioTrack Emotion Recognition execution "
            "through FaceAnalysis on CAER-S images."
        )
    )

    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run N deterministic images without replacing final "
            "component-execution outputs."
        ),
    )

    return parser.parse_args()


def load_annotation_records():
    """Load the accepted CAER-S annotated image population."""
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    if not TEST_ROOT.is_dir():
        raise RuntimeError(
            f"CAER-S test directory not found: {TEST_ROOT}"
        )

    if not ANNOTATION_PATH.is_file():
        raise RuntimeError(
            f"Annotation file not found: {ANNOTATION_PATH}"
        )

    records = []
    seen_paths = set()

    with ANNOTATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 6:
                raise RuntimeError(
                    f"Malformed annotation row at line {line_number}."
                )

            relative_path = parts[0].replace(
                "\\",
                "/",
            )

            try:
                class_id = int(
                    parts[1]
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid class ID at line {line_number}."
                ) from exc

            if class_id not in CLASS_ID_TO_DATASET_LABEL:
                raise RuntimeError(
                    f"Unexpected class ID at line {line_number}: "
                    f"{class_id}"
                )

            if relative_path in seen_paths:
                raise RuntimeError(
                    f"Duplicate annotated path: {relative_path}"
                )

            path_parts = Path(
                relative_path
            ).parts

            if len(path_parts) < 2:
                raise RuntimeError(
                    f"Invalid relative path at line {line_number}."
                )

            dataset_label = path_parts[0]

            if (
                dataset_label
                != CLASS_ID_TO_DATASET_LABEL[
                    class_id
                ]
            ):
                raise RuntimeError(
                    "Class ID and path label mismatch at line "
                    f"{line_number}: {relative_path}"
                )

            image_path = (
                TEST_ROOT
                / relative_path
            )

            if not image_path.is_file():
                raise RuntimeError(
                    f"Referenced image not found: {image_path}"
                )

            records.append(
                {
                    "relative_path": relative_path,
                    "dataset_label": dataset_label,
                    "class_id": class_id,
                    "image_path": image_path,
                }
            )

            seen_paths.add(
                relative_path
            )

    if len(records) != 13942:
        raise RuntimeError(
            f"Expected 13,942 annotated images, found {len(records):,}."
        )

    return records


def dataset_inventory(
    records,
):
    """Build a read-only dataset inventory for mutation verification."""
    inventory = [
        (
            str(
                ANNOTATION_PATH.relative_to(
                    PROJECT_ROOT
                )
            ).replace(
                "\\",
                "/",
            ),
            ANNOTATION_PATH.stat().st_size,
            ANNOTATION_PATH.stat().st_mtime_ns,
        )
    ]

    for record in records:
        image_path = record[
            "image_path"
        ]

        stat = image_path.stat()

        inventory.append(
            (
                str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                ).replace(
                    "\\",
                    "/",
                ),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )

    return tuple(
        inventory
    )


def build_config():
    """Create the isolated FaceAnalysis configuration."""
    return FaceAnalysisConfig(
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
        emotion=True,
        regions=False,
        temporal=False,
        emotion_model=MODEL_NAME,
        emotion_engine=ENGINE,
    )


def verify_isolated_configuration(
    config,
):
    """Verify that only Emotion Recognition is enabled."""
    expected = {
        "tracking": False,
        "head_pose": False,
        "landmarks": False,
        "quality": False,
        "eyes": False,
        "blink": False,
        "gaze": False,
        "gaze_estimation": False,
        "mouth": False,
        "mouth_motion": False,
        "emotion": True,
        "regions": False,
        "temporal": False,
    }

    for field_name, expected_value in expected.items():
        actual_value = getattr(
            config,
            field_name,
        )

        if actual_value != expected_value:
            raise RuntimeError(
                "Unexpected FaceAnalysis configuration value: "
                f"{field_name}={actual_value!r}"
            )


def deterministic_subset(
    records,
    sample_count,
):
    """Select deterministic images distributed across the full population."""
    if sample_count <= 0:
        return records

    if sample_count > len(records):
        raise ValueError(
            "Smoke-test sample count exceeds the available images."
        )

    indices = np.linspace(
        0,
        len(records) - 1,
        sample_count,
        dtype=int,
    )

    return [
        records[index]
        for index in indices
    ]


def empty_numeric_fields():
    """Return empty fields for rows without Emotion Recognition output."""
    return {
        "box_x1": "",
        "box_y1": "",
        "box_x2": "",
        "box_y2": "",
        "detector_confidence": "",
        "emotion": "",
        "emotion_confidence": "",
        "score_anger": "",
        "score_contempt": "",
        "score_disgust": "",
        "score_fear": "",
        "score_happiness": "",
        "score_neutral": "",
        "score_sadness": "",
        "score_surprise": "",
        "score_sum": "",
        "direct_emotion": "",
        "direct_confidence": "",
        "direct_score_anger": "",
        "direct_score_contempt": "",
        "direct_score_disgust": "",
        "direct_score_fear": "",
        "direct_score_happiness": "",
        "direct_score_neutral": "",
        "direct_score_sadness": "",
        "direct_score_surprise": "",
        "label_match": "",
        "confidence_absolute_error": "",
        "maximum_score_absolute_error": "",
    }


def score_fields(
    scores,
    prefix="",
):
    """Convert the eight-class score dictionary to stable CSV fields."""
    name_prefix = (
        f"{prefix}_"
        if prefix
        else ""
    )

    return {
        f"{name_prefix}score_anger": scores[
            "Anger"
        ],
        f"{name_prefix}score_contempt": scores[
            "Contempt"
        ],
        f"{name_prefix}score_disgust": scores[
            "Disgust"
        ],
        f"{name_prefix}score_fear": scores[
            "Fear"
        ],
        f"{name_prefix}score_happiness": scores[
            "Happiness"
        ],
        f"{name_prefix}score_neutral": scores[
            "Neutral"
        ],
        f"{name_prefix}score_sadness": scores[
            "Sadness"
        ],
        f"{name_prefix}score_surprise": scores[
            "Surprise"
        ],
    }


def clipped_crop(
    image,
    box,
):
    """Create the integer-clipped crop used for numerical cross-checking."""
    height, width = image.shape[:2]

    x1, y1, x2, y2 = [
        int(
            round(
                value
            )
        )
        for value in box
    ]

    x1 = max(
        0,
        min(
            x1,
            width,
        ),
    )
    y1 = max(
        0,
        min(
            y1,
            height,
        ),
    )
    x2 = max(
        0,
        min(
            x2,
            width,
        ),
    )
    y2 = max(
        0,
        min(
            y2,
            height,
        ),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return (
            None,
            (
                x1,
                y1,
                x2,
                y2,
            ),
        )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return (
            None,
            (
                x1,
                y1,
                x2,
                y2,
            ),
        )

    return (
        crop,
        (
            x1,
            y1,
            x2,
            y2,
        ),
    )


def validate_emotion_payload(
    emotion_payload,
):
    """Validate the numerical Emotion Recognition output contract."""
    if not isinstance(
        emotion_payload,
        dict,
    ):
        raise RuntimeError(
            "Emotion payload is not a dictionary."
        )

    if not emotion_payload.get(
        "available",
        False,
    ):
        raise RuntimeError(
            "Emotion payload is not marked available."
        )

    label = emotion_payload.get(
        "emotion"
    )
    confidence = emotion_payload.get(
        "confidence"
    )
    scores = emotion_payload.get(
        "scores"
    )

    if label not in MODEL_CLASSES:
        raise RuntimeError(
            f"Unexpected emotion label: {label!r}"
        )

    if not isinstance(
        scores,
        dict,
    ):
        raise RuntimeError(
            "Emotion score output is not a dictionary."
        )

    if set(
        scores
    ) != set(
        MODEL_CLASSES
    ):
        raise RuntimeError(
            "Emotion score classes do not match the expected eight classes."
        )

    score_array = np.asarray(
        [
            scores[
                class_name
            ]
            for class_name in MODEL_CLASSES
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            score_array
        )
    ):
        raise RuntimeError(
            "Emotion scores contain non-finite values."
        )

    if np.any(
        score_array < 0.0
    ) or np.any(
        score_array > 1.0
    ):
        raise RuntimeError(
            "Emotion scores contain values outside [0, 1]."
        )

    score_sum = float(
        np.sum(
            score_array
        )
    )

    if not math.isclose(
        score_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        raise RuntimeError(
            "Emotion scores do not sum to one."
        )

    argmax_label = MODEL_CLASSES[
        int(
            np.argmax(
                score_array
            )
        )
    ]

    if argmax_label != label:
        raise RuntimeError(
            "Emotion label does not match the maximum class score."
        )

    confidence = float(
        confidence
    )

    if not math.isfinite(
        confidence
    ):
        raise RuntimeError(
            "Emotion confidence is non-finite."
        )

    if not math.isclose(
        confidence,
        float(
            scores[
                label
            ]
        ),
        rel_tol=0.0,
        abs_tol=TOLERANCE,
    ):
        raise RuntimeError(
            "Emotion confidence does not match the predicted-class score."
        )

    return (
        label,
        confidence,
        {
            class_name: float(
                scores[
                    class_name
                ]
            )
            for class_name in MODEL_CLASSES
        },
        score_sum,
    )


def process_image(
    record,
    pipeline,
    direct_model,
):
    """Run isolated FaceAnalysis Emotion Recognition on one image."""
    image = cv2.imread(
        str(
            record[
                "image_path"
            ]
        )
    )

    base = {
        "relative_path": record[
            "relative_path"
        ],
        "dataset_label": record[
            "dataset_label"
        ],
        "class_id": record[
            "class_id"
        ],
    }

    if image is None:
        return [
            {
                **base,
                "face_index": "",
                "status": "EXECUTION_FAILED",
                **empty_numeric_fields(),
                "failure_reason": "image_read_failure",
            }
        ]

    try:
        result = pipeline.predict(
            image
        )
    except Exception as exc:
        return [
            {
                **base,
                "face_index": "",
                "status": "EXECUTION_FAILED",
                **empty_numeric_fields(),
                "failure_reason": (
                    f"pipeline_exception:{type(exc).__name__}"
                ),
            }
        ]

    instances = list(
        result
    )

    if not instances:
        return [
            {
                **base,
                "face_index": "",
                "status": "NO_FACE",
                **empty_numeric_fields(),
                "failure_reason": "",
            }
        ]

    rows = []

    for face_index, instance in enumerate(
        instances
    ):
        row = {
            **base,
            "face_index": face_index,
            "status": "",
            **empty_numeric_fields(),
            "failure_reason": "",
        }

        if instance.box is None:
            row[
                "status"
            ] = "INVALID_FACE_BOX"
            row[
                "failure_reason"
            ] = "missing_face_box"
            rows.append(
                row
            )
            continue

        crop, clipped_box = clipped_crop(
            image,
            instance.box,
        )

        x1, y1, x2, y2 = clipped_box

        row.update(
            {
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "detector_confidence": (
                    float(
                        instance.confidence
                    )
                    if instance.confidence is not None
                    else ""
                ),
            }
        )

        if crop is None:
            row[
                "status"
            ] = "INVALID_FACE_BOX"
            row[
                "failure_reason"
            ] = "invalid_crop_after_clipping"
            rows.append(
                row
            )
            continue

        face_features = (
            instance.face_features
            if isinstance(
                instance.face_features,
                dict,
            )
            else {}
        )

        emotion_payload = face_features.get(
            "emotion"
        )

        if not isinstance(
            emotion_payload,
            dict,
        ) or not emotion_payload.get(
            "available",
            False,
        ):
            row[
                "status"
            ] = "EMOTION_UNAVAILABLE"
            row[
                "failure_reason"
            ] = "emotion_payload_unavailable"
            rows.append(
                row
            )
            continue

        try:
            (
                label,
                confidence,
                scores,
                score_sum,
            ) = validate_emotion_payload(
                emotion_payload
            )
        except Exception as exc:
            row[
                "status"
            ] = "EMOTION_UNAVAILABLE"
            row[
                "failure_reason"
            ] = (
                f"invalid_emotion_payload:{type(exc).__name__}"
            )
            rows.append(
                row
            )
            continue

        try:
            direct = direct_model.predict(
                crop
            )
        except Exception as exc:
            row[
                "status"
            ] = "EXECUTION_FAILED"
            row[
                "failure_reason"
            ] = (
                f"direct_reference_exception:{type(exc).__name__}"
            )
            rows.append(
                row
            )
            continue

        direct_label = direct[
            "emotion"
        ]
        direct_confidence = float(
            direct[
                "confidence"
            ]
        )
        direct_scores = {
            class_name: float(
                direct[
                    "scores"
                ][
                    class_name
                ]
            )
            for class_name in MODEL_CLASSES
        }

        score_errors = [
            abs(
                scores[
                    class_name
                ]
                - direct_scores[
                    class_name
                ]
            )
            for class_name in MODEL_CLASSES
        ]

        maximum_score_error = max(
            score_errors
        )

        confidence_error = abs(
            confidence
            - direct_confidence
        )

        label_match = (
            label
            == direct_label
        )

        row.update(
            {
                "status": "EMOTION_AVAILABLE",
                "emotion": label,
                "emotion_confidence": confidence,
                **score_fields(
                    scores
                ),
                "score_sum": score_sum,
                "direct_emotion": direct_label,
                "direct_confidence": direct_confidence,
                **score_fields(
                    direct_scores,
                    prefix="direct",
                ),
                "label_match": label_match,
                "confidence_absolute_error": confidence_error,
                "maximum_score_absolute_error": maximum_score_error,
                "failure_reason": "",
            }
        )

        rows.append(
            row
        )

    return rows


def validate_rows(
    rows,
):
    """Validate component-execution numerical contracts."""
    available_rows = [
        row
        for row in rows
        if row[
            "status"
        ] == "EMOTION_AVAILABLE"
    ]

    if not available_rows:
        raise RuntimeError(
            "No Emotion Recognition outputs were produced."
        )

    for row in available_rows:
        if row[
            "label_match"
        ] is not True:
            raise RuntimeError(
                "FaceAnalysis and direct FaceEmotion labels disagree."
            )

        if float(
            row[
                "confidence_absolute_error"
            ]
        ) > TOLERANCE:
            raise RuntimeError(
                "FaceAnalysis and direct FaceEmotion confidence disagree."
            )

        if float(
            row[
                "maximum_score_absolute_error"
            ]
        ) > TOLERANCE:
            raise RuntimeError(
                "FaceAnalysis and direct FaceEmotion scores disagree."
            )

        if not math.isclose(
            float(
                row[
                    "score_sum"
                ]
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                "Stored FaceAnalysis Emotion Recognition scores "
                "do not sum to one."
            )


def status_counts(
    rows,
):
    """Count component-execution row statuses."""
    statuses = [
        "EMOTION_AVAILABLE",
        "EMOTION_UNAVAILABLE",
        "INVALID_FACE_BOX",
        "NO_FACE",
        "EXECUTION_FAILED",
    ]

    return {
        status: sum(
            1
            for row in rows
            if row[
                "status"
            ] == status
        )
        for status in statuses
    }


def write_results(
    rows,
    output_path,
):
    """Write the detailed numerical component-execution table."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
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


def build_summary(
    records,
    rows,
    runtime_seconds,
    read_only_verified,
):
    """Build the isolated component-execution summary."""
    counts = status_counts(
        rows
    )

    detected_face_rows = sum(
        1
        for row in rows
        if row[
            "status"
        ] not in {
            "NO_FACE",
            "EXECUTION_FAILED",
        }
    )

    available_rows = [
        row
        for row in rows
        if row[
            "status"
        ] == "EMOTION_AVAILABLE"
    ]

    images_with_emotion = len(
        {
            row[
                "relative_path"
            ]
            for row in available_rows
        }
    )

    maximum_confidence_error = max(
        (
            float(
                row[
                    "confidence_absolute_error"
                ]
            )
            for row in available_rows
        ),
        default=0.0,
    )

    maximum_score_error = max(
        (
            float(
                row[
                    "maximum_score_absolute_error"
                ]
            )
            for row in available_rows
        ),
        default=0.0,
    )

    minimum_score_sum = min(
        (
            float(
                row[
                    "score_sum"
                ]
            )
            for row in available_rows
        ),
        default=float("nan"),
    )

    maximum_score_sum = max(
        (
            float(
                row[
                    "score_sum"
                ]
            )
            for row in available_rows
        ),
        default=float("nan"),
    )

    pass_status = (
        read_only_verified
        and counts[
            "EXECUTION_FAILED"
        ] == 0
        and counts[
            "EMOTION_UNAVAILABLE"
        ] == 0
        and counts[
            "INVALID_FACE_BOX"
        ] == 0
        and detected_face_rows
        == counts[
            "EMOTION_AVAILABLE"
        ]
        and maximum_confidence_error
        <= TOLERANCE
        and maximum_score_error
        <= TOLERANCE
    )

    return {
        "component": "FaceEmotion",
        "evidence_type": "isolated_component_execution",
        "scientific_accuracy_evidence": False,
        "execution_path": "FaceAnalysis",
        "active_components": [
            "Face detector",
            "FaceEmotion",
        ],
        "disabled_optional_components": [
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
            "regions",
            "temporal",
        ],
        "model": {
            "name": MODEL_NAME,
            "engine": ENGINE,
            "classes": MODEL_CLASSES,
        },
        "dataset": {
            "name": "CAER-S",
            "population": "annotated target-face test subset image population",
            "images": len(
                records
            ),
            "ground_truth_used_for_accuracy": False,
            "read_only_verified": read_only_verified,
        },
        "execution": {
            "total_images": len(
                records
            ),
            "total_rows": len(
                rows
            ),
            "detected_face_rows": detected_face_rows,
            "emotion_available_rows": counts[
                "EMOTION_AVAILABLE"
            ],
            "images_with_emotion": images_with_emotion,
            "status_counts": counts,
            "runtime_seconds": runtime_seconds,
        },
        "numerical_consistency": {
            "maximum_confidence_absolute_error": (
                maximum_confidence_error
            ),
            "maximum_score_absolute_error": (
                maximum_score_error
            ),
            "minimum_score_sum": minimum_score_sum,
            "maximum_score_sum": maximum_score_sum,
            "tolerance": TOLERANCE,
        },
        "interpretation": (
            "This test verifies real numerical execution and export of "
            "FaceEmotion through the current FaceAnalysis path with only "
            "the unavoidable face detector and Emotion Recognition active. "
            "It is software execution evidence and does not replace the "
            "controlled scientific CAER-S Emotion Recognition validation."
        ),
        "status": (
            "PASS"
            if pass_status
            else "FAIL"
        ),
    }


def validate_staged_outputs(
    staged_results,
    staged_summary,
    expected_rows,
):
    """Validate staged component-execution outputs before replacement."""
    if not staged_results.is_file():
        raise RuntimeError(
            "Staged component result CSV was not created."
        )

    if not staged_summary.is_file():
        raise RuntimeError(
            "Staged component summary JSON was not created."
        )

    with staged_results.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != RESULT_FIELDS:
            raise RuntimeError(
                "Staged component result schema is invalid."
            )

        staged_rows = list(
            reader
        )

    if len(
        staged_rows
    ) != expected_rows:
        raise RuntimeError(
            "Staged component result row count is invalid."
        )

    summary = json.loads(
        staged_summary.read_text(
            encoding="utf-8",
        )
    )

    if summary.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Staged component summary status is not PASS."
        )


def replace_owned_outputs(
    staging_dir,
):
    """Replace only component-execution-owned outputs."""
    COMPONENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(
        staging_dir
        / RESULTS_PATH.name,
        RESULTS_PATH,
    )

    os.replace(
        staging_dir
        / SUMMARY_PATH.name,
        SUMMARY_PATH,
    )


def execute(
    records,
):
    """Run the real isolated PhysioTrack Emotion Recognition path."""
    config = build_config()

    verify_isolated_configuration(
        config
    )

    pipeline = FaceAnalysis(
        config=config,
        device="cpu",
        verbose=False,
    )

    direct_model = FaceEmotion(
        model_name=MODEL_NAME,
        engine=ENGINE,
    )

    rows = []

    start_time = time.perf_counter()

    for index, record in enumerate(
        records,
        start=1,
    ):
        rows.extend(
            process_image(
                record,
                pipeline,
                direct_model,
            )
        )

        if (
            index % 500 == 0
            or index == len(
                records
            )
        ):
            print(
                f"Processed {index:,}/{len(records):,} images"
            )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    validate_rows(
        rows
    )

    return (
        rows,
        runtime_seconds,
    )


def run_smoke_test(
    records,
    sample_count,
):
    """Run the final component logic without replacing final outputs."""
    selected = deterministic_subset(
        records,
        sample_count,
    )

    print(
        "Running deterministic Emotion Recognition component smoke test..."
    )

    before_inventory = dataset_inventory(
        records
    )

    rows, runtime_seconds = execute(
        selected
    )

    after_inventory = dataset_inventory(
        records
    )

    read_only_verified = (
        before_inventory
        == after_inventory
    )

    summary = build_summary(
        selected,
        rows,
        runtime_seconds,
        read_only_verified,
    )

    counts = summary[
        "execution"
    ][
        "status_counts"
    ]

    print()
    print(
        "Emotion Recognition component smoke-test summary"
    )
    print(
        "=" * 50
    )
    print(
        f"Images: {len(selected):,}"
    )
    print(
        f"Rows: {len(rows):,}"
    )
    print(
        f"EMOTION_AVAILABLE: {counts['EMOTION_AVAILABLE']:,}"
    )
    print(
        f"EMOTION_UNAVAILABLE: {counts['EMOTION_UNAVAILABLE']:,}"
    )
    print(
        f"INVALID_FACE_BOX: {counts['INVALID_FACE_BOX']:,}"
    )
    print(
        f"NO_FACE: {counts['NO_FACE']:,}"
    )
    print(
        f"EXECUTION_FAILED: {counts['EXECUTION_FAILED']:,}"
    )
    print(
        "Dataset read-only verification:",
        (
            "PASS"
            if read_only_verified
            else "FAIL"
        ),
    )
    print(
        "Numerical consistency:",
        summary[
            "status"
        ],
    )
    print(
        f"Runtime seconds: {runtime_seconds:.3f}"
    )

    if summary[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Emotion Recognition component smoke test failed."
        )

    print(
        "Smoke-test status: PASS"
    )


def run_full_execution(
    records,
):
    """Run full isolated component execution with staged output replacement."""
    COMPONENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    before_inventory = dataset_inventory(
        records
    )

    rows, runtime_seconds = execute(
        records
    )

    after_inventory = dataset_inventory(
        records
    )

    read_only_verified = (
        before_inventory
        == after_inventory
    )

    summary = build_summary(
        records,
        rows,
        runtime_seconds,
        read_only_verified,
    )

    if summary[
        "status"
    ] != "PASS":
        raise RuntimeError(
            "Emotion Recognition component execution did not pass "
            "all validation checks."
        )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".emotion_recognition_component_",
            dir=COMPONENT_DIR,
        )
    )

    try:
        staged_results = (
            staging_dir
            / RESULTS_PATH.name
        )
        staged_summary = (
            staging_dir
            / SUMMARY_PATH.name
        )

        write_results(
            rows,
            staged_results,
        )

        staged_summary.write_text(
            json.dumps(
                summary,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        validate_staged_outputs(
            staged_results,
            staged_summary,
            len(
                rows
            ),
        )

        replace_owned_outputs(
            staging_dir
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    counts = summary[
        "execution"
    ][
        "status_counts"
    ]

    print()
    print(
        "Emotion Recognition Isolated Component Execution"
    )
    print(
        "=" * 49
    )
    print(
        f"Images: {len(records):,}"
    )
    print(
        f"Rows: {len(rows):,}"
    )
    print(
        f"Detected face rows: "
        f"{summary['execution']['detected_face_rows']:,}"
    )
    print(
        f"EMOTION_AVAILABLE: {counts['EMOTION_AVAILABLE']:,}"
    )
    print(
        f"EMOTION_UNAVAILABLE: {counts['EMOTION_UNAVAILABLE']:,}"
    )
    print(
        f"INVALID_FACE_BOX: {counts['INVALID_FACE_BOX']:,}"
    )
    print(
        f"NO_FACE: {counts['NO_FACE']:,}"
    )
    print(
        f"EXECUTION_FAILED: {counts['EXECUTION_FAILED']:,}"
    )
    print(
        "Dataset read-only verification: PASS"
    )
    print(
        "Maximum confidence absolute error:",
        (
            summary[
                "numerical_consistency"
            ][
                "maximum_confidence_absolute_error"
            ]
        ),
    )
    print(
        "Maximum score absolute error:",
        (
            summary[
                "numerical_consistency"
            ][
                "maximum_score_absolute_error"
            ]
        ),
    )
    print(
        f"Runtime seconds: {runtime_seconds:.3f}"
    )
    print(
        "Overall status: PASS"
    )
    print()
    print(
        f"Results CSV: {RESULTS_PATH}"
    )
    print(
        f"Summary JSON: {SUMMARY_PATH}"
    )


def main():
    """Run Emotion Recognition isolated component execution."""
    args = parse_arguments()

    records = load_annotation_records()

    if args.smoke_test:
        run_smoke_test(
            records,
            args.smoke_test,
        )
        return

    run_full_execution(
        records
    )


if __name__ == "__main__":
    main()
