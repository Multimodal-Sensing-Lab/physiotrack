from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "generated_controlled"
)
METADATA_PATH = (
    TEST_DATA_DIR
    / "robustness_generated_cases.csv"
)

FINAL_RESULTS_DIR = (
    SCRIPT_DIR
    / "results"
)

RESULTS_FILENAME = "robustness_results.csv"
SUMMARY_FILENAME = "robustness_summary.json"

EXPECTED_CASES = [
    "baseline",
    "dim_lighting",
    "overexposed_lighting",
    "gaussian_blur",
    "motion_blur",
    "small_face",
    "partial_occlusion",
]

STATIC_MODULES = [
    "detection",
    "landmarks",
    "quality",
    "head_pose",
    "eyes",
    "gaze",
    "gaze_estimation",
    "mouth",
    "emotion",
    "regions",
]

EMOTION_LABELS = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

RESULT_FIELDS = [
    "case_id",
    "condition",
    "image_file",
    "image_width",
    "image_height",
    "detected_faces",
    "selected_face_index",
    "status",
    "failure_reason",
    "detection_available",
    "landmarks_available",
    "quality_available",
    "head_pose_available",
    "eyes_available",
    "gaze_available",
    "gaze_estimation_available",
    "mouth_available",
    "emotion_available",
    "regions_available",
    "all_available_numeric_values_finite",
    "detector_confidence",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_width",
    "box_height",
    "box_area",
    "quality_brightness",
    "quality_sharpness",
    "quality_face_area_ratio",
    "head_pose_pitch",
    "head_pose_yaw",
    "head_pose_roll",
    "eye_mean_openness",
    "gaze_mean_iris_x",
    "gaze_mean_iris_y",
    "gaze_estimation_pitch",
    "gaze_estimation_yaw",
    "gaze_association_iou",
    "mouth_openness",
    "emotion_label",
    "emotion_confidence",
    "regions_skin_fraction",
    "regions_association_iou",
    "detector_confidence_abs_delta_vs_baseline",
    "quality_brightness_abs_delta_vs_baseline",
    "quality_sharpness_abs_delta_vs_baseline",
    "quality_face_area_ratio_abs_delta_vs_baseline",
    "head_pose_pitch_abs_delta_vs_baseline",
    "head_pose_yaw_abs_delta_vs_baseline",
    "head_pose_roll_abs_delta_vs_baseline",
    "eye_mean_openness_abs_delta_vs_baseline",
    "gaze_mean_iris_x_abs_delta_vs_baseline",
    "gaze_mean_iris_y_abs_delta_vs_baseline",
    "gaze_estimation_pitch_abs_delta_vs_baseline",
    "gaze_estimation_yaw_abs_delta_vs_baseline",
    "mouth_openness_abs_delta_vs_baseline",
    "emotion_confidence_abs_delta_vs_baseline",
    "regions_skin_fraction_abs_delta_vs_baseline",
    "emotion_changed_vs_baseline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate controlled static robustness conditions using the real "
            "PhysioTrack FaceAnalysis pipeline."
        )
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate generated controlled inputs and pipeline configuration "
            "without running inference or writing results."
        ),
    )
    return parser.parse_args()


def finite_numeric(value: Any) -> bool:
    if value is None:
        return False

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_or_none(value: Any) -> float | None:
    if not finite_numeric(value):
        return None

    return float(value)


def module_available(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, dict):
        if "available" in value:
            return bool(value["available"])

        return len(value) > 0

    if isinstance(value, np.ndarray):
        return value.size > 0

    if isinstance(value, (list, tuple)):
        return len(value) > 0

    return True


def make_config() -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=True,
        landmarks=True,
        quality=True,
        eyes=True,
        blink=False,
        gaze=True,
        gaze_estimation=True,
        mouth=True,
        mouth_motion=False,
        emotion=True,
        regions=True,
        temporal=False,
        gaze_estimation_mode="eth-xgaze",
        gaze_estimation_min_iou=0.10,
    )

    config.validate()

    return config


def get_head_pose(
    face: Any,
    features: dict[str, Any],
) -> Any:
    for key in (
        "head_pose",
        "orientation",
        "pose",
    ):
        if key in features:
            return features[key]

    for name in (
        "head_pose",
        "orientation",
        "pose",
    ):
        if hasattr(face, name):
            value = getattr(face, name)

            if value is not None:
                return value

    return None


def load_metadata() -> list[dict[str, str]]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Generated robustness metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != len(EXPECTED_CASES):
        raise RuntimeError(
            "Generated robustness metadata row count does not match "
            "the accepted seven-case protocol"
        )

    case_ids = [
        row.get("case_id", "")
        for row in rows
    ]

    if case_ids != EXPECTED_CASES:
        raise RuntimeError(
            "Generated robustness metadata case order does not match "
            "the accepted protocol"
        )

    return rows


def input_inventory(
    metadata_rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    inventory = {}

    for row in metadata_rows:
        path = TEST_DATA_DIR / row["output_file"]

        if not path.is_file():
            raise FileNotFoundError(
                f"Generated robustness image not found: {path}"
            )

        stat = path.stat()
        inventory[row["output_file"]] = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    metadata_stat = METADATA_PATH.stat()
    inventory[METADATA_PATH.name] = {
        "size_bytes": int(metadata_stat.st_size),
        "mtime_ns": int(metadata_stat.st_mtime_ns),
    }

    return inventory


def preflight_validation() -> list[dict[str, str]]:
    metadata_rows = load_metadata()

    for row in metadata_rows:
        image_path = TEST_DATA_DIR / row["output_file"]
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read generated robustness image: {image_path}"
            )

        height, width = image.shape[:2]

        if int(row["width"]) != width:
            raise RuntimeError(
                f"Width mismatch for generated robustness image: {image_path}"
            )

        if int(row["height"]) != height:
            raise RuntimeError(
                f"Height mismatch for generated robustness image: {image_path}"
            )

    make_config()

    if (
        FINAL_RESULTS_DIR.exists()
        and not FINAL_RESULTS_DIR.is_dir()
    ):
        raise RuntimeError(
            f"Robustness results path is not a directory: {FINAL_RESULTS_DIR}"
        )

    FINAL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return metadata_rows


def select_primary_face(
    faces: list[Any],
) -> tuple[int, Any]:
    if not faces:
        raise ValueError("No faces available for primary-face selection")

    indexed = list(enumerate(faces))

    def key(item: tuple[int, Any]) -> tuple[float, int]:
        index, face = item
        confidence = finite_or_none(
            getattr(face, "confidence", None)
        )

        return (
            -1.0 if confidence is None else confidence,
            -index,
        )

    selected_index, selected_face = max(
        indexed,
        key=key,
    )

    return selected_index, selected_face


def feature_dict(
    features: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = features.get(key)

    if isinstance(value, dict):
        return value

    return {}


def numeric_contracts_valid(
    face: Any,
    head_pose: dict[str, Any],
    landmarks: dict[str, Any],
    quality: dict[str, Any],
    eyes: dict[str, Any],
    gaze: dict[str, Any],
    gaze_estimation: dict[str, Any],
    mouth: dict[str, Any],
    emotion: dict[str, Any],
    regions: dict[str, Any],
) -> bool:
    if not finite_numeric(
        getattr(face, "confidence", None)
    ):
        return False

    box = getattr(face, "box", None)

    if not isinstance(
        box,
        (list, tuple, np.ndarray),
    ) or len(box) != 4:
        return False

    if not all(
        finite_numeric(value)
        for value in box
    ):
        return False

    x1, y1, x2, y2 = [
        float(value)
        for value in box
    ]

    if x2 <= x1 or y2 <= y1:
        return False

    if module_available(landmarks):
        if landmarks.get("count") != 478:
            return False

    if module_available(quality):
        for key in (
            "confidence",
            "brightness",
            "sharpness",
            "face_area_ratio",
        ):
            if not finite_numeric(
                quality.get(key)
            ):
                return False

        if not 0.0 <= float(
            quality["brightness"]
        ) <= 1.0:
            return False

        if float(
            quality["sharpness"]
        ) < 0.0:
            return False

        if not 0.0 <= float(
            quality["face_area_ratio"]
        ) <= 1.0:
            return False

    if module_available(head_pose):
        if not all(
            finite_numeric(
                head_pose.get(key)
            )
            for key in (
                "pitch",
                "yaw",
                "roll",
            )
        ):
            return False

    if module_available(eyes):
        for key in (
            "left_openness",
            "right_openness",
            "mean_openness",
        ):
            if not finite_numeric(
                eyes.get(key)
            ):
                return False

        expected_mean = (
            float(eyes["left_openness"])
            + float(eyes["right_openness"])
        ) / 2.0

        if not math.isclose(
            float(eyes["mean_openness"]),
            expected_mean,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return False

    if module_available(gaze):
        for key in (
            "right_iris_x",
            "right_iris_y",
            "left_iris_x",
            "left_iris_y",
            "mean_iris_x",
            "mean_iris_y",
        ):
            if not finite_numeric(
                gaze.get(key)
            ):
                return False

    if module_available(gaze_estimation):
        gaze_vector = gaze_estimation.get(
            "gaze_vector"
        )

        if not isinstance(
            gaze_vector,
            (list, tuple, np.ndarray),
        ) or len(gaze_vector) != 3:
            return False

        if not all(
            finite_numeric(value)
            for value in gaze_vector
        ):
            return False

        vector_norm = math.sqrt(
            sum(
                float(value) ** 2
                for value in gaze_vector
            )
        )

        if not math.isclose(
            vector_norm,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return False

        for key in (
            "pitch",
            "yaw",
            "association_iou",
        ):
            if not finite_numeric(
                gaze_estimation.get(key)
            ):
                return False

        if not 0.0 <= float(
            gaze_estimation["association_iou"]
        ) <= 1.0:
            return False

    if module_available(mouth):
        for key in (
            "mouth_openness",
            "mouth_width",
            "mouth_height",
        ):
            if not finite_numeric(
                mouth.get(key)
            ):
                return False

        if float(
            mouth["mouth_width"]
        ) <= 0.0:
            return False

    if module_available(emotion):
        label = emotion.get("emotion")
        scores = emotion.get("scores")

        if label not in EMOTION_LABELS:
            return False

        if not isinstance(scores, dict):
            return False

        if set(scores) != set(EMOTION_LABELS):
            return False

        if not all(
            finite_numeric(value)
            and 0.0 <= float(value) <= 1.0
            for value in scores.values()
        ):
            return False

        score_sum = sum(
            float(value)
            for value in scores.values()
        )

        if not math.isclose(
            score_sum,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return False

        dominant = max(
            scores,
            key=scores.get,
        )

        if dominant != label:
            return False

        if not finite_numeric(
            emotion.get("confidence")
        ):
            return False

        if not math.isclose(
            float(emotion["confidence"]),
            float(scores[label]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            return False

    if module_available(regions):
        for key in (
            "skin_pixel_count",
            "skin_fraction",
            "association_iou",
        ):
            if not finite_numeric(
                regions.get(key)
            ):
                return False

        if not 0.0 <= float(
            regions["skin_fraction"]
        ) <= 1.0:
            return False

        if not 0.0 <= float(
            regions["association_iou"]
        ) <= 1.0:
            return False

    return True


def empty_result(
    metadata: dict[str, str],
    width: int,
    height: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    row = {
        field: None
        for field in RESULT_FIELDS
    }

    row.update(
        {
            "case_id": metadata["case_id"],
            "condition": metadata["condition"],
            "image_file": metadata["output_file"],
            "image_width": width,
            "image_height": height,
            "detected_faces": 0,
            "status": status,
            "failure_reason": reason,
            "detection_available": False,
            "landmarks_available": False,
            "quality_available": False,
            "head_pose_available": False,
            "eyes_available": False,
            "gaze_available": False,
            "gaze_estimation_available": False,
            "mouth_available": False,
            "emotion_available": False,
            "regions_available": False,
            "all_available_numeric_values_finite": (
                True
                if status == "NO_FACE"
                else False
            ),
        }
    )

    return row


def evaluate_case(
    pipeline: FaceAnalysis,
    metadata: dict[str, str],
) -> dict[str, Any]:
    image_path = TEST_DATA_DIR / metadata["output_file"]
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read robustness image: {image_path}"
        )

    height, width = image.shape[:2]

    try:
        result = pipeline.predict(image)
        faces = list(result)

    except Exception as exc:
        return empty_result(
            metadata,
            width,
            height,
            "EXECUTION_FAILED",
            f"{type(exc).__name__}: {exc}",
        )

    if not faces:
        return empty_result(
            metadata,
            width,
            height,
            "NO_FACE",
            "",
        )

    selected_index, face = select_primary_face(
        faces
    )

    features = (
        face.face_features
        if isinstance(
            face.face_features,
            dict,
        )
        else {}
    )

    landmarks = feature_dict(
        features,
        "landmarks",
    )
    quality = feature_dict(
        features,
        "quality",
    )
    eyes = feature_dict(
        features,
        "eyes",
    )
    gaze = feature_dict(
        features,
        "gaze",
    )
    gaze_estimation = feature_dict(
        features,
        "gaze_estimation",
    )
    mouth = feature_dict(
        features,
        "mouth",
    )
    emotion = feature_dict(
        features,
        "emotion",
    )
    regions = feature_dict(
        features,
        "regions",
    )

    head_pose_raw = get_head_pose(
        face,
        features,
    )
    head_pose = (
        head_pose_raw
        if isinstance(
            head_pose_raw,
            dict,
        )
        else {}
    )

    availability = {
        "detection_available": True,
        "landmarks_available":
            module_available(landmarks),
        "quality_available":
            module_available(quality),
        "head_pose_available":
            module_available(head_pose),
        "eyes_available":
            module_available(eyes),
        "gaze_available":
            module_available(gaze),
        "gaze_estimation_available":
            module_available(gaze_estimation),
        "mouth_available":
            module_available(mouth),
        "emotion_available":
            module_available(emotion),
        "regions_available":
            module_available(regions),
    }

    contracts_valid = numeric_contracts_valid(
        face,
        head_pose,
        landmarks,
        quality,
        eyes,
        gaze,
        gaze_estimation,
        mouth,
        emotion,
        regions,
    )

    box = [
        float(value)
        for value in face.box
    ]
    x1, y1, x2, y2 = box

    row = {
        field: None
        for field in RESULT_FIELDS
    }

    row.update(
        {
            "case_id": metadata["case_id"],
            "condition": metadata["condition"],
            "image_file": metadata["output_file"],
            "image_width": width,
            "image_height": height,
            "detected_faces": len(faces),
            "selected_face_index": selected_index,
            "status": (
                "DETECTED"
                if contracts_valid
                else "INVALID_NUMERIC_OUTPUT"
            ),
            "failure_reason": (
                ""
                if contracts_valid
                else "One or more available outputs violate numerical contracts"
            ),
            **availability,
            "all_available_numeric_values_finite":
                contracts_valid,
            "detector_confidence":
                finite_or_none(
                    face.confidence
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
                x2 - x1,
            "box_height":
                y2 - y1,
            "box_area":
                (x2 - x1) * (y2 - y1),
            "quality_brightness":
                finite_or_none(
                    quality.get(
                        "brightness"
                    )
                ),
            "quality_sharpness":
                finite_or_none(
                    quality.get(
                        "sharpness"
                    )
                ),
            "quality_face_area_ratio":
                finite_or_none(
                    quality.get(
                        "face_area_ratio"
                    )
                ),
            "head_pose_pitch":
                finite_or_none(
                    head_pose.get(
                        "pitch"
                    )
                ),
            "head_pose_yaw":
                finite_or_none(
                    head_pose.get(
                        "yaw"
                    )
                ),
            "head_pose_roll":
                finite_or_none(
                    head_pose.get(
                        "roll"
                    )
                ),
            "eye_mean_openness":
                finite_or_none(
                    eyes.get(
                        "mean_openness"
                    )
                ),
            "gaze_mean_iris_x":
                finite_or_none(
                    gaze.get(
                        "mean_iris_x"
                    )
                ),
            "gaze_mean_iris_y":
                finite_or_none(
                    gaze.get(
                        "mean_iris_y"
                    )
                ),
            "gaze_estimation_pitch":
                finite_or_none(
                    gaze_estimation.get(
                        "pitch"
                    )
                ),
            "gaze_estimation_yaw":
                finite_or_none(
                    gaze_estimation.get(
                        "yaw"
                    )
                ),
            "gaze_association_iou":
                finite_or_none(
                    gaze_estimation.get(
                        "association_iou"
                    )
                ),
            "mouth_openness":
                finite_or_none(
                    mouth.get(
                        "mouth_openness"
                    )
                ),
            "emotion_label":
                emotion.get(
                    "emotion"
                ),
            "emotion_confidence":
                finite_or_none(
                    emotion.get(
                        "confidence"
                    )
                ),
            "regions_skin_fraction":
                finite_or_none(
                    regions.get(
                        "skin_fraction"
                    )
                ),
            "regions_association_iou":
                finite_or_none(
                    regions.get(
                        "association_iou"
                    )
                ),
        }
    )

    return row


def abs_delta(
    value: Any,
    baseline_value: Any,
) -> float | None:
    if not finite_numeric(value):
        return None

    if not finite_numeric(baseline_value):
        return None

    return abs(
        float(value)
        - float(baseline_value)
    )


def add_baseline_deltas(
    rows: list[dict[str, Any]],
) -> None:
    baseline_rows = [
        row
        for row in rows
        if row[
            "case_id"
        ] == "baseline"
    ]

    if len(baseline_rows) != 1:
        raise RuntimeError(
            "Exactly one baseline robustness result is required"
        )

    baseline = baseline_rows[0]

    if baseline["status"] != "DETECTED":
        raise RuntimeError(
            "Baseline robustness case must produce a valid detected face"
        )

    mappings = [
        (
            "detector_confidence",
            "detector_confidence_abs_delta_vs_baseline",
        ),
        (
            "quality_brightness",
            "quality_brightness_abs_delta_vs_baseline",
        ),
        (
            "quality_sharpness",
            "quality_sharpness_abs_delta_vs_baseline",
        ),
        (
            "quality_face_area_ratio",
            "quality_face_area_ratio_abs_delta_vs_baseline",
        ),
        (
            "head_pose_pitch",
            "head_pose_pitch_abs_delta_vs_baseline",
        ),
        (
            "head_pose_yaw",
            "head_pose_yaw_abs_delta_vs_baseline",
        ),
        (
            "head_pose_roll",
            "head_pose_roll_abs_delta_vs_baseline",
        ),
        (
            "eye_mean_openness",
            "eye_mean_openness_abs_delta_vs_baseline",
        ),
        (
            "gaze_mean_iris_x",
            "gaze_mean_iris_x_abs_delta_vs_baseline",
        ),
        (
            "gaze_mean_iris_y",
            "gaze_mean_iris_y_abs_delta_vs_baseline",
        ),
        (
            "gaze_estimation_pitch",
            "gaze_estimation_pitch_abs_delta_vs_baseline",
        ),
        (
            "gaze_estimation_yaw",
            "gaze_estimation_yaw_abs_delta_vs_baseline",
        ),
        (
            "mouth_openness",
            "mouth_openness_abs_delta_vs_baseline",
        ),
        (
            "emotion_confidence",
            "emotion_confidence_abs_delta_vs_baseline",
        ),
        (
            "regions_skin_fraction",
            "regions_skin_fraction_abs_delta_vs_baseline",
        ),
    ]

    for row in rows:
        for value_field, delta_field in mappings:
            row[delta_field] = abs_delta(
                row.get(value_field),
                baseline.get(value_field),
            )

        row[
            "emotion_changed_vs_baseline"
        ] = (
            None
            if (
                row.get(
                    "emotion_label"
                ) is None
                or baseline.get(
                    "emotion_label"
                ) is None
            )
            else (
                row[
                    "emotion_label"
                ]
                != baseline[
                    "emotion_label"
                ]
            )
        )


def count_available_modules(
    row: dict[str, Any],
) -> int:
    fields = [
        "detection_available",
        "landmarks_available",
        "quality_available",
        "head_pose_available",
        "eyes_available",
        "gaze_available",
        "gaze_estimation_available",
        "mouth_available",
        "emotion_available",
        "regions_available",
    ]

    return sum(
        bool(
            row.get(field)
        )
        for field in fields
    )


def build_summary(
    rows: list[dict[str, Any]],
    inventory_unchanged: bool,
) -> dict[str, Any]:
    baseline = next(
        row
        for row in rows
        if row[
            "case_id"
        ] == "baseline"
    )

    execution_failed = sum(
        row[
            "status"
        ] == "EXECUTION_FAILED"
        for row in rows
    )
    no_face = sum(
        row[
            "status"
        ] == "NO_FACE"
        for row in rows
    )
    invalid_numeric = sum(
        row[
            "status"
        ] == "INVALID_NUMERIC_OUTPUT"
        for row in rows
    )
    detected = sum(
        row[
            "status"
        ] == "DETECTED"
        for row in rows
    )

    case_summaries = []

    for row in rows:
        case_summaries.append(
            {
                "case_id":
                    row["case_id"],
                "condition":
                    row["condition"],
                "status":
                    row["status"],
                "detected_faces":
                    row["detected_faces"],
                "available_static_modules":
                    count_available_modules(
                        row
                    ),
                "total_static_modules":
                    len(STATIC_MODULES),
                "detector_confidence":
                    row[
                        "detector_confidence"
                    ],
                "emotion_label":
                    row[
                        "emotion_label"
                    ],
                "emotion_changed_vs_baseline":
                    row[
                        "emotion_changed_vs_baseline"
                    ],
            }
        )

    execution_status = (
        "PASS"
        if (
            execution_failed == 0
            and invalid_numeric == 0
            and inventory_unchanged
        )
        else "FAIL"
    )

    return {
        "test":
            "controlled_static_robustness",
        "evidence_type":
            "controlled_robustness_behavior",
        "scientific_accuracy_evidence":
            False,
        "protocol": {
            "source_population":
                (
                    "Seven deterministic stress-test images generated from "
                    "one accepted frontal integration image."
                ),
            "primary_face_selection":
                (
                    "Highest detector-confidence face within each generated "
                    "single-person stress-test image."
                ),
            "comparison":
                (
                    "Each stress condition is compared descriptively with "
                    "the unmodified baseline. No arbitrary robustness pass/fail "
                    "threshold is imposed on prediction drift."
                ),
            "temporal_scope":
                (
                    "Static-image compatible components only. Tracking, blink, "
                    "mouth motion, and temporal aggregation are intentionally "
                    "not evaluated from these single-image controlled inputs."
                ),
        },
        "configuration": {
            "active_components": STATIC_MODULES,
            "not_applicable_static_input_components": [
                "tracking",
                "blink",
                "mouth_motion",
                "temporal",
            ],
            "gaze_estimation_mode":
                "eth-xgaze",
            "gaze_estimation_min_iou":
                0.10,
        },
        "execution": {
            "cases":
                len(rows),
            "detected_cases":
                detected,
            "no_face_cases":
                no_face,
            "execution_failed_cases":
                execution_failed,
            "invalid_numeric_cases":
                invalid_numeric,
            "input_read_only_verified":
                inventory_unchanged,
        },
        "baseline": {
            "detected_faces":
                baseline[
                    "detected_faces"
                ],
            "detector_confidence":
                baseline[
                    "detector_confidence"
                ],
            "emotion_label":
                baseline[
                    "emotion_label"
                ],
            "available_static_modules":
                count_available_modules(
                    baseline
                ),
        },
        "cases":
            case_summaries,
        "interpretation":
            (
                "This evaluation characterizes deterministic degradation "
                "behavior under controlled image perturbations. A NO_FACE "
                "outcome is an observed robustness limitation, not a script "
                "execution failure. The test does not provide predictive "
                "accuracy or a universal robustness score."
            ),
        "execution_status":
            execution_status,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_staging(
    staging_dir: Path,
) -> None:
    results_path = (
        staging_dir
        / RESULTS_FILENAME
    )
    summary_path = (
        staging_dir
        / SUMMARY_FILENAME
    )

    for path in (
        results_path,
        summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected staged robustness output missing: {path}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged robustness output is empty: {path}"
            )

    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != len(EXPECTED_CASES):
        raise RuntimeError(
            "Staged robustness result row count mismatch"
        )

    if [
        row["case_id"]
        for row in rows
    ] != EXPECTED_CASES:
        raise RuntimeError(
            "Staged robustness result ordering mismatch"
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    if (
        summary.get("test")
        != "controlled_static_robustness"
    ):
        raise RuntimeError(
            "Unexpected robustness summary test identifier"
        )

    if int(
        summary.get(
            "execution",
            {},
        ).get(
            "cases",
            -1,
        )
    ) != len(EXPECTED_CASES):
        raise RuntimeError(
            "Robustness summary case count mismatch"
        )


def promote_staged_files(
    staging_dir: Path,
) -> None:
    targets = [
        RESULTS_FILENAME,
        SUMMARY_FILENAME,
    ]

    backups: dict[str, Path] = {}
    promoted: list[Path] = []

    try:
        for filename in targets:
            final_path = (
                FINAL_RESULTS_DIR
                / filename
            )
            staged_path = (
                staging_dir
                / filename
            )

            if final_path.exists():
                backup_path = (
                    staging_dir
                    / f".backup_{filename}"
                )
                os.replace(
                    final_path,
                    backup_path,
                )
                backups[
                    filename
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

        for filename, backup_path in backups.items():
            if backup_path.exists():
                os.replace(
                    backup_path,
                    FINAL_RESULTS_DIR
                    / filename,
                )

        raise

    for backup_path in backups.values():
        if backup_path.exists():
            backup_path.unlink()


def main() -> None:
    args = parse_args()

    metadata_rows = preflight_validation()

    if args.preflight_only:
        print(
            "Controlled robustness evaluator preflight: PASS"
        )
        print(
            f"Cases: {len(metadata_rows)}"
        )
        print(
            "Static pipeline components:"
        )

        for module in STATIC_MODULES:
            print(
                f"- {module}"
            )

        print(
            "Not applicable for single-image controlled inputs:"
        )
        print(
            "- tracking"
        )
        print(
            "- blink"
        )
        print(
            "- mouth_motion"
        )
        print(
            "- temporal"
        )

        return

    before_inventory = input_inventory(
        metadata_rows
    )

    pipeline = FaceAnalysis(
        config=make_config()
    )

    rows = []

    try:
        for index, metadata in enumerate(
            metadata_rows,
            start=1,
        ):
            print(
                f"[{index}/{len(metadata_rows)}] "
                f"{metadata['case_id']}"
            )

            row = evaluate_case(
                pipeline,
                metadata,
            )
            rows.append(row)

            print(
                f"  status={row['status']} | "
                f"faces={row['detected_faces']} | "
                f"available_modules={count_available_modules(row)}/"
                f"{len(STATIC_MODULES)}"
            )

    finally:
        pipeline.close()

    add_baseline_deltas(rows)

    after_inventory = input_inventory(
        metadata_rows
    )
    inventory_unchanged = (
        before_inventory
        == after_inventory
    )

    summary = build_summary(
        rows,
        inventory_unchanged,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".robustness_eval_",
            dir=FINAL_RESULTS_DIR,
        )
    )

    try:
        write_csv(
            staging_dir
            / RESULTS_FILENAME,
            rows,
        )

        with (
            staging_dir
            / SUMMARY_FILENAME
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        validate_staging(
            staging_dir
        )
        promote_staged_files(
            staging_dir
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    print()
    print(
        "Controlled Robustness Evaluation"
    )
    print(
        "=" * 32
    )
    print(
        f"Cases: {summary['execution']['cases']}"
    )
    print(
        "Detected cases: "
        f"{summary['execution']['detected_cases']}"
    )
    print(
        "NO_FACE cases: "
        f"{summary['execution']['no_face_cases']}"
    )
    print(
        "Execution failures: "
        f"{summary['execution']['execution_failed_cases']}"
    )
    print(
        "Invalid numerical cases: "
        f"{summary['execution']['invalid_numeric_cases']}"
    )
    print(
        "Input read-only verification: "
        f"{'PASS' if summary['execution']['input_read_only_verified'] else 'FAIL'}"
    )
    print(
        "Execution status: "
        f"{summary['execution_status']}"
    )
    print(
        f"Results CSV: {FINAL_RESULTS_DIR / RESULTS_FILENAME}"
    )
    print(
        f"Summary JSON: {FINAL_RESULTS_DIR / SUMMARY_FILENAME}"
    )


if __name__ == "__main__":
    main()
