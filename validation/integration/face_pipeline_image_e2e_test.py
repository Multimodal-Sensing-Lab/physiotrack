from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "images"
)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "image_e2e"
)

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

NOT_APPLICABLE_MODULES = [
    "tracking",
    "blink",
    "mouth_motion",
    "temporal",
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


def get_image_paths() -> list[Path]:
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test-data directory not found: {TEST_DATA_DIR}"
        )

    image_paths = sorted(
        path
        for path in TEST_DATA_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files found in: {TEST_DATA_DIR}"
        )

    return image_paths


def image_label(
    image_path: Path,
) -> str:
    return str(
        image_path.relative_to(SCRIPT_DIR)
    ).replace("\\", "/")


def clean_output_directory() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def finite_numeric(value: Any) -> bool:
    if value is None:
        return False

    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def module_available(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, dict):
        if "available" in value:
            return bool(
                value["available"]
            )

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
            value = getattr(
                face,
                name,
            )

            if value is not None:
                return value

    return None


def validate_static_values(
    values: dict[str, Any],
) -> dict[str, bool]:
    eyes = values[
        "eyes"
    ]

    mouth = values[
        "mouth"
    ]

    return {
        "eye_openness_finite":
            (
                isinstance(
                    eyes,
                    dict,
                )
                and eyes.get(
                    "available",
                    False,
                )
                and finite_numeric(
                    eyes.get(
                        "mean_openness"
                    )
                )
            ),
        "mouth_openness_finite":
            (
                isinstance(
                    mouth,
                    dict,
                )
                and mouth.get(
                    "available",
                    False,
                )
                and finite_numeric(
                    mouth.get(
                        "mouth_openness"
                    )
                )
            ),
    }


def static_contracts_valid(
    face: Any,
    values: dict[str, Any],
) -> bool:
    landmarks = values.get(
        "landmarks"
    )
    quality = values.get(
        "quality"
    )
    head_pose = values.get(
        "head_pose"
    )
    eyes = values.get(
        "eyes"
    )
    gaze = values.get(
        "gaze"
    )
    gaze_estimation = values.get(
        "gaze_estimation"
    )
    mouth = values.get(
        "mouth"
    )
    emotion = values.get(
        "emotion"
    )
    regions = values.get(
        "regions"
    )

    if not finite_numeric(
        face.confidence
    ):
        return False

    box = face.box

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

    if not isinstance(landmarks, dict):
        return False

    if landmarks.get(
        "count"
    ) != 478:
        return False

    if not isinstance(quality, dict):
        return False

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

    if not math.isclose(
        float(quality["confidence"]),
        float(face.confidence),
        rel_tol=1e-9,
        abs_tol=1e-9,
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

    if not isinstance(head_pose, dict):
        return False

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

    if not isinstance(eyes, dict):
        return False

    eye_values = [
        eyes.get(
            "left_openness"
        ),
        eyes.get(
            "right_openness"
        ),
        eyes.get(
            "mean_openness"
        ),
    ]

    if not all(
        finite_numeric(value)
        for value in eye_values
    ):
        return False

    expected_eye_mean = (
        float(eye_values[0])
        + float(eye_values[1])
    ) / 2.0

    if not math.isclose(
        float(eye_values[2]),
        expected_eye_mean,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return False

    if not isinstance(gaze, dict):
        return False

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

    expected_gaze_x = (
        float(gaze["right_iris_x"])
        + float(gaze["left_iris_x"])
    ) / 2.0

    expected_gaze_y = (
        float(gaze["right_iris_y"])
        + float(gaze["left_iris_y"])
    ) / 2.0

    if not math.isclose(
        float(gaze["mean_iris_x"]),
        expected_gaze_x,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return False

    if not math.isclose(
        float(gaze["mean_iris_y"]),
        expected_gaze_y,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return False

    if not isinstance(
        gaze_estimation,
        dict,
    ):
        return False

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

    if not isinstance(mouth, dict):
        return False

    for key in (
        "mouth_openness",
        "mouth_width",
        "mouth_height",
    ):
        if not finite_numeric(
            mouth.get(key)
        ):
            return False

    mouth_width = float(
        mouth["mouth_width"]
    )

    if mouth_width <= 0.0:
        return False

    expected_mouth_openness = (
        float(mouth["mouth_height"])
        / mouth_width
    )

    if not math.isclose(
        float(mouth["mouth_openness"]),
        expected_mouth_openness,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return False

    if not isinstance(emotion, dict):
        return False

    emotion_label = emotion.get(
        "emotion"
    )
    emotion_scores = emotion.get(
        "scores"
    )

    if emotion_label not in EMOTION_LABELS:
        return False

    if not isinstance(
        emotion_scores,
        dict,
    ):
        return False

    if set(
        emotion_scores
    ) != set(EMOTION_LABELS):
        return False

    if not all(
        finite_numeric(value)
        and float(value) >= 0.0
        for value in emotion_scores.values()
    ):
        return False

    score_sum = sum(
        float(value)
        for value in emotion_scores.values()
    )

    if not math.isclose(
        score_sum,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        return False

    dominant_label = max(
        emotion_scores,
        key=emotion_scores.get,
    )

    if dominant_label != emotion_label:
        return False

    if not finite_numeric(
        emotion.get(
            "confidence"
        )
    ):
        return False

    if not math.isclose(
        float(emotion["confidence"]),
        float(emotion_scores[emotion_label]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        return False

    if not isinstance(regions, dict):
        return False

    pixel_counts = regions.get(
        "pixel_counts"
    )

    if not isinstance(
        pixel_counts,
        dict,
    ) or not pixel_counts:
        return False

    if not all(
        finite_numeric(value)
        and float(value) >= 0.0
        for value in pixel_counts.values()
    ):
        return False

    for key in (
        "skin_pixel_count",
        "skin_fraction",
        "association_iou",
    ):
        if not finite_numeric(
            regions.get(key)
        ):
            return False

    if float(
        regions["skin_pixel_count"]
    ) < 0.0:
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


def temporal_outputs_absent_for_face(
    face: Any,
    features: dict[str, Any],
) -> bool:
    if face.id is not None:
        return False

    blink = features.get(
        "blink"
    )

    if isinstance(blink, dict):
        if blink.get(
            "available",
            False,
        ):
            return False

        if blink.get(
            "eye_state"
        ) not in (
            None,
            "unknown",
        ):
            return False

        if blink.get(
            "blink",
            False,
        ):
            return False

        if blink.get(
            "blink_count",
            0,
        ) not in (
            None,
            0,
        ):
            return False

        if blink.get(
            "blink_duration"
        ) is not None:
            return False

        if blink.get(
            "blink_rate"
        ) is not None:
            return False

    mouth_motion = features.get(
        "mouth_motion"
    )

    if isinstance(
        mouth_motion,
        dict,
    ):
        if mouth_motion.get(
            "available",
            False,
        ):
            return False

        if mouth_motion.get(
            "mouth_movement"
        ) is not None:
            return False

        if mouth_motion.get(
            "mouth_velocity"
        ) is not None:
            return False

    temporal = features.get(
        "temporal"
    )

    if isinstance(temporal, dict):
        if temporal.get(
            "available",
            False,
        ):
            return False

        if temporal.get(
            "summary"
        ) is not None:
            return False

    return True


def module_status(
    successful: int,
    total: int,
) -> str:
    if total <= 0:
        return "UNAVAILABLE"

    if successful == total:
        return "PASS"

    if successful > 0:
        return "PARTIAL"

    return "UNAVAILABLE"


def list_value(
    value: Any,
    index: int,
) -> Any:
    if not isinstance(
        value,
        (list, tuple),
    ):
        return None

    if index >= len(value):
        return None

    return value[index]


def record_to_csv_row(
    record: dict[str, Any],
    region_class_names: list[str],
) -> dict[str, Any]:
    features = record.get(
        "face_features",
        {},
    )

    if not isinstance(
        features,
        dict,
    ):
        features = {}

    head_pose = record.get(
        "head_pose",
        {},
    )

    if not isinstance(
        head_pose,
        dict,
    ):
        head_pose = {}

    landmarks = features.get(
        "landmarks",
        {},
    )

    quality = features.get(
        "quality",
        {},
    )

    eyes = features.get(
        "eyes",
        {},
    )

    gaze = features.get(
        "gaze",
        {},
    )

    gaze_estimation = features.get(
        "gaze_estimation",
        {},
    )

    mouth = features.get(
        "mouth",
        {},
    )

    emotion = features.get(
        "emotion",
        {},
    )

    regions = features.get(
        "regions",
        {},
    )

    if not isinstance(landmarks, dict):
        landmarks = {}

    if not isinstance(quality, dict):
        quality = {}

    if not isinstance(eyes, dict):
        eyes = {}

    if not isinstance(gaze, dict):
        gaze = {}

    if not isinstance(gaze_estimation, dict):
        gaze_estimation = {}

    if not isinstance(mouth, dict):
        mouth = {}

    if not isinstance(emotion, dict):
        emotion = {}

    if not isinstance(regions, dict):
        regions = {}

    box = record.get(
        "box"
    )

    gaze_vector = (
        gaze_estimation.get(
            "gaze_vector"
        )
    )

    emotion_scores = emotion.get(
        "scores",
        {},
    )

    if not isinstance(
        emotion_scores,
        dict,
    ):
        emotion_scores = {}

    region_pixel_counts = regions.get(
        "pixel_counts",
        {},
    )

    if not isinstance(
        region_pixel_counts,
        dict,
    ):
        region_pixel_counts = {}

    row = {
        "image":
            record.get(
                "image"
            ),
        "face_index":
            record.get(
                "face_index"
            ),
        "person_id":
            record.get(
                "person_id"
            ),
        "detection_confidence":
            record.get(
                "confidence"
            ),
        "box_x1":
            list_value(
                box,
                0,
            ),
        "box_y1":
            list_value(
                box,
                1,
            ),
        "box_x2":
            list_value(
                box,
                2,
            ),
        "box_y2":
            list_value(
                box,
                3,
            ),
        "landmarks_count":
            landmarks.get(
                "count"
            ),
        "head_pose_pitch":
            head_pose.get(
                "pitch"
            ),
        "head_pose_yaw":
            head_pose.get(
                "yaw"
            ),
        "head_pose_roll":
            head_pose.get(
                "roll"
            ),
        "quality_confidence":
            quality.get(
                "confidence"
            ),
        "quality_brightness":
            quality.get(
                "brightness"
            ),
        "quality_sharpness":
            quality.get(
                "sharpness"
            ),
        "quality_face_area_ratio":
            quality.get(
                "face_area_ratio"
            ),
        "eye_left_openness":
            eyes.get(
                "left_openness"
            ),
        "eye_right_openness":
            eyes.get(
                "right_openness"
            ),
        "eye_mean_openness":
            eyes.get(
                "mean_openness"
            ),
        "gaze_right_iris_x":
            gaze.get(
                "right_iris_x"
            ),
        "gaze_right_iris_y":
            gaze.get(
                "right_iris_y"
            ),
        "gaze_left_iris_x":
            gaze.get(
                "left_iris_x"
            ),
        "gaze_left_iris_y":
            gaze.get(
                "left_iris_y"
            ),
        "gaze_mean_iris_x":
            gaze.get(
                "mean_iris_x"
            ),
        "gaze_mean_iris_y":
            gaze.get(
                "mean_iris_y"
            ),
        "gaze_estimation_pitch":
            gaze_estimation.get(
                "pitch"
            ),
        "gaze_estimation_yaw":
            gaze_estimation.get(
                "yaw"
            ),
        "gaze_vector_x":
            list_value(
                gaze_vector,
                0,
            ),
        "gaze_vector_y":
            list_value(
                gaze_vector,
                1,
            ),
        "gaze_vector_z":
            list_value(
                gaze_vector,
                2,
            ),
        "gaze_association_iou":
            gaze_estimation.get(
                "association_iou"
            ),
        "mouth_openness":
            mouth.get(
                "mouth_openness"
            ),
        "mouth_width":
            mouth.get(
                "mouth_width"
            ),
        "mouth_height":
            mouth.get(
                "mouth_height"
            ),
        "emotion_label":
            emotion.get(
                "emotion"
            ),
        "emotion_confidence":
            emotion.get(
                "confidence"
            ),
        "regions_skin_pixel_count":
            regions.get(
                "skin_pixel_count"
            ),
        "regions_skin_fraction":
            regions.get(
                "skin_fraction"
            ),
        "regions_association_iou":
            regions.get(
                "association_iou"
            ),
    }

    for label in EMOTION_LABELS:
        row[
            "emotion_score_"
            + label.lower()
        ] = emotion_scores.get(
            label
        )

    for class_name in region_class_names:
        row[
            "regions_pixels_"
            + class_name
        ] = region_pixel_counts.get(
            class_name
        )

    for module in STATIC_MODULES:
        row[
            "module_"
            + module
        ] = (
            record.get(
                "module_status",
                {},
            ).get(
                module
            )
        )

    return row


def failed_summary(
    image_path: Path,
    reason: str,
) -> dict[str, Any]:
    modules = {}

    for module in STATIC_MODULES:
        modules[module] = {
            "status":
                "UNAVAILABLE",
            "successful_face_samples":
                0,
            "total_face_samples":
                0,
            "coverage_percent":
                0.0,
        }

    for module in NOT_APPLICABLE_MODULES:
        modules[module] = {
            "status":
                "NOT_APPLICABLE",
            "reason":
                (
                    "Requires temporal sequence input "
                    "and is not evaluated from a single image."
                ),
        }

    return {
        "image":
            image_label(
                image_path
            ),
        "resolution":
            None,
        "detected_faces":
            0,
        "frame_records":
            0,
        "modules":
            modules,
        "eye_openness": {
            "finite_samples":
                0,
            "matches_available_eye_samples":
                False,
        },
        "mouth_openness": {
            "finite_samples":
                0,
            "matches_available_mouth_samples":
                False,
        },
        "temporal_outputs_absent":
            True,
        "failure_reason":
            reason,
        "overall_status":
            "FAIL",
    }


def run_image(
    pipeline: FaceAnalysis,
    image_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    result = pipeline.predict(
        image
    )

    faces = list(
        result
    )

    if not faces:
        raise RuntimeError(
            f"{image_label(image_path)}: no faces were detected."
        )

    export_records = (
        FaceResultExporter.frame_records(
            result,
            frame_index=None,
            timestamp=None,
        )
    )

    if len(export_records) != len(faces):
        raise RuntimeError(
            f"{image_label(image_path)}: frame export record count "
            "does not match detected face count."
        )

    records: list[dict[str, Any]] = []

    module_success_counts = {
        module: 0
        for module in STATIC_MODULES
    }

    valid_eye_openness_samples = 0
    valid_mouth_openness_samples = 0
    all_static_contracts_valid = True

    for (
        face_index,
        face,
    ) in enumerate(
        faces
    ):
        features = (
            face.face_features
            if face.face_features is not None
            else {}
        )

        head_pose = get_head_pose(
            face,
            features,
        )

        values = {
            "landmarks":
                features.get(
                    "landmarks"
                ),
            "quality":
                features.get(
                    "quality"
                ),
            "head_pose":
                head_pose,
            "eyes":
                features.get(
                    "eyes"
                ),
            "gaze":
                features.get(
                    "gaze"
                ),
            "gaze_estimation":
                features.get(
                    "gaze_estimation"
                ),
            "mouth":
                features.get(
                    "mouth"
                ),
            "emotion":
                features.get(
                    "emotion"
                ),
            "regions":
                features.get(
                    "regions"
                ),
        }

        status = {
            "detection":
                True,
        }

        module_success_counts[
            "detection"
        ] += 1

        for module, value in (
            values.items()
        ):
            available = (
                module_available(
                    value
                )
            )

            status[module] = (
                available
            )

            if available:
                module_success_counts[
                    module
                ] += 1

        static_value_checks = (
            validate_static_values(
                values
            )
        )

        if not static_contracts_valid(
            face,
            values,
        ):
            all_static_contracts_valid = False

        if static_value_checks[
            "eye_openness_finite"
        ]:
            valid_eye_openness_samples += 1

        if static_value_checks[
            "mouth_openness_finite"
        ]:
            valid_mouth_openness_samples += 1

        records.append(
            {
                "image":
                    image_label(
                        image_path
                    ),
                "face_index":
                    face_index,
                "person_id":
                    to_jsonable(
                        face.id
                    ),
                "box":
                    to_jsonable(
                        face.box
                    ),
                "confidence":
                    to_jsonable(
                        face.confidence
                    ),
                "module_status":
                    status,
                "static_value_checks":
                    static_value_checks,
                "head_pose":
                    to_jsonable(
                        head_pose
                    ),
                "face_features":
                    to_jsonable(
                        features
                    ),
            }
        )

    module_summary = {}

    for module in STATIC_MODULES:
        successful = (
            module_success_counts[
                module
            ]
        )

        module_summary[module] = {
            "status":
                module_status(
                    successful,
                    len(faces),
                ),
            "successful_face_samples":
                successful,
            "total_face_samples":
                len(faces),
            "coverage_percent":
                (
                    100.0
                    * successful
                    / len(faces)
                ),
        }

    for module in NOT_APPLICABLE_MODULES:
        module_summary[module] = {
            "status":
                "NOT_APPLICABLE",
            "reason":
                (
                    "Requires temporal sequence input "
                    "and is not evaluated from a single image."
                ),
        }

    all_static_modules_complete = all(
        module_summary[
            module
        ][
            "status"
        ]
        == "PASS"
        for module in STATIC_MODULES
    )

    eye_openness_values_valid = (
        valid_eye_openness_samples > 0
        and valid_eye_openness_samples
        == module_success_counts[
            "eyes"
        ]
    )

    mouth_openness_values_valid = (
        valid_mouth_openness_samples > 0
        and valid_mouth_openness_samples
        == module_success_counts[
            "mouth"
        ]
    )

    temporal_outputs_absent = all(
        temporal_outputs_absent_for_face(
            face,
            (
                face.face_features
                if face.face_features is not None
                else {}
            ),
        )
        for face in faces
    )

    overall_pass = (
        len(faces) > 0
        and all_static_modules_complete
        and all_static_contracts_valid
        and eye_openness_values_valid
        and mouth_openness_values_valid
        and temporal_outputs_absent
    )

    summary = {
        "image":
            image_label(
                image_path
            ),
        "resolution": {
            "width":
                width,
            "height":
                height,
        },
        "detected_faces":
            len(faces),
        "frame_records":
            len(export_records),
        "modules":
            module_summary,
        "eye_openness": {
            "finite_samples":
                valid_eye_openness_samples,
            "matches_available_eye_samples":
                eye_openness_values_valid,
        },
        "mouth_openness": {
            "finite_samples":
                valid_mouth_openness_samples,
            "matches_available_mouth_samples":
                mouth_openness_values_valid,
        },
        "temporal_outputs_absent":
            temporal_outputs_absent,
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    failed_checks = []

    if not all_static_modules_complete:
        failed_checks.append(
            "one_or_more_static_modules_not_complete"
        )

    if not all_static_contracts_valid:
        failed_checks.append(
            "one_or_more_static_numerical_contracts_invalid"
        )

    if not eye_openness_values_valid:
        failed_checks.append(
            "eye_openness_values_invalid"
        )

    if not mouth_openness_values_valid:
        failed_checks.append(
            "mouth_openness_values_invalid"
        )

    if not temporal_outputs_absent:
        failed_checks.append(
            "temporal_output_present_for_image_test"
        )

    if failed_checks:
        summary[
            "failure_reason"
        ] = ", ".join(
            failed_checks
        )

    return records, summary


def main() -> None:
    image_paths = get_image_paths()

    clean_output_directory()

    config = make_config()

    all_records: list[
        dict[str, Any]
    ] = []

    image_summaries: list[
        dict[str, Any]
    ] = []

    for image_path in image_paths:
        print()
        print("=" * 82)
        print(f"Testing image: {image_path}")
        print("=" * 82)

        pipeline = FaceAnalysis(
            config=config,
        )

        try:
            records, summary = (
                run_image(
                    pipeline,
                    image_path,
                )
            )

        except Exception as exc:
            records = []

            summary = failed_summary(
                image_path,
                str(exc),
            )

        finally:
            pipeline.close()

        all_records.extend(
            records
        )

        image_summaries.append(
            summary
        )

        print(
            "Detected faces:",
            summary[
                "detected_faces"
            ],
        )

        print(
            "Status:",
            summary[
                "overall_status"
            ],
        )

        if summary.get(
            "failure_reason"
        ):
            print(
                "Reason:",
                summary[
                    "failure_reason"
                ],
            )

    overall_pass = all(
        summary[
            "overall_status"
        ]
        == "PASS"
        for summary in image_summaries
    )

    results_json_path = (
        OUTPUT_DIR
        / "image_e2e_results.json"
    )

    frames_csv_path = (
        OUTPUT_DIR
        / "image_e2e_frames.csv"
    )

    summary_json_path = (
        OUTPUT_DIR
        / "image_e2e_summary.json"
    )

    modules_csv_path = (
        OUTPUT_DIR
        / "image_e2e_modules.csv"
    )

    with results_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "test_type":
                    "image_end_to_end_integration",
                "images":
                    image_summaries,
                "records":
                    all_records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    region_class_names = sorted(
        {
            class_name
            for record in all_records
            for features in [
                (
                    record.get(
                        "face_features",
                        {},
                    )
                )
            ]
            if isinstance(
                features,
                dict,
            )
            for regions in [
                (
                    features.get(
                        "regions",
                        {},
                    )
                )
            ]
            if isinstance(
                regions,
                dict,
            )
            for pixel_counts in [
                (
                    regions.get(
                        "pixel_counts",
                        {},
                    )
                )
            ]
            if isinstance(
                pixel_counts,
                dict,
            )
            for class_name in pixel_counts
        }
    )

    frame_fieldnames = [
        "image",
        "face_index",
        "person_id",
        "detection_confidence",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "landmarks_count",
        "head_pose_pitch",
        "head_pose_yaw",
        "head_pose_roll",
        "quality_confidence",
        "quality_brightness",
        "quality_sharpness",
        "quality_face_area_ratio",
        "eye_left_openness",
        "eye_right_openness",
        "eye_mean_openness",
        "gaze_right_iris_x",
        "gaze_right_iris_y",
        "gaze_left_iris_x",
        "gaze_left_iris_y",
        "gaze_mean_iris_x",
        "gaze_mean_iris_y",
        "gaze_estimation_pitch",
        "gaze_estimation_yaw",
        "gaze_vector_x",
        "gaze_vector_y",
        "gaze_vector_z",
        "gaze_association_iou",
        "mouth_openness",
        "mouth_width",
        "mouth_height",
        "emotion_label",
        "emotion_confidence",
        *[
            "emotion_score_"
            + label.lower()
            for label in EMOTION_LABELS
        ],
        "regions_skin_pixel_count",
        "regions_skin_fraction",
        "regions_association_iou",
        *[
            "regions_pixels_"
            + class_name
            for class_name in region_class_names
        ],
        *[
            "module_"
            + module
            for module in STATIC_MODULES
        ],
    ]

    with frames_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=frame_fieldnames,
        )

        writer.writeheader()

        for record in all_records:
            writer.writerow(
                record_to_csv_row(
                    record,
                    region_class_names,
                )
            )

    combined_summary = {
        "test_type":
            "image_end_to_end_integration",
        "images":
            image_summaries,
        "image_count":
            len(image_summaries),
        "passed_images":
            sum(
                summary[
                    "overall_status"
                ]
                == "PASS"
                for summary in image_summaries
            ),
        "failed_images":
            sum(
                summary[
                    "overall_status"
                ]
                != "PASS"
                for summary in image_summaries
            ),
        "total_detected_faces":
            sum(
                summary[
                    "detected_faces"
                ]
                for summary in image_summaries
            ),
        "total_exported_face_records":
            len(
                all_records
            ),
        "evaluated_static_modules":
            STATIC_MODULES,
        "not_applicable_modules":
            {
                module:
                    (
                        "Requires temporal sequence input "
                        "and is not evaluated from a single image."
                    )
                for module
                in NOT_APPLICABLE_MODULES
            },
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with modules_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "image",
            "module",
            "status",
            "successful_face_samples",
            "total_face_samples",
            "coverage_percent",
            "reason",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in image_summaries:
            for module in (
                STATIC_MODULES
                + NOT_APPLICABLE_MODULES
            ):
                item = summary[
                    "modules"
                ][
                    module
                ]

                writer.writerow(
                    {
                        "image":
                            summary[
                                "image"
                            ],
                        "module":
                            module,
                        "status":
                            item[
                                "status"
                            ],
                        "successful_face_samples":
                            item.get(
                                "successful_face_samples"
                            ),
                        "total_face_samples":
                            item.get(
                                "total_face_samples"
                            ),
                        "coverage_percent":
                            item.get(
                                "coverage_percent"
                            ),
                        "reason":
                            item.get(
                                "reason"
                            ),
                    }
                )

    print()
    print("=" * 82)
    print(
        "PhysioTrack Image End-to-End "
        "Integration Test"
    )
    print("=" * 82)

    print(
        "Images tested:",
        len(image_summaries),
    )

    print(
        "Passed images:",
        combined_summary[
            "passed_images"
        ],
    )

    print(
        "Failed images:",
        combined_summary[
            "failed_images"
        ],
    )

    print(
        "Total detected faces:",
        combined_summary[
            "total_detected_faces"
        ],
    )

    print(
        "Exported face records:",
        combined_summary[
            "total_exported_face_records"
        ],
    )

    print(
        "Temporal modules:",
        "NOT_APPLICABLE",
    )

    print(
        "Overall status:",
        combined_summary[
            "overall_status"
        ],
    )

    print()
    print("Saved:")
    print(
        results_json_path
    )
    print(
        frames_csv_path
    )
    print(
        summary_json_path
    )
    print(
        modules_csv_path
    )

    if not overall_pass:
        raise RuntimeError(
            "Image end-to-end integration test completed with "
            "one or more failed images."
        )

    print()
    print(
        "Image end-to-end integration test: PASS"
    )


if __name__ == "__main__":
    main()