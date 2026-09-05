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


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "whole_project"
)

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}

OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "whole_project_e2e"
)

MODULES = [
    "detection",
    "tracking",
    "landmarks",
    "quality",
    "head_pose",
    "eyes",
    "blink",
    "gaze",
    "gaze_estimation",
    "mouth",
    "mouth_motion",
    "emotion",
    "regions",
    "temporal",
]


def get_video_paths() -> list[Path]:
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test-data directory not found: {TEST_DATA_DIR}"
        )

    video_paths = sorted(
        path
        for path in TEST_DATA_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    )

    if not video_paths:
        raise FileNotFoundError(
            f"No supported video files found in: {TEST_DATA_DIR}"
        )

    return video_paths


def video_label(
    video_path: Path,
) -> str:
    return str(
        video_path.relative_to(SCRIPT_DIR)
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


def flatten_value(
    prefix: str,
    value: Any,
    output: dict[str, Any],
) -> None:
    if value is None:
        output[prefix] = None
        return

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        output[prefix] = value
        return

    if isinstance(value, np.generic):
        output[prefix] = value.item()
        return

    if isinstance(value, np.ndarray):
        flatten_value(
            prefix,
            value.tolist(),
            output,
        )
        return

    if isinstance(value, dict):
        if not value:
            output[prefix] = None
            return

        for key, item in value.items():
            child_prefix = (
                f"{prefix}_{key}"
                if prefix
                else str(key)
            )

            flatten_value(
                child_prefix,
                item,
                output,
            )

        return

    if isinstance(value, (list, tuple)):
        if not value:
            output[prefix] = None
            return

        for index, item in enumerate(value):
            flatten_value(
                f"{prefix}_{index}",
                item,
                output,
            )

        return

    if hasattr(value, "__dict__"):
        flatten_value(
            prefix,
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            },
            output,
        )

        return

    output[prefix] = str(value)


def make_frame_csv_row(
    record: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "video":
            record["video"],
        "frame_index":
            record["frame_index"],
        "timestamp_seconds":
            record["timestamp_seconds"],
        "face_index":
            record["face_index"],
        "track_id":
            record["track_id"],
    }

    box = record.get(
        "box"
    )

    if isinstance(
        box,
        (list, tuple),
    ):
        row["box_x1"] = (
            box[0]
            if len(box) > 0
            else None
        )

        row["box_y1"] = (
            box[1]
            if len(box) > 1
            else None
        )

        row["box_x2"] = (
            box[2]
            if len(box) > 2
            else None
        )

        row["box_y2"] = (
            box[3]
            if len(box) > 3
            else None
        )

    else:
        row["box_x1"] = None
        row["box_y1"] = None
        row["box_x2"] = None
        row["box_y2"] = None

    head_pose = record.get(
        "head_pose"
    )

    if head_pose is not None:
        flatten_value(
            "head_pose",
            head_pose,
            row,
        )

    features = record.get(
        "face_features",
        {},
    )

    if isinstance(
        features,
        dict,
    ):
        for module, value in features.items():
            flatten_value(
                module,
                value,
                row,
            )

    module_status = record.get(
        "module_status",
        {},
    )

    for module in MODULES:
        row[
            f"module_{module}_available"
        ] = module_status.get(
            module,
            False,
        )

    return row


def make_config() -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=True,
        head_pose=True,
        landmarks=True,
        quality=True,
        eyes=True,
        blink=True,
        gaze=True,
        gaze_estimation=True,
        mouth=True,
        mouth_motion=True,
        emotion=True,
        regions=True,
        temporal=True,
        gaze_estimation_mode="eth-xgaze",
        gaze_estimation_min_iou=0.10,
    )

    config.validate()

    return config


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


def finite_numeric(value: Any) -> bool:
    if value is None:
        return False

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def finite_values(
    values: list[Any],
) -> bool:
    return all(
        finite_numeric(value)
        for value in values
    )


def statistic_block_valid(
    value: Any,
) -> bool:
    if not isinstance(value, dict):
        return False

    required = (
        "mean",
        "std",
        "min",
        "max",
    )

    if not all(
        key in value
        for key in required
    ):
        return False

    if not finite_values(
        [
            value[key]
            for key in required
        ]
    ):
        return False

    mean = float(value["mean"])
    std = float(value["std"])
    minimum = float(value["min"])
    maximum = float(value["max"])

    return (
        std >= 0.0
        and minimum <= maximum
        and mean >= minimum - 1e-12
        and mean <= maximum + 1e-12
    )


def numerical_contract_valid(
    track_id: Any,
    box: Any,
    head_pose: Any,
    features: dict[str, Any],
    fps: float,
) -> bool:
    if not (
        isinstance(box, (list, tuple, np.ndarray))
        and len(box) == 4
        and finite_values(list(box))
        and float(box[2]) >= float(box[0])
        and float(box[3]) >= float(box[1])
    ):
        return False

    landmarks = features.get("landmarks")

    if not (
        isinstance(landmarks, dict)
        and landmarks.get("available", False)
        and landmarks.get("count") == 478
    ):
        return False

    quality = features.get("quality")

    if not (
        isinstance(quality, dict)
        and quality.get("available", False)
        and finite_values(
            [
                quality.get("confidence"),
                quality.get("brightness"),
                quality.get("sharpness"),
                quality.get("face_area_ratio"),
            ]
        )
        and 0.0 <= float(quality["confidence"]) <= 1.0
        and 0.0 <= float(quality["brightness"]) <= 1.0
        and float(quality["sharpness"]) >= 0.0
        and 0.0 <= float(quality["face_area_ratio"]) <= 1.0
    ):
        return False

    if not (
        isinstance(head_pose, dict)
        and finite_values(
            [
                head_pose.get("pitch"),
                head_pose.get("yaw"),
                head_pose.get("roll"),
            ]
        )
    ):
        return False

    eyes = features.get("eyes")

    if not (
        isinstance(eyes, dict)
        and eyes.get("available", False)
        and finite_values(
            [
                eyes.get("left_openness"),
                eyes.get("right_openness"),
                eyes.get("mean_openness"),
            ]
        )
        and float(eyes["left_openness"]) >= 0.0
        and float(eyes["right_openness"]) >= 0.0
        and float(eyes["mean_openness"]) >= 0.0
        and math.isclose(
            float(eyes["mean_openness"]),
            (
                float(eyes["left_openness"])
                + float(eyes["right_openness"])
            )
            / 2.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    blink = features.get("blink")

    if not (
        isinstance(blink, dict)
        and blink.get("available", False)
        and blink.get("eye_state") in {"open", "closed"}
        and isinstance(blink.get("blink"), bool)
        and finite_numeric(blink.get("blink_count"))
        and float(blink["blink_count"]) >= 0.0
        and (
            blink.get("blink_rate") is None
            or (
                finite_numeric(blink.get("blink_rate"))
                and float(blink["blink_rate"]) >= 0.0
            )
        )
        and (
            blink.get("blink_duration") is None
            or (
                finite_numeric(blink.get("blink_duration"))
                and float(blink["blink_duration"]) >= 0.0
            )
        )
    ):
        return False

    gaze = features.get("gaze")

    if not (
        isinstance(gaze, dict)
        and gaze.get("available", False)
        and finite_values(
            [
                gaze.get("right_iris_x"),
                gaze.get("right_iris_y"),
                gaze.get("left_iris_x"),
                gaze.get("left_iris_y"),
                gaze.get("mean_iris_x"),
                gaze.get("mean_iris_y"),
            ]
        )
        and math.isclose(
            float(gaze["mean_iris_x"]),
            (
                float(gaze["right_iris_x"])
                + float(gaze["left_iris_x"])
            )
            / 2.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(gaze["mean_iris_y"]),
            (
                float(gaze["right_iris_y"])
                + float(gaze["left_iris_y"])
            )
            / 2.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    gaze_estimation = features.get("gaze_estimation")

    if not isinstance(gaze_estimation, dict):
        return False

    gaze_vector = gaze_estimation.get("gaze_vector")

    if not (
        gaze_estimation.get("available", False)
        and isinstance(gaze_vector, (list, tuple, np.ndarray))
        and len(gaze_vector) == 3
        and finite_values(list(gaze_vector))
        and finite_values(
            [
                gaze_estimation.get("pitch"),
                gaze_estimation.get("yaw"),
                gaze_estimation.get("association_iou"),
            ]
        )
        and math.isclose(
            float(
                np.linalg.norm(
                    np.asarray(
                        gaze_vector,
                        dtype=float,
                    )
                )
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and 0.0
        <= float(gaze_estimation["association_iou"])
        <= 1.0
    ):
        return False

    mouth = features.get("mouth")

    if not (
        isinstance(mouth, dict)
        and mouth.get("available", False)
        and finite_values(
            [
                mouth.get("mouth_openness"),
                mouth.get("mouth_width"),
                mouth.get("mouth_height"),
            ]
        )
        and float(mouth["mouth_width"]) > 0.0
        and float(mouth["mouth_height"]) >= 0.0
        and float(mouth["mouth_openness"]) >= 0.0
        and math.isclose(
            float(mouth["mouth_openness"]),
            float(mouth["mouth_height"])
            / float(mouth["mouth_width"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    emotion = features.get("emotion")

    if not isinstance(emotion, dict):
        return False

    scores = emotion.get("scores")
    emotion_label = emotion.get("emotion")

    if not (
        emotion.get("available", False)
        and isinstance(scores, dict)
        and len(scores) > 0
        and all(
            finite_numeric(value)
            and float(value) >= 0.0
            for value in scores.values()
        )
        and finite_numeric(emotion.get("confidence"))
        and math.isclose(
            sum(float(value) for value in scores.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        and emotion_label in scores
        and emotion_label == max(scores, key=scores.get)
        and math.isclose(
            float(emotion["confidence"]),
            float(scores[emotion_label]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    regions = features.get("regions")

    if not isinstance(regions, dict):
        return False

    pixel_counts = regions.get("pixel_counts")

    if not (
        regions.get("available", False)
        and isinstance(pixel_counts, dict)
        and len(pixel_counts) > 0
        and all(
            finite_numeric(value)
            and float(value) >= 0.0
            for value in pixel_counts.values()
        )
        and finite_values(
            [
                regions.get("skin_pixel_count"),
                regions.get("skin_fraction"),
                regions.get("association_iou"),
            ]
        )
        and float(regions["skin_pixel_count"]) >= 0.0
        and 0.0 <= float(regions["skin_fraction"]) <= 1.0
        and 0.0 <= float(regions["association_iou"]) <= 1.0
        and (
            "skin" not in pixel_counts
            or math.isclose(
                float(regions["skin_pixel_count"]),
                float(pixel_counts["skin"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    ):
        return False

    temporal = features.get("temporal")

    if not isinstance(temporal, dict):
        return False

    temporal_summary = temporal.get("summary")

    if not (
        temporal.get("available", False)
        and isinstance(temporal_summary, dict)
        and temporal_summary.get("person_id") == track_id
        and finite_numeric(temporal_summary.get("window_frames"))
        and float(temporal_summary["window_frames"]) >= 1.0
        and finite_numeric(temporal_summary.get("window_sec"))
        and math.isclose(
            float(temporal_summary["window_sec"]),
            float(temporal_summary["window_frames"]) / fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    temporal_blocks = [
        temporal_summary.get("head_pose", {}).get("yaw"),
        temporal_summary.get("head_pose", {}).get("pitch"),
        temporal_summary.get("head_pose", {}).get("roll"),
        temporal_summary.get("eyes", {}).get("mean_openness"),
        temporal_summary.get("gaze", {}).get("mean_iris_x"),
        temporal_summary.get("gaze", {}).get("mean_iris_y"),
        temporal_summary.get("mouth", {}).get("openness"),
        temporal_summary.get("mouth", {}).get("movement"),
        temporal_summary.get("quality", {}).get("brightness"),
        temporal_summary.get("quality", {}).get("sharpness"),
        temporal_summary.get("quality", {}).get("face_area_ratio"),
    ]

    if not all(
        statistic_block_valid(block)
        for block in temporal_blocks
    ):
        return False

    blink_summary = temporal_summary.get("blink")
    emotion_summary = temporal_summary.get("emotion")

    return (
        isinstance(blink_summary, dict)
        and finite_numeric(blink_summary.get("events"))
        and float(blink_summary["events"]) >= 0.0
        and isinstance(emotion_summary, dict)
        and isinstance(emotion_summary.get("dominant"), str)
        and len(emotion_summary["dominant"]) > 0
    )


def get_faces(prediction: Any) -> list[Any]:
    if prediction is None:
        return []

    if hasattr(prediction, "faces"):
        faces = prediction.faces

        if faces is None:
            return []

        return list(faces)

    if isinstance(prediction, (list, tuple)):
        return list(prediction)

    try:
        return list(prediction)
    except TypeError:
        return []


def get_track_id(face: Any) -> Any:
    for name in (
        "track_id",
        "id",
        "person_id",
    ):
        if hasattr(face, name):
            value = getattr(
                face,
                name,
            )

            if value is not None:
                return value

    return None


def get_box(face: Any) -> Any:
    for name in (
        "box",
        "bbox",
        "bounding_box",
    ):
        if hasattr(face, name):
            value = getattr(
                face,
                name,
            )

            if value is not None:
                return value

    return None


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


def failed_video_summary(
    video_path: Path,
    reason: str,
) -> dict[str, Any]:
    module_summary = {}

    for module in MODULES:
        module_summary[module] = {
            "status":
                "ERROR",
            "successful_face_samples":
                0,
            "total_face_samples":
                0,
            "coverage_percent":
                0.0,
        }

    return {
        "test_type":
            "final_whole_project_end_to_end",
        "video":
            video_label(
                video_path
            ),
        "resolution":
            None,
        "fps":
            None,
        "reported_video_frames":
            None,
        "processed_frames":
            0,
        "frames_with_faces":
            0,
        "frames_without_faces":
            0,
        "frames_with_one_face":
            0,
        "frames_with_multiple_faces":
            0,
        "total_face_samples":
            0,
        "unique_track_ids":
            [],
        "modules":
            module_summary,
        "frame_count_matches_video":
            False,
        "tracking_observed":
            False,
        "record_count_matches":
            False,
        "all_modules_observed":
            False,
        "blink_configuration":
            None,
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
        "mouth_motion": {
            "available_samples":
                0,
            "finite_samples":
                0,
            "nonzero_samples":
                0,
            "matches_available_samples":
                False,
            "velocity_consistent_with_movement_and_fps":
                False,
            "person_ids":
                [],
            "initialization":
                {},
        },
        "failure_reason":
            reason,
        "overall_status":
            "FAIL",
    }


def run_video(
    video_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    reported_video_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    config = make_config()

    blink_configuration_valid = (
        config.blink_threshold == 0.22
        and config.min_closed_frames == 3
    )

    print("=" * 86)

    print(
        "PhysioTrack Final Whole-Project "
        "End-to-End Test"
    )

    print("=" * 86)

    print(
        f"Video: {video_path}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        "Reported video frames:",
        reported_video_frames,
    )

    print()

    print(
        "Initializing full FaceAnalysis pipeline..."
    )

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    success_counts = {
        module: 0
        for module in MODULES
    }

    records: list[
        dict[str, Any]
    ] = []

    frame_face_counts: list[
        int
    ] = []

    processed_frames = 0
    frames_with_faces = 0
    total_faces = 0

    track_ids: set[
        Any
    ] = set()

    valid_eye_openness_samples = 0
    valid_mouth_openness_samples = 0

    mouth_motion_available_samples = 0
    valid_mouth_motion_samples = 0
    nonzero_mouth_motion_samples = 0
    mouth_velocity_consistent = True
    mouth_motion_person_ids: set[Any] = set()
    mouth_motion_initialization: dict[
        Any,
        dict[str, Any],
    ] = {}

    processing_error = None
    numerical_contracts_valid = True

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            try:
                prediction = pipeline.predict(
                    frame
                )

            except Exception as exc:
                processing_error = (
                    "processing_error_at_frame_"
                    f"{processed_frames}: {exc}"
                )

                break

            faces = get_faces(
                prediction
            )

            frame_face_counts.append(
                len(faces)
            )

            if faces:
                frames_with_faces += 1

            total_faces += len(
                faces
            )

            for face_index, face in enumerate(
                faces
            ):
                features = getattr(
                    face,
                    "face_features",
                    {},
                )

                if features is None:
                    features = {}

                track_id = get_track_id(
                    face
                )

                if track_id is not None:
                    track_ids.add(
                        track_id
                    )

                status = {
                    "detection":
                        True,
                    "tracking":
                        track_id is not None,
                }

                success_counts[
                    "detection"
                ] += 1

                if status[
                    "tracking"
                ]:
                    success_counts[
                        "tracking"
                    ] += 1

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
                    "blink":
                        features.get(
                            "blink"
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
                    "mouth_motion":
                        features.get(
                            "mouth_motion"
                        ),
                    "emotion":
                        features.get(
                            "emotion"
                        ),
                    "regions":
                        features.get(
                            "regions"
                        ),
                    "temporal":
                        features.get(
                            "temporal"
                        ),
                }

                for module, value in (
                    values.items()
                ):
                    available = (
                        module_available(
                            value
                        )
                    )

                    status[
                        module
                    ] = available

                    if available:
                        success_counts[
                            module
                        ] += 1

                eyes_value = values[
                    "eyes"
                ]

                if (
                    isinstance(
                        eyes_value,
                        dict,
                    )
                    and eyes_value.get(
                        "available",
                        False,
                    )
                    and finite_numeric(
                        eyes_value.get(
                            "mean_openness"
                        )
                    )
                ):
                    valid_eye_openness_samples += 1

                mouth_value = values[
                    "mouth"
                ]

                if (
                    isinstance(
                        mouth_value,
                        dict,
                    )
                    and mouth_value.get(
                        "available",
                        False,
                    )
                    and finite_numeric(
                        mouth_value.get(
                            "mouth_openness"
                        )
                    )
                ):
                    valid_mouth_openness_samples += 1

                mouth_motion_value = values[
                    "mouth_motion"
                ]

                if (
                    isinstance(
                        mouth_motion_value,
                        dict,
                    )
                    and mouth_motion_value.get(
                        "available",
                        False,
                    )
                ):
                    mouth_motion_available_samples += 1

                    movement = mouth_motion_value.get(
                        "mouth_movement"
                    )

                    velocity = mouth_motion_value.get(
                        "mouth_velocity"
                    )

                    values_are_valid = (
                        finite_numeric(
                            movement
                        )
                        and finite_numeric(
                            velocity
                        )
                        and float(
                            movement
                        )
                        >= 0.0
                        and float(
                            velocity
                        )
                        >= 0.0
                    )

                    if values_are_valid:
                        valid_mouth_motion_samples += 1

                        if track_id is not None:
                            mouth_motion_person_ids.add(
                                track_id
                            )

                        if track_id not in mouth_motion_initialization:
                            mouth_motion_initialization[
                                track_id
                            ] = {
                                "frame_index":
                                    processed_frames,
                                "movement":
                                    float(
                                        movement
                                    ),
                                "velocity":
                                    float(
                                        velocity
                                    ),
                                "is_zero":
                                    (
                                        math.isclose(
                                            float(
                                                movement
                                            ),
                                            0.0,
                                            rel_tol=0.0,
                                            abs_tol=1e-12,
                                        )
                                        and math.isclose(
                                            float(
                                                velocity
                                            ),
                                            0.0,
                                            rel_tol=0.0,
                                            abs_tol=1e-12,
                                        )
                                    ),
                            }

                        if (
                            float(
                                movement
                            )
                            > 0.0
                            or float(
                                velocity
                            )
                            > 0.0
                        ):
                            nonzero_mouth_motion_samples += 1

                        mouth_velocity_consistent = (
                            mouth_velocity_consistent
                            and math.isclose(
                                float(
                                    velocity
                                ),
                                float(
                                    movement
                                )
                                * fps,
                                rel_tol=0.0,
                                abs_tol=1e-9,
                            )
                        )

                    else:
                        mouth_velocity_consistent = False

                numerical_contracts_valid = (
                    numerical_contracts_valid
                    and numerical_contract_valid(
                        track_id=track_id,
                        box=get_box(face),
                        head_pose=head_pose,
                        features=features,
                        fps=fps,
                    )
                )

                records.append(
                    {
                        "video":
                            video_label(
                                video_path
                            ),
                        "frame_index":
                            processed_frames,
                        "timestamp_seconds":
                            processed_frames
                            / fps,
                        "face_index":
                            face_index,
                        "track_id":
                            to_jsonable(
                                track_id
                            ),
                        "box":
                            to_jsonable(
                                get_box(
                                    face
                                )
                            ),
                        "module_status":
                            status,
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

            processed_frames += 1

            if (
                processed_frames % 100
                == 0
            ):
                print(
                    f"{video_path.name}: "
                    f"processed {processed_frames} frames"
                )

    finally:
        capture.release()
        pipeline.close()

    module_summary = {}

    for module in MODULES:
        successful = (
            success_counts[
                module
            ]
        )

        if total_faces > 0:
            coverage = (
                successful
                / total_faces
                * 100.0
            )

        else:
            coverage = 0.0

        if total_faces == 0:
            module_status = (
                "NO_FACES"
            )

        elif successful > 0:
            module_status = (
                "PASS"
            )

        else:
            module_status = (
                "UNAVAILABLE"
            )

        module_summary[
            module
        ] = {
            "status":
                module_status,
            "successful_face_samples":
                successful,
            "total_face_samples":
                total_faces,
            "coverage_percent":
                coverage,
        }

    frames_without_faces = (
        processed_frames
        - frames_with_faces
    )

    frames_with_one_face = sum(
        count == 1
        for count in frame_face_counts
    )

    frames_with_multiple_faces = sum(
        count > 1
        for count in frame_face_counts
    )

    all_modules_observed = all(
        module_summary[
            module
        ][
            "status"
        ]
        == "PASS"
        for module in MODULES
    )

    frame_count_matches = (
        processing_error is None
        and (
            reported_video_frames <= 0
            or processed_frames
            == reported_video_frames
        )
    )

    tracking_observed = (
        len(
            track_ids
        )
        > 0
        and success_counts[
            "tracking"
        ]
        > 0
    )

    record_count_matches = (
        len(
            records
        )
        == total_faces
    )

    eye_openness_values_valid = (
        valid_eye_openness_samples > 0
        and valid_eye_openness_samples
        == success_counts[
            "eyes"
        ]
    )

    mouth_openness_values_valid = (
        valid_mouth_openness_samples > 0
        and valid_mouth_openness_samples
        == success_counts[
            "mouth"
        ]
    )

    mouth_motion_values_valid = (
        mouth_motion_available_samples > 0
        and valid_mouth_motion_samples
        == mouth_motion_available_samples
        and mouth_motion_available_samples
        == success_counts[
            "mouth_motion"
        ]
    )

    mouth_motion_nonzero_observed = (
        nonzero_mouth_motion_samples > 0
    )

    mouth_motion_initialization_valid = (
        len(
            mouth_motion_initialization
        )
        > 0
        and all(
            item[
                "is_zero"
            ]
            for item in mouth_motion_initialization.values()
        )
    )

    mouth_motion_tracking_ids_valid = (
        len(
            mouth_motion_person_ids
        )
        > 0
        and all(
            person_id is not None
            for person_id in mouth_motion_person_ids
        )
    )

    failed_checks = []

    if processing_error is not None:
        failed_checks.append(
            processing_error
        )

    if processed_frames <= 0:
        failed_checks.append(
            "no_frames_processed"
        )

    if not frame_count_matches:
        failed_checks.append(
            "frame_count_mismatch"
        )

    if total_faces <= 0:
        failed_checks.append(
            "no_faces"
        )

    if not tracking_observed:
        failed_checks.append(
            "tracking_not_observed"
        )

    if not record_count_matches:
        failed_checks.append(
            "record_count_mismatch"
        )

    if not all_modules_observed:
        failed_checks.append(
            "one_or_more_modules_unobserved"
        )

    if not blink_configuration_valid:
        failed_checks.append(
            "blink_configuration_mismatch"
        )

    if not eye_openness_values_valid:
        failed_checks.append(
            "eye_openness_values_invalid"
        )

    if not mouth_openness_values_valid:
        failed_checks.append(
            "mouth_openness_values_invalid"
        )

    if not mouth_motion_values_valid:
        failed_checks.append(
            "mouth_motion_values_invalid"
        )

    if not mouth_motion_nonzero_observed:
        failed_checks.append(
            "mouth_motion_never_nonzero"
        )

    if not mouth_velocity_consistent:
        failed_checks.append(
            "mouth_velocity_inconsistent_with_movement_and_fps"
        )

    if not mouth_motion_initialization_valid:
        failed_checks.append(
            "mouth_motion_initialization_invalid"
        )

    if not mouth_motion_tracking_ids_valid:
        failed_checks.append(
            "mouth_motion_tracking_ids_invalid"
        )

    if not numerical_contracts_valid:
        failed_checks.append(
            "numerical_contracts_invalid"
        )

    overall_pass = (
        not failed_checks
    )

    normalized_track_ids = sorted(
        [
            str(
                to_jsonable(
                    value
                )
            )
            for value in track_ids
        ]
    )

    summary = {
        "test_type":
            "final_whole_project_end_to_end",
        "video":
            video_label(
                video_path
            ),
        "resolution": {
            "width":
                width,
            "height":
                height,
        },
        "fps":
            fps,
        "reported_video_frames":
            reported_video_frames,
        "processed_frames":
            processed_frames,
        "frames_with_faces":
            frames_with_faces,
        "frames_without_faces":
            frames_without_faces,
        "frames_with_one_face":
            frames_with_one_face,
        "frames_with_multiple_faces":
            frames_with_multiple_faces,
        "total_face_samples":
            total_faces,
        "unique_track_ids":
            normalized_track_ids,
        "modules":
            module_summary,
        "frame_count_matches_video":
            frame_count_matches,
        "tracking_observed":
            tracking_observed,
        "record_count_matches":
            record_count_matches,
        "all_modules_observed":
            all_modules_observed,
        "blink_configuration": {
            "threshold":
                config.blink_threshold,
            "min_closed_frames":
                config.min_closed_frames,
            "matches_validated_configuration":
                blink_configuration_valid,
        },
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
        "mouth_motion": {
            "available_samples":
                mouth_motion_available_samples,
            "finite_samples":
                valid_mouth_motion_samples,
            "nonzero_samples":
                nonzero_mouth_motion_samples,
            "matches_available_samples":
                mouth_motion_values_valid,
            "velocity_consistent_with_movement_and_fps":
                mouth_velocity_consistent,
            "person_ids":
                sorted(
                    [
                        str(
                            to_jsonable(
                                value
                            )
                        )
                        for value in mouth_motion_person_ids
                    ]
                ),
            "initialization":
                {
                    str(
                        to_jsonable(
                            key
                        )
                    ):
                        to_jsonable(
                            value
                        )
                    for key, value in mouth_motion_initialization.items()
                },
            "initialization_valid":
                mouth_motion_initialization_valid,
            "tracking_ids_valid":
                mouth_motion_tracking_ids_valid,
        },
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    if failed_checks:
        summary[
            "failure_reason"
        ] = ", ".join(
            failed_checks
        )

    return (
        records,
        summary,
    )


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    all_records: list[
        dict[str, Any]
    ] = []

    video_summaries: list[
        dict[str, Any]
    ] = []

    for video_path in video_paths:
        print()
        print("=" * 86)

        print(
            f"Testing video: {video_path}"
        )

        print("=" * 86)

        try:
            records, summary = (
                run_video(
                    video_path
                )
            )

        except Exception as exc:
            records = []

            summary = failed_video_summary(
                video_path,
                str(
                    exc
                ),
            )

        all_records.extend(
            records
        )

        video_summaries.append(
            summary
        )

        print()

        print(
            "Processed frames:",
            summary[
                "processed_frames"
            ],
        )

        print(
            "Frames with faces:",
            summary[
                "frames_with_faces"
            ],
        )

        print(
            "Frames without faces:",
            summary[
                "frames_without_faces"
            ],
        )

        print(
            "Frames with one face:",
            summary[
                "frames_with_one_face"
            ],
        )

        print(
            "Frames with multiple faces:",
            summary[
                "frames_with_multiple_faces"
            ],
        )

        print(
            "Total face samples:",
            summary[
                "total_face_samples"
            ],
        )

        print(
            "Unique track IDs:",
            summary[
                "unique_track_ids"
            ],
        )

        print(
            "Mouth-motion available samples:",
            summary[
                "mouth_motion"
            ][
                "available_samples"
            ],
        )

        print(
            "Mouth-motion finite samples:",
            summary[
                "mouth_motion"
            ][
                "finite_samples"
            ],
        )

        print(
            "Mouth-motion non-zero samples:",
            summary[
                "mouth_motion"
            ][
                "nonzero_samples"
            ],
        )

        print(
            "Mouth-velocity consistency:",
            summary[
                "mouth_motion"
            ][
                "velocity_consistent_with_movement_and_fps"
            ],
        )

        print(
            "Mouth-motion initialization valid:",
            summary[
                "mouth_motion"
            ].get(
                "initialization_valid",
                False,
            ),
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
        for summary in video_summaries
    )

    combined_summary = {
        "test_type":
            "final_whole_project_end_to_end",
        "videos":
            video_summaries,
        "video_count":
            len(
                video_summaries
            ),
        "passed_videos":
            sum(
                summary[
                    "overall_status"
                ]
                == "PASS"
                for summary in video_summaries
            ),
        "failed_videos":
            sum(
                summary[
                    "overall_status"
                ]
                != "PASS"
                for summary in video_summaries
            ),
        "total_processed_frames":
            sum(
                summary[
                    "processed_frames"
                ]
                for summary in video_summaries
            ),
        "total_face_samples":
            sum(
                summary[
                    "total_face_samples"
                ]
                for summary in video_summaries
            ),
        "mouth_motion": {
            "available_samples":
                sum(
                    summary[
                        "mouth_motion"
                    ][
                        "available_samples"
                    ]
                    for summary in video_summaries
                ),
            "finite_samples":
                sum(
                    summary[
                        "mouth_motion"
                    ][
                        "finite_samples"
                    ]
                    for summary in video_summaries
                ),
            "nonzero_samples":
                sum(
                    summary[
                        "mouth_motion"
                    ][
                        "nonzero_samples"
                    ]
                    for summary in video_summaries
                ),
            "all_videos_velocity_consistent":
                all(
                    summary[
                        "mouth_motion"
                    ][
                        "velocity_consistent_with_movement_and_fps"
                    ]
                    for summary in video_summaries
                    if summary[
                        "overall_status"
                    ]
                    == "PASS"
                ),
            "all_videos_initialization_valid":
                all(
                    summary[
                        "mouth_motion"
                    ].get(
                        "initialization_valid",
                        False,
                    )
                    for summary in video_summaries
                    if summary[
                        "overall_status"
                    ]
                    == "PASS"
                ),
        },
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    detailed_json_path = (
        OUTPUT_DIR
        / "whole_project_e2e_results.json"
    )

    summary_json_path = (
        OUTPUT_DIR
        / "whole_project_e2e_summary.json"
    )

    frames_csv_path = (
        OUTPUT_DIR
        / "whole_project_e2e_frames.csv"
    )

    modules_csv_path = (
        OUTPUT_DIR
        / "whole_project_e2e_modules.csv"
    )

    frame_rows = [
        make_frame_csv_row(
            record
        )
        for record in all_records
    ]

    metadata_fieldnames = [
        "video",
        "frame_index",
        "timestamp_seconds",
        "face_index",
        "track_id",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
    ]

    all_fieldnames = {
        key
        for row in frame_rows
        for key in row
    }

    feature_fieldnames = sorted(
        field
        for field in all_fieldnames
        if (
            field not in metadata_fieldnames
            and not field.startswith(
                "module_"
            )
        )
    )

    module_fieldnames = [
        f"module_{module}_available"
        for module in MODULES
    ]

    frame_fieldnames = (
        metadata_fieldnames
        + feature_fieldnames
        + module_fieldnames
    )

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

        for row in frame_rows:
            writer.writerow(
                row
            )

    movement_column = (
        "mouth_motion_mouth_movement"
    )

    velocity_column = (
        "mouth_motion_mouth_velocity"
    )

    mouth_available_column = (
        "mouth_motion_available"
    )

    required_mouth_motion_columns = {
        movement_column,
        velocity_column,
        mouth_available_column,
    }

    frame_csv_mouth_motion_columns_present = (
        required_mouth_motion_columns
        <= set(
            frame_fieldnames
        )
    )

    if frame_csv_mouth_motion_columns_present:
        exported_movement_values = [
            row.get(
                movement_column
            )
            for row in frame_rows
            if row.get(
                mouth_available_column
            )
        ]

        exported_velocity_values = [
            row.get(
                velocity_column
            )
            for row in frame_rows
            if row.get(
                mouth_available_column
            )
        ]

        frame_csv_mouth_motion_numeric = (
            len(
                exported_movement_values
            )
            > 0
            and len(
                exported_movement_values
            )
            == len(
                exported_velocity_values
            )
            and all(
                finite_numeric(
                    value
                )
                and float(
                    value
                )
                >= 0.0
                for value in exported_movement_values
            )
            and all(
                finite_numeric(
                    value
                )
                and float(
                    value
                )
                >= 0.0
                for value in exported_velocity_values
            )
        )

        frame_csv_mouth_motion_nonzero = any(
            (
                float(
                    movement
                )
                > 0.0
                or float(
                    velocity
                )
                > 0.0
            )
            for movement, velocity in zip(
                exported_movement_values,
                exported_velocity_values,
            )
        )

    else:
        frame_csv_mouth_motion_numeric = False
        frame_csv_mouth_motion_nonzero = False

    combined_summary[
        "mouth_motion"
    ][
        "frame_csv_columns_present"
    ] = frame_csv_mouth_motion_columns_present

    combined_summary[
        "mouth_motion"
    ][
        "frame_csv_numeric"
    ] = frame_csv_mouth_motion_numeric

    combined_summary[
        "mouth_motion"
    ][
        "frame_csv_nonzero_observed"
    ] = frame_csv_mouth_motion_nonzero

    if not (
        frame_csv_mouth_motion_columns_present
        and frame_csv_mouth_motion_numeric
        and frame_csv_mouth_motion_nonzero
    ):
        combined_summary[
            "overall_status"
        ] = "FAIL"

        overall_pass = False

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

    with detailed_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "summary":
                    combined_summary,
                "videos":
                    video_summaries,
                "frames":
                    all_records,
            },
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
            "video",
            "video_status",
            "failure_reason",
            "module",
            "status",
            "successful_face_samples",
            "total_face_samples",
            "coverage_percent",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for summary in video_summaries:
            for module in MODULES:
                item = summary[
                    "modules"
                ][
                    module
                ]

                writer.writerow(
                    {
                        "video":
                            summary[
                                "video"
                            ],
                        "video_status":
                            summary[
                                "overall_status"
                            ],
                        "failure_reason":
                            summary.get(
                                "failure_reason"
                            ),
                        "module":
                            module,
                        "status":
                            item[
                                "status"
                            ],
                        "successful_face_samples":
                            item[
                                "successful_face_samples"
                            ],
                        "total_face_samples":
                            item[
                                "total_face_samples"
                            ],
                        "coverage_percent":
                            item[
                                "coverage_percent"
                            ],
                    }
                )

    print()
    print("=" * 86)

    print(
        "Final Whole-Project Results"
    )

    print("=" * 86)

    print(
        "Videos tested:",
        len(
            video_summaries
        ),
    )

    print(
        "Passed videos:",
        combined_summary[
            "passed_videos"
        ],
    )

    print(
        "Failed videos:",
        combined_summary[
            "failed_videos"
        ],
    )

    print(
        "Total processed frames:",
        combined_summary[
            "total_processed_frames"
        ],
    )

    print(
        "Total face samples:",
        combined_summary[
            "total_face_samples"
        ],
    )

    print(
        "Mouth-motion available samples:",
        combined_summary[
            "mouth_motion"
        ][
            "available_samples"
        ],
    )

    print(
        "Mouth-motion finite samples:",
        combined_summary[
            "mouth_motion"
        ][
            "finite_samples"
        ],
    )

    print(
        "Mouth-motion non-zero samples:",
        combined_summary[
            "mouth_motion"
        ][
            "nonzero_samples"
        ],
    )

    print(
        "Mouth-motion CSV columns present:",
        combined_summary[
            "mouth_motion"
        ][
            "frame_csv_columns_present"
        ],
    )

    print(
        "Mouth-motion CSV numeric:",
        combined_summary[
            "mouth_motion"
        ][
            "frame_csv_numeric"
        ],
    )

    print(
        "Mouth-motion CSV non-zero observed:",
        combined_summary[
            "mouth_motion"
        ][
            "frame_csv_nonzero_observed"
        ],
    )

    print(
        "FINAL WHOLE-PROJECT E2E TEST:",
        combined_summary[
            "overall_status"
        ],
    )

    print()
    print("Saved:")
    print(
        detailed_json_path
    )
    print(
        summary_json_path
    )
    print(
        frames_csv_path
    )
    print(
        modules_csv_path
    )

    if not overall_pass:
        raise RuntimeError(
            "Whole-project end-to-end test "
            "completed with one or more failed videos."
        )


if __name__ == "__main__":
    main()