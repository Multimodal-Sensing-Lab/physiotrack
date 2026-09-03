from __future__ import annotations

import csv
import json
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

    processing_error = None

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