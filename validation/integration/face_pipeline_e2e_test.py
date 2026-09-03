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
    / "single_person"
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
    / "face_pipeline_e2e"
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

    if isinstance(value, (str, int, float, bool)):
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
        "run":
            record["run"],
        "gaze_estimation_enabled":
            record[
                "gaze_estimation_enabled"
            ],
        "frame_index":
            record["frame_index"],
        "timestamp":
            record["timestamp"],
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


def make_config(
    gaze_estimation_enabled: bool,
) -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=True,
        head_pose=True,
        landmarks=True,
        quality=True,
        eyes=True,
        blink=True,
        gaze=True,
        gaze_estimation=gaze_estimation_enabled,
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
            return bool(value["available"])

        return len(value) > 0

    if isinstance(value, np.ndarray):
        return value.size > 0

    if isinstance(value, (list, tuple)):
        return len(value) > 0

    return True


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
            value = getattr(face, name)

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
            value = getattr(face, name)

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
            value = getattr(face, name)

            if value is not None:
                return value

    return None


def run_pipeline(
    video_path: Path,
    run_name: str,
    gaze_estimation_enabled: bool,
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

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            f"Invalid video FPS: {fps}"
        )

    config = make_config(
        gaze_estimation_enabled
    )

    print()
    print(f"Video: {video_path}")
    print(f"Starting run: {run_name}")
    print(f"FPS: {fps}")
    print(f"Video frames: {frame_count}")

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    records: list[dict[str, Any]] = []

    success_counts = {
        module: 0
        for module in MODULES
    }

    processed_frames = 0
    frames_with_faces = 0
    total_faces = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            prediction = pipeline.predict(
                frame
            )

            faces = get_faces(
                prediction
            )

            if faces:
                frames_with_faces += 1

            total_faces += len(faces)

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

                status = {
                    "detection":
                        True,
                    "tracking":
                        track_id is not None,
                }

                if status["detection"]:
                    success_counts[
                        "detection"
                    ] += 1

                if status["tracking"]:
                    success_counts[
                        "tracking"
                    ] += 1

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
                        get_head_pose(
                            face,
                            features,
                        ),
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

                for module, value in values.items():
                    available = (
                        module_available(
                            value
                        )
                    )

                    status[module] = (
                        available
                    )

                    if available:
                        success_counts[
                            module
                        ] += 1

                records.append(
                    {
                        "video":
                            video_label(
                                video_path
                            ),
                        "run":
                            run_name,
                        "gaze_estimation_enabled":
                            gaze_estimation_enabled,
                        "frame_index":
                            processed_frames,
                        "timestamp":
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
                                get_box(face)
                            ),
                        "module_status":
                            status,
                        "face_features":
                            to_jsonable(
                                features
                            ),
                        "head_pose":
                            to_jsonable(
                                values[
                                    "head_pose"
                                ]
                            ),
                    }
                )

            processed_frames += 1

    finally:
        capture.release()
        pipeline.close()

    module_summary = {}

    for module in MODULES:
        count = success_counts[
            module
        ]

        if (
            module
            == "gaze_estimation"
            and not gaze_estimation_enabled
        ):
            module_status = "ABSENT"

        elif total_faces == 0:
            module_status = "NO_FACES"

        elif count > 0:
            module_status = "PASS"

        else:
            module_status = "UNAVAILABLE"

        module_summary[module] = {
            "status":
                module_status,
            "successful_face_samples":
                count,
        }

    summary = {
        "video":
            video_label(
                video_path
            ),
        "run":
            run_name,
        "gaze_estimation_enabled":
            gaze_estimation_enabled,
        "fps":
            fps,
        "video_frames":
            frame_count,
        "processed_frames":
            processed_frames,
        "frames_with_faces":
            frames_with_faces,
        "total_faces":
            total_faces,
        "modules":
            module_summary,
    }

    return records, summary


def failed_run_summary(
    video_path: Path,
    run_name: str,
    gaze_estimation_enabled: bool,
    reason: str,
) -> dict[str, Any]:
    module_summary = {}

    for module in MODULES:
        module_summary[module] = {
            "status":
                "ERROR",
            "successful_face_samples":
                0,
        }

    return {
        "video":
            video_label(
                video_path
            ),
        "run":
            run_name,
        "gaze_estimation_enabled":
            gaze_estimation_enabled,
        "fps":
            None,
        "video_frames":
            None,
        "processed_frames":
            0,
        "frames_with_faces":
            0,
        "total_faces":
            0,
        "modules":
            module_summary,
        "failure_reason":
            reason,
    }


def save_results(
    records: list[dict[str, Any]],
    video_summaries: list[dict[str, Any]],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_DIR
        / "face_pipeline_e2e_results.json"
    )

    frame_csv_path = (
        OUTPUT_DIR
        / "face_pipeline_e2e_frames.csv"
    )

    summary_csv_path = (
        OUTPUT_DIR
        / "face_pipeline_e2e_summary.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "test_type":
                    "end_to_end_integration",
                "video_summaries":
                    video_summaries,
                "frames":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    frame_rows = [
        make_frame_csv_row(
            record
        )
        for record in records
    ]

    metadata_fieldnames = [
        "video",
        "run",
        "gaze_estimation_enabled",
        "frame_index",
        "timestamp",
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

    with frame_csv_path.open(
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

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "video",
            "status",
            "failure_reason",
            "module",
            "disabled_status",
            "enabled_status",
            "disabled_successful_face_samples",
            "enabled_successful_face_samples",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for video_summary in video_summaries:
            disabled = video_summary[
                "disabled"
            ]

            enabled = video_summary[
                "enabled"
            ]

            for module in MODULES:
                writer.writerow(
                    {
                        "video":
                            video_summary[
                                "video"
                            ],
                        "status":
                            video_summary[
                                "status"
                            ],
                        "failure_reason":
                            video_summary.get(
                                "failure_reason"
                            ),
                        "module":
                            module,
                        "disabled_status":
                            disabled[
                                "modules"
                            ][
                                module
                            ][
                                "status"
                            ],
                        "enabled_status":
                            enabled[
                                "modules"
                            ][
                                module
                            ][
                                "status"
                            ],
                        "disabled_successful_face_samples":
                            disabled[
                                "modules"
                            ][
                                module
                            ][
                                "successful_face_samples"
                            ],
                        "enabled_successful_face_samples":
                            enabled[
                                "modules"
                            ][
                                module
                            ][
                                "successful_face_samples"
                            ],
                    }
                )

    print()
    print("Saved:")
    print(json_path)
    print(frame_csv_path)
    print(summary_csv_path)


def print_summary(
    video_summaries: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 82)

    print(
        "PhysioTrack Face Pipeline "
        "End-to-End Integration Test"
    )

    print("=" * 82)

    for video_summary in video_summaries:
        disabled = video_summary[
            "disabled"
        ]

        enabled = video_summary[
            "enabled"
        ]

        print()
        print(
            "Video:",
            video_summary[
                "video"
            ],
        )

        print(
            "Status:",
            video_summary[
                "status"
            ],
        )

        if video_summary.get(
            "failure_reason"
        ):
            print(
                "Reason:",
                video_summary[
                    "failure_reason"
                ],
            )

        print(
            "FPS:",
            disabled.get(
                "fps"
            ),
        )

        print(
            "Frames:",
            disabled.get(
                "video_frames"
            ),
        )

        print()

        print(
            "Disabled run - processed frames:",
            disabled[
                "processed_frames"
            ],
        )

        print(
            "Disabled run - total faces:",
            disabled[
                "total_faces"
            ],
        )

        print(
            "Enabled run - processed frames:",
            enabled[
                "processed_frames"
            ],
        )

        print(
            "Enabled run - total faces:",
            enabled[
                "total_faces"
            ],
        )

        print()

        print(
            f"{'Module':<22}"
            f"{'Disabled':<16}"
            f"{'Enabled':<16}"
        )

        print("-" * 54)

        for module in MODULES:
            disabled_status = (
                disabled[
                    "modules"
                ][
                    module
                ][
                    "status"
                ]
            )

            enabled_status = (
                enabled[
                    "modules"
                ][
                    module
                ][
                    "status"
                ]
            )

            print(
                f"{module:<22}"
                f"{disabled_status:<16}"
                f"{enabled_status:<16}"
            )

        print("-" * 54)

    print("=" * 82)


def validate_integration_summaries(
    disabled: dict[str, Any],
    enabled: dict[str, Any],
) -> None:
    for summary in (
        disabled,
        enabled,
    ):
        if summary[
            "processed_frames"
        ] <= 0:
            raise RuntimeError(
                f"{summary['video']} | "
                f"{summary['run']}: "
                "no video frames were processed."
            )

        if (
            summary["video_frames"] > 0
            and summary[
                "processed_frames"
            ]
            != summary[
                "video_frames"
            ]
        ):
            raise RuntimeError(
                f"{summary['video']} | "
                f"{summary['run']}: "
                "processed frame count does not "
                "match the video-reported frame count."
            )

        if summary[
            "total_faces"
        ] <= 0:
            raise RuntimeError(
                f"{summary['video']} | "
                f"{summary['run']}: "
                "no faces were detected."
            )

    if (
        disabled[
            "modules"
        ][
            "gaze_estimation"
        ][
            "status"
        ]
        != "ABSENT"
    ):
        raise RuntimeError(
            f"{disabled['video']}: "
            "gaze estimation produced output "
            "while disabled."
        )

    if (
        disabled[
            "modules"
        ][
            "gaze_estimation"
        ][
            "successful_face_samples"
        ]
        != 0
    ):
        raise RuntimeError(
            f"{disabled['video']}: "
            "gaze estimation produced "
            "successful samples while disabled."
        )

    if (
        enabled[
            "modules"
        ][
            "gaze_estimation"
        ][
            "status"
        ]
        != "PASS"
        or enabled[
            "modules"
        ][
            "gaze_estimation"
        ][
            "successful_face_samples"
        ]
        <= 0
    ):
        raise RuntimeError(
            f"{enabled['video']}: "
            "gaze estimation was enabled "
            "but no successful output was observed."
        )

    for module in MODULES:
        if module == "gaze_estimation":
            continue

        if (
            disabled[
                "modules"
            ][
                module
            ][
                "status"
            ]
            != "PASS"
        ):
            raise RuntimeError(
                f"{disabled['video']}: "
                f"{module} was unavailable "
                "in the gaze-disabled run."
            )

        if (
            enabled[
                "modules"
            ][
                module
            ][
                "status"
            ]
            != "PASS"
        ):
            raise RuntimeError(
                f"{enabled['video']}: "
                f"{module} was unavailable "
                "in the gaze-enabled run."
            )


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    records: list[
        dict[str, Any]
    ] = []

    video_summaries: list[
        dict[str, Any]
    ] = []

    for video_path in video_paths:
        print()
        print("=" * 82)
        print(f"Testing video: {video_path}")
        print("=" * 82)

        print()
        print(
            "Running baseline pipeline "
            "with gaze estimation disabled..."
        )

        disabled_records = []
        disabled_error = None

        try:
            (
                disabled_records,
                disabled_summary,
            ) = run_pipeline(
                video_path=video_path,
                run_name="gaze_disabled",
                gaze_estimation_enabled=False,
            )

        except Exception as exc:
            disabled_error = str(
                exc
            )

            disabled_summary = (
                failed_run_summary(
                    video_path=video_path,
                    run_name="gaze_disabled",
                    gaze_estimation_enabled=False,
                    reason=disabled_error,
                )
            )

            print(
                "Gaze-disabled run: FAIL"
            )

            print(
                "Reason:",
                disabled_error,
            )

        print()
        print(
            "Running pipeline with "
            "gaze estimation enabled..."
        )

        enabled_records = []
        enabled_error = None

        try:
            (
                enabled_records,
                enabled_summary,
            ) = run_pipeline(
                video_path=video_path,
                run_name="gaze_enabled",
                gaze_estimation_enabled=True,
            )

        except Exception as exc:
            enabled_error = str(
                exc
            )

            enabled_summary = (
                failed_run_summary(
                    video_path=video_path,
                    run_name="gaze_enabled",
                    gaze_estimation_enabled=True,
                    reason=enabled_error,
                )
            )

            print(
                "Gaze-enabled run: FAIL"
            )

            print(
                "Reason:",
                enabled_error,
            )

        validation_error = None

        if (
            disabled_error is None
            and enabled_error is None
        ):
            try:
                validate_integration_summaries(
                    disabled_summary,
                    enabled_summary,
                )

            except Exception as exc:
                validation_error = str(
                    exc
                )

                print(
                    "Integration validation: FAIL"
                )

                print(
                    "Reason:",
                    validation_error,
                )

        failure_reasons = [
            reason
            for reason in (
                (
                    "gaze_disabled: "
                    + disabled_error
                    if disabled_error
                    else None
                ),
                (
                    "gaze_enabled: "
                    + enabled_error
                    if enabled_error
                    else None
                ),
                (
                    "validation: "
                    + validation_error
                    if validation_error
                    else None
                ),
            )
            if reason is not None
        ]

        video_pass = (
            not failure_reasons
        )

        records.extend(
            disabled_records
        )

        records.extend(
            enabled_records
        )

        video_summary = {
            "video":
                video_label(
                    video_path
                ),
            "disabled":
                disabled_summary,
            "enabled":
                enabled_summary,
            "status":
                (
                    "PASS"
                    if video_pass
                    else "FAIL"
                ),
        }

        if failure_reasons:
            video_summary[
                "failure_reason"
            ] = "; ".join(
                failure_reasons
            )

        video_summaries.append(
            video_summary
        )

        print()
        print(
            "Video status:",
            video_summary[
                "status"
            ],
        )

    save_results(
        records,
        video_summaries,
    )

    print_summary(
        video_summaries
    )

    overall_pass = all(
        video_summary[
            "status"
        ]
        == "PASS"
        for video_summary in video_summaries
    )

    print()
    print(
        "Face pipeline end-to-end "
        "integration test:",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    )

    print(
        "Videos tested:",
        len(video_summaries),
    )

    print(
        "Passed videos:",
        sum(
            video_summary[
                "status"
            ]
            == "PASS"
            for video_summary in video_summaries
        ),
    )

    print(
        "Failed videos:",
        sum(
            video_summary[
                "status"
            ]
            != "PASS"
            for video_summary in video_summaries
        ),
    )

    if not overall_pass:
        raise RuntimeError(
            "Face pipeline end-to-end integration "
            "test completed with one or more "
            "failed videos."
        )


if __name__ == "__main__":
    main()