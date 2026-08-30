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
TEST_DATA_DIR = SCRIPT_DIR / "test_data"

VIDEO_PATH = (
    TEST_DATA_DIR
    / "istockphoto-1370809321-640_adpp_is.mp4"
)

VIDEO_LABEL = str(
    VIDEO_PATH.relative_to(SCRIPT_DIR)
).replace("\\", "/")

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


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    clean_output_directory()

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
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

    print("=" * 86)
    print(
        "PhysioTrack Final Whole-Project "
        "End-to-End Test"
    )
    print("=" * 86)

    print(f"Video: {VIDEO_PATH}")
    print(
        f"Resolution: {width} x {height}"
    )
    print(f"FPS: {fps}")
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

    records: list[dict[str, Any]] = []
    frame_face_counts: list[int] = []

    processed_frames = 0
    frames_with_faces = 0
    total_faces = 0

    track_ids: set[Any] = set()

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

            frame_face_counts.append(
                len(faces)
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

                if track_id is not None:
                    track_ids.add(
                        track_id
                    )

                status = {
                    "detection": True,
                    "tracking":
                        track_id is not None,
                }

                success_counts[
                    "detection"
                ] += 1

                if status["tracking"]:
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

                    status[module] = (
                        available
                    )

                    if available:
                        success_counts[
                            module
                        ] += 1

                records.append(
                    {
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
                                get_box(face)
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
                    "Processed frames:",
                    processed_frames,
                )

    finally:
        capture.release()
        pipeline.close()

    module_summary = {}

    for module in MODULES:
        successful = (
            success_counts[module]
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
            status = "NO_FACES"

        elif successful > 0:
            status = "PASS"

        else:
            status = "UNAVAILABLE"

        module_summary[module] = {
            "status":
                status,
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
        module_summary[module][
            "status"
        ]
        == "PASS"
        for module in MODULES
    )

    frame_count_matches = (
        reported_video_frames <= 0
        or processed_frames == reported_video_frames
    )

    tracking_observed = (
        len(track_ids) > 0
        and success_counts["tracking"] > 0
    )

    record_count_matches = (
        len(records) == total_faces
    )

    overall_pass = (
        processed_frames > 0
        and frame_count_matches
        and total_faces > 0
        and tracking_observed
        and record_count_matches
        and all_modules_observed
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
            VIDEO_LABEL,
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
                    summary,
                "frames":
                    records,
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
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with frames_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "face_index",
            "track_id",
            *MODULES,
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            row = {
                "frame_index":
                    record[
                        "frame_index"
                    ],
                "timestamp_seconds":
                    record[
                        "timestamp_seconds"
                    ],
                "face_index":
                    record[
                        "face_index"
                    ],
                "track_id":
                    record[
                        "track_id"
                    ],
            }

            for module in MODULES:
                row[module] = (
                    record[
                        "module_status"
                    ].get(
                        module,
                        False,
                    )
                )

            writer.writerow(
                row
            )

    with modules_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
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

        for module in MODULES:
            item = module_summary[
                module
            ]

            writer.writerow(
                {
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
        "Processed frames:",
        processed_frames,
    )

    print(
        "Frames with faces:",
        frames_with_faces,
    )

    print(
        "Frames without faces:",
        frames_without_faces,
    )

    print(
        "Frames with one face:",
        frames_with_one_face,
    )

    print(
        "Frames with multiple faces:",
        frames_with_multiple_faces,
    )

    print(
        "Total face samples:",
        total_faces,
    )

    print(
        "Unique track IDs:",
        normalized_track_ids,
    )

    print()

    print(
        f"{'Module':<22}"
        f"{'Status':<14}"
        f"{'Samples':<14}"
        f"{'Coverage':<14}"
    )

    print("-" * 64)

    for module in MODULES:
        item = module_summary[
            module
        ]

        print(
            f"{module:<22}"
            f"{item['status']:<14}"
            f"{item['successful_face_samples']:<14}"
            f"{item['coverage_percent']:.2f}%"
        )

    print("-" * 64)

    print(
        "All modules observed:",
        all_modules_observed,
    )

    print(
        "FINAL WHOLE-PROJECT E2E TEST:",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
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
        failed_checks = []

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

        raise RuntimeError(
            "Whole-project end-to-end test failed: "
            + ", ".join(failed_checks)
        )


if __name__ == "__main__":
    main()