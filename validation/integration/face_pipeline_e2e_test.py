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

VIDEO_PATH = TEST_DATA_DIR / "face_blink_pose.mp4"

VIDEO_LABEL = str(
    VIDEO_PATH.relative_to(SCRIPT_DIR)
).replace("\\", "/")

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
        return [to_jsonable(item) for item in value]

    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


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
    for name in ("track_id", "id", "person_id"):
        if hasattr(face, name):
            value = getattr(face, name)

            if value is not None:
                return value

    return None


def get_box(face: Any) -> Any:
    for name in ("box", "bbox", "bounding_box"):
        if hasattr(face, name):
            value = getattr(face, name)

            if value is not None:
                return value

    return None


def get_head_pose(
    face: Any,
    features: dict[str, Any],
) -> Any:
    for key in ("head_pose", "orientation", "pose"):
        if key in features:
            return features[key]

    for name in ("head_pose", "orientation", "pose"):
        if hasattr(face, name):
            value = getattr(face, name)

            if value is not None:
                return value

    return None


def run_pipeline(
    run_name: str,
    gaze_estimation_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
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

            prediction = pipeline.predict(frame)

            faces = get_faces(prediction)

            if faces:
                frames_with_faces += 1

            total_faces += len(faces)

            for face_index, face in enumerate(faces):
                features = getattr(
                    face,
                    "face_features",
                    {},
                )

                if features is None:
                    features = {}

                track_id = get_track_id(face)

                status = {
                    "detection": True,
                    "tracking": track_id is not None,
                }

                if status["detection"]:
                    success_counts["detection"] += 1

                if status["tracking"]:
                    success_counts["tracking"] += 1

                values = {
                    "landmarks":
                        features.get("landmarks"),
                    "quality":
                        features.get("quality"),
                    "head_pose":
                        get_head_pose(
                            face,
                            features,
                        ),
                    "eyes":
                        features.get("eyes"),
                    "blink":
                        features.get("blink"),
                    "gaze":
                        features.get("gaze"),
                    "gaze_estimation":
                        features.get(
                            "gaze_estimation"
                        ),
                    "mouth":
                        features.get("mouth"),
                    "mouth_motion":
                        features.get(
                            "mouth_motion"
                        ),
                    "emotion":
                        features.get("emotion"),
                    "regions":
                        features.get("regions"),
                    "temporal":
                        features.get("temporal"),
                }

                for module, value in values.items():
                    available = module_available(
                        value
                    )

                    status[module] = available

                    if available:
                        success_counts[
                            module
                        ] += 1

                records.append(
                    {
                        "run": run_name,
                        "gaze_estimation_enabled":
                            gaze_estimation_enabled,
                        "frame_index":
                            processed_frames,
                        "timestamp":
                            processed_frames / fps,
                        "face_index":
                            face_index,
                        "track_id":
                            to_jsonable(track_id),
                        "box":
                            to_jsonable(
                                get_box(face)
                            ),
                        "module_status":
                            status,
                        "face_features":
                            to_jsonable(features),
                        "head_pose":
                            to_jsonable(
                                values["head_pose"]
                            ),
                    }
                )

            processed_frames += 1

    finally:
        capture.release()
        pipeline.close()

    module_summary = {}

    for module in MODULES:
        count = success_counts[module]

        if (
            module == "gaze_estimation"
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
            "status": module_status,
            "successful_face_samples": count,
        }

    summary = {
        "run": run_name,
        "gaze_estimation_enabled":
            gaze_estimation_enabled,
        "video": VIDEO_LABEL,
        "fps": fps,
        "video_frames": frame_count,
        "processed_frames": processed_frames,
        "frames_with_faces": frames_with_faces,
        "total_faces": total_faces,
        "modules": module_summary,
    }

    return records, summary


def save_results(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
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
                "summaries":
                    summaries,
                "frames":
                    records,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with frame_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "run",
            "gaze_estimation_enabled",
            "frame_index",
            "timestamp",
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

            for module in MODULES:
                row[module] = (
                    record[
                        "module_status"
                    ].get(
                        module,
                        False,
                    )
                )

            writer.writerow(row)

    disabled = summaries[0]
    enabled = summaries[1]

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
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

        for module in MODULES:
            writer.writerow(
                {
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
    summaries: list[dict[str, Any]],
) -> None:
    disabled = summaries[0]
    enabled = summaries[1]

    print()
    print("=" * 82)

    print(
        "PhysioTrack Face Pipeline "
        "End-to-End Integration Test"
    )

    print("=" * 82)

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {disabled['fps']}")
    print(
        f"Frames: "
        f"{disabled['video_frames']}"
    )

    print()

    print(
        "Disabled run - processed frames:",
        disabled["processed_frames"],
    )

    print(
        "Disabled run - total faces:",
        disabled["total_faces"],
    )

    print(
        "Enabled run - processed frames:",
        enabled["processed_frames"],
    )

    print(
        "Enabled run - total faces:",
        enabled["total_faces"],
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

    print("=" * 82)



def validate_integration_summaries(
    disabled: dict[str, Any],
    enabled: dict[str, Any],
) -> None:
    for summary in (disabled, enabled):
        if summary["processed_frames"] <= 0:
            raise RuntimeError(
                f"{summary['run']}: no video frames were processed."
            )

        if (
            summary["video_frames"] > 0
            and summary["processed_frames"]
            != summary["video_frames"]
        ):
            raise RuntimeError(
                f"{summary['run']}: processed frame count does not "
                "match the video-reported frame count."
            )

        if summary["total_faces"] <= 0:
            raise RuntimeError(
                f"{summary['run']}: no faces were detected."
            )

    if (
        disabled["modules"]["gaze_estimation"]["status"]
        != "ABSENT"
    ):
        raise RuntimeError(
            "Gaze estimation produced output while disabled."
        )

    if (
        disabled["modules"]["gaze_estimation"][
            "successful_face_samples"
        ]
        != 0
    ):
        raise RuntimeError(
            "Gaze estimation produced successful samples while disabled."
        )

    if (
        enabled["modules"]["gaze_estimation"]["status"]
        != "PASS"
        or enabled["modules"]["gaze_estimation"][
            "successful_face_samples"
        ]
        <= 0
    ):
        raise RuntimeError(
            "Gaze estimation was enabled but no successful output "
            "was observed."
        )

    for module in MODULES:
        if module == "gaze_estimation":
            continue

        if disabled["modules"][module]["status"] != "PASS":
            raise RuntimeError(
                f"{module} was unavailable in the gaze-disabled run."
            )

        if enabled["modules"][module]["status"] != "PASS":
            raise RuntimeError(
                f"{module} was unavailable in the gaze-enabled run."
            )


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{VIDEO_PATH}"
        )

    clean_output_directory()

    print(
        "Running baseline pipeline "
        "with gaze estimation disabled..."
    )

    disabled_records, disabled_summary = (
        run_pipeline(
            run_name="gaze_disabled",
            gaze_estimation_enabled=False,
        )
    )

    print()
    print(
        "Running pipeline with "
        "gaze estimation enabled..."
    )

    enabled_records, enabled_summary = (
        run_pipeline(
            run_name="gaze_enabled",
            gaze_estimation_enabled=True,
        )
    )

    records = (
        disabled_records
        + enabled_records
    )

    summaries = [
        disabled_summary,
        enabled_summary,
    ]

    validate_integration_summaries(
        disabled_summary,
        enabled_summary,
    )

    save_results(
        records,
        summaries,
    )

    print_summary(
        summaries
    )

    print()
    print(
        "Face pipeline end-to-end integration test: PASS"
    )


if __name__ == "__main__":
    main()