from __future__ import annotations

import json
from pathlib import Path

import cv2

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "media_for_test"
    / "istockphoto-1370809321-640_adpp_is.mp4"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "whole_project_video_analysis_results"
)


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    total_video_frames = int(
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

    print("=" * 82)
    print(
        "PhysioTrack Whole-Project "
        "Video Analysis"
    )
    print("=" * 82)
    print(f"Video: {VIDEO_PATH}")
    print(
        f"Resolution: {width} x {height}"
    )
    print(f"FPS: {fps}")
    print(
        f"Video frames: {total_video_frames}"
    )
    print()
    print(
        "Running all enabled face-analysis "
        "components..."
    )

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    frame_records = []
    window_records = []

    processed_frames = 0
    detected_faces = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            timestamp = (
                processed_frames / fps
            )

            result = pipeline.predict(
                frame
            )

            current_frame_records = (
                FaceResultExporter.frame_records(
                    result,
                    frame_index=processed_frames,
                    timestamp=timestamp,
                )
            )

            current_window_records = (
                FaceResultExporter.window_records(
                    result,
                    frame_index=processed_frames,
                    timestamp=timestamp,
                )
            )

            frame_records.extend(
                current_frame_records
            )

            window_records.extend(
                current_window_records
            )

            detected_faces += len(
                current_frame_records
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_json_path = (
        OUTPUT_DIR
        / "whole_project_frames.json"
    )

    frame_csv_path = (
        OUTPUT_DIR
        / "whole_project_frames.csv"
    )

    window_json_path = (
        OUTPUT_DIR
        / "whole_project_windows.json"
    )

    window_csv_path = (
        OUTPUT_DIR
        / "whole_project_windows.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "whole_project_analysis_summary.json"
    )

    FaceResultExporter.save_json(
        frame_records,
        frame_json_path,
    )

    FaceResultExporter.save_csv(
        frame_records,
        frame_csv_path,
    )

    FaceResultExporter.save_json(
        window_records,
        window_json_path,
    )

    FaceResultExporter.save_csv(
        window_records,
        window_csv_path,
    )

    summary = {
        "test_type":
            "whole_project_video_analysis",
        "video":
            str(VIDEO_PATH),
        "resolution": {
            "width":
                width,
            "height":
                height,
        },
        "fps":
            fps,
        "video_frames":
            total_video_frames,
        "processed_frames":
            processed_frames,
        "detected_faces":
            detected_faces,
        "frame_records":
            len(frame_records),
        "window_records":
            len(window_records),
        "all_components_enabled": {
            "tracking": True,
            "head_pose": True,
            "landmarks": True,
            "quality": True,
            "eyes": True,
            "blink": True,
            "gaze": True,
            "gaze_estimation": True,
            "mouth": True,
            "mouth_motion": True,
            "emotion": True,
            "regions": True,
            "temporal": True,
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 82)
    print(
        "Whole-Project Video Analysis "
        "Completed"
    )
    print("=" * 82)

    print(
        "Processed frames:",
        processed_frames,
    )

    print(
        "Detected face records:",
        detected_faces,
    )

    print(
        "Frame records:",
        len(frame_records),
    )

    print(
        "Window records:",
        len(window_records),
    )

    print()

    print(
        "The saved frame records contain "
        "the numerical outputs produced by "
        "the enabled analysis components."
    )

    print()

    print("Saved:")
    print(frame_json_path)
    print(frame_csv_path)
    print(window_json_path)
    print(window_csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()