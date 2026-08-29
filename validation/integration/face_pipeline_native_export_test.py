from __future__ import annotations

import json
from pathlib import Path

import cv2

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


VIDEO_PATH = Path(
    r"C:\Users\xx901\Documents\PhysioTrack_Thesis\physiotrack"
    r"\media_for_test\face_blink_pose.mp4"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "native_export"
)


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    total_video_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
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

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    frame_records = []
    window_records = []

    processed_frames = 0
    detected_faces = 0
    gaze_available = 0
    gaze_estimation_available = 0
    blink_available = 0
    mouth_motion_available = 0
    temporal_available = 0
    blink_events = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            timestamp = processed_frames / fps

            result = pipeline.predict(frame)

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

            for instance in result:
                detected_faces += 1

                features = (
                    instance.face_features
                    if instance.face_features is not None
                    else {}
                )

                gaze = features.get("gaze", {})
                gaze_estimation = features.get(
                    "gaze_estimation",
                    {},
                )
                blink = features.get("blink", {})
                mouth_motion = features.get(
                    "mouth_motion",
                    {},
                )
                temporal = features.get(
                    "temporal",
                    {},
                )

                if gaze.get("available", False):
                    gaze_available += 1

                if gaze_estimation.get(
                    "available",
                    False,
                ):
                    gaze_estimation_available += 1

                if blink.get("available", False):
                    blink_available += 1

                if blink.get("blink", False):
                    blink_events += 1

                if mouth_motion.get(
                    "available",
                    False,
                ):
                    mouth_motion_available += 1

                if temporal.get(
                    "available",
                    False,
                ):
                    temporal_available += 1

            processed_frames += 1

    finally:
        capture.release()
        pipeline.close()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_json_path = (
        OUTPUT_DIR
        / "face_pipeline_frames.json"
    )

    frame_csv_path = (
        OUTPUT_DIR
        / "face_pipeline_frames.csv"
    )

    window_json_path = (
        OUTPUT_DIR
        / "face_pipeline_windows.json"
    )

    window_csv_path = (
        OUTPUT_DIR
        / "face_pipeline_windows.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "face_pipeline_native_export_summary.json"
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

    frame_has_old_gaze = all(
        record.get(
            "face_features",
            {},
        ).get(
            "gaze",
            {},
        ).get(
            "available",
            False,
        )
        for record in frame_records
    )

    frame_has_gaze_estimation = all(
        record.get(
            "face_features",
            {},
        ).get(
            "gaze_estimation",
            {},
        ).get(
            "available",
            False,
        )
        for record in frame_records
    )

    windows_have_expected_structure = all(
        all(
            key in record
            for key in (
                "head_pose",
                "eyes",
                "gaze",
                "mouth",
                "quality",
                "blink",
                "emotion",
            )
        )
        for record in window_records
    )

    summary = {
        "video": str(VIDEO_PATH),
        "fps": fps,
        "video_frames": total_video_frames,
        "processed_frames": processed_frames,
        "detected_faces": detected_faces,
        "frame_records": len(frame_records),
        "window_records": len(window_records),
        "gaze_available": gaze_available,
        "gaze_estimation_available":
            gaze_estimation_available,
        "blink_available": blink_available,
        "blink_events": blink_events,
        "mouth_motion_available":
            mouth_motion_available,
        "temporal_available":
            temporal_available,
        "frame_has_old_gaze":
            frame_has_old_gaze,
        "frame_has_gaze_estimation":
            frame_has_gaze_estimation,
        "windows_have_expected_structure":
            windows_have_expected_structure,
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
    print("=" * 72)
    print(
        "PhysioTrack Native Export "
        "Integration Test"
    )
    print("=" * 72)
    print(f"FPS: {fps}")
    print(
        f"Video frames: {total_video_frames}"
    )
    print(
        f"Processed frames: {processed_frames}"
    )
    print(
        f"Detected faces: {detected_faces}"
    )
    print(
        f"Frame records: {len(frame_records)}"
    )
    print(
        f"Window records: {len(window_records)}"
    )
    print(
        f"Old gaze available: {gaze_available}"
    )
    print(
        "Gaze estimation available:",
        gaze_estimation_available,
    )
    print(
        f"Blink available: {blink_available}"
    )
    print(
        f"Blink events: {blink_events}"
    )
    print(
        "Mouth motion available:",
        mouth_motion_available,
    )
    print(
        "Temporal available:",
        temporal_available,
    )
    print(
        "Frame records contain old gaze:",
        frame_has_old_gaze,
    )
    print(
        "Frame records contain gaze estimation:",
        frame_has_gaze_estimation,
    )
    print(
        "Window records have expected structure:",
        windows_have_expected_structure,
    )
    print("=" * 72)

    print()
    print("Saved:")
    print(frame_json_path)
    print(frame_csv_path)
    print(window_json_path)
    print(window_csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()