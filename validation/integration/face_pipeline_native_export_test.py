from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


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
    / "native_export"
)


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


def finite_numeric(value) -> bool:
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


def numeric_summary_valid(
    summary,
) -> bool:
    if not isinstance(
        summary,
        dict,
    ):
        return False

    return all(
        finite_numeric(
            summary.get(key)
        )
        for key in (
            "mean",
            "std",
            "min",
            "max",
        )
    )


def failed_video_summary(
    video_path: Path,
    reason: str,
) -> dict:
    return {
        "video":
            video_label(
                video_path
            ),
        "fps":
            None,
        "video_frames":
            None,
        "processed_frames":
            0,
        "detected_faces":
            0,
        "frame_records":
            0,
        "window_records":
            0,
        "gaze_available":
            0,
        "gaze_estimation_available":
            0,
        "blink_available":
            0,
        "blink_events":
            0,
        "mouth_motion_available":
            0,
        "temporal_available":
            0,
        "blink_configuration":
            None,
        "integration_checks":
            {},
        "failure_reason":
            reason,
        "overall_status":
            "FAIL",
    }


def run_video(
    video_path: Path,
) -> tuple[
    list[dict],
    list[dict],
    dict,
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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

            timestamp = (
                processed_frames
                / fps
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

            for record in current_frame_records:
                record["video"] = video_label(
                    video_path
                )

            for record in current_window_records:
                record["video"] = video_label(
                    video_path
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

                gaze = features.get(
                    "gaze",
                    {},
                )

                gaze_estimation = features.get(
                    "gaze_estimation",
                    {},
                )

                blink = features.get(
                    "blink",
                    {},
                )

                mouth_motion = features.get(
                    "mouth_motion",
                    {},
                )

                temporal = features.get(
                    "temporal",
                    {},
                )

                if gaze.get(
                    "available",
                    False,
                ):
                    gaze_available += 1

                if gaze_estimation.get(
                    "available",
                    False,
                ):
                    gaze_estimation_available += 1

                if blink.get(
                    "available",
                    False,
                ):
                    blink_available += 1

                if blink.get(
                    "blink",
                    False,
                ):
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

    frame_feature_dicts_valid = all(
        isinstance(
            record.get(
                "face_features"
            ),
            dict,
        )
        for record in frame_records
    )

    frame_has_old_gaze_key = all(
        "gaze"
        in record.get(
            "face_features",
            {},
        )
        for record in frame_records
    )

    frame_has_gaze_estimation_key = all(
        "gaze_estimation"
        in record.get(
            "face_features",
            {},
        )
        for record in frame_records
    )

    frame_eye_openness_valid = all(
        (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "eyes"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "eyes"
            ].get(
                "available",
                False,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "eyes"
                ].get(
                    "mean_openness"
                )
            )
        )
        for record in frame_records
    )

    frame_mouth_openness_valid = all(
        (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "mouth"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "mouth"
            ].get(
                "available",
                False,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth"
                ].get(
                    "mouth_openness"
                )
            )
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

    window_eye_openness_valid = all(
        (
            isinstance(
                record.get(
                    "eyes"
                ),
                dict,
            )
            and numeric_summary_valid(
                record[
                    "eyes"
                ].get(
                    "mean_openness"
                )
            )
        )
        for record in window_records
    )

    window_mouth_openness_valid = all(
        (
            isinstance(
                record.get(
                    "mouth"
                ),
                dict,
            )
            and numeric_summary_valid(
                record[
                    "mouth"
                ].get(
                    "openness"
                )
            )
        )
        for record in window_records
    )

    blink_configuration_valid = (
        config.blink_threshold == 0.22
        and config.min_closed_frames == 3
    )

    frame_count_matches = (
        total_video_frames <= 0
        or processed_frames == total_video_frames
    )

    export_record_counts_match = (
        len(frame_records) == detected_faces
        and len(window_records) == detected_faces
    )

    integration_checks = {
        "processed_frames_positive":
            processed_frames > 0,
        "frame_count_matches_video":
            frame_count_matches,
        "detected_faces_positive":
            detected_faces > 0,
        "frame_records_nonempty":
            len(frame_records) > 0,
        "window_records_nonempty":
            len(window_records) > 0,
        "export_record_counts_match":
            export_record_counts_match,
        "frame_feature_dicts_valid":
            frame_feature_dicts_valid,
        "frame_has_old_gaze_key":
            frame_has_old_gaze_key,
        "frame_has_gaze_estimation_key":
            frame_has_gaze_estimation_key,
        "frame_eye_openness_valid":
            frame_eye_openness_valid,
        "frame_mouth_openness_valid":
            frame_mouth_openness_valid,
        "old_gaze_observed":
            gaze_available > 0,
        "gaze_estimation_observed":
            gaze_estimation_available > 0,
        "windows_have_expected_structure":
            windows_have_expected_structure,
        "window_eye_openness_valid":
            window_eye_openness_valid,
        "window_mouth_openness_valid":
            window_mouth_openness_valid,
        "blink_configuration_valid":
            blink_configuration_valid,
    }

    overall_pass = all(
        integration_checks.values()
    )

    summary = {
        "video":
            video_label(
                video_path
            ),
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
        "gaze_available":
            gaze_available,
        "gaze_estimation_available":
            gaze_estimation_available,
        "blink_available":
            blink_available,
        "blink_events":
            blink_events,
        "mouth_motion_available":
            mouth_motion_available,
        "temporal_available":
            temporal_available,
        "blink_configuration": {
            "threshold":
                config.blink_threshold,
            "min_closed_frames":
                config.min_closed_frames,
        },
        "integration_checks":
            integration_checks,
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    if not overall_pass:
        failed_checks = [
            name
            for name, passed
            in integration_checks.items()
            if not passed
        ]

        raise RuntimeError(
            f"{summary['video']}: "
            "native export integration test failed: "
            + ", ".join(failed_checks)
        )

    return (
        frame_records,
        window_records,
        summary,
    )


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    frame_records = []
    window_records = []
    video_summaries = []

    for video_path in video_paths:
        print()
        print("=" * 72)
        print(f"Video: {video_path}")
        print("=" * 72)

        try:
            (
                current_frame_records,
                current_window_records,
                current_summary,
            ) = run_video(
                video_path
            )

        except Exception as exc:
            current_frame_records = []
            current_window_records = []

            current_summary = (
                failed_video_summary(
                    video_path,
                    str(exc),
                )
            )

            print(
                "Status: FAIL"
            )

            print(
                "Reason:",
                current_summary[
                    "failure_reason"
                ],
            )

        frame_records.extend(
            current_frame_records
        )

        window_records.extend(
            current_window_records
        )

        video_summaries.append(
            current_summary
        )

        print(
            "Processed frames:",
            current_summary[
                "processed_frames"
            ],
        )

        print(
            "Detected faces:",
            current_summary[
                "detected_faces"
            ],
        )

        print(
            "Frame records:",
            current_summary[
                "frame_records"
            ],
        )

        print(
            "Window records:",
            current_summary[
                "window_records"
            ],
        )

        print(
            "Blink events:",
            current_summary[
                "blink_events"
            ],
        )

        print(
            "Status:",
            current_summary[
                "overall_status"
            ],
        )

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

    overall_pass = all(
        summary[
            "overall_status"
        ]
        == "PASS"
        for summary in video_summaries
    )

    combined_summary = {
        "test_type":
            "native_export_integration",
        "videos":
            video_summaries,
        "video_count":
            len(video_summaries),
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
        "total_detected_faces":
            sum(
                summary[
                    "detected_faces"
                ]
                for summary in video_summaries
            ),
        "total_frame_records":
            len(frame_records),
        "total_window_records":
            len(window_records),
        "overall_status":
            (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_summary,
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

    print(
        "Videos tested:",
        len(video_summaries),
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
        "Total detected faces:",
        combined_summary[
            "total_detected_faces"
        ],
    )

    print(
        "Total frame records:",
        len(frame_records),
    )

    print(
        "Total window records:",
        len(window_records),
    )

    print(
        "Overall status:",
        combined_summary[
            "overall_status"
        ],
    )

    print("=" * 72)

    print()
    print("Saved:")
    print(frame_json_path)
    print(frame_csv_path)
    print(window_json_path)
    print(window_csv_path)
    print(summary_path)

    if not overall_pass:
        raise RuntimeError(
            "Native export integration test "
            "completed with one or more failed videos."
        )

    print()
    print(
        "Native export integration test: PASS"
    )


if __name__ == "__main__":
    main()