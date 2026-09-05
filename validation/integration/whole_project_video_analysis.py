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
    / "whole_project_video_analysis"
)



def finite_numeric(
    value,
) -> bool:
    """Return True for finite real numerical values."""
    if isinstance(
        value,
        bool,
    ):
        return False

    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return False

    return math.isfinite(
        float(value)
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


def failed_video_summary(
    video_path: Path,
    reason: str,
) -> dict:
    return {
        "test_type":
            "whole_project_video_analysis",
        "video":
            video_label(
                video_path
            ),
        "resolution":
            None,
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
        "mouth_motion_available":
            0,
        "mouth_motion_numeric":
            0,
        "mouth_motion_nonzero":
            0,
        "mouth_motion_velocity_consistent":
            False,
        "mouth_motion_person_ids":
            [],
        "frame_count_matches_video":
            False,
        "export_counts_match":
            False,
        "analysis_status":
            "FAIL",
        "failure_reason":
            reason,
        "all_components_enabled": {
            "tracking":
                True,
            "head_pose":
                True,
            "landmarks":
                True,
            "quality":
                True,
            "eyes":
                True,
            "blink":
                True,
            "gaze":
                True,
            "gaze_estimation":
                True,
            "mouth":
                True,
            "mouth_motion":
                True,
            "emotion":
                True,
            "regions":
                True,
            "temporal":
                True,
        },
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
        capture.get(
            cv2.CAP_PROP_FPS
        )
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

    config = make_config()

    print("=" * 82)
    print(
        "PhysioTrack Whole-Project "
        "Video Analysis"
    )
    print("=" * 82)

    print(f"Video: {video_path}")

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

    mouth_motion_available = 0
    mouth_motion_numeric = 0
    mouth_motion_nonzero = 0
    mouth_motion_velocity_consistent = True
    mouth_motion_person_ids = set()

    processing_error = None

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            timestamp = (
                processed_frames
                / fps
            )

            try:
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

            except Exception as exc:
                processing_error = (
                    "processing_error_at_frame_"
                    f"{processed_frames}: {exc}"
                )

                break

            for record in current_frame_records:
                record[
                    "video"
                ] = video_label(
                    video_path
                )

            for record in current_window_records:
                record[
                    "video"
                ] = video_label(
                    video_path
                )

            for record in current_frame_records:
                features = record.get(
                    "face_features",
                    {},
                )

                mouth_motion = features.get(
                    "mouth_motion",
                    {},
                )

                if mouth_motion.get(
                    "available",
                    False,
                ):
                    mouth_motion_available += 1

                    person_id = record.get(
                        "person_id"
                    )

                    mouth_motion_person_ids.add(
                        person_id
                    )

                    movement = mouth_motion.get(
                        "mouth_movement"
                    )

                    velocity = mouth_motion.get(
                        "mouth_velocity"
                    )

                    values_are_numeric = (
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

                    if values_are_numeric:
                        mouth_motion_numeric += 1

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
                            mouth_motion_nonzero += 1

                        mouth_motion_velocity_consistent = (
                            mouth_motion_velocity_consistent
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
                        mouth_motion_velocity_consistent = False

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
                    f"{video_path.name}: "
                    f"processed {processed_frames} frames"
                )

    finally:
        capture.release()
        pipeline.close()

    frame_count_matches = (
        processing_error is None
        and (
            total_video_frames <= 0
            or processed_frames
            == total_video_frames
        )
    )

    export_counts_match = (
        len(frame_records)
        == detected_faces
        and len(window_records)
        == detected_faces
    )

    failed_checks = []

    if processing_error is not None:
        failed_checks.append(
            processing_error
        )

    if processed_frames <= 0:
        failed_checks.append(
            "no_video_frames_processed"
        )

    if not frame_count_matches:
        failed_checks.append(
            "frame_count_mismatch"
        )

    if detected_faces <= 0:
        failed_checks.append(
            "no_face_records_produced"
        )

    if not export_counts_match:
        failed_checks.append(
            "export_record_count_mismatch"
        )

    if mouth_motion_available <= 0:
        failed_checks.append(
            "mouth_motion_not_observed"
        )

    if (
        mouth_motion_numeric
        != mouth_motion_available
    ):
        failed_checks.append(
            "mouth_motion_contains_non_numeric_values"
        )

    if mouth_motion_nonzero <= 0:
        failed_checks.append(
            "mouth_motion_never_nonzero"
        )

    if not mouth_motion_velocity_consistent:
        failed_checks.append(
            "mouth_velocity_inconsistent_with_movement_and_fps"
        )

    if not mouth_motion_person_ids:
        failed_checks.append(
            "mouth_motion_has_no_person_ids"
        )

    analysis_status = (
        "PASS"
        if not failed_checks
        else "FAIL"
    )

    summary = {
        "test_type":
            "whole_project_video_analysis",
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
        "mouth_motion_available":
            mouth_motion_available,
        "mouth_motion_numeric":
            mouth_motion_numeric,
        "mouth_motion_nonzero":
            mouth_motion_nonzero,
        "mouth_motion_velocity_consistent":
            mouth_motion_velocity_consistent,
        "mouth_motion_person_ids":
            sorted(
                mouth_motion_person_ids,
                key=lambda value: (
                    value is None,
                    str(value),
                ),
            ),
        "frame_count_matches_video":
            frame_count_matches,
        "export_counts_match":
            export_counts_match,
        "analysis_status":
            analysis_status,
        "all_components_enabled": {
            "tracking":
                True,
            "head_pose":
                True,
            "landmarks":
                True,
            "quality":
                True,
            "eyes":
                True,
            "blink":
                True,
            "gaze":
                True,
            "gaze_estimation":
                True,
            "mouth":
                True,
            "mouth_motion":
                True,
            "emotion":
                True,
            "regions":
                True,
            "temporal":
                True,
        },
    }

    if failed_checks:
        summary[
            "failure_reason"
        ] = ", ".join(
            failed_checks
        )

    return (
        frame_records,
        window_records,
        summary,
    )


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    all_frame_records = []
    all_window_records = []
    video_summaries = []

    for video_path in video_paths:
        print()
        print("=" * 82)
        print(f"Testing video: {video_path}")
        print("=" * 82)

        try:
            (
                frame_records,
                window_records,
                summary,
            ) = run_video(
                video_path
            )

        except Exception as exc:
            frame_records = []
            window_records = []

            summary = failed_video_summary(
                video_path,
                str(exc),
            )

        all_frame_records.extend(
            frame_records
        )

        all_window_records.extend(
            window_records
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
            "Detected face records:",
            summary[
                "detected_faces"
            ],
        )

        print(
            "Frame records:",
            summary[
                "frame_records"
            ],
        )

        print(
            "Window records:",
            summary[
                "window_records"
            ],
        )

        print(
            "Mouth-motion available:",
            summary[
                "mouth_motion_available"
            ],
        )

        print(
            "Mouth-motion numeric:",
            summary[
                "mouth_motion_numeric"
            ],
        )

        print(
            "Mouth-motion non-zero:",
            summary[
                "mouth_motion_nonzero"
            ],
        )

        print(
            "Mouth-velocity consistency:",
            summary[
                "mouth_motion_velocity_consistent"
            ],
        )

        print(
            "Mouth-motion person IDs:",
            summary[
                "mouth_motion_person_ids"
            ],
        )

        print(
            "Status:",
            summary[
                "analysis_status"
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
        all_frame_records,
        frame_json_path,
    )

    FaceResultExporter.save_csv(
        all_frame_records,
        frame_csv_path,
    )

    FaceResultExporter.save_json(
        all_window_records,
        window_json_path,
    )

    FaceResultExporter.save_csv(
        all_window_records,
        window_csv_path,
    )

    overall_pass = all(
        summary[
            "analysis_status"
        ]
        == "PASS"
        for summary in video_summaries
    )

    combined_summary = {
        "test_type":
            "whole_project_video_analysis",
        "videos":
            video_summaries,
        "video_count":
            len(video_summaries),
        "passed_videos":
            sum(
                summary[
                    "analysis_status"
                ]
                == "PASS"
                for summary in video_summaries
            ),
        "failed_videos":
            sum(
                summary[
                    "analysis_status"
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
            len(
                all_frame_records
            ),
        "total_window_records":
            len(
                all_window_records
            ),
        "mouth_motion_available":
            sum(
                summary[
                    "mouth_motion_available"
                ]
                for summary in video_summaries
            ),
        "mouth_motion_numeric":
            sum(
                summary[
                    "mouth_motion_numeric"
                ]
                for summary in video_summaries
            ),
        "mouth_motion_nonzero":
            sum(
                summary[
                    "mouth_motion_nonzero"
                ]
                for summary in video_summaries
            ),
        "mouth_motion_velocity_consistent":
            all(
                summary[
                    "mouth_motion_velocity_consistent"
                ]
                for summary in video_summaries
                if summary[
                    "analysis_status"
                ]
                == "PASS"
            ),
        "analysis_status":
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
    print("=" * 82)

    print(
        "Whole-Project Video Analysis "
        "Completed"
    )

    print("=" * 82)

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
        "Total detected face records:",
        combined_summary[
            "total_detected_faces"
        ],
    )

    print(
        "Total frame records:",
        len(
            all_frame_records
        ),
    )

    print(
        "Total window records:",
        len(
            all_window_records
        ),
    )

    print(
        "Total mouth-motion available records:",
        combined_summary[
            "mouth_motion_available"
        ],
    )

    print(
        "Total mouth-motion numeric records:",
        combined_summary[
            "mouth_motion_numeric"
        ],
    )

    print(
        "Total mouth-motion non-zero records:",
        combined_summary[
            "mouth_motion_nonzero"
        ],
    )

    print(
        "Mouth-velocity consistency:",
        combined_summary[
            "mouth_motion_velocity_consistent"
        ],
    )

    print(
        "Overall status:",
        combined_summary[
            "analysis_status"
        ],
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

    if not overall_pass:
        raise RuntimeError(
            "Whole-project video analysis "
            "completed with one or more failed videos."
        )


if __name__ == "__main__":
    main()