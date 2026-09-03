from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import pandas as pd

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "multi_person"
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
    / "multi_person_e2e"
)


FEATURE_NAMES = [
    "landmarks",
    "quality",
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


def failed_video_result(
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
        "reported_frame_count":
            None,
        "processed_frames":
            0,
        "total_faces":
            0,
        "person_ids":
            [],
        "person_frame_counts":
            {},
        "multi_face_frames":
            0,
        "duplicate_track_id_frames":
            [],
        "gaze_estimation_person_ids":
            [],
        "frame_records":
            [],
        "window_records":
            [],
        "summary_rows":
            [],
        "frame_face_counts":
            [],
        "gaze_estimation_failures":
            [],
        "failure_reason":
            reason,
        "status":
            "FAIL",
    }


def run_video(
    video_path: Path,
) -> dict:
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

    reported_frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
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

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    exporter = FaceResultExporter()

    frame_records = []
    window_records = []

    processed_frames = 0
    total_faces = 0

    person_frame_counts = {}

    feature_available_counts = {
        name: {}
        for name in FEATURE_NAMES
    }

    feature_missing_counts = {
        name: {}
        for name in FEATURE_NAMES
    }

    head_pose_available = {}
    head_pose_missing = {}

    gaze_estimation_failures = []
    frame_face_counts = []
    duplicate_track_id_frames = []
    multi_face_frames = 0
    gaze_estimation_person_ids = set()

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

            face_count = len(result)

            total_faces += face_count

            if face_count >= 2:
                multi_face_frames += 1

            current_frame_person_ids = [
                instance.id
                for instance in result
                if instance.id is not None
            ]

            if (
                len(current_frame_person_ids)
                != len(set(current_frame_person_ids))
            ):
                duplicate_track_id_frames.append(
                    {
                        "video":
                            video_label(
                                video_path
                            ),
                        "frame_index":
                            processed_frames,
                        "timestamp":
                            timestamp,
                        "person_ids":
                            current_frame_person_ids,
                    }
                )

            frame_face_counts.append(
                {
                    "video":
                        video_label(
                            video_path
                        ),
                    "frame_index":
                        processed_frames,
                    "timestamp":
                        timestamp,
                    "face_count":
                        face_count,
                }
            )

            current_frame_records = (
                exporter.frame_records(
                    result,
                    frame_index=processed_frames,
                    timestamp=timestamp,
                )
            )

            current_window_records = (
                exporter.window_records(
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
                person_id = instance.id

                person_frame_counts[
                    person_id
                ] = (
                    person_frame_counts.get(
                        person_id,
                        0,
                    )
                    + 1
                )

                features = (
                    instance.face_features
                    if instance.face_features is not None
                    else {}
                )

                orientation_available = (
                    instance.orientation is not None
                )

                if orientation_available:
                    head_pose_available[
                        person_id
                    ] = (
                        head_pose_available.get(
                            person_id,
                            0,
                        )
                        + 1
                    )

                else:
                    head_pose_missing[
                        person_id
                    ] = (
                        head_pose_missing.get(
                            person_id,
                            0,
                        )
                        + 1
                    )

                for feature_name in FEATURE_NAMES:
                    feature = features.get(
                        feature_name,
                        {},
                    )

                    is_available = bool(
                        feature.get(
                            "available",
                            False,
                        )
                    )

                    if is_available:
                        feature_available_counts[
                            feature_name
                        ][
                            person_id
                        ] = (
                            feature_available_counts[
                                feature_name
                            ].get(
                                person_id,
                                0,
                            )
                            + 1
                        )

                    else:
                        feature_missing_counts[
                            feature_name
                        ][
                            person_id
                        ] = (
                            feature_missing_counts[
                                feature_name
                            ].get(
                                person_id,
                                0,
                            )
                            + 1
                        )

                    if (
                        feature_name
                        == "gaze_estimation"
                        and is_available
                    ):
                        gaze_estimation_person_ids.add(
                            person_id
                        )

                    if (
                        feature_name
                        == "gaze_estimation"
                        and not is_available
                    ):
                        gaze_estimation_failures.append(
                            {
                                "video":
                                    video_label(
                                        video_path
                                    ),
                                "frame_index":
                                    processed_frames,
                                "timestamp":
                                    timestamp,
                                "person_id":
                                    person_id,
                                "box":
                                    instance.box,
                                "confidence":
                                    instance.confidence,
                                "association_iou":
                                    feature.get(
                                        "association_iou"
                                    ),
                            }
                        )

            processed_frames += 1

            if (
                processed_frames % 25
                == 0
            ):
                print(
                    f"{video_path.name}: "
                    f"processed {processed_frames} frames"
                )

    finally:
        capture.release()
        pipeline.close()

    person_ids = sorted(
        person_frame_counts
    )

    summary_rows = []

    for person_id in person_ids:
        person_total = (
            person_frame_counts[
                person_id
            ]
        )

        summary_rows.append(
            {
                "video":
                    video_label(
                        video_path
                    ),
                "person_id":
                    person_id,
                "module":
                    "head_pose",
                "available":
                    head_pose_available.get(
                        person_id,
                        0,
                    ),
                "missing":
                    head_pose_missing.get(
                        person_id,
                        0,
                    ),
                "total_person_frames":
                    person_total,
                "availability_percent":
                    (
                        100.0
                        * head_pose_available.get(
                            person_id,
                            0,
                        )
                        / person_total
                        if person_total
                        else 0.0
                    ),
            }
        )

        for feature_name in FEATURE_NAMES:
            available = (
                feature_available_counts[
                    feature_name
                ].get(
                    person_id,
                    0,
                )
            )

            missing = (
                feature_missing_counts[
                    feature_name
                ].get(
                    person_id,
                    0,
                )
            )

            summary_rows.append(
                {
                    "video":
                        video_label(
                            video_path
                        ),
                    "person_id":
                        person_id,
                    "module":
                        feature_name,
                    "available":
                        available,
                    "missing":
                        missing,
                    "total_person_frames":
                        person_total,
                    "availability_percent":
                        (
                            100.0
                            * available
                            / person_total
                            if person_total
                            else 0.0
                        ),
                }
            )

    failed_checks = []

    if processed_frames <= 0:
        failed_checks.append(
            "no_video_frames_processed"
        )

    if (
        reported_frame_count > 0
        and processed_frames
        != reported_frame_count
    ):
        failed_checks.append(
            "frame_count_mismatch"
        )

    if len(person_ids) < 2:
        failed_checks.append(
            "fewer_than_two_tracked_person_ids"
        )

    if multi_face_frames <= 0:
        failed_checks.append(
            "no_multi_face_frames"
        )

    if duplicate_track_id_frames:
        failed_checks.append(
            "duplicate_track_ids_within_frame"
        )

    if len(
        gaze_estimation_person_ids
    ) < 2:
        failed_checks.append(
            "gaze_estimation_not_observed_for_two_persons"
        )

    if total_faces <= 0:
        failed_checks.append(
            "no_face_instances"
        )

    if (
        len(frame_records)
        != total_faces
    ):
        failed_checks.append(
            "frame_export_count_mismatch"
        )

    if (
        len(window_records)
        != total_faces
    ):
        failed_checks.append(
            "window_export_count_mismatch"
        )

    if (
        sum(
            person_frame_counts.values()
        )
        != total_faces
    ):
        failed_checks.append(
            "person_frame_accounting_mismatch"
        )

    if (
        sum(
            head_pose_available.values()
        )
        <= 0
    ):
        failed_checks.append(
            "head_pose_never_available"
        )

    for feature_name in FEATURE_NAMES:
        observed = sum(
            feature_available_counts[
                feature_name
            ].values()
        )

        if observed <= 0:
            failed_checks.append(
                f"{feature_name}_never_available"
            )

    status = (
        "PASS"
        if not failed_checks
        else "FAIL"
    )

    result = {
        "video":
            video_label(
                video_path
            ),
        "fps":
            fps,
        "reported_frame_count":
            reported_frame_count,
        "processed_frames":
            processed_frames,
        "total_faces":
            total_faces,
        "person_ids":
            person_ids,
        "person_frame_counts":
            person_frame_counts,
        "multi_face_frames":
            multi_face_frames,
        "duplicate_track_id_frames":
            duplicate_track_id_frames,
        "gaze_estimation_person_ids":
            sorted(
                gaze_estimation_person_ids
            ),
        "frame_records":
            frame_records,
        "window_records":
            window_records,
        "summary_rows":
            summary_rows,
        "frame_face_counts":
            frame_face_counts,
        "gaze_estimation_failures":
            gaze_estimation_failures,
        "status":
            status,
    }

    if failed_checks:
        result[
            "failure_reason"
        ] = ", ".join(
            failed_checks
        )

    return result


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    all_frame_records = []
    all_window_records = []
    all_summary_rows = []
    all_frame_face_counts = []
    all_gaze_estimation_failures = []
    video_summaries = []

    for video_path in video_paths:
        print()
        print("=" * 78)
        print(f"Testing video: {video_path}")
        print("=" * 78)

        try:
            result = run_video(
                video_path
            )

        except Exception as exc:
            result = failed_video_result(
                video_path,
                str(exc),
            )

            print(
                "Status: FAIL"
            )

            print(
                "Reason:",
                result[
                    "failure_reason"
                ],
            )

        all_frame_records.extend(
            result[
                "frame_records"
            ]
        )

        all_window_records.extend(
            result[
                "window_records"
            ]
        )

        all_summary_rows.extend(
            result[
                "summary_rows"
            ]
        )

        all_frame_face_counts.extend(
            result[
                "frame_face_counts"
            ]
        )

        all_gaze_estimation_failures.extend(
            result[
                "gaze_estimation_failures"
            ]
        )

        video_summary = {
            "video":
                result[
                    "video"
                ],
            "fps":
                result[
                    "fps"
                ],
            "reported_frame_count":
                result[
                    "reported_frame_count"
                ],
            "processed_frames":
                result[
                    "processed_frames"
                ],
            "total_faces":
                result[
                    "total_faces"
                ],
            "person_ids":
                result[
                    "person_ids"
                ],
            "person_frame_counts":
                result[
                    "person_frame_counts"
                ],
            "multi_face_frames":
                result[
                    "multi_face_frames"
                ],
            "gaze_estimation_person_ids":
                result[
                    "gaze_estimation_person_ids"
                ],
            "status":
                result[
                    "status"
                ],
        }

        if result.get(
            "failure_reason"
        ):
            video_summary[
                "failure_reason"
            ] = result[
                "failure_reason"
            ]

        video_summaries.append(
            video_summary
        )

        print(
            "Processed frames:",
            result[
                "processed_frames"
            ],
        )

        print(
            "Total face instances:",
            result[
                "total_faces"
            ],
        )

        print(
            "Person IDs:",
            result[
                "person_ids"
            ],
        )

        print(
            "Frames with multiple faces:",
            result[
                "multi_face_frames"
            ],
        )

        print(
            "Status:",
            result[
                "status"
            ],
        )

        if result.get(
            "failure_reason"
        ):
            print(
                "Reason:",
                result[
                    "failure_reason"
                ],
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_json_path = (
        OUTPUT_DIR
        / "multi_person_full_frames.json"
    )

    frame_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_frames.csv"
    )

    window_json_path = (
        OUTPUT_DIR
        / "multi_person_full_windows.json"
    )

    window_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_windows.csv"
    )

    summary_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_summary.csv"
    )

    face_counts_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_frame_face_counts.csv"
    )

    gaze_failures_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_gaze_estimation_failures.csv"
    )

    video_summary_csv_path = (
        OUTPUT_DIR
        / "multi_person_full_video_summary.csv"
    )

    video_summary_json_path = (
        OUTPUT_DIR
        / "multi_person_full_video_summary.json"
    )

    exporter = FaceResultExporter()

    exporter.save_json(
        all_frame_records,
        frame_json_path,
    )

    exporter.save_csv(
        all_frame_records,
        frame_csv_path,
    )

    exporter.save_json(
        all_window_records,
        window_json_path,
    )

    exporter.save_csv(
        all_window_records,
        window_csv_path,
    )

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    face_counts_df = pd.DataFrame(
        all_frame_face_counts
    )

    face_counts_df.to_csv(
        face_counts_csv_path,
        index=False,
    )

    gaze_failures_df = pd.DataFrame(
        all_gaze_estimation_failures
    )

    gaze_failures_df.to_csv(
        gaze_failures_csv_path,
        index=False,
    )

    video_summary_rows = []

    for summary in video_summaries:
        video_summary_rows.append(
            {
                "video":
                    summary[
                        "video"
                    ],
                "fps":
                    summary[
                        "fps"
                    ],
                "reported_frame_count":
                    summary[
                        "reported_frame_count"
                    ],
                "processed_frames":
                    summary[
                        "processed_frames"
                    ],
                "total_faces":
                    summary[
                        "total_faces"
                    ],
                "person_ids":
                    json.dumps(
                        summary[
                            "person_ids"
                        ]
                    ),
                "person_frame_counts":
                    json.dumps(
                        summary[
                            "person_frame_counts"
                        ]
                    ),
                "multi_face_frames":
                    summary[
                        "multi_face_frames"
                    ],
                "gaze_estimation_person_ids":
                    json.dumps(
                        summary[
                            "gaze_estimation_person_ids"
                        ]
                    ),
                "status":
                    summary[
                        "status"
                    ],
                "failure_reason":
                    summary.get(
                        "failure_reason"
                    ),
            }
        )

    video_summary_df = pd.DataFrame(
        video_summary_rows
    )

    video_summary_df.to_csv(
        video_summary_csv_path,
        index=False,
    )

    overall_pass = all(
        summary[
            "status"
        ]
        == "PASS"
        for summary in video_summaries
    )

    video_summary_payload = {
        "test_type":
            "multi_person_end_to_end_integration",
        "videos":
            video_summaries,
        "video_count":
            len(
                video_summaries
            ),
        "passed_videos":
            sum(
                summary[
                    "status"
                ]
                == "PASS"
                for summary in video_summaries
            ),
        "failed_videos":
            sum(
                summary[
                    "status"
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
        "total_face_instances":
            sum(
                summary[
                    "total_faces"
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

    with video_summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            video_summary_payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "=== Full Multi-Person End-to-End Summary ==="
    )

    print(
        "Videos tested:",
        len(video_summaries),
    )

    print(
        "Passed videos:",
        video_summary_payload[
            "passed_videos"
        ],
    )

    print(
        "Failed videos:",
        video_summary_payload[
            "failed_videos"
        ],
    )

    for summary in video_summaries:
        print()
        print(
            "Video:",
            summary[
                "video"
            ],
        )

        print(
            "FPS:",
            summary[
                "fps"
            ],
        )

        print(
            "Reported frame count:",
            summary[
                "reported_frame_count"
            ],
        )

        print(
            "Processed frames:",
            summary[
                "processed_frames"
            ],
        )

        print(
            "Total face instances:",
            summary[
                "total_faces"
            ],
        )

        print(
            "Person IDs:",
            summary[
                "person_ids"
            ],
        )

        print(
            "Frames with multiple faces:",
            summary[
                "multi_face_frames"
            ],
        )

        print(
            "Person IDs with gaze estimation:",
            summary[
                "gaze_estimation_person_ids"
            ],
        )

        print(
            "Status:",
            summary[
                "status"
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

    print()
    print(
        "=== Module Availability Summary ==="
    )

    if len(
        summary_df
    ) == 0:
        print(
            "No module availability rows were produced."
        )

    else:
        print(
            summary_df.to_string(
                index=False
            )
        )

    print()
    print(
        "=== Gaze Estimation Failures ==="
    )

    if len(
        gaze_failures_df
    ) == 0:
        print(
            "None"
        )

    else:
        print(
            gaze_failures_df.to_string(
                index=False
            )
        )

    print()
    print(
        "=== Output Files ==="
    )

    for path in [
        frame_json_path,
        frame_csv_path,
        window_json_path,
        window_csv_path,
        summary_csv_path,
        face_counts_csv_path,
        gaze_failures_csv_path,
        video_summary_csv_path,
        video_summary_json_path,
    ]:
        print(
            path
        )

    print()
    print(
        "Full multi-person end-to-end "
        "integration and export test:",
        video_summary_payload[
            "overall_status"
        ],
    )

    if not overall_pass:
        raise RuntimeError(
            "Full multi-person end-to-end "
            "integration and export test completed "
            "with one or more failed videos."
        )


if __name__ == "__main__":
    main()