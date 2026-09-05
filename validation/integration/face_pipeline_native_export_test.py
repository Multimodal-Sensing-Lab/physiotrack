from __future__ import annotations

import json
import math
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


def values_finite(
    mapping: dict,
    keys: tuple[str, ...],
) -> bool:
    return all(
        finite_numeric(
            mapping.get(key)
        )
        for key in keys
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
    mouth_motion_numeric = 0
    mouth_motion_nonzero = 0
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

                    movement = mouth_motion.get(
                        "mouth_movement"
                    )

                    velocity = mouth_motion.get(
                        "mouth_velocity"
                    )

                    if (
                        finite_numeric(
                            movement
                        )
                        and finite_numeric(
                            velocity
                        )
                    ):
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

    frame_landmarks_valid = all(
        (
            record.get(
                "face_features",
                {},
            ).get(
                "landmarks",
                {},
            ).get(
                "available",
                False,
            )
            and record[
                "face_features"
            ][
                "landmarks"
            ].get(
                "count"
            )
            == 478
        )
        for record in frame_records
    )

    frame_quality_valid = all(
        (
            record.get(
                "face_features",
                {},
            ).get(
                "quality",
                {},
            ).get(
                "available",
                False,
            )
            and values_finite(
                record[
                    "face_features"
                ][
                    "quality"
                ],
                (
                    "confidence",
                    "brightness",
                    "sharpness",
                    "face_area_ratio",
                ),
            )
            and math.isclose(
                float(
                    record[
                        "face_features"
                    ][
                        "quality"
                    ][
                        "confidence"
                    ]
                ),
                float(
                    record.get(
                        "confidence"
                    )
                ),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        for record in frame_records
    )

    frame_head_pose_valid = all(
        values_finite(
            record.get(
                "orientation",
                {},
            ),
            (
                "pitch",
                "yaw",
                "roll",
            ),
        )
        for record in frame_records
    )

    frame_gaze_valid = all(
        (
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
            and values_finite(
                record[
                    "face_features"
                ][
                    "gaze"
                ],
                (
                    "right_iris_x",
                    "right_iris_y",
                    "left_iris_x",
                    "left_iris_y",
                    "mean_iris_x",
                    "mean_iris_y",
                ),
            )
            and math.isclose(
                float(
                    record[
                        "face_features"
                    ][
                        "gaze"
                    ][
                        "mean_iris_x"
                    ]
                ),
                (
                    float(
                        record[
                            "face_features"
                        ][
                            "gaze"
                        ][
                            "right_iris_x"
                        ]
                    )
                    + float(
                        record[
                            "face_features"
                        ][
                            "gaze"
                        ][
                            "left_iris_x"
                        ]
                    )
                )
                / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(
                    record[
                        "face_features"
                    ][
                        "gaze"
                    ][
                        "mean_iris_y"
                    ]
                ),
                (
                    float(
                        record[
                            "face_features"
                        ][
                            "gaze"
                        ][
                            "right_iris_y"
                        ]
                    )
                    + float(
                        record[
                            "face_features"
                        ][
                            "gaze"
                        ][
                            "left_iris_y"
                        ]
                    )
                )
                / 2.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        for record in frame_records
    )

    frame_gaze_estimation_valid = all(
        (
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
            and isinstance(
                record[
                    "face_features"
                ][
                    "gaze_estimation"
                ].get(
                    "gaze_vector"
                ),
                list,
            )
            and len(
                record[
                    "face_features"
                ][
                    "gaze_estimation"
                ][
                    "gaze_vector"
                ]
            )
            == 3
            and all(
                finite_numeric(value)
                for value in record[
                    "face_features"
                ][
                    "gaze_estimation"
                ][
                    "gaze_vector"
                ]
            )
            and math.isclose(
                math.sqrt(
                    sum(
                        float(value) ** 2
                        for value in record[
                            "face_features"
                        ][
                            "gaze_estimation"
                        ][
                            "gaze_vector"
                        ]
                    )
                ),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and values_finite(
                record[
                    "face_features"
                ][
                    "gaze_estimation"
                ],
                (
                    "pitch",
                    "yaw",
                    "association_iou",
                ),
            )
        )
        for record in frame_records
    )

    frame_emotion_valid = all(
        (
            record.get(
                "face_features",
                {},
            ).get(
                "emotion",
                {},
            ).get(
                "available",
                False,
            )
            and isinstance(
                record[
                    "face_features"
                ][
                    "emotion"
                ].get(
                    "scores"
                ),
                dict,
            )
            and math.isclose(
                sum(
                    float(value)
                    for value in record[
                        "face_features"
                    ][
                        "emotion"
                    ][
                        "scores"
                    ].values()
                ),
                1.0,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            and record[
                "face_features"
            ][
                "emotion"
            ].get(
                "emotion"
            )
            in record[
                "face_features"
            ][
                "emotion"
            ][
                "scores"
            ]
            and math.isclose(
                float(
                    record[
                        "face_features"
                    ][
                        "emotion"
                    ][
                        "confidence"
                    ]
                ),
                float(
                    record[
                        "face_features"
                    ][
                        "emotion"
                    ][
                        "scores"
                    ][
                        record[
                            "face_features"
                        ][
                            "emotion"
                        ][
                            "emotion"
                        ]
                    ]
                ),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        )
        for record in frame_records
    )

    frame_regions_valid = all(
        (
            record.get(
                "face_features",
                {},
            ).get(
                "regions",
                {},
            ).get(
                "available",
                False,
            )
            and isinstance(
                record[
                    "face_features"
                ][
                    "regions"
                ].get(
                    "pixel_counts"
                ),
                dict,
            )
            and all(
                finite_numeric(value)
                and float(value) >= 0.0
                for value in record[
                    "face_features"
                ][
                    "regions"
                ][
                    "pixel_counts"
                ].values()
            )
            and values_finite(
                record[
                    "face_features"
                ][
                    "regions"
                ],
                (
                    "skin_pixel_count",
                    "skin_fraction",
                    "association_iou",
                ),
            )
        )
        for record in frame_records
    )

    frame_temporal_numeric_valid = all(
        (
            record.get(
                "face_features",
                {},
            ).get(
                "temporal",
                {},
            ).get(
                "available",
                False,
            )
            and isinstance(
                record[
                    "face_features"
                ][
                    "temporal"
                ].get(
                    "summary"
                ),
                dict,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "temporal"
                ][
                    "summary"
                ].get(
                    "window_frames"
                )
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "temporal"
                ][
                    "summary"
                ].get(
                    "window_sec"
                )
            )
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

    frame_mouth_motion_valid = all(
        (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "mouth_motion"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "mouth_motion"
            ].get(
                "available",
                False,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_movement"
                )
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_velocity"
                )
            )
            and float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_movement"
                ]
            )
            >= 0.0
            and float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_velocity"
                ]
            )
            >= 0.0
        )
        for record in frame_records
    )

    frame_mouth_velocity_consistent = all(
        math.isclose(
            float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_velocity"
                ]
            ),
            float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_movement"
                ]
            )
            * fps,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for record in frame_records
        if (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "mouth_motion"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "mouth_motion"
            ].get(
                "available",
                False,
            )
        )
    )

    frame_mouth_motion_nonzero_observed = any(
        (
            float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_movement"
                ]
            )
            > 0.0
            or float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_velocity"
                ]
            )
            > 0.0
        )
        for record in frame_records
        if (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "mouth_motion"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "mouth_motion"
            ].get(
                "available",
                False,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_movement"
                )
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_velocity"
                )
            )
        )
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

    numerical_checks = {
        "frame_landmarks_valid":
            frame_landmarks_valid,
        "frame_quality_valid":
            frame_quality_valid,
        "frame_head_pose_valid":
            frame_head_pose_valid,
        "frame_gaze_valid":
            frame_gaze_valid,
        "frame_gaze_estimation_valid":
            frame_gaze_estimation_valid,
        "frame_emotion_valid":
            frame_emotion_valid,
        "frame_regions_valid":
            frame_regions_valid,
        "frame_temporal_numeric_valid":
            frame_temporal_numeric_valid,
        "frame_mouth_motion_valid":
            frame_mouth_motion_valid,
        "frame_mouth_velocity_consistent":
            frame_mouth_velocity_consistent,
        "frame_mouth_motion_nonzero_observed":
            frame_mouth_motion_nonzero_observed,
        "mouth_motion_numeric_matches_available":
            mouth_motion_numeric
            == mouth_motion_available,
    }

    overall_pass = (
        all(
            integration_checks.values()
        )
        and all(
            numerical_checks.values()
        )
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
            in {
                **integration_checks,
                **numerical_checks,
            }.items()
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



def validate_saved_exports(
    frame_records: list[dict],
    frame_json_path: Path,
    frame_csv_path: Path,
) -> dict[str, bool]:
    """Verify numerical values survive JSON and flattened CSV export."""
    with frame_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_json_records = json.load(
            file
        )

    frame_csv = pd.read_csv(
        frame_csv_path
    )

    movement_column = (
        "face_features.mouth_motion.mouth_movement"
    )

    velocity_column = (
        "face_features.mouth_motion.mouth_velocity"
    )

    available_column = (
        "face_features.mouth_motion.available"
    )

    json_record_count_matches = (
        len(
            saved_json_records
        )
        == len(
            frame_records
        )
    )

    csv_record_count_matches = (
        len(
            frame_csv
        )
        == len(
            frame_records
        )
    )

    csv_has_mouth_motion_columns = all(
        column in frame_csv.columns
        for column in (
            available_column,
            movement_column,
            velocity_column,
        )
    )

    required_numeric_columns = (
        "confidence",
        "orientation.pitch",
        "orientation.yaw",
        "orientation.roll",
        "face_features.landmarks.count",
        "face_features.quality.confidence",
        "face_features.quality.brightness",
        "face_features.quality.sharpness",
        "face_features.quality.face_area_ratio",
        "face_features.eyes.left_openness",
        "face_features.eyes.right_openness",
        "face_features.eyes.mean_openness",
        "face_features.gaze.mean_iris_x",
        "face_features.gaze.mean_iris_y",
        "face_features.gaze_estimation.pitch",
        "face_features.gaze_estimation.yaw",
        "face_features.gaze_estimation.association_iou",
        "face_features.mouth.mouth_openness",
        "face_features.mouth.mouth_width",
        "face_features.mouth.mouth_height",
        "face_features.emotion.confidence",
        "face_features.regions.skin_pixel_count",
        "face_features.regions.skin_fraction",
        "face_features.regions.association_iou",
        "face_features.temporal.summary.window_frames",
        "face_features.temporal.summary.window_sec",
    )

    csv_has_numeric_columns = all(
        column in frame_csv.columns
        for column in required_numeric_columns
    )

    csv_numeric_columns_valid = (
        csv_has_numeric_columns
        and all(
            pd.to_numeric(
                frame_csv[column],
                errors="coerce",
            ).notna().all()
            for column in required_numeric_columns
        )
    )

    json_numeric_contracts_valid = (
        len(
            saved_json_records
        )
        > 0
        and all(
            (
                values_finite(
                    record.get(
                        "orientation",
                        {},
                    ),
                    (
                        "pitch",
                        "yaw",
                        "roll",
                    ),
                )
                and record.get(
                    "face_features",
                    {},
                ).get(
                    "landmarks",
                    {},
                ).get(
                    "count"
                )
                == 478
            )
            for record in saved_json_records
        )
    )

    json_mouth_motion_numeric = (
        len(
            saved_json_records
        )
        > 0
        and all(
            (
                isinstance(
                    record.get(
                        "face_features",
                        {},
                    ).get(
                        "mouth_motion"
                    ),
                    dict,
                )
                and record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "available",
                    False,
                )
                and finite_numeric(
                    record[
                        "face_features"
                    ][
                        "mouth_motion"
                    ].get(
                        "mouth_movement"
                    )
                )
                and finite_numeric(
                    record[
                        "face_features"
                    ][
                        "mouth_motion"
                    ].get(
                        "mouth_velocity"
                    )
                )
            )
            for record in saved_json_records
        )
    )

    if csv_has_mouth_motion_columns:
        csv_movement = pd.to_numeric(
            frame_csv[
                movement_column
            ],
            errors="coerce",
        )

        csv_velocity = pd.to_numeric(
            frame_csv[
                velocity_column
            ],
            errors="coerce",
        )

        csv_available = frame_csv[
            available_column
        ].astype(
            str
        ).str.lower().isin(
            {
                "true",
                "1",
            }
        )

        csv_mouth_motion_numeric = (
            bool(
                csv_available.all()
            )
            and bool(
                csv_movement.notna().all()
            )
            and bool(
                csv_velocity.notna().all()
            )
            and bool(
                (
                    csv_movement
                    >= 0.0
                ).all()
            )
            and bool(
                (
                    csv_velocity
                    >= 0.0
                ).all()
            )
        )

        csv_mouth_motion_nonzero = bool(
            (
                (
                    csv_movement
                    > 0.0
                )
                | (
                    csv_velocity
                    > 0.0
                )
            ).any()
        )

    else:
        csv_mouth_motion_numeric = False
        csv_mouth_motion_nonzero = False

    json_mouth_motion_nonzero = any(
        (
            float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_movement"
                ]
            )
            > 0.0
            or float(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ][
                    "mouth_velocity"
                ]
            )
            > 0.0
        )
        for record in saved_json_records
        if (
            isinstance(
                record.get(
                    "face_features",
                    {},
                ).get(
                    "mouth_motion"
                ),
                dict,
            )
            and record[
                "face_features"
            ][
                "mouth_motion"
            ].get(
                "available",
                False,
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_movement"
                )
            )
            and finite_numeric(
                record[
                    "face_features"
                ][
                    "mouth_motion"
                ].get(
                    "mouth_velocity"
                )
            )
        )
    )

    return {
        "json_record_count_matches":
            json_record_count_matches,
        "csv_record_count_matches":
            csv_record_count_matches,
        "json_numeric_contracts_valid":
            json_numeric_contracts_valid,
        "csv_has_numeric_columns":
            csv_has_numeric_columns,
        "csv_numeric_columns_valid":
            csv_numeric_columns_valid,
        "csv_has_mouth_motion_columns":
            csv_has_mouth_motion_columns,
        "json_mouth_motion_numeric":
            json_mouth_motion_numeric,
        "csv_mouth_motion_numeric":
            csv_mouth_motion_numeric,
        "json_mouth_motion_nonzero_observed":
            json_mouth_motion_nonzero,
        "csv_mouth_motion_nonzero_observed":
            csv_mouth_motion_nonzero,
    }

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

    saved_export_checks = validate_saved_exports(
        frame_records,
        frame_json_path,
        frame_csv_path,
    )

    overall_pass = (
        all(
            summary[
                "overall_status"
            ]
            == "PASS"
            for summary in video_summaries
        )
        and all(
            saved_export_checks.values()
        )
    )

    combined_summary = {
        "test_type":
            "native_export_integration",
        "videos":
            video_summaries,
        "video_count":
            len(video_summaries),
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
        sum(
            summary[
                "overall_status"
            ]
            == "PASS"
            for summary in video_summaries
        ),
    )

    print(
        "Failed videos:",
        sum(
            summary[
                "overall_status"
            ]
            != "PASS"
            for summary in video_summaries
        ),
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
        "JSON mouth-motion numeric export:",
        saved_export_checks[
            "json_mouth_motion_numeric"
        ],
    )

    print(
        "CSV mouth-motion numeric export:",
        saved_export_checks[
            "csv_mouth_motion_numeric"
        ],
    )

    print(
        "CSV mouth-motion columns present:",
        saved_export_checks[
            "csv_has_mouth_motion_columns"
        ],
    )

    print(
        "Mouth-motion non-zero values exported:",
        (
            saved_export_checks[
                "json_mouth_motion_nonzero_observed"
            ]
            and saved_export_checks[
                "csv_mouth_motion_nonzero_observed"
            ]
        ),
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