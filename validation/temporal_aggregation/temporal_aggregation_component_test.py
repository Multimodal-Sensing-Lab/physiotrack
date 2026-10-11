from collections import defaultdict, deque
from pathlib import Path
import csv
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"

INTEGRATION_DATA_ROOT = (
    REPO_ROOT
    / "validation"
    / "integration"
    / "test_data"
)

VIDEO_DIRS = (
    INTEGRATION_DATA_ROOT
    / "single_person",
    INTEGRATION_DATA_ROOT
    / "multi_person",
)

RESULTS_ROOT = SCRIPT_DIR / "results"
RESULTS_DIR = RESULTS_ROOT / "component_execution"

FRAME_RESULTS_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_component_results.csv"
)
CHECKS_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_component_checks.csv"
)
METRICS_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_component_metrics.csv"
)
SUMMARY_PATH = (
    RESULTS_DIR
    / "temporal_aggregation_component_summary.json"
)

LEGACY_OUTPUT_PATHS = (
    RESULTS_ROOT
    / "temporal_aggregation_component_results.csv",
    RESULTS_ROOT
    / "temporal_aggregation_component_checks.csv",
    RESULTS_ROOT
    / "temporal_aggregation_component_metrics.csv",
    RESULTS_ROOT
    / "temporal_aggregation_component_summary.json",
)

EXPECTED_ANALYSIS_SOURCE = (
    SRC_ROOT
    / "physiotrack"
    / "face"
    / "analysis.py"
)
EXPECTED_TEMPORAL_SOURCE = (
    SRC_ROOT
    / "physiotrack"
    / "face"
    / "temporal.py"
)

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}

NUMERIC_TOLERANCE = 1e-10

NUMERIC_FEATURES = (
    (
        "head_pose_yaw",
        ("head_pose", "yaw"),
    ),
    (
        "head_pose_pitch",
        ("head_pose", "pitch"),
    ),
    (
        "head_pose_roll",
        ("head_pose", "roll"),
    ),
    (
        "eyes_mean_openness",
        ("eyes", "mean_openness"),
    ),
    (
        "gaze_mean_iris_x",
        ("gaze", "mean_iris_x"),
    ),
    (
        "gaze_mean_iris_y",
        ("gaze", "mean_iris_y"),
    ),
    (
        "mouth_openness",
        ("mouth", "openness"),
    ),
    (
        "mouth_movement",
        ("mouth", "movement"),
    ),
    (
        "quality_brightness",
        ("quality", "brightness"),
    ),
    (
        "quality_sharpness",
        ("quality", "sharpness"),
    ),
    (
        "quality_face_area_ratio",
        ("quality", "face_area_ratio"),
    ),
)

STATISTICS = (
    "mean",
    "std",
    "min",
    "max",
)

BASE_FRAME_FIELDS = (
    "video",
    "frame_index",
    "timestamp_sec",
    "person_id",
    "face_index",
    "temporal_available",
    "window_frames_observed",
    "window_frames_expected",
    "window_sec_observed",
    "window_sec_expected",
    "raw_head_pose_yaw",
    "raw_head_pose_pitch",
    "raw_head_pose_roll",
    "raw_eyes_available",
    "raw_eyes_mean_openness",
    "raw_gaze_available",
    "raw_gaze_mean_iris_x",
    "raw_gaze_mean_iris_y",
    "raw_mouth_available",
    "raw_mouth_openness",
    "raw_mouth_motion_available",
    "raw_mouth_movement",
    "raw_quality_available",
    "raw_quality_brightness",
    "raw_quality_sharpness",
    "raw_quality_face_area_ratio",
    "raw_blink",
    "raw_emotion_available",
    "raw_emotion",
    "blink_events_observed",
    "blink_events_expected",
    "dominant_emotion_observed",
    "dominant_emotion_expected",
    "row_status",
)

FRAME_FIELDS = list(
    BASE_FRAME_FIELDS
)

for feature_name, _ in NUMERIC_FEATURES:
    for statistic in STATISTICS:
        FRAME_FIELDS.extend(
            [
                f"{feature_name}_{statistic}_observed",
                f"{feature_name}_{statistic}_expected",
                f"{feature_name}_{statistic}_absolute_error",
            ]
        )

CHECK_FIELDS = (
    "video",
    "frame_index",
    "person_id",
    "check",
    "expected",
    "observed",
    "absolute_error",
    "status",
)

METRIC_FIELDS = (
    "metric",
    "value",
)


def import_current_project():
    """Import the current repository implementation and verify source paths."""
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(
            f"PhysioTrack source directory not found: {SRC_ROOT}"
        )

    src_string = str(
        SRC_ROOT
    )

    if src_string not in sys.path:
        sys.path.insert(
            0,
            src_string,
        )

    from physiotrack.face.analysis import FaceAnalysis
    from physiotrack.face.config import FaceAnalysisConfig
    from physiotrack.face.temporal import FaceTemporalAggregator

    analysis_source = Path(
        inspect.getfile(
            FaceAnalysis
        )
    ).resolve()

    temporal_source = Path(
        inspect.getfile(
            FaceTemporalAggregator
        )
    ).resolve()

    if analysis_source != EXPECTED_ANALYSIS_SOURCE.resolve():
        raise RuntimeError(
            "Imported FaceAnalysis does not match the current repository "
            f"source. Expected {EXPECTED_ANALYSIS_SOURCE.resolve()}, "
            f"got {analysis_source}"
        )

    if temporal_source != EXPECTED_TEMPORAL_SOURCE.resolve():
        raise RuntimeError(
            "Imported FaceTemporalAggregator does not match the current "
            f"repository source. Expected {EXPECTED_TEMPORAL_SOURCE.resolve()}, "
            f"got {temporal_source}"
        )

    return (
        FaceAnalysis,
        FaceAnalysisConfig,
        analysis_source,
        temporal_source,
    )


def discover_videos():
    """Discover supported validation videos from repository-relative fixtures."""
    videos = []

    for video_dir in VIDEO_DIRS:
        if not video_dir.is_dir():
            raise FileNotFoundError(
                f"Required integration fixture directory not found: "
                f"{video_dir.relative_to(REPO_ROOT)}"
            )

        discovered = sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in SUPPORTED_VIDEO_EXTENSIONS
        )

        if not discovered:
            raise FileNotFoundError(
                f"No supported videos found in: "
                f"{video_dir.relative_to(REPO_ROOT)}"
            )

        videos.extend(
            discovered
        )

    return videos


def relative_path(
    path,
):
    """Return a repository-relative path using forward slashes."""
    return str(
        path.resolve().relative_to(
            REPO_ROOT.resolve()
        )
    ).replace(
        "\\",
        "/",
    )


def finite_float(
    value,
):
    """Return a finite float or None."""
    if value is None:
        return None

    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        numeric
    ):
        return None

    return numeric


def independent_summary(
    values,
):
    """Calculate population statistics independently from the project code."""
    valid_values = [
        value
        for value in (
            finite_float(
                item
            )
            for item in values
        )
        if value is not None
    ]

    if not valid_values:
        return None

    count = len(
        valid_values
    )
    mean = sum(
        valid_values
    ) / count

    variance = sum(
        (
            value
            - mean
        ) ** 2
        for value in valid_values
    ) / count

    return {
        "mean": float(
            mean
        ),
        "std": float(
            math.sqrt(
                variance
            )
        ),
        "min": float(
            min(
                valid_values
            )
        ),
        "max": float(
            max(
                valid_values
            )
        ),
    }


def independent_dominant_emotion(
    frames,
):
    """Return the most frequent available emotion with stable first-seen ties."""
    counts = {}
    order = []

    for frame in frames:
        if not frame[
            "emotion_available"
        ]:
            continue

        emotion = frame[
            "emotion"
        ]

        if emotion is None:
            continue

        if emotion not in counts:
            counts[
                emotion
            ] = 0
            order.append(
                emotion
            )

        counts[
            emotion
        ] += 1

    if not counts:
        return None

    best_emotion = order[
        0
    ]
    best_count = counts[
        best_emotion
    ]

    for emotion in order[
        1:
    ]:
        count = counts[
            emotion
        ]

        if count > best_count:
            best_emotion = emotion
            best_count = count

    return best_emotion


def extract_raw_features(
    instance,
):
    """Extract only the actual upstream values used by temporal aggregation."""
    features = (
        instance.face_features
        if instance.face_features is not None
        else {}
    )

    orientation = (
        instance.orientation
        if instance.orientation is not None
        else {}
    )

    eyes = features.get(
        "eyes",
        {},
    )
    gaze = features.get(
        "gaze",
        {},
    )
    mouth = features.get(
        "mouth",
        {},
    )
    mouth_motion = features.get(
        "mouth_motion",
        {},
    )
    quality = features.get(
        "quality",
        {},
    )
    blink = features.get(
        "blink",
        {},
    )
    emotion = features.get(
        "emotion",
        {},
    )

    return {
        "head_pose_yaw": orientation.get(
            "yaw"
        ),
        "head_pose_pitch": orientation.get(
            "pitch"
        ),
        "head_pose_roll": orientation.get(
            "roll"
        ),
        "eyes_available": bool(
            eyes.get(
                "available",
                False,
            )
        ),
        "eyes_mean_openness": eyes.get(
            "mean_openness"
        ),
        "gaze_available": bool(
            gaze.get(
                "available",
                False,
            )
        ),
        "gaze_mean_iris_x": gaze.get(
            "mean_iris_x"
        ),
        "gaze_mean_iris_y": gaze.get(
            "mean_iris_y"
        ),
        "mouth_available": bool(
            mouth.get(
                "available",
                False,
            )
        ),
        "mouth_openness": mouth.get(
            "mouth_openness"
        ),
        "mouth_motion_available": bool(
            mouth_motion.get(
                "available",
                False,
            )
        ),
        "mouth_movement": mouth_motion.get(
            "mouth_movement"
        ),
        "quality_available": bool(
            quality.get(
                "available",
                False,
            )
        ),
        "quality_brightness": quality.get(
            "brightness"
        ),
        "quality_sharpness": quality.get(
            "sharpness"
        ),
        "quality_face_area_ratio": quality.get(
            "face_area_ratio"
        ),
        "blink": bool(
            blink.get(
                "blink",
                False,
            )
        ),
        "emotion_available": bool(
            emotion.get(
                "available",
                False,
            )
        ),
        "emotion": emotion.get(
            "emotion"
        ),
    }


def values_for_feature(
    frames,
    feature_name,
):
    """Return independently filtered source values for one summary feature."""
    if feature_name.startswith(
        "head_pose_"
    ):
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                feature_name
            ]
            is not None
        ]

    if feature_name == "eyes_mean_openness":
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                "eyes_available"
            ]
        ]

    if feature_name in {
        "gaze_mean_iris_x",
        "gaze_mean_iris_y",
    }:
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                "gaze_available"
            ]
        ]

    if feature_name == "mouth_openness":
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                "mouth_available"
            ]
        ]

    if feature_name == "mouth_movement":
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                "mouth_motion_available"
            ]
        ]

    if feature_name.startswith(
        "quality_"
    ):
        return [
            frame[
                feature_name
            ]
            for frame in frames
            if frame[
                "quality_available"
            ]
        ]

    raise KeyError(
        f"Unsupported feature for independent validation: {feature_name}"
    )


def get_observed_summary_value(
    summary,
    path,
):
    """Read one numerical summary object from the actual temporal output."""
    current = summary

    for key in path:
        if current is None:
            return None

        current = current.get(
            key
        )

    return current


def format_value(
    value,
):
    """Format a result value for CSV output."""
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.16g}"

    return str(
        value
    )


def compare_value(
    expected,
    observed,
    tolerance=NUMERIC_TOLERANCE,
):
    """Compare scalar values and return pass status plus absolute error."""
    expected_numeric = finite_float(
        expected
    )
    observed_numeric = finite_float(
        observed
    )

    if (
        expected_numeric is not None
        and observed_numeric is not None
    ):
        absolute_error = abs(
            observed_numeric
            - expected_numeric
        )

        return (
            absolute_error
            <= tolerance,
            absolute_error,
        )

    return (
        observed == expected,
        None,
    )


def add_check(
    checks,
    *,
    video,
    frame_index,
    person_id,
    check,
    expected,
    observed,
):
    """Append one independently evaluated correctness check."""
    passed, absolute_error = compare_value(
        expected,
        observed,
    )

    checks.append(
        {
            "video": video,
            "frame_index": frame_index,
            "person_id": person_id,
            "check": check,
            "expected": format_value(
                expected
            ),
            "observed": format_value(
                observed
            ),
            "absolute_error": (
                ""
                if absolute_error is None
                else format_value(
                    absolute_error
                )
            ),
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
        }
    )

    return (
        passed,
        absolute_error,
    )


def build_config(
    FaceAnalysisConfig,
):
    """Enable exactly the real upstream components summarized by Temporal."""
    return FaceAnalysisConfig(
        tracking=True,
        head_pose=True,
        landmarks=True,
        quality=True,
        eyes=True,
        blink=True,
        gaze=True,
        gaze_estimation=False,
        mouth=True,
        mouth_motion=True,
        emotion=True,
        regions=False,
        temporal=True,
        tracker_type="ocsort",
        blink_threshold=0.22,
        min_closed_frames=3,
        temporal_window_sec=5.0,
    )


def process_video(
    video_path,
    FaceAnalysis,
    FaceAnalysisConfig,
):
    """Run the real FaceAnalysis path and audit every temporal output."""
    video_name = relative_path(
        video_path
    )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_name}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    reported_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        not math.isfinite(
            fps
        )
        or fps <= 0
    ):
        capture.release()

        raise RuntimeError(
            f"Invalid video FPS for {video_name}: {fps}"
        )

    config = build_config(
        FaceAnalysisConfig
    )

    analysis = FaceAnalysis(
        config=config,
        fps=fps,
        device="cpu",
    )

    window_frames = max(
        1,
        int(
            round(
                fps
                * config.temporal_window_sec
            )
        ),
    )

    buffers = defaultdict(
        lambda: deque(
            maxlen=window_frames
        )
    )

    active_person_ids = set()
    frame_rows = []
    checks = []

    processed_frames = 0
    frames_with_faces = 0
    total_face_records = 0
    duplicate_person_id_frames = 0

    try:
        frame_index = 0

        while True:
            ok, frame = capture.read()

            if not ok:
                break

            result = analysis.predict(
                frame
            )

            processed_frames += 1

            instances = list(
                result
            )

            current_person_ids = {
                instance.id
                for instance in instances
                if instance.id is not None
            }

            missing_person_ids = (
                active_person_ids
                - current_person_ids
            )

            for person_id in missing_person_ids:
                buffers.pop(
                    person_id,
                    None,
                )

            active_person_ids = current_person_ids

            if instances:
                frames_with_faces += 1

            ids_in_frame = [
                instance.id
                for instance in instances
                if instance.id is not None
            ]

            if len(
                ids_in_frame
            ) != len(
                set(
                    ids_in_frame
                )
            ):
                duplicate_person_id_frames += 1

            timestamp_sec = (
                frame_index
                / fps
            )

            for face_index, instance in enumerate(
                instances
            ):
                total_face_records += 1

                person_id = instance.id
                features = (
                    instance.face_features
                    if instance.face_features is not None
                    else {}
                )

                temporal = features.get(
                    "temporal",
                    {},
                )

                temporal_available = bool(
                    temporal.get(
                        "available",
                        False,
                    )
                )

                observed_summary = temporal.get(
                    "summary"
                )

                raw = extract_raw_features(
                    instance
                )

                row = {
                    "video": video_name,
                    "frame_index": frame_index,
                    "timestamp_sec": format_value(
                        timestamp_sec
                    ),
                    "person_id": person_id,
                    "face_index": face_index,
                    "temporal_available": temporal_available,
                    "raw_head_pose_yaw": format_value(
                        raw[
                            "head_pose_yaw"
                        ]
                    ),
                    "raw_head_pose_pitch": format_value(
                        raw[
                            "head_pose_pitch"
                        ]
                    ),
                    "raw_head_pose_roll": format_value(
                        raw[
                            "head_pose_roll"
                        ]
                    ),
                    "raw_eyes_available": raw[
                        "eyes_available"
                    ],
                    "raw_eyes_mean_openness": format_value(
                        raw[
                            "eyes_mean_openness"
                        ]
                    ),
                    "raw_gaze_available": raw[
                        "gaze_available"
                    ],
                    "raw_gaze_mean_iris_x": format_value(
                        raw[
                            "gaze_mean_iris_x"
                        ]
                    ),
                    "raw_gaze_mean_iris_y": format_value(
                        raw[
                            "gaze_mean_iris_y"
                        ]
                    ),
                    "raw_mouth_available": raw[
                        "mouth_available"
                    ],
                    "raw_mouth_openness": format_value(
                        raw[
                            "mouth_openness"
                        ]
                    ),
                    "raw_mouth_motion_available": raw[
                        "mouth_motion_available"
                    ],
                    "raw_mouth_movement": format_value(
                        raw[
                            "mouth_movement"
                        ]
                    ),
                    "raw_quality_available": raw[
                        "quality_available"
                    ],
                    "raw_quality_brightness": format_value(
                        raw[
                            "quality_brightness"
                        ]
                    ),
                    "raw_quality_sharpness": format_value(
                        raw[
                            "quality_sharpness"
                        ]
                    ),
                    "raw_quality_face_area_ratio": format_value(
                        raw[
                            "quality_face_area_ratio"
                        ]
                    ),
                    "raw_blink": raw[
                        "blink"
                    ],
                    "raw_emotion_available": raw[
                        "emotion_available"
                    ],
                    "raw_emotion": raw[
                        "emotion"
                    ],
                }

                row_pass = True

                passed, _ = add_check(
                    checks,
                    video=video_name,
                    frame_index=frame_index,
                    person_id=person_id,
                    check="temporal_available",
                    expected=(
                        person_id
                        is not None
                    ),
                    observed=temporal_available,
                )

                row_pass &= passed

                if (
                    person_id is None
                    or observed_summary is None
                ):
                    row[
                        "window_frames_observed"
                    ] = ""
                    row[
                        "window_frames_expected"
                    ] = ""
                    row[
                        "window_sec_observed"
                    ] = ""
                    row[
                        "window_sec_expected"
                    ] = ""
                    row[
                        "blink_events_observed"
                    ] = ""
                    row[
                        "blink_events_expected"
                    ] = ""
                    row[
                        "dominant_emotion_observed"
                    ] = ""
                    row[
                        "dominant_emotion_expected"
                    ] = ""

                    for feature_name, _ in NUMERIC_FEATURES:
                        for statistic in STATISTICS:
                            row[
                                f"{feature_name}_{statistic}_observed"
                            ] = ""
                            row[
                                f"{feature_name}_{statistic}_expected"
                            ] = ""
                            row[
                                f"{feature_name}_{statistic}_absolute_error"
                            ] = ""

                    row[
                        "row_status"
                    ] = (
                        "PASS"
                        if row_pass
                        else "FAIL"
                    )

                    frame_rows.append(
                        row
                    )

                    continue

                buffers[
                    person_id
                ].append(
                    raw
                )

                frames = list(
                    buffers[
                        person_id
                    ]
                )

                expected_window_frames = len(
                    frames
                )
                expected_window_sec = (
                    expected_window_frames
                    / fps
                )

                observed_window_frames = observed_summary.get(
                    "window_frames"
                )
                observed_window_sec = observed_summary.get(
                    "window_sec"
                )

                row[
                    "window_frames_observed"
                ] = format_value(
                    observed_window_frames
                )
                row[
                    "window_frames_expected"
                ] = format_value(
                    expected_window_frames
                )
                row[
                    "window_sec_observed"
                ] = format_value(
                    observed_window_sec
                )
                row[
                    "window_sec_expected"
                ] = format_value(
                    expected_window_sec
                )

                for check_name, expected, observed in (
                    (
                        "person_id",
                        person_id,
                        observed_summary.get(
                            "person_id"
                        ),
                    ),
                    (
                        "window_frames",
                        expected_window_frames,
                        observed_window_frames,
                    ),
                    (
                        "window_sec",
                        expected_window_sec,
                        observed_window_sec,
                    ),
                ):
                    passed, _ = add_check(
                        checks,
                        video=video_name,
                        frame_index=frame_index,
                        person_id=person_id,
                        check=check_name,
                        expected=expected,
                        observed=observed,
                    )

                    row_pass &= passed

                for feature_name, summary_path in NUMERIC_FEATURES:
                    expected_summary = independent_summary(
                        values_for_feature(
                            frames,
                            feature_name,
                        )
                    )

                    observed_feature_summary = get_observed_summary_value(
                        observed_summary,
                        summary_path,
                    )

                    for statistic in STATISTICS:
                        expected_value = (
                            None
                            if expected_summary is None
                            else expected_summary[
                                statistic
                            ]
                        )

                        observed_value = (
                            None
                            if observed_feature_summary is None
                            else observed_feature_summary.get(
                                statistic
                            )
                        )

                        passed, absolute_error = add_check(
                            checks,
                            video=video_name,
                            frame_index=frame_index,
                            person_id=person_id,
                            check=(
                                f"{feature_name}.{statistic}"
                            ),
                            expected=expected_value,
                            observed=observed_value,
                        )

                        row_pass &= passed

                        row[
                            f"{feature_name}_{statistic}_observed"
                        ] = format_value(
                            observed_value
                        )
                        row[
                            f"{feature_name}_{statistic}_expected"
                        ] = format_value(
                            expected_value
                        )
                        row[
                            f"{feature_name}_{statistic}_absolute_error"
                        ] = (
                            ""
                            if absolute_error is None
                            else format_value(
                                absolute_error
                            )
                        )

                expected_blinks = sum(
                    1
                    for buffered_frame in frames
                    if buffered_frame[
                        "blink"
                    ]
                )

                observed_blinks = (
                    observed_summary
                    .get(
                        "blink",
                        {},
                    )
                    .get(
                        "events"
                    )
                )

                passed, _ = add_check(
                    checks,
                    video=video_name,
                    frame_index=frame_index,
                    person_id=person_id,
                    check="blink.events",
                    expected=expected_blinks,
                    observed=observed_blinks,
                )

                row_pass &= passed

                expected_emotion = independent_dominant_emotion(
                    frames
                )

                observed_emotion = (
                    observed_summary
                    .get(
                        "emotion",
                        {},
                    )
                    .get(
                        "dominant"
                    )
                )

                passed, _ = add_check(
                    checks,
                    video=video_name,
                    frame_index=frame_index,
                    person_id=person_id,
                    check="emotion.dominant",
                    expected=expected_emotion,
                    observed=observed_emotion,
                )

                row_pass &= passed

                row[
                    "blink_events_observed"
                ] = format_value(
                    observed_blinks
                )
                row[
                    "blink_events_expected"
                ] = format_value(
                    expected_blinks
                )
                row[
                    "dominant_emotion_observed"
                ] = format_value(
                    observed_emotion
                )
                row[
                    "dominant_emotion_expected"
                ] = format_value(
                    expected_emotion
                )
                row[
                    "row_status"
                ] = (
                    "PASS"
                    if row_pass
                    else "FAIL"
                )

                frame_rows.append(
                    row
                )

            frame_index += 1

    finally:
        capture.release()
        analysis.close()

    return {
        "video": video_name,
        "fps": fps,
        "reported_frames": reported_frames,
        "processed_frames": processed_frames,
        "frames_with_faces": frames_with_faces,
        "total_face_records": total_face_records,
        "duplicate_person_id_frames": duplicate_person_id_frames,
        "window_frames": window_frames,
        "frame_rows": frame_rows,
        "checks": checks,
    }


def validate_result_integrity(
    frame_rows,
    checks,
):
    """Validate table integrity before files are accepted."""
    if not frame_rows:
        raise RuntimeError(
            "No real FaceAnalysis face records were generated"
        )

    if not checks:
        raise RuntimeError(
            "No real temporal correctness checks were generated"
        )

    frame_keys = set()

    for row in frame_rows:
        key = (
            row[
                "video"
            ],
            int(
                row[
                    "frame_index"
                ]
            ),
            str(
                row[
                    "person_id"
                ]
            ),
            int(
                row[
                    "face_index"
                ]
            ),
        )

        if key in frame_keys:
            raise RuntimeError(
                f"Duplicate frame-face result key found: {key}"
            )

        frame_keys.add(
            key
        )

    check_keys = set()

    for row in checks:
        key = (
            row[
                "video"
            ],
            int(
                row[
                    "frame_index"
                ]
            ),
            str(
                row[
                    "person_id"
                ]
            ),
            row[
                "check"
            ],
        )

        if key in check_keys:
            raise RuntimeError(
                f"Duplicate temporal check key found: {key}"
            )

        check_keys.add(
            key
        )

        if row[
            "status"
        ] not in {
            "PASS",
            "FAIL",
        }:
            raise RuntimeError(
                f"Unexpected check status: {row['status']}"
            )


def build_metrics(
    video_summaries,
    frame_rows,
    checks,
    runtime_sec,
):
    """Build compact aggregate metrics for the real component run."""
    total_checks = len(
        checks
    )
    passed_checks = sum(
        1
        for row in checks
        if row[
            "status"
        ]
        == "PASS"
    )
    failed_checks = (
        total_checks
        - passed_checks
    )

    failed_rows = sum(
        1
        for row in frame_rows
        if row[
            "row_status"
        ]
        != "PASS"
    )

    numeric_errors = [
        float(
            row[
                "absolute_error"
            ]
        )
        for row in checks
        if row[
            "absolute_error"
        ]
        != ""
    ]

    max_error = (
        max(
            numeric_errors
        )
        if numeric_errors
        else 0.0
    )

    return [
        {
            "metric": "videos_discovered",
            "value": len(
                video_summaries
            ),
        },
        {
            "metric": "videos_passed",
            "value": sum(
                1
                for item in video_summaries
                if item[
                    "status"
                ]
                == "PASS"
            ),
        },
        {
            "metric": "videos_failed",
            "value": sum(
                1
                for item in video_summaries
                if item[
                    "status"
                ]
                == "FAIL"
            ),
        },
        {
            "metric": "processed_frames",
            "value": sum(
                item[
                    "processed_frames"
                ]
                for item in video_summaries
                if item[
                    "status"
                ]
                == "PASS"
            ),
        },
        {
            "metric": "face_records",
            "value": len(
                frame_rows
            ),
        },
        {
            "metric": "temporal_available_records",
            "value": sum(
                1
                for row in frame_rows
                if str(
                    row[
                        "temporal_available"
                    ]
                ).lower()
                == "true"
            ),
        },
        {
            "metric": "failed_frame_rows",
            "value": failed_rows,
        },
        {
            "metric": "total_checks",
            "value": total_checks,
        },
        {
            "metric": "passed_checks",
            "value": passed_checks,
        },
        {
            "metric": "failed_checks",
            "value": failed_checks,
        },
        {
            "metric": "max_numeric_absolute_error",
            "value": max_error,
        },
        {
            "metric": "runtime_sec",
            "value": runtime_sec,
        },
        {
            "metric": "overall_status",
            "value": (
                "PASS"
                if failed_checks == 0
                and failed_rows == 0
                and all(
                    item[
                        "status"
                    ]
                    == "PASS"
                    for item in video_summaries
                )
                else "FAIL"
            ),
        },
    ]


def write_csv(
    path,
    rows,
    fieldnames,
):
    """Write a CSV file with an explicit stable header."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def validate_staged_outputs(
    staging_dir,
):
    """Validate every staged artifact before promotion."""
    required = (
        staging_dir
        / FRAME_RESULTS_PATH.name,
        staging_dir
        / CHECKS_PATH.name,
        staging_dir
        / METRICS_PATH.name,
        staging_dir
        / SUMMARY_PATH.name,
    )

    for path in required:
        if not path.is_file():
            raise RuntimeError(
                f"Missing staged output: {path.name}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Empty staged output: {path.name}"
            )

    with (
        staging_dir
        / FRAME_RESULTS_PATH.name
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != FRAME_FIELDS:
            raise RuntimeError(
                "Staged frame-result CSV schema mismatch"
            )

        rows = list(
            reader
        )

        if not rows:
            raise RuntimeError(
                "Staged frame-result CSV contains no rows"
            )

        if any(
            row[
                "row_status"
            ]
            != "PASS"
            for row in rows
        ):
            raise RuntimeError(
                "Staged frame-result CSV contains failed rows"
            )

    with (
        staging_dir
        / CHECKS_PATH.name
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if tuple(
            reader.fieldnames
            or ()
        ) != CHECK_FIELDS:
            raise RuntimeError(
                "Staged check CSV schema mismatch"
            )

        rows = list(
            reader
        )

        if not rows:
            raise RuntimeError(
                "Staged check CSV contains no rows"
            )

        if any(
            row[
                "status"
            ]
            != "PASS"
            for row in rows
        ):
            raise RuntimeError(
                "Staged check CSV contains failed checks"
            )

    summary = json.loads(
        (
            staging_dir
            / SUMMARY_PATH.name
        ).read_text(
            encoding="utf-8"
        )
    )

    if summary.get(
        "overall_status"
    ) != "PASS":
        raise RuntimeError(
            "Staged summary does not report overall PASS"
        )


def promote_outputs(
    staging_dir,
):
    """Transactionally replace only outputs owned by this script."""
    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_paths = (
        FRAME_RESULTS_PATH,
        CHECKS_PATH,
        METRICS_PATH,
        SUMMARY_PATH,
    )

    rollback_dir = Path(
        tempfile.mkdtemp(
            prefix=".temporal_component_rollback_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    replaced = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                shutil.copy2(
                    final_path,
                    rollback_dir
                    / final_path.name,
                )

        for final_path in final_paths:
            staged_path = (
                staging_dir
                / final_path.name
            )

            os.replace(
                staged_path,
                final_path,
            )

            replaced.append(
                final_path
            )

    except Exception:
        for final_path in replaced:
            if final_path.exists():
                final_path.unlink()

        for backup_path in rollback_dir.iterdir():
            os.replace(
                backup_path,
                RESULTS_DIR
                / backup_path.name,
            )

        raise

    finally:
        shutil.rmtree(
            rollback_dir,
            ignore_errors=True,
        )


def cleanup_legacy_outputs():
    """Remove only obsolete root-level outputs previously owned by this script."""
    removed = []

    for path in LEGACY_OUTPUT_PATHS:
        if path.is_file():
            path.unlink()
            removed.append(
                path.name
            )

    return removed


def main():
    """Run the real FaceAnalysis temporal-component validation."""
    print(
        "PhysioTrack Temporal Aggregation Real Component Validation"
    )
    print(
        "========================================================="
    )

    (
        FaceAnalysis,
        FaceAnalysisConfig,
        analysis_source,
        temporal_source,
    ) = import_current_project()

    videos = discover_videos()

    print(
        "Preflight: PASS"
    )
    print(
        "FaceAnalysis source: "
        f"{relative_path(analysis_source)}"
    )
    print(
        "Temporal source: "
        f"{relative_path(temporal_source)}"
    )
    print(
        f"Videos discovered: {len(videos)}"
    )

    for video in videos:
        print(
            f"  {relative_path(video)}"
        )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".temporal_component_staging_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    start_time = time.perf_counter()

    all_frame_rows = []
    all_checks = []
    video_summaries = []

    try:
        for video in videos:
            video_name = relative_path(
                video
            )

            print(
                f"Processing: {video_name}"
            )

            try:
                result = process_video(
                    video,
                    FaceAnalysis,
                    FaceAnalysisConfig,
                )

                video_failed_checks = sum(
                    1
                    for row in result[
                        "checks"
                    ]
                    if row[
                        "status"
                    ]
                    != "PASS"
                )

                video_failed_rows = sum(
                    1
                    for row in result[
                        "frame_rows"
                    ]
                    if row[
                        "row_status"
                    ]
                    != "PASS"
                )

                status = (
                    "PASS"
                    if video_failed_checks == 0
                    and video_failed_rows == 0
                    and result[
                        "duplicate_person_id_frames"
                    ]
                    == 0
                    else "FAIL"
                )

                all_frame_rows.extend(
                    result[
                        "frame_rows"
                    ]
                )
                all_checks.extend(
                    result[
                        "checks"
                    ]
                )

                video_summaries.append(
                    {
                        "video": video_name,
                        "status": status,
                        "fps": result[
                            "fps"
                        ],
                        "reported_frames": result[
                            "reported_frames"
                        ],
                        "processed_frames": result[
                            "processed_frames"
                        ],
                        "frames_with_faces": result[
                            "frames_with_faces"
                        ],
                        "face_records": result[
                            "total_face_records"
                        ],
                        "window_frames": result[
                            "window_frames"
                        ],
                        "duplicate_person_id_frames": result[
                            "duplicate_person_id_frames"
                        ],
                        "failed_rows": video_failed_rows,
                        "failed_checks": video_failed_checks,
                        "error": None,
                    }
                )

                print(
                    "  "
                    f"Frames: {result['processed_frames']}, "
                    f"face records: {result['total_face_records']}, "
                    f"failed checks: {video_failed_checks}, "
                    f"status: {status}"
                )

            except Exception as exc:
                video_summaries.append(
                    {
                        "video": video_name,
                        "status": "FAIL",
                        "fps": None,
                        "reported_frames": None,
                        "processed_frames": 0,
                        "frames_with_faces": 0,
                        "face_records": 0,
                        "window_frames": None,
                        "duplicate_person_id_frames": None,
                        "failed_rows": None,
                        "failed_checks": None,
                        "error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

                print(
                    "  "
                    f"FAILED: {type(exc).__name__}: {exc}"
                )

        validate_result_integrity(
            all_frame_rows,
            all_checks,
        )

        runtime_sec = (
            time.perf_counter()
            - start_time
        )

        metrics = build_metrics(
            video_summaries,
            all_frame_rows,
            all_checks,
            runtime_sec,
        )

        metric_map = {
            row[
                "metric"
            ]: row[
                "value"
            ]
            for row in metrics
        }

        write_csv(
            staging_dir
            / FRAME_RESULTS_PATH.name,
            all_frame_rows,
            FRAME_FIELDS,
        )
        write_csv(
            staging_dir
            / CHECKS_PATH.name,
            all_checks,
            CHECK_FIELDS,
        )
        write_csv(
            staging_dir
            / METRICS_PATH.name,
            metrics,
            METRIC_FIELDS,
        )

        summary = {
            "purpose": (
                "Real PhysioTrack FaceAnalysis execution validation for "
                "FaceTemporalAggregator with independent frame-level "
                "recomputation of every current temporal summary field."
            ),
            "analysis_source": relative_path(
                analysis_source
            ),
            "temporal_source": relative_path(
                temporal_source
            ),
            "configuration": {
                "tracking": True,
                "head_pose": True,
                "landmarks": True,
                "quality": True,
                "eyes": True,
                "blink": True,
                "gaze": True,
                "gaze_estimation": False,
                "mouth": True,
                "mouth_motion": True,
                "emotion": True,
                "regions": False,
                "temporal": True,
                "tracker_type": "ocsort",
                "blink_threshold": 0.22,
                "min_closed_frames": 3,
                "temporal_window_sec": 5.0,
            },
            "scope_note": (
                "GazeEstimator and face-region segmentation are disabled "
                "because the current FaceTemporalAggregator does not summarize "
                "their outputs. All upstream components currently consumed by "
                "the temporal aggregator are enabled."
            ),
            "output_directory": (
                "validation/temporal_aggregation/results/component_execution"
            ),
            "videos": video_summaries,
            "metrics": metric_map,
            "overall_status": metric_map[
                "overall_status"
            ],
        }

        (
            staging_dir
            / SUMMARY_PATH.name
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        validate_staged_outputs(
            staging_dir
        )

        print(
            f"Processed frames: {metric_map['processed_frames']}"
        )
        print(
            f"Face records: {metric_map['face_records']}"
        )
        print(
            f"Temporal records: "
            f"{metric_map['temporal_available_records']}"
        )
        print(
            f"Total checks: {metric_map['total_checks']}"
        )
        print(
            f"Passed checks: {metric_map['passed_checks']}"
        )
        print(
            f"Failed checks: {metric_map['failed_checks']}"
        )
        print(
            "Maximum numerical absolute error: "
            f"{float(metric_map['max_numeric_absolute_error']):.16g}"
        )
        print(
            "Staged outputs: PASS"
        )

        if metric_map[
            "overall_status"
        ] != "PASS":
            raise RuntimeError(
                "Real Temporal Aggregation component validation failed"
            )

        promote_outputs(
            staging_dir
        )

        removed_legacy = cleanup_legacy_outputs()

        print(
            "Final output promotion: PASS"
        )
        print(
            "Output directory: "
            "results/component_execution/"
        )

        if removed_legacy:
            print(
                "Legacy root-level component outputs removed: "
                f"{len(removed_legacy)}"
            )

        print(
            "Overall status: PASS"
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
