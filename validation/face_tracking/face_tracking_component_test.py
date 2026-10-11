from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import (
    Face,
    FaceAnalysis,
    FaceAnalysisConfig,
    FaceTracker,
)


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FACE_TRACKING_ECCV2016"
)

RESULTS_DIR = (
    VALIDATION_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_CSV_PATH = (
    OUTPUT_DIR
    / "face_tracking_component_results.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "face_tracking_component_summary.json"
)

VIDEO_CONFIGS = [
    (
        "Apink",
        "Apink.mp4",
        "Apink_gt.xml",
    ),
    (
        "BrunoMars",
        "BrunoMars.mp4",
        "BrunoMars_gt.xml",
    ),
    (
        "Darling",
        "Darling.mp4",
        "Darling_gt.xml",
    ),
    (
        "GirlsAloud",
        "GirlsAloud.mp4",
        "GirlsAloud_gt.xml",
    ),
    (
        "HelloBubble",
        "HelloBubble.mp4",
        "HelloBubble_gt.xml",
    ),
    (
        "PussycatDolls",
        "PussycatDolls.mp4",
        "stickwitu_gt.xml",
    ),
    (
        "T-ara",
        "T-ara.mov",
        "Tara_gt.xml",
    ),
    (
        "Westlife",
        "Westlife.mp4",
        "Westlife_gt.xml",
    ),
]

EXPECTED_VIDEO_COUNT = 8
EXPECTED_TOTAL_FRAMES = 42007

FIELDNAMES = [
    "sequence",
    "source_video",
    "frame_number",
    "timestamp_seconds",
    "image_width",
    "image_height",
    "track_index",
    "tracks_in_frame",
    "track_id",
    "class_id",
    "class_name",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_width",
    "box_height",
    "box_area",
    "confidence",
    "status",
    "failure_reason",
]

UNRELATED_PIPELINE_ATTRIBUTES = (
    "orientation",
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
)


def scalar_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (
            TypeError,
            ValueError,
        ):
            pass

    return value


def finite_numeric(
    value: Any,
) -> bool:
    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def read_ground_truth_end_frame(
    xml_path: Path,
) -> int:
    root = ET.parse(
        xml_path
    ).getroot()

    end_frame = int(
        root.attrib[
            "end_frame"
        ]
    )

    if end_frame <= 0:
        raise RuntimeError(
            "Ground-truth end_frame must be positive: "
            f"{xml_path}"
        )

    return end_frame


def video_metadata(
    video_path: Path,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video during preflight: {video_path}"
        )

    frame_count = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
    )

    height = int(
        round(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
    )

    cap.release()

    if frame_count <= 0:
        raise RuntimeError(
            f"Video reports no frames: {video_path}"
        )

    if not finite_numeric(
        fps
    ) or fps <= 0:
        raise RuntimeError(
            f"Video reports invalid FPS: {video_path}"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Video reports invalid dimensions: {video_path}"
        )

    return {
        "frame_count":
            frame_count,
        "fps":
            fps,
        "width":
            width,
        "height":
            height,
    }


def preflight() -> list[dict[str, Any]]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {DATASET_ROOT}"
        )

    if len(
        VIDEO_CONFIGS
    ) != EXPECTED_VIDEO_COUNT:
        raise RuntimeError(
            "Unexpected configured video count."
        )

    prepared = []
    total_expected_frames = 0

    for (
        sequence,
        video_name,
        gt_name,
    ) in VIDEO_CONFIGS:
        video_path = (
            DATASET_ROOT
            / "videos"
            / video_name
        )

        gt_path = (
            DATASET_ROOT
            / "ground_truth"
            / "GT"
            / gt_name
        )

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Required video not found: {video_path}"
            )

        if not gt_path.is_file():
            raise FileNotFoundError(
                f"Required ground-truth XML not found: {gt_path}"
            )

        expected_frames = (
            read_ground_truth_end_frame(
                gt_path
            )
        )

        metadata = video_metadata(
            video_path
        )

        if (
            metadata[
                "frame_count"
            ]
            != expected_frames
        ):
            raise RuntimeError(
                "Video frame count does not match ground-truth end_frame "
                f"for {sequence}: "
                f"{metadata['frame_count']} != {expected_frames}"
            )

        total_expected_frames += (
            expected_frames
        )

        prepared.append(
            {
                "sequence":
                    sequence,
                "video_name":
                    video_name,
                "video_path":
                    video_path,
                "gt_name":
                    gt_name,
                "gt_path":
                    gt_path,
                "expected_frames":
                    expected_frames,
                "fps":
                    metadata[
                        "fps"
                    ],
                "width":
                    metadata[
                        "width"
                    ],
                "height":
                    metadata[
                        "height"
                    ],
            }
        )

    if (
        total_expected_frames
        != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            "Unexpected total benchmark frame count: "
            f"{total_expected_frames} "
            f"(expected {EXPECTED_TOTAL_FRAMES})"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_path = (
        OUTPUT_DIR
        / ".write_probe"
    )

    try:
        with probe_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "preflight"
            )
    finally:
        if probe_path.exists():
            probe_path.unlink()

    return prepared


def make_config() -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=True,
        head_pose=False,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
        tracker_type="ocsort",
    )

    config.validate()

    return config


def make_pipeline() -> FaceAnalysis:
    detector = Face(
        conf=0.25,
        iou=0.45,
        device="cpu",
        verbose=False,
    )

    tracker = FaceTracker(
        tracker_type="ocsort",
        device="cpu",
    )

    pipeline = FaceAnalysis(
        detector=detector,
        tracker=tracker,
        config=make_config(),
        device="cpu",
        verbose=False,
    )

    assert_tracking_only_pipeline(
        pipeline
    )

    return pipeline


def assert_tracking_only_pipeline(
    pipeline: FaceAnalysis,
) -> None:
    if pipeline.detector is None:
        raise RuntimeError(
            "Required Face Detection dependency is not initialized."
        )

    if pipeline.tracker is None:
        raise RuntimeError(
            "Face Tracking component is not initialized."
        )

    unexpected_enabled = [
        name
        for name in UNRELATED_PIPELINE_ATTRIBUTES
        if getattr(
            pipeline,
            name,
        ) is not None
    ]

    if unexpected_enabled:
        raise RuntimeError(
            "Unrelated FaceAnalysis components are enabled: "
            + ", ".join(
                unexpected_enabled
            )
        )


def normalize_box(
    box: Any,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if isinstance(
        box,
        np.ndarray,
    ):
        values = box.tolist()
    else:
        values = list(
            box
        )

    if len(values) != 4:
        raise RuntimeError(
            "Track box does not contain four coordinates."
        )

    if not all(
        finite_numeric(value)
        for value in values
    ):
        raise RuntimeError(
            "Track box contains a non-finite coordinate."
        )

    return tuple(
        float(value)
        for value in values
    )


def track_row(
    sequence: str,
    video_name: str,
    frame_number: int,
    fps: float,
    image_width: int,
    image_height: int,
    track_index: int,
    tracks_in_frame: int,
    track: Any,
) -> tuple[
    dict[str, Any],
    bool,
    bool,
]:
    x1, y1, x2, y2 = (
        normalize_box(
            track.box
        )
    )

    width = x2 - x1
    height = y2 - y1

    valid_box = (
        width > 0.0
        and height > 0.0
    )

    track_id = scalar_value(
        track.id
    )

    missing_id = (
        track_id is None
    )

    confidence = scalar_value(
        track.confidence
    )

    if confidence is not None:
        if not finite_numeric(
            confidence
        ):
            raise RuntimeError(
                "Track confidence is not finite."
            )

        confidence = float(
            confidence
        )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise RuntimeError(
                "Track confidence is outside [0, 1]."
            )

    if valid_box:
        box_width = width
        box_height = height
        box_area = (
            width
            * height
        )
    else:
        box_width = None
        box_height = None
        box_area = None

    if missing_id:
        status = (
            "TRACKED_MISSING_ID"
        )
        failure_reason = (
            "Tracked output has no tracking identifier."
        )
    elif not valid_box:
        status = (
            "TRACKED_INVALID_BOX"
        )
        failure_reason = (
            "Raw tracked box has non-positive width or height; "
            "raw coordinates are preserved without correction."
        )
    else:
        status = "TRACKED"
        failure_reason = ""

    row = {
        "sequence":
            sequence,
        "source_video":
            video_name,
        "frame_number":
            frame_number,
        "timestamp_seconds":
            (
                (frame_number - 1)
                / fps
            ),
        "image_width":
            image_width,
        "image_height":
            image_height,
        "track_index":
            track_index,
        "tracks_in_frame":
            tracks_in_frame,
        "track_id":
            track_id,
        "class_id":
            scalar_value(
                track.cls
            ),
        "class_name":
            scalar_value(
                track.cls_name
            ),
        "box_x1":
            x1,
        "box_y1":
            y1,
        "box_x2":
            x2,
        "box_y2":
            y2,
        "box_width":
            box_width,
        "box_height":
            box_height,
        "box_area":
            box_area,
        "confidence":
            confidence,
        "status":
            status,
        "failure_reason":
            failure_reason,
    }

    return (
        row,
        valid_box,
        missing_id,
    )


def empty_frame_row(
    sequence: str,
    video_name: str,
    frame_number: int,
    fps: float,
    image_width: int | None,
    image_height: int | None,
    status: str,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "sequence":
            sequence,
        "source_video":
            video_name,
        "frame_number":
            frame_number,
        "timestamp_seconds":
            (
                (frame_number - 1)
                / fps
            ),
        "image_width":
            image_width,
        "image_height":
            image_height,
        "track_index":
            None,
        "tracks_in_frame":
            0,
        "track_id":
            None,
        "class_id":
            None,
        "class_name":
            None,
        "box_x1":
            None,
        "box_y1":
            None,
        "box_x2":
            None,
        "box_y2":
            None,
        "box_width":
            None,
        "box_height":
            None,
        "box_area":
            None,
        "confidence":
            None,
        "status":
            status,
        "failure_reason":
            failure_reason,
    }


def run_sequence(
    item: dict[str, Any],
    writer: csv.DictWriter,
) -> dict[str, Any]:
    sequence = item[
        "sequence"
    ]

    video_name = item[
        "video_name"
    ]

    video_path = item[
        "video_path"
    ]

    expected_frames = item[
        "expected_frames"
    ]

    fps = float(
        item[
            "fps"
        ]
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    pipeline = make_pipeline()

    frames_processed = 0
    frame_read_failures = 0
    prediction_failures = 0
    frames_with_tracks = 0
    frames_without_tracks = 0
    track_observations = 0
    invalid_box_observations = 0
    missing_id_observations = 0
    rows_written = 0
    unique_track_ids = set()

    start_time = time.perf_counter()

    print()
    print(
        f"=== {sequence} ==="
    )

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frames_processed += 1
            frame_number = (
                frames_processed
            )

            image_height, image_width = (
                frame.shape[:2]
            )

            try:
                result = pipeline.predict(
                    frame
                )
            except Exception as exc:
                prediction_failures += 1

                writer.writerow(
                    empty_frame_row(
                        sequence=sequence,
                        video_name=video_name,
                        frame_number=frame_number,
                        fps=fps,
                        image_width=image_width,
                        image_height=image_height,
                        status="PREDICTION_FAILURE",
                        failure_reason=str(
                            exc
                        ),
                    )
                )

                rows_written += 1
                continue

            tracks = list(
                result
            )

            tracks_in_frame = len(
                tracks
            )

            if tracks_in_frame == 0:
                frames_without_tracks += 1

                writer.writerow(
                    empty_frame_row(
                        sequence=sequence,
                        video_name=video_name,
                        frame_number=frame_number,
                        fps=fps,
                        image_width=image_width,
                        image_height=image_height,
                        status="NO_TRACKS",
                    )
                )

                rows_written += 1
            else:
                frames_with_tracks += 1
                track_observations += (
                    tracks_in_frame
                )

                for (
                    track_index,
                    track,
                ) in enumerate(
                    tracks,
                    start=1,
                ):
                    (
                        row,
                        valid_box,
                        missing_id,
                    ) = track_row(
                        sequence=sequence,
                        video_name=video_name,
                        frame_number=frame_number,
                        fps=fps,
                        image_width=image_width,
                        image_height=image_height,
                        track_index=track_index,
                        tracks_in_frame=tracks_in_frame,
                        track=track,
                    )

                    writer.writerow(
                        row
                    )

                    if not valid_box:
                        invalid_box_observations += 1

                    if missing_id:
                        missing_id_observations += 1
                    else:
                        unique_track_ids.add(
                            int(
                                row[
                                    "track_id"
                                ]
                            )
                        )

                    rows_written += 1

            if (
                frames_processed % 500 == 0
                or frames_processed
                == expected_frames
            ):
                print(
                    "Processed "
                    f"{frames_processed}/{expected_frames} frames"
                )

    finally:
        cap.release()
        pipeline.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if (
        frames_processed
        != expected_frames
    ):
        frame_read_failures = abs(
            expected_frames
            - frames_processed
        )

    failed_frames = (
        frame_read_failures
        + prediction_failures
    )

    print(
        f"Completed {sequence}: "
        f"{frames_processed} frames, "
        f"{track_observations} tracked observations"
    )

    return {
        "sequence":
            sequence,
        "source_video":
            video_name,
        "expected_frames":
            expected_frames,
        "frames_processed":
            frames_processed,
        "failed_frames":
            failed_frames,
        "frame_read_failures":
            frame_read_failures,
        "prediction_failures":
            prediction_failures,
        "frames_with_tracks":
            frames_with_tracks,
        "frames_without_tracks":
            frames_without_tracks,
        "track_observations":
            track_observations,
        "unique_track_ids":
            len(
                unique_track_ids
            ),
        "invalid_box_observations":
            invalid_box_observations,
        "missing_id_observations":
            missing_id_observations,
        "rows_written":
            rows_written,
        "source_fps":
            fps,
        "runtime_seconds":
            elapsed,
        "runtime_minutes":
            elapsed / 60.0,
    }


def generate_outputs(
    prepared: list[dict[str, Any]],
    staging_dir: Path,
) -> dict[str, Any]:
    staged_csv_path = (
        staging_dir
        / RESULTS_CSV_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_JSON_PATH.name
    )

    sequence_summaries = []
    total_rows = 0
    total_frames = 0
    total_failed_frames = 0
    total_track_observations = 0
    total_invalid_boxes = 0
    total_missing_ids = 0
    total_frames_with_tracks = 0
    total_frames_without_tracks = 0

    start_time = time.perf_counter()

    with staged_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for item in prepared:
            sequence_summary = (
                run_sequence(
                    item=item,
                    writer=writer,
                )
            )

            sequence_summaries.append(
                sequence_summary
            )

            total_rows += (
                sequence_summary[
                    "rows_written"
                ]
            )

            total_frames += (
                sequence_summary[
                    "frames_processed"
                ]
            )

            total_failed_frames += (
                sequence_summary[
                    "failed_frames"
                ]
            )

            total_track_observations += (
                sequence_summary[
                    "track_observations"
                ]
            )

            total_invalid_boxes += (
                sequence_summary[
                    "invalid_box_observations"
                ]
            )

            total_missing_ids += (
                sequence_summary[
                    "missing_id_observations"
                ]
            )

            total_frames_with_tracks += (
                sequence_summary[
                    "frames_with_tracks"
                ]
            )

            total_frames_without_tracks += (
                sequence_summary[
                    "frames_without_tracks"
                ]
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    overall_status = (
        "PASS"
        if (
            total_frames
            == EXPECTED_TOTAL_FRAMES
            and total_failed_frames == 0
        )
        else "FAIL"
    )

    summary = {
        "test_type":
            "isolated_physiotrack_face_tracking_component_execution",
        "purpose":
            (
                "Software execution evidence for the real PhysioTrack "
                "FaceAnalysis tracking path. This is not a tracking-accuracy benchmark."
            ),
        "dataset":
            "ECCV 2016 Music Video Face Tracking Dataset",
        "videos_expected":
            EXPECTED_VIDEO_COUNT,
        "videos_processed":
            len(
                sequence_summaries
            ),
        "expected_total_frames":
            EXPECTED_TOTAL_FRAMES,
        "frames_processed":
            total_frames,
        "failed_frames":
            total_failed_frames,
        "frames_with_tracks":
            total_frames_with_tracks,
        "frames_without_tracks":
            total_frames_without_tracks,
        "track_observations":
            total_track_observations,
        "invalid_box_observations":
            total_invalid_boxes,
        "missing_id_observations":
            total_missing_ids,
        "rows_written":
            total_rows,
        "device":
            "CPU",
        "detector_confidence":
            0.25,
        "detector_iou":
            0.45,
        "tracker":
            "OC-SORT",
        "enabled_component":
            "face_tracking",
        "required_dependency":
            "face_detection",
        "disabled_components": [
            "head_pose",
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
        ],
        "sequence_results":
            sequence_summaries,
        "runtime_seconds":
            elapsed,
        "runtime_minutes":
            elapsed / 60.0,
        "overall_status":
            overall_status,
    }

    with staged_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


def validate_staged_outputs(
    staging_dir: Path,
) -> dict[str, Any]:
    staged_csv_path = (
        staging_dir
        / RESULTS_CSV_PATH.name
    )

    staged_summary_path = (
        staging_dir
        / SUMMARY_JSON_PATH.name
    )

    if not staged_csv_path.is_file():
        raise RuntimeError(
            "Staged Face Tracking result CSV was not created."
        )

    if not staged_summary_path.is_file():
        raise RuntimeError(
            "Staged Face Tracking summary JSON was not created."
        )

    with staged_summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if summary.get(
        "videos_processed"
    ) != EXPECTED_VIDEO_COUNT:
        raise RuntimeError(
            "Staged summary video count is incorrect."
        )

    if summary.get(
        "frames_processed"
    ) != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(
            "Staged summary frame count is incorrect."
        )

    if summary.get(
        "failed_frames"
    ) != 0:
        raise RuntimeError(
            "Face Tracking component execution completed with frame failures; "
            "existing accepted outputs will not be replaced."
        )

    if summary.get(
        "overall_status"
    ) != "PASS":
        raise RuntimeError(
            "Staged Face Tracking component execution did not produce PASS status."
        )

    expected_sequence_frames = {
        item[
            "sequence"
        ]:
            item[
                "expected_frames"
            ]
        for item in summary[
            "sequence_results"
        ]
    }

    if set(
        expected_sequence_frames
    ) != {
        item[
            0
        ]
        for item in VIDEO_CONFIGS
    }:
        raise RuntimeError(
            "Staged summary sequence set is incorrect."
        )

    frame_keys = set()
    track_observation_rows = 0
    no_track_rows = 0
    invalid_box_rows = 0
    missing_id_rows = 0
    csv_rows = 0

    per_frame_track_indices = {}
    per_frame_track_counts = {}

    with staged_csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(
                "Staged Face Tracking result CSV schema is incorrect."
            )

        for row in reader:
            csv_rows += 1

            sequence = row[
                "sequence"
            ].strip()

            if sequence not in expected_sequence_frames:
                raise RuntimeError(
                    f"Unexpected sequence in result CSV: {sequence}"
                )

            frame_number = int(
                row[
                    "frame_number"
                ]
            )

            if not (
                1
                <= frame_number
                <= expected_sequence_frames[
                    sequence
                ]
            ):
                raise RuntimeError(
                    "Frame number is outside the expected sequence range."
                )

            frame_key = (
                sequence,
                frame_number,
            )

            frame_keys.add(
                frame_key
            )

            status = row[
                "status"
            ]

            if status in {
                "TRACKED",
                "TRACKED_INVALID_BOX",
                "TRACKED_MISSING_ID",
            }:
                track_observation_rows += 1

                track_index = int(
                    row[
                        "track_index"
                    ]
                )

                tracks_in_frame = int(
                    row[
                        "tracks_in_frame"
                    ]
                )

                if (
                    track_index < 1
                    or tracks_in_frame < 1
                    or track_index
                    > tracks_in_frame
                ):
                    raise RuntimeError(
                        "Tracked row has invalid track indexing."
                    )

                per_frame_track_indices.setdefault(
                    frame_key,
                    [],
                ).append(
                    track_index
                )

                per_frame_track_counts.setdefault(
                    frame_key,
                    set(),
                ).add(
                    tracks_in_frame
                )

                for field in (
                    "box_x1",
                    "box_y1",
                    "box_x2",
                    "box_y2",
                ):
                    if not finite_numeric(
                        row[
                            field
                        ]
                    ):
                        raise RuntimeError(
                            "Tracked row contains non-finite raw box coordinates."
                        )

                confidence_value = row[
                    "confidence"
                ].strip()

                if confidence_value:
                    if not finite_numeric(
                        confidence_value
                    ):
                        raise RuntimeError(
                            "Tracked row confidence is not finite."
                        )

                    confidence = float(
                        confidence_value
                    )

                    if not (
                        0.0
                        <= confidence
                        <= 1.0
                    ):
                        raise RuntimeError(
                            "Tracked row confidence is outside [0, 1]."
                        )

                x1 = float(
                    row[
                        "box_x1"
                    ]
                )
                y1 = float(
                    row[
                        "box_y1"
                    ]
                )
                x2 = float(
                    row[
                        "box_x2"
                    ]
                )
                y2 = float(
                    row[
                        "box_y2"
                    ]
                )

                if status == "TRACKED":
                    if not row[
                        "track_id"
                    ].strip():
                        raise RuntimeError(
                            "Valid tracked row has no tracking identifier."
                        )

                    for field in (
                        "box_width",
                        "box_height",
                        "box_area",
                    ):
                        if not finite_numeric(
                            row[
                                field
                            ]
                        ):
                            raise RuntimeError(
                                f"Valid tracked row contains invalid {field}."
                            )

                    width = float(
                        row[
                            "box_width"
                        ]
                    )
                    height = float(
                        row[
                            "box_height"
                        ]
                    )
                    area = float(
                        row[
                            "box_area"
                        ]
                    )

                    if (
                        width <= 0.0
                        or height <= 0.0
                    ):
                        raise RuntimeError(
                            "Valid tracked row has non-positive box dimensions."
                        )

                    if not math.isclose(
                        width,
                        x2 - x1,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Tracked row box width is inconsistent."
                        )

                    if not math.isclose(
                        height,
                        y2 - y1,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Tracked row box height is inconsistent."
                        )

                    if not math.isclose(
                        area,
                        width * height,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        raise RuntimeError(
                            "Tracked row box area is inconsistent."
                        )

                elif status == "TRACKED_INVALID_BOX":
                    invalid_box_rows += 1

                    if (
                        x2 > x1
                        and y2 > y1
                    ):
                        raise RuntimeError(
                            "Invalid-box row contains valid geometry."
                        )

                    for field in (
                        "box_width",
                        "box_height",
                        "box_area",
                    ):
                        if row[
                            field
                        ].strip():
                            raise RuntimeError(
                                "Invalid-box row contains derived geometry."
                            )

                else:
                    missing_id_rows += 1

                    if row[
                        "track_id"
                    ].strip():
                        raise RuntimeError(
                            "Missing-ID row contains a tracking identifier."
                        )

            elif status == "NO_TRACKS":
                no_track_rows += 1

                if int(
                    row[
                        "tracks_in_frame"
                    ]
                ) != 0:
                    raise RuntimeError(
                        "NO_TRACKS row has non-zero track count."
                    )

                if row[
                    "track_index"
                ].strip():
                    raise RuntimeError(
                        "NO_TRACKS row contains a track index."
                    )

                if row[
                    "track_id"
                ].strip():
                    raise RuntimeError(
                        "NO_TRACKS row contains a tracking identifier."
                    )

            else:
                raise RuntimeError(
                    f"Unexpected result status: {status}"
                )

    expected_frame_keys = {
        (
            sequence,
            frame_number,
        )
        for (
            sequence,
            expected_frames,
        ) in expected_sequence_frames.items()
        for frame_number in range(
            1,
            expected_frames + 1,
        )
    }

    if frame_keys != expected_frame_keys:
        raise RuntimeError(
            "Staged CSV does not account for every expected video frame."
        )

    for (
        frame_key,
        indices,
    ) in per_frame_track_indices.items():
        if len(
            per_frame_track_counts[
                frame_key
            ]
        ) != 1:
            raise RuntimeError(
                "Inconsistent tracks_in_frame values within a frame."
            )

        expected_count = next(
            iter(
                per_frame_track_counts[
                    frame_key
                ]
            )
        )

        if sorted(
            indices
        ) != list(
            range(
                1,
                expected_count + 1,
            )
        ):
            raise RuntimeError(
                "Track indices are not consecutive within a frame."
            )

    if csv_rows != summary.get(
        "rows_written"
    ):
        raise RuntimeError(
            "Staged CSV row count does not match the summary."
        )

    if track_observation_rows != summary.get(
        "track_observations"
    ):
        raise RuntimeError(
            "Tracked observation count does not match the summary."
        )

    if invalid_box_rows != summary.get(
        "invalid_box_observations"
    ):
        raise RuntimeError(
            "Invalid-box count does not match the summary."
        )

    if missing_id_rows != summary.get(
        "missing_id_observations"
    ):
        raise RuntimeError(
            "Missing-ID count does not match the summary."
        )

    if no_track_rows != summary.get(
        "frames_without_tracks"
    ):
        raise RuntimeError(
            "NO_TRACKS row count does not match the summary."
        )

    return summary


def replace_owned_outputs(
    staging_dir: Path,
) -> None:
    staged_paths = [
        (
            staging_dir
            / RESULTS_CSV_PATH.name,
            RESULTS_CSV_PATH,
        ),
        (
            staging_dir
            / SUMMARY_JSON_PATH.name,
            SUMMARY_JSON_PATH,
        ),
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    moved_backups = []
    installed_outputs = []

    try:
        for _, final_path in staged_paths:
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                moved_backups.append(
                    (
                        backup_path,
                        final_path,
                    )
                )

        for (
            staged_path,
            final_path,
        ) in staged_paths:
            os.replace(
                staged_path,
                final_path,
            )

            installed_outputs.append(
                final_path
            )

    except Exception:
        for final_path in installed_outputs:
            if final_path.exists():
                final_path.unlink()

        for (
            backup_path,
            final_path,
        ) in reversed(
            moved_backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main() -> None:
    print(
        "PhysioTrack isolated Face Tracking component execution"
    )

    print(
        "Preflight..."
    )

    prepared = preflight()

    print(
        "Preflight: PASS"
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_tracking_component_",
            dir=OUTPUT_DIR,
        )
    )

    try:
        summary = generate_outputs(
            prepared=prepared,
            staging_dir=staging_dir,
        )

        print()
        print(
            "Validating staged outputs..."
        )

        validated_summary = (
            validate_staged_outputs(
                staging_dir=staging_dir,
            )
        )

        if validated_summary != summary:
            raise RuntimeError(
                "In-memory and staged Face Tracking summaries differ."
            )

        replace_owned_outputs(
            staging_dir
        )

    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )

    print()
    print(
        "Finished."
    )

    print(
        "Videos processed:",
        summary[
            "videos_processed"
        ],
    )

    print(
        "Frames processed:",
        summary[
            "frames_processed"
        ],
    )

    print(
        "Failed frames:",
        summary[
            "failed_frames"
        ],
    )

    print(
        "Frames with tracks:",
        summary[
            "frames_with_tracks"
        ],
    )

    print(
        "Frames without tracks:",
        summary[
            "frames_without_tracks"
        ],
    )

    print(
        "Track observations:",
        summary[
            "track_observations"
        ],
    )

    print(
        "Invalid-box observations:",
        summary[
            "invalid_box_observations"
        ],
    )

    print(
        "Missing-ID observations:",
        summary[
            "missing_id_observations"
        ],
    )

    print(
        "Result rows:",
        summary[
            "rows_written"
        ],
    )

    print(
        "Runtime minutes:",
        f"{summary['runtime_minutes']:.2f}",
    )

    print(
        "Overall status:",
        summary[
            "overall_status"
        ],
    )

    print()
    print(
        "Saved:"
    )

    print(
        RESULTS_CSV_PATH
    )

    print(
        SUMMARY_JSON_PATH
    )


if __name__ == "__main__":
    main()
