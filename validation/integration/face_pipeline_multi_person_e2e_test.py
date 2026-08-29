from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.export import FaceResultExporter


VIDEO_PATH = Path(
    r"C:\Users\xx901\Documents\PhysioTrack_Thesis"
    r"\physiotrack\media_for_test\multi_person2.mp4"
)

OUTPUT_DIR = Path(
    r"C:\Users\xx901\Documents\PhysioTrack_Thesis"
    r"\physiotrack\validation\integration\results_multi_person"
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


def main() -> None:
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

            frame_face_counts.append(
                {
                    "frame_index": processed_frames,
                    "timestamp": timestamp,
                    "face_count": face_count,
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
                        and not is_available
                    ):
                        gaze_estimation_failures.append(
                            {
                                "frame_index": processed_frames,
                                "timestamp": timestamp,
                                "person_id": person_id,
                                "box": instance.box,
                                "confidence": instance.confidence,
                                "association_iou": feature.get(
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
                    f"Processed {processed_frames} frames"
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

    exporter.save_json(
        frame_records,
        frame_json_path,
    )

    exporter.save_csv(
        frame_records,
        frame_csv_path,
    )

    exporter.save_json(
        window_records,
        window_json_path,
    )

    exporter.save_csv(
        window_records,
        window_csv_path,
    )

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
                "person_id": person_id,
                "module": "head_pose",
                "available": (
                    head_pose_available.get(
                        person_id,
                        0,
                    )
                ),
                "missing": (
                    head_pose_missing.get(
                        person_id,
                        0,
                    )
                ),
                "total_person_frames": person_total,
                "availability_percent": (
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
                    "person_id": person_id,
                    "module": feature_name,
                    "available": available,
                    "missing": missing,
                    "total_person_frames": person_total,
                    "availability_percent": (
                        100.0
                        * available
                        / person_total
                        if person_total
                        else 0.0
                    ),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_csv_path,
        index=False,
    )

    face_counts_df = pd.DataFrame(
        frame_face_counts
    )

    face_counts_df.to_csv(
        face_counts_csv_path,
        index=False,
    )

    gaze_failures_df = pd.DataFrame(
        gaze_estimation_failures
    )

    gaze_failures_df.to_csv(
        gaze_failures_csv_path,
        index=False,
    )

    print()
    print(
        "=== Full Multi-Person End-to-End Summary ==="
    )

    print(
        f"Video FPS: {fps}"
    )

    print(
        f"Reported frame count: {reported_frame_count}"
    )

    print(
        f"Processed frames: {processed_frames}"
    )

    print(
        f"Total face instances: {total_faces}"
    )

    print(
        f"Person IDs: {person_ids}"
    )

    print(
        f"Frame records: {len(frame_records)}"
    )

    print(
        f"Window records: {len(window_records)}"
    )

    print(
        "Gaze estimation failures: "
        f"{len(gaze_estimation_failures)}"
    )

    print()
    print(
        "=== Person Frame Counts ==="
    )

    for person_id in person_ids:
        print(
            f"ID {person_id}: "
            f"{person_frame_counts[person_id]}"
        )

    print()
    print(
        "=== Module Availability Summary ==="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "=== Frames With Unexpected Face Count ==="
    )

    unexpected_face_counts = (
        face_counts_df[
            face_counts_df[
                "face_count"
            ] != 2
        ]
    )

    if len(
        unexpected_face_counts
    ) == 0:
        print(
            "None"
        )
    else:
        print(
            unexpected_face_counts.to_string(
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
    ]:
        print(
            path
        )

    if processed_frames <= 0:
        raise RuntimeError(
            "No video frames were processed."
        )

    if (
        reported_frame_count > 0
        and processed_frames
        != reported_frame_count
    ):
        raise RuntimeError(
            "Processed frame count does not match "
            "the video-reported frame count."
        )

    if person_ids != [
        1,
        2,
    ]:
        raise RuntimeError(
            f"Unexpected person IDs: {person_ids}"
        )

    if len(
        unexpected_face_counts
    ) != 0:
        raise RuntimeError(
            "The video did not contain exactly "
            "two tracked face instances in every frame."
        )

    for person_id in person_ids:
        if (
            person_frame_counts[
                person_id
            ]
            != processed_frames
        ):
            raise RuntimeError(
                f"ID {person_id} was not present "
                "in every processed frame."
            )

    for person_id in person_ids:
        if (
            head_pose_available.get(
                person_id,
                0,
            )
            != processed_frames
        ):
            raise RuntimeError(
                f"Head pose was not available "
                f"for all frames for ID {person_id}."
            )

    for feature_name in FEATURE_NAMES:
        for person_id in person_ids:
            available = (
                feature_available_counts[
                    feature_name
                ].get(
                    person_id,
                    0,
                )
            )

            if (
                available
                != processed_frames
            ):
                raise RuntimeError(
                    f"{feature_name} was not "
                    f"available for all frames "
                    f"for ID {person_id}."
                )

    expected_records = (
        processed_frames
        * 2
    )

    if (
        len(frame_records)
        != expected_records
    ):
        raise RuntimeError(
            "Unexpected number of frame records."
        )

    if (
        len(window_records)
        != expected_records
    ):
        raise RuntimeError(
            "Unexpected number of window records."
        )

    print()
    print(
        "Full multi-person end-to-end "
        "integration and export test: PASS"
    )


if __name__ == "__main__":
    main()