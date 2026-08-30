from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig, GazeEstimator


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = SCRIPT_DIR / "test_data"

VIDEO_PATH = TEST_DATA_DIR / "face_blink_pose.mp4"

VIDEO_LABEL = str(
    VIDEO_PATH.relative_to(SCRIPT_DIR)
).replace("\\", "/")

OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "runtime_smoke"
)


def run_case(
    name: str,
    gaze_enabled: bool,
    gaze_estimation_enabled: bool,
) -> dict[str, int | str | bool]:
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    reported_frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Invalid FPS: {fps}"
        )

    config = FaceAnalysisConfig(
        tracking=True,
        head_pose=True,
        landmarks=True,
        quality=True,
        eyes=True,
        blink=True,
        gaze=gaze_enabled,
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

    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )

    processed_frames = 0
    faces_seen = 0
    old_gaze_available = 0
    new_gaze_available = 0

    try:
        for _ in range(5):
            ok, frame = capture.read()

            if not ok:
                break

            result = pipeline.predict(frame)

            for face in result:
                faces_seen += 1

                features = (
                    face.face_features
                    if face.face_features is not None
                    else {}
                )

                old_gaze = features.get(
                    "gaze"
                )

                new_gaze = features.get(
                    "gaze_estimation"
                )

                if (
                    isinstance(old_gaze, dict)
                    and old_gaze.get(
                        "available",
                        False,
                    )
                ):
                    old_gaze_available += 1

                if (
                    isinstance(new_gaze, dict)
                    and new_gaze.get(
                        "available",
                        False,
                    )
                ):
                    new_gaze_available += 1

            processed_frames += 1

    finally:
        capture.release()
        pipeline.close()

    expected_smoke_frames = (
        min(5, reported_frame_count)
        if reported_frame_count > 0
        else processed_frames
    )

    if processed_frames == 0:
        raise RuntimeError(
            f"{name}: no frames processed"
        )

    if (
        reported_frame_count > 0
        and processed_frames
        != expected_smoke_frames
    ):
        raise RuntimeError(
            f"{name}: incomplete smoke-test frame read"
        )

    if faces_seen == 0:
        raise RuntimeError(
            f"{name}: no faces detected"
        )

    if gaze_enabled:
        if old_gaze_available == 0:
            raise RuntimeError(
                f"{name}: old gaze enabled but unavailable"
            )
    else:
        if old_gaze_available != 0:
            raise RuntimeError(
                f"{name}: old gaze disabled but produced output"
            )

    if gaze_estimation_enabled:
        if new_gaze_available == 0:
            raise RuntimeError(
                f"{name}: gaze estimation enabled but unavailable"
            )
    else:
        if new_gaze_available != 0:
            raise RuntimeError(
                f"{name}: gaze estimation disabled but produced output"
            )

    print(
        f"{name}: PASS | "
        f"frames={processed_frames} | "
        f"faces={faces_seen} | "
        f"old_gaze_available={old_gaze_available} | "
        f"gaze_estimation_available={new_gaze_available}"
    )

    return {
        "case": name,
        "gaze_enabled": gaze_enabled,
        "gaze_estimation_enabled": gaze_estimation_enabled,
        "processed_frames": processed_frames,
        "faces_seen": faces_seen,
        "old_gaze_available": old_gaze_available,
        "gaze_estimation_available": new_gaze_available,
        "status": "PASS",
    }


def clean_output_directory() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    clean_output_directory()

    print(
        "Import GazeEstimator: PASS"
    )

    cases = [
        (
            "both_disabled",
            False,
            False,
        ),
        (
            "old_gaze_only",
            True,
            False,
        ),
        (
            "gaze_estimation_only",
            False,
            True,
        ),
        (
            "both_enabled",
            True,
            True,
        ),
    ]

    case_results = []

    for (
        name,
        gaze_enabled,
        gaze_estimation_enabled,
    ) in cases:
        case_results.append(
            run_case(
                name=name,
                gaze_enabled=gaze_enabled,
                gaze_estimation_enabled=
                    gaze_estimation_enabled,
            )
        )

    summary_path = (
        OUTPUT_DIR
        / "runtime_smoke_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "test_type": "runtime_smoke",
                "video": VIDEO_LABEL,
                "cases": case_results,
                "overall_status": "PASS",
            },
            file,
            indent=2,
        )

    print()
    print(
        "Runtime smoke test: PASS"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()