from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig, GazeEstimator


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
    / "runtime_smoke"
)


MOTION_ZERO_TOLERANCE = 1e-12


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


def validate_mouth_motion_dependency() -> dict[str, str]:
    config = FaceAnalysisConfig(
        mouth=False,
        mouth_motion=True,
    )

    try:
        config.validate()
    except ValueError as exc:
        expected_message = (
            "mouth_motion requires mouth=True"
        )

        if expected_message not in str(exc):
            raise RuntimeError(
                "Unexpected mouth-motion dependency error: "
                f"{exc}"
            ) from exc

        print(
            "Mouth-motion dependency validation: PASS"
        )

        return {
            "case": "mouth_motion_requires_mouth",
            "status": "PASS",
            "expected_error": expected_message,
        }

    raise RuntimeError(
        "mouth_motion=True with mouth=False did not raise ValueError"
    )


def run_case(
    video_path: Path,
    name: str,
    gaze_enabled: bool,
    gaze_estimation_enabled: bool,
    mouth_enabled: bool,
    mouth_motion_enabled: bool,
) -> dict[str, int | str | bool]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
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
        mouth=mouth_enabled,
        mouth_motion=mouth_motion_enabled,
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
    mouth_available = 0
    mouth_motion_available = 0
    mouth_motion_numeric = 0
    mouth_motion_nonzero = 0
    first_motion_by_person = {}

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

                mouth = features.get(
                    "mouth"
                )

                mouth_motion = features.get(
                    "mouth_motion"
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

                if (
                    isinstance(mouth, dict)
                    and mouth.get(
                        "available",
                        False,
                    )
                ):
                    mouth_available += 1

                if not isinstance(
                    mouth_motion,
                    dict,
                ):
                    raise RuntimeError(
                        f"{name}: mouth_motion feature is missing"
                    )

                movement = mouth_motion.get(
                    "mouth_movement"
                )

                velocity = mouth_motion.get(
                    "mouth_velocity"
                )

                if mouth_motion.get(
                    "available",
                    False,
                ):
                    mouth_motion_available += 1

                    if (
                        movement is None
                        or velocity is None
                    ):
                        raise RuntimeError(
                            f"{name}: available mouth motion has missing values"
                        )

                    movement = float(
                        movement
                    )

                    velocity = float(
                        velocity
                    )

                    if (
                        not math.isfinite(
                            movement
                        )
                        or not math.isfinite(
                            velocity
                        )
                    ):
                        raise RuntimeError(
                            f"{name}: mouth motion produced non-finite values"
                        )

                    mouth_motion_numeric += 1

                    if (
                        abs(movement)
                        > MOTION_ZERO_TOLERANCE
                        or abs(velocity)
                        > MOTION_ZERO_TOLERANCE
                    ):
                        mouth_motion_nonzero += 1

                    person_id = face.id

                    if person_id not in first_motion_by_person:
                        first_motion_by_person[
                            person_id
                        ] = (
                            movement,
                            velocity,
                        )

                else:
                    if (
                        movement is not None
                        or velocity is not None
                    ):
                        raise RuntimeError(
                            f"{name}: unavailable mouth motion contains numerical values"
                        )

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

    if mouth_enabled:
        if mouth_available == 0:
            raise RuntimeError(
                f"{name}: mouth enabled but unavailable"
            )
    else:
        if mouth_available != 0:
            raise RuntimeError(
                f"{name}: mouth disabled but produced output"
            )

    if mouth_motion_enabled:
        if mouth_motion_available == 0:
            raise RuntimeError(
                f"{name}: mouth motion enabled but unavailable"
            )

        if (
            mouth_motion_numeric
            != mouth_motion_available
        ):
            raise RuntimeError(
                f"{name}: mouth-motion numerical accounting mismatch"
            )

        if not first_motion_by_person:
            raise RuntimeError(
                f"{name}: no mouth-motion initialization values observed"
            )

        for person_id, (
            movement,
            velocity,
        ) in first_motion_by_person.items():
            if (
                abs(movement)
                > MOTION_ZERO_TOLERANCE
                or abs(velocity)
                > MOTION_ZERO_TOLERANCE
            ):
                raise RuntimeError(
                    f"{name}: person {person_id} mouth-motion initialization "
                    "is not zero"
                )
    else:
        if mouth_motion_available != 0:
            raise RuntimeError(
                f"{name}: mouth motion disabled but produced available output"
            )

        if mouth_motion_numeric != 0:
            raise RuntimeError(
                f"{name}: mouth motion disabled but produced numerical output"
            )

    print(
        f"{name}: PASS | "
        f"frames={processed_frames} | "
        f"faces={faces_seen} | "
        f"old_gaze_available={old_gaze_available} | "
        f"gaze_estimation_available={new_gaze_available} | "
        f"mouth_available={mouth_available} | "
        f"mouth_motion_available={mouth_motion_available} | "
        f"mouth_motion_numeric={mouth_motion_numeric} | "
        f"mouth_motion_nonzero={mouth_motion_nonzero}"
    )

    return {
        "case": name,
        "gaze_enabled": gaze_enabled,
        "gaze_estimation_enabled": gaze_estimation_enabled,
        "mouth_enabled": mouth_enabled,
        "mouth_motion_enabled": mouth_motion_enabled,
        "processed_frames": processed_frames,
        "faces_seen": faces_seen,
        "old_gaze_available": old_gaze_available,
        "gaze_estimation_available": new_gaze_available,
        "mouth_available": mouth_available,
        "mouth_motion_available": mouth_motion_available,
        "mouth_motion_numeric": mouth_motion_numeric,
        "mouth_motion_nonzero": mouth_motion_nonzero,
        "mouth_motion_initialized_persons": len(
            first_motion_by_person
        ),
        "status": "PASS",
    }


def failed_case_result(
    name: str,
    gaze_enabled: bool,
    gaze_estimation_enabled: bool,
    mouth_enabled: bool,
    mouth_motion_enabled: bool,
    reason: str,
) -> dict[str, int | str | bool | None]:
    return {
        "case":
            name,
        "gaze_enabled":
            gaze_enabled,
        "gaze_estimation_enabled":
            gaze_estimation_enabled,
        "mouth_enabled":
            mouth_enabled,
        "mouth_motion_enabled":
            mouth_motion_enabled,
        "processed_frames":
            None,
        "faces_seen":
            None,
        "old_gaze_available":
            None,
        "gaze_estimation_available":
            None,
        "mouth_available":
            None,
        "mouth_motion_available":
            None,
        "mouth_motion_numeric":
            None,
        "mouth_motion_nonzero":
            None,
        "mouth_motion_initialized_persons":
            None,
        "failure_reason":
            reason,
        "status":
            "FAIL",
    }


def clean_output_directory() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    video_paths = get_video_paths()

    clean_output_directory()

    print(
        "Import GazeEstimator: PASS"
    )

    dependency_result = (
        validate_mouth_motion_dependency()
    )

    cases = [
        (
            "both_disabled",
            False,
            False,
            True,
            True,
        ),
        (
            "old_gaze_only",
            True,
            False,
            True,
            True,
        ),
        (
            "gaze_estimation_only",
            False,
            True,
            True,
            True,
        ),
        (
            "both_enabled",
            True,
            True,
            True,
            True,
        ),
        (
            "mouth_without_motion",
            False,
            False,
            True,
            False,
        ),
        (
            "mouth_and_motion_disabled",
            False,
            False,
            False,
            False,
        ),
    ]

    video_results = []

    for video_path in video_paths:
        print()
        print("=" * 72)
        print(f"Video: {video_path}")
        print("=" * 72)

        case_results = []

        for (
            name,
            gaze_enabled,
            gaze_estimation_enabled,
            mouth_enabled,
            mouth_motion_enabled,
        ) in cases:
            try:
                case_result = run_case(
                    video_path=video_path,
                    name=name,
                    gaze_enabled=gaze_enabled,
                    gaze_estimation_enabled=
                        gaze_estimation_enabled,
                    mouth_enabled=mouth_enabled,
                    mouth_motion_enabled=
                        mouth_motion_enabled,
                )

            except Exception as exc:
                case_result = failed_case_result(
                    name=name,
                    gaze_enabled=gaze_enabled,
                    gaze_estimation_enabled=
                        gaze_estimation_enabled,
                    mouth_enabled=mouth_enabled,
                    mouth_motion_enabled=
                        mouth_motion_enabled,
                    reason=str(exc),
                )

                print(
                    f"{name}: FAIL | "
                    f"reason={exc}"
                )

            case_results.append(
                case_result
            )

        video_pass = all(
            result[
                "status"
            ]
            == "PASS"
            for result in case_results
        )

        video_results.append(
            {
                "video":
                    video_label(
                        video_path
                    ),
                "cases":
                    case_results,
                "status":
                    (
                        "PASS"
                        if video_pass
                        else "FAIL"
                    ),
            }
        )

    overall_pass = (
        dependency_result[
            "status"
        ]
        == "PASS"
        and all(
            result[
                "status"
            ]
            == "PASS"
            for result in video_results
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
                "test_type":
                    "runtime_smoke",
                "mouth_motion_dependency":
                    dependency_result,
                "videos":
                    video_results,
                "video_count":
                    len(video_results),
                "passed_videos":
                    sum(
                        result[
                            "status"
                        ]
                        == "PASS"
                        for result in video_results
                    ),
                "failed_videos":
                    sum(
                        result[
                            "status"
                        ]
                        != "PASS"
                        for result in video_results
                    ),
                "overall_status":
                    (
                        "PASS"
                        if overall_pass
                        else "FAIL"
                    ),
            },
            file,
            indent=2,
        )

    print()
    print(
        "Runtime smoke test:",
        (
            "PASS"
            if overall_pass
            else "FAIL"
        ),
    )

    print(
        "Videos tested:",
        len(video_results),
    )

    print(f"Saved: {summary_path}")

    if not overall_pass:
        raise RuntimeError(
            "Runtime smoke test completed with "
            "one or more failed cases."
        )


if __name__ == "__main__":
    main()
