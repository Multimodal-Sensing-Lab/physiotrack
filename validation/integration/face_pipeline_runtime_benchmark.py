from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig


SCRIPT_DIR = Path(__file__).resolve().parent

TEST_SCENARIOS = {
    "single_person":
        SCRIPT_DIR
        / "test_data"
        / "single_person",
    "multi_person":
        SCRIPT_DIR
        / "test_data"
        / "multi_person",
    "whole_project":
        SCRIPT_DIR
        / "test_data"
        / "whole_project",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}

FINAL_OUTPUT_DIR = (
    SCRIPT_DIR
    / "results"
    / "runtime_performance"
)

EXPECTED_OUTPUT_FILES = (
    "face_pipeline_runtime_results.csv",
    "face_pipeline_runtime_frame_times.csv",
    "face_pipeline_runtime_summary.json",
)

RESULT_FIELDS = [
    "scenario",
    "video",
    "resolution",
    "input_fps",
    "reported_frames",
    "frames_read",
    "successful_frames",
    "failed_frames",
    "total_faces",
    "pipeline_initialization_seconds",
    "wall_seconds",
    "processing_seconds",
    "effective_processing_fps",
    "successful_processing_fps",
    "wall_fps",
    "mean_ms_per_frame",
    "median_ms_per_frame",
    "p90_ms_per_frame",
    "p95_ms_per_frame",
    "max_ms_per_frame",
    "first_frame_ms",
    "real_time_factor",
    "dataset_read_only_verified",
    "status",
    "failure_reason",
]

FRAME_FIELDS = [
    "scenario",
    "video",
    "frame_index",
    "timestamp_seconds",
    "processing_seconds",
    "processing_ms",
    "faces_detected",
    "status",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure full PhysioTrack face-pipeline runtime performance "
            "on the accepted integration videos."
        )
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate inputs and the final full-pipeline configuration "
            "without running the benchmark."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process at most N frames from each discovered video. "
            "Smoke-test outputs are validated but do not replace final "
            "runtime-performance results."
        ),
    )

    args = parser.parse_args()

    if (
        args.smoke_test is not None
        and args.smoke_test <= 0
    ):
        parser.error(
            "--smoke-test must be a positive integer"
        )

    return args


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


def video_label(
    video_path: Path,
) -> str:
    return str(
        video_path.relative_to(SCRIPT_DIR)
    ).replace("\\", "/")


def discover_videos() -> list[tuple[str, Path]]:
    discovered = []

    for scenario, directory in TEST_SCENARIOS.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Required runtime test-data directory not found: {directory}"
            )

        video_paths = sorted(
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
            )
        )

        if not video_paths:
            raise FileNotFoundError(
                f"No supported runtime videos found in: {directory}"
            )

        for video_path in video_paths:
            discovered.append(
                (
                    scenario,
                    video_path,
                )
            )

    return discovered


def input_inventory(
    discovered: list[tuple[str, Path]],
) -> dict[str, dict[str, int]]:
    inventory = {}

    for _, video_path in discovered:
        stat = video_path.stat()
        inventory[
            video_label(
                video_path
            )
        ] = {
            "size_bytes":
                int(
                    stat.st_size
                ),
            "mtime_ns":
                int(
                    stat.st_mtime_ns
                ),
        }

    return inventory


def preflight_video(
    scenario: str,
    video_path: Path,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open runtime benchmark video: {video_path}"
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

        if (
            not math.isfinite(
                fps
            )
            or fps <= 0.0
        ):
            raise RuntimeError(
                f"Invalid FPS for runtime benchmark video: {video_path}"
            )

        if (
            width <= 0
            or height <= 0
        ):
            raise RuntimeError(
                f"Invalid video resolution during preflight: {video_path}"
            )

        ok, frame = capture.read()

        if (
            not ok
            or frame is None
        ):
            raise RuntimeError(
                f"Could not read first frame during preflight: {video_path}"
            )

        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):
            raise RuntimeError(
                f"Video metadata/frame resolution mismatch: {video_path}"
            )

    finally:
        capture.release()

    return {
        "scenario":
            scenario,
        "video":
            video_label(
                video_path
            ),
        "input_fps":
            fps,
        "reported_frames":
            reported_frames,
        "width":
            width,
        "height":
            height,
    }


def preflight_validation() -> tuple[
    list[tuple[str, Path]],
    list[dict[str, Any]],
]:
    discovered = discover_videos()

    preflight_records = [
        preflight_video(
            scenario,
            video_path,
        )
        for scenario, video_path in discovered
    ]

    make_config()

    results_root = FINAL_OUTPUT_DIR.parent

    if (
        results_root.exists()
        and not results_root.is_dir()
    ):
        raise RuntimeError(
            f"Integration results root is not a directory: {results_root}"
        )

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        discovered,
        preflight_records,
    )


def percentile(
    values: list[float],
    q: float,
) -> float | None:
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def finite_or_none(
    value: float | None,
) -> float | None:
    if value is None:
        return None

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        return None

    return value


def collect_package_versions() -> dict[str, str | None]:
    package_names = [
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "onnxruntime",
        "mediapipe",
        "torch",
        "torchvision",
        "ptgaze",
        "emotiefflib",
        "ultralytics",
    ]

    versions = {}

    for package_name in package_names:
        try:
            versions[
                package_name
            ] = importlib.metadata.version(
                package_name
            )
        except importlib.metadata.PackageNotFoundError:
            versions[
                package_name
            ] = None

    versions[
        "opencv_runtime"
    ] = cv2.__version__
    versions[
        "numpy_runtime"
    ] = np.__version__

    return versions


def collect_environment() -> dict[str, Any]:
    memory_gib = None

    try:
        import psutil

        memory_gib = (
            psutil.virtual_memory().total
            / (1024 ** 3)
        )
    except Exception:
        memory_gib = None

    gpu = {
        "cuda_available":
            None,
        "cuda_device_count":
            None,
        "device_names":
            [],
    }

    try:
        import torch

        cuda_available = bool(
            torch.cuda.is_available()
        )
        gpu[
            "cuda_available"
        ] = cuda_available

        if cuda_available:
            device_count = int(
                torch.cuda.device_count()
            )
            gpu[
                "cuda_device_count"
            ] = device_count
            gpu[
                "device_names"
            ] = [
                str(
                    torch.cuda.get_device_name(
                        index
                    )
                )
                for index in range(
                    device_count
                )
            ]
        else:
            gpu[
                "cuda_device_count"
            ] = 0

    except Exception:
        pass

    return {
        "platform":
            platform.platform(),
        "system":
            platform.system(),
        "release":
            platform.release(),
        "machine":
            platform.machine(),
        "processor":
            platform.processor(),
        "python_version":
            platform.python_version(),
        "python_executable":
            Path(
                sys.executable
            ).name,
        "logical_cpu_count":
            os.cpu_count(),
        "memory_gib":
            finite_or_none(
                memory_gib
            ),
        "gpu":
            gpu,
        "packages":
            collect_package_versions(),
    }


def result_failure(
    scenario: str,
    video_path: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "scenario":
            scenario,
        "video":
            video_label(
                video_path
            ),
        "resolution":
            None,
        "input_fps":
            None,
        "reported_frames":
            None,
        "frames_read":
            0,
        "successful_frames":
            0,
        "failed_frames":
            0,
        "total_faces":
            0,
        "pipeline_initialization_seconds":
            None,
        "wall_seconds":
            None,
        "processing_seconds":
            None,
        "effective_processing_fps":
            None,
        "successful_processing_fps":
            None,
        "wall_fps":
            None,
        "mean_ms_per_frame":
            None,
        "median_ms_per_frame":
            None,
        "p90_ms_per_frame":
            None,
        "p95_ms_per_frame":
            None,
        "max_ms_per_frame":
            None,
        "first_frame_ms":
            None,
        "real_time_factor":
            None,
        "dataset_read_only_verified":
            False,
        "status":
            "VIDEO_EXECUTION_FAILED",
        "failure_reason":
            reason,
    }


def run_video(
    scenario: str,
    video_path: Path,
    smoke_limit: int | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open runtime benchmark video: {video_path}"
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

    if (
        not math.isfinite(
            fps
        )
        or fps <= 0.0
    ):
        capture.release()

        raise RuntimeError(
            f"Invalid video FPS: {video_path}"
        )

    config = make_config()

    initialization_start = time.perf_counter()
    pipeline = FaceAnalysis(
        config=config,
        fps=fps,
    )
    initialization_seconds = (
        time.perf_counter()
        - initialization_start
    )

    frame_records = []
    frame_times_ms = []

    frames_read = 0
    successful_frames = 0
    failed_frames = 0
    total_faces = 0

    processing_seconds = 0.0

    wall_start = time.perf_counter()

    try:
        while True:
            if (
                smoke_limit is not None
                and frames_read >= smoke_limit
            ):
                break

            ok, frame = capture.read()

            if not ok:
                break

            current_index = frames_read
            timestamp_seconds = (
                current_index
                / fps
            )

            process_start = time.perf_counter()

            try:
                faces = pipeline.predict(
                    frame
                )
                status = "PROCESSED"
                failure_reason = ""
                faces_detected = len(
                    faces
                )
                successful_frames += 1
                total_faces += faces_detected

            except Exception as exc:
                status = "PROCESSING_FAILED"
                failure_reason = (
                    f"{type(exc).__name__}: {exc}"
                )
                faces_detected = 0
                failed_frames += 1

            process_elapsed = (
                time.perf_counter()
                - process_start
            )
            processing_seconds += process_elapsed

            process_ms = (
                process_elapsed
                * 1000.0
            )
            frame_times_ms.append(
                process_ms
            )

            frame_records.append(
                {
                    "scenario":
                        scenario,
                    "video":
                        video_label(
                            video_path
                        ),
                    "frame_index":
                        current_index,
                    "timestamp_seconds":
                        timestamp_seconds,
                    "processing_seconds":
                        process_elapsed,
                    "processing_ms":
                        process_ms,
                    "faces_detected":
                        faces_detected,
                    "status":
                        status,
                    "failure_reason":
                        failure_reason,
                }
            )

            frames_read += 1

    finally:
        wall_seconds = (
            time.perf_counter()
            - wall_start
        )
        capture.release()
        pipeline.close()

    if frames_read == 0:
        raise RuntimeError(
            f"No frames were read from runtime benchmark video: {video_path}"
        )

    if (
        smoke_limit is None
        and reported_frames > 0
        and frames_read != reported_frames
    ):
        raise RuntimeError(
            "Full runtime benchmark did not read the complete reported "
            f"video frame count for {video_path}: "
            f"{frames_read} != {reported_frames}"
        )

    mean_ms = float(
        statistics.fmean(
            frame_times_ms
        )
    )
    median_ms = float(
        statistics.median(
            frame_times_ms
        )
    )

    effective_fps = (
        frames_read
        / processing_seconds
        if processing_seconds > 0.0
        else None
    )
    successful_fps = (
        successful_frames
        / processing_seconds
        if processing_seconds > 0.0
        else None
    )
    wall_fps = (
        frames_read
        / wall_seconds
        if wall_seconds > 0.0
        else None
    )
    real_time_factor = (
        effective_fps
        / fps
        if (
            effective_fps is not None
            and fps > 0.0
        )
        else None
    )

    status = (
        "PASS"
        if failed_frames == 0
        else "PASS_WITH_FRAME_FAILURES"
    )

    result = {
        "scenario":
            scenario,
        "video":
            video_label(
                video_path
            ),
        "resolution":
            f"{width}x{height}",
        "input_fps":
            fps,
        "reported_frames":
            reported_frames,
        "frames_read":
            frames_read,
        "successful_frames":
            successful_frames,
        "failed_frames":
            failed_frames,
        "total_faces":
            total_faces,
        "pipeline_initialization_seconds":
            initialization_seconds,
        "wall_seconds":
            wall_seconds,
        "processing_seconds":
            processing_seconds,
        "effective_processing_fps":
            finite_or_none(
                effective_fps
            ),
        "successful_processing_fps":
            finite_or_none(
                successful_fps
            ),
        "wall_fps":
            finite_or_none(
                wall_fps
            ),
        "mean_ms_per_frame":
            mean_ms,
        "median_ms_per_frame":
            median_ms,
        "p90_ms_per_frame":
            percentile(
                frame_times_ms,
                90.0,
            ),
        "p95_ms_per_frame":
            percentile(
                frame_times_ms,
                95.0,
            ),
        "max_ms_per_frame":
            max(
                frame_times_ms
            ),
        "first_frame_ms":
            frame_times_ms[
                0
            ],
        "real_time_factor":
            finite_or_none(
                real_time_factor
            ),
        "dataset_read_only_verified":
            True,
        "status":
            status,
        "failure_reason":
            "",
    }

    return (
        result,
        frame_records,
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
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


def build_summary(
    preflight_records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    smoke_limit: int | None,
    inventory_unchanged: bool,
) -> dict[str, Any]:
    successful_video_results = [
        row
        for row in results
        if row[
            "status"
        ] in {
            "PASS",
            "PASS_WITH_FRAME_FAILURES",
        }
    ]

    total_frames = sum(
        int(
            row[
                "frames_read"
            ]
        )
        for row in successful_video_results
    )
    total_successful_frames = sum(
        int(
            row[
                "successful_frames"
            ]
        )
        for row in successful_video_results
    )
    total_failed_frames = sum(
        int(
            row[
                "failed_frames"
            ]
        )
        for row in successful_video_results
    )
    total_faces = sum(
        int(
            row[
                "total_faces"
            ]
        )
        for row in successful_video_results
    )
    total_processing_seconds = sum(
        float(
            row[
                "processing_seconds"
            ]
        )
        for row in successful_video_results
        if row[
            "processing_seconds"
        ] is not None
    )
    total_wall_seconds = sum(
        float(
            row[
                "wall_seconds"
            ]
        )
        for row in successful_video_results
        if row[
            "wall_seconds"
        ] is not None
    )

    processing_ms = [
        float(
            row[
                "processing_ms"
            ]
        )
        for row in frame_records
    ]

    aggregate_effective_fps = (
        total_frames
        / total_processing_seconds
        if total_processing_seconds > 0.0
        else None
    )
    aggregate_wall_fps = (
        total_frames
        / total_wall_seconds
        if total_wall_seconds > 0.0
        else None
    )

    execution_failures = sum(
        1
        for row in results
        if row[
            "status"
        ] == "VIDEO_EXECUTION_FAILED"
    )

    full_run = (
        smoke_limit is None
    )

    return {
        "test":
            "face_pipeline_runtime_performance",
        "evidence_type":
            "system_runtime_performance",
        "scientific_accuracy_evidence":
            False,
        "timing_scope": {
            "pipeline_processing":
                (
                    "Per-frame time around FaceAnalysis.predict(frame). "
                    "Video decoding and output-file writing are excluded "
                    "from processing_seconds."
                ),
            "wall_time":
                (
                    "Wall-clock time for the video processing loop, including "
                    "video-frame reads and FaceAnalysis.predict calls. Final "
                    "CSV/JSON writing is excluded."
                ),
            "initialization":
                (
                    "FaceAnalysis construction is measured separately and is "
                    "not included in processing_seconds."
                ),
        },
        "configuration": {
            "active_components": [
                "face_detection",
                "tracking",
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
            "gaze_estimation_mode":
                "eth-xgaze",
            "gaze_estimation_min_iou":
                0.10,
        },
        "run_mode": {
            "full_run":
                full_run,
            "smoke_test_frames_per_video":
                smoke_limit,
        },
        "preflight":
            preflight_records,
        "execution": {
            "videos":
                len(
                    results
                ),
            "video_execution_failures":
                execution_failures,
            "frames_read":
                total_frames,
            "successful_frames":
                total_successful_frames,
            "failed_frames":
                total_failed_frames,
            "total_faces":
                total_faces,
            "processing_seconds":
                total_processing_seconds,
            "wall_seconds":
                total_wall_seconds,
            "effective_processing_fps":
                finite_or_none(
                    aggregate_effective_fps
                ),
            "wall_fps":
                finite_or_none(
                    aggregate_wall_fps
                ),
            "mean_ms_per_frame":
                (
                    float(
                        statistics.fmean(
                            processing_ms
                        )
                    )
                    if processing_ms
                    else None
                ),
            "median_ms_per_frame":
                (
                    float(
                        statistics.median(
                            processing_ms
                        )
                    )
                    if processing_ms
                    else None
                ),
            "p90_ms_per_frame":
                percentile(
                    processing_ms,
                    90.0,
                ),
            "p95_ms_per_frame":
                percentile(
                    processing_ms,
                    95.0,
                ),
        },
        "dataset": {
            "source":
                "validation/integration/test_data",
            "read_only_verified":
                inventory_unchanged,
        },
        "environment":
            collect_environment(),
        "interpretation":
            (
                "This package measures computational processing performance "
                "of the final full PhysioTrack face-analysis configuration. "
                "It is not a predictive-accuracy benchmark. Per-video runtime "
                "results must be interpreted with their native resolution, "
                "source FPS, face count, and scenario because the accepted "
                "integration videos have different workloads."
            ),
        "status":
            (
                "PASS"
                if (
                    execution_failures == 0
                    and total_failed_frames == 0
                    and inventory_unchanged
                )
                else "PASS_WITH_FAILURES"
            ),
    }


def validate_staged_outputs(
    staging_dir: Path,
) -> None:
    for filename in EXPECTED_OUTPUT_FILES:
        path = staging_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Expected staged runtime output was not created: {path}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Staged runtime output is empty: {path}"
            )

    results_path = (
        staging_dir
        / "face_pipeline_runtime_results.csv"
    )
    frames_path = (
        staging_dir
        / "face_pipeline_runtime_frame_times.csv"
    )
    summary_path = (
        staging_dir
        / "face_pipeline_runtime_summary.json"
    )

    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        result_rows = list(
            csv.DictReader(
                file
            )
        )

    with frames_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        frame_rows = list(
            csv.DictReader(
                file
            )
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if not result_rows:
        raise RuntimeError(
            "Runtime result CSV contains no video rows"
        )

    if not frame_rows:
        raise RuntimeError(
            "Runtime frame-time CSV contains no frame rows"
        )

    if (
        summary.get(
            "test"
        )
        != "face_pipeline_runtime_performance"
    ):
        raise RuntimeError(
            "Unexpected runtime summary test identifier"
        )

    result_frame_total = sum(
        int(
            row[
                "frames_read"
            ]
        )
        for row in result_rows
        if row[
            "status"
        ] in {
            "PASS",
            "PASS_WITH_FRAME_FAILURES",
        }
    )

    if result_frame_total != len(
        frame_rows
    ):
        raise RuntimeError(
            "Runtime per-video/frame-time row accounting mismatch"
        )

    summary_execution = summary.get(
        "execution",
        {},
    )

    if int(
        summary_execution.get(
            "frames_read",
            -1,
        )
    ) != len(
        frame_rows
    ):
        raise RuntimeError(
            "Runtime summary/frame-time row accounting mismatch"
        )

    for row in frame_rows:
        processing_ms = float(
            row[
                "processing_ms"
            ]
        )

        if (
            not math.isfinite(
                processing_ms
            )
            or processing_ms < 0.0
        ):
            raise RuntimeError(
                "Runtime frame-time CSV contains invalid processing_ms"
            )


def commit_staged_output_directory(
    staging_dir: Path,
) -> None:
    results_root = FINAL_OUTPUT_DIR.parent
    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{FINAL_OUTPUT_DIR.name}_backup_",
            dir=results_root,
        )
    )
    backup_dir.rmdir()

    final_backed_up = False
    staging_promoted = False

    try:
        if FINAL_OUTPUT_DIR.exists():
            os.replace(
                FINAL_OUTPUT_DIR,
                backup_dir,
            )
            final_backed_up = True

        os.replace(
            staging_dir,
            FINAL_OUTPUT_DIR,
        )
        staging_promoted = True

    except Exception:
        if (
            staging_promoted
            and FINAL_OUTPUT_DIR.exists()
        ):
            shutil.rmtree(
                FINAL_OUTPUT_DIR,
                ignore_errors=True,
            )

        if (
            final_backed_up
            and backup_dir.exists()
        ):
            os.replace(
                backup_dir,
                FINAL_OUTPUT_DIR,
            )

        raise

    if backup_dir.exists():
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def run_benchmark(
    smoke_limit: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    (
        discovered,
        preflight_records,
    ) = preflight_validation()

    before_inventory = input_inventory(
        discovered
    )

    results = []
    frame_records = []

    for (
        video_index,
        (
            scenario,
            video_path,
        ),
    ) in enumerate(
        discovered,
        start=1,
    ):
        print()
        print(
            "=" * 72
        )
        print(
            f"[{video_index}/{len(discovered)}] "
            f"{scenario}: {video_label(video_path)}"
        )
        print(
            "=" * 72
        )

        try:
            result, video_frame_records = run_video(
                scenario,
                video_path,
                smoke_limit,
            )
            results.append(
                result
            )
            frame_records.extend(
                video_frame_records
            )

            print(
                "Status: "
                f"{result['status']}"
            )
            print(
                "Frames: "
                f"{result['frames_read']}"
            )
            print(
                "Faces: "
                f"{result['total_faces']}"
            )
            print(
                "Processing FPS: "
                f"{result['effective_processing_fps']:.4f}"
            )
            print(
                "Mean ms/frame: "
                f"{result['mean_ms_per_frame']:.4f}"
            )

        except Exception as exc:
            reason = (
                f"{type(exc).__name__}: {exc}"
            )
            results.append(
                result_failure(
                    scenario,
                    video_path,
                    reason,
                )
            )

            print(
                "Status: VIDEO_EXECUTION_FAILED"
            )
            print(
                f"Reason: {reason}"
            )

    after_inventory = input_inventory(
        discovered
    )
    inventory_unchanged = (
        before_inventory
        == after_inventory
    )

    for result in results:
        if result[
            "status"
        ] in {
            "PASS",
            "PASS_WITH_FRAME_FAILURES",
        }:
            result[
                "dataset_read_only_verified"
            ] = inventory_unchanged

    summary = build_summary(
        preflight_records,
        results,
        frame_records,
        smoke_limit,
        inventory_unchanged,
    )

    return (
        results,
        frame_records,
        summary,
    )


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        (
            output_dir
            / "face_pipeline_runtime_results.csv"
        ),
        RESULT_FIELDS,
        results,
    )
    write_csv(
        (
            output_dir
            / "face_pipeline_runtime_frame_times.csv"
        ),
        FRAME_FIELDS,
        frame_records,
    )

    with (
        output_dir
        / "face_pipeline_runtime_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )


def print_summary(
    summary: dict[str, Any],
) -> None:
    execution = summary[
        "execution"
    ]

    print()
    print(
        "Runtime / Processing Performance summary"
    )
    print(
        "=" * 40
    )
    print(
        f"Videos: {execution['videos']}"
    )
    print(
        f"Frames read: {execution['frames_read']}"
    )
    print(
        f"Successful frames: {execution['successful_frames']}"
    )
    print(
        f"Failed frames: {execution['failed_frames']}"
    )
    print(
        f"Total faces: {execution['total_faces']}"
    )
    print(
        "Effective processing FPS: "
        f"{execution['effective_processing_fps']:.4f}"
    )
    print(
        "Mean ms/frame: "
        f"{execution['mean_ms_per_frame']:.4f}"
    )
    print(
        "Median ms/frame: "
        f"{execution['median_ms_per_frame']:.4f}"
    )
    print(
        "P95 ms/frame: "
        f"{execution['p95_ms_per_frame']:.4f}"
    )
    print(
        "Dataset read-only verification: "
        f"{'PASS' if summary['dataset']['read_only_verified'] else 'FAIL'}"
    )
    print(
        f"Status: {summary['status']}"
    )


def main() -> None:
    args = parse_args()

    if args.preflight_only:
        (
            discovered,
            preflight_records,
        ) = preflight_validation()

        print(
            "Runtime benchmark preflight: PASS"
        )
        print(
            f"Videos discovered: {len(discovered)}"
        )

        for record in preflight_records:
            print(
                f"- {record['scenario']}: "
                f"{record['video']} | "
                f"{record['width']}x{record['height']} | "
                f"{record['input_fps']:.6f} FPS | "
                f"reported_frames={record['reported_frames']}"
            )

        return

    (
        results,
        frame_records,
        summary,
    ) = run_benchmark(
        args.smoke_test
    )

    results_root = FINAL_OUTPUT_DIR.parent

    if args.smoke_test is not None:
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=".runtime_performance_smoke_",
                dir=results_root,
            )
        )

        try:
            write_outputs(
                staging_dir,
                results,
                frame_records,
                summary,
            )
            validate_staged_outputs(
                staging_dir
            )
            print_summary(
                summary
            )
            print(
                "Smoke-test outputs validated; "
                "final runtime-performance outputs were not replaced."
            )

        finally:
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )

        return

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".runtime_performance_",
            dir=results_root,
        )
    )

    try:
        write_outputs(
            staging_dir,
            results,
            frame_records,
            summary,
        )
        validate_staged_outputs(
            staging_dir
        )
        commit_staged_output_directory(
            staging_dir
        )
        print_summary(
            summary
        )
        print(
            f"Final outputs: {FINAL_OUTPUT_DIR}"
        )

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )

        raise


if __name__ == "__main__":
    main()
