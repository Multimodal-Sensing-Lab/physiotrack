"""Synthetic qualitative demonstration of the real PhysioTrack face tracker.

Automatically processes supported videos in:
    validation/synthetic_media/face_tracking_media/

Publishes only under:
    validation/face_tracking/results/synthetic_demo/

The ECCV 2016 benchmark, its annotations, and its outputs are not accessed.
Each video has an independent OC-SORT state. This script measures no accuracy.
Each successful video produces the original tracked MP4 and per-frame CSV,
plus three illustrative PNG frames decoded from the annotated MP4.
Only a complete, independently verified run replaces earlier output.
Both JSON and TXT summaries contain the same complete record.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig, FaceTracker


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "face_tracking_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".mpg",
    ".mpeg", ".wmv", ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
})
DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.45
TRACKER_TYPE = "ocsort"
OUTPUT_CODECS = ("mp4v", "avc1")
TRAIL_LENGTH = 18  # Visual presentation only; does not affect the tracker.
STILL_FRACTIONS = (0.25, 0.50, 0.75)  # Frames from annotated MP4, no second inference.
CSV_FIELDS = (
    "source_frame", "track_index", "tracks_in_frame", "track_id", "confidence",
    "box_x1", "box_y1", "box_x2", "box_y2", "drawing_status",
)
DISABLED_MODULES = (
    "orientation", "landmarks", "quality", "eyes", "blink", "gaze",
    "gaze_estimation", "mouth", "mouth_motion", "emotion", "regions", "temporal",
)


class MediaError(RuntimeError):
    """A media or output-verification failure."""


def check_project_imports() -> dict[str, str]:
    """Require implementation classes to originate from this project checkout."""
    expected_root = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    paths = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig, FaceTracker):
        actual = Path(inspect.getfile(cls)).resolve()
        if not actual.is_relative_to(expected_root):
            raise RuntimeError(
                f"{cls.__name__} does not come from the local physiotrack-v2 checkout. "
                "Install the repository in editable mode in the active environment."
            )
        paths[cls.__name__] = actual.relative_to(REPOSITORY_ROOT).as_posix()
    return paths


def discover_videos() -> list[Path]:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            "Media directory missing: validation/synthetic_media/face_tracking_media/"
        )
    videos = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda p: (
        p.relative_to(INPUT_DIR).as_posix().casefold(),
        p.relative_to(INPUT_DIR).as_posix(),
    ))
    if not videos:
        raise MediaError(
            "No supported videos found in validation/synthetic_media/face_tracking_media/. "
            "Existing results were not modified."
        )
    return videos


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def library_version(package: str) -> str | None:
    """Record exact installed package versions without importing optional libraries."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def summary_as_text(summary: dict[str, Any]) -> str:
    """Readable, complete projection of JSON; no synthetic ground-truth claims."""
    lines = [
        "PhysioTrack-v2 Face Tracking - Synthetic Qualitative Demonstration",
        "=" * 68,
        "",
        "Human-readable rendering of synthetic_demo_summary.json",
        "This is not a tracking accuracy benchmark; no ground truth is used.",
        "",
    ]

    def walk(value: Any, indent: int = 0, key: str | None = None) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            if not value:
                lines.append(f"{prefix}(empty)")
            for name, item in value.items():
                walk(item, indent, str(name))
        elif isinstance(value, list):
            if key is not None:
                lines.append(f"{prefix}{key}: ({len(value)} items)")
                indent += 2
                prefix = " " * indent
            if not value:
                lines.append(f"{prefix}(none)")
            for index, item in enumerate(value, 1):
                walk(item, indent, f"item_{index:03d}")
        else:
            lines.append(f"{prefix}{key}: {json.dumps(value, ensure_ascii=False, allow_nan=False)}")

    walk(summary)
    return "\n".join(lines) + "\n"


def output_paths(source: Path) -> tuple[Path, Path]:
    """Preserve input nesting, with a stable hashed basename for uniqueness."""
    relative = source.relative_to(INPUT_DIR)
    suffix_hash = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.name)[:90].strip(". ")
    if not safe_name:
        safe_name = "media"
    base = f"{safe_name}_{suffix_hash}"
    return (
        Path("videos") / relative.parent / f"{base}_tracked.mp4",
        Path("tracks") / relative.parent / f"{base}_observations.csv",
    )


def make_config() -> FaceAnalysisConfig:
    """Exactly the tracking-only configuration used in component validation."""
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
        tracker_type=TRACKER_TYPE,
    )
    config.validate()
    return config


def create_pipeline(detector: Face) -> FaceAnalysis:
    """Instantiate a NEW tracker for each independent video sequence."""
    tracker = FaceTracker(tracker_type=TRACKER_TYPE, device=DEVICE)
    pipeline = FaceAnalysis(
        detector=detector,
        tracker=tracker,
        config=make_config(),
        device=DEVICE,
        verbose=False,
    )
    if (pipeline.detector is not detector or pipeline.tracker is not tracker
            or any(getattr(pipeline, attr) is not None for attr in DISABLED_MODULES)):
        raise RuntimeError("Unexpected modules enabled in the tracking-only pipeline")
    return pipeline


def track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour computed from the real track identifier."""
    digest = hashlib.sha256(str(track_id).encode("ascii")).digest()
    return (int(60 + digest[0] % 175),
            int(60 + digest[1] % 175),
            int(60 + digest[2] % 175))


def draw_text(frame: np.ndarray, text: str, x: int, y: int,
              scale: float, color: tuple[int, int, int]) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    height, width = frame.shape[:2]
    xx = max(0, min(x, max(0, width - tw - 8)))
    yy = max(th + 5, min(y, height - baseline - 3))
    cv2.rectangle(frame, (xx, max(0, yy - th - 5)),
                  (min(width - 1, xx + tw + 7), min(height - 1, yy + baseline + 4)),
                  color, -1)
    cv2.putText(frame, text, (xx + 3, yy), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 1, cv2.LINE_AA)


def annotate_tracks(frame: np.ndarray, result: Any,
                    history: dict[int, deque], frame_no: int,
                    known_ids: set[int]) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Draw *only* tracks provided by PhysioTrack; do not fabricate trajectories."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise MediaError("Decoded video frame is not a three-channel BGR image")
    height, width = frame.shape[:2]
    tracks = list(result)
    annotated = frame.copy()
    entries: list[dict[str, Any]] = []
    invalid = 0
    frame_ids = set()
    font_scale = max(0.36, min(0.76, min(width, height) / 950.0))
    thickness = max(1, min(3, min(width, height) // 440))

    for index, track in enumerate(tracks, 1):
        record: dict[str, Any] = {
            "source_frame": frame_no,
            "track_index": index,
            "tracks_in_frame": len(tracks),
            "track_id": "",
            "confidence": "",
            "box_x1": "", "box_y1": "", "box_x2": "", "box_y2": "",
            "drawing_status": "INVALID_TRACK",
        }
        try:
            identifier = track.id
            if identifier is None or not math.isfinite(float(identifier)):
                raise ValueError("Missing or non-finite track ID")
            identity = int(identifier)
            if float(identifier) != float(identity) or identity in frame_ids:
                raise ValueError("Non-integral or duplicate ID in a frame")
            frame_ids.add(identity)
            record["track_id"] = identity
            coords = np.asarray(track.box, dtype=np.float64).reshape(-1)
            if coords.size != 4 or not np.isfinite(coords).all():
                raise ValueError("Invalid track box coordinates")
            x1, y1, x2, y2 = [float(v) for v in coords]
            record.update({"box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2})
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Non-positive track box area")
            confidence = track.confidence
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Invalid tracking confidence")
                record["confidence"] = confidence

            # Raw coordinates are recorded intact; clipping is for rendering only.
            left = max(0, min(width - 1, int(round(x1))))
            top = max(0, min(height - 1, int(round(y1))))
            right = max(0, min(width - 1, int(round(x2))))
            bottom = max(0, min(height - 1, int(round(y2))))
            if right <= left or bottom <= top:
                raise ValueError("Track box cannot be drawn within frame bounds")
            color = track_color(identity)
            known_ids.add(identity)
            center = ((left + right) // 2, (top + bottom) // 2)
            history[identity].append(center)
            trail = np.asarray(history[identity], dtype=np.int32)
            if len(trail) > 1:
                cv2.polylines(annotated, [trail], False, color, max(1, thickness - 1),
                              cv2.LINE_AA)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
            label = f"ID {identity}"
            if confidence is not None:
                label += f"  {confidence:.2f}"
            label_y = top - 5
            if label_y < 17:
                label_y = min(height - 5, bottom + 19)
            draw_text(annotated, label, left, label_y, font_scale, color)
            record["drawing_status"] = "DRAWN"
        except (ValueError, TypeError, OverflowError, AttributeError):
            invalid += 1
        entries.append(record)

    # Keep an overlay of observed counts only (not a claim of GT or benchmark accuracy).
    overlay = f"Frame {frame_no} | Tracks {len(tracks)} | Seen IDs {len(known_ids)}"
    draw_text(annotated, overlay, 6, max(18, int(24 * font_scale / 0.5)),
              max(0.35, font_scale * 0.85), (214, 214, 214))
    return annotated, entries, invalid


def open_video_capture(source: Path, temp_dir: Path, ordinal: int):
    """Open original directly, falling back to an ASCII temporary copy on Windows."""
    cap = cv2.VideoCapture(str(source))
    copy_path = None
    if not cap.isOpened():
        cap.release()
        copy_path = temp_dir / f"input_{ordinal:05d}{source.suffix.lower()}"
        shutil.copyfile(source, copy_path)
        cap = cv2.VideoCapture(str(copy_path))
    if not cap.isOpened():
        cap.release()
        if copy_path is not None:
            copy_path.unlink(missing_ok=True)
        raise MediaError("OpenCV cannot open the input video")
    return cap, copy_path


def open_video_writer(path: Path, width: int, height: int, fps: float):
    for codec in OUTPUT_CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                 fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
        path.unlink(missing_ok=True)
    raise MediaError("OpenCV has no available MP4 encoder")


def verify_video(path: Path, width: int, height: int, fps: float,
                 expected_frames: int) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise MediaError("Output video could not be reopened")
    count = 0
    try:
        w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        output_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if (w, h) != (width, height):
            raise MediaError("Output video dimensions do not match the source")
        if not math.isfinite(output_fps) or abs(output_fps - fps) > max(0.05, fps * 0.01):
            raise MediaError("Output FPS differs from source beyond tolerance")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise MediaError("Output video contains a malformed frame")
            count += 1
        if count != expected_frames:
            raise MediaError(f"Output frame count {count} differs from {expected_frames}")
        return {"verified_output_frames": count, "output_fps": output_fps}
    finally:
        cap.release()


def extract_illustrative_frames(video: Path, output_video_rel: Path,
                                stage: Path, frames: int, width: int,
                                height: int) -> list[str]:
    """Save 25%, 50%, 75% PNG stills from the VERIFIED tracked MP4.

    They show exactly what the annotated video contains; no extra model
    inference is run. The original MP4 and observations CSV are unchanged.
    """
    indices = sorted({max(1, min(frames, int(round(frames * f)))) for f in STILL_FRACTIONS})
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise MediaError("Cannot reopen annotated video for illustrative PNG frames")
    outputs: list[str] = []
    try:
        cursor = 0
        wanted = set(indices)
        while cursor < max(indices):
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape != (height, width, 3):
                raise MediaError("Cannot decode an illustrative frame from annotated video")
            cursor += 1
            if cursor not in wanted:
                continue
            rel = Path("images") / output_video_rel.parent.relative_to("videos") / (
                f"{output_video_rel.stem}_frame_{cursor:06d}.png"
            )
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                raise MediaError("Could not encode an illustrative PNG frame")
            encoded.tofile(str(target))  # Supports Unicode filenames on Windows.
            outputs.append(rel.as_posix())
    finally:
        cap.release()
    if len(outputs) != len(indices):
        raise MediaError("An illustrative PNG frame was not generated")
    return outputs


def verify_png(path: Path, width: int, height: int) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise MediaError("An illustrative image is missing or empty")
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (ValueError, OSError, cv2.error) as exc:
        raise MediaError("An illustrative PNG could not be decoded") from exc
    if decoded is None or decoded.shape != (height, width, 3):
        raise MediaError("An illustrative PNG has an incorrect shape")


def process_video(source: Path, output_video: Path, output_csv: Path,
                  detector: Face, working_dir: Path, ordinal: int) -> dict[str, Any]:
    cap, copied_input = open_video_capture(source, working_dir, ordinal)
    temporary_mp4 = working_dir / f"render_{ordinal:05d}.mp4"
    writer = None
    pipeline = None
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise MediaError("Invalid source video dimensions or FPS")
        pipeline = create_pipeline(detector)  # Independent OC-SORT state per video.
        writer, codec = open_video_writer(temporary_mp4, width, height, fps)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame_number = 0
        frames_with_tracks = 0
        observations = 0
        invalid_observations = 0
        unique_ids: set[int] = set()
        history: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
        with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape != (height, width, 3):
                    raise MediaError("Source frame is unreadable or changed dimensions")
                frame_number += 1
                # Inference must be performed ONCE per frame in decoded order.
                result = pipeline.predict(frame)
                annotated, entries, invalid = annotate_tracks(
                    frame, result, history, frame_number, unique_ids,
                )
                observations += len(entries)
                invalid_observations += invalid
                if entries:
                    frames_with_tracks += 1
                    csv_writer.writerows(entries)
                else:
                    csv_writer.writerow({
                        "source_frame": frame_number,
                        "track_index": "", "tracks_in_frame": 0,
                        "track_id": "", "confidence": "",
                        "box_x1": "", "box_y1": "", "box_x2": "", "box_y2": "",
                        "drawing_status": "NO_TRACK",
                    })
                writer.write(annotated)
                if frame_number % 200 == 0:
                    print(f"    Frames {frame_number}; track observations {observations}", flush=True)
        if frame_number == 0:
            raise MediaError("Source video contains zero decoded frames")
        if declared_frames > 0 and abs(declared_frames - frame_number) > 2:
            raise MediaError(
                f"Source ended after {frame_number} frames; metadata declared {declared_frames}"
            )
        writer.release()
        writer = None
        checked = verify_video(temporary_mp4, width, height, fps, frame_number)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_mp4, output_video)
        return {
            "width": width, "height": height, "source_fps": fps,
            "declared_source_frames": declared_frames if declared_frames > 0 else None,
            "frames_processed": frame_number,
            "frames_with_tracks": frames_with_tracks,
            "frames_without_tracks": frame_number - frames_with_tracks,
            "track_observations": observations,
            "unique_track_ids": len(unique_ids),
            "invalid_track_observations": invalid_observations,
            "output_codec": codec,
            **checked,
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pipeline is not None and hasattr(pipeline, "close"):
            pipeline.close()
        temporary_mp4.unlink(missing_ok=True)
        if copied_input is not None:
            copied_input.unlink(missing_ok=True)


def verify_staging(stage: Path, summary: dict[str, Any]) -> None:
    """Verify the complete media, CSV, image, and summary set before publication."""
    if (summary["status"] != "COMPLETE" or summary["failed_files"] != 0
            or summary["successful_files"] != summary["discovered_files"]
            or summary["discovered_files"] != len(summary["files"])
            or summary["discovered_files"] < 1):
        raise MediaError("An incomplete run cannot replace previous outputs")

    expected: set[str] = {"synthetic_demo_summary.json", "synthetic_demo_summary.txt"}
    total_frames = 0
    total_tracks = 0
    total_no_tracks = 0
    total_invalid = 0
    total_images = 0
    input_names: set[str] = set()
    for record in summary["files"]:
        if record["status"] != "SUCCESS":
            raise MediaError("An input failed; no output may be published")
        input_key = record["input"].casefold()
        if input_key in input_names:
            raise MediaError("Duplicate input in synthetic summary")
        input_names.add(input_key)

        video_rel = record["output_video"]
        csv_rel = record["output_observations_csv"]
        still_rels = record["output_images"]
        if not still_rels:
            raise MediaError("Missing illustrative images for a tracked video")
        for rel in [video_rel, csv_rel, *still_rels]:
            path = Path(rel)
            if (path.is_absolute() or ".." in path.parts or rel in expected
                    or (stage / path).is_symlink()):
                raise MediaError("Unsafe or duplicate staged output name")
            expected.add(rel)

        video = stage / video_rel
        csv_path = stage / csv_rel
        if (not video.is_file() or not csv_path.is_file()
                or video.stat().st_size <= 0 or csv_path.stat().st_size <= 0):
            raise MediaError("Missing or empty staged MP4/CSV")
        verify_video(video, record["width"], record["height"],
                     record["source_fps"], record["frames_processed"])

        no_tracks = 0
        track_rows = 0
        invalid_rows = 0
        frames_with_tracks = 0
        previous_frame = 0
        next_track_index = 1
        expected_tracks = 0
        current_tracks = 0
        seen_ids: set[int] = set()
        observed_ids: set[int] = set()
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise MediaError("Incorrect track CSV schema")
            for row in reader:
                try:
                    frame = int(row["source_frame"])
                except (ValueError, TypeError) as exc:
                    raise MediaError("Invalid CSV source-frame index") from exc
                if not 1 <= frame <= record["frames_processed"]:
                    raise MediaError("Out-of-range CSV source frame")
                if frame != previous_frame:
                    if (previous_frame > 0 and (current_tracks != expected_tracks
                                                or next_track_index != expected_tracks + 1)):
                        raise MediaError("Incorrect CSV track accounting for a frame")
                    if frame != previous_frame + 1:
                        raise MediaError("CSV frames must cover the full ordered video")
                    previous_frame = frame
                    next_track_index = 1
                    current_tracks = 0
                    seen_ids.clear()
                    expected_tracks = int(row["tracks_in_frame"])
                    if expected_tracks < 0:
                        raise MediaError("Negative tracks_in_frame")
                    if expected_tracks > 0:
                        frames_with_tracks += 1
                elif int(row["tracks_in_frame"]) != expected_tracks:
                    raise MediaError("Inconsistent track count within a frame")

                if row["drawing_status"] == "NO_TRACK":
                    if (expected_tracks != 0 or row["track_index"]
                            or row["track_id"] or current_tracks):
                        raise MediaError("Invalid no-track placeholder CSV row")
                    no_tracks += 1
                elif row["drawing_status"] in ("DRAWN", "INVALID_TRACK"):
                    if (expected_tracks == 0 or int(row["track_index"]) != next_track_index):
                        raise MediaError("Incorrect track index in CSV")
                    current_tracks += 1
                    next_track_index += 1
                    track_rows += 1
                    if row["drawing_status"] == "INVALID_TRACK":
                        invalid_rows += 1
                    if row["drawing_status"] == "DRAWN":
                        identity = int(row["track_id"])
                        if identity in seen_ids:
                            raise MediaError("Duplicate drawn track ID within a frame")
                        seen_ids.add(identity)
                        observed_ids.add(identity)
                else:
                    raise MediaError("Unknown CSV drawing status")

        if (previous_frame != record["frames_processed"]
                or (expected_tracks and current_tracks != expected_tracks)
                or no_tracks != record["frames_without_tracks"]
                or frames_with_tracks != record["frames_with_tracks"]
                or track_rows != record["track_observations"]
                or invalid_rows != record["invalid_track_observations"]
                or len(observed_ids) != record["unique_track_ids"]):
            raise MediaError("Staged CSV row counts do not match recorded tracking outputs")

        for rel in still_rels:
            verify_png(stage / rel, record["width"], record["height"])
        total_images += len(still_rels)
        total_frames += record["frames_processed"]
        total_tracks += record["track_observations"]
        total_no_tracks += record["frames_without_tracks"]
        total_invalid += record["invalid_track_observations"]

        if sha256_file(video) != record["output_video_sha256"]:
            raise MediaError("Staged MP4 hash changed")
        if sha256_file(csv_path) != record["output_observations_sha256"]:
            raise MediaError("Staged CSV hash changed")
        if {rel: sha256_file(stage / rel) for rel in still_rels} != record["output_images_sha256"]:
            raise MediaError("Staged illustrative PNG hash changed")

    for key, value in (
        ("total_processed_frames", total_frames),
        ("total_track_observations", total_tracks),
        ("total_frames_without_tracks", total_no_tracks),
        ("total_invalid_track_observations", total_invalid),
        ("total_illustrative_images", total_images),
    ):
        if summary[key] != value:
            raise MediaError(f"Staged summary mismatch: {key}")

    json_path = stage / "synthetic_demo_summary.json"
    txt_path = stage / "synthetic_demo_summary.txt"
    with json_path.open("r", encoding="utf-8") as f:
        if json.load(f) != summary:
            raise MediaError("Staged JSON summary failed verification")
    if txt_path.read_text(encoding="utf-8") != summary_as_text(summary):
        raise MediaError("Staged TXT summary failed verification")

    actual = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in stage.rglob("*")):
        raise MediaError("Staged output inventory has missing or unexpected files")
    if any(not any(p.iterdir()) for p in stage.rglob("*") if p.is_dir()):
        raise MediaError("An empty output folder would be published")


def commit_staged_directory(stage: Path, staging_root: Path,
                            summary: dict[str, Any]) -> None:
    """Publish an entirely verified output set, not a merge with earlier files.

    The old result is retained only until the new final directory has been
    installed and independently checked. Ordinary publication/verification
    errors restore the old result. This is not power-loss atomic.
    """
    backup = staging_root / "previous_synthetic_demo"
    backed_up = False
    published = False
    try:
        if FINAL_DIR.exists():
            if FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir():
                raise MediaError("Final synthetic_demo path is not a normal directory")
            FINAL_DIR.replace(backup)
            backed_up = True
        # Rename the complete staged directory, replacing the old DIRECTORY,
        # never copying/merging files into the previously published directory.
        stage.replace(FINAL_DIR)
        published = True
        # A second check is intentional: nothing is discarded until the new
        # location itself passes the complete validation protocol.
        verify_staging(FINAL_DIR, summary)
    except Exception:
        # Restore the entire old directory if publication OR final checks fail.
        try:
            if published:
                FINAL_DIR.replace(stage)
            if backed_up:
                backup.replace(FINAL_DIR)
        except Exception as rollback_error:
            raise MediaError(
                "Publication verification failed and rollback was unsuccessful; "
                f"do not delete the recovery backup at {backup}"
            ) from rollback_error
        raise

    # At this point the complete NEW final result has been verified. The old
    # backup is obsolete and must not be left inside staging on success.
    if backed_up:
        try:
            shutil.rmtree(backup)
        except OSError as cleanup_error:
            raise MediaError(
                "The new synthetic_demo was verified and published, but the "
                f"obsolete backup could not be removed: {backup}"
            ) from cleanup_error


def cleanup_previous_completed_stages(current_stage_root: Path) -> list[str]:
    """Prune only recognizable leftovers of earlier *completed* publishes.

    Never remove a pending staged synthetic_demo or an unknown directory:
    those may be needed to recover an interrupted or failed run. Called only
    after a new complete final directory has been verified and published.
    """
    skipped = []
    for old_root in sorted(RESULTS_DIR.glob(".face_tracking_synthetic_stage_*")):
        if old_root == current_stage_root or old_root.is_symlink() or not old_root.is_dir():
            continue
        try:
            children = {child.name for child in old_root.iterdir()}
            backup = old_root / "previous_synthetic_demo"
            if (not backup.is_dir() or backup.is_symlink()
                    or "synthetic_demo" in children
                    or not children.issubset({"previous_synthetic_demo", "video_work"})):
                continue
            # Guard against an unexpected symlink inside the known stale root.
            if any(child.is_symlink() for child in old_root.rglob("*")):
                continue
            shutil.rmtree(old_root)
        except OSError:
            skipped.append(str(old_root))
    return skipped


def main() -> None:
    component_paths = check_project_imports()
    videos = discover_videos()
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise MediaError("Unsafe final output path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".face_tracking_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    # Output folders are created only when a corresponding file is produced.
    # Successful tracking videos require videos/, tracks/, and images/.
    working_dir = staging_root / "video_work"
    working_dir.mkdir()
    records = []
    try:
        detector = Face(
            conf=DETECTOR_CONFIDENCE,
            iou=DETECTOR_IOU,
            device=DEVICE,
            verbose=False,
        )
        outputs_seen: set[str] = set()
        print("PhysioTrack face tracking synthetic qualitative demonstration")
        print("Videos discovered:", len(videos))
        print("Inputs: validation/synthetic_media/face_tracking_media/")
        print("Outputs: validation/face_tracking/results/synthetic_demo/")
        for ordinal, source in enumerate(videos, 1):
            relative = source.relative_to(INPUT_DIR).as_posix()
            video_name, csv_name = output_paths(source)
            if any(p.as_posix().casefold() in outputs_seen for p in (video_name, csv_name)):
                raise MediaError("Output-name collision detected")
            outputs_seen.update((video_name.as_posix().casefold(), csv_name.as_posix().casefold()))
            print(f"[{ordinal}/{len(videos)}] video: {relative}", flush=True)
            record: dict[str, Any] = {"input": relative, "status": "FAILED"}
            try:
                record["source_sha256"] = sha256_file(source)
                measurements = process_video(
                    source, stage / video_name, stage / csv_name,
                    detector, working_dir, ordinal,
                )
                images = extract_illustrative_frames(
                    stage / video_name, video_name, stage,
                    measurements["frames_processed"], measurements["width"],
                    measurements["height"],
                )
                if sha256_file(source) != record["source_sha256"]:
                    raise MediaError("Input video changed during processing")
                record.update(measurements)
                record["output_images"] = images
                record["output_video_sha256"] = sha256_file(stage / video_name)
                record["output_observations_sha256"] = sha256_file(stage / csv_name)
                record["output_images_sha256"] = {
                    name: sha256_file(stage / name) for name in images
                }
                record["status"] = "SUCCESS"
                record["output_video"] = video_name.as_posix()
                record["output_observations_csv"] = csv_name.as_posix()
                print(f"    OK: {measurements['frames_processed']} frames; "
                      f"{measurements['track_observations']} observations; "
                      f"{measurements['unique_track_ids']} distinct sequence-local IDs")
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc) if isinstance(exc, MediaError) else "Processing failure; see terminal traceback"
                print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                traceback.print_exc()
                (stage / video_name).unlink(missing_ok=True)
                (stage / csv_name).unlink(missing_ok=True)
                # Any staged images will be discarded; failed runs are never published.
            records.append(record)
        failed = sum(r["status"] != "SUCCESS" for r in records)
        succeeded = len(records) - failed
        summary = {
            "type": "synthetic_qualitative_face_tracking_demonstration",
            "benchmark_evaluation": False,
            "project": "physiotrack-v2",
            "input_directory": "validation/synthetic_media/face_tracking_media",
            "output_directory": "validation/face_tracking/results/synthetic_demo",
            "component_imports": component_paths,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "component_source_sha256": {
                name: sha256_file(REPOSITORY_ROOT / relative)
                for name, relative in component_paths.items()
            },
            "configuration": {
                "detector": "physiotrack.face.Face",
                "pipeline": "physiotrack.face.FaceAnalysis",
                "tracker": "physiotrack.face.FaceTracker",
                "tracker_backend": TRACKER_TYPE,
                "detector_confidence": DETECTOR_CONFIDENCE,
                "detector_iou": DETECTOR_IOU,
                "device": DEVICE,
                "tracking_state": "reset_per_video_using_new_FaceTracker",
                "ground_truth_used": False,
                "benchmark_metrics_calculated": False,
                "visual_trail_length": TRAIL_LENGTH,
                "illustrative_frame_fractions": list(STILL_FRACTIONS),
                "illustrative_frames_are_decoded_from_saved_annotated_video": True,
            },
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "torch": library_version("torch"),
                "ultralytics": library_version("ultralytics"),
                "platform": platform.system(),
            },
            "video_note": "Source audio and variable-frame timestamps are not preserved by OpenCV MP4 encoding. "
                          "IDs are unique within a sequence, not across different videos. "
                          "Track observations are not unique people and are not accuracy measurements.",
            "status": "COMPLETE" if not failed else "REJECTED_WITH_FILE_FAILURES",
            "discovered_files": len(videos),
            "successful_files": succeeded,
            "failed_files": failed,
            "total_processed_frames": sum(r.get("frames_processed", 0) for r in records if r["status"] == "SUCCESS"),
            "total_track_observations": sum(r.get("track_observations", 0) for r in records if r["status"] == "SUCCESS"),
            "total_frames_without_tracks": sum(r.get("frames_without_tracks", 0) for r in records if r["status"] == "SUCCESS"),
            "total_invalid_track_observations": sum(r.get("invalid_track_observations", 0) for r in records if r["status"] == "SUCCESS"),
            "total_illustrative_images": sum(len(r.get("output_images", [])) for r in records if r["status"] == "SUCCESS"),
            "files": records,
        }
        summary_path = stage / "synthetic_demo_summary.json"
        with summary_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")
        (stage / "synthetic_demo_summary.txt").write_text(
            summary_as_text(summary), encoding="utf-8", newline="\n",
        )
        if failed:
            # A partially successful run must not replace a previously complete run.
            failure_report = RESULTS_DIR / "synthetic_demo_last_attempt.json"
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                prefix=".synthetic_demo_failed_", suffix=".json", dir=RESULTS_DIR,
            ) as f:
                temporary_report = Path(f.name)
                json.dump(summary, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(temporary_report, failure_report)
            temporary_txt = RESULTS_DIR / (".synthetic_demo_failed_" + staging_root.name + ".txt")
            temporary_txt.write_text(summary_as_text(summary), encoding="utf-8", newline="\n")
            os.replace(temporary_txt, RESULTS_DIR / "synthetic_demo_last_attempt.txt")
            raise MediaError(
                f"{failed} video(s) failed. Previous completed outputs remain unchanged. "
                "See results/synthetic_demo_last_attempt.json, .txt, and terminal diagnostics."
            )
        verify_staging(stage, summary)
        # Ensure the input inventory did not change during processing.
        if [p.relative_to(INPUT_DIR).as_posix() for p in discover_videos()] != [
            p.relative_to(INPUT_DIR).as_posix() for p in videos
        ]:
            raise MediaError("Input video inventory changed during the run")
        commit_staged_directory(stage, staging_root, summary)
        # Do not report COMPLETE until the *current* stage is cleaned.
        shutil.rmtree(staging_root)
        old_stages_not_cleaned = cleanup_previous_completed_stages(staging_root)
        for obsolete in old_stages_not_cleaned:
            print(f"WARNING: obsolete staging could not be removed: {obsolete}",
                  file=sys.stderr)
        print("\nDemonstration complete: COMPLETE")
        print("Successful videos:", succeeded)
        print("Failed videos:", failed)
        print("Processed frames:", summary["total_processed_frames"])
        print("Track observations:", summary["total_track_observations"])
        print("Frames without tracks:", summary["total_frames_without_tracks"])
        print("Invalid track observations:", summary["total_invalid_track_observations"])
        print("Illustrative PNG frames:", summary["total_illustrative_images"])
        print("Saved summaries: validation/face_tracking/results/synthetic_demo/synthetic_demo_summary.json")
        print("                 validation/face_tracking/results/synthetic_demo/synthetic_demo_summary.txt")
    finally:
        # Keep a backup only in the exceptional case where it is still needed
        # for recovery. Successful publication removes it before reaching here.
        if staging_root.exists():
            if (staging_root / "previous_synthetic_demo").exists():
                print(f"Recovery backup preserved at: {staging_root / 'previous_synthetic_demo'}",
                      file=sys.stderr)
            else:
                shutil.rmtree(staging_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Synthetic tracking demonstration failed: {exc}", file=sys.stderr)
        sys.exit(1)
