"""PhysioTrack-v2 synthetic qualitative EyeOpenness + Blink demonstration.

Installation:
    validation/eye_openness_and_blink/eye_blink_synthetic_demo.py

Input videos:
    validation/synthetic_media/eye_openness_and_blink_media/

Outputs (only this directory is replaced after a *fully validated* run):
    validation/eye_openness_and_blink/results/synthetic_demo/

Uses real Face -> FaceTracker(OC-SORT) -> FaceLandmarks(MediaPipe) ->
EyeOpenness -> BlinkDetector through the project's FaceAnalysis. An EyeOpenness
subclass records the very landmarks passed to EyeOpenness.predict for drawing,
without repeating MediaPipe inference, changing project files or changing the
underlying EAR formula. Default mode runs the complete reproducible workflow:
    - Validate independently annotated Ground Truth + SHA-256 input videos
    - Run the real FaceAnalysis on every synthetic video (new CSV, MP4, PNG)
    - Match temporal blink events to Ground Truth with tIoU >= 0.50
    - Produce per-person, per-video, aggregate results and accuracy-overlaid videos
    - Commit both output folders only after ALL stages succeed
Optional modes:
    --make-gt-template : scaffold EMPTY annotation slots (if no GT exists)
    --evaluate-only    : score verified GT from completed output WITHOUT inference
Synthetic scores are ALWAYS separate from the MPEBlink benchmark. Cross-device
bitwise identical inference is NOT guaranteed without identical models/software.
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
from physiotrack.face.blink import BlinkDetector
from physiotrack.face.eyes import EyeOpenness
from physiotrack.face.landmarks import FaceLandmarks


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "eye_openness_and_blink_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv",
})
EXPECTED_FILENAMES = (
    "01_normal_blink.mp4", "02_fast_blink.mp4", "03_prolonged_closure.mp4",
    "04_multi_person_blink.mp4", "05_head_motion_blink.mp4",
)
DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.45
TRACKER_TYPE = "ocsort"
THRESHOLD = 0.22
MIN_CLOSED_FRAMES = 3
POINT_RADIUS = 1  # DRAWING ONLY. Do not modify EyeOpenness computation.
OUTPUT_CODECS = ("mp4v", "avc1")
HISTORY_LENGTH = 150
SIDEBAR_WIDTH = 376
SHEET_CELL_WIDTH = 740
SHEET_CELL_HEIGHT = 370
CSV_COLUMNS = (
    "source_frame", "time_s", "face_index", "track_id", "detection_confidence",
    "box_x1", "box_y1", "box_x2", "box_y2", "landmarks_available",
    "eyes_available", "left_openness", "right_openness", "mean_openness",
    "blink_available", "eye_state", "blink_pulse", "blink_count", "blink_duration_s",
    "blink_rate_per_min", "drawing_status",
)


class DemoError(RuntimeError):
    """An input, rendering, verification or output-staging failure."""


class RecordingEyeOpenness(EyeOpenness):
    """Read-only recorder of *actual* FaceAnalysis EyeOpenness input landmarks.

    Crucially, FaceLandmarks remains the original concrete class inside
    FaceAnalysis, so its original predict_face(..., other_boxes=...) B06
    association path is preserved for multi-person sequences.
    """

    def __init__(self) -> None:
        self.frame_calls: list[dict[str, Any]] = []

    def begin_frame(self) -> None:
        self.frame_calls.clear()

    def predict(self, landmarks, image_size):
        # Compute via the ORIGINAL EyeOpenness implementation, unchanged.
        output = super().predict(landmarks, image_size=image_size)
        width, height = image_size
        points: dict[str, list[tuple[float, float]]] = {}
        for label, indices in (("left", self.LEFT_EYE), ("right", self.RIGHT_EYE)):
            poly = []
            for index in indices:
                p = landmarks[index]
                poly.append((float(p.x) * width, float(p.y) * height))
            points[label] = poly
        self.frame_calls.append({"points": points, "openness": output})
        return output


def check_project_imports() -> dict[str, str]:
    expected = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    classes = (Face, FaceTracker, FaceAnalysis, FaceAnalysisConfig,
               FaceLandmarks, EyeOpenness, BlinkDetector)
    resolved = {}
    if not expected.is_dir():
        raise DemoError(f"Project source is missing: {expected}")
    for cls in classes:
        path = Path(inspect.getfile(cls)).resolve()
        if not path.is_relative_to(expected):
            raise DemoError(
                f"{cls.__name__} is loaded from {path}, not this physiotrack-v2 checkout. "
                "Activate the project environment and editable install."
            )
        resolved[cls.__name__] = path.relative_to(REPOSITORY_ROOT).as_posix()
    return resolved


def environment_fingerprint() -> dict[str, Any]:
    """Record replay-relevant files/versions; do not claim universal bitwise equality."""
    classes = (Face, FaceTracker, FaceAnalysis, FaceAnalysisConfig,
               FaceLandmarks, EyeOpenness, BlinkDetector)
    source_hashes = {}
    for cls in classes:
        path = Path(inspect.getfile(cls)).resolve()
        source_hashes[path.relative_to(REPOSITORY_ROOT).as_posix()] = source_sha256(path)
    package_versions = {}
    for package in ("numpy", "mediapipe", "torch", "ultralytics", "opencv-python",
                    "opencv-contrib-python", "scipy", "onnxruntime"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "script_sha256": source_sha256(Path(__file__).resolve()),
        "implementation_sha256": dict(sorted(source_hashes.items())),
        "package_versions": package_versions,
        "inference_device": DEVICE,
        "note": "Input hashes and GT annotation hash must match. Exact numerical predictions "
                "also depend on pretrained model weights, dependency versions and runtime; "
                "cross-machine bitwise identity cannot be guaranteed by this script alone."
    }


def discover_videos() -> list[Path]:
    if not INPUT_DIR.is_dir():
        raise DemoError(
            "Synthetic media folder not found. Create "
            "validation/synthetic_media/eye_openness_and_blink_media/ and place videos there."
        )
    inputs = sorted(
        (p for p in INPUT_DIR.rglob("*") if p.is_file() and not p.is_symlink()
         and p.suffix.casefold() in SUPPORTED_VIDEO_EXTENSIONS),
        key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(),
                       p.relative_to(INPUT_DIR).as_posix()),
    )
    if not inputs:
        raise DemoError("No synthetic videos found. Existing outputs were not modified.")
    return inputs


def source_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(buf)
    return sha.hexdigest()


def output_names(source: Path) -> tuple[Path, Path, Path]:
    relative = source.relative_to(INPUT_DIR)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.stem).strip(". ")[:80] or "video"
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
    base = f"{safe}_{digest}"
    return (Path("annotated_videos") / relative.parent / f"{base}_annotated.mp4",
            Path("observations") / relative.parent / f"{base}_observations.csv",
            Path("annotated_images") / relative.parent / f"{base}_example.png")


def make_pipeline(detector: Face, fps: float) -> tuple[FaceAnalysis, RecordingEyeOpenness]:
    config = FaceAnalysisConfig(
        tracking=True, head_pose=False, landmarks=True, quality=False,
        eyes=True, blink=True, gaze=False, gaze_estimation=False, mouth=False,
        mouth_motion=False, emotion=False, regions=False, temporal=False,
        tracker_type=TRACKER_TYPE, blink_threshold=THRESHOLD,
        min_closed_frames=MIN_CLOSED_FRAMES,
    )
    config.validate()
    eye_recorder = RecordingEyeOpenness()
    tracker = FaceTracker(tracker_type=TRACKER_TYPE, device=DEVICE)
    pipeline = FaceAnalysis(
        detector=detector, tracker=tracker, eyes=eye_recorder,
        config=config, fps=fps, device=DEVICE, verbose=False,
    )
    if (pipeline.detector is not detector or pipeline.tracker is not tracker
            or pipeline.eyes is not eye_recorder
            or type(pipeline.landmarks) is not FaceLandmarks
            or type(pipeline.blink) is not BlinkDetector):
        pipeline.close()
        raise DemoError("Unexpected FaceAnalysis implementation or prerequisite modules")
    for attr in ("orientation", "quality", "gaze", "gaze_estimation", "mouth",
                 "mouth_motion", "emotion", "regions", "temporal"):
        if getattr(pipeline, attr) is not None:
            pipeline.close()
            raise DemoError(f"Unrelated component {attr} is unexpectedly enabled")
    if (pipeline.blink.threshold != THRESHOLD
            or pipeline.blink.min_closed_frames != MIN_CLOSED_FRAMES
            or abs(pipeline.blink.fps - fps) > 1e-8):
        pipeline.close()
        raise DemoError("BlinkDetector configuration mismatch")
    return pipeline, eye_recorder


def _optional_real(value) -> float | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _rect(box, width: int, height: int):
    try:
        raw = np.asarray(box, dtype=np.float64).reshape(-1)
        if len(raw) != 4 or not np.isfinite(raw).all():
            return None
        x1, y1, x2, y2 = map(float, raw)
        if not (x2 > x1 and y2 > y1):
            return None
        clipped = (
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(0, min(width - 1, int(round(x2)))),
            max(0, min(height - 1, int(round(y2)))),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return (x1, y1, x2, y2), clipped
    except (TypeError, ValueError, OverflowError):
        return None


def identity_color(identifier: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(identifier).encode("ascii")).digest()
    return (75 + digest[0] % 140, 75 + digest[1] % 140, 75 + digest[2] % 140)


def draw_eye_points(canvas: np.ndarray, eyes: dict[str, Any]) -> int:
    h, w = canvas.shape[:2]
    count = 0
    for label in ("left", "right"):
        point_list = eyes["points"][label]
        if (len(point_list) != 6 or
                not all(math.isfinite(x) and math.isfinite(y) for x, y in point_list)):
            continue
        pts = [(int(round(x)), int(round(y))) for x, y in point_list]
        inside = [(0 <= x < w and 0 <= y < h) for x, y in pts]
        color = (245, 220, 35) if label == "left" else (235, 190, 35)
        # Only draw anatomically indicated eye landmarks from current inference.
        if all(inside):
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)],
                          isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        for (x, y), valid in zip(pts, inside):
            if valid:
                cv2.circle(canvas, (x, y), POINT_RADIUS, (255, 255, 255), -1,
                           lineType=cv2.LINE_AA)
                count += 1
    return count


def draw_text(canvas: np.ndarray, text: str, x: int, y: int,
              scale: float = 0.52, color=(255, 255, 255), weight: int = 1) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, weight, cv2.LINE_AA)


def draw_box_label(canvas: np.ndarray, rect, label: str, color) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    scale = max(0.4, min(0.64, canvas.shape[1] / 1700))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(x1, canvas.shape[1] - tw - 9))
    y = y1 - 8 if y1 > th + 14 else min(canvas.shape[0] - 2, y2 + th + 11)
    cv2.rectangle(canvas, (x, y - th - 5), (x + tw + 8, y + baseline + 4), color, -1)
    draw_text(canvas, label, x + 4, y, scale, (0, 0, 0))


def draw_signal(canvas: np.ndarray, history: deque[float | None], threshold: float,
                x: int, y: int, width: int, height: int) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (85, 85, 85), 1)
    vals = list(history)
    if not vals:
        return
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    ymax = max(0.52, (max(finite) + 0.08) if finite else 0.52)
    def y_coord(value: float) -> int:
        return y + height - int(np.clip(value / ymax, 0, 1) * height)
    ty = y_coord(threshold)
    cv2.line(canvas, (x, ty), (x + width, ty), (45, 187, 225), 1)
    prev = None
    for index, val in enumerate(vals):
        xp = x + int(index * (width - 1) / max(1, HISTORY_LENGTH - 1))
        if val is None or not math.isfinite(val):
            prev = None
        else:
            pt = (xp, y_coord(val))
            if prev is not None:
                cv2.line(canvas, prev, pt, (223, 223, 223), 2, cv2.LINE_AA)
            prev = pt


def draw_frame(frame: np.ndarray, instances, eye_recorder: RecordingEyeOpenness,
               fps: float, frame_index: int, histories, seen_ids,
               event_counts: dict[int, int]) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    height, width = frame.shape[:2]
    canvas = np.zeros((height + height % 2, width + SIDEBAR_WIDTH + ((width + SIDEBAR_WIDTH) % 2), 3), dtype=np.uint8)
    canvas[:height, :width] = frame
    canvas[:, width:] = (29, 29, 29)
    eye_calls = iter(eye_recorder.frame_calls)
    rows = []
    face_summaries = []
    eye_landmark_points_drawn = 0
    blink_pulses = 0
    seen_frame_ids = set()
    for i, face in enumerate(instances):
        feature = face.face_features
        lm_avail = bool(feature["landmarks"]["available"])
        recorded = next(eye_calls, None) if lm_avail else None
        if lm_avail and recorded is None:
            raise DemoError("Missing eye landmark capture for an available face")
        eyes = feature["eyes"]
        blink = feature["blink"]
        eye_avail = bool(eyes["available"])
        blink_avail = bool(blink["available"])
        confidence = _optional_real(face.confidence)
        ident = face.id
        identity = None
        if ident is not None:
            val = _optional_real(ident)
            if val is None or val < 0 or int(val) != val:
                raise DemoError("Invalid tracked face identifier")
            identity = int(val)
            if identity in seen_frame_ids:
                raise DemoError("Duplicate tracking ID in one frame")
            seen_frame_ids.add(identity)
            seen_ids.add(identity)
        means = [_optional_real(eyes.get(k)) for k in
                 ("left_openness", "right_openness", "mean_openness")]
        if eye_avail != all(x is not None for x in means):
            raise DemoError("EyeOpenness availability flag disagrees with recorded values")
        state = str(blink.get("eye_state", "unknown"))
        if (blink_avail and state not in ("open", "closed")) or (
                not blink_avail and state not in ("unknown", "open", "closed")):
            raise DemoError("Invalid reported eye state")
        if eye_avail and blink_avail:
            if state != ("closed" if means[2] < THRESHOLD else "open"):
                raise DemoError("BlinkDetector eye state does not match configured threshold")
        pulse = bool(blink.get("blink", False))
        if pulse and not (blink_avail and identity is not None and state == "open"):
            raise DemoError("Invalid blink pulse on unavailable or closed eye")
        blink_count = blink.get("blink_count", 0)
        if not isinstance(blink_count, (int, np.integer)) or blink_count < 0:
            raise DemoError("Invalid blink counter")
        if identity is not None:
            if pulse:
                event_counts[identity] += 1
                blink_pulses += 1
            if blink_count != event_counts[identity]:
                raise DemoError("Blink count is inconsistent with observed blink pulses")
            histories[identity].append(means[2] if eye_avail else None)
        if lm_avail and recorded is not None:
            for k, outkey in (("left", "left_openness"),
                              ("right", "right_openness")):
                measured = _optional_real(recorded["openness"].get(outkey))
                result_value = _optional_real(eyes.get(outkey))
                if measured != result_value:
                    raise DemoError("Rendered landmarks do not correspond to EyeOpenness outputs")
            eye_landmark_points_drawn += draw_eye_points(canvas, recorded)
        coords = _rect(face.box, width, height)
        rect_status = "INVALID_BOX"
        if coords is not None:
            raw, clipped = coords
            rect_status = "DRAWN"
            color = identity_color(identity) if identity is not None else (160, 160, 160)
            label = f"ID {identity if identity is not None else '?'}  "
            label += f"EAR {means[2]:.3f}" if eye_avail else "EAR --"
            label += f"  {state.upper()}  B {blink_count}"
            draw_box_label(canvas, clipped, label, color)
        else:
            raw = (None, None, None, None)
        rows.append({
            "source_frame": frame_index + 1, "time_s": round(frame_index / fps, 6),
            "face_index": i, "track_id": identity,
            "detection_confidence": confidence,
            "box_x1": raw[0], "box_y1": raw[1], "box_x2": raw[2], "box_y2": raw[3],
            "landmarks_available": int(lm_avail), "eyes_available": int(eye_avail),
            "left_openness": means[0], "right_openness": means[1],
            "mean_openness": means[2], "blink_available": int(blink_avail),
            "eye_state": state, "blink_pulse": int(pulse),
            "blink_count": int(blink_count),
            "blink_duration_s": _optional_real(blink.get("blink_duration")),
            "blink_rate_per_min": _optional_real(blink.get("blink_rate")),
            "drawing_status": rect_status,
        })
        face_summaries.append((identity, means[2] if eye_avail else None,
                               state, blink_count, pulse))
    if next(eye_calls, None) is not None:
        raise DemoError("Extra eye landmark capture was not matched to a face")
    if not rows:
        rows = [{"source_frame": frame_index + 1,
                 "time_s": round(frame_index / fps, 6), "eye_state": "no_face",
                 "drawing_status": "NO_FACE"}]
    right = width + 15
    draw_text(canvas, "PhysioTrack-v2", right, 30, 0.72, (250, 250, 250), 2)
    draw_text(canvas, "Eye Openness + Blink", right, 58, 0.58, (240, 240, 240), 1)
    cv2.line(canvas, (right, 76), (width + SIDEBAR_WIDTH - 15, 76), (95, 95, 95), 1)
    draw_text(canvas, "SYNTHETIC QUALITATIVE DEMO", right, 102, 0.48, (105, 224, 240))
    draw_text(canvas, "Not a benchmark / no ground truth", right, 127, 0.43)
    draw_text(canvas, f"Frame {frame_index + 1}   Time {frame_index / fps:.2f}s", right, 162, 0.53)
    draw_text(canvas, f"Tracked faces: {len(face_summaries)}", right, 192, 0.55)
    draw_text(canvas, f"Threshold {THRESHOLD:.2f}  Min closed {MIN_CLOSED_FRAMES}",
              right, 220, 0.49)
    draw_text(canvas, f"Detected blink pulses: {sum(event_counts.values())}",
              right, 249, 0.49, (140, 250, 140))
    cv2.line(canvas, (right, 264), (width + SIDEBAR_WIDTH - 15, 264), (95, 95, 95), 1)
    draw_text(canvas, "Current tracked persons", right, 290, 0.51)
    max_display = max(1, min(6, (height - 380) // 27))
    for index, (ident, mean, state, count, pulse) in enumerate(face_summaries[:max_display]):
        label = f"ID {ident if ident is not None else '?'} | EAR "
        label += f"{mean:.3f}" if mean is not None else "--"
        label += f" | {state[:3].upper()} | B {count}"
        if pulse:
            label += " *"
        draw_text(canvas, label, right, 319 + 27 * index, 0.42,
                  (130, 245, 150) if pulse else (235, 235, 235))
    if len(face_summaries) > max_display:
        draw_text(canvas, f"+ {len(face_summaries) - max_display} more", right,
                  319 + 27 * max_display, 0.44)
    graph_y = max(380, height - 165)
    draw_text(canvas, "Rolling EyeOpenness", right, graph_y - 14, 0.46)
    eligible = sorted(i for i in seen_frame_ids if i in histories)
    selected = eligible[0] if eligible else None
    if selected is not None:
        draw_signal(canvas, histories[selected], THRESHOLD, right, graph_y,
                    SIDEBAR_WIDTH - 30, 98)
        draw_text(canvas, f"Signal: track {selected} | yellow: threshold", right,
                  graph_y + 117, 0.39)
    else:
        draw_text(canvas, "No current tracked person", right,
                  graph_y + 35, 0.44)
    return canvas, rows, {
        "landmark_points_drawn": eye_landmark_points_drawn,
        "blink_pulses": blink_pulses,
        "face_instances": len(face_summaries),
        "eyes_available": sum(int(r.get("eyes_available", 0)) for r in rows),
        "faces_with_landmarks": sum(int(r.get("landmarks_available", 0)) for r in rows),
    }


def open_video(source: Path, work_dir: Path, serial: int):
    cap = cv2.VideoCapture(str(source))
    copy = None
    if not cap.isOpened():
        cap.release()
        copy = work_dir / f"ascii_input_{serial:03d}{source.suffix.lower()}"
        shutil.copyfile(source, copy)
        cap = cv2.VideoCapture(str(copy))
    if not cap.isOpened():
        cap.release()
        raise DemoError(f"Unable to open video: {source.name}")
    return cap, copy


def open_writer(path: Path, width: int, height: int, fps: float):
    for codec in OUTPUT_CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                 fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
        path.unlink(missing_ok=True)
    raise DemoError("No MP4 video encoder is available in OpenCV")


def verify_video(path: Path, width: int, height: int, fps: float,
                 expected_frames: int) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise DemoError(f"Produced MP4 cannot be decoded: {path.name}")
    count = 0
    try:
        size = (int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if size != (width, height):
            raise DemoError(f"Incorrect output dimensions in {path.name}")
        if not math.isfinite(actual_fps) or abs(actual_fps - fps) > max(0.05, fps * 0.01):
            raise DemoError(f"Incorrect FPS in {path.name}")
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if img is None or img.shape != (height, width, 3):
                raise DemoError(f"Malformed output frame in {path.name}")
            count += 1
        if count != expected_frames:
            raise DemoError(f"Output frame count mismatch: {path.name} ({count} != {expected_frames})")
    finally:
        cap.release()


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".png", image)
    if not ok:
        raise DemoError(f"Failed to encode PNG: {path}")
    enc.tofile(str(path))


def process_video(source: Path, destination_video: Path, destination_csv: Path,
                  destination_image: Path, detector: Face, work_dir: Path,
                  serial: int) -> dict[str, Any]:
    cap, ascii_copy = open_video(source, work_dir, serial)
    writer = None
    pipeline = None
    tmp_video = work_dir / f"render_{serial:03d}.mp4"
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared_frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not (width > 0 and height >= 260 and math.isfinite(fps) and fps > 0):
            raise DemoError("Invalid video resolution or frame rate")
        pipeline, eye_recorder = make_pipeline(detector, fps)
        render_width = width + SIDEBAR_WIDTH + (width + SIDEBAR_WIDTH) % 2
        render_height = height + height % 2
        writer, codec = open_writer(tmp_video, render_width, render_height, fps)
        destination_csv.parent.mkdir(parents=True, exist_ok=True)
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        video_frames = 0
        total_face_instances = 0
        total_eye_avail = 0
        total_lm_avail = 0
        total_points = 0
        frames_without_faces = 0
        event_counts: dict[int, int] = defaultdict(int)
        histories = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
        seen_ids: set[int] = set()
        first_frame_image = None
        first_eyes_image = None
        first_eyes_frame_number = None
        selected_blink_image = None
        selected_frame_number = None
        selected_from_closure = False
        last_closed_image_by_id: dict[int, tuple[int, np.ndarray]] = {}
        with destination_csv.open("w", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="raise")
            csv_writer.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape != (height, width, 3):
                    raise DemoError("An input frame was missing or changed dimensions")
                eye_recorder.begin_frame()
                result = pipeline.predict(frame)
                annotated, rows, counts = draw_frame(
                    frame, list(result), eye_recorder, fps, video_frames,
                    histories, seen_ids, event_counts,
                )
                csv_writer.writerows(rows)
                writer.write(annotated)
                if first_frame_image is None:
                    first_frame_image = annotated.copy()
                if first_eyes_image is None and counts["eyes_available"]:
                    first_eyes_image = annotated.copy()
                    first_eyes_frame_number = video_frames + 1
                if selected_blink_image is None and counts["blink_pulses"]:
                    for row in rows:
                        if int(row.get("blink_pulse", 0)) != 1:
                            continue
                        identity = row.get("track_id")
                        earlier = last_closed_image_by_id.get(identity)
                        if earlier is not None and earlier[0] == video_frames:
                            selected_frame_number, selected_blink_image = earlier
                            selected_from_closure = True
                        else:
                            selected_blink_image = annotated.copy()
                            selected_frame_number = video_frames + 1
                        break
                # Use the actual last closed frame as the representative
                # image when a blink is confirmed on reopening. This does not
                # change blink timing, counts, or the underlying algorithm.
                for row in rows:
                    ident = row.get("track_id")
                    if ident is not None and row.get("eye_state") == "closed":
                        last_closed_image_by_id[ident] = (video_frames + 1, annotated.copy())
                total_face_instances += counts["face_instances"]
                total_eye_avail += counts["eyes_available"]
                total_lm_avail += counts["faces_with_landmarks"]
                total_points += counts["landmark_points_drawn"]
                if counts["face_instances"] == 0:
                    frames_without_faces += 1
                video_frames += 1
                if video_frames % 100 == 0:
                    print(f"    {video_frames} frames; eye samples {total_eye_avail}; "
                          f"predicted blink pulses {sum(event_counts.values())}", flush=True)
        if video_frames == 0:
            raise DemoError("Input video has no decodable frames")
        if declared_frames > 0 and abs(declared_frames - video_frames) > 2:
            raise DemoError("Input frame count differs from video metadata")
        writer.release()
        writer = None
        verify_video(tmp_video, render_width, render_height, fps, video_frames)
        image = (selected_blink_image if selected_blink_image is not None
                 else first_eyes_image if first_eyes_image is not None
                 else first_frame_image)
        save_image(image, destination_image)
        destination_video.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_video, destination_video)
        return {
            "input_width": width, "input_height": height, "fps": fps,
            "declared_frames": declared_frames if declared_frames > 0 else None,
            "processed_frames": video_frames, "frames_without_tracked_faces": frames_without_faces,
            "face_observations": total_face_instances, "landmark_available_observations": total_lm_avail,
            "eye_available_observations": total_eye_avail,
            "landmark_points_rendered": total_points,
            "predicted_blink_pulses": sum(event_counts.values()),
            "predicted_blinks_per_track": {str(i): event_counts[i] for i in sorted(seen_ids)},
            "unique_track_ids": len(seen_ids),
            "representative_frame": selected_frame_number or first_eyes_frame_number or 1,
            "representative_reason": (
                "last_closed_frame_before_first_predicted_blink" if selected_from_closure else
                "first_predicted_blink_reopen" if selected_blink_image is not None else
                "first_eyes_available" if first_eyes_image is not None else "first_frame"
            ),
            "output_codec": codec, "verified_output_frames": video_frames,
        }
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if pipeline is not None:
            pipeline.close()
        tmp_video.unlink(missing_ok=True)
        if ascii_copy is not None:
            ascii_copy.unlink(missing_ok=True)


def make_contact_sheet(stage: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise DemoError("No successful records for contact sheet")
    cells = []
    for record in records:
        image = cv2.imdecode(np.fromfile(str(stage / record["output_image"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise DemoError("Representative image is missing or undecodable")
        canvas = np.full((SHEET_CELL_HEIGHT, SHEET_CELL_WIDTH, 3), 20, dtype=np.uint8)
        h, w = image.shape[:2]
        scale = min((SHEET_CELL_WIDTH - 22) / w,
                    (SHEET_CELL_HEIGHT - 45) / h)
        img = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
        x = (SHEET_CELL_WIDTH - img.shape[1]) // 2
        y = 40 + (SHEET_CELL_HEIGHT - 45 - img.shape[0]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        title = Path(record["input"]).name[:56]
        draw_text(canvas, title, 12, 27, 0.62, (242, 242, 242))
        cells.append(canvas)
    columns = 2 if len(cells) > 1 else 1
    rows = math.ceil(len(cells) / columns)
    sheet = np.full((rows * SHEET_CELL_HEIGHT, columns * SHEET_CELL_WIDTH, 3),
                    20, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, columns)
        sheet[r * SHEET_CELL_HEIGHT:(r + 1) * SHEET_CELL_HEIGHT,
              c * SHEET_CELL_WIDTH:(c + 1) * SHEET_CELL_WIDTH] = cell
    save_image(sheet, stage / "synthetic_demo_examples.png")


def _summary_text(summary: dict[str, Any], heading: str) -> str:
    """Human-readable lossless enumeration of EVERY field in a JSON summary."""
    lines = [heading, '=' * len(heading), '']

    def emit(value: Any, indent: int = 0) -> None:
        prefix = '  ' * indent
        if isinstance(value, dict):
            for name, child in value.items():
                if isinstance(child, (dict, list)):
                    lines.append(f'{prefix}{name}:')
                    emit(child, indent + 1)
                else:
                    lines.append(f'{prefix}{name}: {json.dumps(child, ensure_ascii=False, allow_nan=False)}')
        elif isinstance(value, list):
            for index, child in enumerate(value, 1):
                if isinstance(child, (dict, list)):
                    lines.append(f'{prefix}[{index}]')
                    emit(child, indent + 1)
                else:
                    lines.append(f'{prefix}[{index}] {json.dumps(child, ensure_ascii=False, allow_nan=False)}')
        else:
            lines.append(f'{prefix}{json.dumps(value, ensure_ascii=False, allow_nan=False)}')

    emit(summary)
    return '\n'.join(lines) + '\n'


def _save_summary_pair(stage: Path, stem: str, summary: dict[str, Any], heading: str) -> None:
    with (stage / f'{stem}.json').open('w', encoding='utf-8', newline='\n') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    (stage / f'{stem}.txt').write_text(_summary_text(summary, heading), encoding='utf-8', newline='\n')


def _verify_summary_pair(stage: Path, stem: str, summary: dict[str, Any], heading: str) -> None:
    with (stage / f'{stem}.json').open('r', encoding='utf-8') as f:
        if json.load(f) != summary:
            raise DemoError(f'{stem}.json does not match validated summary')
    txt = (stage / f'{stem}.txt').read_text(encoding='utf-8')
    if txt != _summary_text(summary, heading):
        raise DemoError(f'{stem}.txt does not match validated summary')


QUALITATIVE_SUMMARY_HEADING = 'PhysioTrack Synthetic Eye Openness and Blink - Qualitative Summary'
ACCURACY_SUMMARY_HEADING = 'PhysioTrack Synthetic Eye Openness and Blink - Accuracy Summary'


def verify_stage(stage: Path, summary: dict[str, Any]) -> None:
    """Verify the COMPLETE staged/published qualitative directory and its ownership."""
    if summary.get('status') != 'COMPLETE' or summary.get('failed_videos') != 0:
        raise DemoError('Incomplete qualitative run cannot be published')
    items = summary.get('files')
    if not isinstance(items, list) or not items or len(items) != summary.get('total_videos'):
        raise DemoError('Qualitative video inventory is incomplete')
    expected_paths = {'synthetic_demo_summary.json', 'synthetic_demo_summary.txt',
                      'synthetic_demo_examples.png'}
    total_frames = total_pulses = total_faces = total_eyes = 0
    for item in items:
        if item.get('status') != 'SUCCESS':
            raise DemoError('A failed video was included in the published summary')
        mp4, csv_path, png = (stage / item[key] for key in
                              ('output_video', 'output_csv', 'output_image'))
        expected_paths.update(item[key] for key in ('output_video', 'output_csv', 'output_image'))
        if not all(p.is_file() and p.stat().st_size > 0 for p in (mp4, csv_path, png)):
            raise DemoError('Missing or empty qualitative video/CSV/PNG')
        image = np.fromfile(str(png), dtype=np.uint8)
        if cv2.imdecode(image, cv2.IMREAD_COLOR) is None:
            raise DemoError('Cannot decode qualitative PNG')
        expected_width = int(item['input_width']) + SIDEBAR_WIDTH
        expected_width += expected_width % 2
        expected_height = int(item['input_height']) + int(item['input_height']) % 2
        verify_video(mp4, expected_width, expected_height,
                     float(item['fps']), int(item['processed_frames']))
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise DemoError('Incorrect qualitative observations CSV columns')
            frame_counts: dict[int, int] = defaultdict(int)
            faces = eyes = pulses = no_face = 0
            for row in reader:
                idx = int(row['source_frame'])
                if idx < 1 or idx > item['processed_frames']:
                    raise DemoError('Qualitative CSV frame is out of range')
                frame_counts[idx] += 1
                if row['drawing_status'] == 'NO_FACE':
                    no_face += 1
                    if (row['track_id'] or row['blink_pulse']):
                        raise DemoError('NO_FACE placeholder contains a track or blink')
                else:
                    faces += 1
                    eyes += int(row['eyes_available'] or '0')
                    pulses += int(row['blink_pulse'] or '0')
            if (len(frame_counts) != item['processed_frames'] or
                    no_face != item['frames_without_tracked_faces'] or
                    faces != item['face_observations'] or
                    eyes != item['eye_available_observations'] or
                    pulses != item['predicted_blink_pulses']):
                raise DemoError('Qualitative CSV and summary accounting disagree')
        total_frames += item['processed_frames']
        total_pulses += pulses
        total_faces += faces
        total_eyes += eyes
    contact = np.fromfile(str(stage / 'synthetic_demo_examples.png'), dtype=np.uint8)
    if cv2.imdecode(contact, cv2.IMREAD_COLOR) is None:
        raise DemoError('Qualitative contact sheet cannot be decoded')
    if (total_frames != summary['total_processed_frames'] or
            total_pulses != summary['total_predicted_blink_pulses'] or
            total_faces != summary['total_face_observations'] or
            total_eyes != summary['total_eye_available_observations'] or
            summary['successful_videos'] != len(items)):
        raise DemoError('Qualitative aggregate summary accounting mismatch')
    _verify_summary_pair(stage, 'synthetic_demo_summary', summary,
                         QUALITATIVE_SUMMARY_HEADING)
    actual_paths = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    if actual_paths != expected_paths:
        raise DemoError('Qualitative output inventory contains missing or unexpected files')


# === Independent synthetic-video annotation and accuracy-evaluation support ===
# The default command runs the qualitative pipeline AND this accuracy stage.
ACCURACY_GT_PATH = INPUT_DIR / 'synthetic_blink_ground_truth.json'
ACCURACY_DIR = RESULTS_DIR / 'synthetic_demo_accuracy'
ACCURACY_IOU = 0.50
ACC_PERSON_COLUMNS = ('video','person_label','track_id','gt_blinks','predicted_blinks',
    'tp','fp','fn','precision','recall','f1','matched_mean_tiou','prolonged_closures')
ACC_VIDEO_COLUMNS = ('video','gt_blinks','predicted_blinks','tp','fp','fn',
    'precision','recall','f1','matched_mean_tiou','prolonged_closures','fps','frames')
ACC_EVENT_COLUMNS = ('video','person_label','track_id','outcome',
    'gt_start','gt_end','pred_start','pred_end','temporal_iou')


def _load_completed_demo() -> dict[str, Any]:
    file = FINAL_DIR / 'synthetic_demo_summary.json'
    if not file.is_file():
        raise DemoError('Previous qualitative results are missing. Run the demo first.')
    with file.open('r',encoding='utf-8') as f:
        summary = json.load(f)
    if (summary.get('project') != 'physiotrack-v2' or
            summary.get('status') != 'COMPLETE' or
            summary.get('failed_videos') != 0 or
            not summary.get('files')):
        raise DemoError('Completed synthetic demonstration summary is not valid')
    return summary


def _rows_for_video(summary_item: dict[str, Any], demo_dir: Path | None = None) -> list[dict[str, str]]:
    path = (demo_dir if demo_dir is not None else FINAL_DIR) / summary_item['output_csv']
    if not path.is_file():
        raise DemoError(f'Missing original observation CSV: {path}')
    with path.open('r',newline='',encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise DemoError(f'Unexpected source CSV schema: {path}')
        rows = list(reader)
    expected_rows = (summary_item['face_observations'] +
                     summary_item['frames_without_tracked_faces'])
    if len(rows) != expected_rows:
        raise DemoError(f'CSV row count inconsistent for {summary_item["input"]}')
    return rows


def generate_gt_template() -> None:
    """Create EMPTY, UNREVIEWED GT slots. Never infer GT from model pulses."""
    check_project_imports()
    summary = _load_completed_demo()
    template = {
        'schema_version': 1,
        'annotation_source': 'independent_manual_review_of_original_synthetic_video',
        'frame_indexing': '1_based_inclusive',
        'instructions': (
            'Review ORIGINAL videos frame by frame, independently of the model CSV. '
            'For each true blink, enter [start_frame,end_frame] from the first visible '
            'lid closure movement through the last blink/reopening frame. '
            'Put prolonged eye closures in prolonged_closure_intervals, NOT blink_intervals. '
            'For three persons, assign left/center/right labels using the original video '
            'and verify their track_id correspondence in the annotated video. '
            'Only set each persons reviewed=true and each video reviewed=true after '
            'all frames/events are manually checked; [] then means confirmed zero events. '
            'Tracker IDs are a manual mapping, not the source of GT. '
            'Synthetic accuracy is separate from MPEBlink benchmark.'
        ),
        'video_annotations': []
    }
    for item in summary['files']:
        rows = _rows_for_video(item)
        ids = sorted({int(r['track_id']) for r in rows if r['track_id'].strip()})
        expected = {int(k) for k in item['predicted_blinks_per_track']}
        if set(ids) != expected:
            raise DemoError(f'Track IDs do not agree with original summary: {item["input"]}')
        persons = []
        for index, track_id in enumerate(ids):
            label = ('left','center','right')[index] if len(ids) == 3 else f'person_{index+1}'
            persons.append({'person_label': label, 'track_id': track_id,
                'reviewed': False, 'blink_intervals': [], 'prolonged_closure_intervals': []})
        template['video_annotations'].append({
            'video': item['input'], 'source_sha256': item['source_sha256'],
            'frames': item['processed_frames'], 'fps': item['fps'],
            'reviewed': False, 'persons': persons,
        })
    ACCURACY_GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ACCURACY_GT_PATH.exists():
        raise DemoError(f'Ground-truth file already exists (not overwritten): {ACCURACY_GT_PATH}')
    with ACCURACY_GT_PATH.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(template,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('Created UNREVIEWED manual ground-truth template:',ACCURACY_GT_PATH)
    print('No GT events were guessed. Fill intervals and mark reviewed=true after manual checks.')


def _strict_int(value: Any, context: str) -> int:
    if type(value) is not int:
        raise DemoError(f'{context} must be an integer (not bool, string or float)')
    return value


def _validated_intervals(value: Any, context: str, frames: int):
    if not isinstance(value,list):
        raise DemoError(f'{context} must be a list of [start,end] frame pairs')
    intervals = []
    for i,pair in enumerate(value):
        if not isinstance(pair,list) or len(pair) != 2:
            raise DemoError(f'{context}[{i}] must contain two integer frame numbers')
        start = _strict_int(pair[0],context+' start')
        end = _strict_int(pair[1],context+' end')
        if not 1 <= start <= end <= frames:
            raise DemoError(f'{context}[{i}] outside 1..{frames}')
        intervals.append((start,end))
    intervals.sort()
    if any(a[1] >= b[0] for a,b in zip(intervals,intervals[1:])):
        raise DemoError(f'{context} contains overlapping events')
    return intervals


def load_verified_gt(summary: dict[str, Any], demo_dir: Path | None = None) -> dict[str, Any]:
    if not ACCURACY_GT_PATH.is_file():
        raise DemoError(f'GT annotations missing: {ACCURACY_GT_PATH}. '
                        'First run: python eye_blink_synthetic_demo.py --make-gt-template')
    with ACCURACY_GT_PATH.open('r',encoding='utf-8') as f:
        gt = json.load(f)
    if (gt.get('schema_version') != 1 or
            gt.get('annotation_source') != 'independent_manual_review_of_original_synthetic_video' or
            gt.get('frame_indexing') != '1_based_inclusive'):
        raise DemoError('GT provenance/schema or frame numbering is wrong')
    items = summary['files']
    annotations = gt.get('video_annotations')
    if not isinstance(annotations,list) or len(annotations) != len(items):
        raise DemoError('GT must include exactly one annotation for each source video')
    by_video = {}
    for ann in annotations:
        if not isinstance(ann,dict) or not isinstance(ann.get('video'),str):
            raise DemoError('Malformed video GT entry')
        if ann['video'] in by_video:
            raise DemoError('Repeated video GT entry')
        by_video[ann['video']] = ann
    if set(by_video) != {i['input'] for i in items}:
        raise DemoError('GT video names differ from the original synthetic demo')
    validated = {}
    for item in items:
        name = item['input']
        ann = by_video[name]
        if ann.get('reviewed') is not True:
            raise DemoError(f'Ground truth not independently reviewed: {name}')
        if (ann.get('source_sha256') != item['source_sha256'] or
                ann.get('frames') != item['processed_frames'] or
                abs(float(ann.get('fps',-1))-float(item['fps'])) > 1e-6):
            raise DemoError(f'GT source hash, frame count or FPS mismatch for {name}')
        source = INPUT_DIR / name
        if not source.is_file() or source_sha256(source) != item['source_sha256']:
            raise DemoError(f'Source video missing or modified since original demo: {name}')
        persons = ann.get('persons')
        if not isinstance(persons,list) or not persons:
            raise DemoError(f'GT persons are missing for {name}')
        seen_tracks,seen_labels = set(),set()
        normalized_persons = []
        for p in persons:
            if not isinstance(p,dict) or p.get('reviewed') is not True:
                raise DemoError(f'Person GT not marked reviewed in {name}')
            ident = p.get('track_id')
            if ident is not None:
                ident = _strict_int(ident,f'{name} track_id')
                if ident < 0 or ident in seen_tracks:
                    raise DemoError(f'Invalid or duplicate track ID in {name}')
                seen_tracks.add(ident)
            label = p.get('person_label')
            if not isinstance(label,str) or not label.strip() or label in seen_labels:
                raise DemoError(f'Invalid person label in {name}')
            seen_labels.add(label)
            blinks = _validated_intervals(p.get('blink_intervals'),
                                         f'{name}/{label} blink intervals',item['processed_frames'])
            closures = _validated_intervals(p.get('prolonged_closure_intervals'),
                                            f'{name}/{label} prolonged closures',item['processed_frames'])
            for a in blinks:
                if any(max(a[0],b[0]) <= min(a[1],b[1]) for b in closures):
                    raise DemoError(f'Blink/prolonged closure overlap in {name}/{label}')
            normalized_persons.append({'person_label':label,'track_id':ident,
                                        'blink_intervals':blinks,
                                        'prolonged_closure_intervals':closures})
        rows = _rows_for_video(item, demo_dir)
        model_ids = {int(r['track_id']) for r in rows if r['track_id'].strip()}
        if model_ids != seen_tracks:
            raise DemoError(f'GT person-to-track mapping is incomplete for {name}: '
                            f'expected {sorted(model_ids)}, got {sorted(seen_tracks)}. '
                            'Add any entirely missed persons with track_id=null.')
        validated[name] = {'persons':normalized_persons,'frames':item['processed_frames'],
                           'fps':float(item['fps'])}
    return validated


def _predictions_from_csv(item: dict[str,Any], demo_dir: Path | None = None):
    """Reconstruct *actual* pulse intervals, like mpeblink_blink_eval.build_events."""
    rows = _rows_for_video(item, demo_dir)
    by_track = defaultdict(list)
    pulse_counts = defaultdict(int)
    for row in rows:
        if not row['track_id'].strip():
            continue
        ident = int(row['track_id'])
        frame = int(row['source_frame'])
        pulse = int(row['blink_pulse'])
        if pulse not in (0,1):
            raise DemoError('Invalid CSV blink pulse')
        if pulse:
            if row['eye_state'] != 'open':
                raise DemoError('Blink pulse occurs on a non-open frame')
            duration = float(row['blink_duration_s'])
            if not math.isfinite(duration) or duration <= 0:
                raise DemoError('Pulse missing measured closed duration')
            closed_frames = int(round(duration*float(item['fps'])))
            start,end = frame-closed_frames,frame-1
            if start < 1 or end >= frame or closed_frames < MIN_CLOSED_FRAMES:
                raise DemoError('Pulse has invalid closed-frame interval')
            by_track[ident].append((start,end))
            pulse_counts[ident] += 1
    if sum(pulse_counts.values()) != item['predicted_blink_pulses']:
        raise DemoError(f'Predicted pulses disagree with original summary: {item["input"]}')
    if {str(k):pulse_counts[k] for k in sorted(int(x) for x in item['predicted_blinks_per_track'])} != item['predicted_blinks_per_track']:
        raise DemoError(f'Per-track pulse count disagrees with original summary: {item["input"]}')
    return dict(by_track),rows


def _temporal_iou_inclusive(first,second):
    intersection = max(0,min(first[1],second[1])-max(first[0],second[0])+1)
    union = first[1]-first[0]+1+second[1]-second[0]+1-intersection
    return intersection/union if union else 0.0


def _match_events_as_benchmark(predicted,ground_truth):
    """Identical greedy descending-tIoU one-to-one logic to MPEBlink evaluator."""
    candidates = []
    for pi,p in enumerate(predicted):
        for gi,g in enumerate(ground_truth):
            tiou = _temporal_iou_inclusive(p,g)
            if tiou >= ACCURACY_IOU:
                candidates.append((tiou,pi,gi))
    candidates.sort(reverse=True)
    used_p,used_g,matched = set(),set(),[]
    for overlap,pi,gi in candidates:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            matched.append((pi,gi,overlap))
    return matched,used_p,used_g


def _calculate_counts(gt_events,pred_events,matched):
    tp = len(matched)
    fp = len(pred_events)-tp
    fn = len(gt_events)-tp
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'gt_blinks':len(gt_events),'predicted_blinks':len(pred_events),
            'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
            'matched_mean_tiou':sum(m[2] for m in matched)/tp if tp else None}


def _save_csv_file(path:Path,columns,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def _calc_accuracy(summary,gt,demo_dir: Path | None = None):
    per_person,per_video,event_rows = [],[],[]
    annotated = {}
    all_matches=[]
    for item in summary['files']:
        name=item['input']
        source_gt=gt[name]
        by_track,rows = _predictions_from_csv(item, demo_dir)
        selected = {}
        micro={'gt_blinks':0,'predicted_blinks':0,'tp':0,'fp':0,'fn':0}
        all_iou=[]
        for p in source_gt['persons']:
            ident=p['track_id']
            ge=p['blink_intervals']
            pe=by_track.get(ident,[]) if ident is not None else []
            matched,used_p,used_g = _match_events_as_benchmark(pe,ge)
            s = _calculate_counts(ge,pe,matched)
            for k in micro: micro[k]+=s[k]
            all_iou.extend(m[2] for m in matched)
            p_result = {'video':name,'person_label':p['person_label'],
                'track_id':ident if ident is not None else '',
                **s,'prolonged_closures':len(p['prolonged_closure_intervals'])}
            per_person.append(p_result)
            for pi,gi,iou in matched:
                event_rows.append({'video':name,'person_label':p['person_label'],
                    'track_id':ident,'outcome':'TP','gt_start':ge[gi][0],
                    'gt_end':ge[gi][1],'pred_start':pe[pi][0],
                    'pred_end':pe[pi][1],'temporal_iou':iou})
            for pi,pred_event in enumerate(pe):
                if pi not in used_p:
                    event_rows.append({'video':name,'person_label':p['person_label'],
                        'track_id':ident,'outcome':'FP','gt_start':'','gt_end':'',
                        'pred_start':pred_event[0],'pred_end':pred_event[1],
                        'temporal_iou':''})
            for gi,gt_event in enumerate(ge):
                if gi not in used_g:
                    event_rows.append({'video':name,'person_label':p['person_label'],
                        'track_id':ident,'outcome':'FN','gt_start':gt_event[0],
                        'gt_end':gt_event[1],'pred_start':'','pred_end':'',
                        'temporal_iou':''})
            selected[p['person_label']] = p
        if micro['predicted_blinks'] != item['predicted_blink_pulses']:
            raise DemoError(f'Event count mismatch for {name}')
        scores=_calculate_counts([None]*micro['gt_blinks'],
            [None]*micro['predicted_blinks'],[('','',t) for t in all_iou])
        v_result={'video':name,**scores,
            'prolonged_closures':sum(len(p['prolonged_closure_intervals'])
                                      for p in source_gt['persons']),
            'fps':item['fps'],'frames':item['processed_frames']}
        per_video.append(v_result)
        annotated[name]={'item':item,'gt':source_gt,'scores':v_result,'rows':rows}
        all_matches.extend(all_iou)
    aggregate_counts={k:sum(p[k] for p in per_person) for k in
                       ('gt_blinks','predicted_blinks','tp','fp','fn')}
    overall=_calculate_counts([None]*aggregate_counts['gt_blinks'],
        [None]*aggregate_counts['predicted_blinks'],[('','',t) for t in all_matches])
    if overall['predicted_blinks']!=summary['total_predicted_blink_pulses']:
        raise DemoError('Overall pulse totals disagree with original summary')
    return per_person,per_video,event_rows,overall,annotated


def _frame_in_events(frame,intervals):
    return any(a<=frame<=b for a,b in intervals)


def _draw_gt_panel(img,first_width:int,video_gt,score,frame_num:int):
    """Use unused margin of the ORIGINAL sidebar without masking EAR chart."""
    h,w=img.shape[:2]
    right=first_width+15
    if w < first_width+SIDEBAR_WIDTH or h < 590:
        raise DemoError('Annotated video panel has unexpected dimensions')
    cv2.rectangle(img,(right-3,107),(w-8,137),(29,29,29),-1)
    draw_text(img,'Manual synthetic GT (not MPEBlink)',right,126,0.41,(105,224,240))
    start_y=max(415,h-305)
    if start_y+136 >= h-165:
        raise DemoError('Insufficient sidebar space for GT metrics')
    cv2.rectangle(img,(right-3,start_y-19),(w-8,start_y+133),(29,29,29),-1)
    draw_text(img,'Full-video event metrics (post-hoc)',right,start_y,0.44,(235,235,235))
    draw_text(img,f'GT blinks {score["gt_blinks"]} | Pred {score["predicted_blinks"]}',
              right,start_y+25,0.48,(220,220,220))
    draw_text(img,f'TP {score["tp"]}   FP {score["fp"]}   FN {score["fn"]}',
              right,start_y+49,0.49,(230,230,230))
    draw_text(img,f'P {score["precision"]:.3f}  R {score["recall"]:.3f}  F1 {score["f1"]:.3f}',
              right,start_y+73,0.46,(170,245,170))
    blink_active=[str(p['track_id'] if p['track_id'] is not None else p['person_label'])
                  for p in video_gt['persons'] if _frame_in_events(frame_num,p['blink_intervals'])]
    closure_active=[str(p['track_id'] if p['track_id'] is not None else p['person_label'])
                    for p in video_gt['persons'] if _frame_in_events(frame_num,p['prolonged_closure_intervals'])]
    draw_text(img,'GT blink active: '+(','.join(blink_active) if blink_active else 'none'),
              right,start_y+95,0.43,(180,200,250))
    draw_text(img,'GT long closure: '+(','.join(closure_active) if closure_active else 'none'),
              right,start_y+116,0.43,(185,190,245))
    return img


def _choose_gt_example_frame(video_name:str,video_gt,item):
    """Prefer relevant MANUALLY LABELED event, never select based on prediction."""
    prolonged=sorted([evt for p in video_gt['persons']
        for evt in p['prolonged_closure_intervals']])
    blink=sorted([evt for p in video_gt['persons'] for evt in p['blink_intervals']])
    if 'prolonged_closure' in video_name and prolonged:
        ev=prolonged[0]
    elif blink:
        ev=blink[0]
    elif prolonged:
        ev=prolonged[0]
    else:
        return int(item['representative_frame'])
    return (ev[0]+ev[1])//2


def _accuracy_contact_sheet(stage,accuracy_records):
    cells=[]
    for rec in accuracy_records:
        img=cv2.imdecode(np.fromfile(str(stage/rec['accuracy_image']), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: raise DemoError('Missing accuracy example image')
        cell=np.full((SHEET_CELL_HEIGHT,SHEET_CELL_WIDTH,3),20,dtype=np.uint8)
        h,w=img.shape[:2]
        scale=min((SHEET_CELL_WIDTH-22)/w,(SHEET_CELL_HEIGHT-67)/h)
        resized=cv2.resize(img,(max(1,round(w*scale)),max(1,round(h*scale))),
                           interpolation=cv2.INTER_AREA)
        x=(SHEET_CELL_WIDTH-resized.shape[1])//2
        y=56+(SHEET_CELL_HEIGHT-67-resized.shape[0])//2
        cell[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        sc=rec['scores']
        draw_text(cell,Path(rec['video']).name[:55],12,24,0.60)
        draw_text(cell,(f'GT {sc["gt_blinks"]} | Pred {sc["predicted_blinks"]} | '
                        f'P {sc["precision"]:.2f}  R {sc["recall"]:.2f}  F1 {sc["f1"]:.2f}'),
                  12,45,0.46,(170,245,170))
        cells.append(cell)
    cols=2 if len(cells)>1 else 1
    sheet=np.full((math.ceil(len(cells)/cols)*SHEET_CELL_HEIGHT,
                   cols*SHEET_CELL_WIDTH,3),20,dtype=np.uint8)
    for i,cell in enumerate(cells):
        r,c=divmod(i,cols)
        sheet[r*SHEET_CELL_HEIGHT:(r+1)*SHEET_CELL_HEIGHT,
              c*SHEET_CELL_WIDTH:(c+1)*SHEET_CELL_WIDTH]=cell
    save_image(sheet,stage/'synthetic_accuracy_examples.png')


def _render_accuracy_videos(stage,accuracy_data,demo_dir: Path | None = None):
    records=[]
    for name,obj in accuracy_data.items():
        item,gt,score=obj['item'],obj['gt'],obj['scores']
        rel_base=Path(item['output_video']).name
        dest_video=Path('annotated_videos')/rel_base.replace('_annotated.mp4','_accuracy.mp4')
        dest_png=Path('annotated_images')/rel_base.replace('_annotated.mp4','_accuracy_example.png')
        source=(demo_dir if demo_dir is not None else FINAL_DIR)/item['output_video']
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened():
            cap.release()
            raise DemoError(f'Cannot decode source annotated video: {name}')
        writer=None
        frame_num=0
        chosen=_choose_gt_example_frame(name,gt,item)
        snapshot=None
        try:
            out_width=int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            out_height=int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            first_width=int(item['input_width'])
            fps=float(cap.get(cv2.CAP_PROP_FPS))
            if (out_width<first_width+SIDEBAR_WIDTH or
                    abs(fps-float(item['fps']))>0.05):
                raise DemoError('Original annotated video dimensions/FPS mismatch')
            (stage/dest_video).parent.mkdir(parents=True,exist_ok=True)
            writer,codec=open_writer(stage/dest_video,out_width,out_height,fps)
            while True:
                ok,img=cap.read()
                if not ok:break
                frame_num+=1
                img=_draw_gt_panel(img,first_width,gt,score,frame_num)
                writer.write(img)
                if frame_num==chosen:
                    snapshot=img.copy()
            writer.release(); writer=None
            if frame_num!=item['processed_frames'] or snapshot is None:
                raise DemoError('Accuracy annotation output frame count mismatch')
            verify_video(stage/dest_video,out_width,out_height,fps,frame_num)
            save_image(snapshot,stage/dest_png)
            records.append({'video':name,'accuracy_video':dest_video.as_posix(),
                            'accuracy_image':dest_png.as_posix(),
                            'example_frame':chosen,'scores':score,
                            'output_width':out_width, 'output_height':out_height,
                            'fps':fps, 'frames':frame_num})
            print(f'  GT overlay: {name}: TP {score["tp"]}, FP {score["fp"]}, FN {score["fn"]}, '
                  f'F1={score["f1"]:.4f}',flush=True)
        finally:
            cap.release()
            if writer is not None:writer.release()
    _accuracy_contact_sheet(stage,records)
    return records


def preflight_gt_for_inputs(videos: list[Path]) -> None:
    """Reject absent/stale/incomplete GT BEFORE expensive neural inference."""
    if not ACCURACY_GT_PATH.is_file():
        raise DemoError(
            f"Verified Ground Truth is required for the single-command workflow: "
            f"{ACCURACY_GT_PATH}. Place the reviewed JSON in synthetic_media first. "
            "The script never invents GT or scores its own predictions as GT."
        )
    with ACCURACY_GT_PATH.open("r", encoding="utf-8") as f:
        gt = json.load(f)
    if (gt.get("schema_version") != 1 or
            gt.get("annotation_source") != "independent_manual_review_of_original_synthetic_video" or
            gt.get("frame_indexing") != "1_based_inclusive"):
        raise DemoError("Unsupported GT provenance/schema/frame numbering")
    expected = set(EXPECTED_FILENAMES)
    actual = {p.relative_to(INPUT_DIR).as_posix() for p in videos}
    if len(videos) != len(expected) or actual != expected:
        raise DemoError(
            f"Exactly these five original videos are required for reproducible scoring: "
            f"{sorted(expected)}. Found: {sorted(actual)}"
        )
    annotations = gt.get("video_annotations")
    if not isinstance(annotations, list) or len(annotations) != len(expected):
        raise DemoError("GT must contain exactly five video annotations")
    ann_by_name = {}
    for ann in annotations:
        if not isinstance(ann, dict) or not isinstance(ann.get("video"), str):
            raise DemoError("Invalid GT video entry")
        if ann["video"] in ann_by_name:
            raise DemoError("Duplicate GT video name")
        ann_by_name[ann["video"]] = ann
    if set(ann_by_name) != expected:
        raise DemoError("GT video names do not match expected inputs")
    for video in videos:
        name = video.relative_to(INPUT_DIR).as_posix()
        ann = ann_by_name[name]
        if ann.get("reviewed") is not True:
            raise DemoError(f"GT video not reviewed: {name}")
        if ann.get("source_sha256") != source_sha256(video):
            raise DemoError(f"Source video SHA-256 does not match reviewed GT: {name}")
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise DemoError(f"Cannot open GT-matched source video: {name}")
            frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if (type(ann.get("frames")) is not int or frames != ann["frames"] or
                not math.isfinite(fps) or fps <= 0 or
                abs(float(ann.get("fps", -1)) - fps) > 1e-3):
            raise DemoError(f"GT frame count/FPS mismatch: {name}")
        people = ann.get("persons")
        if not isinstance(people, list) or not people:
            raise DemoError(f"Missing GT people: {name}")
        labels, ids = set(), set()
        for person in people:
            if not isinstance(person, dict) or person.get("reviewed") is not True:
                raise DemoError(f"Unreviewed person GT: {name}")
            label = person.get("person_label")
            if not isinstance(label, str) or not label.strip() or label in labels:
                raise DemoError(f"Invalid person GT label: {name}")
            labels.add(label)
            ident = person.get("track_id")
            if ident is not None:
                ident = _strict_int(ident, f"{name} track_id")
                if ident < 0 or ident in ids:
                    raise DemoError(f"Duplicated/invalid GT track_id: {name}")
                ids.add(ident)
            blinks = _validated_intervals(person.get("blink_intervals"),
                                          f"{name}/{label} blinks", frames)
            closures = _validated_intervals(person.get("prolonged_closure_intervals"),
                                            f"{name}/{label} prolonged closures", frames)
            if any(max(a[0], b[0]) <= min(a[1], b[1])
                   for a in blinks for b in closures):
                raise DemoError(f"Overlapping blink and sustained closure GT: {name}/{label}")
    print("Verified GT preflight: PASS (reviewed, matching SHA-256, frames, FPS, events)")


def verify_accuracy_stage(stage: Path, report: dict[str, Any]) -> None:
    """Validate the entire accuracy publication, including both summary formats."""
    if (report.get('NOT_mpeblink_benchmark') is not True or
            not isinstance(report.get('created'), list) or not report['created']):
        raise DemoError('Synthetic accuracy report is missing essential metadata')
    expected = {'synthetic_accuracy_summary.json', 'synthetic_accuracy_summary.txt',
                'accuracy_by_person.csv', 'accuracy_by_video.csv',
                'event_match_details.csv', 'synthetic_accuracy_examples.png'}
    for name, expected_length in (
            ('accuracy_by_person.csv', report['gt_persons']),
            ('accuracy_by_video.csv', report['gt_videos']),
            ('event_match_details.csv', sum(report['overall_micro'][key]
                                           for key in ('tp', 'fp', 'fn')))):
        with (stage / name).open('r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != expected_length:
            raise DemoError(f'Accuracy table {name} has incorrect row count')
        if name == 'accuracy_by_video.csv':
            if (sum(int(x['tp']) for x in rows) != report['overall_micro']['tp'] or
                    sum(int(x['fp']) for x in rows) != report['overall_micro']['fp'] or
                    sum(int(x['fn']) for x in rows) != report['overall_micro']['fn']):
                raise DemoError('Accuracy per-video and overall totals disagree')
    for record in report['created']:
        video, image = (stage / record[key] for key in ('accuracy_video', 'accuracy_image'))
        expected.update([record['accuracy_video'], record['accuracy_image']])
        if not video.is_file() or video.stat().st_size == 0:
            raise DemoError('Missing or empty accuracy video')
        if cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
            raise DemoError('Missing or invalid accuracy PNG')
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise DemoError('Accuracy video cannot be opened')
            w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        source = next((entry for entry in report['by_video'] if entry['video'] == record['video']), None)
        if source is None:
            raise DemoError('Accuracy video has no corresponding metrics row')
        if (w != record['output_width'] or h != record['output_height'] or
                not math.isfinite(fps) or abs(fps - record['fps']) > max(0.05, record['fps'] * 0.01) or
                abs(fps - float(source['fps'])) > max(0.05, float(source['fps']) * 0.01) or
                record['frames'] != source['frames']):
            raise DemoError('Accuracy video properties differ from recorded source metadata')
        verify_video(video, w, h, fps, int(source['frames']))
    overview = stage / 'synthetic_accuracy_examples.png'
    if cv2.imdecode(np.fromfile(str(overview), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
        raise DemoError('Synthetic accuracy contact sheet cannot be decoded')
    _verify_summary_pair(stage, 'synthetic_accuracy_summary', report,
                         ACCURACY_SUMMARY_HEADING)
    actual = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    if actual != expected:
        raise DemoError('Accuracy output inventory contains missing or unexpected files')


def _write_accuracy_stage(summary: dict[str, Any], demo_dir: Path,
                          stage: Path) -> dict[str, Any]:
    """Create/verify synthetic accuracy outputs WITHOUT changing published results."""
    gt = load_verified_gt(summary, demo_dir=demo_dir)
    per_person, per_video, event_rows, overall, annotated = _calc_accuracy(
        summary, gt, demo_dir=demo_dir)
    stage.mkdir(parents=True, exist_ok=False)
    _save_csv_file(stage / 'accuracy_by_person.csv', ACC_PERSON_COLUMNS, per_person)
    _save_csv_file(stage / 'accuracy_by_video.csv', ACC_VIDEO_COLUMNS, per_video)
    _save_csv_file(stage / 'event_match_details.csv', ACC_EVENT_COLUMNS, event_rows)
    files = _render_accuracy_videos(stage, annotated, demo_dir=demo_dir)
    report = {
        'project': 'physiotrack-v2',
        'evaluation': 'Independent manually annotated synthetic video accuracy',
        'NOT_mpeblink_benchmark': True,
        'annotation_file': ACCURACY_GT_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        'annotation_sha256': source_sha256(ACCURACY_GT_PATH),
        'source_qualitative_summary_sha256': source_sha256(
            demo_dir / 'synthetic_demo_summary.json'),
        'script_sha256': source_sha256(Path(__file__).resolve()),
        'matching': 'greedy one-to-one temporal IoU, inclusive event frames',
        'matching_threshold': ACCURACY_IOU,
        'frame_numbering': '1-based inclusive',
        'scoring_population': 'all annotated persons, per video; including track_id null if fully missed',
        'zero_division_convention': 'precision/recall/F1 shown as 0.0 if denominator zero; '
                                    'event count always shown alongside scores',
        'prolonged_closures_scoring': 'not GT blink events; model pulses on long closures count as FP '
            'unless they independently overlap annotated blink events by tIoU >= 0.50',
        'not_comparable_directly_to_mpeblink':
            'Artificial media, separately annotated from model results',
        'source_qualitative_outputs_preserved': demo_dir.resolve() == FINAL_DIR.resolve(),
        'gt_videos': len(per_video), 'gt_persons': len(per_person),
        'overall_micro': overall, 'by_video': per_video, 'created': files,
    }
    _save_summary_pair(stage, 'synthetic_accuracy_summary', report, ACCURACY_SUMMARY_HEADING)
    with (stage / 'accuracy_by_video.csv').open('r', newline='', encoding='utf-8') as f:
        check_rows = list(csv.DictReader(f))
    if len(check_rows) != len(per_video):
        raise DemoError('Incomplete accuracy-by-video table')
    if (sum(int(x['tp']) for x in check_rows) != overall['tp'] or
            sum(int(x['fp']) for x in check_rows) != overall['fp'] or
            sum(int(x['fn']) for x in check_rows) != overall['fn']):
        raise DemoError('Accuracy summary and CSV event totals disagree')
    with (stage / 'event_match_details.csv').open('r', newline='', encoding='utf-8') as f:
        event_records = list(csv.DictReader(f))
    if len(event_records) != overall['tp'] + overall['fp'] + overall['fn']:
        raise DemoError('Accuracy event detail row count disagrees with summary')
    for item in files:
        video = stage / item['accuracy_video']
        image = stage / item['accuracy_image']
        if not video.is_file() or not image.is_file() or cv2.imread(str(image)) is None:
            raise DemoError('Missing/invalid staged accuracy media')
    if cv2.imread(str(stage / 'synthetic_accuracy_examples.png')) is None:
        raise DemoError('Synthetic accuracy contact sheet could not be decoded')
    with (stage / 'synthetic_accuracy_summary.json').open('r', encoding='utf-8') as f:
        if json.load(f) != report:
            raise DemoError('Staged accuracy summary was not written correctly')
    verify_accuracy_stage(stage, report)
    return report


def _commit_output_pairs(
        staged: list[tuple[Path, Path]], staging_root: Path,
        post_publish_checks: list[Any] | None = None) -> None:
    """Replace exact owned directories, verify published results, roll back on failure.

    Normal successful completion deletes previous output snapshots. Never merges
    old and new files. Staging is on the same results filesystem as final outputs.
    """
    if post_publish_checks is not None and len(post_publish_checks) != len(staged):
        raise DemoError('Publication verification callbacks do not match outputs')
    destinations = [str(final.resolve(strict=False)).casefold() for _, final in staged]
    if len(destinations) != len(set(destinations)):
        raise DemoError('Duplicate publication destinations')
    for stage, final in staged:
        if not stage.is_dir() or stage.is_symlink():
            raise DemoError(f'Missing or unsafe staged result directory: {stage}')
        if final.parent.resolve() != RESULTS_DIR.resolve():
            raise DemoError('Publication target is outside this validation results directory')
        if final.is_symlink() or (final.exists() and not final.is_dir()):
            raise DemoError(f'Unsafe publication destination: {final}')
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for idx, (_, final) in enumerate(staged):
            if final.exists():
                backup = staging_root / f'previous_output_{idx}'
                if backup.exists():
                    raise DemoError('Unexpected pre-existing recovery backup')
                final.replace(backup)
                backups.append((backup, final))
        for stage, final in staged:
            stage.replace(final)
            published.append(final)
        if post_publish_checks is not None:
            for check in post_publish_checks:
                check()
    except Exception:
        rollback_errors = []
        for final in reversed(published):
            try:
                if final.is_dir() and not final.is_symlink():
                    shutil.rmtree(final)
            except OSError as error:
                rollback_errors.append(f'Cannot remove incomplete output {final}: {error}')
        for backup, original in reversed(backups):
            try:
                if original.exists():
                    raise OSError('Incomplete output still exists; cannot safely restore')
                backup.replace(original)
            except OSError as error:
                rollback_errors.append(f'Cannot restore previous output {original}: {error}')
        if rollback_errors:
            (staging_root / 'RECOVERY_REQUIRED.txt').write_text(
                'Publication rollback failed. Do not delete this staging directory.\n'
                + '\n'.join(rollback_errors) + '\n', encoding='utf-8')
            raise DemoError('Publication rollback was incomplete; recovery copies remain '
                            f'in {staging_root}. ' + '; '.join(rollback_errors))
        raise
    for backup, _ in backups:
        shutil.rmtree(backup)



def run_accuracy_evaluation() -> None:
    """Compatibility mode: assess already-completed demo outputs without inference."""
    check_project_imports()
    summary = _load_completed_demo()
    if ACCURACY_DIR.exists() and (ACCURACY_DIR.is_symlink() or not ACCURACY_DIR.is_dir()):
        raise DemoError('Unsafe previous accuracy output path')
    temp_root = Path(tempfile.mkdtemp(prefix='.accuracy_stage_', dir=RESULTS_DIR))
    try:
        report = _write_accuracy_stage(summary, FINAL_DIR, temp_root / 'accuracy_evaluation')
        _commit_output_pairs(
            [(temp_root / 'accuracy_evaluation', ACCURACY_DIR)], temp_root,
            [lambda: verify_accuracy_stage(ACCURACY_DIR, report)],
        )
        overall = report['overall_micro']
        print('\nSYNTHETIC ACCURACY EVALUATION: COMPLETE (separate from MPEBlink)')
        print('Videos:', report['gt_videos'], 'Persons:', report['gt_persons'])
        print('GT blink events:', overall['gt_blinks'], 'Predicted blink events:', overall['predicted_blinks'])
        print('TP:', overall['tp'], 'FP:', overall['fp'], 'FN:', overall['fn'])
        print('Precision:', f'{overall["precision"]:.4%}')
        print('Recall:', f'{overall["recall"]:.4%}')
        print('F1:', f'{overall["f1"]:.4%}')
        print('Saved:', ACCURACY_DIR)
    finally:
        if (temp_root / 'RECOVERY_REQUIRED.txt').exists():
            print(f'Recovery required; preserved staging folder: {temp_root}', file=sys.stderr)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

def main() -> None:
    paths = check_project_imports()
    videos = discover_videos()
    preflight_gt_for_inputs(videos)
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise DemoError("Unsafe final output path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".eye_blink_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    work_dir = staging_root / "work"
    work_dir.mkdir()
    records = []
    try:
        print("PhysioTrack-v2 EyeOpenness + Blink Synthetic Qualitative Demonstration")
        print("Videos:", len(videos))
        print("Input: validation/synthetic_media/eye_openness_and_blink_media/")
        print("Output: validation/eye_openness_and_blink/results/synthetic_demo/")
        detector = Face(device=DEVICE, conf=DETECTOR_CONFIDENCE,
                        iou=DETECTOR_IOU, verbose=False)
        known_outputs = set()
        for serial, source in enumerate(videos, 1):
            relative = source.relative_to(INPUT_DIR).as_posix()
            path_video, path_csv, path_image = output_names(source)
            for output in (path_video, path_csv, path_image):
                key = output.as_posix().casefold()
                if key in known_outputs:
                    raise DemoError("Output path collision between input files")
                known_outputs.add(key)
            record: dict[str, Any] = {"input": relative}
            print(f"[{serial}/{len(videos)}] {relative}", flush=True)
            try:
                record["source_sha256"] = source_sha256(source)
                measurements = process_video(
                    source, stage / path_video, stage / path_csv, stage / path_image,
                    detector, work_dir, serial,
                )
                record.update(measurements)
                record.update({
                    "status": "SUCCESS", "output_video": path_video.as_posix(),
                    "output_csv": path_csv.as_posix(), "output_image": path_image.as_posix(),
                })
                print(f"    OK: {measurements['processed_frames']} frames; "
                      f"{measurements['eye_available_observations']} available eye samples; "
                      f"{measurements['predicted_blink_pulses']} predicted pulses", flush=True)
            except Exception as exc:
                record["status"] = "FAILED"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {record['error']}", file=sys.stderr)
                traceback.print_exc()
            records.append(record)
        failed = [record for record in records if record["status"] != "SUCCESS"]
        if not failed:
            make_contact_sheet(stage, records)
        summary = {
            "project": "physiotrack-v2",
            "type": "synthetic_qualitative_eye_openness_blink_demonstration",
            "benchmark_evaluation": False,
            "ground_truth_used": False,
            "accuracy_metrics_computed": False,
            "dataset_used": "locally_supplied_synthetic_videos_only",
            "input_directory": "validation/synthetic_media/eye_openness_and_blink_media",
            "output_directory": "validation/eye_openness_and_blink/results/synthetic_demo",
            "expected_initial_videos": list(EXPECTED_FILENAMES),
            "component_imports": paths,
            "configuration": {
                "pipeline": "PhysioTrack FaceAnalysis",
                "face_detector": "PhysioTrack Face", "face_tracker": "PhysioTrack FaceTracker",
                "tracker_type": TRACKER_TYPE, "face_landmarker": "PhysioTrack FaceLandmarks",
                "eye_openness": "PhysioTrack EyeOpenness",
                "eye_recorder": "RecordingEyeOpenness delegates computation to EyeOpenness.predict",
                "blink_detector": "PhysioTrack BlinkDetector",
                "blink_threshold": THRESHOLD, "min_closed_frames": MIN_CLOSED_FRAMES,
                "landmarks_other_boxes_association": "preserved_in_original_FaceAnalysis_path",
                "device": DEVICE, "point_radius_pixels": POINT_RADIUS,
                "per_video_tracker_and_blink_state": "fresh_pipeline_per_video",
                "detector_confidence": DETECTOR_CONFIDENCE, "detector_iou": DETECTOR_IOU,
            },
            "methodological_note": (
                "Qualitative demonstration only: uses automatic detection/tracking, not "
                "MPEBlink ground-truth bounding boxes. Detection/tracking errors may "
                "affect downstream blink observations. The annotated blink pulses are "
                "model predictions, not validated blink counts. Videos may not contain "
                "biologically accurate blinks; no sensitivity, precision, recall or F1 "
                "is claimed. Input audio and original variable-frame timestamps are "
                "not retained by OpenCV re-encoding."
            ),
            "environment": {"python": platform.python_version(),
                            "opencv": cv2.__version__, "numpy": np.__version__,
                            "platform": platform.system()},
            "reproducibility": environment_fingerprint(),
            "status": "COMPLETE" if not failed else "FAILED_NO_REPLACEMENT",
            "total_videos": len(videos), "successful_videos": len(videos) - len(failed),
            "failed_videos": len(failed),
            "total_processed_frames": sum(i.get("processed_frames", 0) for i in records),
            "total_face_observations": sum(i.get("face_observations", 0) for i in records),
            "total_eye_available_observations": sum(
                i.get("eye_available_observations", 0) for i in records),
            "total_predicted_blink_pulses": sum(i.get("predicted_blink_pulses", 0) for i in records),
            "files": records,
        }
        _save_summary_pair(stage, 'synthetic_demo_summary', summary, QUALITATIVE_SUMMARY_HEADING)
        if failed:
            report_path = RESULTS_DIR / "synthetic_demo_last_attempt.json"
            with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", newline="\n", delete=False,
                    prefix=".synthetic_failed_", suffix=".json", dir=RESULTS_DIR) as f:
                temp_report = Path(f.name)
                json.dump(summary, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(temp_report, report_path)
            raise DemoError(f"{len(failed)} video(s) failed. Previous completed output "
                            "was not replaced. See synthetic_demo_last_attempt.json")
        verify_stage(stage, summary)
        # Stage the full GT-matched accuracy report and annotated videos from
        # this SAME inference run. Do not read stale, previously published CSVs.
        print("\nCreating synthetic accuracy tables, videos and images...", flush=True)
        accuracy_stage = staging_root / "accuracy_evaluation"
        accuracy_report = _write_accuracy_stage(summary, stage, accuracy_stage)
        _commit_output_pairs(
            [(stage, FINAL_DIR), (accuracy_stage, ACCURACY_DIR)], staging_root,
            [lambda: verify_stage(FINAL_DIR, summary),
             lambda: verify_accuracy_stage(ACCURACY_DIR, accuracy_report)],
        )
        print("\nSYNTHETIC QUALITATIVE + ACCURACY DEMONSTRATION: COMPLETE")
        print("Videos:", len(videos))
        print("Total frames:", summary["total_processed_frames"])
        print("Face observations:", summary["total_face_observations"])
        print("EyeOpenness available:", summary["total_eye_available_observations"])
        print("Predicted blink pulses (not GT):", summary["total_predicted_blink_pulses"])
        print("Saved: validation/eye_openness_and_blink/results/synthetic_demo/")
        print("Qualitative summaries: synthetic_demo_summary.json + synthetic_demo_summary.txt")
        overall = accuracy_report['overall_micro']
        print("Synthetic GT blinks:", overall['gt_blinks'])
        print("Synthetic predicted blink events:", overall['predicted_blinks'])
        print("Synthetic TP:", overall['tp'], "FP:", overall['fp'], "FN:", overall['fn'])
        print("Synthetic Precision:", f"{overall['precision']:.4%}")
        print("Synthetic Recall:", f"{overall['recall']:.4%}")
        print("Synthetic F1:", f"{overall['f1']:.4%}")
        print("Accuracy videos/tables:", "validation/eye_openness_and_blink/results/synthetic_demo_accuracy/")
        print("Accuracy summaries: synthetic_accuracy_summary.json + synthetic_accuracy_summary.txt")
        print("Synthetic results do NOT replace the MPEBlink benchmark.")
    finally:
        if (staging_root / 'RECOVERY_REQUIRED.txt').exists():
            print(f'Recovery required; preserved staging folder: {staging_root}', file=sys.stderr)
        else:
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PhysioTrack-v2 synthetic EyeOpenness + Blink demonstration")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--make-gt-template", action="store_true",
                      help="Create independently reviewable EMPTY GT slots from existing CSVs")
    mode.add_argument("--evaluate-only", action="store_true",
                      help="Compute synthetic precision/recall/F1 and render GT overlays from saved outputs")
    args = parser.parse_args()
    try:
        if args.make_gt_template:
            generate_gt_template()
        elif args.evaluate_only:
            run_accuracy_evaluation()
        else:
            main()  # One command: fresh inference + independent GT metrics + overlays
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
