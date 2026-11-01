"""Synthetic qualitative demonstration of the real PhysioTrack face-landmarks component.

Automatically processes supported images in:
    validation/synthetic_media/face_landmarks_media/

Publishes only under:
    validation/face_landmarks/results/synthetic_demo/

This script is qualitative only. It does not use 300-W ground truth, does not
compute NME, and does not alter accepted benchmark outputs.
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
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "face_landmarks_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
})
DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.45
MAX_DETECTIONS = 20
POINT_RADIUS = 2
POINT_THICKNESS = -1
EXPECTED_LANDMARKS_PER_FACE = 478

CSV_FIELDS = (
    "source_image", "face_index", "faces_in_image", "image_width", "image_height",
    "detector_confidence", "face_box_x1", "face_box_y1", "face_box_x2", "face_box_y2",
    "face_box_width", "face_box_height", "pipeline_landmarks_available",
    "pipeline_landmark_count", "landmark_index", "landmarks_in_face", "x_normalized",
    "y_normalized", "z_normalized", "x_pixel", "y_pixel", "within_image_bounds",
    "status", "failure_reason",
)

UNRELATED_PIPELINE_ATTRIBUTES = (
    "tracker", "orientation", "quality", "eyes", "blink", "gaze",
    "gaze_estimation", "mouth", "mouth_motion", "emotion", "regions", "temporal",
)


class MediaProcessingError(RuntimeError):
    """A media, component, or publication failure."""


class LandmarkCapture:
    """Capture all native landmark lists returned through the real FaceAnalysis path."""

    def __init__(self, landmarker: Any) -> None:
        self.landmarker = landmarker
        self.calls: list[Any] = []

    def predict_face(self, frame: np.ndarray, box: Any) -> Any:
        landmarks = self.landmarker.predict_face(frame, box)
        self.calls.append(landmarks)
        return landmarks

    def reset_capture(self) -> None:
        self.calls = []

    def close(self) -> None:
        if hasattr(self.landmarker, "close"):
            self.landmarker.close()


def check_project_imports() -> dict[str, str]:
    """Require implementation classes to originate from this project checkout."""
    expected_root = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    paths = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig):
        actual = Path(inspect.getfile(cls)).resolve()
        if not actual.is_relative_to(expected_root):
            raise RuntimeError(
                f"{cls.__name__} does not come from the local physiotrack-v2 checkout. "
                "Install the repository in editable mode in the active environment."
            )
        paths[cls.__name__] = actual.relative_to(REPOSITORY_ROOT).as_posix()
    return paths


def resolve_model_path() -> Path:
    """Resolve the MediaPipe face-landmarker model portably."""
    candidates: list[Path] = []

    env_model = os.environ.get("PHYSIOTRACK_FACE_LANDMARKER_MODEL")
    if env_model:
        candidates.append(Path(env_model))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data) / "physiotrack" / "weights" / "mediapipe" / "face_landmarker.task"
        )

    candidates.append(REPOSITORY_ROOT / "weights" / "mediapipe" / "face_landmarker.task")
    candidates.append(Path.home() / ".cache" / "physiotrack" / "weights" / "mediapipe" / "face_landmarker.task")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate face_landmarker.task.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Set PHYSIOTRACK_FACE_LANDMARKER_MODEL to an explicit model path if needed."
    )


def discover_images() -> list[Path]:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            "Media directory missing: validation/synthetic_media/face_landmarks_media/"
        )
    images = [
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(), p.relative_to(INPUT_DIR).as_posix()))
    if not images:
        raise MediaProcessingError(
            "No supported images found in validation/synthetic_media/face_landmarks_media/. "
            "Existing results were not modified."
        )
    return images


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_provenance(model_path: Path) -> dict[str, str]:
    """Identify the precise model bytes without a machine-specific absolute path."""
    resolved = model_path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        logical_path = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    else:
        logical_path = resolved.name
    return {"path": logical_path, "sha256": sha256_file(resolved)}


def dependency_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def summary_as_text(summary: dict[str, Any]) -> str:
    """Readable TXT containing the full data represented by the JSON summary."""
    lines = ["PhysioTrack Face Landmarks - Synthetic Qualitative Demonstration", "=" * 65, ""]

    def append_value(name: str, value: Any, depth: int) -> None:
        prefix = "  " * depth
        if isinstance(value, dict):
            lines.append(f"{prefix}{name}:")
            for field, item in value.items():
                append_value(str(field), item, depth + 1)
        elif isinstance(value, list):
            lines.append(f"{prefix}{name}: ({len(value)} entries)")
            for index, item in enumerate(value, start=1):
                append_value(f"[{index}]", item, depth + 1)
        else:
            lines.append(f"{prefix}{name}: {json.dumps(value, ensure_ascii=False)}")

    for key, value in summary.items():
        append_value(key, value, 0)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def image_from_path(path: Path) -> np.ndarray:
    """Read via bytes to support Unicode paths on Windows OpenCV builds."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise MediaProcessingError(f"Unable to decode image: {path.name}") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise MediaProcessingError(f"Unable to decode image as three-channel BGR: {path.name}")
    return image


def save_png(image: np.ndarray, path: Path) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise MediaProcessingError("OpenCV could not encode the annotated PNG image")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def output_relative_path(source: Path) -> Path:
    """Preserve input nesting, with a stable hashed basename for uniqueness."""
    relative = source.relative_to(INPUT_DIR)
    suffix_hash = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", relative.stem)[:90].strip(". ")
    if not safe_name:
        safe_name = "image"
    return Path("annotated_images") / relative.parent / f"{safe_name}_{suffix_hash}_annotated.png"


def make_config() -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=True,
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
    )
    config.validate()
    return config


def landmark_only_pipeline(model_path: Path) -> tuple[FaceAnalysis, LandmarkCapture]:
    detector = Face(
        device=DEVICE,
        conf=DETECTOR_CONFIDENCE,
        iou=DETECTOR_IOU,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )
    pipeline = FaceAnalysis(
        detector=detector,
        config=make_config(),
        landmark_model_path=model_path,
        device=DEVICE,
        verbose=False,
    )
    if pipeline.detector is not detector:
        raise RuntimeError("Unexpected detector binding in FaceAnalysis")
    if pipeline.landmarks is None:
        raise RuntimeError("Face landmarks component was not initialized")
    unexpected_enabled = [
        name for name in UNRELATED_PIPELINE_ATTRIBUTES if getattr(pipeline, name) is not None
    ]
    if unexpected_enabled:
        raise RuntimeError(
            "Unrelated FaceAnalysis components are enabled: " + ", ".join(unexpected_enabled)
        )
    capture = LandmarkCapture(pipeline.landmarks)
    pipeline.landmarks = capture
    return pipeline, capture


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


def annotate_image(frame: np.ndarray, prediction: Any, capture: LandmarkCapture,
                   source_image: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise MediaProcessingError("The decoded frame is not a three-channel image")

    faces = list(prediction)
    raw_landmark_calls = list(capture.calls)
    output = frame.copy()
    height, width = frame.shape[:2]
    font_scale = max(0.36, min(0.82, min(height, width) / 900.0))
    line_width = 1 if min(height, width) < 450 else 2
    rows: list[dict[str, Any]] = []

    drawn_faces = 0
    faces_with_landmarks = 0
    faces_without_landmarks = 0
    invalid_faces = 0
    landmark_observations = 0
    out_of_frame_landmarks = 0

    for face_index, face in enumerate(faces, start=1):
        row_prefix: dict[str, Any] = {
            "source_image": source_image,
            "face_index": face_index,
            "faces_in_image": len(faces),
            "image_width": width,
            "image_height": height,
            "detector_confidence": "",
            "face_box_x1": "",
            "face_box_y1": "",
            "face_box_x2": "",
            "face_box_y2": "",
            "face_box_width": "",
            "face_box_height": "",
            "pipeline_landmarks_available": False,
            "pipeline_landmark_count": 0,
            "landmark_index": "",
            "landmarks_in_face": "",
            "x_normalized": "",
            "y_normalized": "",
            "z_normalized": "",
            "x_pixel": "",
            "y_pixel": "",
            "within_image_bounds": "",
            "status": "",
            "failure_reason": "",
        }
        try:
            coords = np.asarray(face.box, dtype=np.float64).reshape(-1)
            if coords.size != 4 or not np.all(np.isfinite(coords)):
                raise ValueError("Invalid face box coordinates")
            x1, y1, x2, y2 = (float(v) for v in coords)
            if not (x2 > x1 and y2 > y1):
                raise ValueError("Non-positive face box area")

            confidence = getattr(face, "confidence", None)
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                    raise ValueError("Invalid detection confidence")
                row_prefix["detector_confidence"] = confidence

            features = getattr(face, "face_features", None) or {}
            landmark_features = features.get("landmarks", {}) if isinstance(features, dict) else {}
            available = bool(landmark_features.get("available", False))
            count = int(landmark_features.get("count", 0) or 0)
            row_prefix["pipeline_landmarks_available"] = available
            row_prefix["pipeline_landmark_count"] = count
            row_prefix["face_box_x1"] = x1
            row_prefix["face_box_y1"] = y1
            row_prefix["face_box_x2"] = x2
            row_prefix["face_box_y2"] = y2
            row_prefix["face_box_width"] = x2 - x1
            row_prefix["face_box_height"] = y2 - y1

            left = max(0, min(width - 1, int(round(x1))))
            top = max(0, min(height - 1, int(round(y1))))
            right = max(0, min(width - 1, int(round(x2))))
            bottom = max(0, min(height - 1, int(round(y2))))
            if right <= left or bottom <= top:
                raise ValueError("Face box cannot be drawn within image bounds")

            color_box = (255, 210, 0)
            color_points = (0, 220, 0)
            cv2.rectangle(output, (left, top), (right, bottom), color_box, line_width)
            drawn_faces += 1

            landmarks = raw_landmark_calls[face_index - 1] if face_index - 1 < len(raw_landmark_calls) else None

            if landmarks is None:
                faces_without_landmarks += 1
                placeholder = dict(row_prefix)
                placeholder["status"] = "NO_LANDMARKS"
                placeholder["failure_reason"] = "Landmarker returned None"
                rows.append(placeholder)
                label = f"face {face_index}: no landmarks"
                draw_text(output, label, left, top - 5 if top > 22 else bottom + 18, font_scale, color_box)
                continue

            try:
                landmark_count = len(landmarks)
            except TypeError as exc:
                raise ValueError("Landmarker output does not define a length") from exc

            if landmark_count == 0:
                faces_without_landmarks += 1
                placeholder = dict(row_prefix)
                placeholder["status"] = "NO_LANDMARKS"
                placeholder["failure_reason"] = "Landmarker returned an empty sequence"
                rows.append(placeholder)
                label = f"face {face_index}: 0 landmarks"
                draw_text(output, label, left, top - 5 if top > 22 else bottom + 18, font_scale, color_box)
                continue

            if not available:
                raise ValueError("Native landmarks were returned, but FaceAnalysis reports them unavailable")
            if count != landmark_count:
                raise ValueError(
                    f"FaceAnalysis reports {count} landmarks, but native output has {landmark_count}"
                )

            label = f"face {face_index}: {landmark_count} landmarks"
            if confidence is not None:
                label += f"  det {confidence:.2f}"
            draw_text(output, label, left, top - 5 if top > 22 else bottom + 18, font_scale, color_box)

            for landmark_index, landmark in enumerate(landmarks):
                try:
                    x_norm = float(landmark.x)
                    y_norm = float(landmark.y)
                    z_norm = float(landmark.z)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid landmark structure at index {landmark_index}") from exc
                if not all(math.isfinite(v) for v in (x_norm, y_norm, z_norm)):
                    raise ValueError(f"Non-finite landmark coordinate at index {landmark_index}")

                x_px = x_norm * width
                y_px = y_norm * height
                within_bounds = 0.0 <= x_px < width and 0.0 <= y_px < height
                if within_bounds:
                    cv2.circle(
                        output,
                        (int(round(x_px)), int(round(y_px))),
                        POINT_RADIUS,
                        color_points,
                        POINT_THICKNESS,
                        cv2.LINE_AA,
                    )
                else:
                    out_of_frame_landmarks += 1

                row = dict(row_prefix)
                row["landmark_index"] = landmark_index
                row["landmarks_in_face"] = landmark_count
                row["x_normalized"] = x_norm
                row["y_normalized"] = y_norm
                row["z_normalized"] = z_norm
                row["x_pixel"] = x_px
                row["y_pixel"] = y_px
                row["within_image_bounds"] = bool(within_bounds)
                row["status"] = "OK"
                rows.append(row)
                landmark_observations += 1

            faces_with_landmarks += 1
        except Exception as exc:
            invalid_faces += 1
            failure = dict(row_prefix)
            failure["status"] = "INVALID_FACE_RESULT"
            failure["failure_reason"] = str(exc)
            rows.append(failure)

    if not faces:
        rows.append({
            "source_image": source_image,
            "face_index": "",
            "faces_in_image": 0,
            "image_width": width,
            "image_height": height,
            "detector_confidence": "",
            "face_box_x1": "",
            "face_box_y1": "",
            "face_box_x2": "",
            "face_box_y2": "",
            "face_box_width": "",
            "face_box_height": "",
            "pipeline_landmarks_available": False,
            "pipeline_landmark_count": 0,
            "landmark_index": "",
            "landmarks_in_face": "",
            "x_normalized": "",
            "y_normalized": "",
            "z_normalized": "",
            "x_pixel": "",
            "y_pixel": "",
            "within_image_bounds": "",
            "status": "NO_FACE_DETECTIONS",
            "failure_reason": "Detector returned no faces",
        })

    summary = {
        "faces_detected": len(faces),
        "faces_drawn": drawn_faces,
        "faces_with_landmarks": faces_with_landmarks,
        "faces_without_landmarks": faces_without_landmarks,
        "invalid_faces": invalid_faces,
        "landmark_observations": landmark_observations,
        "out_of_frame_landmarks": out_of_frame_landmarks,
    }
    return output, rows, summary


def process_image(source: Path, target: Path, stage: Path, pipeline: FaceAnalysis,
                  capture: LandmarkCapture, csv_writer: csv.DictWriter) -> dict[str, Any]:
    relative = source.relative_to(INPUT_DIR).as_posix()
    frame = image_from_path(source)
    capture.reset_capture()
    prediction = pipeline.predict(frame)
    annotated, rows, summary = annotate_image(frame, prediction, capture, relative)
    save_png(annotated, target)
    decoded = image_from_path(target)
    if decoded.shape[:2] != frame.shape[:2]:
        raise MediaProcessingError("Annotated image dimensions differ from source")
    for row in rows:
        csv_writer.writerow(row)
    return {
        "input": relative,
        "source_sha256": sha256_file(source),
        "output_image": target.relative_to(stage).as_posix(),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        **summary,
        "status": "SUCCESS",
    }


def build_contact_sheet(stage: Path, image_records: list[dict[str, Any]]) -> str | None:
    successful = [r for r in image_records if r.get("status") == "SUCCESS"]
    if not successful:
        return None

    thumbs: list[np.ndarray] = []
    thumb_w = 480
    thumb_h = 320
    panel_h = 38

    for rec in successful:
        rel = rec["output_image"]
        img = image_from_path(stage / rel)
        h, w = img.shape[:2]
        scale = min(thumb_w / w, thumb_h / h)
        resized = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
        canvas = np.full((thumb_h + panel_h, thumb_w, 3), 245, dtype=np.uint8)
        y0 = (thumb_h - resized.shape[0]) // 2
        x0 = (thumb_w - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        cv2.rectangle(canvas, (0, thumb_h), (thumb_w - 1, thumb_h + panel_h - 1), (235, 235, 235), -1)
        cv2.putText(canvas, Path(rec["input"]).name, (10, thumb_h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"faces={rec['faces_detected']} landmarks={rec['landmark_observations']}",
                    (10, thumb_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
        thumbs.append(canvas)

    cols = 2 if len(thumbs) > 1 else 1
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows * (thumb_h + panel_h) + (rows - 1) * 16,
                     cols * thumb_w + (cols - 1) * 16, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        y = r * (thumb.shape[0] + 16)
        x = c * (thumb.shape[1] + 16)
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    rel_path = Path("synthetic_demo_examples.png")
    save_png(sheet, stage / rel_path)
    return rel_path.as_posix()


def verify_staged_outputs(stage: Path, summary: dict[str, Any]) -> None:
    """Verify all new artifacts and the full image/CSV/summary accounting."""
    if not stage.is_dir() or stage.is_symlink():
        raise MediaProcessingError("Published output directory is missing or unsafe")
    if summary.get("status") != "COMPLETE" or summary.get("images_failed") != 0:
        raise MediaProcessingError("A failed or incomplete run cannot be published")

    records = summary.get("images", [])
    if (not records or len(records) != summary.get("images_discovered")
            or len(records) != summary.get("images_succeeded")
            or any(rec.get("status") != "SUCCESS" for rec in records)):
        raise MediaProcessingError("Not every discovered image succeeded")

    output_dir = stage / "annotated_images"
    csv_path = stage / "landmark_observations.csv"
    json_path = stage / "synthetic_demo_summary.json"
    text_path = stage / "synthetic_demo_summary.txt"
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise MediaProcessingError("Annotated image directory is missing or unsafe")
    for artifact in (csv_path, json_path, text_path):
        if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size == 0:
            raise MediaProcessingError(f"Required output is missing or empty: {artifact.name}")

    output_names = set()
    source_names = set()
    for rec in records:
        rel_name = rec["output_image"]
        source_name = rec["input"]
        if rel_name in output_names or source_name in source_names:
            raise MediaProcessingError("Duplicate input or output listed in summary")
        output_names.add(rel_name)
        source_names.add(source_name)
        target = stage / rel_name
        if (not target.resolve().is_relative_to(stage.resolve())
                or not target.is_file() or target.is_symlink() or target.stat().st_size == 0):
            raise MediaProcessingError("Required staged image is absent or unsafe")
        decoded = image_from_path(target)
        if decoded.shape[:2] != (rec["height"], rec["width"]):
            raise MediaProcessingError("Annotated image dimensions differ from source")
        original = INPUT_DIR / source_name
        if (not original.is_file() or original.is_symlink()
                or sha256_file(original) != rec["source_sha256"]):
            raise MediaProcessingError("Input image changed during this demonstration")

    actual_names = {
        p.relative_to(stage).as_posix() for p in output_dir.rglob("*") if p.is_file()
    }
    if actual_names != output_names:
        raise MediaProcessingError("Annotated outputs do not exactly match successful sources")

    row_count = 0
    native_point_rows = 0
    no_face_rows = 0
    csv_sources = set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise MediaProcessingError("CSV header does not match the expected schema")
        for row in reader:
            row_count += 1
            source_name = row["source_image"]
            if source_name not in source_names:
                raise MediaProcessingError("CSV contains an unknown image identifier")
            csv_sources.add(source_name)
            status = row["status"]
            if status == "OK":
                native_point_rows += 1
            elif status == "NO_FACE_DETECTIONS":
                no_face_rows += 1
            elif status not in ("NO_LANDMARKS", "INVALID_FACE_RESULT"):
                raise MediaProcessingError("CSV contains an unknown observation status")
    if csv_sources != source_names:
        raise MediaProcessingError("One or more input images have no CSV record")
    if row_count != summary["result_rows"]:
        raise MediaProcessingError("CSV row count does not match the summary")
    if native_point_rows != summary["landmark_observations"]:
        raise MediaProcessingError("Landmark observation count does not match the summary")
    if no_face_rows != summary["images_without_faces"]:
        raise MediaProcessingError("Images-without-faces count does not match the summary")

    metrics = ("faces_detected", "faces_with_landmarks", "faces_without_landmarks",
               "invalid_faces", "landmark_observations", "out_of_frame_landmarks")
    for metric in metrics:
        if sum(int(rec.get(metric, 0)) for rec in records) != summary[metric]:
            raise MediaProcessingError(f"Summary total for {metric} is inconsistent")

    contact_rel = summary.get("summary_contact_sheet")
    if not contact_rel or contact_rel != "synthetic_demo_examples.png":
        raise MediaProcessingError("A complete contact sheet was not created")
    contact_path = stage / contact_rel
    if not contact_path.is_file() or contact_path.stat().st_size == 0:
        raise MediaProcessingError("Missing or empty contact sheet")
    contact = image_from_path(contact_path)
    if contact.shape[0] <= 0 or contact.shape[1] <= 0:
        raise MediaProcessingError("Invalid contact sheet")

    with json_path.open("r", encoding="utf-8") as handle:
        if json.load(handle) != summary:
            raise MediaProcessingError("Staged JSON summary differs from recorded results")
    if text_path.read_text(encoding="utf-8") != summary_as_text(summary):
        raise MediaProcessingError("Staged TXT summary differs from recorded results")

    expected_files = output_names | {
        "landmark_observations.csv", "synthetic_demo_summary.json",
        "synthetic_demo_summary.txt", "synthetic_demo_examples.png",
    }
    existing_files = {
        p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()
    }
    if existing_files != expected_files:
        raise MediaProcessingError("Staged directory contains unexpected or missing files")


def commit_directory(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    """Replace the complete owned directory only after verification, with rollback."""
    backup = staging_root / "previous_synthetic_demo"
    if backup.exists():
        raise MediaProcessingError("Unexpected backup already exists in new staging directory")
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise MediaProcessingError("Final synthetic_demo path is not a normal directory")
    verify_staged_outputs(stage, summary)
    had_old = FINAL_DIR.exists()
    published = False
    try:
        if had_old:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        published = True
        # Check the ACTUAL published files before removing the previous result.
        verify_staged_outputs(FINAL_DIR, summary)
    except BaseException:
        try:
            if published and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)  # Move rejected new outputs out of the final path.
            if had_old and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as restore_error:
            # Keep the backup; never silently erase the only surviving old result.
            raise MediaProcessingError(
                f"Publication failed and old-output restoration failed; "
                f"recover the backup from {backup}"
            ) from restore_error
        raise
    # Only AFTER the published data pass verification is the old version deleted.
    if had_old:
        shutil.rmtree(backup)



def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    component_paths = check_project_imports()
    model_path = resolve_model_path()
    model_details = model_provenance(model_path)
    images = discover_images()
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise MediaProcessingError("Final synthetic_demo path is not a normal directory")
    staging_root = Path(tempfile.mkdtemp(prefix=".face_landmarks_synth_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    (stage / "annotated_images").mkdir()

    try:
        pipeline, capture = landmark_only_pipeline(model_path)
        outputs_seen: set[str] = set()

        print("PhysioTrack face landmarks synthetic qualitative demonstration")
        print("Images discovered:", len(images))
        print("Inputs: validation/synthetic_media/face_landmarks_media/")
        print("Outputs: validation/face_landmarks/results/synthetic_demo/")

        records: list[dict[str, Any]] = []
        csv_path = stage / "landmark_observations.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for ordinal, source in enumerate(images, 1):
                relative = source.relative_to(INPUT_DIR).as_posix()
                target_rel = output_relative_path(source)

                if target_rel.as_posix().casefold() in outputs_seen:
                    raise MediaProcessingError("Output-name collision detected")
                outputs_seen.add(target_rel.as_posix().casefold())

                print(f"[{ordinal}/{len(images)}] image: {relative}", flush=True)

                try:
                    record = process_image(source, stage / target_rel, stage, pipeline, capture, writer)
                    records.append(record)
                    print(
                        f"    OK: faces={record['faces_detected']}; "
                        f"faces_with_landmarks={record['faces_with_landmarks']}; "
                        f"landmarks={record['landmark_observations']}"
                    )
                except Exception as exc:
                    print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                    traceback.print_exc()
                    (stage / target_rel).unlink(missing_ok=True)
                    records.append({
                        "input": relative,
                        "source_sha256": sha256_file(source),
                        "output_image": target_rel.as_posix(),
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc) if isinstance(exc, MediaProcessingError) else "Processing failure; see terminal traceback",
                    })

        successful = sum(r.get("status") == "SUCCESS" for r in records)
        failed = len(records) - successful
        images_without_faces = sum(r.get("status") == "SUCCESS" and r.get("faces_detected", 0) == 0 for r in records)
        faces_detected = sum(int(r.get("faces_detected", 0)) for r in records if r.get("status") == "SUCCESS")
        faces_with_landmarks = sum(int(r.get("faces_with_landmarks", 0)) for r in records if r.get("status") == "SUCCESS")
        faces_without_landmarks = sum(int(r.get("faces_without_landmarks", 0)) for r in records if r.get("status") == "SUCCESS")
        invalid_faces = sum(int(r.get("invalid_faces", 0)) for r in records if r.get("status") == "SUCCESS")
        landmark_observations = sum(int(r.get("landmark_observations", 0)) for r in records if r.get("status") == "SUCCESS")
        out_of_frame_landmarks = sum(int(r.get("out_of_frame_landmarks", 0)) for r in records if r.get("status") == "SUCCESS")

        result_rows = 0
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for _ in reader:
                result_rows += 1

        contact_sheet_rel = build_contact_sheet(stage, records)

        summary = {
            "type": "synthetic_qualitative_face_landmarks_demonstration",
            "benchmark_evaluation": False,
            "project": "physiotrack-v2",
            "input_directory": "validation/synthetic_media/face_landmarks_media",
            "output_directory": "validation/face_landmarks/results/synthetic_demo",
            "component_imports": component_paths,
            "configuration": {
                "detector": "physiotrack.face.Face",
                "pipeline": "physiotrack.face.FaceAnalysis",
                "detector_confidence": DETECTOR_CONFIDENCE,
                "detector_iou": DETECTOR_IOU,
                "max_detections": MAX_DETECTIONS,
                "landmarker_model_path": model_details["path"],
                "landmarker_model_sha256": model_details["sha256"],
                "device": DEVICE,
                "tracking_enabled": False,
                "ground_truth_used": False,
                "benchmark_metrics_calculated": False,
                "expected_native_landmarks_per_face": EXPECTED_LANDMARKS_PER_FACE,
            },
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "platform": platform.system(),
                "mediapipe": dependency_version("mediapipe"),
                "torch": dependency_version("torch"),
                "ultralytics": dependency_version("ultralytics"),
            },
            "note": "Qualitative demonstration only. Uses the real face detector and the real face-landmarks component. No 300-W annotations or NME calculations are used here.",
            "status": "COMPLETE" if not failed else "REJECTED_WITH_FILE_FAILURES",
            "images_discovered": len(images),
            "images_succeeded": successful,
            "images_failed": failed,
            "images_without_faces": images_without_faces,
            "faces_detected": faces_detected,
            "faces_with_landmarks": faces_with_landmarks,
            "faces_without_landmarks": faces_without_landmarks,
            "invalid_faces": invalid_faces,
            "landmark_observations": landmark_observations,
            "out_of_frame_landmarks": out_of_frame_landmarks,
            "result_rows": result_rows,
            "landmark_observations_csv": "landmark_observations.csv",
            "summary_contact_sheet": contact_sheet_rel,
            "images": records,
            "reproducibility": {
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "source_sha256": {name: sha256_file(REPOSITORY_ROOT / rel_path)
                                  for name, rel_path in component_paths.items()},
                "input_sha256_recorded_per_image": True,
                "model_sha256_recorded": True,
                "exact_numerical_reproduction_requires_matching_model_and_environment": True,
            },
        }

        with (stage / "synthetic_demo_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        with (stage / "synthetic_demo_summary.txt").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(summary_as_text(summary))

        if failed:
            failure_report = RESULTS_DIR / "synthetic_demo_last_attempt.json"
            shutil.copyfile(stage / "synthetic_demo_summary.json", failure_report)
            raise MediaProcessingError(
                f"{failed} image(s) failed during processing. Existing published synthetic_demo was not replaced. "
                "See results/synthetic_demo_last_attempt.json and terminal diagnostics."
            )

        verify_staged_outputs(stage, summary)
        if sha256_file(model_path) != model_details["sha256"]:
            raise MediaProcessingError("Landmarker model changed during inference; old results preserved")
        commit_directory(stage, staging_root, summary)

        print("\nFinished.")
        print("Images processed:", successful)
        print("Faces detected:", faces_detected)
        print("Faces with landmarks:", faces_with_landmarks)
        print("Faces without landmarks:", faces_without_landmarks)
        print("Landmark observations:", landmark_observations)
        print("Out-of-frame landmarks:", out_of_frame_landmarks)
        print("Saved summaries:")
        print("  validation/face_landmarks/results/synthetic_demo/synthetic_demo_summary.json")
        print("  validation/face_landmarks/results/synthetic_demo/synthetic_demo_summary.txt")

    finally:
        try:
            if "capture" in locals():
                capture.close()
        finally:
            backup = staging_root / "previous_synthetic_demo"
            if backup.exists():
                print(f"Recovery backup retained at: {backup}", file=sys.stderr)
            else:
                shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()