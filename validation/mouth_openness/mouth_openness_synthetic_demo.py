"""Independent synthetic qualitative demonstration of PhysioTrack MouthOpenness.

Place this script at validation/mouth_openness/mouth_openness_synthetic_demo.py.
Images: validation/synthetic_media/mouth_openness_media/ (recursive discovery).
Outputs: validation/mouth_openness/results/synthetic_demo/ (script-owned only).

Runs the real PhysioTrack Face detector -> FaceAnalysis -> FaceLandmarks ->
MouthOpenness on non-benchmark images. Captures the native per-face landmarks
without replacing the computation and independently checks the three mouth
measurements from the captured MediaPipe points (13, 14, 61, 291).

This is a qualitative illustration, NOT a FELT/RAVDESS accuracy benchmark.
No ground-truth mouth-opening labels or detection-recall metrics are inferred.
Every detection and status is kept in the CSV even if not drawn on the panels.

Staging, full output verification and rollback protect any accepted previous
synthetic_demo results. All benchmark/component-test outputs are untouched.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
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

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPO_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "mouth_openness_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"
SOURCE_ROOT = REPO_ROOT / "src" / "physiotrack"

EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.001  # Capture weak predictions, not a quality threshold.
DETECTOR_IOU = 0.45
MAX_DETECTIONS = 10000
DISPLAY_CONFIDENCE = 0.25  # Annotation ONLY; never discards a CSV prediction.
EXPECTED_LANDMARKS = 478
TOLERANCE = 1e-12
SHEET_COLUMNS = 3
TILE_WIDTH = 470
TILE_HEIGHT = 340

# Presentation only: the original and annotated source are kept at native size;
# text is drawn exclusively in a separate dashboard, never over face pixels.
PANEL_MARGIN = 12
PANEL_HEADER = 65
PANEL_FOOTER = 55
DASHBOARD_WIDTH = 460
DASHBOARD_CARD_HEIGHT = 158
DASHBOARD_CARD_GAP = 14
DASHBOARD_CARDS_TOP = 335

POINTS = {"left_corner": 61, "right_corner": 291, "upper_lip": 13, "lower_lip": 14}
METRICS = ("mouth_openness", "mouth_width", "mouth_height")
CSV_FIELDS = (
    "input", "source_sha256", "output_image", "image_width", "image_height",
    "face_index", "status", "displayed", "detector_confidence",
    "box_x1", "box_y1", "box_x2", "box_y2",
    "landmarks_available", "landmark_count", "mouth_available",
    "mouth_openness", "mouth_width", "mouth_height",
    "independent_mouth_openness", "independent_mouth_width", "independent_mouth_height",
    "openness_absolute_error", "width_absolute_error", "height_absolute_error",
    "left_corner_x", "left_corner_y", "right_corner_x", "right_corner_y",
    "upper_lip_x", "upper_lip_y", "lower_lip_x", "lower_lip_y",
)
ARTIFACT_CSV = "mouth_openness_observations.csv"
ARTIFACT_SHEET = "synthetic_demo_examples.png"
ARTIFACT_JSON = "synthetic_demo_summary.json"
ARTIFACT_TXT = "synthetic_demo_summary.txt"


class DemoError(RuntimeError):
    """Preflight, execution, verification or publication failed."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def components() -> dict[str, Any]:
    """Require the actual current local checkout, not an unrelated installed wheel."""
    if not SOURCE_ROOT.is_dir():
        raise DemoError("Repository src/physiotrack not found relative to this script")
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig
    from physiotrack.face.landmarks import FaceLandmarks
    from physiotrack.face.mouth import MouthOpenness
    from physiotrack.models import Models

    selected = {"Face": Face, "FaceAnalysis": FaceAnalysis,
                "FaceAnalysisConfig": FaceAnalysisConfig,
                "FaceLandmarks": FaceLandmarks, "MouthOpenness": MouthOpenness}
    expected_root = SOURCE_ROOT.resolve()
    imports = {}
    for name, cls in selected.items():
        path = Path(inspect.getfile(cls)).resolve()
        if not path.is_relative_to(expected_root):
            raise DemoError(f"{name} is not loaded from this repository's src/physiotrack")
        imports[name] = path.relative_to(REPO_ROOT).as_posix()

    model_path = Path(Models.resolve(Models.Face.MediaPipe.Landmarks.face_landmarker)).resolve()
    if not model_path.is_file():
        raise DemoError("The MediaPipe FaceLandmarks model could not be resolved")
    selected.update({"model_path": model_path, "imports": imports})
    return selected


def get_sources() -> list[Path]:
    if not INPUT_DIR.is_dir() or INPUT_DIR.is_symlink():
        raise DemoError("Missing/unsafe validation/synthetic_media/mouth_openness_media folder")
    sources = sorted(
        (p for p in INPUT_DIR.rglob("*") if p.is_file() and not p.is_symlink()
         and p.suffix.lower() in EXTENSIONS),
        key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(),
                       p.relative_to(INPUT_DIR).as_posix()),
    )
    if not sources:
        raise DemoError("No synthetic images found; previously accepted results unchanged")
    names = [p.relative_to(INPUT_DIR).as_posix().casefold() for p in sources]
    if len(names) != len(set(names)):
        raise DemoError("Ambiguous input names differing only by case")
    return sources


def output_name(path: Path) -> str:
    relative = path.relative_to(INPUT_DIR).as_posix()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip(" .")[:70] or "image"
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"annotated_images/{name}_{suffix}.png"


def read_image(path: Path) -> np.ndarray:
    try:
        frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise DemoError(f"Cannot read image: {path.name}") from exc
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise DemoError(f"Invalid BGR image: {path.name}")
    return frame


def save_image(path: Path, frame: np.ndarray) -> None:
    ok, blob = cv2.imencode(".png", frame)
    if not ok:
        raise DemoError(f"PNG encoding failed for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    blob.tofile(str(path))


class LandmarkCapture:
    """Transparent proxy of the actual FaceLandmarks.predict_face method."""

    def __init__(self, actual: Any) -> None:
        self.actual = actual
        self.calls: list[tuple[np.ndarray, Any]] = []

    def predict_face(self, frame: np.ndarray, box: Any) -> Any:
        landmarks = self.actual.predict_face(frame, box)
        self.calls.append((np.asarray(box, dtype=float).reshape(-1).copy(), landmarks))
        return landmarks

    def reset(self) -> None:
        self.calls.clear()

    def close(self) -> None:
        if hasattr(self.actual, "close"):
            self.actual.close()


def create_pipeline(classes: dict[str, Any]) -> tuple[Any, LandmarkCapture]:
    config = classes["FaceAnalysisConfig"](
        tracking=False, head_pose=False, landmarks=True, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=True, mouth_motion=False, emotion=False, regions=False, temporal=False,
    )
    config.validate()
    detector = classes["Face"](device=DEVICE, conf=DETECTOR_CONFIDENCE,
                               iou=DETECTOR_IOU, max_det=MAX_DETECTIONS, verbose=False)
    pipeline = classes["FaceAnalysis"](
        detector=detector, config=config,
        landmark_model_path=classes["model_path"], device=DEVICE, verbose=False,
    )
    try:
        if pipeline.detector is not detector:
            raise DemoError("FaceAnalysis is not using the configured real detector")
        if not isinstance(pipeline.landmarks, classes["FaceLandmarks"]):
            raise DemoError("FaceAnalysis did not initialize the real FaceLandmarks")
        if not isinstance(pipeline.mouth, classes["MouthOpenness"]):
            raise DemoError("FaceAnalysis did not initialize the real MouthOpenness")
        for field in ("tracker", "orientation", "quality", "eyes", "blink", "gaze",
                      "gaze_estimation", "mouth_motion", "emotion", "regions", "temporal"):
            if getattr(pipeline, field) is not None:
                raise DemoError(f"Unrelated component unexpectedly active: {field}")
        capture = LandmarkCapture(pipeline.landmarks)
        pipeline.landmarks = capture
        return pipeline, capture
    except BaseException:
        pipeline.close()
        raise


def numeric(value: Any, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise DemoError(f"Missing or invalid numerical value: {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DemoError(f"Invalid numerical value: {name}") from exc
    if not math.isfinite(result):
        raise DemoError(f"Non-finite numerical value: {name}")
    if minimum is not None and result < minimum:
        raise DemoError(f"Out-of-range numerical value: {name}")
    if maximum is not None and result > maximum:
        raise DemoError(f"Out-of-range numerical value: {name}")
    return result


def mouth_from_points(point_xy: dict[str, tuple[float, float]],
                      image_width: int, image_height: int) -> dict[str, float]:
    """Independent image-dimension-aware Euclidean check of the native ratios."""
    def distance(a: str, b: str) -> float:
        ax, ay = point_xy[a]
        bx, by = point_xy[b]
        return math.hypot((ax - bx) * image_width, (ay - by) * image_height) / image_width

    width = distance("left_corner", "right_corner")
    height = distance("upper_lip", "lower_lip")
    if not (math.isfinite(width) and width > 0 and math.isfinite(height) and height >= 0):
        raise DemoError("Captured mouth landmarks imply invalid geometry")
    return {"mouth_width": width, "mouth_height": height,
            "mouth_openness": height / width}


def matching_points(box: list[float] | None, capture: LandmarkCapture,
                    matched: set[int]) -> Any:
    if box is None:
        return None
    for i, (call_box, landmarks) in enumerate(capture.calls):
        if i not in matched and call_box.size == 4 and np.isfinite(call_box).all() and \
                np.allclose(call_box, np.asarray(box, dtype=float), atol=1e-5, rtol=0):
            matched.add(i)
            return landmarks
    return None


def bounded_box(box: Any) -> list[float] | None:
    if box is None:
        return None
    try:
        coords = np.asarray(box, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if coords.size != 4 or not np.isfinite(coords).all():
        return None
    x1, y1, x2, y2 = map(float, coords)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def put_label(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.45,
              fg: tuple[int, int, int] = (250, 250, 250)) -> None:
    """Small captions in the dedicated header/footer; never over the face."""
    height, width = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(x, max(0, width - tw - 9)))
    y = max(th + 8, min(y, height - baseline - 5))
    cv2.rectangle(image, (x, y - th - 7), (min(width - 1, x + tw + 9),
                  min(height - 1, y + baseline + 4)), (20, 20, 20), -1)
    cv2.putText(image, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, 1, cv2.LINE_AA)


def dashboard_text(canvas: np.ndarray, text: str, x: int, baseline: int,
                   scale: float = 0.58, color: tuple[int, int, int] = (235, 235, 235),
                   thickness: int = 1) -> None:
    """Readably print on the dark dashboard, with no label rectangles."""
    cv2.putText(canvas, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wrap_dashboard_text(value: str, max_px: int, scale: float = 0.52) -> list[str]:
    """Wrap source names by words when possible, without losing any characters."""
    def fits(candidate: str) -> bool:
        return cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX,
                               scale, 1)[0][0] <= max_px

    result: list[str] = []
    current = ""
    for token in re.findall(r"\S+\s*", value):
        trial = current + token
        if fits(trial):
            current = trial
            continue
        if current.strip():
            result.append(current.rstrip())
            current = ""
        # A single exceptionally long path fragment still needs safe wrapping.
        for char in token:
            if current and not fits(current + char):
                result.append(current.rstrip())
                current = ""
            current += char
    if current.strip():
        result.append(current.rstrip())
    return result or [""]


def annotate_face(frame: np.ndarray, box: list[float], face_index: int,
                  confidence: float | None, status: str,
                  row: dict[str, Any]) -> None:
    """Show only bounding boxes and measured mouth geometry on source pixels."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(0, min(width - 1, int(round(x2))))
    bottom = max(0, min(height - 1, int(round(y2))))
    if right <= left or bottom <= top:
        return
    color = (60, 220, 60) if status == "MOUTH_AVAILABLE" else (20, 160, 255)
    line_width = max(1, round(min(width, height) / 550))
    cv2.rectangle(frame, (left, top), (right, bottom), color, line_width)
    if status == "MOUTH_AVAILABLE":
        for a, b, linecolor in (("left_corner", "right_corner", (255, 215, 0)),
                                ("upper_lip", "lower_lip", (0, 190, 255))):
            ap = (int(round(float(row[a + "_x"]) * width)),
                  int(round(float(row[a + "_y"]) * height)))
            bp = (int(round(float(row[b + "_x"]) * width)),
                  int(round(float(row[b + "_y"]) * height)))
            ap = (max(0, min(width - 1, ap[0])), max(0, min(height - 1, ap[1])))
            bp = (max(0, min(width - 1, bp[0])), max(0, min(height - 1, bp[1])))
            cv2.line(frame, ap, bp, linecolor, 2, cv2.LINE_AA)
            for p in (ap, bp):
                cv2.circle(frame, p, max(2, int(min(width, height) / 480)),
                           linecolor, -1, cv2.LINE_AA)


def dashboard_body_height(source_height: int, number_displayed: int,
                          input_name: str = "") -> int:
    """Leave space for wrapped filenames and every annotated face."""
    filename_lines = len(wrap_dashboard_text(input_name, DASHBOARD_WIDTH - 46, .53))
    card_top = max(DASHBOARD_CARDS_TOP, 148 + 23 * filename_lines + 100)
    needed = card_top + number_displayed * (DASHBOARD_CARD_HEIGHT + DASHBOARD_CARD_GAP) + 36
    return max(source_height, needed)


def panel_image(original: np.ndarray, annotated: np.ndarray,
                input_name: str, raw: int, drawn: int,
                observations: list[dict[str, Any]]) -> np.ndarray:
    """Original | annotated geometry | separate readable per-face dashboard."""
    h, w = original.shape[:2]
    margin, header, footer = PANEL_MARGIN, PANEL_HEADER, PANEL_FOOTER
    shown = [row for row in observations if row["displayed"]]
    if len(shown) != drawn:
        raise DemoError("Dashboard displayed-face count does not match observations")
    body_height = dashboard_body_height(h, len(shown), input_name)
    total_width = 2 * w + 4 * margin + DASHBOARD_WIDTH
    output = np.full((body_height + header + footer, total_width, 3),
                     (27, 29, 30), dtype=np.uint8)
    left_x, annotated_x, sidebar_x = margin, 2 * margin + w, 3 * margin + 2 * w
    output[header:header + h, left_x:left_x + w] = original
    output[header:header + h, annotated_x:annotated_x + w] = annotated
    scale = max(0.4, min(0.8, w / 1250))
    put_label(output, "Synthetic source", left_x, 36, scale)
    put_label(output, "PhysioTrack MouthOpenness", annotated_x, 36, scale)
    put_label(output, f"{input_name} | raw faces: {raw} | displayed: {drawn}",
              margin, body_height + header + 30,
              max(0.35, min(0.62, w / 1700)))

    # A single monochrome text panel holds every measurement; no dark text
    # rectangles are positioned on forehead or face, including multi-face cases.
    x = sidebar_x + 22
    y0 = header
    dashboard_text(output, "PhysioTrack", x, y0 + 42, .87, (255, 255, 255), 2)
    dashboard_text(output, "MOUTH OPENNESS / SYNTHETIC", x, y0 + 70,
                   .52, (170, 228, 235))
    cv2.line(output, (x, y0 + 88), (sidebar_x + DASHBOARD_WIDTH - 22, y0 + 88),
             (74, 79, 79), 1, cv2.LINE_AA)
    dashboard_text(output, "IMAGE", x, y0 + 121, .52, (174, 181, 184))
    lines = wrap_dashboard_text(input_name, DASHBOARD_WIDTH - 46, .53)
    for i, line in enumerate(lines):
        dashboard_text(output, line, x, y0 + 148 + 23 * i, .53)
    info_y = y0 + 148 + 23 * len(lines)
    dashboard_text(output, f"Detected faces: {raw}", x, info_y + 23, .57)
    dashboard_text(output, f"Displayed faces: {drawn}", x, info_y + 51, .57)
    dashboard_text(output, "Openness = height / width", x, info_y + 82,
                   .49, (170, 228, 235))

    start = y0 + max(DASHBOARD_CARDS_TOP, info_y + 100 - y0)
    for i, row in enumerate(shown):
        top = start + i * (DASHBOARD_CARD_HEIGHT + DASHBOARD_CARD_GAP)
        # Long file names can add header lines; account for them safely.
        if top + DASHBOARD_CARD_HEIGHT + 12 > output.shape[0] - footer:
            raise DemoError("Dashboard insufficient for all displayed faces")
        x1, x2 = sidebar_x + 14, sidebar_x + DASHBOARD_WIDTH - 14
        cv2.rectangle(output, (x1, top), (x2, top + DASHBOARD_CARD_HEIGHT),
                      (41, 46, 47), -1)
        cv2.rectangle(output, (x1, top), (x1 + 5, top + DASHBOARD_CARD_HEIGHT),
                      (60, 220, 60) if row["mouth_available"] else (20, 160, 255), -1)
        confidence = row["detector_confidence"]
        heading = (f"Face {row['face_index']}   Confidence: {float(confidence):.3f}"
                   if confidence != "" else f"Face {row['face_index']}   Confidence: N/A")
        dashboard_text(output, heading, x1 + 14, top + 29, .55,
                       (255, 255, 255), 1)
        if row["mouth_available"]:
            dashboard_text(output, f"Openness: {float(row['mouth_openness']):.5f}",
                           x1 + 14, top + 61, .55)
            dashboard_text(output, f"Mouth width: {float(row['mouth_width']):.5f}",
                           x1 + 14, top + 89, .55)
            dashboard_text(output, f"Mouth height: {float(row['mouth_height']):.5f}",
                           x1 + 14, top + 117, .55)
            dashboard_text(output, "Status: mouth available", x1 + 14,
                           top + 143, .47, (170, 228, 235))
        else:
            dashboard_text(output, "Openness: unavailable", x1 + 14, top + 65, .55)
            dashboard_text(output, "Status: " + str(row["status"]).replace("_", " ").lower(),
                           x1 + 14, top + 105, .49, (180, 210, 245))
    if not shown:
        dashboard_text(output, "No faces above the display threshold",
                       x, start + 38, .48, (180, 210, 245))
        dashboard_text(output, "All detections remain in the CSV.",
                       x, start + 67, .47)
    return output


def analyze(frame: np.ndarray, pipeline: Any, capture: LandmarkCapture, source_name: str,
            source_hash: str, annotated_relative: str
            ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    capture.reset()
    prediction = pipeline.predict(frame)
    faces = list(prediction.instances)
    h, w = frame.shape[:2]
    preview = frame.copy()
    all_rows: list[dict[str, Any]] = []
    stats = {"raw_detections": len(faces), "landmarks_available": 0,
             "mouth_available": 0, "mouth_unavailable": 0, "displayed_detections": 0,
             "suppressed_by_display_threshold_or_invalid": 0}
    matched: set[int] = set()
    for index, face in enumerate(faces, start=1):
        confidence = (None if face.confidence is None else
                      numeric(face.confidence, "detector_confidence", 0.0, 1.0))
        box = bounded_box(face.box)
        features = face.face_features if isinstance(face.face_features, dict) else {}
        landmark_info = features.get("landmarks", {})
        mouth_info = features.get("mouth", {})
        if not isinstance(landmark_info, dict) or not isinstance(mouth_info, dict):
            raise DemoError("Unexpected FaceAnalysis feature schema")
        landmarks_avail = bool(landmark_info.get("available", False))
        landmark_count = int(landmark_info.get("count", 0) or 0)
        mouth_avail = bool(mouth_info.get("available", False))
        if landmarks_avail and landmark_count != EXPECTED_LANDMARKS:
            raise DemoError("Available landmarks did not have 478 points")
        if mouth_avail and not landmarks_avail:
            raise DemoError("MouthOpenness available without FaceLandmarks")
        if mouth_avail and box is None:
            raise DemoError("MouthOpenness available for invalid face box")
        row: dict[str, Any] = dict.fromkeys(CSV_FIELDS, "")
        row.update({"input": source_name, "source_sha256": source_hash,
                    "output_image": annotated_relative, "image_width": w,
                    "image_height": h, "face_index": index,
                    "detector_confidence": "" if confidence is None else confidence,
                    "landmarks_available": landmarks_avail, "landmark_count": landmark_count,
                    "mouth_available": mouth_avail, "displayed": False})
        if box is not None:
            row.update(dict(zip(("box_x1", "box_y1", "box_x2", "box_y2"), box)))
        native_landmarks = matching_points(box, capture, matched)
        if mouth_avail:
            if native_landmarks is None or len(native_landmarks) != EXPECTED_LANDMARKS:
                raise DemoError("Cannot capture landmarks underlying an available mouth output")
            coord = {}
            for name, point_index in POINTS.items():
                point = native_landmarks[point_index]
                x, y = numeric(point.x, name + ".x"), numeric(point.y, name + ".y")
                coord[name] = (x, y)
                row[name + "_x"] = x
                row[name + "_y"] = y
            independent = mouth_from_points(coord, w, h)
            for metric, label in zip(METRICS, ("openness", "width", "height")):
                observed = numeric(mouth_info.get(metric), metric, 0.0)
                row[metric] = observed
                row["independent_" + metric] = independent[metric]
                row[label + "_absolute_error"] = abs(observed - independent[metric])
                if row[label + "_absolute_error"] > TOLERANCE:
                    raise DemoError(f"Real MouthOpenness {metric} differs from independently computed geometry")
            if row["mouth_width"] <= 0:
                raise DemoError("Available mouth_width must be positive")
            row["status"] = "MOUTH_AVAILABLE"
            stats["mouth_available"] += 1
        else:
            row["status"] = ("INVALID_FACE_BOX" if box is None else
                             "MOUTH_UNAVAILABLE" if landmarks_avail else "LANDMARKS_UNAVAILABLE")
            stats["mouth_unavailable"] += 1
        if landmarks_avail:
            if native_landmarks is None or len(native_landmarks) != EXPECTED_LANDMARKS:
                raise DemoError("Landmarks were reported available but capture did not match")
            stats["landmarks_available"] += 1
        elif native_landmarks is not None and len(native_landmarks) == EXPECTED_LANDMARKS:
            raise DemoError("Captured available landmarks contradict pipeline unavailable flag")
        if box is not None and confidence is not None and confidence >= DISPLAY_CONFIDENCE:
            annotatable = max(0, min(w - 1, round(box[2]))) > max(0, min(w - 1, round(box[0]))) and \
                          max(0, min(h - 1, round(box[3]))) > max(0, min(h - 1, round(box[1])))
            if annotatable:
                annotate_face(preview, box, index, confidence, row["status"], row)
                row["displayed"] = True
                stats["displayed_detections"] += 1
        all_rows.append(row)
    if len(matched) != len(capture.calls):
        raise DemoError("A landmark prediction was not associated with an exported detected face")
    stats["suppressed_by_display_threshold_or_invalid"] = \
        stats["raw_detections"] - stats["displayed_detections"]
    return panel_image(frame, preview, source_name, len(faces),
                       stats["displayed_detections"], all_rows), all_rows, stats


def thumbnail(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    ratio = min(max_w / img.shape[1], max_h / img.shape[0])
    return cv2.resize(img, (max(1, round(img.shape[1] * ratio)),
                            max(1, round(img.shape[0] * ratio))),
                      interpolation=cv2.INTER_AREA)


def contact_sheet(stage: Path, records: list[dict[str, Any]]) -> np.ndarray:
    num_rows = (len(records) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = np.full((TILE_HEIGHT * num_rows, TILE_WIDTH * SHEET_COLUMNS, 3),
                    245, dtype=np.uint8)
    for i, record in enumerate(records):
        panel = read_image(stage / record["output_image"])
        source_width = record["image_width"]
        paired_view = panel[:, :2 * source_width + 3 * PANEL_MARGIN]
        thumb = thumbnail(paired_view, TILE_WIDTH - 14, TILE_HEIGHT - 58)
        top, left = divmod(i, SHEET_COLUMNS)
        px, py = left * TILE_WIDTH, top * TILE_HEIGHT
        th, tw = thumb.shape[:2]
        sheet[py + 6:py + 6 + th, px + (TILE_WIDTH - tw) // 2:px + (TILE_WIDTH + tw) // 2] = thumb
        message = (f"{record['input']} | faces={record['raw_detections']} "
                   f"| mouth={record['mouth_available']}")
        cv2.putText(sheet, message[:65], (px + 9, py + TILE_HEIGHT - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(sheet, "Synthetic illustration (no reference labels)",
                    (px + 9, py + TILE_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    .37, (75, 75, 75), 1, cv2.LINE_AA)
    return sheet


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def txt_content(summary: dict[str, Any]) -> str:
    return ("PhysioTrack MouthOpenness Synthetic Qualitative Demonstration\n\n" +
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n")


def write_summary(stage: Path, summary: dict[str, Any]) -> None:
    encoded = json.dumps(summary, indent=2, ensure_ascii=False,
                         sort_keys=True, allow_nan=False) + "\n"
    (stage / ARTIFACT_JSON).write_text(encoded, encoding="utf-8")
    (stage / ARTIFACT_TXT).write_text(txt_content(summary), encoding="utf-8")


def verify(stage: Path, summary: dict[str, Any]) -> None:
    """Reopen and independently audit every staged artifact before/after commit."""
    if not stage.is_dir() or stage.is_symlink():
        raise DemoError("Missing or unsafe staged output directory")
    with (stage / ARTIFACT_JSON).open(encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored != summary:
        raise DemoError("Stored JSON summary does not match in-memory summary")
    if (stage / ARTIFACT_TXT).read_text(encoding="utf-8") != txt_content(summary):
        raise DemoError("TXT and JSON summaries differ")
    records = summary["images"]
    if len(records) != summary["files_discovered"] or len(records) != summary["files_processed"]:
        raise DemoError("Input and output image count mismatch")
    with (stage / ARTIFACT_CSV).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise DemoError("CSV field schema mismatch")
        rows = list(reader)
    if len(rows) != summary["face_observations"]:
        raise DemoError("CSV face-row count mismatch")
    if file_hash(stage / ARTIFACT_CSV) != summary["artifact_sha256"][ARTIFACT_CSV]:
        raise DemoError("CSV hash mismatch")
    if file_hash(stage / ARTIFACT_SHEET) != summary["artifact_sha256"][ARTIFACT_SHEET]:
        raise DemoError("Contact-sheet hash mismatch")
    by_input = {rec["input"]: rec for rec in records}
    if len(by_input) != len(records):
        raise DemoError("Duplicate input in summary")
    observed: dict[str, list[dict[str, str]]] = {key: [] for key in by_input}
    for row in rows:
        key = row["input"]
        if key not in observed:
            raise DemoError("CSV row from unexpected input")
        observed[key].append(row)
    counted = {k: 0 for k in ("raw_detections", "landmarks_available", "mouth_available",
                             "mouth_unavailable", "displayed_detections",
                             "suppressed_by_display_threshold_or_invalid")}
    for rec in records:
        source = INPUT_DIR / rec["input"]
        if not source.is_file() or source.is_symlink() or file_hash(source) != rec["source_sha256"]:
            raise DemoError("Input image changed during generation: " + rec["input"])
        source_frame = read_image(source)
        if (source_frame.shape[1], source_frame.shape[0]) != (rec["image_width"], rec["image_height"]):
            raise DemoError("Input image dimensions changed")
        png = stage / rec["output_image"]
        if not png.is_file() or file_hash(png) != rec["output_sha256"]:
            raise DemoError("Annotated PNG missing or hash mismatch")
        panel = read_image(png)
        expected_height = dashboard_body_height(rec["image_height"],
                                               rec["displayed_detections"], rec["input"])
        if panel.shape[1] != 2 * rec["image_width"] + 4 * PANEL_MARGIN + DASHBOARD_WIDTH or \
                panel.shape[0] != expected_height + PANEL_HEADER + PANEL_FOOTER:
            raise DemoError("Incorrect annotated panel dimensions")
        if not np.array_equal(panel[PANEL_HEADER:PANEL_HEADER + rec["image_height"],
                                    PANEL_MARGIN:PANEL_MARGIN + rec["image_width"]], source_frame):
            raise DemoError("Annotated output did not preserve original source panel pixels")
        grouped = observed[rec["input"]]
        if len(grouped) != rec["raw_detections"]:
            raise DemoError("Raw detection count/CSV mismatch")
        if {int(row["face_index"]) for row in grouped} != set(range(1, len(grouped) + 1)):
            raise DemoError("Missing/duplicate face indices in CSV")
        per_image = {k: 0 for k in counted}
        per_image["raw_detections"] = len(grouped)
        for row in grouped:
            if row["source_sha256"] != rec["source_sha256"] or \
                    row["output_image"] != rec["output_image"]:
                raise DemoError("Mismatched CSV source or output name")
            if int(row["image_width"]) != rec["image_width"] or \
                    int(row["image_height"]) != rec["image_height"]:
                raise DemoError("Mismatched row dimensions")
            flag_landmarks = row["landmarks_available"] == "True"
            flag_mouth = row["mouth_available"] == "True"
            drawn = row["displayed"] == "True"
            if row["landmarks_available"] not in ("True", "False") or \
                    row["mouth_available"] not in ("True", "False") or \
                    row["displayed"] not in ("True", "False"):
                raise DemoError("Unrecognized CSV Boolean")
            if flag_landmarks:
                per_image["landmarks_available"] += 1
                if int(row["landmark_count"]) != EXPECTED_LANDMARKS:
                    raise DemoError("Invalid available landmark count")
            if flag_mouth:
                per_image["mouth_available"] += 1
                if not flag_landmarks or row["status"] != "MOUTH_AVAILABLE":
                    raise DemoError("Invalid mouth status or unavailable landmark prerequisite")
                xy = {key: (numeric(row[key + "_x"], key + ".x"),
                            numeric(row[key + "_y"], key + ".y")) for key in POINTS}
                check = mouth_from_points(xy, rec["image_width"], rec["image_height"])
                for metric, error_label in zip(METRICS, ("openness", "width", "height")):
                    native = numeric(row[metric], metric, 0.0)
                    reference = numeric(row["independent_" + metric], "independent_" + metric, 0.0)
                    error = numeric(row[error_label + "_absolute_error"], "error", 0.0)
                    if abs(reference - check[metric]) > TOLERANCE or \
                            abs(native - check[metric]) > TOLERANCE or \
                            abs(error - abs(native - reference)) > TOLERANCE:
                        raise DemoError("CSV geometry consistency mismatch")
            else:
                per_image["mouth_unavailable"] += 1
                if row["status"] not in ("INVALID_FACE_BOX", "MOUTH_UNAVAILABLE", "LANDMARKS_UNAVAILABLE"):
                    raise DemoError("Invalid unavailable mouth status")
                if any(row[key] != "" for key in METRICS):
                    raise DemoError("Unavailability row contains fabricated mouth predictions")
            if drawn:
                per_image["displayed_detections"] += 1
                if row["detector_confidence"] == "" or \
                        numeric(row["detector_confidence"], "confidence", 0, 1) < DISPLAY_CONFIDENCE:
                    raise DemoError("Annotation-threshold filter mismatch")
            if row["detector_confidence"] != "":
                numeric(row["detector_confidence"], "confidence", 0, 1)
        per_image["suppressed_by_display_threshold_or_invalid"] = \
            per_image["raw_detections"] - per_image["displayed_detections"]
        for key in counted:
            if per_image[key] != rec[key]:
                raise DemoError(f"Summary per-image count mismatch for {key}")
            counted[key] += per_image[key]
    if summary["totals"] != counted:
        raise DemoError("Summary total counters do not match CSV")
    if not np.array_equal(read_image(stage / ARTIFACT_SHEET), contact_sheet(stage, records)):
        raise DemoError("Contact-sheet pixels do not match source panels")
    expected_files = {ARTIFACT_JSON, ARTIFACT_TXT, ARTIFACT_CSV, ARTIFACT_SHEET}
    expected_files.update(rec["output_image"] for rec in records)
    actual_files = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise DemoError("Unexpected/missing synthetic-demo outputs: " + str(actual_files ^ expected_files))


def publish(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    """Replace only this script's own output directory, restoring it on failure."""
    verify(stage, summary)
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError("Unsafe previously published synthetic_demo path")
    backup = staging_root / "previous_accepted_synthetic_demo"
    previous = FINAL_DIR.exists()
    installed = False
    try:
        if previous:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        installed = True
        verify(FINAL_DIR, summary)
    except BaseException:
        try:
            if installed and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if previous and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as restore_error:
            raise DemoError(f"Publication failed, manual restoration required from {backup}") from restore_error
        raise
    if previous:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            print(f"WARNING: Published results are valid; stale backup remains at {backup}: {exc}")


def provenance(classes: dict[str, Any]) -> dict[str, Any]:
    paths = {Path(__file__).resolve()}
    for cls in (classes["Face"], classes["FaceAnalysis"], classes["FaceAnalysisConfig"],
                classes["FaceLandmarks"], classes["MouthOpenness"]):
        paths.add(Path(inspect.getfile(cls)).resolve())
    hashes = {}
    for file in sorted(paths):
        relative = file.relative_to(REPO_ROOT).as_posix() if file.is_relative_to(REPO_ROOT) else file.name
        hashes[relative] = file_hash(file)
    return {"python": platform.python_version(), "platform": platform.system(),
            "opencv": cv2.__version__, "numpy": np.__version__,
            "mediapipe": version("mediapipe"), "ultralytics": version("ultralytics"),
            "physiotrack": version("physiotrack"),
            "landmark_model_file": classes["model_path"].name,
            "landmark_model_sha256": file_hash(classes["model_path"]),
            "file_sha256": hashes}


def run() -> None:
    classes = components()
    sources = get_sources()
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError("Unsafe final output directory")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".mouth_openness_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    (stage / "annotated_images").mkdir()
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    pipeline = None
    failure_messages: list[str] = []
    try:
        print("PhysioTrack MouthOpenness Synthetic Qualitative Demonstration")
        print("Inputs: validation/synthetic_media/mouth_openness_media/")
        print("Outputs: validation/mouth_openness/results/synthetic_demo/")
        print("Images discovered:", len(sources))
        pipeline, capture = create_pipeline(classes)
        for i, source in enumerate(sources, 1):
            name = source.relative_to(INPUT_DIR).as_posix()
            dest = output_name(source)
            print(f"[{i}/{len(sources)}] {name}", flush=True)
            try:
                original = read_image(source)
                input_hash = file_hash(source)
                panel, observations, stats = analyze(original, pipeline, capture, name, input_hash, dest)
                panel_file = stage / dest
                save_image(panel_file, panel)
                if not np.array_equal(read_image(panel_file), panel):
                    raise DemoError("Encoded annotated panel did not retain its pixels")
                records.append({"input": name, "source_sha256": input_hash,
                                "output_image": dest, "output_sha256": file_hash(panel_file),
                                "image_width": original.shape[1], "image_height": original.shape[0],
                                "status": "SUCCESS" if stats["raw_detections"] else "NO_FACE",
                                **stats})
                rows.extend(observations)
                print(f"    faces={stats['raw_detections']} landmarks={stats['landmarks_available']} "
                      f"mouth={stats['mouth_available']} displayed={stats['displayed_detections']}",
                      flush=True)
            except Exception as exc:
                report = f"{name}: {type(exc).__name__}: {exc}"
                print("    FAILED:", report, flush=True)
                failure_messages.append(report)
        if failure_messages:
            raise DemoError(f"{len(failure_messages)} images failed: " + "; ".join(failure_messages[:5]))
        csv_write(stage / ARTIFACT_CSV, rows)
        save_image(stage / ARTIFACT_SHEET, contact_sheet(stage, records))
        counted_keys = ("raw_detections", "landmarks_available", "mouth_available",
                        "mouth_unavailable", "displayed_detections",
                        "suppressed_by_display_threshold_or_invalid")
        totals = {key: sum(r[key] for r in records) for key in counted_keys}
        summary = {
            "project": "physiotrack-v2", "component": "MouthOpenness",
            "type": "synthetic_qualitative_demonstration", "scientific_benchmark": False,
            "uses_ground_truth_mouth_labels": False, "calculates_mouth_accuracy_metrics": False,
            "source_directory": "validation/synthetic_media/mouth_openness_media",
            "output_directory": "validation/mouth_openness/results/synthetic_demo",
            "configuration": {
                "pipeline": "real Face detector -> FaceAnalysis -> FaceLandmarks -> MouthOpenness",
                "device": DEVICE, "detector_confidence": DETECTOR_CONFIDENCE,
                "detector_iou": DETECTOR_IOU, "maximum_detections": MAX_DETECTIONS,
                "display_confidence_threshold": DISPLAY_CONFIDENCE,
                "expected_landmark_count_when_available": EXPECTED_LANDMARKS,
                "numeric_check_tolerance": TOLERANCE,
                "landmark_indices": POINTS,
                "mouth_openness_definition": "pixel-distance(13,14) / pixel-distance(61,291)",
                "mouth_width_and_height": "Euclidean pixel distances divided by image width",
                "tracking_enabled": False, "unrelated_components_disabled": True,
                "visualization_layout": "unchanged source + geometrical overlay + separate side dashboard",
                "visualization_dashboard_width": DASHBOARD_WIDTH,
            },
            "component_imports": classes["imports"], "reproducibility": provenance(classes),
            "files_discovered": len(sources), "files_processed": len(records),
            "files_failed": 0, "face_observations": len(rows), "totals": totals,
            "images": records,
            "artifact_sha256": {ARTIFACT_CSV: file_hash(stage / ARTIFACT_CSV),
                                ARTIFACT_SHEET: file_hash(stage / ARTIFACT_SHEET)},
            "interpretation": (
                "Independent non-benchmark illustrations of native PhysioTrack FaceDetection, "
                "FaceLandmarks and MouthOpenness outputs. Synthetic expression and opening "
                "descriptions are nominal and are not quantitative ground truth. "
                "The report checks numerical geometry and output availability, not "
                "mouth-opening accuracy, detection recall, landmark accuracy, or temporal "
                "mouth movement. All detector outputs are retained in the CSV even if "
                "suppressed in annotated figures below the display confidence threshold."
            ),
            "status": "COMPLETE",
        }
        write_summary(stage, summary)
        print("Validating all staged outputs ...", flush=True)
        verify(stage, summary)
        publish(stage, staging_root, summary)
        print("Committed verified synthetic outputs:", FINAL_DIR)
        print(f"Images: {len(records)} | raw detections: {totals['raw_detections']} "
              f"| mouth available: {totals['mouth_available']} "
              f"| displayed detections: {totals['displayed_detections']}")
        print("Overall synthetic demonstration status: COMPLETE")
    finally:
        if pipeline is not None:
            pipeline.close()
        # Never delete a backup which may be the only recoverable previous version.
        if not (staging_root / "previous_accepted_synthetic_demo").exists():
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("Synthetic MouthOpenness demo FAILED. Previous accepted results remain where possible.")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
