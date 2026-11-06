"""Independent PhysioTrack Emotion Recognition synthetic qualitative demonstration.

Location: validation/emotion_recognition/emotion_recognition_synthetic_demo.py
Inputs:   validation/synthetic_media/emotion_recognition_media/01.png .. 12.png
Outputs:  validation/emotion_recognition/results/synthetic_demo/

The actual local FaceAnalysis FaceDetector -> FaceEmotion pipeline is executed on
12 numbered synthetic images (any OpenCV-decodable supported image format and
positive native dimensions). Every detected face is exported with its native eight-class
scores and checked independently by re-running the original FaceEmotion model on
the exact detected-face crop (as in emotion_recognition_component_test.py).

This is NOT a CAER-S benchmark. Scenario descriptions are nominal, not verified
ground-truth emotion labels: no accuracy, F1, or classification success rates are
calculated for these synthetic inputs. No post-hoc remapping or score changes.

Results are generated in a temporary sibling directory, reopened and numerically
verified, and then installed as a single replaceable synthetic_demo directory.
Any pre-existing directory is restored if a publishing step fails. Benchmark,
qualitative benchmark and component_execution outputs are never touched.
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
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "physiotrack"
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "emotion_recognition_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

EXPECTED_COUNT = 12
MODEL_NAME = "enet_b0_8_best_afew"
ENGINE = "onnx"
DEVICE = "cpu"
TOLERANCE = 1e-6
SCORE_SUM_TOLERANCE = 1e-5
CLASSES = (
    "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral",
    "Sadness", "Surprise",
)
SCENARIOS = {
    1: ("Neutral", "Neutral frontal expression"),
    2: ("Anger", "Angry expression"),
    3: ("Sadness", "Sad expression"),
    4: ("Fear", "Wide-eyed fearful expression with slightly open mouth"),
    5: ("Disgust", "Disgust expression with wrinkled nose"),
    6: ("Anger", "Angry expression with partial mouth occlusion by hand"),
    7: ("Happiness", "Visible natural smile"),
    8: ("Contempt", "Asymmetric contempt-like expression"),
    9: ("Surprise", "Wide-eyed surprised expression with open mouth"),
    10: ("Happiness", "Head turned with a slight smile"),
    11: ("Mixed", "Three distinct faces: happy, sad, neutral"),
    12: ("Sadness", "Sad expression under uneven low lighting"),
}

CSV_NAME = "emotion_recognition_observations.csv"
SHEET_NAME = "synthetic_demo_examples.png"
JSON_NAME = "synthetic_demo_summary.json"
TXT_NAME = "synthetic_demo_summary.txt"

# A standalone black sidebar: text NEVER overlaps facial pixels.
PANEL_GAP = 12
HEADER = 64
FOOTER = 58
SIDEBAR_WIDTH = 492
SIDEBAR_FACE_TOP = 288
SIDEBAR_CARD_HEIGHT = 352
SIDEBAR_CARD_GAP = 12
TILE_WIDTH = 470
TILE_HEIGHT = 350

CSV_FIELDS = (
    "input", "source_sha256", "output_image", "scenario_number",
    "scenario_hint", "image_width", "image_height", "face_index", "status",
    "box_x1", "box_y1", "box_x2", "box_y2", "detector_confidence",
    "emotion", "emotion_confidence",
    "score_anger", "score_contempt", "score_disgust", "score_fear",
    "score_happiness", "score_neutral", "score_sadness", "score_surprise",
    "score_sum", "direct_emotion", "direct_confidence",
    "direct_score_anger", "direct_score_contempt", "direct_score_disgust",
    "direct_score_fear", "direct_score_happiness", "direct_score_neutral",
    "direct_score_sadness", "direct_score_surprise", "label_match",
    "confidence_absolute_error", "maximum_score_absolute_error",
    "failure_reason",
)


class DemoError(RuntimeError):
    """An actionable validation or publication failure."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dependency_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_components() -> dict[str, Any]:
    """Guard against importing a different installed PhysioTrack wheel."""
    if not PACKAGE_ROOT.is_dir():
        raise DemoError("Expected src/physiotrack inside the repository checkout")
    source_path = str(REPOSITORY_ROOT / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from physiotrack.face.analysis import FaceAnalysis
    from physiotrack.face.config import FaceAnalysisConfig
    from physiotrack.face.emotion import FaceEmotion

    mapping = {
        "FaceAnalysis": FaceAnalysis,
        "FaceAnalysisConfig": FaceAnalysisConfig,
        "FaceEmotion": FaceEmotion,
    }
    imported = {}
    for key, cls in mapping.items():
        source = Path(inspect.getfile(cls)).resolve()
        if not source.is_relative_to(PACKAGE_ROOT.resolve()):
            raise DemoError(f"{key} was not imported from the local src/physiotrack")
        imported[key] = source.relative_to(REPOSITORY_ROOT).as_posix()
    mapping["imports"] = imported
    return mapping


def find_inputs() -> list[tuple[int, Path]]:
    if not INPUT_DIR.is_dir() or INPUT_DIR.is_symlink():
        raise DemoError("Missing or unsafe validation/synthetic_media/emotion_recognition_media")
    files = sorted((p for p in INPUT_DIR.rglob("*") if p.is_file() and not p.is_symlink()),
                   key=lambda p: p.relative_to(INPUT_DIR).as_posix().lower())
    image_files = [p for p in files if p.suffix.lower() in
                   (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")]
    if len(image_files) != EXPECTED_COUNT:
        raise DemoError(f"Expected exactly {EXPECTED_COUNT} synthetic images; found {len(image_files)}")
    numbered: dict[int, Path] = {}
    for p in image_files:
        match = re.match(r"^(\d{2})(?:$|[_-])", p.stem)
        if not match:
            raise DemoError("Input needs two-digit prefix 01..12: " + p.name)
        n = int(match.group(1))
        if n not in SCENARIOS or n in numbered:
            raise DemoError("Unexpected or duplicate synthetic scenario number: " + p.name)
        numbered[n] = p
    if set(numbered) != set(SCENARIOS):
        raise DemoError("Synthetic scenario 01..12 is incomplete")
    return sorted(numbered.items())


def read_image(path: Path) -> np.ndarray:
    try:
        pixels = np.fromfile(str(path), dtype=np.uint8)
        result = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise DemoError(f"Cannot read image: {path}") from exc
    if result is None or result.ndim != 3 or result.shape[2] != 3:
        raise DemoError(f"Invalid 3-channel image: {path}")
    # Image size/aspect ratio are properties of the media, not validation gates.
    # No resizing/cropping occurs before FaceAnalysis; source pixels stay intact.
    if min(result.shape[:2]) < 1:
        raise DemoError(f"Empty decoded image: {path}")
    return result


def save_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        raise DemoError("PNG encoding failed: " + str(path))
    buffer.tofile(str(path))


def output_name(number: int, image: Path) -> str:
    relative = image.relative_to(INPUT_DIR).as_posix()
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"annotated_images/{number:02d}_{suffix}_annotated.png"


def numeric(value: Any, field: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise DemoError(f"Missing or invalid numeric value {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DemoError(f"Invalid numeric value {field}") from exc
    if not math.isfinite(result):
        raise DemoError(f"Non-finite {field}")
    if minimum is not None and result < minimum:
        raise DemoError(f"{field} below permitted range")
    if maximum is not None and result > maximum:
        raise DemoError(f"{field} above permitted range")
    return result


def config_for_emotion(config_cls: Any) -> Any:
    config = config_cls(
        tracking=False, head_pose=False, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=True, regions=False, temporal=False,
        emotion_model=MODEL_NAME, emotion_engine=ENGINE,
    )
    for name in ("tracking", "head_pose", "landmarks", "quality", "eyes", "blink",
                 "gaze", "gaze_estimation", "mouth", "mouth_motion", "regions", "temporal"):
        if getattr(config, name) is not False:
            raise DemoError("Unrelated FaceAnalysis component was enabled: " + name)
    if config.emotion is not True or config.emotion_model != MODEL_NAME or config.emotion_engine != ENGINE:
        raise DemoError("The fixed FaceEmotion model and engine are not configured")
    if hasattr(config, "validate"):
        config.validate()
    return config


def validate_emotion_payload(payload: Any) -> tuple[str, float, dict[str, float], float]:
    """Validate exactly the native eight-class, unmodified output semantics."""
    if not isinstance(payload, dict) or not payload.get("available", False):
        raise DemoError("Emotion payload is unavailable")
    label, raw_scores = payload.get("emotion"), payload.get("scores")
    if label not in CLASSES or not isinstance(raw_scores, dict) or set(raw_scores) != set(CLASSES):
        raise DemoError("Emotion payload label or eight-class score schema is invalid")
    scores = {c: numeric(raw_scores[c], "score_" + c, 0.0, 1.0) for c in CLASSES}
    score_sum = float(sum(scores.values()))
    if not math.isclose(score_sum, 1.0, abs_tol=SCORE_SUM_TOLERANCE, rel_tol=0):
        raise DemoError("Emotion scores do not sum to one")
    if CLASSES[int(np.argmax(np.asarray([scores[c] for c in CLASSES])))] != label:
        raise DemoError("Predicted label is not the argmax of native scores")
    conf = numeric(payload.get("confidence"), "emotion_confidence", 0.0, 1.0)
    if abs(conf - scores[label]) > TOLERANCE:
        raise DemoError("Confidence differs from predicted-class probability")
    return label, conf, scores, score_sum


def clipped_crop(frame: np.ndarray, box: Any) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Identical round-to-nearest and image clipping to component_execution."""
    h, w = frame.shape[:2]
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError) as exc:
        raise DemoError("Invalid face bounding box") from exc
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    box_tuple = (x1, y1, x2, y2)
    if x2 <= x1 or y2 <= y1:
        return None, box_tuple
    crop = frame[y1:y2, x1:x2]
    return (crop if crop.size else None), box_tuple


def blank_row() -> dict[str, Any]:
    return {key: "" for key in CSV_FIELDS}


def analyze_image(original: np.ndarray, pipeline: Any, direct_model: Any,
                  input_name: str, image_sha: str, output_image: str,
                  number: int) -> tuple[list[dict[str, Any]], dict[str, int], np.ndarray]:
    """Run real FaceAnalysis; keep *every* detection and independent comparison."""
    result = pipeline.predict(original)
    instances = list(result)
    h, w = original.shape[:2]
    annotated = original.copy()
    observations: list[dict[str, Any]] = []
    counts = {"raw_detections": len(instances), "emotion_available": 0,
              "emotion_unavailable": 0, "invalid_face_box": 0,
              "no_face": 0, "displayed_detections": 0}

    if not instances:
        row = blank_row()
        row.update({"input": input_name, "source_sha256": image_sha,
                    "output_image": output_image, "scenario_number": number,
                    "scenario_hint": SCENARIOS[number][0],
                    "image_width": w, "image_height": h, "status": "NO_FACE",
                    "failure_reason": "", "label_match": ""})
        counts["no_face"] = 1
        observations.append(row)

    for face_index, face in enumerate(instances, 1):
        row = blank_row()
        row.update({"input": input_name, "source_sha256": image_sha,
                    "output_image": output_image, "scenario_number": number,
                    "scenario_hint": SCENARIOS[number][0],
                    "image_width": w, "image_height": h, "face_index": face_index})
        detector_conf = ("" if face.confidence is None else
                         numeric(face.confidence, "detector_confidence", 0.0, 1.0))
        row["detector_confidence"] = detector_conf
        if face.box is None:
            row["status"] = "INVALID_FACE_BOX"
            row["failure_reason"] = "missing_face_box"
            counts["invalid_face_box"] += 1
            observations.append(row)
            continue
        crop, box = clipped_crop(original, face.box)
        for name, value in zip(("box_x1", "box_y1", "box_x2", "box_y2"), box):
            row[name] = value
        if crop is None:
            row["status"] = "INVALID_FACE_BOX"
            row["failure_reason"] = "invalid_crop_after_clipping"
            counts["invalid_face_box"] += 1
            observations.append(row)
            continue
        features = face.face_features if isinstance(face.face_features, dict) else {}
        payload = features.get("emotion")
        if not isinstance(payload, dict) or not payload.get("available", False):
            row["status"] = "EMOTION_UNAVAILABLE"
            row["failure_reason"] = "emotion_payload_unavailable"
            counts["emotion_unavailable"] += 1
        else:
            label, confidence, scores, score_sum = validate_emotion_payload(payload)
            direct = direct_model.predict(crop)
            direct_label, direct_conf, direct_scores, _ = validate_emotion_payload(
                {"available": True, **direct})
            if label != direct_label:
                raise DemoError(f"Pipeline vs direct prediction mismatch: {input_name}, face {face_index}")
            confidence_error = abs(confidence - direct_conf)
            score_error = max(abs(scores[c] - direct_scores[c]) for c in CLASSES)
            if confidence_error > TOLERANCE or score_error > TOLERANCE:
                raise DemoError(f"Pipeline vs direct numeric mismatch: {input_name}, face {face_index}")
            row.update({"status": "EMOTION_AVAILABLE", "emotion": label,
                        "emotion_confidence": confidence,
                        "score_sum": score_sum, "direct_emotion": direct_label,
                        "direct_confidence": direct_conf, "label_match": True,
                        "confidence_absolute_error": confidence_error,
                        "maximum_score_absolute_error": score_error})
            for class_name in CLASSES:
                key = class_name.lower()
                row["score_" + key] = scores[class_name]
                row["direct_score_" + key] = direct_scores[class_name]
            counts["emotion_available"] += 1

        x1, y1, x2, y2 = box
        # All valid face detections are drawn; display is not filtered by scores.
        line_width = max(2, round(min(h, w) / 500))
        color = (55, 220, 55) if row["status"] == "EMOTION_AVAILABLE" else (0, 160, 255)
        cv2.rectangle(annotated, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), color, line_width)
        counts["displayed_detections"] += 1
        observations.append(row)

    return observations, counts, annotated


def font(canvas: np.ndarray, text: str, x: int, y: int, size: float = 0.56,
         color: tuple[int, int, int] = (231, 232, 231), thickness: int = 1) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, thickness, cv2.LINE_AA)


def wrap(value: str, max_width: int, size: float = 0.55) -> list[str]:
    def fits(text: str) -> bool:
        return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] <= max_width
    lines, current = [], ""
    for token in re.findall(r"\S+\s*", value):
        trial = current + token
        if fits(trial):
            current = trial
        else:
            if current:
                lines.append(current.rstrip())
                current = ""
            for char in token:
                if current and not fits(current + char):
                    lines.append(current.rstrip())
                    current = ""
                current += char
    if current:
        lines.append(current.rstrip())
    return lines or [""]


def draw_panel(original: np.ndarray, annotated: np.ndarray, input_name: str,
               number: int, rows: list[dict[str, Any]], counts: dict[str, int]) -> np.ndarray:
    """Unmodified input + annotated copy + separate legible eight-score cards."""
    h, w = original.shape[:2]
    title_lines = wrap(input_name, SIDEBAR_WIDTH - 46, 0.52)
    cards_top = max(SIDEBAR_FACE_TOP, 153 + len(title_lines) * 25 + 95)
    cards_count = max(1, len(rows))
    body = max(h, cards_top + cards_count * (SIDEBAR_CARD_HEIGHT + SIDEBAR_CARD_GAP) + 12)
    width = 2 * w + 4 * PANEL_GAP + SIDEBAR_WIDTH
    canvas = np.full((HEADER + body + FOOTER, width, 3), (25, 27, 28), dtype=np.uint8)
    first_x = PANEL_GAP
    second_x = 2 * PANEL_GAP + w
    sidebar_x = 3 * PANEL_GAP + 2 * w
    canvas[HEADER:HEADER + h, first_x:first_x + w] = original
    canvas[HEADER:HEADER + h, second_x:second_x + w] = annotated
    font(canvas, "Synthetic source", first_x, 38, .72, (255, 255, 255), 2)
    font(canvas, "PhysioTrack FaceEmotion", second_x, 38, .72, (255, 255, 255), 2)
    footer = f"{input_name} | nominal: {SCENARIOS[number][0]} | raw faces: {counts['raw_detections']} | available: {counts['emotion_available']}"
    for i, line in enumerate(wrap(footer, 2 * w + SIDEBAR_WIDTH + PANEL_GAP, .51)):
        font(canvas, line, PANEL_GAP, HEADER + body + 28 + i * 20, .51)
    x = sidebar_x + 22
    font(canvas, "PhysioTrack", x, HEADER + 42, .88, (255, 255, 255), 2)
    font(canvas, "EMOTION RECOGNITION", x, HEADER + 72, .55, (183, 231, 232))
    cv2.line(canvas, (x, HEADER + 88),
             (sidebar_x + SIDEBAR_WIDTH - 22, HEADER + 88), (72, 78, 78), 1, cv2.LINE_AA)
    font(canvas, "IMAGE", x, HEADER + 120, .51, (166, 171, 173))
    for i, line in enumerate(title_lines):
        font(canvas, line, x, HEADER + 151 + i * 25, .52)
    row_y = HEADER + 153 + 25 * len(title_lines)
    font(canvas, f"Scenario: {number:02d} (nominal only)", x, row_y + 19, .51)
    font(canvas, f"Detected faces: {counts['raw_detections']}", x, row_y + 45, .55)
    font(canvas, f"FaceEmotion available: {counts['emotion_available']}", x, row_y + 70, .55)
    top = HEADER + cards_top
    for idx, row in enumerate(rows):
        card_y = top + idx * (SIDEBAR_CARD_HEIGHT + SIDEBAR_CARD_GAP)
        cv2.rectangle(canvas, (sidebar_x + 13, card_y),
                      (sidebar_x + SIDEBAR_WIDTH - 13, card_y + SIDEBAR_CARD_HEIGHT - 8),
                      (36, 42, 43), -1)
        accent = (55, 220, 55) if row["status"] == "EMOTION_AVAILABLE" else (0, 160, 255)
        cv2.rectangle(canvas, (sidebar_x + 13, card_y),
                      (sidebar_x + 18, card_y + SIDEBAR_CARD_HEIGHT - 8), accent, -1)
        prefix = f"Face {row['face_index']}" if row["face_index"] != "" else "No face"
        font(canvas, f"{prefix}  |  {row['status']}", x, card_y + 27, .54, (255, 255, 255), 2)
        if row["status"] == "EMOTION_AVAILABLE":
            font(canvas, f"Predicted: {row['emotion']}", x, card_y + 57, .63, (255, 255, 255), 2)
            font(canvas, f"Confidence: {float(row['emotion_confidence']):.6f}", x, card_y + 84, .54)
            font(canvas, "Native 8-class probabilities", x, card_y + 109, .49, (175, 218, 222))
            for j, name in enumerate(CLASSES):
                score = float(row["score_" + name.lower()])
                mark = "*" if name == row["emotion"] else " "
                font(canvas, f"{mark} {name:<10}: {score:.6f}", x,
                     card_y + 134 + j * 25, .50,
                     (225, 244, 245) if mark == "*" else (206, 211, 213))
        else:
            font(canvas, "No valid emotion scores", x, card_y + 69, .58)
            if row["failure_reason"]:
                font(canvas, str(row["failure_reason"])[:50], x, card_y + 103, .43)
    return canvas


def thumbnail(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    ratio = min(max_w / image.shape[1], max_h / image.shape[0])
    return cv2.resize(image, (max(1, int(round(image.shape[1] * ratio))),
                              max(1, int(round(image.shape[0] * ratio)))),
                      interpolation=cv2.INTER_AREA)


def contact_sheet(folder: Path, records: list[dict[str, Any]]) -> np.ndarray:
    columns, n = 3, len(records)
    rows = int(math.ceil(n / columns))
    output = np.full((rows * TILE_HEIGHT, columns * TILE_WIDTH, 3), (246, 246, 246), dtype=np.uint8)
    for i, record in enumerate(records):
        panel = read_panel_png(folder / record["output_image"])
        small = thumbnail(panel, TILE_WIDTH - 16, TILE_HEIGHT - 70)
        col, row = i % columns, i // columns
        x = col * TILE_WIDTH + (TILE_WIDTH - small.shape[1]) // 2
        y = row * TILE_HEIGHT + 6
        output[y:y + small.shape[0], x:x + small.shape[1]] = small
        label = f"{record['scenario_number']:02d}.png | faces {record['raw_detections']} | available {record['emotion_available']}"
        font(output, label, col * TILE_WIDTH + 12, row * TILE_HEIGHT + TILE_HEIGHT - 36,
             .46, (15, 15, 15))
        font(output, "Synthetic illustration (no reference emotion labels)",
             col * TILE_WIDTH + 12, row * TILE_HEIGHT + TILE_HEIGHT - 15,
             .36, (75, 75, 75))
    return output


def read_panel_png(path: Path) -> np.ndarray:
    try:
        frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise DemoError("Could not read annotated PNG: " + str(path)) from exc
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise DemoError("Invalid saved annotated PNG: " + str(path))
    return frame


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "PhysioTrack Emotion Recognition Synthetic Qualitative Demonstration",
        "===============================================================",
        "",
        "Type: execution and visualization, NOT scientific emotion accuracy",
        "Model: " + MODEL_NAME + " (" + ENGINE + ")",
        "Source: " + summary["source_directory"],
        "Output: " + summary["output_directory"],
        f"Synthetic images: {summary['images_processed']}",
        f"Image processing failures: {summary['images_failed']}",
        f"CSV observations: {summary['observation_rows']}",
        f"Raw detected faces: {summary['totals']['raw_detections']}",
        f"Emotion available: {summary['totals']['emotion_available']}",
        f"Emotion unavailable: {summary['totals']['emotion_unavailable']}",
        f"Invalid face boxes: {summary['totals']['invalid_face_box']}",
        f"Images with no faces: {summary['totals']['no_face']}",
        f"Max direct confidence difference: {summary['numerical_consistency']['maximum_confidence_absolute_error']:.12g}",
        f"Max direct score difference: {summary['numerical_consistency']['maximum_score_absolute_error']:.12g}",
        f"Runtime seconds: {summary['runtime_seconds']:.3f}",
        f"Status: {summary['status']}",
        "",
        "This is not a CAER-S evaluation; nominal synthetic expressions are not",
        "ground truth. No synthetic emotion accuracy, F1 or other accuracy",
        "metrics are computed. Every real detector output is represented in CSV.",
        "",
        "Per-image results:",
    ]
    for record in summary["images"]:
        lines.append(f"  {record['input']}: detected={record['raw_detections']}, "
                     f"available={record['emotion_available']}, status={record['status']}")
    return "\n".join(lines) + "\n"


def write_summaries(folder: Path, summary: dict[str, Any]) -> None:
    (folder / JSON_NAME).write_text(json.dumps(summary, indent=2, sort_keys=True,
                                              ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (folder / TXT_NAME).write_text(summary_text(summary), encoding="utf-8")


def audit_rows(rows: list[dict[str, str]], records: list[dict[str, Any]]) -> dict[str, int]:
    """Recalculate all counts, score sums and direct comparisons from the CSV."""
    known = {r["input"]: r for r in records}
    if len(known) != len(records):
        raise DemoError("Duplicate inputs in synthetic summary")
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in known}
    totals = dict.fromkeys(("raw_detections", "emotion_available", "emotion_unavailable",
                            "invalid_face_box", "no_face", "displayed_detections"), 0)
    for row in rows:
        if row["input"] not in grouped:
            raise DemoError("Unexpected input in observations CSV")
        grouped[row["input"]].append(row)
    for name, record in known.items():
        group = grouped[name]
        no_face = [r for r in group if r["status"] == "NO_FACE"]
        face_rows = [r for r in group if r["status"] != "NO_FACE"]
        if no_face and (len(no_face) != 1 or face_rows):
            raise DemoError("Invalid NO_FACE sentinel count")
        if not face_rows and not no_face:
            raise DemoError("Missing image observation")
        if {int(r["face_index"]) for r in face_rows} != set(range(1, len(face_rows) + 1)):
            raise DemoError("Missing or duplicate face indices")
        if any(r["source_sha256"] != record["source_sha256"] or
               r["output_image"] != record["output_image"] or
               int(r["scenario_number"]) != record["scenario_number"] or
               int(r["image_width"]) != record["image_width"] or
               int(r["image_height"]) != record["image_height"] or
               r["scenario_hint"] != record["scenario_hint"] for r in group):
            raise DemoError("Observation source or scenario metadata mismatch")
        per = dict.fromkeys(totals, 0)
        per["raw_detections"] = len(face_rows)
        per["no_face"] = len(no_face)
        per["displayed_detections"] = sum(r["status"] != "INVALID_FACE_BOX" for r in face_rows)
        for row in face_rows:
            if row["detector_confidence"] != "":
                numeric(row["detector_confidence"], "detector_confidence", 0.0, 1.0)
            status = row["status"]
            if status not in ("EMOTION_AVAILABLE", "EMOTION_UNAVAILABLE", "INVALID_FACE_BOX"):
                raise DemoError("Unexpected emotion execution status: " + status)
            if status == "INVALID_FACE_BOX":
                per["invalid_face_box"] += 1
                if row["emotion"] or row["emotion_confidence"]:
                    raise DemoError("Invalid-face row contains emotion prediction")
                continue
            x1, y1, x2, y2 = [int(row[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2")]
            if not (0 <= x1 < x2 <= record["image_width"] and
                    0 <= y1 < y2 <= record["image_height"]):
                raise DemoError("Recorded face crop outside input image")
            if status == "EMOTION_UNAVAILABLE":
                per["emotion_unavailable"] += 1
                if row["emotion"] or row["emotion_confidence"]:
                    raise DemoError("Unavailable row contains fabricated prediction")
                continue
            per["emotion_available"] += 1
            scores = {c: numeric(row["score_" + c.lower()], c, 0.0, 1.0) for c in CLASSES}
            direct = {c: numeric(row["direct_score_" + c.lower()], "direct " + c, 0.0, 1.0)
                      for c in CLASSES}
            label = row["emotion"]
            if label not in CLASSES or label != row["direct_emotion"] or row["label_match"] != "True":
                raise DemoError("Pipeline and direct labels disagree")
            conf = numeric(row["emotion_confidence"], "confidence", 0.0, 1.0)
            direct_conf = numeric(row["direct_confidence"], "direct confidence", 0.0, 1.0)
            raw_sum = sum(scores.values())
            direct_sum = sum(direct.values())
            if abs(raw_sum - 1.0) > SCORE_SUM_TOLERANCE or \
                    abs(direct_sum - 1.0) > SCORE_SUM_TOLERANCE or \
                    abs(raw_sum - numeric(row["score_sum"], "score_sum")) > SCORE_SUM_TOLERANCE:
                raise DemoError("Native or direct recorded scores do not sum to one")
            if CLASSES[int(np.argmax([scores[c] for c in CLASSES]))] != label:
                raise DemoError("Predicted label differs from score argmax")
            if abs(conf - scores[label]) > TOLERANCE or abs(direct_conf - direct[label]) > TOLERANCE:
                raise DemoError("Recorded confidence differs from class score")
            max_error = max(abs(scores[c] - direct[c]) for c in CLASSES)
            conf_error = abs(conf - direct_conf)
            if abs(max_error - numeric(row["maximum_score_absolute_error"], "score error")) > TOLERANCE or \
                    abs(conf_error - numeric(row["confidence_absolute_error"], "confidence error")) > TOLERANCE:
                raise DemoError("Direct-check error recording mismatch")
            if max_error > TOLERANCE or conf_error > TOLERANCE:
                raise DemoError("Direct FaceEmotion result does not match FaceAnalysis")
        for key in per:
            if per[key] != record[key]:
                raise DemoError(f"Per-image summary count mismatch: {name}: {key}")
            totals[key] += per[key]
    return totals


def verify_outputs(folder: Path, summary: dict[str, Any]) -> None:
    """Reopen every image, verify manifests, CSV, math and contact sheet."""
    if not folder.is_dir() or folder.is_symlink():
        raise DemoError("Unsafe synthetic output folder")
    loaded = json.loads((folder / JSON_NAME).read_text(encoding="utf-8"))
    if loaded != summary:
        raise DemoError("Stored JSON does not match execution summary")
    if (folder / TXT_NAME).read_text(encoding="utf-8") != summary_text(summary):
        raise DemoError("TXT and JSON summaries contradict each other")
    if sha256(folder / CSV_NAME) != summary["artifact_sha256"][CSV_NAME] or \
            sha256(folder / SHEET_NAME) != summary["artifact_sha256"][SHEET_NAME]:
        raise DemoError("CSV or contact-sheet hash mismatch")
    with (folder / CSV_NAME).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise DemoError("Unexpected observation CSV schema")
        observed = list(reader)
    if len(observed) != summary["observation_rows"]:
        raise DemoError("CSV observation count mismatch")
    recalculated = audit_rows(observed, summary["images"])
    if recalculated != summary["totals"]:
        raise DemoError("Summary totals are not backed by CSV rows")
    if len(summary["images"]) != EXPECTED_COUNT:
        raise DemoError("Incorrect synthetic input population")
    for record in summary["images"]:
        source = INPUT_DIR / record["input"]
        if not source.is_file() or source.is_symlink() or sha256(source) != record["source_sha256"]:
            raise DemoError("A source image changed during synthetic execution")
        image = read_image(source)
        annotated_path = folder / record["output_image"]
        if not annotated_path.is_file() or sha256(annotated_path) != record["output_sha256"]:
            raise DemoError("Annotated image hash differs from summary")
        panel = read_panel_png(annotated_path)
        if panel.shape[1] != 2 * record["image_width"] + 4 * PANEL_GAP + SIDEBAR_WIDTH:
            raise DemoError("Annotated panel width is incorrect")
        if panel.shape[0] < HEADER + record["image_height"] + FOOTER:
            raise DemoError("Annotated panel height is insufficient")
        if not np.array_equal(panel[HEADER:HEADER + record["image_height"],
                                    PANEL_GAP:PANEL_GAP + record["image_width"]], image):
            raise DemoError("Source image was modified inside annotated panel")
    re_rendered = contact_sheet(folder, summary["images"])
    if not np.array_equal(read_panel_png(folder / SHEET_NAME), re_rendered):
        raise DemoError("Contact-sheet PNG differs from recomputed layout")
    expected = {CSV_NAME, JSON_NAME, TXT_NAME, SHEET_NAME}
    expected.update(rec["output_image"] for rec in summary["images"])
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    if actual != expected:
        raise DemoError("Unexpected or missing files in synthetic_demo directory: " + str(actual ^ expected))


def publish(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    """Replace only synthetic_demo and recover old results on any publication failure."""
    verify_outputs(stage, summary)
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError("Unsafe previous synthetic_demo output")
    previous = FINAL_DIR.exists()
    backup = staging_root / "previous_accepted_synthetic_demo"
    installed = False
    try:
        if previous:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        installed = True
        verify_outputs(FINAL_DIR, summary)
    except BaseException:
        try:
            if installed and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if previous and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as restore_exc:
            raise DemoError(f"Rollback requires manual recovery from: {backup}") from restore_exc
        raise
    if previous and backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            print("WARNING: Previous result backup is preserved at", backup)


def provenance(classes: dict[str, Any]) -> dict[str, Any]:
    paths = {Path(__file__).resolve()}
    for key in ("FaceAnalysis", "FaceAnalysisConfig", "FaceEmotion"):
        paths.add(Path(inspect.getfile(classes[key])).resolve())
    hashes = {}
    for p in sorted(paths):
        key = p.relative_to(REPOSITORY_ROOT).as_posix() if p.is_relative_to(REPOSITORY_ROOT) else p.name
        hashes[key] = sha256(p)
    return {
        "python": platform.python_version(), "platform": platform.system(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "physiotrack": dependency_version("physiotrack"),
        "ultralytics": dependency_version("ultralytics"),
        "emotiefflib": dependency_version("emotiefflib"),
        "onnxruntime": dependency_version("onnxruntime"),
        "source_files_sha256": hashes,
    }


def run() -> None:
    classes = load_components()
    sources = find_inputs()
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise DemoError("Existing synthetic_demo is unsafe to replace")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".emotion_synthetic_stage_", dir=RESULTS_DIR))
    stage = stage_root / "synthetic_demo"
    stage.mkdir()
    (stage / "annotated_images").mkdir()
    pipeline = None
    direct_model = None
    records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        print("PhysioTrack Emotion Recognition Synthetic Qualitative Demo")
        print("Input: validation/synthetic_media/emotion_recognition_media/ (12 numbered images, native resolutions)")
        print("Output: validation/emotion_recognition/results/synthetic_demo/")
        print("Using real FaceAnalysis emotion path + independent FaceEmotion checks")
        config = config_for_emotion(classes["FaceAnalysisConfig"])
        pipeline = classes["FaceAnalysis"](config=config, device=DEVICE, verbose=False)
        if not isinstance(pipeline.emotion, classes["FaceEmotion"]):
            raise DemoError("FaceAnalysis did not construct the intended FaceEmotion model")
        for disabled in ("tracker", "orientation", "landmarks", "quality", "eyes", "blink",
                         "gaze", "gaze_estimation", "mouth", "mouth_motion", "regions", "temporal"):
            if getattr(pipeline, disabled) is not None:
                raise DemoError("Unexpected active optional module: " + disabled)
        direct_model = classes["FaceEmotion"](model_name=MODEL_NAME, engine=ENGINE)
        for i, (number, source) in enumerate(sources, 1):
            relative = source.relative_to(INPUT_DIR).as_posix()
            source_sha = sha256(source)
            original = read_image(source)
            destination = output_name(number, source)
            print(f"[{i:02d}/{len(sources):02d}] {relative}", flush=True)
            rows, counts, annotated = analyze_image(original, pipeline, direct_model,
                                                   relative, source_sha, destination, number)
            panel = draw_panel(original, annotated, relative, number, rows, counts)
            save_png(stage / destination, panel)
            if not np.array_equal(read_panel_png(stage / destination), panel):
                raise DemoError("Saved image panel does not match original pixels")
            record = {
                "scenario_number": number, "scenario_hint": SCENARIOS[number][0],
                "scenario_description": SCENARIOS[number][1], "input": relative,
                "source_sha256": source_sha, "output_image": destination,
                "output_sha256": sha256(stage / destination),
                "image_width": original.shape[1], "image_height": original.shape[0],
                "status": ("NO_FACE" if counts["no_face"] else
                           "EMOTION_UNAVAILABLE" if counts["emotion_unavailable"] or counts["invalid_face_box"]
                           else "SUCCESS"),
                **counts,
            }
            records.append(record)
            all_rows.extend(rows)
            print(f"    raw faces={counts['raw_detections']} | emotion available={counts['emotion_available']}"
                  f" | unavailable={counts['emotion_unavailable']} | no face={counts['no_face']}", flush=True)
        if len(records) != EXPECTED_COUNT:
            raise DemoError("Not all synthetic images were processed")
        write_csv(stage / CSV_NAME, all_rows)
        save_png(stage / SHEET_NAME, contact_sheet(stage, records))
        totals = {key: sum(int(r[key]) for r in records)
                  for key in ("raw_detections", "emotion_available", "emotion_unavailable",
                              "invalid_face_box", "no_face", "displayed_detections")}
        direct_rows = [r for r in all_rows if r["status"] == "EMOTION_AVAILABLE"]
        limitations = []
        if totals["no_face"]:
            limitations.append(f"No face was detected in {totals['no_face']} synthetic images")
        if totals["emotion_unavailable"] or totals["invalid_face_box"]:
            limitations.append("Some real detected faces had no usable emotion output")
        multi_faces = records[10]["raw_detections"]
        if multi_faces < 3:
            limitations.append(f"Scenario 11 supplied three faces but {multi_faces} were detected")
        predicted_counts = dict(sorted(Counter(r["emotion"] for r in direct_rows).items()))
        summary = {
            "project": "physiotrack-v2", "component": "FaceEmotion",
            "evidence_type": "synthetic_qualitative_demonstration",
            "scientific_accuracy_evidence": False,
            "uses_ground_truth_emotion_labels": False,
            "computes_accuracy_f1_or_detection_recall": False,
            "source_directory": "validation/synthetic_media/emotion_recognition_media",
            "output_directory": "validation/emotion_recognition/results/synthetic_demo",
            "images_expected": EXPECTED_COUNT, "images_processed": len(records),
            "images_failed": 0, "observation_rows": len(all_rows),
            "configuration": {
                "execution_path": "FaceAnalysis -> Face detector -> FaceEmotion",
                "model": MODEL_NAME, "engine": ENGINE, "device": DEVICE,
                "enabled_optional_components": ["emotion"],
                "disabled_optional_components": [
                    "tracking", "head_pose", "landmarks", "quality", "eyes", "blink",
                    "gaze", "gaze_estimation", "mouth", "mouth_motion", "regions", "temporal"],
                "class_order": list(CLASSES), "probability_sum_tolerance": SCORE_SUM_TOLERANCE,
                "direct_comparison_absolute_tolerance": TOLERANCE,
                "visualization": "source unchanged + bounding box image + readable detached eight-class cards",
                "annotation_dashboard_width": SIDEBAR_WIDTH,
                "synthetic_nominal_scenarios": {
                    f"{number:02d}": {"nominal": hint, "description": desc}
                    for number, (hint, desc) in SCENARIOS.items()},
            },
            "component_imports": classes["imports"],
            "reproducibility": provenance(classes),
            "totals": totals, "predicted_class_counts": predicted_counts,
            "numerical_consistency": {
                "maximum_confidence_absolute_error": max(
                    (float(r["confidence_absolute_error"]) for r in direct_rows), default=0.0),
                "maximum_score_absolute_error": max(
                    (float(r["maximum_score_absolute_error"]) for r in direct_rows), default=0.0),
                "comparisons": len(direct_rows),
            },
            "images": records, "runtime_seconds": time.perf_counter() - start,
            "artifact_sha256": {CSV_NAME: sha256(stage / CSV_NAME),
                                SHEET_NAME: sha256(stage / SHEET_NAME)},
            "limitations": limitations,
            "interpretation": (
                "Demonstrates the native FaceAnalysis/FaceEmotion output on independent synthetic "
                "image cases. Nominal depicted expressions are illustration descriptions, NOT "
                "ground-truth labels. No accuracy is calculated or claimed. Every face (including "
                "unavailable predictions) is preserved in CSV and in the scene display, and each "
                "valid native eight-class score vector is compared with direct FaceEmotion inference "
                "on the identical detector crop. The accepted CAER-S benchmark remains separate."),
            "status": "COMPLETE_WITH_LIMITATIONS" if limitations else "COMPLETE",
        }
        write_summaries(stage, summary)
        print("Auditing staged synthetic outputs ...", flush=True)
        verify_outputs(stage, summary)
        publish(stage, stage_root, summary)
        print("Published:", FINAL_DIR)
        print(f"Images: {len(records)} | Detected face rows: {totals['raw_detections']} | "
              f"Emotion available: {totals['emotion_available']} | Missing images: {totals['no_face']}")
        print("Maximum independent score error:",
              summary["numerical_consistency"]["maximum_score_absolute_error"])
        print("Synthetic demo status:", summary["status"])
    finally:
        for obj in (pipeline, direct_model):
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        # A failed rollback may leave the last valid accepted version here.
        if not (stage_root / "previous_accepted_synthetic_demo").exists():
            shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("Emotion Recognition synthetic demo FAILED; previous accepted outputs were not intentionally replaced.")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
