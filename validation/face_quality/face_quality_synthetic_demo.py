"""Independent synthetic qualitative demonstration of PhysioTrack FaceQuality.

Install as validation/face_quality/face_quality_synthetic_demo.py.
Input:  validation/synthetic_media/face_quality_media/ (recursive discovery).
Output: validation/face_quality/results/synthetic_demo/ (owned exclusively here).

Executes REAL Face -> FaceAnalysis -> FaceQuality on synthetic/non-benchmark
images. Saves all raw per-face descriptors and separate visually filtered
annotations. Makes NO quality-accuracy, robustness-rate, or benchmark claims.

A fresh staging directory is validated before publication. A successful run
replaces the entire prior synthetic_demo tree; a failed run preserves it.
This script never modifies 300-W data or benchmark/component-test outputs.
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

# Import from this repository as ensured by the active PhysioTrack-v2 environment.
from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.quality import FaceQuality

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "face_quality_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.001
NMS_IOU = 0.45
MAX_DETECTIONS = 10000
DISPLAY_CONFIDENCE = 0.25  # Rendering filter only; ALL model outputs are exported.
NUMERIC_TOLERANCE = 1e-12
SHEET_COLS = 3
SHEET_TILE_W = 470
SHEET_TILE_H = 320

CSV_FIELDS = (
    "input", "source_sha256", "output_image", "image_width", "image_height",
    "face_index", "status", "displayed", "detector_confidence", "quality_confidence",
    "box_x1", "box_y1", "box_x2", "box_y2", "brightness", "sharpness",
    "face_area_ratio", "expected_brightness", "expected_sharpness",
    "expected_face_area_ratio", "brightness_absolute_error",
    "sharpness_absolute_error", "face_area_ratio_absolute_error",
    "confidence_absolute_error",
)


class DemoError(RuntimeError):
    """Unsafe or incomplete demonstration execution."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def project_imports() -> dict[str, str]:
    base = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    if not base.is_dir():
        raise DemoError("Local repository src/physiotrack directory not found")
    paths: dict[str, str] = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig, FaceQuality):
        location = Path(inspect.getfile(cls)).resolve()
        if not location.is_relative_to(base):
            raise DemoError(
                f"{cls.__name__} is not imported from this physiotrack-v2 checkout. "
                "Activate its environment/editable installation."
            )
        paths[cls.__name__] = location.relative_to(REPOSITORY_ROOT).as_posix()
    return paths


def discover_images() -> list[Path]:
    if not INPUT_DIR.is_dir() or INPUT_DIR.is_symlink():
        raise DemoError("Missing or unsafe validation/synthetic_media/face_quality_media folder")
    images = sorted(
        (p for p in INPUT_DIR.rglob("*") if p.is_file() and not p.is_symlink()
         and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(),
                       p.relative_to(INPUT_DIR).as_posix()),
    )
    if not images:
        raise DemoError("No supported synthetic image found; old results unchanged")
    paths = [p.relative_to(INPUT_DIR).as_posix().casefold() for p in images]
    if len(paths) != len(set(paths)):
        raise DemoError("Ambiguous input filenames that differ only by letter case")
    return images


def relative_output(source: Path) -> str:
    relative = source.relative_to(INPUT_DIR).as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip(". ")[:70] or "image"
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"annotated_images/{safe_stem}_{suffix}.png"


def read_image(path: Path) -> np.ndarray:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise DemoError(f"Cannot read image: {path.name}") from exc
    if (image is None or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 1):
        raise DemoError(f"Invalid image: {path.name}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise DemoError(f"Cannot encode PNG {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def make_pipeline() -> FaceAnalysis:
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=False, quality=True,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )
    config.validate()
    detector = Face(
        device=DEVICE, conf=DETECTOR_CONFIDENCE, iou=NMS_IOU,
        max_det=MAX_DETECTIONS, verbose=False,
    )
    pipeline = FaceAnalysis(detector=detector, config=config, device=DEVICE, verbose=False)
    try:
        if pipeline.detector is None or pipeline.quality is None or not isinstance(pipeline.quality, FaceQuality):
            raise DemoError("Expected Face detector and FaceQuality to be active")
        for attribute in (
            "tracker", "orientation", "landmarks", "eyes", "blink", "gaze",
            "gaze_estimation", "mouth", "mouth_motion", "emotion", "regions", "temporal",
        ):
            if getattr(pipeline, attribute) is not None:
                raise DemoError(f"Unrelated FaceAnalysis component unexpectedly active: {attribute}")
        return pipeline
    except BaseException:
        pipeline.close()
        raise


def valid_confidence(raw: Any, *, allow_none: bool = True) -> float | None:
    if raw is None:
        if allow_none:
            return None
        raise DemoError("Missing detector confidence")
    value = float(raw)
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise DemoError("Detector/quality confidence must be finite and inside [0,1]")
    return value


def independent_descriptors(frame: np.ndarray, box: list[float]) -> dict[str, float] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean() / 255.0),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "face_area_ratio": float((x2 - x1) * (y2 - y1) / (width * height)),
    }


def draw_text(image: np.ndarray, label: str, x: int, y: int, size: float = 0.5,
              color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Readable text over any synthetic lighting; does not affect measurements."""
    width, height = image.shape[1], image.shape[0]
    x = max(3, min(x, width - 3))
    y = max(15, min(y, height - 4))
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                color, 1, cv2.LINE_AA)


def make_panel(source: np.ndarray, annotated: np.ndarray,
               input_name: str, raw_count: int, displayed_count: int) -> np.ndarray:
    height, width = source.shape[:2]
    margin, header, footer = 14, 72, 58
    canvas = np.full((height + header + footer, 2 * width + 3 * margin, 3),
                     30, dtype=np.uint8)
    canvas[header:header + height, margin:margin + width] = source
    canvas[header:header + height, 2 * margin + width:2 * margin + 2 * width] = annotated
    title_size = max(0.38, min(0.85, width / 1100.0))
    draw_text(canvas, "Synthetic source", margin, 37, title_size)
    draw_text(canvas, "PhysioTrack FaceQuality", 2 * margin + width, 37, title_size)
    draw_text(canvas, f"{input_name} | raw detections: {raw_count} | displayed (conf >= {DISPLAY_CONFIDENCE:.2f}): {displayed_count}",
              margin, height + header + 30, max(0.33, min(0.64, width / 1400.0)))
    return canvas


def analyze_image(frame: np.ndarray, pipeline: FaceAnalysis,
                  input_name: str, input_hash: str, output_name: str
                  ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    result = pipeline.predict(frame)
    faces = list(result.instances)
    height, width = frame.shape[:2]
    annotated = frame.copy()
    observations: list[dict[str, Any]] = []
    displayed = available = unavailable = invalid = 0

    for face_index, face in enumerate(faces):
        box = face.box
        confidence = valid_confidence(face.confidence)
        if box is None:
            # Preserve a real unavailable result rather than inventing features.
            coords = None
        else:
            coords_arr = np.asarray(box, dtype=float).reshape(-1)
            coords = coords_arr.tolist() if coords_arr.size == 4 and np.isfinite(coords_arr).all() else None
        feature = face.face_features.get("quality", {}) if isinstance(face.face_features, dict) else {}
        if not isinstance(feature, dict):
            raise DemoError("Unexpected FaceAnalysis quality feature schema")
        is_available = bool(feature.get("available", False))
        should_display = confidence is not None and confidence >= DISPLAY_CONFIDENCE and coords is not None
        expected = independent_descriptors(frame, coords) if coords is not None else None
        row: dict[str, Any] = {
            "input": input_name, "source_sha256": input_hash, "output_image": output_name,
            "image_width": width, "image_height": height, "face_index": face_index,
            "status": "QUALITY_UNAVAILABLE", "displayed": should_display,
            "detector_confidence": "" if confidence is None else confidence,
            "quality_confidence": "", "box_x1": "", "box_y1": "", "box_x2": "", "box_y2": "",
            "brightness": "", "sharpness": "", "face_area_ratio": "",
            "expected_brightness": "", "expected_sharpness": "", "expected_face_area_ratio": "",
            "brightness_absolute_error": "", "sharpness_absolute_error": "",
            "face_area_ratio_absolute_error": "", "confidence_absolute_error": "",
        }
        if coords is not None:
            for field, value in zip(("box_x1", "box_y1", "box_x2", "box_y2"), coords):
                row[field] = value
        if is_available:
            if expected is None:
                raise DemoError("FaceQuality reports data for an invalid/empty clipped face crop")
            required = ("confidence", "brightness", "sharpness", "face_area_ratio")
            if not all(key in feature for key in required):
                raise DemoError("FaceQuality available feature omits required descriptors")
            quality_conf = valid_confidence(feature["confidence"])
            if (confidence is None) != (quality_conf is None):
                raise DemoError("FaceQuality did not preserve confidence availability")
            errors: dict[str, float] = {}
            for name in ("brightness", "sharpness", "face_area_ratio"):
                val = float(feature[name])
                if not math.isfinite(val):
                    raise DemoError(f"Non-finite FaceQuality {name}")
                if name == "brightness" and not 0.0 <= val <= 1.0:
                    raise DemoError("Brightness must lie within [0,1]")
                if name == "sharpness" and val < 0.0:
                    raise DemoError("Sharpness must be non-negative")
                if name == "face_area_ratio" and not 0.0 < val <= 1.0:
                    raise DemoError("Face-area ratio must lie inside (0,1]")
                row[name] = val
                row["expected_" + name] = expected[name]
                errors[name] = abs(val - expected[name])
                row[name + "_absolute_error"] = errors[name]
                if errors[name] > NUMERIC_TOLERANCE:
                    raise DemoError(f"FaceQuality {name} disagrees with independently computed formula")
            confidence_error = (abs(confidence - quality_conf) if confidence is not None else 0.0)
            if confidence_error > NUMERIC_TOLERANCE:
                raise DemoError("FaceQuality confidence differs from detector confidence")
            row["confidence_absolute_error"] = confidence_error
            row["quality_confidence"] = "" if quality_conf is None else quality_conf
            row["status"] = "QUALITY_AVAILABLE"
            available += 1
        else:
            unavailable += 1
            if coords is None or expected is None:
                invalid += 1
        if should_display:
            x1, y1, x2, y2 = coords
            left, top = max(0, min(width - 1, int(round(x1)))), max(0, min(height - 1, int(round(y1))))
            right, bottom = max(0, min(width - 1, int(round(x2)))), max(0, min(height - 1, int(round(y2))))
            if right > left and bottom > top:
                color = (38, 220, 80) if is_available else (5, 135, 240)
                thickness = max(1, int(round(min(width, height) / 480)))
                cv2.rectangle(annotated, (left, top), (right, bottom), color, thickness)
                # Put labels in a dark block next to the face, not into the crop used for measurements.
                font = max(0.36, min(0.62, min(width, height) / 1500.0))
                if is_available:
                    labels = [f"F{face_index + 1} conf={confidence:.2f}",
                              f"B={row['brightness']:.3f} S={row['sharpness']:.1f}",
                              f"A={row['face_area_ratio']:.4f}"]
                else:
                    labels = [f"F{face_index + 1} conf={confidence:.2f}", "quality unavailable"]
                text_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][0]
                                 for label in labels)
                text_line_h = max(17, int(27 * font / 0.5))
                block_w = min(width, text_width + 12)
                block_h = len(labels) * text_line_h + 9
                label_x = max(0, min(left, width - block_w))
                label_y = top - block_h - 3 if top - block_h >= 0 else min(height - block_h, bottom + 4)
                label_y = max(0, label_y)
                cv2.rectangle(annotated, (label_x, label_y),
                              (min(width - 1, label_x + block_w), min(height - 1, label_y + block_h)),
                              (24, 24, 24), -1)
                for j, label in enumerate(labels):
                    draw_text(annotated, label, label_x + 5, label_y + (j + 1) * text_line_h,
                              font, (255, 255, 255))
                displayed += 1
            else:
                row["displayed"] = False
        observations.append(row)

    if displayed == 0:
        if not faces:
            label = "No faces detected"
        else:
            label = f"No face above display threshold ({DISPLAY_CONFIDENCE:.2f})"
        size = max(0.45, min(0.9, width / 1100))
        draw_text(annotated, label, 16, min(40, height - 4), size, (140, 220, 255))
    return make_panel(frame, annotated, input_name, len(faces), displayed), observations, {
        "raw_detections": len(faces), "quality_available": available,
        "quality_unavailable": unavailable, "invalid_crops": invalid,
        "displayed_detections": displayed,
        "suppressed_by_display_threshold_or_invalid": len(faces) - displayed,
    }


def fit_tile(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    ratio = min(max_width / img.shape[1], max_height / img.shape[0])
    new_w = max(1, int(round(img.shape[1] * ratio)))
    new_h = max(1, int(round(img.shape[0] * ratio)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sheet_image(stage: Path, records: list[dict[str, Any]]) -> np.ndarray:
    count = len(records)
    rows = (count + SHEET_COLS - 1) // SHEET_COLS
    canvas = np.full((rows * SHEET_TILE_H, SHEET_COLS * SHEET_TILE_W, 3),
                     245, dtype=np.uint8)
    for i, item in enumerate(records):
        png = read_image(stage / item["output_image"])
        thumb = fit_tile(png, SHEET_TILE_W - 20, SHEET_TILE_H - 65)
        x = (i % SHEET_COLS) * SHEET_TILE_W
        y = (i // SHEET_COLS) * SHEET_TILE_H
        th, tw = thumb.shape[:2]
        canvas[y + 9:y + 9 + th, x + (SHEET_TILE_W - tw) // 2:x + (SHEET_TILE_W + tw) // 2] = thumb
        description = f"{item['input']} | faces: {item['raw_detections']} | with quality: {item['quality_available']}"
        cv2.putText(canvas, description[:70], (x + 12, y + SHEET_TILE_H - 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Synthetic qualitative illustration (no GT quality labels)",
                    (x + 12, y + SHEET_TILE_H - 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (75, 75, 75), 1, cv2.LINE_AA)
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary_text(summary: dict[str, Any]) -> str:
    # Structured complete mirror: JSON/TXT can be independently compared.
    return "PhysioTrack FaceQuality Synthetic Qualitative Demonstration\n\n" + \
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def write_reports(stage: Path, summary: dict[str, Any]) -> None:
    data = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    (stage / "synthetic_demo_summary.json").write_text(data, encoding="utf-8")
    (stage / "synthetic_demo_summary.txt").write_text(summary_text(summary), encoding="utf-8")


def verify_outputs(stage: Path, summary: dict[str, Any]) -> None:
    """Audit staged files, CSV, captions, panel pixels, hash manifest, and summary pair."""
    if not stage.is_dir() or stage.is_symlink():
        raise DemoError("Staged output path is not a safe directory")
    if json.loads((stage / "synthetic_demo_summary.json").read_text(encoding="utf-8")) != summary:
        raise DemoError("JSON summary does not match run summary")
    if (stage / "synthetic_demo_summary.txt").read_text(encoding="utf-8") != summary_text(summary):
        raise DemoError("TXT summary does not match JSON")
    if summary["status"] != "COMPLETE" or summary["files_failed"]:
        raise DemoError("An incomplete/failed execution cannot replace previous accepted results")
    records = summary["images"]
    if len(records) != summary["files_discovered"] or summary["files_processed"] != len(records):
        raise DemoError("Source-file accounting mismatch")
    with (stage / "face_quality_observations.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise DemoError("CSV column schema mismatch")
        rows = list(reader)
    if len(rows) != summary["face_observations"]:
        raise DemoError("Face-observation row count mismatch")
    if sum(r["raw_detections"] for r in records) != len(rows):
        raise DemoError("Rows do not account for all raw model detections")
    if len({r["input"] for r in records}) != len(records):
        raise DemoError("Duplicate input record")
    if len({r["output_image"] for r in records}) != len(records):
        raise DemoError("Output-name collision")
    valid_inputs = {r["input"]: r for r in records}
    counts = {name: {"raw": 0, "available": 0, "unavailable": 0, "displayed": 0} for name in valid_inputs}
    indices = set()
    for r in rows:
        name = r["input"]
        if name not in valid_inputs:
            raise DemoError("CSV refers to an unknown input")
        rec = valid_inputs[name]
        if r["source_sha256"] != rec["source_sha256"] or r["output_image"] != rec["output_image"]:
            raise DemoError("CSV source/output provenance mismatch")
        if int(r["image_width"]) != rec["image_width"] or int(r["image_height"]) != rec["image_height"]:
            raise DemoError("CSV image size mismatch")
        ix = int(r["face_index"])
        if (name, ix) in indices or ix < 0:
            raise DemoError("Duplicate/negative per-image face index")
        indices.add((name, ix))
        counts[name]["raw"] += 1
        status = r["status"]
        if status not in {"QUALITY_AVAILABLE", "QUALITY_UNAVAILABLE"}:
            raise DemoError("Unknown FaceQuality execution status")
        if status == "QUALITY_AVAILABLE":
            counts[name]["available"] += 1
            for field in ("brightness", "sharpness", "face_area_ratio", "expected_brightness",
                          "expected_sharpness", "expected_face_area_ratio"):
                if not math.isfinite(float(r[field])):
                    raise DemoError("Non-finite CSV descriptor")
            if not 0.0 <= float(r["brightness"]) <= 1.0 or float(r["sharpness"]) < 0:
                raise DemoError("CSV descriptor outside physical range")
            if not 0.0 < float(r["face_area_ratio"]) <= 1.0:
                raise DemoError("CSV face-area-ratio outside range")
            for descriptor in ("brightness", "sharpness", "face_area_ratio"):
                e = abs(float(r[descriptor]) - float(r["expected_" + descriptor]))
                if abs(e - float(r[descriptor + "_absolute_error"])) > NUMERIC_TOLERANCE or e > NUMERIC_TOLERANCE:
                    raise DemoError("CSV formula discrepancy")
            conf = r["detector_confidence"]
            qc = r["quality_confidence"]
            if (conf == "") != (qc == ""):
                raise DemoError("CSV detector/quality confidence presence mismatch")
            err = abs(float(conf) - float(qc)) if conf != "" else 0.0
            if abs(err - float(r["confidence_absolute_error"])) > NUMERIC_TOLERANCE or err > NUMERIC_TOLERANCE:
                raise DemoError("CSV confidence passthrough mismatch")
        else:
            counts[name]["unavailable"] += 1
            if any(r[key] != "" for key in ("brightness", "sharpness", "face_area_ratio")):
                raise DemoError("Unavailable quality must not be fabricated")
        if r["displayed"] not in ("True", "False"):
            raise DemoError("Malformed display flag")
        if r["displayed"] == "True":
            counts[name]["displayed"] += 1
            if r["detector_confidence"] == "" or float(r["detector_confidence"]) < DISPLAY_CONFIDENCE:
                raise DemoError("CSV display flag violates confidence threshold")
    for rec in records:
        file = stage / rec["output_image"]
        if not file.is_file():
            raise DemoError("Missing staged annotated PNG")
        if sha256_file(file) != rec["output_sha256"]:
            raise DemoError("PNG output SHA-256 mismatch")
        panel = read_image(file)
        if (panel.shape[1] != 2 * rec["image_width"] + 42
                or panel.shape[0] != rec["image_height"] + 130):
            raise DemoError("Unexpected annotated panel dimensions")
        source = INPUT_DIR / rec["input"]
        if not source.is_file() or sha256_file(source) != rec["source_sha256"]:
            raise DemoError("Synthetic source image changed during execution")
        # The original panel must be pixel-identical to the source image.
        orig = read_image(source)
        recovered = panel[72:72 + rec["image_height"], 14:14 + rec["image_width"]]
        if not np.array_equal(orig, recovered):
            raise DemoError("Left panel no longer matches unmodified synthetic source")
        stats = counts[rec["input"]]
        if sorted(ix for name, ix in indices if name == rec["input"]) != list(range(stats["raw"])):
            raise DemoError("Non-contiguous face indexes")
        for key, actual in (("raw_detections", stats["raw"]), ("quality_available", stats["available"]),
                            ("quality_unavailable", stats["unavailable"]),
                            ("displayed_detections", stats["displayed"])):
            if rec[key] != actual:
                raise DemoError(f"Per-image count mismatch: {key}")
    totals = summary["totals"]
    for key in ("raw_detections", "quality_available", "quality_unavailable", "displayed_detections"):
        if totals[key] != sum(rec[key] for rec in records):
            raise DemoError("Summary total mismatch: " + key)
    if sha256_file(stage / "face_quality_observations.csv") != summary["artifact_sha256"]["face_quality_observations.csv"]:
        raise DemoError("CSV hash mismatch")
    sheet_path = stage / "synthetic_demo_examples.png"
    if sha256_file(sheet_path) != summary["artifact_sha256"]["synthetic_demo_examples.png"]:
        raise DemoError("Contact-sheet hash mismatch")
    if not np.array_equal(read_image(sheet_path), sheet_image(stage, records)):
        raise DemoError("Contact-sheet pixels do not match the validated individual panels")
    expected_files = {"synthetic_demo_summary.json", "synthetic_demo_summary.txt",
                      "face_quality_observations.csv", "synthetic_demo_examples.png"}
    expected_files.update(r["output_image"] for r in records)
    actual_files = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise DemoError("Unexpected or missing published file: " + str(actual_files ^ expected_files))


def publish(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    verify_outputs(stage, summary)
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise DemoError("Unsafe prior synthetic_demo directory (symlink or file)")
    backup = staging_root / "previous_accepted_synthetic_demo"
    previous = FINAL_DIR.exists()
    published = False
    try:
        if previous:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        published = True
        verify_outputs(FINAL_DIR, summary)
    except BaseException:
        try:
            if published and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if previous and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as recover_exc:
            raise DemoError(
                f"Output publication failed and manual recovery is needed from {backup}"
            ) from recover_exc
        raise
    if previous:
        shutil.rmtree(backup)


def provenance() -> dict[str, Any]:
    relevant_paths = [Path(__file__).resolve()]
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig, FaceQuality):
        relevant_paths.append(Path(inspect.getfile(cls)).resolve())
    files = {}
    for path in relevant_paths:
        key = path.relative_to(REPOSITORY_ROOT).as_posix() if path.is_relative_to(REPOSITORY_ROOT) else path.name
        files[key] = sha256_file(path)
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "ultralytics": version("ultralytics"),
        "physiotrack": version("physiotrack"),
        "file_sha256": files,
    }


def run() -> None:
    component_paths = project_imports()
    images = discover_images()
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise DemoError("Unsafe final synthetic_demo path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".face_quality_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    (stage / "annotated_images").mkdir()
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    pipeline: FaceAnalysis | None = None
    failure_messages: list[str] = []
    try:
        print("PhysioTrack Face Quality Synthetic Qualitative Demonstration")
        print("Inputs: validation/synthetic_media/face_quality_media/")
        print("Outputs: validation/face_quality/results/synthetic_demo/")
        print("Discovered synthetic images:", len(images))
        pipeline = make_pipeline()
        for i, source in enumerate(images, start=1):
            rel = source.relative_to(INPUT_DIR).as_posix()
            out_rel = relative_output(source)
            print(f"[{i}/{len(images)}] {rel}", flush=True)
            try:
                original = read_image(source)
                source_hash = sha256_file(source)
                panel, observations, counts = analyze_image(original, pipeline, rel, source_hash, out_rel)
                output_file = stage / out_rel
                save_png(output_file, panel)
                panel_reopened = read_image(output_file)
                if not np.array_equal(panel_reopened, panel):
                    raise DemoError("Written panel pixels differ from their source")
                row = {
                    "input": rel, "source_sha256": source_hash,
                    "output_image": out_rel, "output_sha256": sha256_file(output_file),
                    "image_width": original.shape[1], "image_height": original.shape[0],
                    "status": "SUCCESS", **counts,
                }
                records.append(row)
                rows.extend(observations)
                print(f"    faces={counts['raw_detections']}, quality={counts['quality_available']}, "
                      f"displayed={counts['displayed_detections']}", flush=True)
            except Exception as exc:
                message = f"{rel}: {type(exc).__name__}: {exc}"
                print("    FAILED:", message, flush=True)
                failure_messages.append(message)
        if failure_messages:
            raise DemoError(f"{len(failure_messages)} image(s) failed: " + "; ".join(failure_messages[:4]))
        write_csv(stage / "face_quality_observations.csv", rows)
        save_png(stage / "synthetic_demo_examples.png", sheet_image(stage, records))
        totals = {key: sum(rec[key] for rec in records) for key in (
            "raw_detections", "quality_available", "quality_unavailable",
            "displayed_detections", "suppressed_by_display_threshold_or_invalid", "invalid_crops",
        )}
        summary: dict[str, Any] = {
            "project": "physiotrack-v2", "component": "FaceQuality",
            "type": "synthetic_qualitative_demonstration",
            "scientific_benchmark": False,
            "uses_ground_truth_quality_labels": False,
            "calculates_quality_accuracy_metrics": False,
            "source_directory": "validation/synthetic_media/face_quality_media",
            "output_directory": "validation/face_quality/results/synthetic_demo",
            "configuration": {
                "pipeline": "Face detector -> FaceAnalysis -> FaceQuality",
                "device": DEVICE, "detector_confidence": DETECTOR_CONFIDENCE,
                "detector_iou": NMS_IOU, "maximum_detections": MAX_DETECTIONS,
                "display_confidence_threshold": DISPLAY_CONFIDENCE,
                "tracking_enabled": False, "unrelated_components_disabled": True,
                "descriptor_definitions": {
                    "brightness": "mean grayscale pixel intensity of integer-clipped face crop / 255",
                    "sharpness": "population variance of grayscale Laplacian of face crop",
                    "face_area_ratio": "integer-clipped face-box area / full-image pixel area",
                    "quality_confidence": "passed through from face detector (NOT quality accuracy)",
                },
                "numeric_check_tolerance": NUMERIC_TOLERANCE,
            },
            "component_imports": component_paths,
            "reproducibility": provenance(),
            "files_discovered": len(images), "files_processed": len(records),
            "files_failed": 0,
            "face_observations": len(rows), "totals": totals,
            "images": records,
            "artifact_sha256": {
                "face_quality_observations.csv": sha256_file(stage / "face_quality_observations.csv"),
                "synthetic_demo_examples.png": sha256_file(stage / "synthetic_demo_examples.png"),
            },
            "interpretation": (
                "Independent qualitative examples of native PhysioTrack detector and FaceQuality "
                "outputs on generated media. Image descriptions (brightness, blur, apparent scale, "
                "occlusion, pose and no-face) are nominal conditions, not calibrated ground truth. "
                "The experiment does not measure accuracy, image-quality class correctness, "
                "or end-to-end face-detection recall. Faces below the display threshold are "
                "retained in the CSV and counts, not visualized."
            ),
            "status": "COMPLETE",
        }
        write_reports(stage, summary)
        print("Validating all staged outputs ...", flush=True)
        verify_outputs(stage, summary)
        publish(stage, staging_root, summary)
        print("Published verified outputs:", FINAL_DIR)
        print("Images:", len(records), "| detected faces:", totals["raw_detections"],
              "| quality rows:", totals["quality_available"],
              "| displayed faces:", totals["displayed_detections"])
        print("Overall synthetic demonstration status: COMPLETE")
    finally:
        if pipeline is not None:
            pipeline.close()
        # Never destroy the only remaining copy if rollback needs manual recovery.
        if not (staging_root / "previous_accepted_synthetic_demo").exists():
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("Synthetic demonstration FAILED. Existing accepted results have been preserved where possible.")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
