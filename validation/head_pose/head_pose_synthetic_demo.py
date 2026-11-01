from __future__ import annotations

"""Synthetic qualitative head-pose demonstration on locally supplied images.

Run from any working directory after installing physiotrack-v2 in the active
Python environment. Inputs are discovered below validation/synthetic_media;
only validation/head_pose/results/synthetic_demo is replaced by a successful run.
The folder contains annotated images, CSV and JSON/TXT summaries.
No ground-truth pose labels or benchmark metrics are used here.
"""

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.face_orientation import FaceOrientation
from physiotrack.results import Result


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "head_pose_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp",
    ".avif", ".jp2", ".pbm", ".pgm", ".ppm",
})

DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.001
DISPLAY_CONFIDENCE = 0.25
NMS_IOU = 0.45
MAX_DETECTIONS = 10000
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_MAX_WIDTH = 420
CONTACT_SHEET_MAX_HEIGHT = 320

OBSERVATION_FIELDS = (
    "input", "source_sha256", "output_image", "width", "height",
    "face_index", "confidence", "box_x1", "box_y1", "box_x2",
    "box_y2", "yaw", "pitch", "roll",
)


class SyntheticDemoError(RuntimeError):
    """A synthetic-demo media, rendering, or output validation failure."""


def check_project_imports() -> dict[str, str]:
    """Reject imports from any unrelated installed physiotrack package."""
    expected = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    paths: dict[str, str] = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig, FaceOrientation, Result):
        source = Path(inspect.getfile(cls)).resolve()
        if not source.is_relative_to(expected):
            raise RuntimeError(
                f"{cls.__name__} was not imported from this repository's src/physiotrack. "
                "Install the current physiotrack-v2 checkout in editable mode."
            )
        paths[cls.__name__] = source.relative_to(REPOSITORY_ROOT).as_posix()
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images() -> list[Path]:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            "Input directory does not exist: validation/synthetic_media/head_pose_media/"
        )
    images: list[Path] = []
    for path in INPUT_DIR.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    images.sort(
        key=lambda p: (
            p.relative_to(INPUT_DIR).as_posix().casefold(),
            p.relative_to(INPUT_DIR).as_posix(),
        )
    )
    if not images:
        raise RuntimeError(
            "No supported images found in validation/synthetic_media/head_pose_media/."
        )
    return images


def output_relative_path(source: Path) -> Path:
    relative = source.relative_to(INPUT_DIR)
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    safe_basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.stem)[:92].strip(". ")
    if not safe_basename:
        safe_basename = "image"
    return Path("images") / relative.parent / f"{safe_basename}_{digest}_annotated.png"


def image_from_path(path: Path) -> np.ndarray:
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error) as exc:
        raise SyntheticDemoError(f"Unable to decode image: {path}") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise SyntheticDemoError(f"Unable to decode a 3-channel BGR image: {path}")
    return image


def save_png(image: np.ndarray, path: Path) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise SyntheticDemoError("OpenCV could not encode the annotated PNG image")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def make_pipeline() -> FaceAnalysis:
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=True,
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
    )
    config.validate()

    detector = Face(
        device=DEVICE,
        conf=DETECTOR_CONFIDENCE,
        iou=NMS_IOU,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )
    pipeline = FaceAnalysis(
        detector=detector,
        config=config,
        device=DEVICE,
        verbose=False,
    )

    if not isinstance(pipeline.orientation, FaceOrientation):
        raise RuntimeError("The instantiated FaceAnalysis pipeline did not initialize FaceOrientation")

    unrelated_components = {
        "tracker": pipeline.tracker,
        "landmarks": pipeline.landmarks,
        "quality": pipeline.quality,
        "eyes": pipeline.eyes,
        "blink": pipeline.blink,
        "gaze": pipeline.gaze,
        "gaze_estimation": pipeline.gaze_estimation,
        "mouth": pipeline.mouth,
        "mouth_motion": pipeline.mouth_motion,
        "emotion": pipeline.emotion,
        "regions": pipeline.regions,
        "temporal": pipeline.temporal,
    }
    enabled_unrelated = [name for name, component in unrelated_components.items() if component is not None]
    if enabled_unrelated:
        raise RuntimeError(
            "Unexpected unrelated FaceAnalysis components were initialized: " + ", ".join(enabled_unrelated)
        )
    return pipeline


def _validated_box(face: Any, image_width: int, image_height: int) -> tuple[tuple[int, int, int, int], float | None] | None:
    try:
        coords = np.asarray(face.box, dtype=np.float64).reshape(-1)
        if coords.size != 4 or not np.all(np.isfinite(coords)):
            return None
        x1, y1, x2, y2 = (float(v) for v in coords)
        if not (x2 > x1 and y2 > y1):
            return None
        confidence = getattr(face, "confidence", None)
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                return None
        left = max(0, min(image_width - 1, int(round(x1))))
        top = max(0, min(image_height - 1, int(round(y1))))
        right = max(0, min(image_width - 1, int(round(x2))))
        bottom = max(0, min(image_height - 1, int(round(y2))))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom), confidence
    except (TypeError, ValueError, OverflowError):
        return None


def _validated_orientation(face: Any) -> tuple[float, float, float] | None:
    orientation = getattr(face, "orientation", None)
    if orientation is None:
        return None
    try:
        yaw = float(orientation["yaw"])
        pitch = float(orientation["pitch"])
        roll = float(orientation["roll"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite([yaw, pitch, roll])):
        return None
    return yaw, pitch, roll


def _draw_label_block(image: np.ndarray, anchor_x: int, anchor_y: int, lines: list[str], color: tuple[int, int, int], font_scale: float, line_width: int) -> None:
    height, width = image.shape[:2]
    sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1) for text in lines]
    text_width = max(size[0][0] for size in sizes)
    line_height = max(size[0][1] + size[1] + 4 for size in sizes)
    block_width = min(width - 1, text_width + 10)
    block_height = min(height - 1, line_height * len(lines) + 4)

    x1 = max(0, min(anchor_x, max(0, width - block_width - 1)))
    y1 = anchor_y - block_height - 2
    if y1 < 0:
        y1 = max(0, min(anchor_y + 2, max(0, height - block_height - 1)))
    x2 = min(width - 1, x1 + block_width)
    y2 = min(height - 1, y1 + block_height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

    cursor_y = y1 + 4
    for text, ((_, th), baseline) in zip(lines, sizes):
        baseline_y = min(height - 1, cursor_y + th + 1)
        cv2.putText(
            image,
            text,
            (x1 + 4, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cursor_y += th + baseline + 4


def annotate_image(image: np.ndarray, pipeline: FaceAnalysis) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise SyntheticDemoError("The decoded image is not a three-channel BGR frame")

    result = pipeline.predict(image)
    faces = list(result)
    output = image.copy()
    image_height, image_width = image.shape[:2]
    font_scale = max(0.38, min(0.72, min(image_height, image_width) / 950.0))
    line_width = 1 if min(image_height, image_width) < 450 else 2

    valid_pose_rows: list[dict[str, Any]] = []
    faces_with_pose = 0
    faces_without_pose = 0
    invalid_faces = 0
    displayed_faces = 0
    pose_axes_drawn = 0
    faces_skipped_low_confidence = 0

    for face_index, face in enumerate(faces, 1):
        boxed = _validated_box(face, image_width, image_height)
        if boxed is None:
            invalid_faces += 1
            continue
        (left, top, right, bottom), confidence = boxed
        if confidence is not None and confidence < DISPLAY_CONFIDENCE:
            faces_skipped_low_confidence += 1
            continue

        orientation = _validated_orientation(face)
        color = (0, 220, 255) if orientation is not None else (0, 140, 255)

        if orientation is not None:
            # Use PhysioTrack's official Result.plot -> 6DRepNet360 draw_axis,
            # not an independently reimplemented Euler-angle projection.
            # The native renderer draws X red, Y green, and Z blue.
            # Restrict it to the currently displayed face so low-confidence
            # detections do not generate unexpected overlays.
            output = Result(orig_img=output, instances=[face], task="face").plot(
                boxes=False, labels=False, keypoints=False, masks=False,
            )
            pose_axes_drawn += 1

        cv2.rectangle(output, (left, top), (right, bottom), color, line_width)
        displayed_faces += 1

        line1 = f"face {face_index}"
        if confidence is not None:
            line1 += f"  det {confidence:.2f}"

        if orientation is None:
            faces_without_pose += 1
            _draw_label_block(output, left, top, [line1, "pose unavailable"], color, font_scale, line_width)
            continue

        yaw, pitch, roll = orientation
        faces_with_pose += 1
        valid_pose_rows.append({
            "face_index": int(face_index),
            "confidence": float(confidence) if confidence is not None else "",
            "box_x1": int(left),
            "box_y1": int(top),
            "box_x2": int(right),
            "box_y2": int(bottom),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        })
        line2 = f"Y {yaw:+.1f}  P {pitch:+.1f}  R {roll:+.1f}"
        _draw_label_block(output, left, top, [line1, line2], color, font_scale, line_width)

    counts = {
        "faces_detected": len(faces),
        "displayed_faces": displayed_faces,
        "pose_axes_drawn": pose_axes_drawn,
        "faces_skipped_low_confidence": faces_skipped_low_confidence,
        "faces_with_pose": faces_with_pose,
        "faces_without_pose": faces_without_pose,
        "invalid_faces": invalid_faces,
    }
    return output, counts, valid_pose_rows


def process_image(source: Path, target: Path, pipeline: FaceAnalysis) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image = image_from_path(source)
    annotated, counts, observations = annotate_image(image, pipeline)
    save_png(annotated, target)
    decoded = image_from_path(target)
    if decoded.shape[:2] != image.shape[:2]:
        raise SyntheticDemoError("Annotated image dimensions differ from source")
    metrics = {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        **counts,
    }
    return metrics, observations


def write_observations_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _fit_thumbnail(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise SyntheticDemoError("Cannot build a contact sheet from an empty image")
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_contact_sheet(stage: Path, records: list[dict[str, Any]], output_name: str = "synthetic_demo_examples.png") -> None:
    successful = [record for record in records if record.get("status") == "SUCCESS"]
    if not successful:
        return
    padding = 20
    header_h = 58
    footer_h = 48
    cell_w = CONTACT_SHEET_MAX_WIDTH + 2 * padding
    cell_h = CONTACT_SHEET_MAX_HEIGHT + header_h + footer_h + 2 * padding
    columns = min(CONTACT_SHEET_COLUMNS, max(1, len(successful)))
    rows = int(math.ceil(len(successful) / columns))
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 238, dtype=np.uint8)

    for index, record in enumerate(successful):
        image = image_from_path(stage / record["output_image"])
        thumb = _fit_thumbnail(image, CONTACT_SHEET_MAX_WIDTH, CONTACT_SHEET_MAX_HEIGHT)
        row = index // columns
        col = index % columns
        x0 = col * cell_w
        y0 = row * cell_h
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (210, 210, 210), 1)

        rel_input = str(record["input"])
        stats = f"faces={record['faces_detected']} pose={record['faces_with_pose']}"
        cv2.putText(canvas, rel_input, (x0 + padding, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, stats, (x0 + padding, y0 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 70, 70), 1, cv2.LINE_AA)

        h, w = thumb.shape[:2]
        x_img = x0 + (cell_w - w) // 2
        y_img = y0 + header_h + padding
        canvas[y_img:y_img + h, x_img:x_img + w] = thumb
        output_rel = Path(record["output_image"]).name
        cv2.putText(
            canvas,
            output_rel,
            (x0 + padding, min(canvas.shape[0] - 10, y0 + cell_h - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    save_png(canvas, stage / output_name)


def execution_fingerprint() -> dict[str, Any]:
    """Record checkout sources and installed versions, without inventing model hashes."""
    sources = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig, FaceOrientation, Result):
        source = Path(inspect.getfile(cls)).resolve()
        sources[source.relative_to(REPOSITORY_ROOT).as_posix()] = sha256_file(source)
    versions = {}
    for package in ("numpy", "opencv-python", "opencv-contrib-python",
                    "torch", "ultralytics", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_sha256": dict(sorted(sources.items())),
        "package_versions": versions,
        "note": ("Model weights and runtime can change numerical predictions. "
                 "Input and implementation hashes aid reproducibility but do not "
                 "guarantee identical results on another machine."),
    }


def summary_as_text(summary: dict[str, Any]) -> str:
    """Readable report derived from the same in-memory data as the JSON."""
    config = summary["configuration"]
    lines = [
        "PhysioTrack Head Pose - Synthetic Qualitative Demonstration",
        "=" * 56, "",
        "Status: " + summary["status"],
        "Evaluation type: qualitative only (not AFLW benchmark)",
        "Source: " + summary["input_directory"],
        "Output: " + summary["output_directory"], "",
        "Discovered images: " + str(summary["discovered_files"]),
        "Successfully processed images: " + str(summary["successful_files"]),
        "Failed images: " + str(summary["failed_files"]),
        "Faces detected: " + str(summary["faces_detected"]),
        "Displayed faces: " + str(summary["displayed_faces"]),
        "Faces with pose: " + str(summary["faces_with_pose"]),
        "Faces without pose: " + str(summary["faces_without_pose"]),
        "Pose axes drawn: " + str(summary["pose_axes_drawn"]),
        "Detections below display threshold: " + str(summary["faces_skipped_low_confidence"]),
        "Invalid face results: " + str(summary["invalid_faces"]),
        "Pose observation CSV rows: " + str(summary["observation_rows"]), "",
        "Configuration", "-------------",
        "Device: " + str(config["device"]),
        "Detector confidence threshold: " + str(config["detector_confidence"]),
        "Display confidence threshold: " + str(config["display_confidence_threshold"]),
        "Detector IoU: " + str(config["detector_iou"]),
        "Maximum detections: " + str(config["max_detections"]),
        "Ground truth: not used",
        "Benchmark accuracy metrics: not calculated", "",
        "Images", "------",
    ]
    for rec in summary["records"]:
        lines.append(
            f"{rec['input']}: {rec['status']}"
            + (f"; faces={rec['faces_detected']}; pose={rec['faces_with_pose']}"
               if rec["status"] == "SUCCESS" else f"; {rec.get('error', 'processing failure')}")
        )
        if rec.get("source_sha256"):
            lines.append("  Input SHA-256: " + rec["source_sha256"])
        if rec["status"] == "SUCCESS":
            lines.append("  Annotated image: " + rec["output_image"])
    if summary["status"] == "COMPLETE":
        lines.extend(["", "Published files", "---------------",
            "images/", "head_pose_observations.csv",
            "synthetic_demo_examples.png", "synthetic_demo_summary.json",
            "synthetic_demo_summary.txt",
        ])
    else:
        lines.extend(["", "The failed attempt was not published; any existing accepted results were preserved."])
    lines.extend(["",
        "Interpretation: visual demonstrations and model-predicted angles, not ground-truth pose accuracy.",
        "Reproduction: use identical media, implementation, dependencies and model weights.",
    ])
    return "\n".join(lines) + "\n"


def write_reports(folder: Path, summary: dict[str, Any], prefix: str) -> None:
    with (folder / f"{prefix}.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    (folder / f"{prefix}.txt").write_text(summary_as_text(summary), encoding="utf-8", newline="\n")


def verify_staged_outputs(stage: Path, summary: dict[str, Any]) -> None:
    """Verify exactly one complete, internally consistent set of owned artifacts."""
    if not stage.is_dir() or stage.is_symlink():
        raise SyntheticDemoError("Output directory is missing or unsafe")
    records = summary.get("records", [])
    if (summary.get("status") != "COMPLETE" or summary.get("failed_files") != 0
            or not records or len(records) != summary.get("discovered_files")
            or len(records) != summary.get("successful_files")
            or any(rec.get("status") != "SUCCESS" for rec in records)):
        raise SyntheticDemoError("Only a complete and successful run may be published")

    images_dir = stage / "images"
    if not images_dir.is_dir() or images_dir.is_symlink():
        raise SyntheticDemoError("Annotated images directory is missing or unsafe")
    required_files = ("head_pose_observations.csv", "synthetic_demo_examples.png",
                      "synthetic_demo_summary.json", "synthetic_demo_summary.txt")
    for name in required_files:
        p = stage / name
        if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:
            raise SyntheticDemoError(f"Missing or empty required artifact: {name}")

    expected_images: set[str] = set()
    by_input: dict[str, dict[str, Any]] = {}
    for rec in records:
        rel = rec["output_image"]
        if (rec["input"] in by_input or rel in expected_images
                or not rel.startswith("images/")):
            raise SyntheticDemoError("Duplicate/invalid input or output image")
        by_input[rec["input"]] = rec
        expected_images.add(rel)
        path = stage / rel
        if (not path.resolve().is_relative_to(stage.resolve())
                or not path.is_file() or path.is_symlink() or path.stat().st_size == 0):
            raise SyntheticDemoError(f"Missing/unsafe annotated image: {rel}")
        img = image_from_path(path)
        if img.shape[:2] != (rec["height"], rec["width"]):
            raise SyntheticDemoError(f"Annotated image dimensions disagree for: {rec['input']}")
        original = INPUT_DIR / rec["input"]
        if (not original.is_file() or original.is_symlink()
                or sha256_file(original) != rec["source_sha256"]):
            raise SyntheticDemoError("An input was altered during execution")
        if not (rec["faces_detected"] ==
                rec["displayed_faces"] + rec["faces_skipped_low_confidence"] + rec["invalid_faces"]
                and rec["displayed_faces"] == rec["faces_with_pose"] + rec["faces_without_pose"]
                and rec["pose_axes_drawn"] == rec["faces_with_pose"]):
            raise SyntheticDemoError("Per-image face accounting is inconsistent")

    expected_files = expected_images | set(required_files)
    actual_files: set[str] = set()
    for p in stage.rglob("*"):
        if p.is_symlink():
            raise SyntheticDemoError("Symlinks are not allowed in staged outputs")
        if p.is_file():
            actual_files.add(p.relative_to(stage).as_posix())
    if actual_files != expected_files:
        raise SyntheticDemoError("Unexpected/missing files inside synthetic_demo")

    csv_counts = {name: 0 for name in by_input}
    seen_faces: set[tuple[str, int]] = set()
    total_rows = 0
    with (stage / "head_pose_observations.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OBSERVATION_FIELDS:
            raise SyntheticDemoError("Pose observations CSV schema mismatch")
        for row in reader:
            rec = by_input.get(row["input"])
            if rec is None:
                raise SyntheticDemoError("CSV contains an unknown input")
            for name in ("source_sha256", "output_image", "width", "height"):
                if str(row[name]) != str(rec[name]):
                    raise SyntheticDemoError(f"CSV {name} does not match the image summary")
            try:
                index = int(row["face_index"])
                box = [int(row[x]) for x in ("box_x1", "box_y1", "box_x2", "box_y2")]
                angles = [float(row[x]) for x in ("yaw", "pitch", "roll")]
                confidence = float(row["confidence"]) if row["confidence"] else None
            except (ValueError, TypeError) as exc:
                raise SyntheticDemoError("Invalid numerical pose observation") from exc
            if (not 1 <= index <= rec["faces_detected"]
                    or (row["input"], index) in seen_faces
                    or not (0 <= box[0] < box[2] < rec["width"]
                            and 0 <= box[1] < box[3] < rec["height"])
                    or not all(math.isfinite(v) for v in angles)
                    or (confidence is not None and not (0 <= confidence <= 1))):
                raise SyntheticDemoError("Invalid/duplicate face or pose coordinates in CSV")
            seen_faces.add((row["input"], index))
            csv_counts[row["input"]] += 1
            total_rows += 1
    if total_rows != summary["observation_rows"]:
        raise SyntheticDemoError("Total pose observation rows do not match summary")
    for name, rec in by_input.items():
        if csv_counts[name] != rec["faces_with_pose"]:
            raise SyntheticDemoError(f"Per-image pose observation count mismatch: {name}")

    for metric in ("faces_detected", "displayed_faces", "faces_with_pose",
                   "faces_without_pose", "pose_axes_drawn",
                   "faces_skipped_low_confidence", "invalid_faces"):
        if sum(int(rec[metric]) for rec in records) != summary[metric]:
            raise SyntheticDemoError(f"Aggregate {metric} disagrees with images")
    sheet = image_from_path(stage / "synthetic_demo_examples.png")
    if sheet.shape[0] <= 0 or sheet.shape[1] <= 0:
        raise SyntheticDemoError("Invalid annotated contact sheet")
    with (stage / "synthetic_demo_summary.json").open("r", encoding="utf-8") as handle:
        if json.load(handle) != summary:
            raise SyntheticDemoError("JSON summary differs from the recorded results")
    if (stage / "synthetic_demo_summary.txt").read_text(encoding="utf-8") != summary_as_text(summary):
        raise SyntheticDemoError("TXT summary differs from the recorded results")


def commit_staged_directory(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    """Replace synthetic_demo only, verify after publish, restore old data on failure."""
    backup = staging_root / "previous_synthetic_demo"
    if backup.exists():
        raise SyntheticDemoError("Unexpected backup in new staging directory")
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise SyntheticDemoError("Final synthetic_demo is not a normal directory")
    verify_staged_outputs(stage, summary)
    had_previous = FINAL_DIR.exists()
    published = False
    try:
        if had_previous:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        published = True
        verify_staged_outputs(FINAL_DIR, summary)
    except BaseException:
        try:
            if published and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if had_previous and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as restore_error:
            # A backup that cannot be restored MUST survive staging cleanup.
            raise SyntheticDemoError(
                f"Publication failed; old results require manual recovery from {backup}"
            ) from restore_error
        raise
    if had_previous:
        shutil.rmtree(backup)


def main() -> None:
    component_paths = check_project_imports()
    images = discover_images()
    if FINAL_DIR.exists() and (FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()):
        raise SyntheticDemoError("Unsafe final output path")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    staging_root = Path(tempfile.mkdtemp(prefix=".head_pose_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    (stage / "images").mkdir()

    records: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    pipeline = None

    try:
        provenance = execution_fingerprint()
        pipeline = make_pipeline()
        print("PhysioTrack head pose synthetic qualitative demonstration")
        print("Images discovered:", len(images))
        print("Inputs: validation/synthetic_media/head_pose_media/")
        print("Outputs: validation/head_pose/results/synthetic_demo/")

        outputs_seen: set[str] = set()
        for ordinal, source in enumerate(images, 1):
            relative = source.relative_to(INPUT_DIR).as_posix()
            output_image = output_relative_path(source)
            key = output_image.as_posix().casefold()
            if key in outputs_seen:
                raise SyntheticDemoError("Output-name collision detected")
            outputs_seen.add(key)

            print(f"[{ordinal}/{len(images)}] image: {relative}", flush=True)
            record: dict[str, Any] = {
                "input": relative,
                "status": "FAILED",
                "output_image": output_image.as_posix(),
            }
            try:
                source_digest = sha256_file(source)
                measurements, pose_rows = process_image(source, stage / output_image, pipeline)
                record["source_sha256"] = source_digest
                record.update(measurements)
                record["status"] = "SUCCESS"

                for pose_row in pose_rows:
                    observation_rows.append({
                        "input": relative,
                        "source_sha256": source_digest,
                        "output_image": output_image.as_posix(),
                        "width": record["width"],
                        "height": record["height"],
                        **pose_row,
                    })

                print(
                    f"    OK: faces={measurements['faces_detected']}; "
                    f"faces_with_pose={measurements['faces_with_pose']}; "
                    f"faces_without_pose={measurements['faces_without_pose']}"
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {record['error']}")
            records.append(record)

        create_contact_sheet(stage, records)
        write_observations_csv(stage / "head_pose_observations.csv", observation_rows)

        successful_records = [r for r in records if r["status"] == "SUCCESS"]
        failed_records = [r for r in records if r["status"] != "SUCCESS"]

        summary = {
            "type": "synthetic_qualitative_head_pose_demonstration",
            "benchmark_evaluation": False,
            "project": "physiotrack-v2",
            "input_directory": "validation/synthetic_media/head_pose_media",
            "output_directory": "validation/head_pose/results/synthetic_demo",
            "component_imports": component_paths,
            "reproducibility": provenance,
            "configuration": {
                "detector": "physiotrack.face.Face",
                "pipeline": "physiotrack.face.FaceAnalysis",
                "head_pose_component": "physiotrack.face.face_orientation.FaceOrientation",
                "head_pose_enabled": True,
                "tracking_enabled": False,
                "detector_confidence": DETECTOR_CONFIDENCE,
                "display_confidence_threshold": DISPLAY_CONFIDENCE,
                "detector_iou": NMS_IOU,
                "max_detections": MAX_DETECTIONS,
                "device": DEVICE,
                "ground_truth_used": False,
                "benchmark_metrics_calculated": False,
            },
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "platform": platform.system(),
            },
            "note": (
                "Synthetic qualitative demonstration only. The supplied images are synthetic/non-benchmark media. "
                "No ground-truth head-pose labels, AFLW evaluation targets, or quantitative benchmark metrics are "
                "used by this script."
            ),
            "status": "COMPLETE" if not failed_records else "REJECTED_WITH_FILE_FAILURES",
            "discovered_files": len(images),
            "successful_files": len(successful_records),
            "failed_files": len(failed_records),
            "images_processed": len(successful_records),
            "faces_detected": int(sum(r.get("faces_detected", 0) for r in successful_records)),
            "displayed_faces": int(sum(r.get("displayed_faces", 0) for r in successful_records)),
            "faces_with_pose": int(sum(r.get("faces_with_pose", 0) for r in successful_records)),
            "pose_axes_drawn": int(sum(r.get("pose_axes_drawn", 0) for r in successful_records)),
            "faces_skipped_low_confidence": int(sum(r.get("faces_skipped_low_confidence", 0) for r in successful_records)),
            "faces_without_pose": int(sum(r.get("faces_without_pose", 0) for r in successful_records)),
            "invalid_faces": int(sum(r.get("invalid_faces", 0) for r in successful_records)),
            "observation_rows": len(observation_rows),
            "records": records,
        }

        write_reports(stage, summary, "synthetic_demo_summary")

        if failed_records:
            # Failed attempts never replace the previous published demonstration.
            write_reports(RESULTS_DIR, summary, "synthetic_demo_last_attempt")
            raise SyntheticDemoError(
                f"{len(failed_records)} image(s) failed. Existing synthetic_demo was preserved. "
                "See results/synthetic_demo_last_attempt.json and .txt."
            )
        verify_staged_outputs(stage, summary)
        commit_staged_directory(stage, staging_root, summary)

        # A successful replacement supersedes the last failed-attempt diagnostics.
        for ext in ("json", "txt"):
            try:
                (RESULTS_DIR / f"synthetic_demo_last_attempt.{ext}").unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARNING: could not remove old failed-attempt report: {exc}")

        print("\nFinished.")
        print(f"Images processed: {summary['images_processed']}")
        print(f"Faces detected: {summary['faces_detected']}")
        print(f"Faces with pose: {summary['faces_with_pose']}")
        print(f"Pose coordinate triads drawn: {summary['pose_axes_drawn']}")
        print(f"Detections below display threshold: {summary['faces_skipped_low_confidence']}")
        print(f"Faces without pose: {summary['faces_without_pose']}")
        print(f"Pose observations: {summary['observation_rows']}")
        print(f"Failed files: {summary['failed_files']}")
        print(f"Overall status: {summary['status']}")
        print("Saved summaries: validation/head_pose/results/synthetic_demo/synthetic_demo_summary.{json,txt}")

    finally:
        try:
            if pipeline is not None and hasattr(pipeline, "close"):
                pipeline.close()
        finally:
            # In the exceptional case restoration failed, preserve the old backup.
            if (staging_root / "previous_synthetic_demo").exists():
                print("WARNING: previous results backup preserved at",
                      staging_root / "previous_synthetic_demo")
            else:
                shutil.rmtree(staging_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
