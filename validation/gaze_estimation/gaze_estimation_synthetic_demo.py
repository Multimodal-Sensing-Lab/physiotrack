from __future__ import annotations

"""Synthetic qualitative gaze-estimation demonstration on locally supplied images.

Run from any working directory after installing physiotrack-v2 in the active
Python environment. Inputs are discovered below validation/synthetic_media;
only validation/gaze_estimation/results/synthetic_demo is replaced by a
successful run. The folder contains annotated images, a CSV manifest, and JSON/
TXT summaries. No ground-truth gaze labels or benchmark metrics are used here.
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
import yaml

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.gaze_estimation import GazeEstimator


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "gaze_estimation_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp",
    ".avif", ".jp2", ".pbm", ".pgm", ".ppm",
})

DEVICE = "cpu"
MODEL_MODE = "eth-xgaze"
MIN_IOU = 0.10
DETECTOR_CONFIDENCE = 0.001
DISPLAY_CONFIDENCE = 0.25
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_MAX_WIDTH = 420
CONTACT_SHEET_MAX_HEIGHT = 320
DISPLAY_MAX_ABS_ANGLE_DEG = 35.0
DISPLAY_AXIS_SCALE_RATIO = 0.18
DISPLAY_MAX_VECTOR_RATIO = 0.20
DISPLAY_MIN_VECTOR_PX = 10.0
DISPLAY_CLIP_MARGIN_PX = 14
DISPLAY_REGION_MAX_SIDE = 320
DISPLAY_REGION_NONBLACK_THRESHOLD = 15
EXPECTED_CHECKPOINT_FILENAME = "model.safetensors"
EXPECTED_CHECKPOINT_SHA256 = (
    "d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc"
)

OBSERVATION_FIELDS = (
    "input", "source_sha256", "output_image", "width", "height",
    "face_index", "confidence", "box_x1", "box_y1", "box_x2", "box_y2",
    "origin_x", "origin_y", "endpoint_x", "endpoint_y",
    "gaze_available", "gaze_x", "gaze_y", "gaze_z", "gaze_vector_norm",
    "pitch_deg", "yaw_deg", "association_iou", "status", "failure_reason",
)


class SyntheticDemoError(RuntimeError):
    """Synthetic-demo media, rendering, or output validation failure."""


def check_project_imports() -> dict[str, str]:
    """Reject imports from any unrelated installed physiotrack package."""
    expected = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    paths: dict[str, str] = {}
    for obj in (FaceAnalysis, FaceAnalysisConfig, GazeEstimator):
        source = Path(inspect.getfile(obj)).resolve()
        if not source.is_relative_to(expected):
            raise RuntimeError(
                f"{obj.__name__} was not imported from this repository's src/physiotrack. "
                "Install the current physiotrack-v2 checkout in editable mode."
            )
        paths[obj.__name__] = source.relative_to(REPOSITORY_ROOT).as_posix()
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
            "Input directory does not exist: validation/synthetic_media/gaze_estimation_media/"
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
            "No supported images found in validation/synthetic_media/gaze_estimation_media/."
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


def _camera_yaml_path_for_shape(width: int, height: int, runtime_dir: Path) -> Path:
    if width <= 0 or height <= 0:
        raise SyntheticDemoError("Invalid image size for synthetic camera creation")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"camera_{width}x{height}.yaml"
    if path.exists():
        return path

    focal = float(max(width, height))
    data = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                focal, 0.0, width / 2.0,
                0.0, focal, height / 2.0,
                0.0, 0.0, 1.0,
            ],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def make_config() -> FaceAnalysisConfig:
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=False,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=True,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
        gaze_estimation_mode=MODEL_MODE,
        gaze_estimation_min_iou=MIN_IOU,
    )
    config.validate()
    return config


def validate_pipeline_configuration(pipeline: FaceAnalysis) -> None:
    if pipeline.detector is None:
        raise RuntimeError("FaceAnalysis detector must be enabled.")
    if pipeline.tracker is not None:
        raise RuntimeError("Tracking must be disabled.")
    if pipeline.orientation is not None:
        raise RuntimeError("Head pose must be disabled.")
    if pipeline.landmarks is not None:
        raise RuntimeError("Landmarks must be disabled.")
    if pipeline.quality is not None:
        raise RuntimeError("Face quality must be disabled.")
    if pipeline.eyes is not None:
        raise RuntimeError("Eye-openness must be disabled.")
    if pipeline.blink is not None:
        raise RuntimeError("Blink must be disabled.")
    if pipeline.gaze is not None:
        raise RuntimeError("Legacy gaze must be disabled.")
    if pipeline.gaze_estimation is None:
        raise RuntimeError("GazeEstimator must be enabled.")
    if pipeline.mouth is not None:
        raise RuntimeError("Mouth openness must be disabled.")
    if pipeline.mouth_motion is not None:
        raise RuntimeError("Mouth movement must be disabled.")
    if pipeline.emotion is not None:
        raise RuntimeError("Emotion must be disabled.")
    if pipeline.regions is not None:
        raise RuntimeError("Regions must be disabled.")
    if pipeline.temporal is not None:
        raise RuntimeError("Temporal aggregation must be disabled.")
    if pipeline.config.gaze_estimation_mode != MODEL_MODE:
        raise RuntimeError("Unexpected gaze-estimation mode.")
    if not math.isclose(float(pipeline.config.gaze_estimation_min_iou), MIN_IOU, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Unexpected gaze-estimation minimum IoU.")


def make_pipeline(camera_yaml_path: Path) -> FaceAnalysis:
    pipeline = FaceAnalysis(
        config=make_config(),
        gaze_estimation_camera_path=camera_yaml_path,
        device=DEVICE,
        verbose=False,
    )
    validate_pipeline_configuration(pipeline)
    return pipeline


def checkpoint_provenance(pipeline: FaceAnalysis) -> dict[str, str]:
    estimator = pipeline.gaze_estimation
    checkpoint_path = getattr(estimator, "checkpoint_path", None)
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise RuntimeError("GazeEstimator did not resolve a valid pretrained checkpoint.")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_path.name != EXPECTED_CHECKPOINT_FILENAME:
        raise RuntimeError(f"Unexpected GazeEstimator checkpoint filename: {checkpoint_path.name}")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Unexpected GazeEstimator checkpoint SHA256.")
    return {"checkpoint_file": checkpoint_path.name, "checkpoint_sha256": checkpoint_hash}


def finite_numeric(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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


def validate_gaze_feature(feature: Any) -> dict[str, Any]:
    if not isinstance(feature, dict):
        raise RuntimeError("FaceAnalysis gaze_estimation feature is not a dictionary.")

    available = bool(feature.get("available", False))
    association_iou = feature.get("association_iou")
    if association_iou is not None and not finite_numeric(association_iou):
        raise RuntimeError("GazeEstimator output contains invalid association IoU.")
    association_iou = None if association_iou is None else float(association_iou)

    if not available:
        return {
            "available": False,
            "gaze_x": None,
            "gaze_y": None,
            "gaze_z": None,
            "gaze_vector_norm": None,
            "pitch_deg": None,
            "yaw_deg": None,
            "association_iou": association_iou,
        }

    vector = np.asarray(feature.get("gaze_vector"), dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise RuntimeError("Available GazeEstimator output contains an invalid 3D gaze vector.")
    vector_norm = float(np.linalg.norm(vector))
    if not math.isclose(vector_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("Available GazeEstimator gaze vector is not unit-normalized.")

    pitch = feature.get("pitch")
    yaw = feature.get("yaw")
    if not finite_numeric(pitch):
        raise RuntimeError("Available GazeEstimator output contains invalid pitch.")
    if not finite_numeric(yaw):
        raise RuntimeError("Available GazeEstimator output contains invalid yaw.")
    if association_iou is None:
        raise RuntimeError("Available GazeEstimator output is missing association IoU.")
    if association_iou < MIN_IOU - 1e-12 or association_iou > 1.0 + 1e-12:
        raise RuntimeError("Available GazeEstimator output violates the configured minimum IoU.")

    return {
        "available": True,
        "gaze_x": float(vector[0]),
        "gaze_y": float(vector[1]),
        "gaze_z": float(vector[2]),
        "gaze_vector_norm": vector_norm,
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "association_iou": association_iou,
    }


def visible_image_region(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Cannot position gaze arrows on an empty image")

    scale = min(1.0, DISPLAY_REGION_MAX_SIDE / max(height, width))
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
    )
    visible = (np.max(small, axis=2) > DISPLAY_REGION_NONBLACK_THRESHOLD).astype(np.uint8)
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(visible, connectivity=8)
    if count <= 1:
        return (0, 0, width, height)

    largest = max(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    x, y, region_w, region_h, area = (int(n) for n in stats[largest])
    if area < 0.01 * small_width * small_height or region_w < 6 or region_h < 6:
        return (0, 0, width, height)

    left = max(0, int(math.floor(x * width / small_width)))
    top = max(0, int(math.floor(y * height / small_height)))
    right = min(width, int(math.ceil((x + region_w) * width / small_width)))
    bottom = min(height, int(math.ceil((y + region_h) * height / small_height)))
    if right - left < 8 or bottom - top < 8:
        return (0, 0, width, height)
    return (left, top, right, bottom)


def angle_to_display_endpoint(
    yaw_deg: float,
    pitch_deg: float,
    origin: tuple[int, int],
    image_width: int,
    image_height: int,
    display_region: tuple[int, int, int, int] | None = None,
) -> tuple[int, int]:
    yaw_display = float(np.clip(yaw_deg, -DISPLAY_MAX_ABS_ANGLE_DEG, DISPLAY_MAX_ABS_ANGLE_DEG))
    pitch_display = float(np.clip(pitch_deg, -DISPLAY_MAX_ABS_ANGLE_DEG, DISPLAY_MAX_ABS_ANGLE_DEG))

    left, top, right, bottom = display_region or (0, 0, image_width, image_height)
    region_width = right - left
    region_height = bottom - top

    dx = (yaw_display / DISPLAY_MAX_ABS_ANGLE_DEG) * (DISPLAY_AXIS_SCALE_RATIO * region_width)
    dy = (pitch_display / DISPLAY_MAX_ABS_ANGLE_DEG) * (DISPLAY_AXIS_SCALE_RATIO * region_height)

    vector_length = float(math.hypot(dx, dy))
    max_length = float(min(region_width, region_height) * DISPLAY_MAX_VECTOR_RATIO)
    if vector_length > max_length and vector_length > 0:
        scale = max_length / vector_length
        dx *= scale
        dy *= scale
        vector_length = max_length
    if 0 < vector_length < DISPLAY_MIN_VECTOR_PX:
        scale = DISPLAY_MIN_VECTOR_PX / vector_length
        dx *= scale
        dy *= scale

    endpoint_x = int(round(origin[0] + dx))
    endpoint_y = int(round(origin[1] + dy))

    margin = min(
        DISPLAY_CLIP_MARGIN_PX,
        max(0, (region_width - 1) // 4),
        max(0, (region_height - 1) // 4),
    )
    endpoint_x = max(left + margin, min(right - 1 - margin, endpoint_x))
    endpoint_y = max(top + margin, min(bottom - 1 - margin, endpoint_y))
    return endpoint_x, endpoint_y


def draw_gaze_arrow(
    canvas: np.ndarray,
    origin: tuple[int, int],
    endpoint: tuple[int, int],
    bgr_color: tuple[int, int, int],
) -> None:
    thickness = max(2, int(round(min(canvas.shape[:2]) / 220.0)))
    cv2.arrowedLine(canvas, origin, endpoint, bgr_color, thickness, cv2.LINE_AA, tipLength=0.16)
    cv2.circle(canvas, origin, max(3, thickness + 1), (255, 255, 255), -1, cv2.LINE_AA)


def _extract_origin(face: Any, box: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[int, int]:
    left, top, right, bottom = box
    kps = getattr(face, "kps", None)
    if kps is not None:
        try:
            pts = np.asarray(kps, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                finite = pts[np.all(np.isfinite(pts[:, :2]), axis=1), :2]
                if finite.shape[0] >= 3:
                    candidate = finite[2]
                elif finite.shape[0] >= 2:
                    candidate = finite[:2].mean(axis=0)
                elif finite.shape[0] == 1:
                    candidate = finite[0]
                else:
                    candidate = None
                if candidate is not None:
                    x = max(0, min(image_width - 1, int(round(float(candidate[0])))))
                    y = max(0, min(image_height - 1, int(round(float(candidate[1])))))
                    return x, y
        except (TypeError, ValueError, OverflowError):
            pass

    center_x = int(round((left + right) / 2.0))
    upper_y = int(round(top + 0.42 * (bottom - top)))
    center_x = max(0, min(image_width - 1, center_x))
    upper_y = max(0, min(image_height - 1, upper_y))
    return center_x, upper_y


def _draw_label_block(
    image: np.ndarray,
    anchor_x: int,
    anchor_y: int,
    lines: list[str],
    color: tuple[int, int, int],
    font_scale: float,
    line_width: int,
) -> None:
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


def annotate_image(
    image: np.ndarray,
    pipeline: FaceAnalysis,
) -> tuple[np.ndarray, dict[str, int], list[dict[str, Any]]]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise SyntheticDemoError("The decoded image is not a three-channel BGR frame")

    result = pipeline.predict(image)
    faces = list(result)
    output = image.copy()
    image_height, image_width = image.shape[:2]
    font_scale = max(0.38, min(0.72, min(image_height, image_width) / 950.0))
    line_width = 1 if min(image_height, image_width) < 450 else 2
    display_region = visible_image_region(image)

    valid_rows: list[dict[str, Any]] = []
    faces_with_gaze = 0
    faces_without_gaze = 0
    invalid_faces = 0
    displayed_faces = 0
    arrows_drawn = 0
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

        features = face.face_features if isinstance(getattr(face, "face_features", None), dict) else {}
        if "gaze_estimation" not in features:
            raise RuntimeError("Enabled GazeEstimator output is missing from face_features.")
        legacy_gaze = features.get("gaze")
        if isinstance(legacy_gaze, dict) and legacy_gaze.get("available", False):
            raise RuntimeError(
                "Legacy GazeDescriptor produced an available output while gaze=False in the synthetic demo configuration."
            )

        gaze = validate_gaze_feature(features["gaze_estimation"])
        color = (0, 220, 255) if gaze["available"] else (0, 140, 255)
        cv2.rectangle(output, (left, top), (right, bottom), color, line_width)
        displayed_faces += 1

        origin_x = origin_y = endpoint_x = endpoint_y = None
        line1 = f"face {face_index}"
        if confidence is not None:
            line1 += f"  det {confidence:.2f}"

        if gaze["available"]:
            origin_x, origin_y = _extract_origin(face, (left, top, right, bottom), image_width, image_height)
            endpoint_x, endpoint_y = angle_to_display_endpoint(
                gaze["yaw_deg"],
                gaze["pitch_deg"],
                (origin_x, origin_y),
                image_width,
                image_height,
                display_region=display_region,
            )
            draw_gaze_arrow(output, (origin_x, origin_y), (endpoint_x, endpoint_y), (0, 0, 255))
            arrows_drawn += 1
            faces_with_gaze += 1
            line2 = f"Y {gaze['yaw_deg']:+.1f}  P {gaze['pitch_deg']:+.1f}"
            line3 = f"IoU {gaze['association_iou']:.2f}"
            _draw_label_block(output, left, top, [line1, line2, line3], color, font_scale, line_width)
            status = "OK"
            failure_reason = ""
        else:
            faces_without_gaze += 1
            _draw_label_block(output, left, top, [line1, "gaze unavailable"], color, font_scale, line_width)
            status = "NO_GAZE_OUTPUT"
            failure_reason = "FaceAnalysis GazeEstimator returned available=False."

        valid_rows.append({
            "face_index": int(face_index),
            "confidence": float(confidence) if confidence is not None else "",
            "box_x1": int(left),
            "box_y1": int(top),
            "box_x2": int(right),
            "box_y2": int(bottom),
            "origin_x": "" if origin_x is None else int(origin_x),
            "origin_y": "" if origin_y is None else int(origin_y),
            "endpoint_x": "" if endpoint_x is None else int(endpoint_x),
            "endpoint_y": "" if endpoint_y is None else int(endpoint_y),
            "gaze_available": bool(gaze["available"]),
            "gaze_x": "" if gaze["gaze_x"] is None else float(gaze["gaze_x"]),
            "gaze_y": "" if gaze["gaze_y"] is None else float(gaze["gaze_y"]),
            "gaze_z": "" if gaze["gaze_z"] is None else float(gaze["gaze_z"]),
            "gaze_vector_norm": "" if gaze["gaze_vector_norm"] is None else float(gaze["gaze_vector_norm"]),
            "pitch_deg": "" if gaze["pitch_deg"] is None else float(gaze["pitch_deg"]),
            "yaw_deg": "" if gaze["yaw_deg"] is None else float(gaze["yaw_deg"]),
            "association_iou": "" if gaze["association_iou"] is None else float(gaze["association_iou"]),
            "status": status,
            "failure_reason": failure_reason,
        })

    counts = {
        "faces_detected": len(faces),
        "displayed_faces": displayed_faces,
        "gaze_arrows_drawn": arrows_drawn,
        "faces_skipped_low_confidence": faces_skipped_low_confidence,
        "faces_with_gaze": faces_with_gaze,
        "faces_without_gaze": faces_without_gaze,
        "invalid_faces": invalid_faces,
    }
    return output, counts, valid_rows


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
        stats = f"faces={record['faces_detected']} gaze={record['faces_with_gaze']}"
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
    sources = {}
    for obj in (FaceAnalysis, FaceAnalysisConfig, GazeEstimator):
        source = Path(inspect.getfile(obj)).resolve()
        sources[source.relative_to(REPOSITORY_ROOT).as_posix()] = sha256_file(source)
    versions = {}
    for package in ("numpy", "opencv-python", "opencv-contrib-python", "torch", "ultralytics", "scipy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_sha256": dict(sorted(sources.items())),
        "package_versions": versions,
        "note": (
            "Model weights, image-space camera assumptions and runtime dependencies can change numerical predictions. "
            "Input and implementation hashes aid reproducibility but do not guarantee identical results on another machine."
        ),
    }


def summary_as_text(summary: dict[str, Any]) -> str:
    config = summary["configuration"]
    lines = [
        "PhysioTrack Gaze Estimation - Synthetic Qualitative Demonstration",
        "=" * 62, "",
        "Status: " + summary["status"],
        "Evaluation type: qualitative only (not MPIIFaceGaze benchmark)",
        "Source: " + summary["input_directory"],
        "Output: " + summary["output_directory"], "",
        "Discovered images: " + str(summary["discovered_files"]),
        "Successfully processed images: " + str(summary["successful_files"]),
        "Failed images: " + str(summary["failed_files"]),
        "Faces detected: " + str(summary["faces_detected"]),
        "Displayed faces: " + str(summary["displayed_faces"]),
        "Faces with gaze output: " + str(summary["faces_with_gaze"]),
        "Faces without gaze output: " + str(summary["faces_without_gaze"]),
        "Gaze arrows drawn: " + str(summary["gaze_arrows_drawn"]),
        "Detections below display threshold: " + str(summary["faces_skipped_low_confidence"]),
        "Invalid face results: " + str(summary["invalid_faces"]),
        "Observation CSV rows: " + str(summary["observation_rows"]), "",
        "Configuration", "-------------",
        "Device: " + str(config["device"]),
        "Model mode: " + str(config["gaze_estimation_mode"]),
        "Detector confidence threshold: " + str(config["detector_confidence"]),
        "Display confidence threshold: " + str(config["display_confidence_threshold"]),
        "Gaze min IoU: " + str(config["gaze_estimation_min_iou"]),
        "Synthetic camera model: " + str(config["synthetic_camera_model"]),
        "Ground truth: not used",
        "Benchmark accuracy metrics: not calculated", "",
        "Images", "------",
    ]
    for rec in summary["records"]:
        lines.append(
            f"{rec['input']}: {rec['status']}"
            + (f"; faces={rec['faces_detected']}; gaze={rec['faces_with_gaze']}"
               if rec["status"] == "SUCCESS" else f"; {rec.get('error', 'processing failure')}")
        )
        if rec.get("source_sha256"):
            lines.append("  Input SHA-256: " + rec["source_sha256"])
        if rec["status"] == "SUCCESS":
            lines.append("  Annotated image: " + rec["output_image"])
    if summary["status"] == "COMPLETE":
        lines.extend([
            "", "Published files", "---------------",
            "images/", "gaze_observations.csv",
            "synthetic_demo_examples.png", "synthetic_demo_summary.json",
            "synthetic_demo_summary.txt",
        ])
    else:
        lines.extend(["", "The failed attempt was not published; any existing accepted results were preserved."])
    lines.extend([
        "",
        "Interpretation: visual demonstrations and model-predicted gaze directions only; there are no synthetic ground-truth gaze labels here.",
        "Reproduction: use identical media, implementation, dependencies, model weights and synthetic camera assumptions.",
    ])
    return "\n".join(lines) + "\n"


def write_reports(folder: Path, summary: dict[str, Any], prefix: str) -> None:
    with (folder / f"{prefix}.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    (folder / f"{prefix}.txt").write_text(summary_as_text(summary), encoding="utf-8", newline="\n")


def verify_staged_outputs(stage: Path, summary: dict[str, Any]) -> None:
    if not stage.is_dir() or stage.is_symlink():
        raise SyntheticDemoError("Output directory is missing or unsafe")
    records = summary.get("records", [])
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("failed_files") != 0
        or not records
        or len(records) != summary.get("discovered_files")
        or len(records) != summary.get("successful_files")
        or any(rec.get("status") != "SUCCESS" for rec in records)
    ):
        raise SyntheticDemoError("Only a complete and successful run may be published")

    images_dir = stage / "images"
    if not images_dir.is_dir() or images_dir.is_symlink():
        raise SyntheticDemoError("Annotated images directory is missing or unsafe")
    required_files = (
        "gaze_observations.csv",
        "synthetic_demo_examples.png",
        "synthetic_demo_summary.json",
        "synthetic_demo_summary.txt",
    )
    for name in required_files:
        p = stage / name
        if not p.is_file() or p.is_symlink() or p.stat().st_size == 0:
            raise SyntheticDemoError(f"Missing or empty required artifact: {name}")

    expected_images: set[str] = set()
    by_input: dict[str, dict[str, Any]] = {}
    for rec in records:
        rel = rec["output_image"]
        if rec["input"] in by_input or rel in expected_images or not rel.startswith("images/"):
            raise SyntheticDemoError("Duplicate/invalid input or output image")
        by_input[rec["input"]] = rec
        expected_images.add(rel)
        path = stage / rel
        if not path.resolve().is_relative_to(stage.resolve()) or not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SyntheticDemoError(f"Missing/unsafe annotated image: {rel}")
        img = image_from_path(path)
        if img.shape[:2] != (rec["height"], rec["width"]):
            raise SyntheticDemoError(f"Annotated image dimensions disagree for: {rec['input']}")
        original = INPUT_DIR / rec["input"]
        if not original.is_file() or original.is_symlink() or sha256_file(original) != rec["source_sha256"]:
            raise SyntheticDemoError("An input was altered during execution")
        if not (
            rec["faces_detected"] == rec["displayed_faces"] + rec["faces_skipped_low_confidence"] + rec["invalid_faces"]
            and rec["displayed_faces"] == rec["faces_with_gaze"] + rec["faces_without_gaze"]
            and rec["gaze_arrows_drawn"] == rec["faces_with_gaze"]
        ):
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
    with (stage / "gaze_observations.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OBSERVATION_FIELDS:
            raise SyntheticDemoError("Gaze observations CSV schema mismatch")
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
                confidence = float(row["confidence"]) if row["confidence"] != "" else None
                gaze_available = row["gaze_available"] == "True"
            except (ValueError, TypeError) as exc:
                raise SyntheticDemoError("Invalid numerical gaze observation") from exc
            if (
                not 1 <= index <= rec["faces_detected"]
                or (row["input"], index) in seen_faces
                or not (0 <= box[0] < box[2] < rec["width"] and 0 <= box[1] < box[3] < rec["height"])
                or (confidence is not None and not (0 <= confidence <= 1))
            ):
                raise SyntheticDemoError("Invalid/duplicate face or box coordinates in CSV")
            seen_faces.add((row["input"], index))
            if gaze_available:
                try:
                    origin = [int(row["origin_x"]), int(row["origin_y"])]
                    endpoint = [int(row["endpoint_x"]), int(row["endpoint_y"])]
                    gaze_vals = [float(row[x]) for x in ("gaze_x", "gaze_y", "gaze_z", "gaze_vector_norm", "pitch_deg", "yaw_deg", "association_iou")]
                except (ValueError, TypeError) as exc:
                    raise SyntheticDemoError("Missing/invalid available-gaze values in CSV") from exc
                if (
                    not all(math.isfinite(v) for v in gaze_vals)
                    or not (0 <= origin[0] < rec["width"] and 0 <= origin[1] < rec["height"])
                    or not (0 <= endpoint[0] < rec["width"] and 0 <= endpoint[1] < rec["height"])
                    or not math.isclose(gaze_vals[3], 1.0, rel_tol=0.0, abs_tol=1e-6)
                    or not (MIN_IOU - 1e-12 <= gaze_vals[6] <= 1.0 + 1e-12)
                    or row["status"] != "OK"
                    or row["failure_reason"] != ""
                ):
                    raise SyntheticDemoError("Invalid available-gaze row in CSV")
            else:
                if any(row[key] != "" for key in ("origin_x", "origin_y", "endpoint_x", "endpoint_y", "gaze_x", "gaze_y", "gaze_z", "gaze_vector_norm", "pitch_deg", "yaw_deg")):
                    raise SyntheticDemoError("Unavailable-gaze row contains unexpected non-empty values")
                if row["status"] != "NO_GAZE_OUTPUT":
                    raise SyntheticDemoError("Unavailable-gaze row has an unexpected status")
                if not row["failure_reason"]:
                    raise SyntheticDemoError("Unavailable-gaze row is missing a failure reason")
                if row["association_iou"] != "":
                    try:
                        assoc = float(row["association_iou"])
                    except (ValueError, TypeError) as exc:
                        raise SyntheticDemoError("Invalid unavailable-gaze association IoU") from exc
                    if not math.isfinite(assoc):
                        raise SyntheticDemoError("Unavailable-gaze association IoU is not finite")
            csv_counts[row["input"]] += 1
            total_rows += 1
    if total_rows != summary["observation_rows"]:
        raise SyntheticDemoError("Total gaze observation rows do not match summary")
    for name, rec in by_input.items():
        if csv_counts[name] != rec["displayed_faces"]:
            raise SyntheticDemoError(f"Per-image gaze observation count mismatch: {name}")

    for metric in (
        "faces_detected", "displayed_faces", "faces_with_gaze", "faces_without_gaze",
        "gaze_arrows_drawn", "faces_skipped_low_confidence", "invalid_faces",
    ):
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

    staging_root = Path(tempfile.mkdtemp(prefix=".gaze_estimation_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    runtime_dir = staging_root / "runtime"
    stage.mkdir()
    (stage / "images").mkdir()

    records: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    pipelines: dict[tuple[int, int], FaceAnalysis] = {}
    checkpoint_info: dict[str, str] | None = None

    try:
        provenance = execution_fingerprint()
        print("PhysioTrack gaze estimation synthetic qualitative demonstration")
        print("Images discovered:", len(images))
        print("Inputs: validation/synthetic_media/gaze_estimation_media/")
        print("Outputs: validation/gaze_estimation/results/synthetic_demo/")

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
                image = image_from_path(source)
                image_height, image_width = image.shape[:2]
                camera_yaml = _camera_yaml_path_for_shape(image_width, image_height, runtime_dir)
                key_shape = (image_width, image_height)
                pipeline = pipelines.get(key_shape)
                if pipeline is None:
                    pipeline = make_pipeline(camera_yaml)
                    if checkpoint_info is None:
                        checkpoint_info = checkpoint_provenance(pipeline)
                    pipelines[key_shape] = pipeline
                source_digest = sha256_file(source)
                annotated, counts, rows = annotate_image(image, pipeline)
                save_png(annotated, stage / output_image)
                decoded = image_from_path(stage / output_image)
                if decoded.shape[:2] != image.shape[:2]:
                    raise SyntheticDemoError("Annotated image dimensions differ from source")

                record["source_sha256"] = source_digest
                record["width"] = int(image_width)
                record["height"] = int(image_height)
                record.update(counts)
                record["camera_yaml"] = camera_yaml.name
                record["status"] = "SUCCESS"

                for row in rows:
                    observation_rows.append({
                        "input": relative,
                        "source_sha256": source_digest,
                        "output_image": output_image.as_posix(),
                        "width": record["width"],
                        "height": record["height"],
                        **row,
                    })

                print(
                    f"    OK: faces={record['faces_detected']}; "
                    f"faces_with_gaze={record['faces_with_gaze']}; "
                    f"faces_without_gaze={record['faces_without_gaze']}"
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {record['error']}")
            records.append(record)

        create_contact_sheet(stage, records)
        write_observations_csv(stage / "gaze_observations.csv", observation_rows)

        successful_records = [r for r in records if r["status"] == "SUCCESS"]
        failed_records = [r for r in records if r["status"] != "SUCCESS"]

        summary = {
            "type": "synthetic_qualitative_gaze_estimation_demonstration",
            "benchmark_evaluation": False,
            "project": "physiotrack-v2",
            "input_directory": "validation/synthetic_media/gaze_estimation_media",
            "output_directory": "validation/gaze_estimation/results/synthetic_demo",
            "component_imports": component_paths,
            "reproducibility": provenance,
            "checkpoint": checkpoint_info,
            "configuration": {
                "pipeline": "physiotrack.face.FaceAnalysis",
                "gaze_estimation_component": "physiotrack.face.gaze_estimation.GazeEstimator",
                "tracking_enabled": False,
                "head_pose_enabled": False,
                "landmarks_enabled": False,
                "legacy_gaze_enabled": False,
                "gaze_estimation_enabled": True,
                "gaze_estimation_mode": MODEL_MODE,
                "gaze_estimation_min_iou": MIN_IOU,
                "detector_confidence": DETECTOR_CONFIDENCE,
                "display_confidence_threshold": DISPLAY_CONFIDENCE,
                "device": DEVICE,
                "synthetic_camera_model": "per-image-size pinhole camera with focal=max(width,height) and zero distortion",
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
                "No ground-truth gaze labels or MPIIFaceGaze benchmark targets are used. A generic zero-distortion "
                "camera file is synthesized from each image size so the current GazeEstimator can run on standalone images."
            ),
            "status": "COMPLETE" if not failed_records else "REJECTED_WITH_FILE_FAILURES",
            "discovered_files": len(images),
            "successful_files": len(successful_records),
            "failed_files": len(failed_records),
            "images_processed": len(successful_records),
            "faces_detected": int(sum(r.get("faces_detected", 0) for r in successful_records)),
            "displayed_faces": int(sum(r.get("displayed_faces", 0) for r in successful_records)),
            "faces_with_gaze": int(sum(r.get("faces_with_gaze", 0) for r in successful_records)),
            "faces_without_gaze": int(sum(r.get("faces_without_gaze", 0) for r in successful_records)),
            "gaze_arrows_drawn": int(sum(r.get("gaze_arrows_drawn", 0) for r in successful_records)),
            "faces_skipped_low_confidence": int(sum(r.get("faces_skipped_low_confidence", 0) for r in successful_records)),
            "invalid_faces": int(sum(r.get("invalid_faces", 0) for r in successful_records)),
            "observation_rows": len(observation_rows),
            "records": records,
        }

        write_reports(stage, summary, "synthetic_demo_summary")

        if failed_records:
            write_reports(RESULTS_DIR, summary, "synthetic_demo_last_attempt")
            raise SyntheticDemoError(
                f"{len(failed_records)} image(s) failed. Existing synthetic_demo was preserved. "
                "See results/synthetic_demo_last_attempt.json and .txt."
            )
        verify_staged_outputs(stage, summary)
        commit_staged_directory(stage, staging_root, summary)

        for ext in ("json", "txt"):
            try:
                (RESULTS_DIR / f"synthetic_demo_last_attempt.{ext}").unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARNING: could not remove old failed-attempt report: {exc}")

        print("\nFinished.")
        print(f"Images processed: {summary['images_processed']}")
        print(f"Faces detected: {summary['faces_detected']}")
        print(f"Faces with gaze output: {summary['faces_with_gaze']}")
        print(f"Faces without gaze output: {summary['faces_without_gaze']}")
        print(f"Gaze arrows drawn: {summary['gaze_arrows_drawn']}")
        print(f"Detections below display threshold: {summary['faces_skipped_low_confidence']}")
        print(f"Invalid faces: {summary['invalid_faces']}")
        print(f"Observation rows: {summary['observation_rows']}")
        print(f"Failed files: {summary['failed_files']}")
        print(f"Overall status: {summary['status']}")
        print("Saved summaries: validation/gaze_estimation/results/synthetic_demo/synthetic_demo_summary.{json,txt}")

    finally:
        for pipeline in pipelines.values():
            try:
                if hasattr(pipeline, "close"):
                    pipeline.close()
            except Exception:
                pass
        if (staging_root / "previous_synthetic_demo").exists():
            print("WARNING: previous results backup preserved at", staging_root / "previous_synthetic_demo")
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