"""PhysioTrack-v2 Face Regions synthetic qualitative demonstration.

Install at: validation/face_regions/face_regions_synthetic_demo.py
Read images from: validation/synthetic_media/face_regions_media/
Publish ONLY to: validation/face_regions/results/synthetic_demo/

The REAL FaceAnalysis -> FaceRegions -> SegFace pipeline is used. A controlled
full-aligned-image box (not a learned face detector) matches the scientific
CelebAMask-HQ segmentation protocol. Source images are resized to 512x512.
Outputs include three-panel figures, real label-map PNGs, overlays, a contact
sheet, a per-class CSV, and complete JSON/TXT execution reports.

This demonstration has NO ground-truth labels, computes NO benchmark accuracy,
and never reads/writes the CelebAMask-HQ benchmark outputs. Results are replaced
only after a complete, independently validated successful run. Reproducible
inputs and code hashes do NOT guarantee numerically identical model predictions
across environments with different model weights or library versions.
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
import tempfile
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig, FaceRegions
from physiotrack.results import Instance, Result


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "face_regions_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

SUPPORTED_IMAGE_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
})
DEVICE = "cpu"
SIZE = 512
CLASS_NAMES = (
    "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow",
    "r_brow", "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip",
    "hair", "eye_g", "hat", "ear_r", "neck_l",
)
NUM_CLASSES = len(CLASS_NAMES)
# The color palette is for PRESENTATION only, and is consistent with the
# scientific qualitative rendering. It does not modify model predictions.
COLORS_BGR = np.array([
    [35, 35, 35], [140, 75, 35], [180, 160, 120], [95, 70, 155],
    [55, 190, 230], [40, 150, 245], [180, 90, 210], [210, 115, 195],
    [215, 150, 80], [235, 175, 85], [45, 210, 245], [75, 85, 220],
    [130, 80, 230], [165, 105, 245], [60, 120, 60], [220, 210, 80],
    [80, 190, 115], [170, 120, 80], [120, 175, 150],
], dtype=np.uint8)
CSV_FIELDS = (
    "source_image", "source_sha256", "class_id", "class_name", "pixel_count",
    "pixel_fraction", "image_width", "image_height", "association_iou",
    "pipeline_pixel_count", "native_pixel_count", "pixel_count_match", "status",
)
OUTPUT_SUBDIRS = ("annotated_images", "label_maps", "segmentation_maps", "overlays")


class SyntheticDemoError(RuntimeError):
    """A media, component, consistency, or output-publication failure."""


class ControlledFullImageDetector:
    """Same controlled aligned-image face box as component/benchmark tests."""

    def predict(self, frame: np.ndarray) -> Result:
        height, width = frame.shape[:2]
        return Result(
            orig_img=frame,
            instances=[Instance(
                box=np.array([0.0, 0.0, float(width), float(height)], dtype=float),
                confidence=1.0, cls=0, cls_name="face",
            )],
            task="face",
        )


class RegionCapture:
    """Read-only recorder of the ORIGINAL native FaceRegions.predict output."""

    def __init__(self, regions: FaceRegions) -> None:
        self.regions = regions
        self.last_output: Any = None

    def predict(self, frame: np.ndarray, boxes: Any = None) -> Any:
        output = self.regions.predict(frame, boxes=boxes)
        self.last_output = output
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self.regions, name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_project_imports() -> dict[str, str]:
    """Reject imports from a different installation of physiotrack."""
    root = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    if not root.is_dir():
        raise SyntheticDemoError("Current repository's src/physiotrack is missing")
    found = {}
    for obj in (FaceAnalysis, FaceAnalysisConfig, FaceRegions, Instance, Result):
        source = Path(inspect.getfile(obj)).resolve()
        if not source.is_relative_to(root):
            raise SyntheticDemoError(
                f"{obj.__name__} is not imported from this physiotrack-v2 checkout; "
                "activate the project environment and editable install."
            )
        found[obj.__name__] = source.relative_to(REPOSITORY_ROOT).as_posix()
    return found


def discover_images() -> list[Path]:
    if not INPUT_DIR.is_dir() or INPUT_DIR.is_symlink():
        raise SyntheticDemoError(
            "Missing/unsafe media folder: validation/synthetic_media/face_regions_media/"
        )
    images = sorted(
        (p for p in INPUT_DIR.rglob("*")
         if p.is_file() and not p.is_symlink()
         and p.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda p: (p.relative_to(INPUT_DIR).as_posix().casefold(),
                       p.relative_to(INPUT_DIR).as_posix()),
    )
    if not images:
        raise SyntheticDemoError("No synthetic images found; existing results are unchanged")
    # Reject ambiguous names on case-insensitive systems (notably Windows).
    keys = [p.relative_to(INPUT_DIR).as_posix().casefold() for p in images]
    if len(keys) != len(set(keys)):
        raise SyntheticDemoError("Two input paths differ only by case")
    return images


def output_stem(source: Path) -> str:
    relative = source.relative_to(INPUT_DIR).as_posix()
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip(". ")[:72] or "image"
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"{base}_{digest}"


def read_image(path: Path) -> np.ndarray:
    try:
        array = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise SyntheticDemoError(f"Failed to decode image {path.name}") from exc
    if (decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3
            or min(decoded.shape[:2]) <= 0):
        raise SyntheticDemoError(f"Image is not valid BGR: {path.name}")
    return decoded


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise SyntheticDemoError(f"Could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def make_pipeline() -> tuple[FaceAnalysis, RegionCapture]:
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=True,
        temporal=False,
    )
    config.validate()
    pipeline = FaceAnalysis(
        detector=ControlledFullImageDetector(), config=config,
        device=DEVICE, verbose=False,
    )
    if (pipeline.regions is None or not isinstance(pipeline.regions, FaceRegions)
            or any(getattr(pipeline, attr) is not None for attr in (
                "tracker", "orientation", "landmarks", "quality", "eyes", "blink",
                "gaze", "gaze_estimation", "mouth", "mouth_motion", "emotion", "temporal",
            ))):
        pipeline.close()
        raise SyntheticDemoError("FaceAnalysis did not initialize FaceRegions-only execution")
    capture = RegionCapture(pipeline.regions)
    pipeline.regions = capture
    return pipeline, capture


def validated_label_map(native: Any) -> np.ndarray:
    """Validate numeric values BEFORE converting (prevent uint8 wrapping)."""
    array = np.squeeze(np.asarray(native))
    if array.shape != (SIZE, SIZE) or array.dtype.kind not in "iuf":
        raise SyntheticDemoError(f"Invalid raw SegFace label map: {array.shape}, {array.dtype}")
    if (not np.isfinite(array).all() or (array < 0).any()
            or (array >= NUM_CLASSES).any()
            or (array != np.floor(array)).any()):
        raise SyntheticDemoError("SegFace returned non-finite or out-of-range class IDs")
    return array.astype(np.uint8, copy=False)


def evaluate_real_prediction(pipeline: FaceAnalysis, capture: RegionCapture,
                             frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one ORIGINAL pipeline prediction and cross-check its component data."""
    capture.last_output = None
    result = pipeline.predict(frame)
    if len(result) != 1:
        raise SyntheticDemoError(f"Expected one controlled face, found {len(result)}")
    face = result[0]
    features = face.face_features if isinstance(face.face_features, dict) else {}
    region_feature = features.get("regions")
    if not isinstance(region_feature, dict):
        raise SyntheticDemoError("FaceAnalysis did not provide regions face_features")
    native = capture.last_output
    if not isinstance(native, dict) or not isinstance(native.get("faces"), list):
        raise SyntheticDemoError("Native FaceRegions prediction was not captured")
    if len(native["faces"]) != 1:
        raise SyntheticDemoError("Native FaceRegions prediction has unexpected face count")
    region_face = native["faces"][0]
    if not isinstance(region_face, dict) or not isinstance(region_face.get("regions"), dict):
        raise SyntheticDemoError("Native FaceRegions output is malformed")
    segment_result = native.get("result")
    if segment_result is None or getattr(segment_result, "seg_map", None) is None:
        raise SyntheticDemoError("SegFace did not return a semantic label map")
    label_map = validated_label_map(segment_result.seg_map)
    width, height = frame.shape[1], frame.shape[0]
    full_box = np.array([0.0, 0.0, float(width), float(height)])
    for name, box in (("FaceAnalysis", face.box), ("native FaceRegions", region_face.get("box"))):
        try:
            coords = np.asarray(box, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise SyntheticDemoError(f"Malformed {name} face box") from exc
        if (coords.shape != (4,) or not np.isfinite(coords).all()
                or not np.allclose(coords, full_box, atol=1e-9, rtol=0)):
            raise SyntheticDemoError(f"{name} did not preserve the full-image input box")

    raw_regions = region_face["regions"]
    native_counts: dict[str, int] = {}
    for name, raw_mask in raw_regions.items():
        key = str(name)
        if key not in CLASS_NAMES[1:]:
            raise SyntheticDemoError(f"Unknown native SegFace region: {key}")
        mask = np.asarray(raw_mask)
        if (mask.shape != (SIZE, SIZE) or mask.dtype.kind not in "biuf"
                or not np.isfinite(mask).all()
                or not np.all((mask == 0) | (mask == 1))):
            raise SyntheticDemoError(f"Malformed native boolean mask for {key}")
        count = int(mask.sum())
        if count != int(np.count_nonzero(label_map == CLASS_NAMES.index(key))):
            raise SyntheticDemoError(f"Native mask does not match SegFace label map: {key}")
        native_counts[key] = count
    pixel_counts = region_feature.get("pixel_counts")
    if not isinstance(pixel_counts, dict):
        raise SyntheticDemoError("FaceAnalysis regions pixel_counts is missing")
    try:
        pipeline_counts = {str(k): int(v) for k, v in pixel_counts.items()}
    except (TypeError, ValueError) as exc:
        raise SyntheticDemoError("Invalid FaceAnalysis region pixel counts") from exc
    if (pipeline_counts != native_counts
            or list(pipeline_counts) != [str(v) for v in region_feature.get("classes", [])]
            or bool(region_feature.get("available")) != bool(native_counts)):
        raise SyntheticDemoError("FaceAnalysis and native FaceRegions outputs disagree")
    for cls_id, cls_name in enumerate(CLASS_NAMES[1:], start=1):
        if int(np.count_nonzero(label_map == cls_id)) > 0 and cls_name not in native_counts:
            raise SyntheticDemoError(f"Native FaceRegions omitted predicted class {cls_name}")
    assoc = region_feature.get("association_iou")
    if (assoc is None or not math.isfinite(float(assoc))
            or not math.isclose(float(assoc), 1.0, abs_tol=1e-9, rel_tol=0)):
        raise SyntheticDemoError("Controlled full-image association IoU is not one")
    counts = np.bincount(label_map.ravel(), minlength=NUM_CLASSES)
    if int(counts.sum()) != SIZE * SIZE:
        raise SyntheticDemoError("SegFace label map did not account for every pixel")
    if int(counts[1:].sum()) <= 0:
        raise SyntheticDemoError("No foreground segmentation regions for this synthetic image")
    skin_count = int(region_feature.get("skin_pixel_count", 0))
    skin_fraction = region_feature.get("skin_fraction")
    if (skin_count != pipeline_counts.get("skin", 0)
            or skin_fraction is None or not math.isclose(
                float(skin_fraction), skin_count / float(SIZE * SIZE),
                abs_tol=1e-12, rel_tol=0,
            )):
        raise SyntheticDemoError("FaceAnalysis skin accounting is inconsistent")
    info = {
        "association_iou": float(assoc),
        "class_counts": {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)},
        "native_counts": native_counts,
        "pipeline_counts": pipeline_counts,
        "foreground_classes": [name for name in CLASS_NAMES[1:] if name in native_counts],
        "foreground_pixels": int(counts[1:].sum()),
        "background_pixels": int(counts[0]),
        "skin_fraction": float(skin_fraction),
    }
    return label_map, info


def colorize(label_map: np.ndarray) -> np.ndarray:
    return COLORS_BGR[label_map]


def make_overlay(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(image, 0.48, colorize(label_map), 0.52, 0.0)


def text_on(canvas: np.ndarray, value: str, pos: tuple[int, int],
            scale: float = 0.54, color: tuple[int, int, int] = (232, 232, 232),
            thickness: int = 1) -> None:
    cv2.putText(canvas, value, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_three_panel(image: np.ndarray, label_map: np.ndarray,
                       filename: str, info: dict[str, Any]) -> np.ndarray:
    """Original, colored segmentation and overlay; no fabricated ground truth."""
    margin, gap, cell = 22, 18, SIZE
    width = 2 * margin + 3 * cell + 2 * gap
    height = 890
    canvas = np.full((height, width, 3), 29, dtype=np.uint8)
    panels = (("Synthetic input", image),
              ("SegFace prediction", colorize(label_map)),
              ("Prediction overlay", make_overlay(image, label_map)))
    for j, (title, panel) in enumerate(panels):
        x = margin + j * (cell + gap)
        canvas[88:88 + cell, x:x + cell] = panel
        text_on(canvas, title, (x + 7, 66), 0.70, thickness=2)
    text_on(canvas, f"PhysioTrack Face Regions  |  {filename[:92]}", (margin, 27), 0.61)
    text_on(canvas, "Illustrative synthetic media - NO ground-truth segmentation or benchmark scores",
            (margin, 626), 0.57)
    text_on(canvas,
            f"Controlled full-image box | SegFace | CPU | {SIZE} x {SIZE} | "
            f"foreground classes: {len(info['foreground_classes'])}", (margin, 654), 0.56)
    text_on(canvas,
            f"Foreground pixels: {info['foreground_pixels']:,} | "
            f"association IoU: {info['association_iou']:.2f} (pipeline association, NOT segmentation accuracy)",
            (margin, 684), 0.54)
    text_on(canvas, "Semantic legend (SegFace class IDs):", (margin, 719), 0.49)
    for class_id, name in enumerate(CLASS_NAMES):
        row, col = divmod(class_id, 5)
        xx, yy = margin + col * 315, 746 + row * 32
        cv2.rectangle(canvas, (xx, yy - 12), (xx + 20, yy + 8),
                      tuple(int(v) for v in COLORS_BGR[class_id]), -1)
        text_on(canvas, f"{class_id:02d} {name}", (xx + 29, yy + 5), 0.47)
    return canvas


def render_contact_sheet(stage: Path, records: list[dict[str, Any]]) -> None:
    columns, cell_w, cell_h = 2, 680, 315
    rows = math.ceil(len(records) / columns)
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 238, np.uint8)
    for index, rec in enumerate(records):
        if rec.get("status") != "SUCCESS":
            continue
        visual = read_image(stage / rec["overlays_path"])
        thumb = cv2.resize(visual, (256, 256), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        x, y = col * cell_w, row * cell_h
        cv2.rectangle(canvas, (x, y), (x + cell_w - 1, y + cell_h - 1), (178, 178, 178), 1)
        canvas[y + 36:y + 292, x + 16:x + 272] = thumb
        display = Path(rec["input"]).name[:39]
        text_on(canvas, display, (x + 286, y + 67), 0.53, (30, 30, 30))
        text_on(canvas, f"classes: {len(rec['foreground_classes'])}",
                (x + 286, y + 102), 0.56, (40, 40, 40))
        text_on(canvas, "SegFace prediction", (x + 286, y + 134), 0.53, (40, 40, 40))
        text_on(canvas, "Synthetic qualitative demo", (x + 286, y + 167), 0.45, (50, 50, 50))
    save_png(stage / "synthetic_demo_examples.png", canvas)


def summary_as_text(summary: dict[str, Any]) -> str:
    lines = ["PhysioTrack-v2 Face Regions - Synthetic Qualitative Demonstration",
             "=" * 70, "", "Full human-readable projection of synthetic_demo_summary.json", ""]

    def walk(value: Any, indent: int = 0, name: str | None = None) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            if name is not None:
                lines.append(prefix + name + ":")
                indent += 2
            if not value:
                lines.append(" " * indent + "(empty)")
            for k, v in value.items():
                walk(v, indent, str(k))
        elif isinstance(value, list):
            if name is not None:
                lines.append(f"{prefix}{name}: ({len(value)} items)")
                indent += 2
            if not value:
                lines.append(" " * indent + "(none)")
            for i, v in enumerate(value, 1):
                walk(v, indent, f"item_{i:03d}")
        else:
            rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
            lines.append(f"{prefix}{name}: {rendered}")

    walk(summary)
    return "\n".join(lines) + "\n"


def write_reports(folder: Path, summary: dict[str, Any], basename: str) -> None:
    with (folder / f"{basename}.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    (folder / f"{basename}.txt").write_text(summary_as_text(summary), encoding="utf-8", newline="\n")


def provenance() -> dict[str, Any]:
    sources = {}
    for obj in (FaceAnalysis, FaceAnalysisConfig, FaceRegions, Instance, Result):
        p = Path(inspect.getfile(obj)).resolve()
        sources[p.relative_to(REPOSITORY_ROOT).as_posix()] = sha256_file(p)
    packages = {}
    for name in ("numpy", "opencv-python", "opencv-contrib-python", "torch", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation_sha256": dict(sorted(sources.items())),
        "package_versions": packages,
        "warning": "Exact predictions depend on weights and runtime; no weight hash is asserted here.",
    }


def relative_outputs(source: Path) -> dict[str, str]:
    stem = output_stem(source)
    return {
        "annotated_path": f"annotated_images/{stem}_panels.png",
        "label_maps_path": f"label_maps/{stem}_labels.png",
        "segmentation_maps_path": f"segmentation_maps/{stem}_colors.png",
        "overlays_path": f"overlays/{stem}_overlay.png",
    }


def process_image(source: Path, stage: Path, pipeline: FaceAnalysis,
                  capture: RegionCapture) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_image = read_image(source)
    original_height, original_width = raw_image.shape[:2]
    image = cv2.resize(raw_image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    label_map, info = evaluate_real_prediction(pipeline, capture, image)
    output_paths = relative_outputs(source)
    for key, visual in (
        ("annotated_path", render_three_panel(image, label_map, source.name, info)),
        ("label_maps_path", label_map),
        ("segmentation_maps_path", colorize(label_map)),
        ("overlays_path", make_overlay(image, label_map)),
    ):
        save_png(stage / output_paths[key], visual)
    source_rel = source.relative_to(INPUT_DIR).as_posix()
    source_hash = sha256_file(source)
    class_rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        count = int(info["class_counts"][name])
        # Background is derived from the raw class-ID map. FaceRegions exposes
        # masks/counts for FOREGROUND regions only; do not fabricate native or
        # FaceAnalysis background counts.
        native = "" if name == "background" else info["native_counts"].get(name, 0)
        pipeline_count = "" if name == "background" else info["pipeline_counts"].get(name, 0)
        class_rows.append({
            "source_image": source_rel,
            "source_sha256": source_hash,
            "class_id": class_id,
            "class_name": name,
            "pixel_count": count,
            "pixel_fraction": count / float(SIZE * SIZE),
            "image_width": SIZE,
            "image_height": SIZE,
            "association_iou": info["association_iou"],
            "pipeline_pixel_count": pipeline_count,
            "native_pixel_count": native,
            "pixel_count_match": "N/A" if name == "background" else (native == pipeline_count == count),
            "status": "OK",
        })
    record = {
        "input": source_rel,
        "source_sha256": source_hash,
        "original_width": original_width,
        "original_height": original_height,
        "input_width": SIZE,
        "input_height": SIZE,
        "association_iou": info["association_iou"],
        "foreground_classes": info["foreground_classes"],
        "foreground_pixels": info["foreground_pixels"],
        "background_pixels": info["background_pixels"],
        "skin_fraction": info["skin_fraction"],
        "class_counts": info["class_counts"],
        **output_paths,
        "output_sha256": {key: sha256_file(stage / path)
                          for key, path in output_paths.items()},
        "status": "SUCCESS",
    }
    return record, class_rows


def write_rows(stage: Path, rows: list[dict[str, Any]]) -> None:
    with (stage / "face_regions_synthetic_observations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def verify_stage(stage: Path, summary: dict[str, Any]) -> None:
    """Recompute every per-class count from the saved raw class-ID PNGs."""
    records = summary.get("records", [])
    if (summary.get("status") != "COMPLETE" or not records
            or summary.get("discovered_files") != len(records)
            or summary.get("successful_files") != len(records)
            or summary.get("failed_files") != 0):
        raise SyntheticDemoError("Incomplete run cannot replace existing results")
    for part in OUTPUT_SUBDIRS:
        if not (stage / part).is_dir() or (stage / part).is_symlink():
            raise SyntheticDemoError(f"Missing/unsafe output directory: {part}")
    required = ["synthetic_demo_examples.png", "face_regions_synthetic_observations.csv",
                "synthetic_demo_summary.json", "synthetic_demo_summary.txt"]
    for filename in required:
        p = stage / filename
        if not p.is_file() or p.is_symlink() or not p.stat().st_size:
            raise SyntheticDemoError(f"Missing/empty artifact: {filename}")
    contact = read_image(stage / "synthetic_demo_examples.png")
    if contact.shape[:2] != (math.ceil(len(records) / 2) * 315, 1360):
        raise SyntheticDemoError("Invalid contact sheet dimensions")
    if (json.loads((stage / "synthetic_demo_summary.json").read_text(encoding="utf-8"))
            != summary or (stage / "synthetic_demo_summary.txt").read_text(encoding="utf-8")
            != summary_as_text(summary)):
        raise SyntheticDemoError("JSON/TXT reports differ from validated summary")
    seen_inputs: set[str] = set()
    expected_paths: set[str] = set()
    expected_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        source_rel = rec["input"]
        if source_rel.casefold() in seen_inputs or rec["status"] != "SUCCESS":
            raise SyntheticDemoError("Repeated or failed image in complete report")
        seen_inputs.add(source_rel.casefold())
        source = INPUT_DIR / source_rel
        if (not source.is_file() or source.is_symlink()
                or not source.resolve().is_relative_to(INPUT_DIR.resolve())
                or sha256_file(source) != rec["source_sha256"]):
            raise SyntheticDemoError(f"Input altered since inference: {source_rel}")
        for key, rel in relative_outputs(source).items():
            if rec[key] != rel or rel.casefold() in expected_paths:
                raise SyntheticDemoError("Output name collision/inconsistency")
            expected_paths.add(rel.casefold())
            p = stage / rel
            if (not p.resolve().is_relative_to(stage.resolve()) or p.is_symlink()
                    or not p.is_file() or not p.stat().st_size
                    or sha256_file(p) != rec["output_sha256"][key]):
                raise SyntheticDemoError(f"Missing/modified output: {rel}")
        label = cv2.imdecode(np.fromfile(str(stage / rec["label_maps_path"]), dtype=np.uint8),
                             cv2.IMREAD_UNCHANGED)
        if label is None or label.shape != (SIZE, SIZE) or label.dtype != np.uint8:
            raise SyntheticDemoError(f"Invalid saved class-ID PNG: {source_rel}")
        label = validated_label_map(label)
        counts = np.bincount(label.ravel(), minlength=NUM_CLASSES)
        if (sum(rec["class_counts"].values()) != SIZE * SIZE
                or any(int(counts[i]) != rec["class_counts"][name]
                       for i, name in enumerate(CLASS_NAMES))
                or int(counts[1:].sum()) != rec["foreground_pixels"]
                or int(counts[0]) != rec["background_pixels"]):
            raise SyntheticDemoError("Saved label map does not match region counts")
        source_image = read_image(source)
        resized = cv2.resize(source_image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        color = read_image(stage / rec["segmentation_maps_path"])
        overlay = read_image(stage / rec["overlays_path"])
        panel = read_image(stage / rec["annotated_path"])
        if (color.shape != (SIZE, SIZE, 3)
                or overlay.shape != (SIZE, SIZE, 3)
                or panel.shape != (890, 2 * 22 + 3 * SIZE + 2 * 18, 3)
                or not np.array_equal(color, colorize(label))
                or not np.array_equal(overlay, make_overlay(resized, label))):
            raise SyntheticDemoError(f"Rendered segmentation inconsistent: {source_rel}")
        for i, name in enumerate(CLASS_NAMES):
            expected_rows[(source_rel, i)] = {
                "count": int(counts[i]),
                "name": name,
                "hash": rec["source_sha256"],
                "assoc": rec["association_iou"],
            }
    csv_path = stage / "face_regions_synthetic_observations.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise SyntheticDemoError("Unexpected observations CSV columns")
        rows = list(reader)
    if len(rows) != len(expected_rows) or summary["observation_rows"] != len(rows):
        raise SyntheticDemoError("Missing or extra class rows in observations CSV")
    seen_rows: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["source_image"], int(row["class_id"]))
        if key in seen_rows or key not in expected_rows:
            raise SyntheticDemoError("Duplicate or unknown per-class observation")
        seen_rows.add(key)
        expected = expected_rows[key]
        count = expected["count"]
        if (row["class_name"] != expected["name"]
                or row["source_sha256"] != expected["hash"]
                or row["status"] != "OK"
                or row["pixel_count_match"] != ("N/A" if key[1] == 0 else "True")
                or int(row["pixel_count"]) != count
                or (key[1] == 0 and (row["pipeline_pixel_count"] != "" or row["native_pixel_count"] != ""))
                or (key[1] != 0 and (int(row["pipeline_pixel_count"]) != count
                                     or int(row["native_pixel_count"]) != count))
                or int(row["image_width"]) != SIZE
                or int(row["image_height"]) != SIZE
                or not math.isclose(float(row["pixel_fraction"]), count / (SIZE * SIZE), abs_tol=1e-12)
                or not math.isclose(float(row["association_iou"]), expected["assoc"], abs_tol=1e-12)):
            raise SyntheticDemoError("Inconsistent per-class CSV data")
    # Ensure no untracked staged artifacts are published.
    observed_files = {p.relative_to(stage).as_posix().casefold()
                      for p in stage.rglob("*") if p.is_file()}
    required_files = {str(v).casefold() for v in required}
    if observed_files != required_files | expected_paths:
        raise SyntheticDemoError("Unexpected file in published synthetic demonstration")


def commit_stage(stage: Path, staging_root: Path, summary: dict[str, Any]) -> None:
    """Atomically swap this script's ONE owned directory; rollback on failure."""
    backup = staging_root / "previous_synthetic_demo"
    if (FINAL_DIR.exists() or FINAL_DIR.is_symlink()) and (
        FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()
    ):
        raise SyntheticDemoError("Unsafe published synthetic_demo directory")
    verify_stage(stage, summary)
    old_exists = FINAL_DIR.exists()
    published = False
    try:
        if old_exists:
            FINAL_DIR.replace(backup)
        stage.replace(FINAL_DIR)
        published = True
        verify_stage(FINAL_DIR, summary)
    except BaseException:
        try:
            if published and FINAL_DIR.exists():
                FINAL_DIR.replace(stage)
            if old_exists and backup.exists():
                backup.replace(FINAL_DIR)
        except BaseException as exc:
            raise SyntheticDemoError(
                f"Publication failed; previous output backup preserved at {backup}"
            ) from exc
        raise
    if old_exists:
        shutil.rmtree(backup)


def main() -> None:
    imports = check_project_imports()
    images = discover_images()
    if (FINAL_DIR.exists() or FINAL_DIR.is_symlink()) and (
        FINAL_DIR.is_symlink() or not FINAL_DIR.is_dir()
    ):
        raise SyntheticDemoError("Unsafe existing synthetic_demo directory")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".face_regions_synthetic_stage_", dir=RESULTS_DIR))
    stage = staging_root / "synthetic_demo"
    stage.mkdir()
    for part in OUTPUT_SUBDIRS:
        (stage / part).mkdir()
    pipeline = None
    records: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    try:
        fingerprint = provenance()
        pipeline, capture = make_pipeline()
        print("PhysioTrack Face Regions - Synthetic Qualitative Demo")
        print("Inputs: validation/synthetic_media/face_regions_media/")
        print("Outputs: validation/face_regions/results/synthetic_demo/")
        print(f"Discovered images: {len(images)}; device: {DEVICE}")
        print("Full-image box | 512 x 512 | real SegFace | NO ground truth")
        for index, source in enumerate(images, 1):
            name = source.relative_to(INPUT_DIR).as_posix()
            print(f"[{index}/{len(images)}] {name}", flush=True)
            try:
                rec, rows = process_image(source, stage, pipeline, capture)
                records.append(rec)
                class_rows.extend(rows)
                print(f"    OK: {len(rec['foreground_classes'])} foreground classes", flush=True)
            except Exception as error:
                records.append({
                    "input": name,
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                })
                print(f"    FAILED: {error}", flush=True)
        successful = [r for r in records if r["status"] == "SUCCESS"]
        failed = [r for r in records if r["status"] != "SUCCESS"]
        summary = {
            "project": "physiotrack-v2",
            "component": "Face Regions",
            "evaluation_type": "synthetic_qualitative_demonstration",
            "benchmark_evaluation": False,
            "ground_truth_used": False,
            "benchmark_accuracy_calculated": False,
            "input_directory": "validation/synthetic_media/face_regions_media",
            "output_directory": "validation/face_regions/results/synthetic_demo",
            "configuration": {
                "pipeline": "physiotrack.face.FaceAnalysis",
                "regions": "physiotrack.face.FaceRegions",
                "backend": "SegFace",
                "detector": "controlled_full_aligned_image_box",
                "face_detection_performed": False,
                "input_resolution": f"{SIZE} x {SIZE}",
                "resize_interpolation": "cv2.INTER_LINEAR",
                "device": DEVICE,
                "other_modules_disabled": True,
                "semantic_class_order": list(CLASS_NAMES),
            },
            "component_imports": imports,
            "reproducibility": fingerprint,
            "runtime_environment": {
                "python": platform.python_version(),
                "platform": platform.system(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "discovered_files": len(images),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "observation_rows": len(class_rows),
            "status": "COMPLETE" if not failed else "REJECTED_WITH_FILE_FAILURES",
            "interpretation": (
                "Real SegFace semantic class predictions on independently supplied synthetic "
                "images. No annotation-based accuracy, IoU, Dice or pixel-accuracy score "
                "is measured. The FaceAnalysis association IoU is NOT segmentation IoU. "
                "No CelebAMask-HQ input or result is accessed."
            ),
            "records": records,
        }
        if not failed:
            render_contact_sheet(stage, records)
        write_rows(stage, class_rows)
        write_reports(stage, summary, "synthetic_demo_summary")
        if failed:
            # Do not replace earlier accepted synthetic outputs, even partially.
            write_reports(RESULTS_DIR, summary, "synthetic_demo_last_attempt")
            raise SyntheticDemoError(
                f"{len(failed)} image(s) failed. Existing results preserved. "
                "See results/synthetic_demo_last_attempt.json/.txt."
            )
        verify_stage(stage, summary)
        commit_stage(stage, staging_root, summary)
        for ext in ("json", "txt"):
            try:
                (RESULTS_DIR / f"synthetic_demo_last_attempt.{ext}").unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARNING: could not remove old failure diagnostics: {exc}")
        print("\nFinished: COMPLETE")
        print(f"Images processed: {len(successful)}")
        print(f"Per-class CSV rows: {len(class_rows)}")
        print("Saved: validation/face_regions/results/synthetic_demo/")
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            if (staging_root / "previous_synthetic_demo").exists():
                print("WARNING: Previous results backup preserved in", staging_root)
            else:
                shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
