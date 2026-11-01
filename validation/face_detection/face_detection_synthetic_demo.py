"""Illustrative face-detection demonstrations on locally supplied media.

Run from any working directory after installing physiotrack-v2 in the active
Python environment. Inputs are discovered below validation/synthetic_media;
only results/face_detection/synthetic_demo is replaced by a fully successful,
verified run. Both JSON and TXT summaries are written before publication.
No benchmark predictions, ground truth, or evaluation metrics are used.
"""

from __future__ import annotations

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

import cv2
import numpy as np

from physiotrack.face import Face, FaceAnalysis, FaceAnalysisConfig


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = VALIDATION_DIR.parent
INPUT_DIR = VALIDATION_DIR / "synthetic_media" / "face_detection_media"
RESULTS_DIR = SCRIPT_DIR / "results"
FINAL_DIR = RESULTS_DIR / "synthetic_demo"

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".dib", ".tif", ".tiff", ".webp",
    ".avif", ".jp2", ".pbm", ".pgm", ".ppm",
})
VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".m4v", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg",
    ".wmv", ".flv", ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".vob",
})

DEVICE = "cpu"
DETECTOR_CONFIDENCE = 0.001
DISPLAY_CONFIDENCE = 0.25
NMS_IOU = 0.45
MAX_DETECTIONS = 10000
VIDEO_CODECS = ("mp4v", "avc1")


class MediaProcessingError(RuntimeError):
    """A media read, render, or output verification failure."""


def check_project_imports() -> dict[str, str]:
    """Reject imports from any unrelated installed physiotrack package."""
    expected = (REPOSITORY_ROOT / "src" / "physiotrack").resolve()
    paths: dict[str, str] = {}
    for cls in (Face, FaceAnalysis, FaceAnalysisConfig):
        source = Path(inspect.getfile(cls)).resolve()
        if not source.is_relative_to(expected):
            raise RuntimeError(
                f"{cls.__name__} was not imported from this repository's src/physiotrack. "
                "Install the current physiotrack-v2 checkout in editable mode."
            )
        paths[cls.__name__] = source.relative_to(REPOSITORY_ROOT).as_posix()
    return paths


def find_media() -> list[tuple[Path, str]]:
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            "Input directory does not exist: validation/synthetic_media/face_detection_media/"
        )
    files: list[tuple[Path, str]] = []
    for path in INPUT_DIR.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        extension = path.suffix.lower()
        if extension in IMAGE_EXTENSIONS:
            files.append((path, "image"))
        elif extension in VIDEO_EXTENSIONS:
            files.append((path, "video"))
    files.sort(key=lambda item: (item[0].relative_to(INPUT_DIR).as_posix().casefold(),
                                item[0].relative_to(INPUT_DIR).as_posix()))
    if not files:
        raise RuntimeError(
            "No supported images or videos found in validation/synthetic_media/face_detection_media/."
        )
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_version(package: str) -> str | None:
    """Record installed package versions without requiring optional packages."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def summary_as_text(summary: dict) -> str:
    """Render every summary field, including nested file metadata, as plain text.

    The TXT and JSON come from the same in-memory dictionary. Do not infer
    accuracy scores from this qualitative demonstration.
    """
    lines = [
        "PhysioTrack-v2 Face Detection - Synthetic Qualitative Demonstration",
        "=" * 68,
        "",
        "Human-readable rendering of synthetic_demo_summary.json",
        "No ground truth or detection-accuracy benchmark is used.",
        "",
    ]

    def walk(value: object, indent: int = 0, key: str | None = None) -> None:
        prefix = " " * indent
        if isinstance(value, dict):
            if key is not None:
                lines.append(f"{prefix}{key}:")
                indent += 2
                prefix = " " * indent
            if not value:
                lines.append(f"{prefix}(empty)")
            for name, entry in value.items():
                walk(entry, indent, str(name))
        elif isinstance(value, list):
            if key is not None:
                lines.append(f"{prefix}{key}: ({len(value)} items)")
                indent += 2
                prefix = " " * indent
            if not value:
                lines.append(f"{prefix}(none)")
            for index, entry in enumerate(value, start=1):
                walk(entry, indent, f"item_{index:03d}")
        else:
            rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
            lines.append(f"{prefix}{key}: {rendered}")

    walk(summary)
    return "\n".join(lines) + "\n"


def output_relative_path(source: Path, media_type: str) -> Path:
    """Keep directory nesting and add a stable identifier to avoid name collisions."""
    relative = source.relative_to(INPUT_DIR)
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
    safe_basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.name)[:92].strip(". ")
    if not safe_basename:
        safe_basename = "media"
    output_name = f"{safe_basename}_{digest}_annotated"
    if media_type == "image":
        return Path("images") / relative.parent / f"{output_name}.png"
    return Path("videos") / relative.parent / f"{output_name}.mp4"


def image_from_path(path: Path) -> np.ndarray:
    """Read via bytes to support Unicode paths on Windows OpenCV builds."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise MediaProcessingError("Unable to decode the image") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise MediaProcessingError("Unable to decode the image as a three-channel BGR frame")
    return image


def save_png(image: np.ndarray, path: Path) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise MediaProcessingError("OpenCV could not encode the annotated PNG image")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def detection_only_pipeline() -> FaceAnalysis:
    config = FaceAnalysisConfig(
        tracking=False,
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
        detector=detector, config=config, device=DEVICE, verbose=False,
    )
    if pipeline.detector is None or any(
        getattr(pipeline, attr) is not None
        for attr in (
            "tracker", "orientation", "landmarks", "quality", "eyes", "blink",
            "gaze", "gaze_estimation", "mouth", "mouth_motion", "emotion",
            "regions", "temporal",
        )
    ):
        raise RuntimeError("The instantiated FaceAnalysis pipeline is not detection-only")
    return pipeline


def annotate_frame(frame: np.ndarray, pipeline: FaceAnalysis) -> tuple[np.ndarray, dict[str, int]]:
    """Draw real detector results; only presentation filtering occurs here."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise MediaProcessingError("The decoded frame is not a three-channel image")
    result = pipeline.predict(frame)
    faces = list(result)
    output = frame.copy()
    height, width = frame.shape[:2]
    shown = 0
    invalid = 0
    font_scale = max(0.38, min(0.85, min(height, width) / 900.0))
    line_width = 1 if min(height, width) < 450 else 2

    for face in faces:
        try:
            coords = np.asarray(face.box, dtype=np.float64).reshape(-1)
            if coords.size != 4 or not np.all(np.isfinite(coords)):
                invalid += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            if not (x2 > x1 and y2 > y1):
                invalid += 1
                continue
            confidence = face.confidence
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                    invalid += 1
                    continue
                if confidence < DISPLAY_CONFIDENCE:
                    continue
            # Clip for *drawing only*. Never change the model's underlying results.
            left = max(0, min(width - 1, int(round(x1))))
            top = max(0, min(height - 1, int(round(y1))))
            right = max(0, min(width - 1, int(round(x2))))
            bottom = max(0, min(height - 1, int(round(y2))))
            if right <= left or bottom <= top:
                invalid += 1
                continue
        except (TypeError, ValueError, OverflowError) as exc:
            invalid += 1
            continue

        color = (55, 215, 70)
        cv2.rectangle(output, (left, top), (right, bottom), color, line_width)
        label = f"face {confidence:.2f}" if confidence is not None else "face"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        label_x = max(0, min(left, max(0, width - label_width - 6)))
        label_top = top - label_height - baseline - 6
        if label_top < 0:
            label_top = min(height - 1, bottom + 2)
        label_bottom = min(height - 1, label_top + label_height + baseline + 6)
        cv2.rectangle(
            output, (label_x, label_top),
            (min(width - 1, label_x + label_width + 5), label_bottom),
            color, -1,
        )
        baseline_y = min(height - 1, label_top + label_height + 3)
        cv2.putText(
            output, label, (label_x + 2, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
        shown += 1

    return output, {
        "raw_detections": len(faces),
        "displayed_detections": shown,
        "invalid_boxes_or_confidence": invalid,
    }


def process_image(source: Path, target: Path, pipeline: FaceAnalysis) -> dict:
    image = image_from_path(source)
    annotated, counts = annotate_frame(image, pipeline)
    save_png(annotated, target)
    decoded = image_from_path(target)
    if decoded.shape[:2] != image.shape[:2]:
        raise MediaProcessingError("Annotated image dimensions differ from source")
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "frames_processed": 1,
        **counts,
    }


def open_video_writer(path: Path, width: int, height: int, fps: float) -> tuple[cv2.VideoWriter, str]:
    for codec in VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
        path.unlink(missing_ok=True)
    raise MediaProcessingError("No supported MP4 video encoder is available in OpenCV")


def verify_video(path: Path, width: int, height: int, fps: float, expected_frames: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise MediaProcessingError("Saved annotated video cannot be opened")
    try:
        actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if (actual_width, actual_height) != (width, height):
            raise MediaProcessingError("Encoded video dimensions differ from source")
        if not math.isfinite(actual_fps) or abs(actual_fps - fps) > max(0.05, fps * 0.01):
            raise MediaProcessingError("Encoded video FPS differs from source beyond tolerance")
        actual_frames = 0
        while True:
            success, decoded = capture.read()
            if not success:
                break
            if decoded is None or decoded.shape[:2] != (height, width):
                raise MediaProcessingError("An encoded video frame is unreadable or has changed size")
            actual_frames += 1
        if actual_frames != expected_frames:
            raise MediaProcessingError("Decoded output video frame count differs from processed frame count")
        return {"output_fps": actual_fps, "verified_output_frames": actual_frames}
    finally:
        capture.release()


def process_video(source: Path, target: Path, pipeline: FaceAnalysis, working_dir: Path,
                  index: int) -> dict:
    """Process in order, then verify the *entire* encoded video before publication."""
    capture = cv2.VideoCapture(str(source))
    temporary_input: Path | None = None
    if not capture.isOpened():
        # Some Windows OpenCV backends cannot open Unicode video paths.
        capture.release()
        temporary_input = working_dir / f"input_{index:06d}{source.suffix.lower()}"
        shutil.copyfile(source, temporary_input)
        capture = cv2.VideoCapture(str(temporary_input))
    if not capture.isOpened():
        capture.release()
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)
        raise MediaProcessingError("OpenCV cannot open or decode the input video")
    temporary_video = working_dir / f"video_{index:06d}.mp4"
    writer = None
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width <= 0 or height <= 0:
            raise MediaProcessingError("Input video has invalid frame dimensions")
        if not math.isfinite(fps) or fps <= 0:
            raise MediaProcessingError("Input video has no valid FPS metadata")
        writer, codec = open_video_writer(temporary_video, width, height, fps)
        processed = 0
        total_raw = 0
        total_shown = 0
        total_invalid = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise MediaProcessingError("Input video contains an unreadable or size-changing frame")
            annotated, counts = annotate_frame(frame, pipeline)
            writer.write(annotated)
            processed += 1
            total_raw += counts["raw_detections"]
            total_shown += counts["displayed_detections"]
            total_invalid += counts["invalid_boxes_or_confidence"]
            if processed % 250 == 0:
                print(f"    Processed {processed} video frames...", flush=True)
        if processed == 0:
            raise MediaProcessingError("Input video contains no decodable frames")
        # OpenCV frame-count metadata can be approximate. Large disagreements are suspicious.
        if declared_frames > 0 and abs(declared_frames - processed) > 2:
            raise MediaProcessingError("Input video ended before the declared frame count")
        writer.release()
        writer = None
        checked = verify_video(temporary_video, width, height, fps, processed)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_video.replace(target)
        return {
            "width": width,
            "height": height,
            "source_fps": fps,
            "source_declared_frames": declared_frames if declared_frames > 0 else None,
            "video_codec": codec,
            "frames_processed": processed,
            "raw_detections": total_raw,
            "displayed_detections": total_shown,
            "invalid_boxes_or_confidence": total_invalid,
            **checked,
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        temporary_video.unlink(missing_ok=True)
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)


def verify_staged_outputs(stage: Path, summary: dict) -> None:
    """Reject incomplete runs and verify ALL staged outputs before publication."""
    if summary["status"] != "COMPLETE" or summary["failed_files"] != 0:
        raise RuntimeError("Incomplete media processing: old synthetic_demo remains untouched")
    if summary["successful_files"] != summary["discovered_files"]:
        raise RuntimeError("Not every discovered input was processed successfully")
    if not summary["files"] or len(summary["files"]) != summary["discovered_files"]:
        raise RuntimeError("Missing or inconsistent per-input output records")

    successes = 0
    seen_outputs: set[str] = set()
    totals = {
        "total_processed_frames": 0,
        "total_raw_detections": 0,
        "total_displayed_detections": 0,
        "total_invalid_boxes_or_confidence": 0,
    }
    for record in summary["files"]:
        if record["status"] != "SUCCESS":
            raise RuntimeError("A failed input is present in the staged summary")
        successes += 1
        relative_output = record["output"]
        if not isinstance(relative_output, str) or relative_output in seen_outputs:
            raise RuntimeError("Missing or duplicate staged output path")
        seen_outputs.add(relative_output)
        output = stage / relative_output
        if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
            raise RuntimeError("A staged output is missing, symbolic, or empty")
        if record["media_type"] == "image":
            image = image_from_path(output)
            if image.shape[:2] != (record["height"], record["width"]):
                raise RuntimeError("A staged image has incorrect dimensions")
        elif record["media_type"] == "video":
            verify_video(
                output, record["width"], record["height"],
                record["source_fps"], record["frames_processed"],
            )
        else:
            raise RuntimeError("Unexpected staged media type")
        if sha256_file(output) != record["output_sha256"]:
            raise RuntimeError("A staged media output differs from its SHA-256")
        totals["total_processed_frames"] += record["frames_processed"]
        totals["total_raw_detections"] += record["raw_detections"]
        totals["total_displayed_detections"] += record["displayed_detections"]
        totals["total_invalid_boxes_or_confidence"] += record["invalid_boxes_or_confidence"]

    if successes != summary["successful_files"]:
        raise RuntimeError("Staged file accounting is inconsistent")
    for field, count in totals.items():
        if summary[field] != count:
            raise RuntimeError(f"Staged summary count mismatch: {field}")

    json_file = stage / "synthetic_demo_summary.json"
    txt_file = stage / "synthetic_demo_summary.txt"
    if not json_file.is_file() or not txt_file.is_file():
        raise RuntimeError("Both JSON and TXT summaries are required")
    with json_file.open("r", encoding="utf-8") as handle:
        on_disk = json.load(handle)
    if on_disk != summary:
        raise RuntimeError("The staged JSON summary failed verification")
    if txt_file.read_text(encoding="utf-8") != summary_as_text(summary):
        raise RuntimeError("The staged TXT summary failed verification")

    allowed = seen_outputs | {json_file.name, txt_file.name}
    produced = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*") if path.is_file()
    }
    if produced != allowed:
        raise RuntimeError("The staged output set differs from the verified manifest")

    # Media directories are determined by actual successful outputs, not by
    # a fixed template. This keeps image-only and video-only runs free of
    # empty folders that Git cannot preserve.
    expected_media_dirs = {Path(name).parts[0] for name in seen_outputs}
    actual_media_dirs = {path.name for path in stage.iterdir() if path.is_dir()}
    if actual_media_dirs != expected_media_dirs:
        raise RuntimeError("Unexpected, missing, or empty media-type directory")
    if any(not any(directory.iterdir()) for directory in stage.rglob("*") if directory.is_dir()):
        raise RuntimeError("An empty media output directory was produced")


def commit_directory(stage: Path, staging_root: Path) -> None:
    """Publish a verified complete run; restore previous results on normal errors.

    Only this script's synthetic_demo directory is replaced. Staging and backup
    are on the same filesystem as the destination. Power-loss atomicity is NOT
    claimed by this two-rename procedure.
    """
    backup = staging_root / "previous_synthetic_demo"
    backed_up = False
    if FINAL_DIR.exists():
        if not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink():
            raise RuntimeError("synthetic_demo output path is not a normal directory")
        FINAL_DIR.replace(backup)
        backed_up = True
    try:
        stage.replace(FINAL_DIR)
    except Exception as publish_error:
        if backed_up:
            try:
                backup.replace(FINAL_DIR)
            except Exception as restore_error:
                # The outer finally must NOT remove this last copy.
                raise RuntimeError(
                    f"Publication failed AND automatic rollback failed. "
                    f"Previous outputs remain at: {backup}"
                ) from restore_error
        raise publish_error


def main() -> None:
    sources = check_project_imports()
    media = find_media()
    if FINAL_DIR.exists() and (not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink()):
        raise RuntimeError("The synthetic_demo output path is not a normal directory")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = detection_only_pipeline()
    print("PhysioTrack face detection synthetic demo")
    print("Media files discovered:", len(media))
    print("Inputs: validation/synthetic_media/face_detection_media/")
    print("Outputs: validation/face_detection/results/synthetic_demo/")

    staging_root = Path(tempfile.mkdtemp(prefix=".synthetic_demo_staging_", dir=RESULTS_DIR))
    staged_outputs = staging_root / "synthetic_demo"
    staged_outputs.mkdir()
    # save_png/process_video create their own output parents only when needed.
    # No empty images/ or videos/ directories are published.
    intermediate = staging_root / "video_work"
    if any(media_type == "video" for _, media_type in media):
        intermediate.mkdir()
    try:
        records: list[dict] = []
        output_names: set[str] = set()
        for index, (source, media_type) in enumerate(media, start=1):
            relative = source.relative_to(INPUT_DIR).as_posix()
            relative_output = output_relative_path(source, media_type).as_posix()
            if relative_output.casefold() in output_names:
                raise RuntimeError("Two inputs resolve to the same output name")
            output_names.add(relative_output.casefold())
            record: dict = {
                "input": relative,
                "media_type": media_type,
                "status": "FAILED",
                "output": None,
            }
            print(f"[{index}/{len(media)}] {media_type}: {relative}", flush=True)
            try:
                record["source_sha256"] = sha256_file(source)
                target = staged_outputs / relative_output
                if media_type == "image":
                    measurements = process_image(source, target, pipeline)
                else:
                    measurements = process_video(source, target, pipeline, intermediate, index)
                if sha256_file(source) != record["source_sha256"]:
                    raise MediaProcessingError("Input file changed while being processed")
                record.update(measurements)
                record["output_sha256"] = sha256_file(target)
                record["status"] = "SUCCESS"
                record["output"] = relative_output
                print(f"    OK: {measurements['raw_detections']} raw, "
                      f"{measurements['displayed_detections']} displayed detections")
            except Exception as exc:
                # Failure is recorded; no failed input gets a published partial output.
                record["error_type"] = type(exc).__name__
                record["error"] = (
                    str(exc) if isinstance(exc, MediaProcessingError)
                    else "Media processing failed; inspect terminal diagnostics."
                )
                print(f"    FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
                traceback.print_exc()
                if record["output"] is None:
                    (staged_outputs / relative_output).unlink(missing_ok=True)
            records.append(record)

        success = [r for r in records if r["status"] == "SUCCESS"]
        failure = [r for r in records if r["status"] == "FAILED"]
        summary = {
            "type": "synthetic_qualitative_face_detection_demonstration",
            "benchmark_evaluation": False,
            "project": "physiotrack-v2",
            "input_directory": "validation/synthetic_media/face_detection_media",
            "output_directory": "validation/face_detection/results/synthetic_demo",
            "component_imports": sources,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "component_source_sha256": {
                name: sha256_file(REPOSITORY_ROOT / relative)
                for name, relative in sources.items()
            },
            "detector": {
                "class": "physiotrack.face.Face",
                "pipeline": "physiotrack.face.FaceAnalysis",
                "device": DEVICE,
                "confidence_threshold": DETECTOR_CONFIDENCE,
                "display_confidence_threshold": DISPLAY_CONFIDENCE,
                "nms_iou": NMS_IOU,
                "max_det": MAX_DETECTIONS,
                "tracking": False,
            },
            "environment": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "torch": library_version("torch"),
                "ultralytics": library_version("ultralytics"),
                "platform": platform.system(),
            },
            "video_note": "Independent per-frame detection; frame detections are not unique people. "
                          "MP4 encoding does not preserve source audio or variable frame timing.",
            "status": "COMPLETE" if not failure else "COMPLETE_WITH_FILE_FAILURES",
            "discovered_files": len(records),
            "successful_files": len(success),
            "failed_files": len(failure),
            "total_processed_frames": sum(r["frames_processed"] for r in success),
            "total_raw_detections": sum(r["raw_detections"] for r in success),
            "total_displayed_detections": sum(r["displayed_detections"] for r in success),
            "total_invalid_boxes_or_confidence": sum(
                r["invalid_boxes_or_confidence"] for r in success
            ),
            "files": records,
        }
        with (staged_outputs / "synthetic_demo_summary.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        with (staged_outputs / "synthetic_demo_summary.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(summary_as_text(summary))

        if failure:
            # Never replace previously accepted results with a partial run.
            # Diagnostics remain in terminal output; the existing final folder
            # and its JSON/TXT summaries are not touched.
            raise RuntimeError(
                f"{len(failure)} of {len(records)} inputs failed. "
                "Previously accepted synthetic_demo results were preserved."
            )
        verify_staged_outputs(staged_outputs, summary)
        commit_directory(staged_outputs, staging_root)
        print("\nDemonstration complete:", summary["status"])
        print("Successful files:", summary["successful_files"])
        print("Failed files:", summary["failed_files"])
        print("Raw detections:", summary["total_raw_detections"])
        print("Displayed detections:", summary["total_displayed_detections"])
        print("Saved summaries: validation/face_detection/results/synthetic_demo/")
        print("  synthetic_demo_summary.json")
        print("  synthetic_demo_summary.txt")
    finally:
        try:
            if hasattr(pipeline, "close"):
                pipeline.close()
        finally:
            backup = staging_root / "previous_synthetic_demo"
            # If rollback itself failed, keep the only surviving old results.
            if backup.exists() and not FINAL_DIR.exists():
                print(f"WARNING: Old outputs preserved at {backup}", file=sys.stderr)
            else:
                shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    main()
