from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.results import Instance, Result


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "300W"
)

RESULTS_DIR = (
    VALIDATION_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "component_execution"
)

RESULTS_CSV_PATH = (
    OUTPUT_DIR
    / "face_landmarks_component_results.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "face_landmarks_component_summary.json"
)

DATASETS = [
    (
        "Indoor",
        DATASET_ROOT
        / "01_Indoor",
    ),
    (
        "Outdoor",
        DATASET_ROOT
        / "02_Outdoor",
    ),
]

EXPECTED_IMAGES_PER_SPLIT = 300
EXPECTED_TOTAL_IMAGES = 600
EXPECTED_LANDMARKS_PER_FACE = 478
FACE_BOX_PADDING = 0.20

FIELDNAMES = [
    "split",
    "image_index",
    "image",
    "image_width",
    "image_height",
    "face_box_source",
    "face_box_x1",
    "face_box_y1",
    "face_box_x2",
    "face_box_y2",
    "face_box_width",
    "face_box_height",
    "face_box_area",
    "pipeline_landmarks_available",
    "pipeline_landmark_count",
    "landmark_index",
    "landmarks_in_face",
    "x_normalized",
    "y_normalized",
    "z_normalized",
    "x_pixel",
    "y_pixel",
    "within_image_bounds",
    "status",
    "failure_reason",
]

UNRELATED_PIPELINE_ATTRIBUTES = (
    "tracker",
    "orientation",
    "quality",
    "eyes",
    "blink",
    "gaze",
    "gaze_estimation",
    "mouth",
    "mouth_motion",
    "emotion",
    "regions",
    "temporal",
)


class ControlledFaceBoxDetector:
    """Provide one externally defined face box to the real FaceAnalysis path."""

    def __init__(self):
        self._box = None

    def set_box(
        self,
        box,
    ) -> None:
        self._box = np.asarray(
            box,
            dtype=float,
        )

    def predict(
        self,
        frame,
    ) -> Result:
        if self._box is None:
            raise RuntimeError(
                "Controlled face box was not set before prediction."
            )

        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=None,
                    box=self._box.copy(),
                    confidence=1.0,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


class LandmarkCapture:
    """Capture the native landmark list returned inside FaceAnalysis."""

    def __init__(
        self,
        landmarker,
    ):
        self.landmarker = landmarker
        self.last_landmarks = None

    def predict_face(
        self,
        frame,
        box,
    ):
        self.last_landmarks = (
            self.landmarker.predict_face(
                frame,
                box,
            )
        )

        return self.last_landmarks

    def reset_capture(self) -> None:
        self.last_landmarks = None

    def close(self) -> None:
        if hasattr(
            self.landmarker,
            "close",
        ):
            self.landmarker.close()


def finite_numeric(
    value: Any,
) -> bool:
    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def resolve_model_path() -> Path:
    """Resolve the MediaPipe face-landmarker model portably."""
    candidates = []

    env_model = os.environ.get(
        "PHYSIOTRACK_FACE_LANDMARKER_MODEL"
    )

    if env_model:
        candidates.append(
            Path(env_model)
        )

    local_app_data = os.environ.get(
        "LOCALAPPDATA"
    )

    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "physiotrack"
            / "weights"
            / "mediapipe"
            / "face_landmarker.task"
        )

    candidates.append(
        PROJECT_ROOT
        / "weights"
        / "mediapipe"
        / "face_landmarker.task"
    )

    candidates.append(
        Path.home()
        / ".cache"
        / "physiotrack"
        / "weights"
        / "mediapipe"
        / "face_landmarker.task"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not locate face_landmarker.task.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Set PHYSIOTRACK_FACE_LANDMARKER_MODEL "
        "to an explicit model path if needed."
    )


def load_300w_points(
    path: Path,
) -> np.ndarray:
    """Load the 68 one-based 300-W points and convert them to zero-based."""
    points = []
    inside = False

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line == "{":
                inside = True
                continue

            if line == "}":
                break

            if inside and line:
                x, y = map(
                    float,
                    line.split(),
                )

                points.append(
                    (
                        x,
                        y,
                    )
                )

    points_array = np.asarray(
        points,
        dtype=float,
    )

    if len(points_array) != 68:
        raise RuntimeError(
            f"Expected 68 points in {path.name}, "
            f"found {len(points_array)}"
        )

    points_array -= 1.0

    return points_array


def make_gt_face_box(
    points: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Create the same GT-derived padded input box used by the benchmark."""
    x_min = float(
        points[:, 0].min()
    )

    y_min = float(
        points[:, 1].min()
    )

    x_max = float(
        points[:, 0].max()
    )

    y_max = float(
        points[:, 1].max()
    )

    face_width = (
        x_max
        - x_min
    )

    face_height = (
        y_max
        - y_min
    )

    pad_x = (
        FACE_BOX_PADDING
        * face_width
    )

    pad_y = (
        FACE_BOX_PADDING
        * face_height
    )

    x1 = max(
        0.0,
        x_min - pad_x,
    )

    y1 = max(
        0.0,
        y_min - pad_y,
    )

    x2 = min(
        float(
            image_width - 1
        ),
        x_max + pad_x,
    )

    y2 = min(
        float(
            image_height - 1
        ),
        y_max + pad_y,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise RuntimeError(
            "GT-derived face box is invalid."
        )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


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


def assert_landmark_only_pipeline(
    pipeline: FaceAnalysis,
    detector: ControlledFaceBoxDetector,
) -> None:
    if pipeline.detector is not detector:
        raise RuntimeError(
            "FaceAnalysis is not using the controlled face-box input adapter."
        )

    if pipeline.landmarks is None:
        raise RuntimeError(
            "Face landmarks component is not initialized."
        )

    unexpected_enabled = [
        name
        for name in UNRELATED_PIPELINE_ATTRIBUTES
        if getattr(
            pipeline,
            name,
        ) is not None
    ]

    if unexpected_enabled:
        raise RuntimeError(
            "Unrelated FaceAnalysis components are enabled: "
            + ", ".join(
                unexpected_enabled
            )
        )


def preflight() -> tuple[
    list[tuple[str, Path, Path]],
    Path,
]:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"300-W dataset root not found: {DATASET_ROOT}"
        )

    samples = []

    for split, folder in DATASETS:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing 300-W split directory: {folder}"
            )

        images = sorted(
            folder.glob(
                "*.png"
            )
        )

        annotations = sorted(
            folder.glob(
                "*.pts"
            )
        )

        if len(images) != EXPECTED_IMAGES_PER_SPLIT:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_IMAGES_PER_SPLIT} images, "
                f"found {len(images)}"
            )

        if len(annotations) != EXPECTED_IMAGES_PER_SPLIT:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_IMAGES_PER_SPLIT} annotations, "
                f"found {len(annotations)}"
            )

        image_stems = {
            path.stem
            for path in images
        }

        annotation_stems = {
            path.stem
            for path in annotations
        }

        if image_stems != annotation_stems:
            missing_annotations = sorted(
                image_stems
                - annotation_stems
            )

            missing_images = sorted(
                annotation_stems
                - image_stems
            )

            raise RuntimeError(
                f"{split}: image/annotation mismatch. "
                f"Missing annotations: {missing_annotations[:5]}; "
                f"missing images: {missing_images[:5]}"
            )

        for image_path in images:
            pts_path = image_path.with_suffix(
                ".pts"
            )

            try:
                with image_path.open(
                    "rb"
                ) as file:
                    if not file.read(1):
                        raise RuntimeError(
                            f"Empty image file: {image_path}"
                        )
            except OSError as exc:
                raise RuntimeError(
                    f"Unreadable image file: {image_path}"
                ) from exc

            load_300w_points(
                pts_path
            )

            samples.append(
                (
                    split,
                    image_path,
                    pts_path,
                )
            )

    if len(samples) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Unexpected total 300-W sample count: "
            f"{len(samples)} "
            f"(expected {EXPECTED_TOTAL_IMAGES})"
        )

    model_path = resolve_model_path()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probe_path = (
        OUTPUT_DIR
        / ".write_probe"
    )

    try:
        with probe_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "preflight"
            )
    finally:
        if probe_path.exists():
            probe_path.unlink()

    return (
        samples,
        model_path,
    )


def blank_landmark_row(
    split: str,
    image_index: int,
    image_path: Path,
    width: int,
    height: int,
    face_box: list[float],
    pipeline_available: bool,
    pipeline_count: int,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    x1, y1, x2, y2 = map(
        float,
        face_box,
    )

    box_width = (
        x2
        - x1
    )

    box_height = (
        y2
        - y1
    )

    return {
        "split": split,
        "image_index": image_index,
        "image": image_path.name,
        "image_width": width,
        "image_height": height,
        "face_box_source": "GT_DERIVED_PADDED_20_PERCENT",
        "face_box_x1": x1,
        "face_box_y1": y1,
        "face_box_x2": x2,
        "face_box_y2": y2,
        "face_box_width": box_width,
        "face_box_height": box_height,
        "face_box_area": box_width * box_height,
        "pipeline_landmarks_available": pipeline_available,
        "pipeline_landmark_count": pipeline_count,
        "landmark_index": "",
        "landmarks_in_face": "",
        "x_normalized": "",
        "y_normalized": "",
        "z_normalized": "",
        "x_pixel": "",
        "y_pixel": "",
        "within_image_bounds": "",
        "status": status,
        "failure_reason": failure_reason,
    }


def landmark_row(
    split: str,
    image_index: int,
    image_path: Path,
    width: int,
    height: int,
    face_box: list[float],
    landmark_index: int,
    landmark,
    landmarks_in_face: int,
    pipeline_available: bool,
    pipeline_count: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = map(
        float,
        face_box,
    )

    box_width = (
        x2
        - x1
    )

    box_height = (
        y2
        - y1
    )

    x_normalized = float(
        landmark.x
    )

    y_normalized = float(
        landmark.y
    )

    z_normalized = float(
        landmark.z
    )

    x_pixel = (
        x_normalized
        * width
    )

    y_pixel = (
        y_normalized
        * height
    )

    within_image_bounds = (
        0.0
        <= x_pixel
        < float(width)
        and 0.0
        <= y_pixel
        < float(height)
    )

    return {
        "split": split,
        "image_index": image_index,
        "image": image_path.name,
        "image_width": width,
        "image_height": height,
        "face_box_source": "GT_DERIVED_PADDED_20_PERCENT",
        "face_box_x1": x1,
        "face_box_y1": y1,
        "face_box_x2": x2,
        "face_box_y2": y2,
        "face_box_width": box_width,
        "face_box_height": box_height,
        "face_box_area": box_width * box_height,
        "pipeline_landmarks_available": pipeline_available,
        "pipeline_landmark_count": pipeline_count,
        "landmark_index": landmark_index,
        "landmarks_in_face": landmarks_in_face,
        "x_normalized": x_normalized,
        "y_normalized": y_normalized,
        "z_normalized": z_normalized,
        "x_pixel": x_pixel,
        "y_pixel": y_pixel,
        "within_image_bounds": within_image_bounds,
        "status": "OK",
        "failure_reason": "",
    }


def validate_staged_results(
    csv_path: Path,
    summary_path: Path,
) -> None:
    if not csv_path.is_file():
        raise RuntimeError(
            "Staged component-result CSV was not created."
        )

    if not summary_path.is_file():
        raise RuntimeError(
            "Staged component summary JSON was not created."
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    if summary.get(
        "expected_total_images"
    ) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Staged summary has an incorrect expected image count."
        )

    if summary.get(
        "processed_images"
    ) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Staged summary does not account for all 600 images."
        )

    image_records = {}
    total_rows = 0
    native_observations = 0

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(
                "Staged component-result CSV schema is incorrect."
            )

        for row in reader:
            total_rows += 1

            split = row[
                "split"
            ]

            image = row[
                "image"
            ]

            key = (
                split,
                image,
            )

            record = image_records.setdefault(
                key,
                {
                    "status": None,
                    "rows": 0,
                    "indices": set(),
                },
            )

            status = row[
                "status"
            ]

            if record[
                "status"
            ] is None:
                record[
                    "status"
                ] = status
            elif record[
                "status"
            ] != status:
                raise RuntimeError(
                    "One image has mixed status values in staged results: "
                    f"{split}/{image}"
                )

            record[
                "rows"
            ] += 1

            width = int(
                row[
                    "image_width"
                ]
            )

            height = int(
                row[
                    "image_height"
                ]
            )

            if width <= 0 or height <= 0:
                raise RuntimeError(
                    "Staged results contain invalid image dimensions."
                )

            x1 = float(
                row[
                    "face_box_x1"
                ]
            )

            y1 = float(
                row[
                    "face_box_y1"
                ]
            )

            x2 = float(
                row[
                    "face_box_x2"
                ]
            )

            y2 = float(
                row[
                    "face_box_y2"
                ]
            )

            box_width = float(
                row[
                    "face_box_width"
                ]
            )

            box_height = float(
                row[
                    "face_box_height"
                ]
            )

            box_area = float(
                row[
                    "face_box_area"
                ]
            )

            if not all(
                finite_numeric(
                    value
                )
                for value in (
                    x1,
                    y1,
                    x2,
                    y2,
                    box_width,
                    box_height,
                    box_area,
                )
            ):
                raise RuntimeError(
                    "Staged results contain non-finite face-box values."
                )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                raise RuntimeError(
                    "Staged results contain an invalid face box."
                )

            if not math.isclose(
                box_width,
                x2 - x1,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Face-box width formula is inconsistent."
                )

            if not math.isclose(
                box_height,
                y2 - y1,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Face-box height formula is inconsistent."
                )

            if not math.isclose(
                box_area,
                box_width * box_height,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Face-box area formula is inconsistent."
                )

            if status == "OK":
                landmark_index = int(
                    row[
                        "landmark_index"
                    ]
                )

                if landmark_index in record[
                    "indices"
                ]:
                    raise RuntimeError(
                        "Duplicate landmark index found for one image: "
                        f"{split}/{image}/{landmark_index}"
                    )

                record[
                    "indices"
                ].add(
                    landmark_index
                )

                if int(
                    row[
                        "landmarks_in_face"
                    ]
                ) != EXPECTED_LANDMARKS_PER_FACE:
                    raise RuntimeError(
                        "Successful staged row reports an unexpected "
                        "landmark count."
                    )

                if row[
                    "pipeline_landmarks_available"
                ].strip().lower() != "true":
                    raise RuntimeError(
                        "Successful staged row reports landmarks unavailable."
                    )

                if int(
                    row[
                        "pipeline_landmark_count"
                    ]
                ) != EXPECTED_LANDMARKS_PER_FACE:
                    raise RuntimeError(
                        "Successful staged row has an inconsistent pipeline count."
                    )

                x_normalized = float(
                    row[
                        "x_normalized"
                    ]
                )

                y_normalized = float(
                    row[
                        "y_normalized"
                    ]
                )

                z_normalized = float(
                    row[
                        "z_normalized"
                    ]
                )

                x_pixel = float(
                    row[
                        "x_pixel"
                    ]
                )

                y_pixel = float(
                    row[
                        "y_pixel"
                    ]
                )

                if not all(
                    finite_numeric(
                        value
                    )
                    for value in (
                        x_normalized,
                        y_normalized,
                        z_normalized,
                        x_pixel,
                        y_pixel,
                    )
                ):
                    raise RuntimeError(
                        "Successful staged landmark row contains "
                        "non-finite coordinates."
                    )

                if not math.isclose(
                    x_pixel,
                    x_normalized * width,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        "Staged x-pixel coordinate is inconsistent."
                    )

                if not math.isclose(
                    y_pixel,
                    y_normalized * height,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError(
                        "Staged y-pixel coordinate is inconsistent."
                    )

                native_observations += 1

            elif status in {
                "NO_LANDMARKS",
                "EXECUTION_FAILED",
            }:
                if record[
                    "rows"
                ] > 1:
                    raise RuntimeError(
                        "Non-success image has more than one staged row: "
                        f"{split}/{image}"
                    )

                for field in (
                    "landmark_index",
                    "landmarks_in_face",
                    "x_normalized",
                    "y_normalized",
                    "z_normalized",
                    "x_pixel",
                    "y_pixel",
                    "within_image_bounds",
                ):
                    if row[
                        field
                    ].strip():
                        raise RuntimeError(
                            "Non-success staged row contains unexpected "
                            f"landmark data in {field}."
                        )

            else:
                raise RuntimeError(
                    f"Unexpected staged status: {status}"
                )

    if len(image_records) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            "Staged CSV does not represent exactly 600 unique images."
        )

    split_counts = {
        "Indoor": 0,
        "Outdoor": 0,
    }

    successful_images = 0
    no_landmark_images = 0
    execution_failed_images = 0

    for (
        split,
        image,
    ), record in image_records.items():
        if split not in split_counts:
            raise RuntimeError(
                f"Unexpected split in staged results: {split}"
            )

        split_counts[
            split
        ] += 1

        status = record[
            "status"
        ]

        if status == "OK":
            successful_images += 1

            if record[
                "rows"
            ] != EXPECTED_LANDMARKS_PER_FACE:
                raise RuntimeError(
                    "Successful image does not have exactly 478 rows: "
                    f"{split}/{image}"
                )

            if record[
                "indices"
            ] != set(
                range(
                    EXPECTED_LANDMARKS_PER_FACE
                )
            ):
                raise RuntimeError(
                    "Successful image does not contain landmark indices 0..477: "
                    f"{split}/{image}"
                )

        elif status == "NO_LANDMARKS":
            no_landmark_images += 1

        elif status == "EXECUTION_FAILED":
            execution_failed_images += 1

    if split_counts != {
        "Indoor": EXPECTED_IMAGES_PER_SPLIT,
        "Outdoor": EXPECTED_IMAGES_PER_SPLIT,
    }:
        raise RuntimeError(
            "Staged CSV split coverage is incorrect."
        )

    expected_rows = (
        successful_images
        * EXPECTED_LANDMARKS_PER_FACE
        + no_landmark_images
        + execution_failed_images
    )

    if total_rows != expected_rows:
        raise RuntimeError(
            "Staged CSV row accounting is inconsistent."
        )

    if native_observations != summary.get(
        "native_landmark_observations"
    ):
        raise RuntimeError(
            "Staged CSV native-landmark count does not match summary."
        )

    if total_rows != summary.get(
        "result_rows"
    ):
        raise RuntimeError(
            "Staged CSV row count does not match summary."
        )

    if successful_images != summary.get(
        "images_with_landmarks"
    ):
        raise RuntimeError(
            "Staged successful-image count does not match summary."
        )

    if no_landmark_images != summary.get(
        "images_without_landmarks"
    ):
        raise RuntimeError(
            "Staged no-landmark count does not match summary."
        )

    if execution_failed_images != summary.get(
        "execution_failed_images"
    ):
        raise RuntimeError(
            "Staged execution-failure count does not match summary."
        )


def replace_owned_outputs(
    staged_csv: Path,
    staged_summary: Path,
    staging_dir: Path,
) -> None:
    pairs = [
        (
            staged_csv,
            RESULTS_CSV_PATH,
        ),
        (
            staged_summary,
            SUMMARY_JSON_PATH,
        ),
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backups = []
    installed = []

    try:
        for _, final_path in pairs:
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                backups.append(
                    (
                        backup_path,
                        final_path,
                    )
                )

        for staged_path, final_path in pairs:
            os.replace(
                staged_path,
                final_path,
            )

            installed.append(
                final_path
            )

    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()

        for backup_path, final_path in reversed(
            backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def main() -> None:
    print(
        "=" * 82
    )
    print(
        "PhysioTrack Face Landmarks Isolated Component Execution"
    )
    print(
        "=" * 82
    )

    samples, model_path = preflight()

    print(
        "Preflight: PASS"
    )
    print(
        f"Dataset: {DATASET_ROOT}"
    )
    print(
        f"Images: {len(samples)}"
    )
    print(
        f"Landmark model: {model_path.name}"
    )
    print(
        "Input face box: GT-derived landmark box with 20% padding"
    )
    print(
        "Accuracy metrics: not computed by this component-execution test"
    )
    print()

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".face_landmarks_component_",
            dir=OUTPUT_DIR,
        )
    )

    staged_csv = (
        staging_dir
        / RESULTS_CSV_PATH.name
    )

    staged_summary = (
        staging_dir
        / SUMMARY_JSON_PATH.name
    )

    detector = ControlledFaceBoxDetector()

    pipeline = FaceAnalysis(
        detector=detector,
        config=make_config(),
        landmark_model_path=model_path,
        device="cpu",
        verbose=False,
    )

    assert_landmark_only_pipeline(
        pipeline,
        detector,
    )

    original_landmarker = pipeline.landmarks

    capture = LandmarkCapture(
        original_landmarker
    )

    pipeline.landmarks = capture

    split_summary = {
        split: {
            "expected_images": EXPECTED_IMAGES_PER_SPLIT,
            "processed_images": 0,
            "images_with_landmarks": 0,
            "images_without_landmarks": 0,
            "execution_failed_images": 0,
            "native_landmark_observations": 0,
            "rows": 0,
            "out_of_frame_landmark_observations": 0,
        }
        for split, _ in DATASETS
    }

    processed_images = 0
    images_with_landmarks = 0
    images_without_landmarks = 0
    execution_failed_images = 0
    native_landmark_observations = 0
    out_of_frame_landmark_observations = 0
    result_rows = 0

    start_time = time.perf_counter()

    try:
        with staged_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDNAMES,
            )

            writer.writeheader()

            for image_index, (
                split,
                image_path,
                pts_path,
            ) in enumerate(
                samples,
                start=1,
            ):
                frame = cv2.imread(
                    str(image_path)
                )

                if frame is None:
                    raise RuntimeError(
                        f"Could not load image after successful preflight: {image_path}"
                    )

                height, width = frame.shape[:2]

                gt_68 = load_300w_points(
                    pts_path
                )

                face_box = make_gt_face_box(
                    gt_68,
                    width,
                    height,
                )

                detector.set_box(
                    face_box
                )

                capture.reset_capture()

                split_info = split_summary[
                    split
                ]

                try:
                    prediction = pipeline.predict(
                        frame
                    )

                    if len(prediction) != 1:
                        raise RuntimeError(
                            "Controlled one-face input produced an unexpected "
                            f"FaceAnalysis result count: {len(prediction)}"
                        )

                    instance = prediction[
                        0
                    ]

                    features = (
                        instance.face_features
                        or {}
                    )

                    landmark_features = features.get(
                        "landmarks",
                        {},
                    )

                    pipeline_available = bool(
                        landmark_features.get(
                            "available",
                            False,
                        )
                    )

                    pipeline_count = int(
                        landmark_features.get(
                            "count",
                            0,
                        )
                    )

                    landmarks = capture.last_landmarks

                    if landmarks is None:
                        if pipeline_available:
                            raise RuntimeError(
                                "FaceAnalysis reports landmarks available, "
                                "but the captured component output is None."
                            )

                        if pipeline_count != 0:
                            raise RuntimeError(
                                "FaceAnalysis reports a non-zero landmark count "
                                "when no native landmarks were returned."
                            )

                        row = blank_landmark_row(
                            split,
                            image_index,
                            image_path,
                            width,
                            height,
                            face_box,
                            pipeline_available,
                            pipeline_count,
                            "NO_LANDMARKS",
                            "Landmark component returned no landmarks.",
                        )

                        writer.writerow(
                            row
                        )

                        images_without_landmarks += 1
                        split_info[
                            "images_without_landmarks"
                        ] += 1

                        result_rows += 1
                        split_info[
                            "rows"
                        ] += 1

                    else:
                        landmarks = list(
                            landmarks
                        )

                        landmark_count = len(
                            landmarks
                        )

                        if landmark_count != EXPECTED_LANDMARKS_PER_FACE:
                            raise RuntimeError(
                                "Unexpected native landmark count: "
                                f"{landmark_count} "
                                f"(expected {EXPECTED_LANDMARKS_PER_FACE})"
                            )

                        if not pipeline_available:
                            raise RuntimeError(
                                "Native landmarks were returned, but FaceAnalysis "
                                "reports landmarks unavailable."
                            )

                        if pipeline_count != landmark_count:
                            raise RuntimeError(
                                "FaceAnalysis landmark count does not match the "
                                "captured native component output."
                            )

                        for landmark_index, landmark in enumerate(
                            landmarks
                        ):
                            if not all(
                                finite_numeric(
                                    value
                                )
                                for value in (
                                    landmark.x,
                                    landmark.y,
                                    landmark.z,
                                )
                            ):
                                raise RuntimeError(
                                    "Landmark component returned a non-finite "
                                    f"coordinate at index {landmark_index}."
                                )

                            row = landmark_row(
                                split,
                                image_index,
                                image_path,
                                width,
                                height,
                                face_box,
                                landmark_index,
                                landmark,
                                landmark_count,
                                pipeline_available,
                                pipeline_count,
                            )

                            writer.writerow(
                                row
                            )

                            if not row[
                                "within_image_bounds"
                            ]:
                                out_of_frame_landmark_observations += 1
                                split_info[
                                    "out_of_frame_landmark_observations"
                                ] += 1

                            native_landmark_observations += 1
                            split_info[
                                "native_landmark_observations"
                            ] += 1

                            result_rows += 1
                            split_info[
                                "rows"
                            ] += 1

                        images_with_landmarks += 1
                        split_info[
                            "images_with_landmarks"
                        ] += 1

                except Exception as exc:
                    row = blank_landmark_row(
                        split,
                        image_index,
                        image_path,
                        width,
                        height,
                        face_box,
                        False,
                        0,
                        "EXECUTION_FAILED",
                        str(exc),
                    )

                    writer.writerow(
                        row
                    )

                    execution_failed_images += 1
                    split_info[
                        "execution_failed_images"
                    ] += 1

                    result_rows += 1
                    split_info[
                        "rows"
                    ] += 1

                processed_images += 1
                split_info[
                    "processed_images"
                ] += 1

                if (
                    image_index % 50 == 0
                    or image_index == EXPECTED_TOTAL_IMAGES
                ):
                    print(
                        "Processed "
                        f"{image_index}/{EXPECTED_TOTAL_IMAGES} images"
                    )

    finally:
        pipeline.close()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    overall_status = (
        "PASS"
        if (
            processed_images == EXPECTED_TOTAL_IMAGES
            and execution_failed_images == 0
        )
        else "FAIL"
    )

    summary = {
        "component": "Face Landmarks",
        "execution_type": "isolated_component_execution",
        "dataset": "300-W evaluation set",
        "dataset_root": "datasets/300W",
        "input_face_box": "GT-derived 68-point landmark bounding box with 20% padding",
        "ground_truth_use": (
            "Ground-truth annotations are used only to define the controlled "
            "face input box and dataset coverage; this script does not compute "
            "landmark accuracy metrics."
        ),
        "pipeline": "PhysioTrack FaceAnalysis",
        "target_component": "FaceLandmarks",
        "model": model_path.name,
        "device": "cpu",
        "tracking_enabled": False,
        "unrelated_components_disabled": True,
        "expected_total_images": EXPECTED_TOTAL_IMAGES,
        "processed_images": processed_images,
        "images_with_landmarks": images_with_landmarks,
        "images_without_landmarks": images_without_landmarks,
        "execution_failed_images": execution_failed_images,
        "expected_landmarks_per_successful_image": EXPECTED_LANDMARKS_PER_FACE,
        "native_landmark_observations": native_landmark_observations,
        "out_of_frame_landmark_observations": out_of_frame_landmark_observations,
        "result_rows": result_rows,
        "splits": split_summary,
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60.0,
        "images_per_second": (
            processed_images / elapsed
            if elapsed > 0.0
            else None
        ),
        "status": overall_status,
        "interpretation": (
            "This output is software execution evidence for the real PhysioTrack "
            "Face Landmarks component. It is not a replacement for the accepted "
            "300-W NME benchmark."
        ),
    }

    with staged_summary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

    print()
    print(
        "Validating staged component outputs..."
    )

    validate_staged_results(
        staged_csv,
        staged_summary,
    )

    replace_owned_outputs(
        staged_csv,
        staged_summary,
        staging_dir,
    )

    if staging_dir.exists():
        shutil.rmtree(
            staging_dir
        )

    print(
        "Staged validation: PASS"
    )
    print(
        "Committed final component-execution outputs."
    )
    print()
    print(
        "Execution summary:"
    )
    print(
        f"Images processed: {processed_images}"
    )
    print(
        f"Images with landmarks: {images_with_landmarks}"
    )
    print(
        f"Images without landmarks: {images_without_landmarks}"
    )
    print(
        f"Execution failed images: {execution_failed_images}"
    )
    print(
        "Native landmark observations: "
        f"{native_landmark_observations}"
    )
    print(
        "Out-of-frame landmark observations: "
        f"{out_of_frame_landmark_observations}"
    )
    print(
        f"Result rows: {result_rows}"
    )
    print(
        f"Runtime: {elapsed / 60.0:.2f} minutes"
    )
    print(
        f"Overall status: {overall_status}"
    )
    print()
    print(
        "Saved:"
    )
    print(
        RESULTS_CSV_PATH
    )
    print(
        SUMMARY_JSON_PATH
    )


if __name__ == "__main__":
    main()
