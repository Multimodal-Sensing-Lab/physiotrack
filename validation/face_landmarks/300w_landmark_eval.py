from pathlib import Path
import csv
import os
import time

import cv2
import numpy as np

from physiotrack.face.landmarks import FaceLandmarks
from mediapipe_300w_mapping import get_mediapipe_300w_51


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "300W"

RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "300w_landmark_results.csv"
SUMMARY_TXT = RESULTS_DIR / "300w_landmark_summary.txt"

DATASETS = [
    ("Indoor", DATASET_ROOT / "01_Indoor"),
    ("Outdoor", DATASET_ROOT / "02_Outdoor"),
]


def resolve_model_path():
    """Resolve the MediaPipe face landmarker model without a user-specific path."""
    candidates = []

    env_model = os.environ.get(
        "PHYSIOTRACK_FACE_LANDMARKER_MODEL"
    )

    if env_model:
        candidates.append(Path(env_model))

    local_app_data = os.environ.get("LOCALAPPDATA")

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


def verify_dataset():
    """Verify the expected 300-W evaluation package."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    print("Dataset verification:")
    print(f"Dataset: {DATASET_ROOT}")

    total_images = 0
    total_annotations = 0

    for split, folder in DATASETS:
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset split: {folder}"
            )

        images = sorted(folder.glob("*.png"))
        annotations = sorted(folder.glob("*.pts"))

        image_stems = {
            path.stem
            for path in images
        }

        annotation_stems = {
            path.stem
            for path in annotations
        }

        missing_annotations = sorted(
            image_stems - annotation_stems
        )

        missing_images = sorted(
            annotation_stems - image_stems
        )

        if missing_annotations:
            raise RuntimeError(
                f"{split}: images without .pts annotations: "
                f"{missing_annotations[:10]}"
            )

        if missing_images:
            raise RuntimeError(
                f"{split}: .pts annotations without images: "
                f"{missing_images[:10]}"
            )

        if len(images) != 300:
            raise RuntimeError(
                f"{split}: expected 300 PNG images, "
                f"found {len(images)}"
            )

        if len(annotations) != 300:
            raise RuntimeError(
                f"{split}: expected 300 PTS files, "
                f"found {len(annotations)}"
            )

        print(
            f"{split}: "
            f"{len(images)} images, "
            f"{len(annotations)} annotations"
        )

        total_images += len(images)
        total_annotations += len(annotations)

    print(
        f"Total: {total_images} images, "
        f"{total_annotations} annotations"
    )

    if total_images != 600:
        raise RuntimeError(
            f"Expected 600 images, found {total_images}"
        )

    print("Dataset verification passed.\n")


def load_300w_points(path):
    """Load the 68 landmark coordinates from a 300-W .pts file."""
    points = []
    inside = False

    with open(
        path,
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
                    (x, y)
                )

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) != 68:
        raise RuntimeError(
            f"Expected 68 points in {path.name}, "
            f"found {len(points)}"
        )

    # Convert one-based 300-W image coordinates to zero-based.
    points -= 1.0

    return points


def make_gt_face_box(
    points,
    image_width,
    image_height,
):
    """Create a padded face box from the 68 GT landmarks."""
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

    face_width = x_max - x_min
    face_height = y_max - y_min

    pad_x = 0.20 * face_width
    pad_y = 0.20 * face_height

    x1 = max(
        0.0,
        x_min - pad_x,
    )

    y1 = max(
        0.0,
        y_min - pad_y,
    )

    x2 = min(
        float(image_width - 1),
        x_max + pad_x,
    )

    y2 = min(
        float(image_height - 1),
        y_max + pad_y,
    )

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def evaluate_image(
    image_path,
    pts_path,
    landmarker,
    mapping,
):
    """Evaluate landmark localization for one annotated face."""
    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = frame.shape[:2]

    gt_68 = load_300w_points(
        pts_path
    )

    gt_51 = gt_68[17:]

    face_box = make_gt_face_box(
        gt_68,
        width,
        height,
    )

    landmarks = landmarker.predict_face(
        frame,
        face_box,
    )

    if landmarks is None:
        return {
            "status": "failed_detection",
            "nme": np.nan,
            "nme_percent": np.nan,
            "mean_pixel_error": np.nan,
            "interocular": np.nan,
        }

    predicted = []

    for index in mapping:
        landmark = landmarks[index]

        predicted.append(
            (
                landmark.x * width,
                landmark.y * height,
            )
        )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if len(predicted) != 51:
        raise RuntimeError(
            f"Expected 51 predicted points, "
            f"found {len(predicted)}"
        )

    point_errors = np.linalg.norm(
        gt_51 - predicted,
        axis=1,
    )

    interocular = np.linalg.norm(
        gt_51[19]
        - gt_51[28]
    )

    if interocular <= 0:
        raise RuntimeError(
            f"Invalid interocular distance for "
            f"{image_path.name}"
        )

    mean_pixel_error = float(
        point_errors.mean()
    )

    nme = (
        mean_pixel_error
        / interocular
    )

    return {
        "status": "ok",
        "nme": float(nme),
        "nme_percent": float(
            nme * 100.0
        ),
        "mean_pixel_error": (
            mean_pixel_error
        ),
        "interocular": float(
            interocular
        ),
    }


def summarize_split(
    rows,
    split,
):
    """Calculate summary statistics for one split."""
    split_rows = [
        row
        for row in rows
        if row["split"] == split
    ]

    successful = [
        row
        for row in split_rows
        if row["status"] == "ok"
    ]

    failures = (
        len(split_rows)
        - len(successful)
    )

    nmes = np.asarray(
        [
            row["nme_percent"]
            for row in successful
        ],
        dtype=float,
    )

    if len(nmes) > 0:
        mean_nme = float(
            np.mean(nmes)
        )

        median_nme = float(
            np.median(nmes)
        )

        std_nme = float(
            np.std(nmes)
        )
    else:
        mean_nme = np.nan
        median_nme = np.nan
        std_nme = np.nan

    return {
        "images": len(split_rows),
        "successful": len(successful),
        "failed": failures,
        "detection_rate": (
            len(successful)
            / len(split_rows)
            * 100.0
            if split_rows
            else 0.0
        ),
        "mean_nme": mean_nme,
        "median_nme": median_nme,
        "std_nme": std_nme,
    }


def main():
    verify_dataset()

    model_path = resolve_model_path()

    print("Model verification:")
    print(
        "MediaPipe model: "
        f"{model_path.name}"
    )
    print("Model verification passed.\n")

    mapping = (
        get_mediapipe_300w_51()
    )

    if len(mapping) != 51:
        raise RuntimeError(
            f"Expected 51 mapped landmarks, "
            f"found {len(mapping)}"
        )

    if len(set(mapping)) != 51:
        raise RuntimeError(
            "The 51-point mapping contains "
            "duplicate MediaPipe indices."
        )

    print(
        "Landmark mapping: "
        f"{len(mapping)} unique points\n"
    )

    landmarker = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    rows = []

    total_start = (
        time.perf_counter()
    )

    try:
        for split, folder in DATASETS:
            images = sorted(
                folder.glob("*.png")
            )

            print(
                f"\n=== {split} ==="
            )

            print(
                f"Images: {len(images)}"
            )

            for number, image_path in enumerate(
                images,
                start=1,
            ):
                pts_path = (
                    image_path
                    .with_suffix(".pts")
                )

                result = evaluate_image(
                    image_path,
                    pts_path,
                    landmarker,
                    mapping,
                )

                rows.append(
                    {
                        "split": split,
                        "image": (
                            image_path.name
                        ),
                        "status": (
                            result["status"]
                        ),
                        "nme": (
                            result["nme"]
                        ),
                        "nme_percent": (
                            result[
                                "nme_percent"
                            ]
                        ),
                        "mean_pixel_error": (
                            result[
                                "mean_pixel_error"
                            ]
                        ),
                        "interocular_px": (
                            result[
                                "interocular"
                            ]
                        ),
                    }
                )

                if number % 50 == 0:
                    print(
                        f"Processed "
                        f"{number}/"
                        f"{len(images)} images"
                    )

    finally:
        landmarker.close()

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "image",
                "status",
                "nme",
                "nme_percent",
                "mean_pixel_error",
                "interocular_px",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    indoor = summarize_split(
        rows,
        "Indoor",
    )

    outdoor = summarize_split(
        rows,
        "Outdoor",
    )

    successful_all = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    overall_nmes = np.asarray(
        [
            row["nme_percent"]
            for row in successful_all
        ],
        dtype=float,
    )

    overall_failed = (
        len(rows)
        - len(successful_all)
    )

    overall_detection_rate = (
        len(successful_all)
        / len(rows)
        * 100.0
        if rows
        else 0.0
    )

    if len(overall_nmes) > 0:
        overall_mean = float(
            np.mean(
                overall_nmes
            )
        )

        overall_median = float(
            np.median(
                overall_nmes
            )
        )

        overall_std = float(
            np.std(
                overall_nmes
            )
        )
    else:
        overall_mean = np.nan
        overall_median = np.nan
        overall_std = np.nan

    print(
        "\n=== 300-W 51-point results ===\n"
    )

    for name, summary in [
        ("Indoor", indoor),
        ("Outdoor", outdoor),
    ]:
        print(name)

        print(
            f"  Images: "
            f"{summary['images']}"
        )

        print(
            f"  Successful: "
            f"{summary['successful']}"
        )

        print(
            f"  Failed detections: "
            f"{summary['failed']}"
        )

        print(
            f"  Detection rate: "
            f"{summary['detection_rate']:.2f}%"
        )

        print(
            f"  Mean NME: "
            f"{summary['mean_nme']:.4f}%"
        )

        print(
            f"  Median NME: "
            f"{summary['median_nme']:.4f}%"
        )

        print(
            f"  Std NME: "
            f"{summary['std_nme']:.4f}%"
        )

        print()

    print("OVERALL")

    print(
        f"  Images: "
        f"{len(rows)}"
    )

    print(
        f"  Successful: "
        f"{len(successful_all)}"
    )

    print(
        f"  Failed detections: "
        f"{overall_failed}"
    )

    print(
        f"  Detection rate: "
        f"{overall_detection_rate:.2f}%"
    )

    print(
        f"  Mean NME: "
        f"{overall_mean:.4f}%"
    )

    print(
        f"  Median NME: "
        f"{overall_median:.4f}%"
    )

    print(
        f"  Std NME: "
        f"{overall_std:.4f}%"
    )

    print(
        f"\nTotal runtime: "
        f"{total_elapsed:.2f} seconds"
    )

    print(
        f"Processing speed: "
        f"{len(rows) / total_elapsed:.2f} images/s"
    )

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "300-W Facial Landmark Validation\n\n"
        )

        file.write("Dataset:\n")
        file.write(
            "300-W evaluation set "
            "(300 Indoor + 300 Outdoor images)\n\n"
        )

        file.write(
            "Dataset coverage:\n"
        )
        file.write(
            "Indoor images: 300\n"
        )
        file.write(
            "Outdoor images: 300\n"
        )
        file.write(
            "Total images: 600\n"
        )
        file.write(
            "Annotation format: "
            "68-point PTS files\n\n"
        )

        file.write("Protocol:\n")
        file.write(
            "51-point landmark evaluation\n"
        )
        file.write(
            "17 face-border landmarks excluded\n"
        )
        file.write(
            "Normalization: inter-ocular distance "
            "between outer eye corners\n"
        )
        file.write(
            "Prediction: "
            "PhysioTrack FaceLandmarks.predict_face()\n"
        )
        file.write(
            "Model: MediaPipe Face Landmarker "
            "(478 landmarks)\n"
        )
        file.write(
            "Evaluation adapter: fixed 51-point "
            "MediaPipe-to-300-W anatomical mapping\n"
        )
        file.write(
            "Face initialization: GT-derived padded "
            "bounding box from the 68-point annotation\n"
        )
        file.write(
            "Padding: 20 percent of GT landmark "
            "bounding-box width and height\n"
        )
        file.write(
            "Purpose: isolate landmark localization "
            "from face detection and face selection\n"
        )
        file.write(
            "The results are controlled component "
            "validation values, not official 300-W "
            "competition leaderboard results.\n\n"
        )

        for name, summary in [
            ("Indoor", indoor),
            ("Outdoor", outdoor),
        ]:
            file.write(
                f"{name}\n"
            )

            file.write(
                f"Images: "
                f"{summary['images']}\n"
            )

            file.write(
                f"Successful: "
                f"{summary['successful']}\n"
            )

            file.write(
                f"Failed detections: "
                f"{summary['failed']}\n"
            )

            file.write(
                f"Detection rate: "
                f"{summary['detection_rate']:.2f}%\n"
            )

            file.write(
                f"Mean NME: "
                f"{summary['mean_nme']:.4f}%\n"
            )

            file.write(
                f"Median NME: "
                f"{summary['median_nme']:.4f}%\n"
            )

            file.write(
                f"Std NME: "
                f"{summary['std_nme']:.4f}%\n\n"
            )

        file.write("OVERALL\n")

        file.write(
            f"Images: {len(rows)}\n"
        )

        file.write(
            f"Successful: "
            f"{len(successful_all)}\n"
        )

        file.write(
            f"Failed detections: "
            f"{overall_failed}\n"
        )

        file.write(
            f"Detection rate: "
            f"{overall_detection_rate:.2f}%\n"
        )

        file.write(
            f"Mean NME: "
            f"{overall_mean:.4f}%\n"
        )

        file.write(
            f"Median NME: "
            f"{overall_median:.4f}%\n"
        )

        file.write(
            f"Std NME: "
            f"{overall_std:.4f}%\n\n"
        )

        file.write("Runtime:\n")

        file.write(
            f"Total runtime: "
            f"{total_elapsed:.2f} seconds\n"
        )

        file.write(
            f"Processing speed: "
            f"{len(rows) / total_elapsed:.2f} images/s\n"
        )

        file.write(
            "\nGenerated outputs:\n"
        )

        file.write(
            "Detailed results CSV: "
            "results/300w_landmark_results.csv\n"
        )

        file.write(
            "Validation summary: "
            "results/300w_landmark_summary.txt\n"
        )

        file.write(
            "CED data: "
            "results/300w_landmark_ced.csv\n"
        )

        file.write(
            "Thesis table CSV: "
            "results/300w_landmark_thesis_table.csv\n"
        )

        file.write(
            "Thesis table Markdown: "
            "results/300w_landmark_thesis_table.md\n"
        )

        file.write(
            "CED figure: "
            "results/figures/300w_landmark_ced.png\n"
        )

    print("\nSaved:")
    print(RESULTS_CSV)
    print(SUMMARY_TXT)


if __name__ == "__main__":
    main()