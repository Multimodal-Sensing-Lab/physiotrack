from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.io import loadmat

from ptgaze.gaze_estimator import GazeEstimator
from ptgaze.main import (
    download_ethxgaze_model,
    expanduser_all,
    load_mode_config,
)


DATASET_ROOT = Path(
    r"C:\Users\xx901\Documents\PhysioTrack_Thesis"
    r"\datasets\MPIIFaceGaze\Data"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

MODEL_MODE = "eth-xgaze"
DEVICE = "cpu"


def angular_error_degrees(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    ground_truth = ground_truth.astype(
        np.float64,
        copy=False,
    )
    prediction = prediction.astype(
        np.float64,
        copy=False,
    )

    gt_norm = np.linalg.norm(ground_truth)
    pred_norm = np.linalg.norm(prediction)

    if gt_norm <= 0 or pred_norm <= 0:
        raise ValueError(
            "Cannot compute angular error for a zero-length vector."
        )

    ground_truth = ground_truth / gt_norm
    prediction = prediction / pred_norm

    dot_product = float(
        np.clip(
            np.dot(
                ground_truth,
                prediction,
            ),
            -1.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(dot_product)
    )


def create_camera_yaml(
    person_dir: Path,
    image_width: int,
    image_height: int,
) -> Path:
    calibration_dir = person_dir / "Calibration"

    camera_mat_path = (
        calibration_dir / "Camera.mat"
    )

    camera_data = loadmat(camera_mat_path)

    camera_matrix = np.asarray(
        camera_data["cameraMatrix"],
        dtype=np.float64,
    )

    distortion = np.asarray(
        camera_data["distCoeffs"],
        dtype=np.float64,
    ).reshape(-1)

    output_path = (
        calibration_dir
        / "ptgaze_camera.yaml"
    )

    data = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": (
                camera_matrix
                .reshape(-1)
                .tolist()
            ),
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(len(distortion)),
            "data": distortion.tolist(),
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )

    return output_path


def build_estimator(
    image_path: Path,
    camera_yaml_path: Path,
    checkpoint_path: str,
) -> GazeEstimator:
    args = argparse.Namespace(
        mode=MODEL_MODE,
        face_detector="mediapipe",
        device=DEVICE,
        image=str(image_path),
        video=None,
        camera=str(camera_yaml_path),
        output_dir=None,
        ext=None,
        no_screen=True,
    )

    config = load_mode_config(args)

    config.gaze_estimator.checkpoint = (
        checkpoint_path
    )

    expanduser_all(config)

    return GazeEstimator(config)


def parse_annotation_line(
    line: str,
) -> tuple[
    str,
    np.ndarray,
]:
    parts = line.strip().split()

    if len(parts) != 28:
        raise ValueError(
            f"Unexpected annotation field count: {len(parts)}"
        )

    image_relative_path = parts[0]

    values = np.asarray(
        [
            float(value)
            for value in parts[1:27]
        ],
        dtype=np.float64,
    )

    face_center = values[20:23]
    gaze_target = values[23:26]

    gaze_vector = (
        gaze_target - face_center
    )

    norm = np.linalg.norm(gaze_vector)

    if (
        norm <= 0
        or not np.all(
            np.isfinite(gaze_vector)
        )
    ):
        raise ValueError(
            "Invalid ground-truth gaze vector."
        )

    gaze_vector = gaze_vector / norm

    return (
        image_relative_path,
        gaze_vector,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_csv_path = (
        OUTPUT_DIR
        / "mpiifacegaze_ethxgaze_per_person.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "mpiifacegaze_ethxgaze_summary.txt"
    )

    checkpoint_path = (
        download_ethxgaze_model()
        .as_posix()
    )

    all_errors: list[float] = []

    total_annotations = 0
    successful_predictions = 0

    image_read_failures = 0
    face_detection_failures = 0
    prediction_failures = 0
    invalid_annotation_rows = 0

    person_results: list[dict] = []

    start_time = time.time()

    person_dirs = sorted(
        path
        for path in DATASET_ROOT.glob("p*")
        if path.is_dir()
    )

    print(
        "=== MPIIFaceGaze Cross-Dataset "
        "Gaze Estimation Evaluation ==="
    )

    print(
        f"Participants: {len(person_dirs)}"
    )

    print(
        f"Model: {MODEL_MODE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print()

    for person_dir in person_dirs:
        annotation_path = (
            person_dir
            / f"{person_dir.name}.txt"
        )

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
        ) as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        total_annotations += len(lines)

        reference_image = None

        for line in lines:
            try:
                image_relative_path, _ = (
                    parse_annotation_line(line)
                )
            except ValueError:
                continue

            candidate = (
                person_dir
                / image_relative_path
            )

            image = cv2.imread(
                str(candidate)
            )

            if image is not None:
                reference_image = candidate
                break

        if reference_image is None:
            print(
                f"{person_dir.name}: "
                "no readable reference image"
            )

            image_read_failures += len(lines)
            continue

        reference = cv2.imread(
            str(reference_image)
        )

        image_height, image_width = (
            reference.shape[:2]
        )

        camera_yaml_path = (
            create_camera_yaml(
                person_dir,
                image_width,
                image_height,
            )
        )

        estimator = build_estimator(
            reference_image,
            camera_yaml_path,
            checkpoint_path,
        )

        person_errors: list[float] = []

        person_image_failures = 0
        person_detection_failures = 0
        person_prediction_failures = 0
        person_invalid_annotations = 0

        person_start = time.time()

        for index, line in enumerate(
            lines,
            start=1,
        ):
            try:
                (
                    image_relative_path,
                    ground_truth,
                ) = parse_annotation_line(
                    line
                )

            except Exception:
                invalid_annotation_rows += 1
                person_invalid_annotations += 1
                continue

            image_path = (
                person_dir
                / image_relative_path
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                image_read_failures += 1
                person_image_failures += 1
                continue

            try:
                faces = (
                    estimator
                    .detect_faces(image)
                )

            except Exception:
                face_detection_failures += 1
                person_detection_failures += 1
                continue

            if not faces:
                face_detection_failures += 1
                person_detection_failures += 1
                continue

            try:
                estimator.estimate_gaze(
                    image,
                    faces[0],
                )

                prediction = np.asarray(
                    faces[0].gaze_vector,
                    dtype=np.float64,
                )

                if not np.all(
                    np.isfinite(prediction)
                ):
                    raise ValueError(
                        "Non-finite prediction."
                    )

                error = (
                    angular_error_degrees(
                        ground_truth,
                        prediction,
                    )
                )

            except Exception:
                prediction_failures += 1
                person_prediction_failures += 1
                continue

            person_errors.append(
                error
            )

            all_errors.append(
                error
            )

            successful_predictions += 1

            if (
                index % 500 == 0
                or index == len(lines)
            ):
                print(
                    f"{person_dir.name}: "
                    f"{index}/{len(lines)}"
                )

        estimator.close()

        person_runtime = (
            time.time() - person_start
        )

        if person_errors:
            person_array = np.asarray(
                person_errors,
                dtype=np.float64,
            )

            person_mean = float(
                person_array.mean()
            )

            person_median = float(
                np.median(person_array)
            )

            person_std = float(
                person_array.std()
            )

        else:
            person_mean = math.nan
            person_median = math.nan
            person_std = math.nan

        person_results.append(
            {
                "participant": (
                    person_dir.name
                ),
                "annotations": len(lines),
                "successful_predictions": (
                    len(person_errors)
                ),
                "image_read_failures": (
                    person_image_failures
                ),
                "face_detection_failures": (
                    person_detection_failures
                ),
                "prediction_failures": (
                    person_prediction_failures
                ),
                "invalid_annotation_rows": (
                    person_invalid_annotations
                ),
                "mean_angular_error_deg": (
                    person_mean
                ),
                "median_angular_error_deg": (
                    person_median
                ),
                "std_angular_error_deg": (
                    person_std
                ),
                "runtime_seconds": (
                    person_runtime
                ),
            }
        )

        print(
            f"{person_dir.name}: "
            f"success={len(person_errors)}, "
            f"mean={person_mean:.4f} deg, "
            f"median={person_median:.4f} deg"
        )

        print()

    total_runtime = (
        time.time() - start_time
    )

    error_array = np.asarray(
        all_errors,
        dtype=np.float64,
    )

    if len(error_array):
        mean_error = float(
            error_array.mean()
        )

        median_error = float(
            np.median(error_array)
        )

        std_error = float(
            error_array.std()
        )

        min_error = float(
            error_array.min()
        )

        max_error = float(
            error_array.max()
        )

        p90_error = float(
            np.percentile(
                error_array,
                90,
            )
        )

        p95_error = float(
            np.percentile(
                error_array,
                95,
            )
        )

    else:
        mean_error = math.nan
        median_error = math.nan
        std_error = math.nan
        min_error = math.nan
        max_error = math.nan
        p90_error = math.nan
        p95_error = math.nan

    with open(
        sequence_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "participant",
            "annotations",
            "successful_predictions",
            "image_read_failures",
            "face_detection_failures",
            "prediction_failures",
            "invalid_annotation_rows",
            "mean_angular_error_deg",
            "median_angular_error_deg",
            "std_angular_error_deg",
            "runtime_seconds",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in person_results:
            writer.writerow(row)

    summary_lines = [
        "MPIIFaceGaze Cross-Dataset Gaze Estimation Evaluation",
        "",
        f"Model: {MODEL_MODE}",
        f"Device: {DEVICE}",
        f"Participants: {len(person_dirs)}",
        f"Total annotations: {total_annotations}",
        f"Successful predictions: {successful_predictions}",
        f"Image read failures: {image_read_failures}",
        f"Face detection failures: {face_detection_failures}",
        f"Prediction failures: {prediction_failures}",
        f"Invalid annotation rows: {invalid_annotation_rows}",
        "",
        f"Mean angular error: {mean_error:.6f} deg",
        f"Median angular error: {median_error:.6f} deg",
        f"Std angular error: {std_error:.6f} deg",
        f"Minimum angular error: {min_error:.6f} deg",
        f"Maximum angular error: {max_error:.6f} deg",
        f"90th percentile angular error: {p90_error:.6f} deg",
        f"95th percentile angular error: {p95_error:.6f} deg",
        "",
        f"Runtime: {total_runtime:.2f} seconds",
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(summary_lines)
        )

    print(
        "=== Final Summary ==="
    )

    for line in summary_lines:
        print(line)

    print()

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {sequence_csv_path}"
    )


if __name__ == "__main__":
    main()