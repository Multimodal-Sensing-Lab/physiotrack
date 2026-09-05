from pathlib import Path
import json
import shutil
import tempfile
import time

import cv2

from physiotrack.face import Face


def main():
    validation_dir = Path(__file__).resolve().parent
    project_root = validation_dir.parents[2]

    wider_root = project_root / "datasets" / "WIDER_FACE"

    images_dir = wider_root / "WIDER_val" / "images"

    results_dir = validation_dir / "results"
    output_dir = results_dir / "predictions"

    inference_summary_path = (
        results_dir
        / "wider_face_inference_summary.json"
    )

    if not images_dir.is_dir():
        raise FileNotFoundError(
            "WIDER FACE validation images were not found: "
            f"{images_dir}"
        )

    image_paths = sorted(
        images_dir.rglob("*.jpg")
    )

    if not image_paths:
        raise RuntimeError(
            "No WIDER FACE validation images were found: "
            f"{images_dir}"
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(
            "Prediction output path exists but is not a directory: "
            f"{output_dir}"
        )

    if (
        inference_summary_path.exists()
        and not inference_summary_path.is_file()
    ):
        raise RuntimeError(
            "Inference summary path exists but is not a file: "
            f"{inference_summary_path}"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = Face(
        device="cpu",
        verbose=False,
        conf=0.001,
        max_det=10000,
    )

    print("WIDER FACE inference preflight: PASS")
    print("Total images:", len(image_paths))

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".wider_face_inference_",
            dir=results_dir,
        )
    )

    staged_output_dir = (
        staging_root
        / "predictions"
    )

    staged_summary_path = (
        staging_root
        / "wider_face_inference_summary.json"
    )

    staged_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        start_time = time.perf_counter()

        failed_images = []
        total_detections = 0

        max_detections = 0
        max_detection_image = None

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                failed_images.append(
                    str(image_path)
                )
                continue

            result = detector.predict(
                image
            )

            num_detections = len(result)

            total_detections += num_detections

            if num_detections > max_detections:
                max_detections = num_detections
                max_detection_image = image_path

            relative_path = image_path.relative_to(
                images_dir
            )

            event_name = (
                relative_path.parent.name
            )

            event_output_dir = (
                staged_output_dir
                / event_name
            )

            event_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            prediction_file = (
                event_output_dir
                / f"{image_path.stem}.txt"
            )

            with open(
                prediction_file,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    f"{event_name}/{image_path.name}\n"
                )

                file.write(
                    f"{num_detections}\n"
                )

                for instance in result:
                    x1, y1, x2, y2 = instance.box

                    width = x2 - x1
                    height = y2 - y1

                    file.write(
                        f"{x1:.3f} "
                        f"{y1:.3f} "
                        f"{width:.3f} "
                        f"{height:.3f} "
                        f"{instance.confidence:.6f}\n"
                    )

            if (
                index % 100 == 0
                or index == len(image_paths)
            ):
                print(
                    f"Processed "
                    f"{index}/{len(image_paths)}"
                )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        prediction_files = sorted(
            staged_output_dir.rglob("*.txt")
        )

        images_processed = (
            len(image_paths)
            - len(failed_images)
        )

        max_detection_image_relative = None

        if max_detection_image is not None:
            max_detection_image_relative = str(
                max_detection_image.relative_to(
                    images_dir
                )
            ).replace("\\", "/")

        inference_summary = {
            "dataset": (
                "WIDER FACE validation split"
            ),
            "validation_images": len(image_paths),
            "images_processed": images_processed,
            "failed_images": len(failed_images),
            "prediction_files": len(
                prediction_files
            ),
            "total_detections": total_detections,
            "maximum_detections_in_one_image": (
                max_detections
            ),
            "image_with_maximum_detections": (
                max_detection_image_relative
            ),
            "confidence_threshold": 0.001,
            "max_det": 10000,
            "device": "CPU",
            "runtime_seconds": elapsed,
            "runtime_minutes": elapsed / 60.0,
        }

        with open(
            staged_summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                inference_summary,
                file,
                indent=2,
            )

        if len(prediction_files) != images_processed:
            raise RuntimeError(
                "Staged prediction-file count does not match the number "
                "of successfully processed images: "
                f"{len(prediction_files)} != {images_processed}"
            )

        for prediction_file in prediction_files:
            with open(
                prediction_file,
                "r",
                encoding="utf-8",
            ) as file:
                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            if len(lines) < 2:
                raise RuntimeError(
                    "Malformed staged prediction file: "
                    f"{prediction_file}"
                )

            count = int(lines[1])

            if len(lines[2:]) != count:
                raise RuntimeError(
                    "Detection count does not match staged prediction "
                    f"rows: {prediction_file}"
                )

            for line in lines[2:]:
                values = line.split()

                if len(values) != 5:
                    raise RuntimeError(
                        "Unexpected staged prediction row: "
                        f"{prediction_file}"
                    )

                [float(value) for value in values]

        with open(
            staged_summary_path,
            "r",
            encoding="utf-8",
        ) as file:
            validated_summary = json.load(
                file
            )

        if (
            validated_summary.get("validation_images")
            != len(image_paths)
            or validated_summary.get("images_processed")
            != images_processed
            or validated_summary.get("prediction_files")
            != len(prediction_files)
        ):
            raise RuntimeError(
                "Staged inference summary is inconsistent with staged "
                "prediction outputs."
            )

        previous_output_dir = (
            staging_root
            / "previous_predictions"
        )

        previous_summary_path = (
            staging_root
            / "previous_inference_summary.json"
        )

        output_committed = False
        summary_committed = False

        try:
            if output_dir.exists():
                output_dir.replace(
                    previous_output_dir
                )

            staged_output_dir.replace(
                output_dir
            )
            output_committed = True

            if inference_summary_path.exists():
                inference_summary_path.replace(
                    previous_summary_path
                )

            staged_summary_path.replace(
                inference_summary_path
            )
            summary_committed = True

        except Exception:
            if (
                summary_committed
                and inference_summary_path.exists()
            ):
                inference_summary_path.unlink()

            if previous_summary_path.exists():
                previous_summary_path.replace(
                    inference_summary_path
                )

            if output_committed and output_dir.exists():
                shutil.rmtree(
                    output_dir
                )

            if previous_output_dir.exists():
                previous_output_dir.replace(
                    output_dir
                )

            raise

    finally:
        if staging_root.exists():
            shutil.rmtree(
                staging_root
            )

    print()
    print("Finished.")

    print(
        "Images processed:",
        images_processed,
    )

    print(
        "Failed images:",
        len(failed_images),
    )

    print(
        "Prediction files:",
        len(prediction_files),
    )

    print(
        "Total detections:",
        total_detections,
    )

    print(
        "Maximum detections in one image:",
        max_detections,
    )

    print(
        "Image with maximum detections:",
        max_detection_image,
    )

    print(
        "Total time:",
        round(
            elapsed / 60,
            2,
        ),
        "minutes",
    )

if __name__ == "__main__":
    main()
