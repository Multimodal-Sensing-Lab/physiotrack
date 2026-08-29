from pathlib import Path
import json
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

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = Face(
        device="cpu",
        verbose=False,
        conf=0.001,
        max_det=10000,
    )

    image_paths = sorted(
        images_dir.rglob("*.jpg")
    )

    print("Total images:", len(image_paths))

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
            output_dir
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

    prediction_files = list(
        output_dir.rglob("*.txt")
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
        inference_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            inference_summary,
            file,
            indent=2,
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