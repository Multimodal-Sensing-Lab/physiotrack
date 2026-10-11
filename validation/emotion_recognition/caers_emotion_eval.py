from pathlib import Path
import argparse
import csv
import math
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face.emotion import FaceEmotion


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "CAER-S"
TEST_ROOT = DATASET_ROOT / "test"
ANNOTATION_PATH = DATASET_ROOT / "test.txt"

RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "caers_emotion_results.csv"
SUMMARY_PATH = RESULTS_DIR / "caers_emotion_summary.txt"

MODEL_NAME = "enet_b0_8_best_afew"
ENGINE = "onnx"

CLASS_ID_TO_DATASET_LABEL = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

DATASET_TO_MODEL_LABEL = {
    "Anger": "Anger",
    "Disgust": "Disgust",
    "Fear": "Fear",
    "Happy": "Happiness",
    "Neutral": "Neutral",
    "Sad": "Sadness",
    "Surprise": "Surprise",
}

EXPECTED_MODEL_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

RESULT_FIELDS = [
    "relative_path",
    "dataset_label",
    "expected_model_label",
    "class_id",
    "original_x1",
    "original_y1",
    "original_x2",
    "original_y2",
    "clipped_x1",
    "clipped_y1",
    "clipped_x2",
    "clipped_y2",
    "box_was_clipped",
    "crop_width",
    "crop_height",
    "status",
    "predicted_label",
    "confidence",
    "score_anger",
    "score_contempt",
    "score_disgust",
    "score_fear",
    "score_happiness",
    "score_neutral",
    "score_sadness",
    "score_surprise",
    "score_sum",
    "correct",
    "failure_reason",
]


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PhysioTrack FaceEmotion on the annotated "
            "CAER-S target-face test subset."
        )
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run dataset and protocol preflight checks only.",
    )

    parser.add_argument(
        "--smoke-test",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run the final evaluator logic on N deterministic samples "
            "without replacing final benchmark outputs."
        ),
    )

    return parser.parse_args()


def load_annotations():
    """Load and validate the CAER-S target-face annotation file."""
    records = []
    seen_paths = set()

    with ANNOTATION_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 6:
                raise RuntimeError(
                    f"Malformed annotation row at line {line_number}: "
                    f"{line}"
                )

            relative_path = parts[0].replace(
                "\\",
                "/",
            )

            if relative_path in seen_paths:
                raise RuntimeError(
                    "Duplicate annotation path found: "
                    f"{relative_path}"
                )

            try:
                class_id = int(
                    parts[1]
                )
                x1, y1, x2, y2 = map(
                    int,
                    parts[2:6],
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid numeric annotation at line {line_number}: "
                    f"{line}"
                ) from exc

            if class_id not in CLASS_ID_TO_DATASET_LABEL:
                raise RuntimeError(
                    f"Unexpected class ID at line {line_number}: "
                    f"{class_id}"
                )

            path_parts = Path(
                relative_path
            ).parts

            if len(path_parts) < 2:
                raise RuntimeError(
                    f"Invalid relative path at line {line_number}: "
                    f"{relative_path}"
                )

            dataset_label = path_parts[0]
            expected_dataset_label = (
                CLASS_ID_TO_DATASET_LABEL[
                    class_id
                ]
            )

            if dataset_label != expected_dataset_label:
                raise RuntimeError(
                    "Class ID and directory label disagree at line "
                    f"{line_number}: {relative_path}, class_id={class_id}"
                )

            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(
                    f"Invalid face box at line {line_number}: "
                    f"{(x1, y1, x2, y2)}"
                )

            records.append(
                {
                    "relative_path": relative_path,
                    "dataset_label": dataset_label,
                    "expected_model_label": (
                        DATASET_TO_MODEL_LABEL[
                            dataset_label
                        ]
                    ),
                    "class_id": class_id,
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

            seen_paths.add(
                relative_path
            )

    if not records:
        raise RuntimeError(
            "No annotation rows were loaded."
        )

    return records


def preflight():
    """Validate all required benchmark inputs before inference."""
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(
            f"CAER-S dataset directory not found: {DATASET_ROOT}"
        )

    if not TEST_ROOT.is_dir():
        raise RuntimeError(
            f"CAER-S test directory not found: {TEST_ROOT}"
        )

    if not ANNOTATION_PATH.is_file():
        raise RuntimeError(
            f"CAER-S annotation file not found: {ANNOTATION_PATH}"
        )

    records = load_annotations()

    missing_images = []
    unreadable_images = []
    invalid_after_clipping = []
    out_of_bounds_count = 0

    class_counts = {
        label: 0
        for label in DATASET_TO_MODEL_LABEL
    }

    for record in records:
        relative_path = record[
            "relative_path"
        ]

        image_path = (
            TEST_ROOT
            / relative_path
        )

        if not image_path.is_file():
            missing_images.append(
                relative_path
            )
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            unreadable_images.append(
                relative_path
            )
            continue

        height, width = image.shape[:2]

        x1, y1, x2, y2 = record[
            "box"
        ]

        clipped_x1 = max(
            0,
            min(x1, width),
        )
        clipped_y1 = max(
            0,
            min(y1, height),
        )
        clipped_x2 = max(
            0,
            min(x2, width),
        )
        clipped_y2 = max(
            0,
            min(y2, height),
        )

        if (
            clipped_x1 != x1
            or clipped_y1 != y1
            or clipped_x2 != x2
            or clipped_y2 != y2
        ):
            out_of_bounds_count += 1

        if (
            clipped_x2 <= clipped_x1
            or clipped_y2 <= clipped_y1
        ):
            invalid_after_clipping.append(
                relative_path
            )

        class_counts[
            record["dataset_label"]
        ] += 1

    if missing_images:
        raise RuntimeError(
            "Missing annotated images: "
            f"{len(missing_images)}"
        )

    if unreadable_images:
        raise RuntimeError(
            "Unreadable annotated images: "
            f"{len(unreadable_images)}"
        )

    if invalid_after_clipping:
        raise RuntimeError(
            "Invalid boxes after clipping: "
            f"{len(invalid_after_clipping)}"
        )

    test_images = list(
        TEST_ROOT.rglob("*.png")
    )

    if len(test_images) == 0:
        raise RuntimeError(
            "No PNG images were found in the CAER-S test split."
        )

    coverage = (
        len(records)
        / len(test_images)
    )

    print(
        "CAER-S Emotion Recognition Preflight"
    )
    print(
        "=" * 40
    )
    print(
        f"Dataset root: {DATASET_ROOT}"
    )
    print(
        f"Annotation rows: {len(records):,}"
    )
    print(
        f"Total CAER-S test images: {len(test_images):,}"
    )
    print(
        f"Annotated subset coverage: {coverage * 100.0:.2f}%"
    )
    print(
        f"Boxes requiring clipping: {out_of_bounds_count:,}"
    )

    print(
        "\nAnnotated images per class:"
    )

    for label in DATASET_TO_MODEL_LABEL:
        print(
            f"  {label:10s}: {class_counts[label]:,}"
        )

    print(
        "\nPreflight status: PASS"
    )

    return records


def select_smoke_records(
    records,
    sample_count,
):
    """Select deterministic records distributed across the full subset."""
    if sample_count <= 0:
        return records

    if sample_count > len(records):
        raise ValueError(
            "Smoke-test sample count exceeds the number of annotations."
        )

    indices = np.linspace(
        0,
        len(records) - 1,
        sample_count,
        dtype=int,
    )

    return [
        records[index]
        for index in indices
    ]


def clip_box(
    box,
    width,
    height,
):
    """Clip a face box to valid image boundaries."""
    x1, y1, x2, y2 = box

    clipped_x1 = max(
        0,
        min(x1, width),
    )
    clipped_y1 = max(
        0,
        min(y1, height),
    )
    clipped_x2 = max(
        0,
        min(x2, width),
    )
    clipped_y2 = max(
        0,
        min(y2, height),
    )

    return (
        clipped_x1,
        clipped_y1,
        clipped_x2,
        clipped_y2,
    )


def empty_score_values():
    """Return empty values for all model score columns."""
    return {
        "score_anger": "",
        "score_contempt": "",
        "score_disgust": "",
        "score_fear": "",
        "score_happiness": "",
        "score_neutral": "",
        "score_sadness": "",
        "score_surprise": "",
        "score_sum": "",
    }


def score_values(
    scores,
):
    """Convert the model score dictionary to stable CSV fields."""
    return {
        "score_anger": scores["Anger"],
        "score_contempt": scores["Contempt"],
        "score_disgust": scores["Disgust"],
        "score_fear": scores["Fear"],
        "score_happiness": scores["Happiness"],
        "score_neutral": scores["Neutral"],
        "score_sadness": scores["Sadness"],
        "score_surprise": scores["Surprise"],
        "score_sum": sum(
            scores.values()
        ),
    }


def evaluate_record(
    record,
    model,
):
    """Evaluate one annotated target-face image."""
    image_path = (
        TEST_ROOT
        / record["relative_path"]
    )

    original_x1, original_y1, original_x2, original_y2 = (
        record["box"]
    )

    base_result = {
        "relative_path": record[
            "relative_path"
        ],
        "dataset_label": record[
            "dataset_label"
        ],
        "expected_model_label": record[
            "expected_model_label"
        ],
        "class_id": record[
            "class_id"
        ],
        "original_x1": original_x1,
        "original_y1": original_y1,
        "original_x2": original_x2,
        "original_y2": original_y2,
        "clipped_x1": "",
        "clipped_y1": "",
        "clipped_x2": "",
        "clipped_y2": "",
        "box_was_clipped": "",
        "crop_width": "",
        "crop_height": "",
        "status": "failed",
        "predicted_label": "",
        "confidence": "",
        **empty_score_values(),
        "correct": False,
        "failure_reason": "",
    }

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        base_result[
            "failure_reason"
        ] = "image_read_failure"
        return base_result

    height, width = image.shape[:2]

    clipped_box = clip_box(
        record["box"],
        width,
        height,
    )

    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = (
        clipped_box
    )

    box_was_clipped = (
        clipped_box
        != record["box"]
    )

    base_result.update(
        {
            "clipped_x1": clipped_x1,
            "clipped_y1": clipped_y1,
            "clipped_x2": clipped_x2,
            "clipped_y2": clipped_y2,
            "box_was_clipped": box_was_clipped,
            "crop_width": (
                clipped_x2
                - clipped_x1
            ),
            "crop_height": (
                clipped_y2
                - clipped_y1
            ),
        }
    )

    if (
        clipped_x2 <= clipped_x1
        or clipped_y2 <= clipped_y1
    ):
        base_result[
            "failure_reason"
        ] = "invalid_crop_after_clipping"
        return base_result

    face_crop = image[
        clipped_y1:clipped_y2,
        clipped_x1:clipped_x2,
    ]

    if face_crop.size == 0:
        base_result[
            "failure_reason"
        ] = "empty_face_crop"
        return base_result

    try:
        prediction = model.predict(
            face_crop
        )
    except Exception as exc:
        base_result[
            "failure_reason"
        ] = (
            f"model_exception:{type(exc).__name__}"
        )
        return base_result

    predicted_label = prediction.get(
        "emotion"
    )
    confidence = prediction.get(
        "confidence"
    )
    scores = prediction.get(
        "scores"
    )

    if not isinstance(
        scores,
        dict,
    ):
        base_result[
            "failure_reason"
        ] = "invalid_score_dictionary"
        return base_result

    if set(scores) != set(
        EXPECTED_MODEL_CLASSES
    ):
        base_result[
            "failure_reason"
        ] = "unexpected_model_classes"
        return base_result

    score_array = np.asarray(
        [
            scores[class_name]
            for class_name in EXPECTED_MODEL_CLASSES
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            score_array
        )
    ):
        base_result[
            "failure_reason"
        ] = "non_finite_scores"
        return base_result

    score_sum = float(
        np.sum(
            score_array
        )
    )

    if not math.isclose(
        score_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-5,
    ):
        base_result[
            "failure_reason"
        ] = "score_sum_not_one"
        return base_result

    if predicted_label not in EXPECTED_MODEL_CLASSES:
        base_result[
            "failure_reason"
        ] = "unexpected_predicted_label"
        return base_result

    if confidence is None:
        base_result[
            "failure_reason"
        ] = "missing_confidence"
        return base_result

    confidence = float(
        confidence
    )

    if not math.isfinite(
        confidence
    ):
        base_result[
            "failure_reason"
        ] = "non_finite_confidence"
        return base_result

    expected_model_label = record[
        "expected_model_label"
    ]

    base_result.update(
        {
            "status": "ok",
            "predicted_label": predicted_label,
            "confidence": confidence,
            **score_values(
                scores
            ),
            "correct": (
                predicted_label
                == expected_model_label
            ),
            "failure_reason": "",
        }
    )

    return base_result


def classification_metrics(
    rows,
):
    """Compute primary classification metrics with failures retained as FN."""
    total = len(
        rows
    )

    successful = [
        row
        for row in rows
        if row["status"] == "ok"
    ]

    failures = total - len(
        successful
    )

    correct = sum(
        bool(
            row["correct"]
        )
        for row in rows
    )

    accuracy = (
        correct / total
        if total
        else float("nan")
    )

    per_class = []

    for class_name in DATASET_TO_MODEL_LABEL.values():
        tp = sum(
            1
            for row in successful
            if (
                row["expected_model_label"] == class_name
                and row["predicted_label"] == class_name
            )
        )

        fp = sum(
            1
            for row in successful
            if (
                row["expected_model_label"] != class_name
                and row["predicted_label"] == class_name
            )
        )

        fn = sum(
            1
            for row in rows
            if (
                row["expected_model_label"] == class_name
                and not (
                    row["status"] == "ok"
                    and row["predicted_label"] == class_name
                )
            )
        )

        support = sum(
            1
            for row in rows
            if row["expected_model_label"] == class_name
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        per_class.append(
            {
                "class_name": class_name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = float(
        np.mean(
            [
                row["f1"]
                for row in per_class
            ]
        )
    )

    support_total = sum(
        row["support"]
        for row in per_class
    )

    weighted_f1 = (
        sum(
            row["f1"]
            * row["support"]
            for row in per_class
        )
        / support_total
        if support_total
        else float("nan")
    )

    contempt_count = sum(
        1
        for row in successful
        if row["predicted_label"] == "Contempt"
    )

    contempt_rate = (
        contempt_count / total
        if total
        else float("nan")
    )

    return {
        "total": total,
        "successful": len(
            successful
        ),
        "failures": failures,
        "availability": (
            len(successful)
            / total
            if total
            else float("nan")
        ),
        "correct": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "contempt_count": contempt_count,
        "contempt_rate": contempt_rate,
        "per_class": per_class,
    }


def write_results(
    rows,
    output_path,
):
    """Write detailed per-image benchmark results."""
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def write_summary(
    metrics,
    elapsed_seconds,
    output_path,
):
    """Write the scientific benchmark summary."""
    lines = [
        "CAER-S Emotion Recognition Validation",
        "=====================================",
        "",
        "Protocol",
        "--------",
        "Dataset: CAER-S",
        "Split: test",
        "Evaluation subset: annotated target-face subset from test.txt",
        f"Model: {MODEL_NAME}",
        f"Engine: {ENGINE}",
        "Target-face crop: annotation bounding box clipped to image bounds",
        "CAER-S Happy mapped to model Happiness",
        "CAER-S Sad mapped to model Sadness",
        "Model Contempt predictions retained and counted as incorrect",
        "",
        "Counts",
        "------",
        f"Annotated samples: {metrics['total']}",
        f"Successful predictions: {metrics['successful']}",
        f"Failed predictions: {metrics['failures']}",
        f"Availability: {metrics['availability']:.6f}",
        f"Correct predictions: {metrics['correct']}",
        "",
        "Primary Metrics",
        "---------------",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Macro F1: {metrics['macro_f1']:.6f}",
        f"Weighted F1: {metrics['weighted_f1']:.6f}",
        f"Contempt predictions: {metrics['contempt_count']}",
        f"Contempt prediction rate: {metrics['contempt_rate']:.6f}",
        "",
        "Per-Class Metrics",
        "-----------------",
    ]

    for row in metrics[
        "per_class"
    ]:
        lines.append(
            (
                f"{row['class_name']}: "
                f"support={row['support']}, "
                f"precision={row['precision']:.6f}, "
                f"recall={row['recall']:.6f}, "
                f"f1={row['f1']:.6f}"
            )
        )

    lines.extend(
        [
            "",
            "Runtime",
            "-------",
            f"Elapsed seconds: {elapsed_seconds:.3f}",
            "",
            "Interpretation",
            "--------------",
            (
                "Accuracy uses all annotated samples as the denominator. "
                "A failed prediction therefore cannot be counted as correct."
            ),
            (
                "Per-class recall and F1 retain failed samples as false "
                "negatives for their ground-truth classes."
            ),
            (
                "This validation covers the annotated target-face subset "
                "rather than the complete CAER-S test split."
            ),
            "",
        ]
    )

    output_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def validate_staged_outputs(
    rows,
    results_path,
    summary_path,
):
    """Validate staged evaluator outputs before final replacement."""
    if not results_path.is_file():
        raise RuntimeError(
            "Staged detailed result file was not created."
        )

    if not summary_path.is_file():
        raise RuntimeError(
            "Staged summary file was not created."
        )

    with results_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != RESULT_FIELDS:
            raise RuntimeError(
                "Staged result schema does not match the expected schema."
            )

        staged_rows = list(
            reader
        )

    if len(staged_rows) != len(
        rows
    ):
        raise RuntimeError(
            "Staged result row count does not match evaluated sample count."
        )

    if summary_path.stat().st_size == 0:
        raise RuntimeError(
            "Staged summary file is empty."
        )


def replace_owned_outputs(
    staging_dir,
):
    """Replace only evaluator-owned final outputs after staged validation."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_results = (
        staging_dir
        / RESULTS_PATH.name
    )
    staged_summary = (
        staging_dir
        / SUMMARY_PATH.name
    )

    os.replace(
        staged_results,
        RESULTS_PATH,
    )

    os.replace(
        staged_summary,
        SUMMARY_PATH,
    )


def run_evaluation(
    records,
):
    """Run PhysioTrack FaceEmotion on the supplied annotation records."""
    model = FaceEmotion(
        model_name=MODEL_NAME,
        engine=ENGINE,
    )

    rows = []

    start_time = time.perf_counter()

    for index, record in enumerate(
        records,
        start=1,
    ):
        rows.append(
            evaluate_record(
                record,
                model,
            )
        )

        if (
            index % 500 == 0
            or index == len(records)
        ):
            print(
                f"Processed {index:,}/{len(records):,}"
            )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    return (
        rows,
        elapsed_seconds,
    )


def run_smoke_test(
    records,
    sample_count,
):
    """Run the final evaluator logic without replacing benchmark outputs."""
    smoke_records = select_smoke_records(
        records,
        sample_count,
    )

    print(
        "\nRunning deterministic smoke test..."
    )

    rows, elapsed_seconds = run_evaluation(
        smoke_records
    )

    metrics = classification_metrics(
        rows
    )

    print(
        "\nSmoke-test summary"
    )
    print(
        "=" * 40
    )
    print(
        f"Samples: {metrics['total']}"
    )
    print(
        f"Successful: {metrics['successful']}"
    )
    print(
        f"Failures: {metrics['failures']}"
    )
    print(
        f"Elapsed seconds: {elapsed_seconds:.3f}"
    )

    if metrics[
        "failures"
    ] != 0:
        raise RuntimeError(
            "Smoke test contains failed predictions."
        )

    print(
        "Smoke-test status: PASS"
    )


def run_full_benchmark(
    records,
):
    """Run the full benchmark with staged transactional output replacement."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=".caers_emotion_eval_",
            dir=RESULTS_DIR,
        )
    )

    try:
        rows, elapsed_seconds = run_evaluation(
            records
        )

        metrics = classification_metrics(
            rows
        )

        staged_results = (
            staging_path
            / RESULTS_PATH.name
        )
        staged_summary = (
            staging_path
            / SUMMARY_PATH.name
        )

        write_results(
            rows,
            staged_results,
        )

        write_summary(
            metrics,
            elapsed_seconds,
            staged_summary,
        )

        validate_staged_outputs(
            rows,
            staged_results,
            staged_summary,
        )

        replace_owned_outputs(
            staging_path
        )

    finally:
        shutil.rmtree(
            staging_path,
            ignore_errors=True,
        )

    print(
        "\nCAER-S Emotion Recognition Validation"
    )
    print(
        "=" * 40
    )
    print(
        f"Annotated samples: {metrics['total']:,}"
    )
    print(
        f"Successful predictions: {metrics['successful']:,}"
    )
    print(
        f"Failed predictions: {metrics['failures']:,}"
    )
    print(
        f"Availability: {metrics['availability'] * 100.0:.2f}%"
    )
    print(
        f"Accuracy: {metrics['accuracy']:.6f}"
    )
    print(
        f"Macro F1: {metrics['macro_f1']:.6f}"
    )
    print(
        f"Weighted F1: {metrics['weighted_f1']:.6f}"
    )
    print(
        f"Contempt predictions: {metrics['contempt_count']:,}"
    )
    print(
        f"Elapsed seconds: {elapsed_seconds:.3f}"
    )
    print(
        "\nFinal evaluator outputs:"
    )
    print(
        RESULTS_PATH
    )
    print(
        SUMMARY_PATH
    )


def main():
    """Run CAER-S emotion-recognition benchmark validation."""
    args = parse_arguments()

    records = preflight()

    if args.preflight_only:
        return

    if args.smoke_test:
        run_smoke_test(
            records,
            args.smoke_test,
        )
        return

    run_full_benchmark(
        records
    )


if __name__ == "__main__":
    main()
