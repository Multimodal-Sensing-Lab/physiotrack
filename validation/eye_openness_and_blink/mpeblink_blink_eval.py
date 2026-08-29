from pathlib import Path
import argparse
import csv
import json
import time

import cv2
import numpy as np

from physiotrack.face.eyes import EyeOpenness
from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.models import Models


DATASET_ROOT = Path(
    r"C:\Users\xx901\Documents\PhysioTrack_Thesis"
    r"\datasets\MPEBlink2\mpeblink2.0"
)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTED_THRESHOLD = 0.44
SELECTED_MIN_CLOSED_FRAMES = 3

EVENT_IOU_THRESHOLD = 0.50

CALIBRATION_THRESHOLDS = np.arange(
    0.20,
    0.81,
    0.02,
)

CALIBRATION_MIN_CLOSED_FRAMES = [
    1,
    2,
    3,
]


def temporal_iou(first, second):
    """Calculate temporal IoU between two inclusive frame intervals."""
    start = max(first[0], second[0])
    end = min(first[1], second[1])

    intersection = max(
        0,
        end - start + 1,
    )

    first_length = (
        first[1] - first[0] + 1
    )

    second_length = (
        second[1] - second[0] + 1
    )

    union = (
        first_length
        + second_length
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def build_events(
    openness,
    threshold,
    min_closed_frames,
):
    """Convert an eye-openness sequence into blink intervals."""
    events = []

    closed_start = None
    closed_count = 0

    for frame_index, value in enumerate(openness):
        if not np.isfinite(value):
            closed_start = None
            closed_count = 0
            continue

        if value < threshold:
            if closed_start is None:
                closed_start = frame_index

            closed_count += 1
            continue

        if (
            closed_start is not None
            and closed_count
            >= min_closed_frames
        ):
            events.append(
                (
                    closed_start,
                    frame_index - 1,
                )
            )

        closed_start = None
        closed_count = 0

    if (
        closed_start is not None
        and closed_count
        >= min_closed_frames
    ):
        events.append(
            (
                closed_start,
                len(openness) - 1,
            )
        )

    return events


def match_events(
    predicted,
    ground_truth,
    iou_threshold,
):
    """Greedily match blink events using temporal IoU."""
    candidates = []

    for pred_index, pred_event in enumerate(predicted):
        for gt_index, gt_event in enumerate(ground_truth):
            overlap = temporal_iou(
                pred_event,
                gt_event,
            )

            if overlap >= iou_threshold:
                candidates.append(
                    (
                        overlap,
                        pred_index,
                        gt_index,
                    )
                )

    candidates.sort(
        reverse=True
    )

    used_predictions = set()
    used_ground_truth = set()

    matches = []

    for (
        overlap,
        pred_index,
        gt_index,
    ) in candidates:
        if pred_index in used_predictions:
            continue

        if gt_index in used_ground_truth:
            continue

        used_predictions.add(
            pred_index
        )

        used_ground_truth.add(
            gt_index
        )

        matches.append(
            (
                pred_index,
                gt_index,
                overlap,
            )
        )

    true_positive = len(matches)

    false_positive = (
        len(predicted)
        - true_positive
    )

    false_negative = (
        len(ground_truth)
        - true_positive
    )

    return (
        true_positive,
        false_positive,
        false_negative,
        matches,
    )


def binary_auc(labels, scores):
    """Calculate ROC AUC without external machine-learning dependencies."""
    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positive_count = int(
        np.sum(labels == 1)
    )

    negative_count = int(
        np.sum(labels == 0)
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        return float("nan")

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[order]

    ranks = np.empty(
        len(scores),
        dtype=np.float64,
    )

    start = 0

    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end]
            == sorted_scores[start]
        ):
            end += 1

        average_rank = (
            start + 1 + end
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    positive_rank_sum = np.sum(
        ranks[labels == 1]
    )

    auc = (
        positive_rank_sum
        - positive_count
        * (
            positive_count + 1
        )
        / 2.0
    ) / (
        positive_count
        * negative_count
    )

    return float(auc)


def make_gt_blink_mask(
    length,
    events,
):
    """Create a frame-level mask from annotated blink intervals."""
    mask = np.zeros(
        length,
        dtype=bool,
    )

    for start, end in events:
        start = max(
            0,
            int(start),
        )

        end = min(
            length - 1,
            int(end),
        )

        if end >= start:
            mask[
                start:end + 1
            ] = True

    return mask


def extract_split(split):
    """Extract PhysioTrack eye-openness sequences using ground-truth face boxes."""
    split_root = (
        DATASET_ROOT
        / split
    )

    if not split_root.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_root}"
        )

    model_path = Models.resolve(
        Models.Face.MediaPipe.Landmarks.face_landmarker
    )

    landmarks_model = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    eye_model = EyeOpenness()

    video_dirs = sorted(
        [
            path
            for path in split_root.iterdir()
            if path.is_dir()
        ],
        key=lambda path: int(
            path.name
        ),
    )

    records = []

    stats = {
        "videos": len(video_dirs),
        "annotation_frames": 0,
        "video_frames_read": 0,
        "person_sequences": 0,
        "valid_face_boxes": 0,
        "successful_eye_samples": 0,
        "missing_bbox": 0,
        "invalid_bbox": 0,
        "landmark_failures": 0,
        "video_frame_mismatches": 0,
        "video_read_failures": 0,
    }

    start_time = time.perf_counter()

    try:
        for video_number, video_dir in enumerate(
            video_dirs,
            start=1,
        ):
            annotation_path = (
                video_dir
                / "annotation_WFLW.json"
            )

            video_path = (
                video_dir
                / "video.mp4"
            )

            with open(
                annotation_path,
                "r",
                encoding="utf-8",
            ) as file:
                annotation = json.load(
                    file
                )

            expected_frames = int(
                annotation["length"]
            )

            stats[
                "annotation_frames"
            ] += expected_frames

            person_keys = sorted(
                [
                    key
                    for key in annotation
                    if key.startswith(
                        "person"
                    )
                ]
            )

            stats[
                "person_sequences"
            ] += len(
                person_keys
            )

            sequences = {
                person_key: np.full(
                    expected_frames,
                    np.nan,
                    dtype=np.float32,
                )
                for person_key
                in person_keys
            }

            capture = cv2.VideoCapture(
                str(video_path)
            )

            if not capture.isOpened():
                stats[
                    "video_read_failures"
                ] += 1
                continue

            fps = float(
                capture.get(
                    cv2.CAP_PROP_FPS
                )
            )

            if fps <= 0:
                capture.release()
                raise RuntimeError(
                    f"Invalid FPS in {video_path}"
                )

            frame_index = 0

            while frame_index < expected_frames:
                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                stats[
                    "video_frames_read"
                ] += 1

                image_height, image_width = (
                    frame.shape[:2]
                )

                for person_key in person_keys:
                    boxes = annotation[
                        person_key
                    ]["bbox"]

                    if frame_index >= len(
                        boxes
                    ):
                        stats[
                            "missing_bbox"
                        ] += 1
                        continue

                    bbox = boxes[
                        frame_index
                    ]

                    if bbox is None:
                        stats[
                            "missing_bbox"
                        ] += 1
                        continue

                    if (
                        not isinstance(
                            bbox,
                            (list, tuple),
                        )
                        or len(bbox) != 4
                    ):
                        stats[
                            "invalid_bbox"
                        ] += 1
                        continue

                    x, y, width, height = (
                        bbox
                    )

                    if (
                        width <= 0
                        or height <= 0
                    ):
                        stats[
                            "invalid_bbox"
                        ] += 1
                        continue

                    x1 = max(
                        0.0,
                        float(x),
                    )

                    y1 = max(
                        0.0,
                        float(y),
                    )

                    x2 = min(
                        float(image_width),
                        float(
                            x + width
                        ),
                    )

                    y2 = min(
                        float(image_height),
                        float(
                            y + height
                        ),
                    )

                    if (
                        x2 <= x1
                        or y2 <= y1
                    ):
                        stats[
                            "invalid_bbox"
                        ] += 1
                        continue

                    box = np.asarray(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        dtype=float,
                    )

                    stats[
                        "valid_face_boxes"
                    ] += 1

                    landmarks = (
                        landmarks_model.predict_face(
                            frame,
                            box,
                        )
                    )

                    if landmarks is None:
                        stats[
                            "landmark_failures"
                        ] += 1
                        continue

                    eye_result = (
                        eye_model.predict(
                            landmarks
                        )
                    )

                    openness = (
                        eye_result.get(
                            "mean_openness"
                        )
                    )

                    if (
                        openness is None
                        or not np.isfinite(
                            openness
                        )
                    ):
                        continue

                    sequences[
                        person_key
                    ][frame_index] = float(
                        openness
                    )

                    stats[
                        "successful_eye_samples"
                    ] += 1

                frame_index += 1

            capture.release()

            if frame_index != expected_frames:
                stats[
                    "video_frame_mismatches"
                ] += 1

            for person_key in person_keys:
                gt_events = [
                    (
                        int(event[0]),
                        int(event[1]),
                    )
                    for event
                    in annotation[
                        person_key
                    ]["blink"]
                ]

                records.append(
                    {
                        "video_id": int(
                            video_dir.name
                        ),
                        "person_id": (
                            person_key
                        ),
                        "fps": fps,
                        "length": (
                            expected_frames
                        ),
                        "openness": sequences[
                            person_key
                        ],
                        "gt_events": (
                            gt_events
                        ),
                    }
                )

            print(
                f"Processed {split} video "
                f"{video_number}/{len(video_dirs)}"
            )

    finally:
        landmarks_model.close()

    stats[
        "runtime_seconds"
    ] = (
        time.perf_counter()
        - start_time
    )

    if stats["valid_face_boxes"] > 0:
        stats[
            "eye_availability"
        ] = (
            stats[
                "successful_eye_samples"
            ]
            / stats[
                "valid_face_boxes"
            ]
        )
    else:
        stats[
            "eye_availability"
        ] = 0.0

    return records, stats


def evaluate_records(
    records,
    threshold,
    min_closed_frames,
    event_iou_threshold=0.50,
):
    """Evaluate blink events and eye-openness discrimination."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_predicted = 0
    total_ground_truth = 0

    matched_ious = []

    onset_errors_frames = []
    offset_errors_frames = []
    duration_errors_seconds = []

    sequence_count_errors = []
    sequence_rate_errors = []

    openness_values = []
    blink_labels = []

    sequence_rows = []

    for record in records:
        openness = record[
            "openness"
        ]

        gt_events = record[
            "gt_events"
        ]

        fps = record[
            "fps"
        ]

        predicted = build_events(
            openness,
            threshold,
            min_closed_frames,
        )

        (
            tp,
            fp,
            fn,
            matches,
        ) = match_events(
            predicted,
            gt_events,
            event_iou_threshold,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        total_predicted += len(
            predicted
        )

        total_ground_truth += len(
            gt_events
        )

        for (
            pred_index,
            gt_index,
            overlap,
        ) in matches:
            pred_event = predicted[
                pred_index
            ]

            gt_event = gt_events[
                gt_index
            ]

            matched_ious.append(
                overlap
            )

            onset_errors_frames.append(
                abs(
                    pred_event[0]
                    - gt_event[0]
                )
            )

            offset_errors_frames.append(
                abs(
                    pred_event[1]
                    - gt_event[1]
                )
            )

            pred_duration = (
                pred_event[1]
                - pred_event[0]
                + 1
            ) / fps

            gt_duration = (
                gt_event[1]
                - gt_event[0]
                + 1
            ) / fps

            duration_errors_seconds.append(
                abs(
                    pred_duration
                    - gt_duration
                )
            )

        video_minutes = (
            record["length"]
            / fps
            / 60.0
        )

        predicted_rate = (
            len(predicted)
            / video_minutes
            if video_minutes > 0
            else 0.0
        )

        gt_rate = (
            len(gt_events)
            / video_minutes
            if video_minutes > 0
            else 0.0
        )

        count_error = abs(
            len(predicted)
            - len(gt_events)
        )

        rate_error = abs(
            predicted_rate
            - gt_rate
        )

        sequence_count_errors.append(
            count_error
        )

        sequence_rate_errors.append(
            rate_error
        )

        sequence_rows.append(
            {
                "video_id": (
                    record["video_id"]
                ),
                "person_id": (
                    record["person_id"]
                ),
                "fps": fps,
                "frames": (
                    record["length"]
                ),
                "gt_blinks": len(
                    gt_events
                ),
                "predicted_blinks": len(
                    predicted
                ),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "gt_blink_rate_per_min": (
                    gt_rate
                ),
                "predicted_blink_rate_per_min": (
                    predicted_rate
                ),
                "absolute_count_error": (
                    count_error
                ),
                "absolute_rate_error_per_min": (
                    rate_error
                ),
            }
        )

        gt_mask = make_gt_blink_mask(
            record["length"],
            gt_events,
        )

        finite_mask = np.isfinite(
            openness
        )

        if np.any(finite_mask):
            valid_openness = (
                openness[
                    finite_mask
                ]
            )

            valid_labels = (
                gt_mask[
                    finite_mask
                ].astype(
                    np.int8
                )
            )

            openness_values.extend(
                valid_openness.tolist()
            )

            blink_labels.extend(
                valid_labels.tolist()
            )

    precision = (
        total_tp
        / (
            total_tp
            + total_fp
        )
        if (
            total_tp
            + total_fp
        )
        else 0.0
    )

    recall = (
        total_tp
        / (
            total_tp
            + total_fn
        )
        if (
            total_tp
            + total_fn
        )
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
        )
        else 0.0
    )

    openness_array = np.asarray(
        openness_values,
        dtype=np.float64,
    )

    labels_array = np.asarray(
        blink_labels,
        dtype=np.int8,
    )

    blink_openness = (
        openness_array[
            labels_array == 1
        ]
    )

    nonblink_openness = (
        openness_array[
            labels_array == 0
        ]
    )

    openness_auc = binary_auc(
        labels_array,
        -openness_array,
    )

    metrics = {
        "event_iou_threshold": (
            event_iou_threshold
        ),
        "threshold": threshold,
        "min_closed_frames": (
            min_closed_frames
        ),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_blinks": (
            total_predicted
        ),
        "ground_truth_blinks": (
            total_ground_truth
        ),
        "mean_matched_tiou": (
            float(
                np.mean(
                    matched_ious
                )
            )
            if matched_ious
            else float("nan")
        ),
        "median_matched_tiou": (
            float(
                np.median(
                    matched_ious
                )
            )
            if matched_ious
            else float("nan")
        ),
        "mean_onset_error_frames": (
            float(
                np.mean(
                    onset_errors_frames
                )
            )
            if onset_errors_frames
            else float("nan")
        ),
        "mean_offset_error_frames": (
            float(
                np.mean(
                    offset_errors_frames
                )
            )
            if offset_errors_frames
            else float("nan")
        ),
        "mean_duration_error_seconds": (
            float(
                np.mean(
                    duration_errors_seconds
                )
            )
            if duration_errors_seconds
            else float("nan")
        ),
        "blink_count_mae_per_sequence": (
            float(
                np.mean(
                    sequence_count_errors
                )
            )
            if sequence_count_errors
            else float("nan")
        ),
        "blink_rate_mae_per_min": (
            float(
                np.mean(
                    sequence_rate_errors
                )
            )
            if sequence_rate_errors
            else float("nan")
        ),
        "eye_openness_auc": (
            openness_auc
        ),
        "blink_openness_mean": (
            float(
                np.mean(
                    blink_openness
                )
            )
            if len(blink_openness)
            else float("nan")
        ),
        "blink_openness_median": (
            float(
                np.median(
                    blink_openness
                )
            )
            if len(blink_openness)
            else float("nan")
        ),
        "nonblink_openness_mean": (
            float(
                np.mean(
                    nonblink_openness
                )
            )
            if len(nonblink_openness)
            else float("nan")
        ),
        "nonblink_openness_median": (
            float(
                np.median(
                    nonblink_openness
                )
            )
            if len(nonblink_openness)
            else float("nan")
        ),
    }

    return metrics, sequence_rows


def save_sequence_results(
    split,
    sequence_rows,
):
    path = (
        RESULTS_DIR
        / f"mpeblink_{split}_sequence_results.csv"
    )

    if not sequence_rows:
        return path

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                sequence_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            sequence_rows
        )

    return path


def save_summary(
    split,
    stats,
    metrics,
):
    path = (
        RESULTS_DIR
        / f"mpeblink_{split}_summary.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "MPEBlink 2.0 Eye Openness and Blink Evaluation\n"
        )
        file.write(
            "============================================\n\n"
        )

        file.write(
            f"Split: {split}\n"
        )
        file.write(
            "Face initialization: ground-truth bounding boxes\n"
        )
        file.write(
            "Eye descriptor: PhysioTrack EyeOpenness\n"
        )
        file.write(
            "Blink method: threshold-based temporal detector\n"
        )
        file.write(
            f"Blink threshold: {metrics['threshold']:.4f}\n"
        )
        file.write(
            f"Minimum closed frames: {metrics['min_closed_frames']}\n"
        )
        file.write(
            f"Event temporal IoU threshold: {metrics['event_iou_threshold']:.2f}\n\n"
        )

        file.write(
            "Dataset processing\n"
        )
        file.write(
            "------------------\n"
        )
        file.write(
            f"Videos: {stats['videos']}\n"
        )
        file.write(
            f"Person sequences: {stats['person_sequences']}\n"
        )
        file.write(
            f"Annotation frames: {stats['annotation_frames']}\n"
        )
        file.write(
            f"Video frames read: {stats['video_frames_read']}\n"
        )
        file.write(
            f"Valid face-box samples: {stats['valid_face_boxes']}\n"
        )
        file.write(
            f"Successful eye-openness samples: {stats['successful_eye_samples']}\n"
        )
        file.write(
            f"Eye-openness availability: {stats['eye_availability'] * 100.0:.2f}%\n"
        )
        file.write(
            f"Missing bounding boxes: {stats['missing_bbox']}\n"
        )
        file.write(
            f"Invalid bounding boxes: {stats['invalid_bbox']}\n"
        )
        file.write(
            f"Landmark failures: {stats['landmark_failures']}\n"
        )
        file.write(
            f"Video frame mismatches: {stats['video_frame_mismatches']}\n"
        )
        file.write(
            f"Video read failures: {stats['video_read_failures']}\n"
        )
        file.write(
            f"Runtime: {stats['runtime_seconds'] / 60.0:.2f} minutes\n\n"
        )

        file.write(
            "Blink event evaluation\n"
        )
        file.write(
            "----------------------\n"
        )
        file.write(
            f"Ground-truth blinks: {metrics['ground_truth_blinks']}\n"
        )
        file.write(
            f"Predicted blinks: {metrics['predicted_blinks']}\n"
        )
        file.write(
            f"True positives: {metrics['true_positive']}\n"
        )
        file.write(
            f"False positives: {metrics['false_positive']}\n"
        )
        file.write(
            f"False negatives: {metrics['false_negative']}\n"
        )
        file.write(
            f"Precision: {metrics['precision']:.6f}\n"
        )
        file.write(
            f"Recall: {metrics['recall']:.6f}\n"
        )
        file.write(
            f"F1: {metrics['f1']:.6f}\n"
        )
        file.write(
            f"Mean matched temporal IoU: {metrics['mean_matched_tiou']:.6f}\n"
        )
        file.write(
            f"Median matched temporal IoU: {metrics['median_matched_tiou']:.6f}\n"
        )
        file.write(
            f"Mean onset error: {metrics['mean_onset_error_frames']:.4f} frames\n"
        )
        file.write(
            f"Mean offset error: {metrics['mean_offset_error_frames']:.4f} frames\n"
        )
        file.write(
            f"Mean blink-duration error: {metrics['mean_duration_error_seconds']:.6f} s\n"
        )
        file.write(
            f"Blink-count MAE per sequence: {metrics['blink_count_mae_per_sequence']:.6f}\n"
        )
        file.write(
            f"Blink-rate MAE: {metrics['blink_rate_mae_per_min']:.6f} blinks/min\n\n"
        )

        file.write(
            "Eye-openness analysis\n"
        )
        file.write(
            "---------------------\n"
        )
        file.write(
            f"Blink-frame openness mean: {metrics['blink_openness_mean']:.6f}\n"
        )
        file.write(
            f"Blink-frame openness median: {metrics['blink_openness_median']:.6f}\n"
        )
        file.write(
            f"Non-blink openness mean: {metrics['nonblink_openness_mean']:.6f}\n"
        )
        file.write(
            f"Non-blink openness median: {metrics['nonblink_openness_median']:.6f}\n"
        )
        file.write(
            f"Blink-vs-non-blink ROC AUC using negative openness: {metrics['eye_openness_auc']:.6f}\n"
        )

    return path


def calibrate(records):
    """Select blink parameters on the validation split only."""
    results = []

    for min_closed_frames in (
        CALIBRATION_MIN_CLOSED_FRAMES
    ):
        for threshold in (
            CALIBRATION_THRESHOLDS
        ):
            metrics, _ = evaluate_records(
                records,
                float(threshold),
                min_closed_frames,
                EVENT_IOU_THRESHOLD,
            )

            results.append(
                metrics
            )

            print(
                f"threshold={threshold:.2f} | "
                f"min_frames={min_closed_frames} | "
                f"P={metrics['precision']:.4f} | "
                f"R={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f}"
            )

    results.sort(
        key=lambda item: (
            item["f1"],
            item["mean_matched_tiou"],
        ),
        reverse=True,
    )

    path = (
        RESULTS_DIR
        / "mpeblink_val_calibration.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    return results[0], path


def run_validation_calibration():
    records, stats = extract_split(
        "val"
    )

    best, calibration_path = (
        calibrate(
            records
        )
    )

    print(
        "\n=== Selected Validation Configuration ==="
    )
    print(
        f"blink_threshold: "
        f"{best['threshold']:.4f}"
    )
    print(
        f"min_closed_frames: "
        f"{best['min_closed_frames']}"
    )
    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )
    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )

    print(
        "\nCalibration saved:"
    )
    print(
        calibration_path
    )

    print(
        "\nExtraction runtime:"
    )
    print(
        f"{stats['runtime_seconds'] / 60.0:.2f} minutes"
    )


def run_final_test():
    print(
        "Running final MPEBlink 2.0 test evaluation."
    )
    print(
        "The parameters below were frozen using the validation split."
    )
    print(
        f"blink_threshold = "
        f"{SELECTED_THRESHOLD:.2f}"
    )
    print(
        f"min_closed_frames = "
        f"{SELECTED_MIN_CLOSED_FRAMES}"
    )

    records, stats = extract_split(
        "test"
    )

    metrics, sequence_rows = (
        evaluate_records(
            records,
            SELECTED_THRESHOLD,
            SELECTED_MIN_CLOSED_FRAMES,
            EVENT_IOU_THRESHOLD,
        )
    )

    sequence_path = (
        save_sequence_results(
            "test",
            sequence_rows,
        )
    )

    summary_path = save_summary(
        "test",
        stats,
        metrics,
    )

    print(
        "\n=== Final Test Results ==="
    )

    print(
        f"Videos: "
        f"{stats['videos']}"
    )

    print(
        f"Person sequences: "
        f"{stats['person_sequences']}"
    )

    print(
        f"Eye-openness availability: "
        f"{stats['eye_availability'] * 100.0:.2f}%"
    )

    print(
        f"Landmark failures: "
        f"{stats['landmark_failures']}"
    )

    print(
        f"Ground-truth blinks: "
        f"{metrics['ground_truth_blinks']}"
    )

    print(
        f"Predicted blinks: "
        f"{metrics['predicted_blinks']}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Mean matched temporal IoU: "
        f"{metrics['mean_matched_tiou']:.4f}"
    )

    print(
        f"Blink-count MAE per sequence: "
        f"{metrics['blink_count_mae_per_sequence']:.4f}"
    )

    print(
        f"Blink-rate MAE: "
        f"{metrics['blink_rate_mae_per_min']:.4f} blinks/min"
    )

    print(
        f"Mean blink-duration error: "
        f"{metrics['mean_duration_error_seconds']:.4f} s"
    )

    print(
        f"Eye-openness ROC AUC: "
        f"{metrics['eye_openness_auc']:.4f}"
    )

    print(
        f"Blink-frame median openness: "
        f"{metrics['blink_openness_median']:.4f}"
    )

    print(
        f"Non-blink median openness: "
        f"{metrics['nonblink_openness_median']:.4f}"
    )

    print(
        f"Runtime: "
        f"{stats['runtime_seconds'] / 60.0:.2f} minutes"
    )

    print(
        "\nSaved:"
    )
    print(
        summary_path
    )
    print(
        sequence_path
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PhysioTrack eye openness "
            "and blink detection on MPEBlink 2.0."
        )
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Run parameter calibration on the "
            "validation split instead of the final test."
        ),
    )

    args = parser.parse_args()

    if args.calibrate:
        run_validation_calibration()
    else:
        run_final_test()


if __name__ == "__main__":
    main()