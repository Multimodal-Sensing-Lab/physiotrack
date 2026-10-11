from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face.blink import BlinkDetector
from physiotrack.face.eyes import EyeOpenness
from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.models import Models


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DATASET_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "MPEBlink2"
    / "mpeblink2.0"
)

RESULTS_DIR = (
    SCRIPT_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CALIBRATION_CSV = (
    RESULTS_DIR
    / "mpeblink_val_calibration.csv"
)

TEST_SUMMARY_PATH = (
    RESULTS_DIR
    / "mpeblink_test_summary.txt"
)

TEST_SEQUENCE_RESULTS_PATH = (
    RESULTS_DIR
    / "mpeblink_test_sequence_results.csv"
)

SELECTED_THRESHOLD = 0.22
SELECTED_MIN_CLOSED_FRAMES = 3

EVENT_IOU_THRESHOLD = 0.50

CALIBRATION_THRESHOLDS = np.arange(
    0.10,
    0.51,
    0.01,
)

CALIBRATION_MIN_CLOSED_FRAMES = [
    1,
    2,
    3,
    4,
]

EXPECTED_SPLIT_VIDEOS = {
    "val": 169,
    "test": 212,
}

EVALUATION_SPLITS = [
    "val",
    "test",
]


def make_staging_dir(prefix):
    """Create an evaluator-owned staging directory under results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=RESULTS_DIR))


def replace_owned_files(staged_to_final, staging_dir):
    """Replace only evaluator-owned files with rollback protection."""
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = []
    installed = []
    try:
        for staged_path, final_path in staged_to_final:
            if not staged_path.is_file():
                raise RuntimeError(f"Missing staged evaluator output: {staged_path}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
        for staged_path, final_path in staged_to_final:
            os.replace(staged_path, final_path)
            installed.append(final_path)
    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()
        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, final_path)
        raise


def validate_calibration_output(path):
    """Validate staged calibration output before replacement."""
    if not path.is_file():
        raise RuntimeError(f"Calibration CSV was not created: {path}")
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected_rows = len(CALIBRATION_THRESHOLDS) * len(CALIBRATION_MIN_CLOSED_FRAMES)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} calibration rows, found {len(rows)}.")
    required = {"threshold", "min_closed_frames", "precision", "recall", "f1", "mean_matched_tiou"}
    if not rows or not required.issubset(rows[0].keys()):
        raise RuntimeError("Staged calibration CSV is missing required columns.")
    seen = set()
    best_key = None
    best_threshold = None
    best_min_frames = None
    for row in rows:
        threshold = float(row["threshold"])
        min_frames = int(float(row["min_closed_frames"]))
        values = np.asarray([threshold, min_frames, float(row["precision"]), float(row["recall"]), float(row["f1"]), float(row["mean_matched_tiou"])], dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Staged calibration CSV contains non-finite required values.")
        key = (round(threshold, 12), min_frames)
        if key in seen:
            raise RuntimeError(f"Duplicate calibration configuration found: {key}")
        seen.add(key)
        rank = (float(row["f1"]), float(row["mean_matched_tiou"]))
        if best_key is None or rank > best_key:
            best_key = rank
            best_threshold = threshold
            best_min_frames = min_frames
    if len(seen) != expected_rows:
        raise RuntimeError("Calibration configuration grid is incomplete.")
    if not np.isclose(best_threshold, SELECTED_THRESHOLD, rtol=0.0, atol=1e-12) or best_min_frames != SELECTED_MIN_CLOSED_FRAMES:
        raise RuntimeError("Staged calibration does not reproduce the frozen selected configuration.")


def validate_test_outputs(summary_path, sequence_path):
    """Validate staged final-test outputs before replacement."""
    if not summary_path.is_file() or not sequence_path.is_file():
        raise RuntimeError("Staged final-test outputs are incomplete.")
    with sequence_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 687:
        raise RuntimeError(f"Expected 687 test person-sequence rows, found {len(rows)}.")
    required = {"video_id", "person_id", "fps", "frames", "gt_blinks", "predicted_blinks", "true_positive", "false_positive", "false_negative", "absolute_count_error", "absolute_rate_error_per_min"}
    if not rows or not required.issubset(rows[0].keys()):
        raise RuntimeError("Staged sequence-results CSV is missing required columns.")
    keys = set()
    gt_total = pred_total = tp_total = fp_total = fn_total = 0
    for row in rows:
        key = (int(row["video_id"]), row["person_id"])
        if key in keys:
            raise RuntimeError(f"Duplicate test sequence row found: {key}")
        keys.add(key)
        fps = float(row["fps"]); frames = int(float(row["frames"]))
        gt = int(float(row["gt_blinks"])); pred = int(float(row["predicted_blinks"]))
        tp = int(float(row["true_positive"])); fp = int(float(row["false_positive"])); fn = int(float(row["false_negative"]))
        count_error = float(row["absolute_count_error"]); rate_error = float(row["absolute_rate_error_per_min"])
        vals=np.asarray([fps,frames,gt,pred,tp,fp,fn,count_error,rate_error],dtype=float)
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(f"Non-finite staged sequence values for {key}.")
        if fps <= 0 or frames <= 0:
            raise RuntimeError(f"Invalid fps/frame accounting for {key}.")
        if pred != tp + fp or gt != tp + fn:
            raise RuntimeError(f"Event accounting mismatch for {key}.")
        if not np.isclose(count_error, abs(pred-gt), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Blink-count error mismatch for {key}.")
        gt_rate = gt / frames * fps * 60.0
        pred_rate = pred / frames * fps * 60.0
        if not np.isclose(rate_error, abs(pred_rate-gt_rate), rtol=0.0, atol=1e-9):
            raise RuntimeError(f"Blink-rate error mismatch for {key}.")
        gt_total += gt; pred_total += pred; tp_total += tp; fp_total += fp; fn_total += fn
    precision = tp_total/(tp_total+fp_total) if tp_total+fp_total else 0.0
    recall = tp_total/(tp_total+fn_total) if tp_total+fn_total else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    summary = summary_path.read_text(encoding="utf-8")
    expected = ["Split: test", "Blink threshold: 0.2200", "Minimum closed frames: 3", "Videos: 212", "Person sequences: 687", f"Ground-truth blinks: {gt_total}", f"Predicted blinks: {pred_total}", f"True positives: {tp_total}", f"False positives: {fp_total}", f"False negatives: {fn_total}", f"Precision: {precision:.6f}", f"Recall: {recall:.6f}", f"F1: {f1:.6f}"]
    for line in expected:
        if line not in summary:
            raise RuntimeError(f"Staged test summary is inconsistent with sequence results: {line}")


def remove_file_if_exists(path):
    """Remove one generated file if it exists."""
    if path.is_file():
        path.unlink()


def clean_calibration_outputs():
    """Remove only outputs owned by validation calibration."""
    remove_file_if_exists(
        CALIBRATION_CSV
    )


def clean_test_outputs():
    """Remove only final-test outputs owned by this evaluator."""
    paths = [
        TEST_SUMMARY_PATH,
        TEST_SEQUENCE_RESULTS_PATH,
    ]

    for path in paths:
        remove_file_if_exists(
            path
        )


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
    fps,
):
    """Convert EyeOpenness values into events using PhysioTrack BlinkDetector."""
    detector = BlinkDetector(
        threshold=threshold,
        fps=fps,
        min_closed_frames=min_closed_frames,
    )

    events = []

    for frame_index, value in enumerate(openness):
        detector_value = (
            float(value)
            if np.isfinite(value)
            else None
        )

        result = detector.update(
            detector_value,
            person_id=0,
        )

        if not result["blink"]:
            continue

        duration_seconds = result[
            "blink_duration"
        ]

        if duration_seconds is None:
            continue

        closed_frames = int(
            round(
                duration_seconds
                * fps
            )
        )

        if closed_frames < 1:
            continue

        end_frame = frame_index - 1
        start_frame = (
            end_frame
            - closed_frames
            + 1
        )

        events.append(
            (
                start_frame,
                end_frame,
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


def validate_annotation(
    annotation,
    annotation_path,
):
    """Validate the annotation schema while preserving benchmark semantics.

    Blink events whose boundaries extend beyond the declared video length are
    recorded as dataset-integrity observations rather than rejected. The
    event-level evaluation retains the original annotated boundaries, matching
    the previous benchmark protocol. Frame-level eye-openness labels remain
    clipped safely by make_gt_blink_mask().
    """
    if "length" not in annotation:
        raise RuntimeError(
            f"Missing length in annotation: {annotation_path}"
        )

    expected_frames = int(
        annotation["length"]
    )

    if expected_frames <= 0:
        raise RuntimeError(
            f"Invalid annotation length in: {annotation_path}"
        )

    person_keys = sorted(
        [
            key
            for key in annotation
            if key.startswith(
                "person"
            )
        ]
    )

    if not person_keys:
        raise RuntimeError(
            f"No person annotations found in: {annotation_path}"
        )

    integrity = {
        "out_of_range_blink_events": 0,
    }

    for person_key in person_keys:
        person = annotation[
            person_key
        ]

        if "bbox" not in person:
            raise RuntimeError(
                f"Missing bbox for {person_key} in: {annotation_path}"
            )

        if "blink" not in person:
            raise RuntimeError(
                f"Missing blink events for {person_key} in: "
                f"{annotation_path}"
            )

        if len(
            person["bbox"]
        ) != expected_frames:
            raise RuntimeError(
                f"Bounding-box length mismatch for {person_key} in: "
                f"{annotation_path}"
            )

        for event in person[
            "blink"
        ]:
            # MPEBlink 2.0 stores each blink annotation as a sequence
            # whose first two values are the inclusive start and end
            # frame indices. Additional values are dataset metadata and
            # are not used by this evaluation protocol.
            if (
                not isinstance(
                    event,
                    (list, tuple),
                )
                or len(event) < 2
            ):
                raise RuntimeError(
                    f"Invalid blink annotation for {person_key} in: "
                    f"{annotation_path}"
                )

            try:
                start = int(
                    event[0]
                )

                end = int(
                    event[1]
                )
            except (
                TypeError,
                ValueError,
            ) as error:
                raise RuntimeError(
                    f"Non-numeric blink boundary for {person_key} in: "
                    f"{annotation_path}"
                ) from error

            if end < start:
                raise RuntimeError(
                    f"Blink interval has end before start for "
                    f"{person_key} in: {annotation_path}"
                )

            if (
                start < 0
                or end >= expected_frames
            ):
                integrity[
                    "out_of_range_blink_events"
                ] += 1

    return (
        expected_frames,
        person_keys,
        integrity,
    )


def validate_dataset_layout():
    """Validate only the validation and test splits used by this benchmark."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "MPEBlink 2.0 dataset root was not found:\n"
            f"{DATASET_ROOT}\n\n"
            "Expected workspace layout:\n"
            "datasets/MPEBlink2/mpeblink2.0/"
        )

    accounting = {}

    # The benchmark uses validation for parameter selection and test for the
    # frozen final evaluation. The training split is intentionally excluded
    # from reproducibility preflight because it is not used by this protocol.
    for split in EVALUATION_SPLITS:
        split_root = (
            DATASET_ROOT
            / split
        )

        if not split_root.is_dir():
            raise FileNotFoundError(
                f"Dataset split not found: {split_root}"
            )

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

        expected_videos = (
            EXPECTED_SPLIT_VIDEOS[
                split
            ]
        )

        if len(
            video_dirs
        ) != expected_videos:
            raise RuntimeError(
                f"Unexpected {split} video-directory count: "
                f"{len(video_dirs)} != {expected_videos}"
            )

        annotation_frames = 0
        person_sequences = 0
        blink_events = 0
        out_of_range_blink_events = 0

        for video_dir in video_dirs:
            annotation_path = (
                video_dir
                / "annotation_WFLW.json"
            )

            video_path = (
                video_dir
                / "video.mp4"
            )

            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Annotation file not found: {annotation_path}"
                )

            if not video_path.is_file():
                raise FileNotFoundError(
                    f"Video file not found: {video_path}"
                )

            with annotation_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                annotation = json.load(
                    file
                )

            (
                expected_frames,
                person_keys,
                integrity,
            ) = validate_annotation(
                annotation,
                annotation_path,
            )

            annotation_frames += (
                expected_frames
            )

            person_sequences += len(
                person_keys
            )

            out_of_range_blink_events += integrity[
                "out_of_range_blink_events"
            ]

            for person_key in person_keys:
                blink_events += len(
                    annotation[
                        person_key
                    ][
                        "blink"
                    ]
                )

        accounting[
            split
        ] = {
            "videos": len(
                video_dirs
            ),
            "annotation_frames": (
                annotation_frames
            ),
            "person_sequences": (
                person_sequences
            ),
            "blink_events": (
                blink_events
            ),
            "out_of_range_blink_events": (
                out_of_range_blink_events
            ),
        }

    return accounting


def extract_split(split):
    """Extract PhysioTrack eye-openness sequences using ground-truth face boxes."""
    split_root = (
        DATASET_ROOT
        / split
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
        "out_of_range_blink_events": 0,
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

            with annotation_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                annotation = json.load(
                    file
                )

            (
                expected_frames,
                person_keys,
                integrity,
            ) = validate_annotation(
                annotation,
                annotation_path,
            )

            stats[
                "annotation_frames"
            ] += expected_frames

            stats[
                "person_sequences"
            ] += len(
                person_keys
            )

            stats[
                "out_of_range_blink_events"
            ] += integrity[
                "out_of_range_blink_events"
            ]

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
                    bbox = annotation[
                        person_key
                    ][
                        "bbox"
                    ][
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
                            landmarks,
                            image_size=(
                                image_width,
                                image_height,
                            ),
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
                    ][
                        frame_index
                    ] = float(
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
                    ][
                        "blink"
                    ]
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

    if stats[
        "valid_face_boxes"
    ] > 0:
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

    return (
        records,
        stats,
    )


def evaluate_records(
    records,
    threshold,
    min_closed_frames,
    event_iou_threshold=EVENT_IOU_THRESHOLD,
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
            fps,
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
            record[
                "length"
            ]
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
                    record[
                        "video_id"
                    ]
                ),
                "person_id": (
                    record[
                        "person_id"
                    ]
                ),
                "fps": fps,
                "frames": (
                    record[
                        "length"
                    ]
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
            record[
                "length"
            ],
            gt_events,
        )

        finite_mask = np.isfinite(
            openness
        )

        if np.any(
            finite_mask
        ):
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
        "matched_blinks": len(
            matched_ious
        ),
        "mean_matched_tiou": (
            float(
                np.mean(
                    matched_ious
                )
            )
            if matched_ious
            else float(
                "nan"
            )
        ),
        "median_matched_tiou": (
            float(
                np.median(
                    matched_ious
                )
            )
            if matched_ious
            else float(
                "nan"
            )
        ),
        "mean_onset_error_frames": (
            float(
                np.mean(
                    onset_errors_frames
                )
            )
            if onset_errors_frames
            else float(
                "nan"
            )
        ),
        "mean_offset_error_frames": (
            float(
                np.mean(
                    offset_errors_frames
                )
            )
            if offset_errors_frames
            else float(
                "nan"
            )
        ),
        "mean_duration_error_seconds": (
            float(
                np.mean(
                    duration_errors_seconds
                )
            )
            if duration_errors_seconds
            else float(
                "nan"
            )
        ),
        "blink_count_mae_per_sequence": (
            float(
                np.mean(
                    sequence_count_errors
                )
            )
            if sequence_count_errors
            else float(
                "nan"
            )
        ),
        "blink_rate_mae_per_min": (
            float(
                np.mean(
                    sequence_rate_errors
                )
            )
            if sequence_rate_errors
            else float(
                "nan"
            )
        ),
        "eye_openness_auc": (
            openness_auc
        ),
        "eye_openness_samples": int(
            len(
                openness_array
            )
        ),
        "blink_eye_samples": int(
            len(
                blink_openness
            )
        ),
        "nonblink_eye_samples": int(
            len(
                nonblink_openness
            )
        ),
        "blink_openness_mean": (
            float(
                np.mean(
                    blink_openness
                )
            )
            if len(
                blink_openness
            )
            else float(
                "nan"
            )
        ),
        "blink_openness_median": (
            float(
                np.median(
                    blink_openness
                )
            )
            if len(
                blink_openness
            )
            else float(
                "nan"
            )
        ),
        "nonblink_openness_mean": (
            float(
                np.mean(
                    nonblink_openness
                )
            )
            if len(
                nonblink_openness
            )
            else float(
                "nan"
            )
        ),
        "nonblink_openness_median": (
            float(
                np.median(
                    nonblink_openness
                )
            )
            if len(
                nonblink_openness
            )
            else float(
                "nan"
            )
        ),
    }

    return (
        metrics,
        sequence_rows,
    )


def save_sequence_results(
    split,
    sequence_rows,
    output_path=None,
):
    """Save per-person sequence results for independent verification."""
    path = output_path if output_path is not None else (RESULTS_DIR / f"mpeblink_{split}_sequence_results.csv")

    if not sequence_rows:
        raise RuntimeError(
            f"No sequence rows were produced for split: {split}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                sequence_rows[
                    0
                ].keys()
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
    output_path=None,
):
    """Save the evaluator-level quantitative summary."""
    path = output_path if output_path is not None else (RESULTS_DIR / f"mpeblink_{split}_summary.txt")

    with path.open(
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
            "Landmarks: PhysioTrack FaceLandmarks using MediaPipe\n"
        )
        file.write(
            "Eye descriptor: PhysioTrack EyeOpenness\n"
        )
        file.write(
            "Blink method: threshold-based temporal detector\n"
        )
        file.write(
            "Parameter selection: validation split only\n"
        )
        file.write(
            f"Blink threshold: {metrics['threshold']:.4f}\n"
        )
        file.write(
            f"Minimum closed frames: "
            f"{metrics['min_closed_frames']}\n"
        )
        file.write(
            f"Event temporal IoU threshold: "
            f"{metrics['event_iou_threshold']:.2f}\n\n"
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
            f"Successful eye-openness samples: "
            f"{stats['successful_eye_samples']}\n"
        )
        file.write(
            f"Eye-openness availability: "
            f"{stats['eye_availability'] * 100.0:.2f}%\n"
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
            f"Video frame mismatches: "
            f"{stats['video_frame_mismatches']}\n"
        )
        file.write(
            f"Video read failures: {stats['video_read_failures']}\n"
        )
        file.write(
            f"Out-of-range blink annotations retained: "
            f"{stats['out_of_range_blink_events']}\n"
        )
        file.write(
            f"Runtime: "
            f"{stats['runtime_seconds'] / 60.0:.2f} minutes\n\n"
        )

        file.write(
            "Blink event evaluation\n"
        )
        file.write(
            "----------------------\n"
        )
        file.write(
            f"Ground-truth blinks: "
            f"{metrics['ground_truth_blinks']}\n"
        )
        file.write(
            f"Predicted blinks: "
            f"{metrics['predicted_blinks']}\n"
        )
        file.write(
            f"True positives: "
            f"{metrics['true_positive']}\n"
        )
        file.write(
            f"False positives: "
            f"{metrics['false_positive']}\n"
        )
        file.write(
            f"False negatives: "
            f"{metrics['false_negative']}\n"
        )
        file.write(
            f"Precision: "
            f"{metrics['precision']:.6f}\n"
        )
        file.write(
            f"Recall: "
            f"{metrics['recall']:.6f}\n"
        )
        file.write(
            f"F1: "
            f"{metrics['f1']:.6f}\n"
        )
        file.write(
            f"Mean matched temporal IoU: "
            f"{metrics['mean_matched_tiou']:.6f}\n"
        )
        file.write(
            f"Median matched temporal IoU: "
            f"{metrics['median_matched_tiou']:.6f}\n"
        )
        file.write(
            f"Mean onset error: "
            f"{metrics['mean_onset_error_frames']:.4f} frames\n"
        )
        file.write(
            f"Mean offset error: "
            f"{metrics['mean_offset_error_frames']:.4f} frames\n"
        )
        file.write(
            f"Mean blink-duration error: "
            f"{metrics['mean_duration_error_seconds']:.6f} s\n"
        )
        file.write(
            f"Blink-count MAE per sequence: "
            f"{metrics['blink_count_mae_per_sequence']:.6f}\n"
        )
        file.write(
            f"Blink-rate MAE: "
            f"{metrics['blink_rate_mae_per_min']:.6f} blinks/min\n\n"
        )

        file.write(
            "Eye-openness analysis\n"
        )
        file.write(
            "---------------------\n"
        )
        file.write(
            "ROC AUC population: finite EyeOpenness samples only\n"
        )
        file.write(
            f"Eye-openness samples: "
            f"{metrics['eye_openness_samples']}\n"
        )
        file.write(
            f"Blink-frame eye samples: "
            f"{metrics['blink_eye_samples']}\n"
        )
        file.write(
            f"Non-blink eye samples: "
            f"{metrics['nonblink_eye_samples']}\n"
        )
        file.write(
            f"Blink-frame openness mean: "
            f"{metrics['blink_openness_mean']:.6f}\n"
        )
        file.write(
            f"Blink-frame openness median: "
            f"{metrics['blink_openness_median']:.6f}\n"
        )
        file.write(
            f"Non-blink openness mean: "
            f"{metrics['nonblink_openness_mean']:.6f}\n"
        )
        file.write(
            f"Non-blink openness median: "
            f"{metrics['nonblink_openness_median']:.6f}\n"
        )
        file.write(
            "Blink-vs-non-blink ROC AUC using negative openness: "
            f"{metrics['eye_openness_auc']:.6f}\n"
        )

    return path


def calibrate(records, output_path=CALIBRATION_CSV):
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
                float(
                    threshold
                ),
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
            item[
                "f1"
            ],
            item[
                "mean_matched_tiou"
            ],
        ),
        reverse=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    return (
        results[
            0
        ],
        output_path,
    )


def run_validation_calibration():
    """Run validation-only parameter selection transactionally."""
    staging_dir = make_staging_dir(".mpeblink_blink_eval_calibration_")
    staged_calibration = staging_dir / CALIBRATION_CSV.name
    try:
        print("Staging directory:", staging_dir)
        records, stats = extract_split("val")
        best, calibration_path = calibrate(records, output_path=staged_calibration)
        print("\nValidating staged calibration output...")
        validate_calibration_output(calibration_path)
        replace_owned_files([(calibration_path, CALIBRATION_CSV)], staging_dir)
        print("Committed final calibration output.")
        print("\n=== Selected Validation Configuration ===")
        print(f"blink_threshold: {best['threshold']:.4f}")
        print(f"min_closed_frames: {best['min_closed_frames']}")
        print(f"Precision: {best['precision']:.4f}")
        print(f"Recall: {best['recall']:.4f}")
        print(f"F1: {best['f1']:.4f}")
        print("\nCalibration saved:")
        print(CALIBRATION_CSV)
        print("\nExtraction runtime:")
        print(f"{stats['runtime_seconds'] / 60.0:.2f} minutes")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def run_final_test():
    """Run the frozen final test transactionally."""
    staging_dir = make_staging_dir(".mpeblink_blink_eval_test_")
    staged_summary = staging_dir / TEST_SUMMARY_PATH.name
    staged_sequence = staging_dir / TEST_SEQUENCE_RESULTS_PATH.name
    try:
        print("Staging directory:", staging_dir)
        print("Running final MPEBlink 2.0 test evaluation.")
        print("The parameters below were frozen using the validation split.")
        print(f"blink_threshold = {SELECTED_THRESHOLD:.2f}")
        print(f"min_closed_frames = {SELECTED_MIN_CLOSED_FRAMES}")
        records, stats = extract_split("test")
        metrics, sequence_rows = evaluate_records(records, SELECTED_THRESHOLD, SELECTED_MIN_CLOSED_FRAMES, EVENT_IOU_THRESHOLD)
        sequence_path = save_sequence_results("test", sequence_rows, output_path=staged_sequence)
        summary_path = save_summary("test", stats, metrics, output_path=staged_summary)
        print("\nValidating staged evaluator outputs...")
        validate_test_outputs(summary_path, sequence_path)
        replace_owned_files([(summary_path, TEST_SUMMARY_PATH), (sequence_path, TEST_SEQUENCE_RESULTS_PATH)], staging_dir)
        print("Committed final evaluator outputs.")
        print("\n=== Final Test Results ===")
        print(f"Videos: {stats['videos']}")
        print(f"Person sequences: {stats['person_sequences']}")
        print(f"Eye-openness availability: {stats['eye_availability'] * 100.0:.2f}%")
        print(f"Landmark failures: {stats['landmark_failures']}")
        print(f"Ground-truth blinks: {metrics['ground_truth_blinks']}")
        print(f"Predicted blinks: {metrics['predicted_blinks']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Mean matched temporal IoU: {metrics['mean_matched_tiou']:.4f}")
        print(f"Blink-count MAE per sequence: {metrics['blink_count_mae_per_sequence']:.4f}")
        print(f"Blink-rate MAE: {metrics['blink_rate_mae_per_min']:.4f} blinks/min")
        print(f"Mean blink-duration error: {metrics['mean_duration_error_seconds']:.4f} s")
        print(f"Eye-openness ROC AUC: {metrics['eye_openness_auc']:.4f}")
        print(f"Blink-frame median openness: {metrics['blink_openness_median']:.4f}")
        print(f"Non-blink median openness: {metrics['nonblink_openness_median']:.4f}")
        print(f"Runtime: {stats['runtime_seconds'] / 60.0:.2f} minutes")
        print("\nSaved:")
        print(TEST_SUMMARY_PATH)
        print(TEST_SEQUENCE_RESULTS_PATH)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args():
    """Parse evaluator execution mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PhysioTrack eye openness "
            "and blink detection on MPEBlink 2.0."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the dataset structure and annotations, then exit "
            "without model inference."
        ),
    )

    mode.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Re-run the original validation-split calibration for verification."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    accounting = (
        validate_dataset_layout()
    )

    print(
        "MPEBlink 2.0 dataset preflight: PASS"
    )

    print(
        "Dataset root:",
        DATASET_ROOT,
    )

    for split in EVALUATION_SPLITS:
        values = accounting[
            split
        ]

        print(
            f"{split}: "
            f"videos={values['videos']}, "
            f"annotation_frames={values['annotation_frames']}, "
            f"person_sequences={values['person_sequences']}, "
            f"blink_events={values['blink_events']}, "
            f"out_of_range_blink_events="
            f"{values['out_of_range_blink_events']}"
        )

    if args.preflight_only:
        print(
            "Preflight-only mode: no model inference was run."
        )
        return

    if args.calibrate:
        run_validation_calibration()
        return

    run_final_test()


if __name__ == "__main__":
    main()
