from pathlib import Path
import csv
import math
import textwrap

import cv2
import numpy as np
from scipy.io import loadmat


IOU_THRESHOLD = 0.50
VISUALIZATION_THRESHOLD = 0.25

EXPECTED_VALIDATION_IMAGES = 3226
EXPECTED_PREDICTION_FILES = 3226

EXPECTED_AP_REFERENCE = {
    "Easy": 0.958883,
    "Medium": 0.948828,
    "Hard": 0.871830,
}

SETTING_FILES = {
    "Easy": "wider_easy_val.mat",
    "Medium": "wider_medium_val.mat",
    "Hard": "wider_hard_val.mat",
}

ROLE_SPECS = [
    ("easy_clear_01", "Easy", "clear"),
    ("easy_clear_02", "Easy", "clear"),
    ("medium_scale_01", "Medium", "scale"),
    ("medium_scale_02", "Medium", "scale"),
    ("hard_readable_01", "Hard", "readable"),
    ("hard_challenge_01", "Hard", "challenge"),
    ("group_medium_01", "Medium", "group"),
    ("group_medium_02", "Medium", "group"),
    ("group_hard_01", "Hard", "group"),
    ("group_hard_02", "Hard", "group"),
]

CANVAS_WIDTH = 1920
IMAGE_AREA_WIDTH = 1280
PANEL_WIDTH = CANVAS_WIDTH - IMAGE_AREA_WIDTH
MIN_CANVAS_HEIGHT = 1180

ANNOTATED_DIRNAME = "annotated_images"

GRID_COLUMNS = 2
GRID_CELL_WIDTH = 960
GRID_CELL_HEIGHT = 620

GREEN = (70, 220, 90)
ORANGE = (0, 165, 255)
RED = (40, 40, 235)
GRAY = (150, 150, 150)

PANEL_BG = 18
IMAGE_BG = 18


def load_benchmark_ap(table_path):
    if not table_path.is_file():
        raise FileNotFoundError(
            "Final quantitative results table was not found: "
            f"{table_path}\n"
            "Run wider_face_inference.py, wider_face_eval.py, and "
            "wider_face_plot.py first."
        )

    benchmark_ap = {}

    with open(
        table_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"Difficulty", "Average Precision"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise RuntimeError(
                "Unexpected columns in quantitative results table: "
                f"{reader.fieldnames}"
            )

        for row in reader:
            subset = row["Difficulty"].strip()
            if subset not in EXPECTED_AP_REFERENCE:
                continue
            benchmark_ap[subset] = float(row["Average Precision"])

    missing = [
        subset
        for subset in EXPECTED_AP_REFERENCE
        if subset not in benchmark_ap
    ]
    if missing:
        raise RuntimeError(
            "Missing benchmark AP values in quantitative results table: "
            + ", ".join(missing)
        )

    for subset, expected in EXPECTED_AP_REFERENCE.items():
        actual = benchmark_ap[subset]
        if abs(actual - expected) > 1e-6:
            raise RuntimeError(
                f"{subset} AP in the final quantitative table is "
                f"{actual:.6f}, but the accepted reference is "
                f"{expected:.6f}. Review the quantitative validation "
                "before generating qualitative figures."
            )

    return benchmark_ap


def read_prediction_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 2:
        return np.empty((0, 5), dtype=np.float64)

    count = int(lines[1])

    if count == 0:
        return np.empty((0, 5), dtype=np.float64)

    detections = []

    for line in lines[2:2 + count]:
        values = [float(value) for value in line.split()[:5]]
        detections.append(values)

    detections = np.asarray(detections, dtype=np.float64)

    if len(detections):
        detections = detections[np.argsort(-detections[:, 4])]

    return detections


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    result = boxes[:, :4].copy()
    result[:, 2] = result[:, 0] + result[:, 2]
    result[:, 3] = result[:, 1] + result[:, 3]
    return result


def compute_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)

    pred = pred_boxes[:, None, :]
    gt = gt_boxes[None, :, :]

    x1 = np.maximum(pred[:, :, 0], gt[:, :, 0])
    y1 = np.maximum(pred[:, :, 1], gt[:, :, 1])
    x2 = np.minimum(pred[:, :, 2], gt[:, :, 2])
    y2 = np.minimum(pred[:, :, 3], gt[:, :, 3])

    width = np.maximum(0.0, x2 - x1 + 1.0)
    height = np.maximum(0.0, y2 - y1 + 1.0)
    intersection = width * height

    pred_area = (
        (pred[:, :, 2] - pred[:, :, 0] + 1.0)
        * (pred[:, :, 3] - pred[:, :, 1] + 1.0)
    )
    gt_area = (
        (gt[:, :, 2] - gt[:, :, 0] + 1.0)
        * (gt[:, :, 3] - gt[:, :, 1] + 1.0)
    )
    union = pred_area + gt_area - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def evaluate_for_display(predictions_xywh, gt_boxes_xywh, keep_indices):
    predictions = np.asarray(predictions_xywh, dtype=np.float64)

    if predictions.size == 0:
        predictions = np.empty((0, 5), dtype=np.float64)

    visible_predictions = predictions[
        predictions[:, 4] >= VISUALIZATION_THRESHOLD
    ].copy()

    if len(visible_predictions):
        visible_predictions = visible_predictions[
            np.argsort(-visible_predictions[:, 4])
        ]

    pred_boxes = xywh_to_xyxy(
        visible_predictions[:, :4]
        if len(visible_predictions)
        else np.empty((0, 4), dtype=np.float64)
    )

    gt_boxes = xywh_to_xyxy(gt_boxes_xywh)

    keep_indices = np.asarray(keep_indices, dtype=int).reshape(-1)

    eligible_mask = np.zeros(len(gt_boxes), dtype=bool)
    if len(keep_indices):
        eligible_mask[keep_indices] = True

    matched_eligible = np.zeros(len(gt_boxes), dtype=bool)
    matched_gt_index = []
    statuses = []
    matched_iou = []

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    for pred_index in range(len(pred_boxes)):
        if len(gt_boxes) == 0:
            statuses.append("FP")
            matched_iou.append(0.0)
            matched_gt_index.append(-1)
            continue

        best_gt = int(np.argmax(iou_matrix[pred_index]))
        best_iou = float(iou_matrix[pred_index, best_gt])

        if best_iou < IOU_THRESHOLD:
            statuses.append("FP")
            matched_iou.append(best_iou)
            matched_gt_index.append(best_gt)
            continue

        if not eligible_mask[best_gt]:
            statuses.append("IGNORED")
            matched_iou.append(best_iou)
            matched_gt_index.append(best_gt)
            continue

        if not matched_eligible[best_gt]:
            matched_eligible[best_gt] = True
            statuses.append("TP")
            matched_iou.append(best_iou)
            matched_gt_index.append(best_gt)
        else:
            statuses.append("FP")
            matched_iou.append(best_iou)
            matched_gt_index.append(best_gt)

    tp = int(sum(status == "TP" for status in statuses))
    fp = int(sum(status == "FP" for status in statuses))
    ignored_predictions = int(sum(status == "IGNORED" for status in statuses))
    eligible_gt_count = int(np.sum(eligible_mask))
    fn = int(eligible_gt_count - np.sum(matched_eligible))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / eligible_gt_count if eligible_gt_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    tp_ious = [
        iou
        for status, iou in zip(statuses, matched_iou)
        if status == "TP"
    ]
    tp_confidences = [
        float(visible_predictions[index, 4])
        for index, status in enumerate(statuses)
        if status == "TP"
    ]

    return {
        "visible_predictions": visible_predictions,
        "pred_boxes": pred_boxes,
        "gt_boxes": gt_boxes,
        "eligible_mask": eligible_mask,
        "matched_eligible": matched_eligible,
        "statuses": statuses,
        "matched_iou": matched_iou,
        "matched_gt_index": matched_gt_index,
        "eligible_gt_count": eligible_gt_count,
        "predictions_shown": int(len(visible_predictions)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored_predictions,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(np.mean(tp_ious)) if tp_ious else 0.0,
        "mean_tp_confidence": (
            float(np.mean(tp_confidences))
            if tp_confidences
            else 0.0
        ),
    }


def build_records(images_dir, pred_dir, gt_dir):
    val_data = loadmat(gt_dir / "wider_face_val.mat")
    event_list = val_data["event_list"]
    file_list = val_data["file_list"]
    face_bbx_list = val_data["face_bbx_list"]

    subset_gt = {
        name: loadmat(gt_dir / filename)["gt_list"]
        for name, filename in SETTING_FILES.items()
    }

    records = []

    total_images = sum(
        len(file_list[event_index][0])
        for event_index in range(len(event_list))
    )

    processed = 0

    for event_index in range(len(event_list)):
        event_name = event_list[event_index][0][0]
        event_files = file_list[event_index][0]
        event_gt = face_bbx_list[event_index][0]

        for image_index in range(len(event_files)):
            image_name = event_files[image_index][0][0]
            relative_image = f"{event_name}/{image_name}.jpg"

            image_path = images_dir / event_name / f"{image_name}.jpg"
            prediction_path = pred_dir / event_name / f"{image_name}.txt"

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing validation image: {image_path}"
                )

            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Missing prediction file: {prediction_path}"
                )

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(
                    f"Could not read validation image: {image_path}"
                )

            image_height, image_width = image.shape[:2]
            image_diagonal = math.sqrt(image_width ** 2 + image_height ** 2)

            gt_boxes = event_gt[image_index][0]
            predictions = read_prediction_file(prediction_path)

            for subset, gt_list in subset_gt.items():
                keep_indices = (
                    gt_list[event_index][0][image_index][0]
                    .flatten()
                    .astype(int)
                    - 1
                )

                if len(keep_indices) == 0:
                    continue

                metrics = evaluate_for_display(
                    predictions_xywh=predictions,
                    gt_boxes_xywh=gt_boxes,
                    keep_indices=keep_indices,
                )

                eligible_boxes = gt_boxes[keep_indices]

                face_widths = np.maximum(eligible_boxes[:, 2], 0.0)
                face_heights = np.maximum(eligible_boxes[:, 3], 0.0)
                face_diagonals = np.sqrt(face_widths ** 2 + face_heights ** 2)

                median_face_diagonal_ratio = float(
                    np.median(face_diagonals) / max(image_diagonal, 1.0)
                )
                max_face_diagonal_ratio = float(
                    np.max(face_diagonals) / max(image_diagonal, 1.0)
                )

                records.append(
                    {
                        "subset": subset,
                        "relative_image": relative_image,
                        "image_path": image_path,
                        "prediction_path": prediction_path,
                        "event_name": event_name,
                        "image_name": image_name,
                        "image_height": int(image_height),
                        "image_width": int(image_width),
                        "total_gt_faces": int(len(gt_boxes)),
                        "median_face_diagonal_ratio": median_face_diagonal_ratio,
                        "max_face_diagonal_ratio": max_face_diagonal_ratio,
                        "metrics": metrics,
                    }
                )

            processed += 1

            if processed % 500 == 0 or processed == total_images:
                print(f"Scanned {processed}/{total_images} validation images")

    return records


def base_quality_score(record):
    metrics = record["metrics"]

    return (
        5.0 * metrics["recall"]
        + 4.0 * metrics["precision"]
        + 2.0 * metrics["mean_iou"]
        + 0.5 * metrics["mean_tp_confidence"]
        + 28.0 * record["median_face_diagonal_ratio"]
        + 8.0 * record["max_face_diagonal_ratio"]
        - 0.10 * record["total_gt_faces"]
        - 0.07 * metrics["predictions_shown"]
        - 0.20 * metrics["ignored_predictions"]
        - 0.80 * metrics["fp"]
        - 1.00 * metrics["fn"]
    )


def role_specific_score(record, role):
    score = base_quality_score(record)
    metrics = record["metrics"]

    if role.startswith("easy_"):
        target = 3.0
        score -= 0.60 * abs(metrics["eligible_gt_count"] - target)

    elif role.startswith("medium_scale_"):
        target = 7.0
        score -= 0.35 * abs(metrics["eligible_gt_count"] - target)

    elif role == "hard_readable_01":
        target = 10.0
        score += 0.10 * min(metrics["eligible_gt_count"], 18)
        score -= 0.20 * abs(metrics["eligible_gt_count"] - target)

    elif role == "hard_challenge_01":
        error_count = metrics["fp"] + metrics["fn"]
        score += 1.0 if error_count > 0 else -20.0
        score -= 0.35 * abs(error_count - 1)
        score += 10.0 * record["median_face_diagonal_ratio"]
        score -= 0.05 * metrics["predictions_shown"]

    elif role.startswith("group_medium_"):
        target = 10.0
        score += 0.25 * min(metrics["eligible_gt_count"], 18)
        score -= 0.18 * abs(metrics["eligible_gt_count"] - target)
        score -= 0.03 * max(metrics["predictions_shown"] - 18, 0)

    elif role.startswith("group_hard_"):
        target = 11.0
        score += 0.28 * min(metrics["eligible_gt_count"], 20)
        score -= 0.20 * abs(metrics["eligible_gt_count"] - target)
        score -= 0.04 * max(metrics["predictions_shown"] - 20, 0)

    return score


def sort_candidates(candidates, role):
    return sorted(
        candidates,
        key=lambda record: (
            -role_specific_score(record, role),
            record["relative_image"],
        ),
    )


def candidate_tiers(role, subset):
    def easy_strict(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 5
            and 2 <= m["eligible_gt_count"] <= 5
            and m["predictions_shown"] <= 8
            and m["fp"] == 0
            and m["fn"] == 0
            and m["precision"] >= 0.99
            and m["recall"] >= 0.99
            and m["mean_iou"] >= 0.88
        )

    def easy_relaxed(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 7
            and 2 <= m["eligible_gt_count"] <= 6
            and m["predictions_shown"] <= 10
            and m["fp"] <= 1
            and m["fn"] == 0
            and m["precision"] >= 0.90
            and m["recall"] >= 0.95
        )

    def medium_scale_strict(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 12
            and 5 <= m["eligible_gt_count"] <= 10
            and m["predictions_shown"] <= 14
            and m["fp"] <= 1
            and m["fn"] == 0
            and m["precision"] >= 0.92
            and m["recall"] >= 0.98
            and m["mean_iou"] >= 0.82
        )

    def medium_scale_relaxed(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 16
            and 5 <= m["eligible_gt_count"] <= 12
            and m["predictions_shown"] <= 18
            and m["fp"] <= 2
            and m["fn"] <= 1
            and m["precision"] >= 0.80
            and m["recall"] >= 0.85
        )

    def hard_success_strict(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 25
            and 4 <= m["eligible_gt_count"] <= 20
            and m["predictions_shown"] <= 28
            and m["fp"] <= 2
            and m["fn"] <= 2
            and m["precision"] >= 0.82
            and m["recall"] >= 0.80
            and m["mean_iou"] >= 0.72
        )

    def hard_success_relaxed(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] <= 35
            and 4 <= m["eligible_gt_count"] <= 25
            and m["predictions_shown"] <= 36
            and m["fp"] <= 4
            and m["fn"] <= 4
            and m["precision"] >= 0.65
            and m["recall"] >= 0.65
        )

    def hard_challenge_strict(record):
        m = record["metrics"]
        error_count = m["fp"] + m["fn"]
        return (
            record["total_gt_faces"] <= 25
            and 4 <= m["eligible_gt_count"] <= 20
            and m["predictions_shown"] <= 28
            and 1 <= error_count <= 3
            and m["precision"] >= 0.70
            and m["recall"] >= 0.65
            and record["median_face_diagonal_ratio"] >= 0.018
        )

    def hard_challenge_relaxed(record):
        m = record["metrics"]
        error_count = m["fp"] + m["fn"]
        return (
            record["total_gt_faces"] <= 35
            and 3 <= m["eligible_gt_count"] <= 25
            and m["predictions_shown"] <= 36
            and 1 <= error_count <= 5
            and m["precision"] >= 0.55
            and m["recall"] >= 0.50
            and record["median_face_diagonal_ratio"] >= 0.012
        )

    def group_medium_strict(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] >= 8
            and record["total_gt_faces"] <= 22
            and m["eligible_gt_count"] >= 8
            and m["eligible_gt_count"] <= 18
            and m["predictions_shown"] <= 24
            and m["fp"] <= 3
            and m["fn"] <= 2
            and m["precision"] >= 0.75
            and m["recall"] >= 0.80
        )

    def group_medium_relaxed(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] >= 7
            and record["total_gt_faces"] <= 26
            and m["eligible_gt_count"] >= 7
            and m["eligible_gt_count"] <= 20
            and m["predictions_shown"] <= 28
            and m["fp"] <= 4
            and m["fn"] <= 3
            and m["precision"] >= 0.65
            and m["recall"] >= 0.70
        )

    def group_hard_strict(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] >= 8
            and record["total_gt_faces"] <= 28
            and m["eligible_gt_count"] >= 8
            and m["eligible_gt_count"] <= 20
            and m["predictions_shown"] <= 30
            and m["fp"] <= 4
            and m["fn"] <= 3
            and m["precision"] >= 0.68
            and m["recall"] >= 0.68
        )

    def group_hard_relaxed(record):
        m = record["metrics"]
        return (
            record["total_gt_faces"] >= 7
            and record["total_gt_faces"] <= 34
            and m["eligible_gt_count"] >= 7
            and m["eligible_gt_count"] <= 24
            and m["predictions_shown"] <= 36
            and m["fp"] <= 5
            and m["fn"] <= 4
            and m["precision"] >= 0.58
            and m["recall"] >= 0.58
        )

    def subset_only(record):
        return record["subset"] == subset

    if role.startswith("easy_"):
        return [easy_strict, easy_relaxed, subset_only]

    if role.startswith("medium_scale_"):
        return [medium_scale_strict, medium_scale_relaxed, subset_only]

    if role == "hard_readable_01":
        return [hard_success_strict, hard_success_relaxed, subset_only]

    if role == "hard_challenge_01":
        return [hard_challenge_strict, hard_challenge_relaxed]

    if role.startswith("group_medium_"):
        return [group_medium_strict, group_medium_relaxed, subset_only]

    if role.startswith("group_hard_"):
        return [group_hard_strict, group_hard_relaxed, subset_only]

    raise ValueError(f"Unsupported role: {role}")


def choose_examples(records):
    selected = []
    used_images = set()

    for role, subset, selection_mode in ROLE_SPECS:
        subset_records = [
            record
            for record in records
            if record["subset"] == subset
            and record["relative_image"] not in used_images
        ]

        if not subset_records:
            raise RuntimeError(
                f"No records available for subset {subset}"
            )

        chosen = None

        for predicate in candidate_tiers(role, subset):
            filtered = [
                record
                for record in subset_records
                if predicate(record)
            ]
            if filtered:
                chosen = sort_candidates(filtered, role)[0]
                break

        if chosen is None:
            raise RuntimeError(
                "No scientifically readable candidate found for "
                f"{role}. Do not replace it with an overcrowded "
                "fallback without review."
            )

        chosen = dict(chosen)
        chosen["role"] = role
        chosen["selection_mode"] = selection_mode

        selected.append(chosen)
        used_images.add(chosen["relative_image"])

    return selected


def wrap_text(text, width):
    return textwrap.wrap(text, width=width) if text else [""]


def put_text(
    image,
    text,
    x,
    y,
    scale=0.5,
    color=(235, 235, 235),
    thickness=1,
):
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def fit_image_to_canvas(source, canvas_height):
    source_height, source_width = source.shape[:2]

    scale = min(
        IMAGE_AREA_WIDTH / source_width,
        canvas_height / source_height,
    )

    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        source,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (canvas_height, IMAGE_AREA_WIDTH, 3),
        IMAGE_BG,
        dtype=np.uint8,
    )

    offset_x = (IMAGE_AREA_WIDTH - new_width) // 2
    offset_y = (canvas_height - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    return canvas, scale, offset_x, offset_y


def transform_box(box, scale, offset_x, offset_y):
    x1, y1, x2, y2 = box

    return (
        int(round(x1 * scale + offset_x)),
        int(round(y1 * scale + offset_y)),
        int(round(x2 * scale + offset_x)),
        int(round(y2 * scale + offset_y)),
    )


def draw_box(image, box, color, thickness, label=None):
    x1, y1, x2, y2 = [int(value) for value in box]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    if not label:
        return

    font_scale = 0.46
    text_thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_thickness,
    )

    label_x1 = max(0, x1)
    label_y2 = max(text_height + baseline + 4, y1)
    label_y1 = max(0, label_y2 - text_height - baseline - 6)
    label_x2 = min(image.shape[1] - 1, label_x1 + text_width + 6)

    cv2.rectangle(
        image,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1,
    )

    put_text(
        image,
        label,
        label_x1 + 3,
        label_y2 - baseline - 2,
        font_scale,
        (10, 10, 10),
        1,
    )


def estimate_panel_height(record):
    path_lines = wrap_text(record["relative_image"], 34)

    estimated_y = 40
    estimated_y += 38
    estimated_y += 28
    estimated_y += 14
    estimated_y += 25 * 3
    estimated_y += 21 * len(path_lines)
    estimated_y += 18
    estimated_y += 20
    estimated_y += 30
    estimated_y += 23 * 11
    estimated_y += 16
    estimated_y += 20
    estimated_y += 29
    estimated_y += 23 * 3
    estimated_y += 18
    estimated_y += 27
    estimated_y += 22 * 4
    estimated_y += 18
    estimated_y += 19 * 3
    estimated_y += 30

    return max(MIN_CANVAS_HEIGHT, estimated_y)


def build_panel(record, canvas_height, benchmark_ap):
    metrics = record["metrics"]

    panel = np.full(
        (canvas_height, PANEL_WIDTH, 3),
        PANEL_BG,
        dtype=np.uint8,
    )

    margin = 24
    value_x = 375
    y = 40

    put_text(
        panel,
        "PhysioTrack",
        margin,
        y,
        0.95,
        (255, 255, 255),
        2,
    )
    y += 34

    put_text(
        panel,
        "WIDER FACE qualitative benchmark",
        margin,
        y,
        0.52,
        (185, 185, 185),
        1,
    )
    y += 34

    header_rows = [
        ("Subset", record["subset"]),
        ("Selection", record["selection_mode"]),
        ("Image", None),
    ]

    for key, value in header_rows:
        put_text(
            panel,
            f"{key}:",
            margin,
            y,
            0.50,
            (160, 160, 160),
            1,
        )

        if value is not None:
            put_text(
                panel,
                str(value),
                130,
                y,
                0.52,
                (245, 245, 245),
                1,
            )
            y += 24
        else:
            y += 22
            for line in wrap_text(record["relative_image"], 34):
                put_text(
                    panel,
                    line,
                    130,
                    y,
                    0.47,
                    (245, 245, 245),
                    1,
                )
                y += 20

    y += 8
    cv2.line(
        panel,
        (margin, y),
        (PANEL_WIDTH - margin, y),
        (75, 75, 75),
        1,
    )
    y += 30

    put_text(
        panel,
        "Per-image display metrics",
        margin,
        y,
        0.58,
        (255, 255, 255),
        1,
    )
    y += 28

    metric_rows = [
        ("Eligible GT", metrics["eligible_gt_count"]),
        ("Predictions shown", metrics["predictions_shown"]),
        ("TP", metrics["tp"]),
        ("FP", metrics["fp"]),
        ("FN", metrics["fn"]),
        ("Ignored predictions", metrics["ignored_predictions"]),
        ("Precision", f"{metrics['precision']:.3f}"),
        ("Recall", f"{metrics['recall']:.3f}"),
        ("F1", f"{metrics['f1']:.3f}"),
        ("Mean matched IoU", f"{metrics['mean_iou']:.3f}"),
        ("Mean TP confidence", f"{metrics['mean_tp_confidence']:.3f}"),
    ]

    for key, value in metric_rows:
        put_text(
            panel,
            f"{key}:",
            margin,
            y,
            0.45,
            (180, 180, 180),
            1,
        )
        put_text(
            panel,
            str(value),
            value_x,
            y,
            0.48,
            (250, 250, 250),
            1,
        )
        y += 23

    y += 8
    cv2.line(
        panel,
        (margin, y),
        (PANEL_WIDTH - margin, y),
        (75, 75, 75),
        1,
    )
    y += 30

    put_text(
        panel,
        "Protocol",
        margin,
        y,
        0.58,
        (255, 255, 255),
        1,
    )
    y += 28

    protocol_rows = [
        ("Matching IoU", f"{IOU_THRESHOLD:.2f}"),
        ("Raw display threshold", f"{VISUALIZATION_THRESHOLD:.2f}"),
        (
            "Full benchmark AP",
            f"{record['subset']} {benchmark_ap[record['subset']]:.6f}",
        ),
    ]

    for key, value in protocol_rows:
        put_text(
            panel,
            f"{key}:",
            margin,
            y,
            0.45,
            (180, 180, 180),
            1,
        )
        put_text(
            panel,
            str(value),
            value_x,
            y,
            0.46,
            (245, 245, 245),
            1,
        )
        y += 23

    y += 10

    put_text(
        panel,
        "Legend",
        margin,
        y,
        0.58,
        (255, 255, 255),
        1,
    )
    y += 28

    legend_rows = [
        ("TP prediction", GREEN),
        ("FP prediction", ORANGE),
        ("FN eligible GT", RED),
        ("Ignored/non-subset prediction", GRAY),
    ]

    for label, color in legend_rows:
        cv2.rectangle(
            panel,
            (margin, y - 12),
            (margin + 18, y + 3),
            color,
            -1,
        )
        put_text(
            panel,
            label,
            margin + 30,
            y,
            0.44,
            (225, 225, 225),
            1,
        )
        y += 22

    y += 10

    footnotes = [
        "Raw display threshold affects visualization only.",
        "Benchmark AP is read from the final quantitative results table.",
        "Images use real saved predictions and official WIDER FACE ground truth.",
    ]

    for line in footnotes:
        put_text(
            panel,
            line,
            margin,
            y,
            0.38,
            (145, 145, 145),
            1,
        )
        y += 19

    if y > (canvas_height - 10):
        raise RuntimeError("Panel overflow detected.")

    return panel


def render_example(record, benchmark_ap):
    image = cv2.imread(str(record["image_path"]))
    if image is None:
        raise RuntimeError(
            f"Could not read image: {record['image_path']}"
        )

    metrics = record["metrics"]

    canvas_height = estimate_panel_height(record)

    image_canvas, scale, offset_x, offset_y = fit_image_to_canvas(
        image,
        canvas_height,
    )

    eligible_indices = np.where(metrics["eligible_mask"])[0]
    matched_mask = metrics["matched_eligible"]
    gt_boxes = metrics["gt_boxes"]

    for gt_index in eligible_indices:
        if matched_mask[gt_index]:
            continue

        transformed = transform_box(
            gt_boxes[gt_index],
            scale,
            offset_x,
            offset_y,
        )

        draw_box(
            image_canvas,
            transformed,
            RED,
            2,
            "FN",
        )

    for pred_index, pred_box in enumerate(metrics["pred_boxes"]):
        status = metrics["statuses"][pred_index]
        confidence = float(metrics["visible_predictions"][pred_index, 4])
        iou = float(metrics["matched_iou"][pred_index])

        if status == "TP":
            color = GREEN
            thickness = 2
            label = f"TP c={confidence:.2f} IoU={iou:.2f}"
        elif status == "FP":
            color = ORANGE
            thickness = 2
            label = f"FP c={confidence:.2f}"
        else:
            color = GRAY
            thickness = 1
            label = f"ignored c={confidence:.2f}"

        transformed = transform_box(
            pred_box,
            scale,
            offset_x,
            offset_y,
        )

        draw_box(
            image_canvas,
            transformed,
            color,
            thickness,
            label,
        )

    panel = build_panel(record, canvas_height, benchmark_ap)
    output = np.hstack([image_canvas, panel])

    if output.shape[1] != CANVAS_WIDTH:
        raise RuntimeError(
            f"Unexpected output width: {output.shape[1]}"
        )

    return output


def write_selection_csv(selected, output_path, benchmark_ap):
    fieldnames = [
        "role",
        "subset",
        "selection_mode",
        "relative_image",
        "image_width",
        "image_height",
        "total_gt_faces",
        "eligible_gt_count",
        "predictions_shown",
        "tp",
        "fp",
        "fn",
        "ignored_predictions",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "mean_tp_confidence",
        "visualization_threshold",
        "matching_iou",
        "benchmark_ap",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in selected:
            metrics = record["metrics"]

            writer.writerow(
                {
                    "role": record["role"],
                    "subset": record["subset"],
                    "selection_mode": record["selection_mode"],
                    "relative_image": record["relative_image"],
                    "image_width": record["image_width"],
                    "image_height": record["image_height"],
                    "total_gt_faces": record["total_gt_faces"],
                    "eligible_gt_count": metrics["eligible_gt_count"],
                    "predictions_shown": metrics["predictions_shown"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "ignored_predictions": metrics["ignored_predictions"],
                    "precision": f"{metrics['precision']:.6f}",
                    "recall": f"{metrics['recall']:.6f}",
                    "f1": f"{metrics['f1']:.6f}",
                    "mean_matched_iou": f"{metrics['mean_iou']:.6f}",
                    "mean_tp_confidence": (
                        f"{metrics['mean_tp_confidence']:.6f}"
                    ),
                    "visualization_threshold": VISUALIZATION_THRESHOLD,
                    "matching_iou": IOU_THRESHOLD,
                    "benchmark_ap": benchmark_ap[record["subset"]],
                }
            )


def fit_to_cell(image, cell_width, cell_height):
    height, width = image.shape[:2]

    scale = min(cell_width / width, cell_height / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    cell = np.full(
        (cell_height, cell_width, 3),
        245,
        dtype=np.uint8,
    )

    offset_x = (cell_width - new_width) // 2
    offset_y = (cell_height - new_height) // 2

    cell[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    return cell


def create_grid(rendered_images, output_path):
    cells = [
        fit_to_cell(image, GRID_CELL_WIDTH, GRID_CELL_HEIGHT)
        for image in rendered_images
    ]

    rows = []

    for start in range(0, len(cells), GRID_COLUMNS):
        row = cells[start:start + GRID_COLUMNS]

        while len(row) < GRID_COLUMNS:
            row.append(
                np.full(
                    (GRID_CELL_HEIGHT, GRID_CELL_WIDTH, 3),
                    245,
                    dtype=np.uint8,
                )
            )

        rows.append(np.hstack(row))

    grid = np.vstack(rows)

    if not cv2.imwrite(str(output_path), grid):
        raise RuntimeError(
            f"Failed to save grid: {output_path}"
        )


def clear_previous_outputs(annotated_dir, selection_csv, grid_path):
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for path in annotated_dir.iterdir():
        if path.is_file():
            path.unlink()

    if selection_csv.exists():
        selection_csv.unlink()

    if grid_path.exists():
        grid_path.unlink()


def main():
    validation_dir = Path(__file__).resolve().parent
    project_root = validation_dir.parents[2]

    wider_root = project_root / "datasets" / "WIDER_FACE"

    images_dir = wider_root / "WIDER_val" / "images"
    gt_dir = (
        wider_root
        / "eval_tools"
        / "eval_tools"
        / "ground_truth"
    )

    results_dir = validation_dir / "results"
    pred_dir = results_dir / "predictions"
    quantitative_table_path = results_dir / "wider_face_thesis_table.csv"

    qualitative_dir = results_dir / "qualitative"
    annotated_dir = qualitative_dir / ANNOTATED_DIRNAME
    figures_dir = results_dir / "figures"
    selection_csv = qualitative_dir / "wider_face_qualitative_selection.csv"
    grid_path = figures_dir / "wider_face_qualitative_examples.png"

    required_paths = [
        images_dir,
        pred_dir,
        quantitative_table_path,
        gt_dir / "wider_face_val.mat",
        gt_dir / "wider_easy_val.mat",
        gt_dir / "wider_medium_val.mat",
        gt_dir / "wider_hard_val.mat",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path not found: {path}"
            )

    image_count = len(list(images_dir.rglob("*.jpg")))
    prediction_count = len(list(pred_dir.rglob("*.txt")))

    if image_count != EXPECTED_VALIDATION_IMAGES:
        raise RuntimeError(
            "Unexpected validation image count: "
            f"{image_count}. Expected {EXPECTED_VALIDATION_IMAGES}."
        )

    if prediction_count != EXPECTED_PREDICTION_FILES:
        raise RuntimeError(
            "Unexpected prediction file count: "
            f"{prediction_count}. Expected {EXPECTED_PREDICTION_FILES}."
        )

    benchmark_ap = load_benchmark_ap(quantitative_table_path)

    qualitative_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    clear_previous_outputs(annotated_dir, selection_csv, grid_path)

    print("WIDER FACE qualitative preflight: PASS")
    print("Validation images:", image_count)
    print("Prediction files:", prediction_count)
    print("Raw visualization threshold:", VISUALIZATION_THRESHOLD)
    print("Matching IoU:", IOU_THRESHOLD)
    print("Benchmark AP loaded from:", quantitative_table_path)
    for subset in ("Easy", "Medium", "Hard"):
        print(f"{subset} AP: {benchmark_ap[subset]:.6f}")
    print()
    print("Cleaned previous qualitative outputs.")
    print("Scanning benchmark results for deterministic qualitative selection...")

    records = build_records(images_dir, pred_dir, gt_dir)

    print()
    print("Selecting final qualitative examples...")

    selected = choose_examples(records)

    rendered_images = []

    for record in selected:
        metrics = record["metrics"]

        print(
            f"{record['role']}: "
            f"{record['relative_image']} | "
            f"subset={record['subset']} | "
            f"eligible={metrics['eligible_gt_count']} | "
            f"shown={metrics['predictions_shown']} | "
            f"TP={metrics['tp']} | "
            f"FP={metrics['fp']} | "
            f"FN={metrics['fn']} | "
            f"P={metrics['precision']:.3f} | "
            f"R={metrics['recall']:.3f}"
        )

        rendered = render_example(record, benchmark_ap)

        output_path = annotated_dir / f"{record['role']}.png"

        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(
                f"Failed to save annotated image: {output_path}"
            )

        rendered_images.append(rendered)

    write_selection_csv(selected, selection_csv, benchmark_ap)
    create_grid(rendered_images, grid_path)

    print()
    print("Saved annotated images:")
    print(annotated_dir)
    print("Saved qualitative table:")
    print(selection_csv)
    print("Saved qualitative grid:")
    print(grid_path)
    print()
    print(
        "DONE. Qualitative figures use real saved predictions and "
        "official WIDER FACE ground truth. Quantitative AP remains unchanged."
    )


if __name__ == "__main__":
    main()