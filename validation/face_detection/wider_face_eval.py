from pathlib import Path
import csv
import shutil
import tempfile

import numpy as np
from scipy.io import loadmat


IOU_THRESHOLD = 0.5
NUM_THRESHOLDS = 1000


def box_iou(gt_boxes, pred_box):
    gt_boxes = gt_boxes.astype(np.float64)
    pred_box = pred_box.astype(np.float64)

    x1 = np.maximum(gt_boxes[:, 0], pred_box[0])
    y1 = np.maximum(gt_boxes[:, 1], pred_box[1])
    x2 = np.minimum(gt_boxes[:, 2], pred_box[2])
    y2 = np.minimum(gt_boxes[:, 3], pred_box[3])

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    intersection = width * height

    gt_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1)
        * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    )

    pred_area = (
        (pred_box[2] - pred_box[0] + 1)
        * (pred_box[3] - pred_box[1] + 1)
    )

    union = gt_area + pred_area - intersection
    iou = intersection / union

    iou[width <= 0] = 0
    iou[height <= 0] = 0

    return iou


def read_prediction_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(lines) < 2:
        return np.empty((0, 5), dtype=np.float64)

    count = int(lines[1])

    if count == 0:
        return np.empty((0, 5), dtype=np.float64)

    detections = []

    for line in lines[2:2 + count]:
        values = list(map(float, line.split()))
        detections.append(values[:5])

    detections = np.asarray(
        detections,
        dtype=np.float64,
    )

    if len(detections):
        order = np.argsort(
            -detections[:, 4]
        )
        detections = detections[order]

    return detections


def load_predictions(pred_dir, event_list, file_list):
    predictions = []

    for event_index in range(len(event_list)):
        event_name = event_list[event_index][0][0]
        event_predictions = []

        images = file_list[event_index][0]

        for image_entry in images:
            image_name = image_entry[0][0]

            prediction_file = (
                pred_dir
                / event_name
                / f"{image_name}.txt"
            )

            if not prediction_file.exists():
                raise FileNotFoundError(
                    f"Missing prediction file: {prediction_file}"
                )

            event_predictions.append(
                read_prediction_file(prediction_file)
            )

        predictions.append(event_predictions)

    return predictions


def normalize_scores(predictions):
    min_score = np.inf
    max_score = -np.inf

    for event in predictions:
        for pred in event:
            if len(pred) == 0:
                continue

            min_score = min(
                min_score,
                pred[:, 4].min(),
            )

            max_score = max(
                max_score,
                pred[:, 4].max(),
            )

    if (
        not np.isfinite(min_score)
        or not np.isfinite(max_score)
    ):
        return predictions

    if max_score == min_score:
        return predictions

    for event in predictions:
        for pred in event:
            if len(pred) == 0:
                continue

            pred[:, 4] = (
                (pred[:, 4] - min_score)
                / (max_score - min_score)
            )

    return predictions


def evaluate_image(
    predictions,
    gt_boxes,
    keep_indices,
):
    pred_recall = np.zeros(
        len(predictions),
        dtype=np.float64,
    )

    proposal_list = np.ones(
        len(predictions),
        dtype=np.float64,
    )

    recall_list = np.zeros(
        len(gt_boxes),
        dtype=np.float64,
    )

    ignore = np.zeros(
        len(gt_boxes),
        dtype=np.float64,
    )

    if len(keep_indices):
        ignore[keep_indices] = 1

    pred_boxes = predictions.copy()
    gt_boxes = gt_boxes.copy()

    pred_boxes[:, 2] = (
        pred_boxes[:, 0]
        + pred_boxes[:, 2]
    )

    pred_boxes[:, 3] = (
        pred_boxes[:, 1]
        + pred_boxes[:, 3]
    )

    gt_boxes[:, 2] = (
        gt_boxes[:, 0]
        + gt_boxes[:, 2]
    )

    gt_boxes[:, 3] = (
        gt_boxes[:, 1]
        + gt_boxes[:, 3]
    )

    for i in range(len(pred_boxes)):
        overlaps = box_iou(
            gt_boxes,
            pred_boxes[i, :4],
        )

        best_index = int(
            np.argmax(overlaps)
        )

        best_overlap = overlaps[
            best_index
        ]

        if best_overlap >= IOU_THRESHOLD:
            if ignore[best_index] == 0:
                recall_list[best_index] = -1
                proposal_list[i] = -1

            elif recall_list[best_index] == 0:
                recall_list[best_index] = 1

        pred_recall[i] = np.sum(
            recall_list == 1
        )

    return pred_recall, proposal_list


def image_pr_info(
    predictions,
    proposal_list,
    pred_recall,
):
    pr_info = np.zeros(
        (NUM_THRESHOLDS, 2),
        dtype=np.float64,
    )

    for t in range(NUM_THRESHOLDS):
        threshold = (
            1
            - ((t + 1) / NUM_THRESHOLDS)
        )

        indices = np.where(
            predictions[:, 4]
            >= threshold
        )[0]

        if len(indices) == 0:
            continue

        last_index = indices[-1]

        valid_proposals = np.where(
            proposal_list[
                :last_index + 1
            ]
            == 1
        )[0]

        pr_info[t, 0] = len(
            valid_proposals
        )

        pr_info[t, 1] = (
            pred_recall[last_index]
        )

    return pr_info


def voc_ap(recall, precision):
    mrec = np.concatenate(
        (
            [0.0],
            recall,
            [1.0],
        )
    )

    mpre = np.concatenate(
        (
            [0.0],
            precision,
            [0.0],
        )
    )

    for i in range(
        len(mpre) - 2,
        -1,
        -1,
    ):
        mpre[i] = max(
            mpre[i],
            mpre[i + 1],
        )

    indices = np.where(
        mrec[1:] != mrec[:-1]
    )[0] + 1

    return np.sum(
        (
            mrec[indices]
            - mrec[indices - 1]
        )
        * mpre[indices]
    )


def evaluate_setting(
    predictions,
    face_bbx_list,
    gt_list,
):
    dataset_pr = np.zeros(
        (
            NUM_THRESHOLDS,
            2,
        ),
        dtype=np.float64,
    )

    total_faces = 0

    for event_index in range(
        len(face_bbx_list)
    ):
        event_gt = (
            face_bbx_list[event_index][0]
        )

        event_keep = (
            gt_list[event_index][0]
        )

        event_pred = (
            predictions[event_index]
        )

        for image_index in range(
            len(event_gt)
        ):
            gt_boxes = (
                event_gt[image_index][0]
            )

            keep_indices = (
                event_keep[image_index][0]
                .flatten()
            )

            # MATLAB indices start from 1
            keep_indices = (
                keep_indices.astype(int)
                - 1
            )

            total_faces += len(
                keep_indices
            )

            pred = (
                event_pred[image_index]
            )

            if (
                len(gt_boxes) == 0
                or len(pred) == 0
            ):
                continue

            (
                pred_recall,
                proposal_list,
            ) = evaluate_image(
                pred,
                gt_boxes,
                keep_indices,
            )

            pr_info = image_pr_info(
                pred,
                proposal_list,
                pred_recall,
            )

            dataset_pr += pr_info

    precision = np.divide(
        dataset_pr[:, 1],
        dataset_pr[:, 0],
        out=np.zeros(
            NUM_THRESHOLDS
        ),
        where=(
            dataset_pr[:, 0] != 0
        ),
    )

    recall = (
        dataset_pr[:, 1]
        / total_faces
    )

    ap = voc_ap(
        recall,
        precision,
    )

    return (
        ap,
        precision,
        recall,
        total_faces,
    )


def main():
    validation_dir = Path(__file__).resolve().parent
    project_root = validation_dir.parents[2]

    wider_root = (
        project_root
        / "datasets"
        / "WIDER_FACE"
    )

    results_dir = (
        validation_dir
        / "results"
    )

    pred_dir = (
        results_dir
        / "predictions"
    )

    results_path = (
        results_dir
        / "wider_face_results.csv"
    )

    gt_dir = (
        wider_root
        / "eval_tools"
        / "eval_tools"
        / "ground_truth"
    )

    required_paths = [
        pred_dir,
        gt_dir / "wider_face_val.mat",
        gt_dir / "wider_easy_val.mat",
        gt_dir / "wider_medium_val.mat",
        gt_dir / "wider_hard_val.mat",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Required evaluation input was not found: "
                f"{path}"
            )

    if not pred_dir.is_dir():
        raise RuntimeError(
            "Prediction input path is not a directory: "
            f"{pred_dir}"
        )

    if results_path.exists() and not results_path.is_file():
        raise RuntimeError(
            "Benchmark results path exists but is not a file: "
            f"{results_path}"
        )

    val_data = loadmat(
        gt_dir / "wider_face_val.mat"
    )

    event_list = val_data["event_list"]
    file_list = val_data["file_list"]
    face_bbx_list = val_data["face_bbx_list"]

    settings = {
        "Easy": "wider_easy_val.mat",
        "Medium": "wider_medium_val.mat",
        "Hard": "wider_hard_val.mat",
    }

    setting_data = {
        name: loadmat(
            gt_dir / filename
        )
        for name, filename in settings.items()
    }

    print("Reading predictions...")

    predictions = load_predictions(
        pred_dir,
        event_list,
        file_list,
    )

    predictions = normalize_scores(
        predictions
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("WIDER FACE evaluation preflight: PASS")

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".wider_face_eval_",
            dir=results_dir,
        )
    )

    staged_results_path = (
        staging_root
        / "wider_face_results.csv"
    )

    try:
        rows = []

        print(
            "\nWIDER FACE validation results"
        )

        for name in settings:
            (
                ap,
                precision,
                recall,
                total_faces,
            ) = evaluate_setting(
                predictions,
                face_bbx_list,
                setting_data[name]["gt_list"],
            )

            print(f"\n{name}")
            print(
                "Evaluated faces:",
                total_faces,
            )
            print(
                "AP:",
                round(float(ap), 6),
            )

            for threshold_index in range(
                NUM_THRESHOLDS
            ):
                rows.append(
                    {
                        "Difficulty": name,
                        "Threshold Index": (
                            threshold_index
                        ),
                        "Score Threshold": (
                            1
                            - (
                                (threshold_index + 1)
                                / NUM_THRESHOLDS
                            )
                        ),
                        "Precision": float(
                            precision[
                                threshold_index
                            ]
                        ),
                        "Recall": float(
                            recall[
                                threshold_index
                            ]
                        ),
                        "Average Precision": float(
                            ap
                        ),
                        "Evaluated Faces": int(
                            total_faces
                        ),
                    }
                )

        fieldnames = [
            "Difficulty",
            "Threshold Index",
            "Score Threshold",
            "Precision",
            "Recall",
            "Average Precision",
            "Evaluated Faces",
        ]

        with open(
            staged_results_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        with open(
            staged_results_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file
            )
            validated_rows = list(
                reader
            )
            validated_fields = reader.fieldnames

        expected_rows = (
            len(settings)
            * NUM_THRESHOLDS
        )

        if validated_fields != fieldnames:
            raise RuntimeError(
                "Unexpected columns in staged benchmark results: "
                f"{validated_fields}"
            )

        if len(validated_rows) != expected_rows:
            raise RuntimeError(
                "Unexpected number of rows in staged benchmark results: "
                f"{len(validated_rows)}. Expected {expected_rows}."
            )

        for name in settings:
            subset_rows = [
                row
                for row in validated_rows
                if row["Difficulty"] == name
            ]

            if len(subset_rows) != NUM_THRESHOLDS:
                raise RuntimeError(
                    f"Unexpected number of {name} threshold rows in "
                    f"staged benchmark results: {len(subset_rows)}"
                )

        staged_results_path.replace(
            results_path
        )

    finally:
        if staging_root.exists():
            shutil.rmtree(
                staging_root
            )

    print()
    print(
        "Saved benchmark results:",
        results_path,
    )

if __name__ == "__main__":
    main()
