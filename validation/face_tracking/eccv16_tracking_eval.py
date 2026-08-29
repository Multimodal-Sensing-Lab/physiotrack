from pathlib import Path
import csv
import time
import xml.etree.ElementTree as ET

import cv2
import motmetrics as mm
import numpy as np

# Compatibility for motmetrics 1.4.0 with NumPy 2.x.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from physiotrack.face import Face, FaceTracker


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "FACE_TRACKING_ECCV2016"
)

RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IOU_THRESHOLD = 0.5

VIDEO_CONFIGS = [
    ("Apink", "Apink.mp4", "Apink_gt.xml"),
    ("BrunoMars", "BrunoMars.mp4", "BrunoMars_gt.xml"),
    ("Darling", "Darling.mp4", "Darling_gt.xml"),
    ("GirlsAloud", "GirlsAloud.mp4", "GirlsAloud_gt.xml"),
    ("HelloBubble", "HelloBubble.mp4", "HelloBubble_gt.xml"),
    ("PussycatDolls", "PussycatDolls.mp4", "stickwitu_gt.xml"),
    ("T-ara", "T-ara.mov", "Tara_gt.xml"),
    ("Westlife", "Westlife.mp4", "Westlife_gt.xml"),
]


def load_ground_truth(xml_path):
    """Load annotated face trajectories and organize them by frame."""
    root = ET.parse(xml_path).getroot()

    frames = {}

    for trajectory in root.findall("Trajectory"):
        object_id = int(trajectory.attrib["obj_id"])

        for frame in trajectory.findall("Frame"):
            frame_no = int(frame.attrib["frame_no"])

            x = float(frame.attrib["x"])
            y = float(frame.attrib["y"])
            width = float(frame.attrib["width"])
            height = float(frame.attrib["height"])

            box = [x, y, x + width, y + height]

            frames.setdefault(frame_no, []).append(
                (object_id, box)
            )

    return frames, int(root.attrib["end_frame"])


def evaluate_video(name, video_path, gt_path):
    """Evaluate one video and return its MOT accumulator and timing data."""
    gt_by_frame, gt_end_frame = load_ground_truth(gt_path)

    detector = Face(
        conf=0.25,
        iou=0.45,
        device="cpu",
        verbose=False,
    )

    tracker = FaceTracker(
        tracker_type="ocsort",
        device="cpu",
    )

    accumulator = mm.MOTAccumulator(auto_id=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_no = 0
    start_time = time.perf_counter()

    print(f"\n=== {name} ===")

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        detections = detector.predict(frame)
        tracks = tracker.track(frame, detections)

        gt_entries = gt_by_frame.get(
            frame_no,
            [],
        )

        gt_ids = [
            obj_id
            for obj_id, _ in gt_entries
        ]

        gt_boxes = [
            box
            for _, box in gt_entries
        ]

        track_ids = [
            int(track.id)
            for track in tracks
        ]

        track_boxes = [
            list(map(float, track.box))
            for track in tracks
        ]

        if gt_boxes and track_boxes:
            distances = mm.distances.iou_matrix(
                np.asarray(
                    gt_boxes,
                    dtype=float,
                ),
                np.asarray(
                    track_boxes,
                    dtype=float,
                ),
                max_iou=(
                    1.0
                    - IOU_THRESHOLD
                ),
            )
        else:
            distances = np.empty(
                (
                    len(gt_boxes),
                    len(track_boxes),
                )
            )

        accumulator.update(
            gt_ids,
            track_ids,
            distances,
        )

        if frame_no % 500 == 0:
            print(
                f"Processed "
                f"{frame_no}/{gt_end_frame} frames"
            )

    cap.release()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if frame_no != gt_end_frame:
        print(
            f"WARNING: processed "
            f"{frame_no} frames, "
            f"but GT end_frame is "
            f"{gt_end_frame}"
        )

    fps = (
        frame_no / elapsed
        if elapsed > 0
        else 0.0
    )

    print(
        f"Completed {name}: "
        f"{frame_no} frames in "
        f"{elapsed / 60:.2f} min "
        f"({fps:.2f} FPS)"
    )

    return (
        accumulator,
        frame_no,
        elapsed,
        fps,
    )


def compute_reporting_values(row):
    """Convert motmetrics values to the reporting format used here."""
    recall = float(row["recall"])
    precision = float(row["precision"])

    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    num_frames = float(
        row["num_frames"]
    )

    faf = (
        float(
            row["num_false_positives"]
        )
        / num_frames
        if num_frames > 0
        else 0.0
    )

    # motmetrics stores IoU distance (1 - IoU) for MOTP.
    motp_raw = float(row["motp"])

    motp_iou = (
        1.0 - motp_raw
        if not np.isnan(motp_raw)
        else float("nan")
    )

    return {
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "FAF": faf,
        "IDS": int(
            row["num_switches"]
        ),
        "Frag": int(
            row["num_fragmentations"]
        ),
        "MOTA": float(
            row["mota"]
        ),
        "MOTP": motp_iou,
        "IDF1": float(
            row["idf1"]
        ),
        "GT_objects": int(
            row["num_objects"]
        ),
        "Predictions": int(
            row["num_predictions"]
        ),
        "Matches": int(
            row["num_matches"]
        ),
        "FN": int(
            row["num_misses"]
        ),
        "FP": int(
            row["num_false_positives"]
        ),
    }


def main():
    metrics = mm.metrics.create()

    metric_names = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_misses",
        "num_false_positives",
        "num_switches",
        "num_fragmentations",
        "recall",
        "precision",
        "idf1",
        "mota",
        "motp",
    ]

    accumulators = []
    names = []
    timing = {}

    total_start = time.perf_counter()

    for (
        name,
        video_name,
        gt_name,
    ) in VIDEO_CONFIGS:
        video_path = (
            DATASET_ROOT
            / "videos"
            / video_name
        )

        gt_path = (
            DATASET_ROOT
            / "ground_truth"
            / "GT"
            / gt_name
        )

        (
            accumulator,
            frames,
            elapsed,
            fps,
        ) = evaluate_video(
            name,
            video_path,
            gt_path,
        )

        accumulators.append(
            accumulator
        )

        names.append(name)

        timing[name] = {
            "frames": frames,
            "seconds": elapsed,
            "fps": fps,
        }

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    summary = metrics.compute_many(
        accumulators,
        names=names,
        metrics=metric_names,
        generate_overall=True,
    )

    print(
        "\n=== Raw MOTMetrics summary ===\n"
    )

    print(
        mm.io.render_summary(
            summary,
            formatters=metrics.formatters,
            namemap=(
                mm.io
                .motchallenge_metric_names
            ),
        )
    )

    reporting_rows = {}

    for name in names + ["OVERALL"]:
        reporting_rows[name] = (
            compute_reporting_values(
                summary.loc[name]
            )
        )

    print(
        "\n=== Paper-compatible metrics ===\n"
    )

    header = (
        f"{'Video':<16}"
        f"{'Recall':>9}"
        f"{'Prec.':>9}"
        f"{'F1':>9}"
        f"{'FAF':>9}"
        f"{'IDS':>8}"
        f"{'Frag':>8}"
        f"{'MOTA':>9}"
        f"{'MOTP':>9}"
        f"{'IDF1':>9}"
    )

    print(header)
    print("-" * len(header))

    for name in names + ["OVERALL"]:
        row = reporting_rows[name]

        print(
            f"{name:<16}"
            f"{row['Recall'] * 100:>8.2f}%"
            f"{row['Precision'] * 100:>8.2f}%"
            f"{row['F1'] * 100:>8.2f}%"
            f"{row['FAF']:>9.4f}"
            f"{row['IDS']:>8d}"
            f"{row['Frag']:>8d}"
            f"{row['MOTA'] * 100:>8.2f}%"
            f"{row['MOTP'] * 100:>8.2f}%"
            f"{row['IDF1'] * 100:>8.2f}%"
        )

    csv_path = (
        RESULTS_DIR
        / "eccv16_tracking_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "Video",
                "Recall_percent",
                "Precision_percent",
                "F1_percent",
                "FAF",
                "IDS",
                "Frag",
                "MOTA_percent",
                "MOTP_IoU_percent",
                "IDF1_percent",
                "GT_objects",
                "Predictions",
                "Matches",
                "FN",
                "FP",
                "Runtime_seconds",
                "Processing_FPS",
            ]
        )

        for name in names + ["OVERALL"]:
            row = reporting_rows[name]

            if name == "OVERALL":
                runtime_seconds = (
                    total_elapsed
                )

                total_frames = sum(
                    timing[item]["frames"]
                    for item in names
                )

                processing_fps = (
                    total_frames
                    / total_elapsed
                    if total_elapsed > 0
                    else 0.0
                )

            else:
                runtime_seconds = (
                    timing[name]["seconds"]
                )

                processing_fps = (
                    timing[name]["fps"]
                )

            writer.writerow(
                [
                    name,
                    f"{row['Recall'] * 100:.4f}",
                    f"{row['Precision'] * 100:.4f}",
                    f"{row['F1'] * 100:.4f}",
                    f"{row['FAF']:.6f}",
                    row["IDS"],
                    row["Frag"],
                    f"{row['MOTA'] * 100:.4f}",
                    f"{row['MOTP'] * 100:.4f}",
                    f"{row['IDF1'] * 100:.4f}",
                    row["GT_objects"],
                    row["Predictions"],
                    row["Matches"],
                    row["FN"],
                    row["FP"],
                    f"{runtime_seconds:.3f}",
                    f"{processing_fps:.4f}",
                ]
            )

    txt_path = (
        RESULTS_DIR
        / "eccv16_tracking_summary.txt"
    )

    total_frames = sum(
        timing[item]["frames"]
        for item in names
    )

    overall_fps = (
        total_frames
        / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "ECCV 2016 Music Video "
            "Face Tracking Validation\n\n"
        )

        file.write("Dataset:\n")
        file.write(
            "ECCV 2016 Music Video "
            "Face Tracking Dataset\n\n"
        )

        file.write(
            "Dataset coverage:\n"
        )
        file.write(
            f"Videos evaluated: "
            f"{len(names)}\n"
        )
        file.write(
            f"Total frames: "
            f"{total_frames}\n"
        )
        file.write(
            f"Ground-truth face annotations: "
            f"{reporting_rows['OVERALL']['GT_objects']}\n\n"
        )

        file.write(
            "Evaluation setup:\n"
        )
        file.write(
            "Face detector confidence: "
            "0.25\n"
        )
        file.write(
            "Face detector IoU: "
            "0.45\n"
        )
        file.write(
            "Tracker: OC-SORT\n"
        )
        file.write(
            "Device: CPU\n"
        )
        file.write(
            "GT/prediction matching "
            f"IoU threshold: "
            f"{IOU_THRESHOLD}\n"
        )
        file.write(
            "Evaluation library: "
            "motmetrics 1.4.0\n\n"
        )

        file.write("Metrics:\n")
        file.write(
            "Recall, Precision, F1, FAF, "
            "IDS, Frag, MOTA, MOTP "
            "(mean matched IoU), IDF1\n\n"
        )

        file.write("Results:\n")

        for name in names + ["OVERALL"]:
            row = reporting_rows[name]

            file.write(
                f"\n{name}\n"
            )

            file.write(
                f"Recall: "
                f"{row['Recall'] * 100:.2f}%\n"
            )

            file.write(
                f"Precision: "
                f"{row['Precision'] * 100:.2f}%\n"
            )

            file.write(
                f"F1: "
                f"{row['F1'] * 100:.2f}%\n"
            )

            file.write(
                f"FAF: "
                f"{row['FAF']:.4f}\n"
            )

            file.write(
                f"IDS: "
                f"{row['IDS']}\n"
            )

            file.write(
                f"Frag: "
                f"{row['Frag']}\n"
            )

            file.write(
                f"MOTA: "
                f"{row['MOTA'] * 100:.2f}%\n"
            )

            file.write(
                f"MOTP: "
                f"{row['MOTP'] * 100:.2f}%\n"
            )

            file.write(
                f"IDF1: "
                f"{row['IDF1'] * 100:.2f}%\n"
            )

        overall = reporting_rows[
            "OVERALL"
        ]

        file.write(
            "\nOverall count statistics:\n"
        )
        file.write(
            f"GT objects: "
            f"{overall['GT_objects']}\n"
        )
        file.write(
            f"Predictions: "
            f"{overall['Predictions']}\n"
        )
        file.write(
            f"Matches: "
            f"{overall['Matches']}\n"
        )
        file.write(
            f"False negatives: "
            f"{overall['FN']}\n"
        )
        file.write(
            f"False positives: "
            f"{overall['FP']}\n"
        )
        file.write(
            f"Identity switches: "
            f"{overall['IDS']}\n"
        )
        file.write(
            f"Fragmentations: "
            f"{overall['Frag']}\n"
        )

        file.write("\nRuntime:\n")
        file.write(
            f"Total frames: "
            f"{total_frames}\n"
        )
        file.write(
            f"Total runtime: "
            f"{total_elapsed / 60:.2f} min\n"
        )
        file.write(
            f"Overall processing speed: "
            f"{overall_fps:.2f} FPS\n"
        )

        file.write(
            "\nGenerated outputs:\n"
        )
        file.write(
            "Detailed results CSV: "
            "results/eccv16_tracking_results.csv\n"
        )
        file.write(
            "Validation summary: "
            "results/eccv16_tracking_summary.txt\n"
        )
        file.write(
            "Thesis table CSV: "
            "results/eccv16_tracking_thesis_table.csv\n"
        )
        file.write(
            "Thesis table Markdown: "
            "results/eccv16_tracking_thesis_table.md\n"
        )
        file.write(
            "Metrics figure: "
            "results/figures/"
            "eccv16_tracking_metrics.png\n"
        )

    print("\nSaved:")
    print(csv_path)
    print(txt_path)

    print(
        f"\nTotal runtime: "
        f"{total_elapsed / 60:.2f} min"
    )

    print(
        f"Overall processing speed: "
        f"{overall_fps:.2f} FPS"
    )


if __name__ == "__main__":
    main()