from pathlib import Path
import csv
import shutil
import xml.etree.ElementTree as ET

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd

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

QUANTITATIVE_CSV = (
    RESULTS_DIR
    / "eccv16_tracking_results.csv"
)

QUALITATIVE_DIR = (
    RESULTS_DIR
    / "qualitative"
)

VIDEOS_DIR = (
    QUALITATIVE_DIR
    / "annotated_videos"
)

FRAMES_DIR = (
    QUALITATIVE_DIR
    / "representative_frames"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

OUTPUT_VIDEO = (
    VIDEOS_DIR
    / "T-ara_face_tracking_qualitative.mp4"
)

SELECTION_CSV = (
    QUALITATIVE_DIR
    / "eccv16_tracking_qualitative_selection.csv"
)

EVENTS_CSV = (
    QUALITATIVE_DIR
    / "eccv16_tracking_qualitative_events.csv"
)

SUMMARY_FIGURE = (
    FIGURES_DIR
    / "eccv16_tracking_qualitative_examples.png"
)



SEQUENCE_NAME = "T-ara"
VIDEO_NAME = "T-ara.mov"
GT_NAME = "Tara_gt.xml"

DETECTOR_CONFIDENCE = 0.25
DETECTOR_IOU = 0.45
MATCHING_IOU = 0.50
DEVICE = "cpu"

TARGET_CLIP_SECONDS = 60.0

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

VIDEO_CANVAS_WIDTH = 1440
VIDEO_CANVAS_HEIGHT = 1080
PANEL_WIDTH = OUTPUT_WIDTH - VIDEO_CANVAS_WIDTH

REPRESENTATIVE_FRAME_COUNT = 6



COLOR_MATCH = (0, 210, 0)
COLOR_SWITCH = (255, 0, 255)
COLOR_FP = (0, 165, 255)
COLOR_MISS = (0, 0, 255)
COLOR_GT = (255, 220, 0)

COLOR_WHITE = (245, 245, 245)
COLOR_LIGHT = (205, 205, 205)
COLOR_MUTED = (160, 160, 160)
COLOR_PANEL = (25, 25, 25)
COLOR_SECTION = (34, 34, 34)
COLOR_LINE = (70, 70, 70)



def load_ground_truth(xml_path):
    """Load ECCV 2016 XML trajectories using one-based source frame numbers."""
    root = ET.parse(xml_path).getroot()

    frames = {}

    for trajectory in root.findall("Trajectory"):
        gt_id = int(trajectory.attrib["obj_id"])

        for frame in trajectory.findall("Frame"):
            frame_no = int(frame.attrib["frame_no"])

            x = float(frame.attrib["x"])
            y = float(frame.attrib["y"])
            width = float(frame.attrib["width"])
            height = float(frame.attrib["height"])

            box = [
                x,
                y,
                x + width,
                y + height,
            ]

            frames.setdefault(frame_no, []).append(
                (gt_id, box)
            )

    return frames, int(root.attrib["end_frame"])


def verify_inputs():
    """Verify the required benchmark files and accepted quantitative results."""
    video_path = (
        DATASET_ROOT
        / "videos"
        / VIDEO_NAME
    )

    gt_path = (
        DATASET_ROOT
        / "ground_truth"
        / "GT"
        / GT_NAME
    )

    missing = []

    for path in [
        video_path,
        gt_path,
        QUANTITATIVE_CSV,
    ]:
        if not path.is_file():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(missing)
        )

    return video_path, gt_path


def load_quantitative_row():
    """Read the accepted T-ara result row from the final quantitative CSV."""
    df = pd.read_csv(QUANTITATIVE_CSV)

    required_columns = {
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
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "The accepted quantitative CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    row = df[df["Video"] == SEQUENCE_NAME]

    if len(row) != 1:
        raise RuntimeError(
            "Could not uniquely locate the accepted T-ara result row."
        )

    return row.iloc[0]


def create_models():
    """Create the exact detector and tracker configuration used in evaluation."""
    detector = Face(
        conf=DETECTOR_CONFIDENCE,
        iou=DETECTOR_IOU,
        device=DEVICE,
        verbose=False,
    )

    tracker = FaceTracker(
        tracker_type="ocsort",
        device=DEVICE,
    )

    return detector, tracker


def extract_tracks(tracks):
    """Convert PhysioTrack track objects into portable numeric records."""
    return [
        (
            int(track.id),
            list(map(float, track.box)),
        )
        for track in tracks
    ]


def build_iou_distance_matrix(
    gt_entries,
    track_entries,
):
    """Build the same IoU-distance matrix used by the quantitative evaluator."""
    gt_boxes = [
        box
        for _, box in gt_entries
    ]

    track_boxes = [
        box
        for _, box in track_entries
    ]

    if gt_boxes and track_boxes:
        return mm.distances.iou_matrix(
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
                - MATCHING_IOU
            ),
        )

    return np.empty(
        (
            len(gt_boxes),
            len(track_boxes),
        )
    )



def compute_reporting_values(row):
    """Convert MOTMetrics values to the same reporting format as the evaluator."""
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

    num_frames = float(row["num_frames"])

    faf = (
        float(row["num_false_positives"])
        / num_frames
        if num_frames > 0
        else 0.0
    )

    motp_raw = float(row["motp"])

    motp_iou = (
        1.0 - motp_raw
        if not np.isnan(motp_raw)
        else float("nan")
    )

    return {
        "Recall_percent": recall * 100.0,
        "Precision_percent": precision * 100.0,
        "F1_percent": f1 * 100.0,
        "FAF": faf,
        "IDS": int(row["num_switches"]),
        "Frag": int(row["num_fragmentations"]),
        "MOTA_percent": float(row["mota"]) * 100.0,
        "MOTP_IoU_percent": motp_iou * 100.0,
        "IDF1_percent": float(row["idf1"]) * 100.0,
        "GT_objects": int(row["num_objects"]),
        "Predictions": int(row["num_predictions"]),
        "Matches": int(row["num_matches"]),
        "FN": int(row["num_misses"]),
        "FP": int(row["num_false_positives"]),
    }


def verify_reproduced_sequence_metrics(
    reproduced,
    accepted,
):
    """
    Verify that the qualitative analysis run reproduces the accepted T-ara metrics.

    This prevents a visually attractive output from being generated from a run
    that does not match the accepted quantitative benchmark configuration.
    """
    integer_columns = [
        "IDS",
        "Frag",
        "GT_objects",
        "Predictions",
        "Matches",
        "FN",
        "FP",
    ]

    float_columns = [
        "Recall_percent",
        "Precision_percent",
        "F1_percent",
        "FAF",
        "MOTA_percent",
        "MOTP_IoU_percent",
        "IDF1_percent",
    ]

    problems = []

    for column in integer_columns:
        expected = int(accepted[column])
        actual = int(reproduced[column])

        if actual != expected:
            problems.append(
                f"{column}: accepted={expected}, reproduced={actual}"
            )

    tolerances = {
        "FAF": 0.0001,
        "Recall_percent": 0.01,
        "Precision_percent": 0.01,
        "F1_percent": 0.01,
        "MOTA_percent": 0.01,
        "MOTP_IoU_percent": 0.01,
        "IDF1_percent": 0.01,
    }

    for column in float_columns:
        expected = float(accepted[column])
        actual = float(reproduced[column])

        if abs(actual - expected) > tolerances[column]:
            problems.append(
                f"{column}: accepted={expected:.6f}, "
                f"reproduced={actual:.6f}"
            )

    if problems:
        raise RuntimeError(
            "The qualitative analysis run did not reproduce the accepted "
            "T-ara quantitative result. No qualitative output should be "
            "accepted from this run.\n\n"
            + "\n".join(problems)
        )


def annotate_switch_history(events):
    """
    Add the previous assigned track ID to exact MOTMetrics SWITCH events.

    SWITCH itself is defined by MOTMetrics. This function only makes the
    previous assignment explicit for visualization.
    """
    events = events.copy()

    events["PreviousHId"] = np.nan

    previous_assignment = {}

    for index, row in events.iterrows():
        event_type = str(row["Type"])

        if event_type not in {
            "MATCH",
            "SWITCH",
        }:
            continue

        gt_id = int(row["OId"])
        track_id = int(row["HId"])

        if event_type == "SWITCH":
            previous_track = previous_assignment.get(
                gt_id
            )

            if previous_track is not None:
                events.at[
                    index,
                    "PreviousHId",
                ] = previous_track

        previous_assignment[
            gt_id
        ] = track_id

    return events


def build_frame_event_table(
    events,
    gt_by_frame,
    total_frames,
):
    """Build exact per-frame MOTMetrics statistics for deterministic selection."""
    stats = []

    grouped = {
        int(frame_no): group
        for frame_no, group
        in events.groupby("source_frame")
    }

    for frame_no in range(
        1,
        total_frames + 1,
    ):
        frame_events = grouped.get(
            frame_no
        )

        counts = {
            "MATCH": 0,
            "SWITCH": 0,
            "MISS": 0,
            "FP": 0,
        }

        mean_iou = 0.0

        if frame_events is not None:
            for event_type in counts:
                counts[event_type] = int(
                    (
                        frame_events["Type"]
                        == event_type
                    ).sum()
                )

            detections = frame_events[
                frame_events["Type"].isin(
                    ["MATCH", "SWITCH"]
                )
            ]

            if len(detections) > 0:
                mean_iou = float(
                    (
                        1.0
                        - detections["D"].astype(float)
                    ).mean()
                )

        gt_count = len(
            gt_by_frame.get(
                frame_no,
                [],
            )
        )

        stats.append(
            {
                "frame_no": frame_no,
                "gt_faces": gt_count,
                "matches": counts["MATCH"],
                "switches": counts["SWITCH"],
                "misses": counts["MISS"],
                "false_positives": counts["FP"],
                "detections": (
                    counts["MATCH"]
                    + counts["SWITCH"]
                ),
                "mean_iou": mean_iou,
            }
        )

    return stats


def rolling_sum(values, window):
    """Efficient rolling sum for deterministic clip selection."""
    values = np.asarray(
        values,
        dtype=float,
    )

    prefix = np.concatenate(
        (
            np.zeros(
                1,
                dtype=float,
            ),
            np.cumsum(values),
        )
    )

    return (
        prefix[window:]
        - prefix[:-window]
    )


def select_tracking_window(
    frame_stats,
    fps,
):
    """
    Select the most informative 60-second T-ara window.

    Primary criterion:
        highest number of exact MOTMetrics SWITCH events.

    Tie-breakers:
        more successful detections,
        fewer misses,
        fewer false positives.

    This keeps the clip focused on temporal tracking behavior while avoiding
    a selection rule that simply maximizes failure.
    """
    window_frames = max(
        1,
        int(
            round(
                TARGET_CLIP_SECONDS
                * fps
            )
        ),
    )

    window_frames = min(
        window_frames,
        len(frame_stats),
    )

    switches = rolling_sum(
        [
            row["switches"]
            for row in frame_stats
        ],
        window_frames,
    )

    detections = rolling_sum(
        [
            row["detections"]
            for row in frame_stats
        ],
        window_frames,
    )

    misses = rolling_sum(
        [
            row["misses"]
            for row in frame_stats
        ],
        window_frames,
    )

    fps_values = rolling_sum(
        [
            row["false_positives"]
            for row in frame_stats
        ],
        window_frames,
    )

    best_index = 0
    best_score = None

    for index in range(
        len(switches)
    ):
        score = (
            int(switches[index]),
            int(detections[index]),
            -int(misses[index]),
            -int(fps_values[index]),
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score
            best_index = index

    start_frame = (
        best_index + 1
    )

    end_frame = (
        start_frame
        + window_frames
        - 1
    )

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "window_frames": window_frames,
        "switches": int(best_score[0]),
        "detections": int(best_score[1]),
        "misses": int(-best_score[2]),
        "false_positives": int(-best_score[3]),
        "selection_rule": (
            "Maximize exact MOTMetrics SWITCH events; "
            "tie-break by more detections, fewer misses, "
            "then fewer false positives."
        ),
    }


def run_full_sequence_analysis(
    video_path,
    gt_by_frame,
    gt_end_frame,
    accepted_row,
):
    """
    Re-run T-ara with the exact accepted detector/tracker configuration.

    The MOTAccumulator is the same evaluation mechanism used by the accepted
    quantitative evaluator. Therefore the qualitative event labels come from
    exact MOTMetrics MATCH/SWITCH/MISS/FP events, not a custom approximation.
    """
    detector, tracker = create_models()

    accumulator = mm.MOTAccumulator(
        auto_id=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open T-ara."
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        not np.isfinite(fps)
        or fps <= 0.0
    ):
        cap.release()

        raise RuntimeError(
            "Could not determine valid video FPS."
        )

    frame_no = 0

    print(
        "\nPass 1/2: exact full-sequence MOTMetrics analysis"
    )

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        detections = detector.predict(
            frame
        )

        tracks = tracker.track(
            frame,
            detections,
        )

        gt_entries = gt_by_frame.get(
            frame_no,
            [],
        )

        track_entries = extract_tracks(
            tracks
        )

        gt_ids = [
            gt_id
            for gt_id, _
            in gt_entries
        ]

        track_ids = [
            track_id
            for track_id, _
            in track_entries
        ]

        distances = build_iou_distance_matrix(
            gt_entries,
            track_entries,
        )

        accumulator.update(
            gt_ids,
            track_ids,
            distances,
        )

        if (
            frame_no
            % 500
            == 0
        ):
            print(
                f"Analyzed "
                f"{frame_no}/"
                f"{gt_end_frame} frames"
            )

    cap.release()

    if frame_no != gt_end_frame:
        raise RuntimeError(
            "Video/ground-truth frame mismatch: "
            f"processed {frame_no}, "
            f"GT end_frame {gt_end_frame}."
        )

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

    summary = metrics.compute(
        accumulator,
        metrics=metric_names,
        name=SEQUENCE_NAME,
    )

    reproduced = compute_reporting_values(
        summary.loc[
            SEQUENCE_NAME
        ]
    )

    verify_reproduced_sequence_metrics(
        reproduced,
        accepted_row,
    )

    events = (
        accumulator
        .mot_events
        .reset_index()
    )

    # MOTAccumulator auto_id starts at FrameId 0.
    # Dataset/source frame numbering is one-based.
    events["source_frame"] = (
        events["FrameId"].astype(int)
        + 1
    )

    events = annotate_switch_history(
        events
    )

    frame_stats = build_frame_event_table(
        events,
        gt_by_frame,
        gt_end_frame,
    )

    selection = select_tracking_window(
        frame_stats,
        fps,
    )

    print(
        "\nExact T-ara quantitative reproduction verified."
    )

    print(
        "Selected qualitative window:"
    )

    print(
        f"Frames: "
        f"{selection['start_frame']}-"
        f"{selection['end_frame']}"
    )

    print(
        f"Duration: "
        f"{selection['window_frames'] / fps:.2f} s"
    )

    print(
        f"Exact MOTMetrics SWITCH events in clip: "
        f"{selection['switches']}"
    )

    print(
        f"Detected face observations in clip: "
        f"{selection['detections']}"
    )

    print(
        f"MISS events in clip: "
        f"{selection['misses']}"
    )

    print(
        f"FP events in clip: "
        f"{selection['false_positives']}"
    )

    return {
        "fps": fps,
        "events": events,
        "frame_stats": frame_stats,
        "selection": selection,
        "reproduced": reproduced,
    }



def clean_previous_qualitative_outputs():
    """
    Remove only outputs owned by this qualitative generator.

    Never remove or modify:
    - accepted quantitative CSV/TXT/Markdown outputs,
    - accepted quantitative metrics figure,
    - dataset files.
    """
    if QUALITATIVE_DIR.exists():
        shutil.rmtree(
            QUALITATIVE_DIR
        )

    QUALITATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIDEOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUMMARY_FIGURE.exists():
        SUMMARY_FIGURE.unlink()



def fit_frame_to_canvas(frame):
    """Letterbox the source frame into a 1440x1080 video canvas."""
    source_height, source_width = (
        frame.shape[:2]
    )

    scale = min(
        VIDEO_CANVAS_WIDTH
        / source_width,
        VIDEO_CANVAS_HEIGHT
        / source_height,
    )

    resized_width = max(
        1,
        int(
            round(
                source_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                source_height
                * scale
            )
        ),
    )

    resized = cv2.resize(
        frame,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros(
        (
            VIDEO_CANVAS_HEIGHT,
            VIDEO_CANVAS_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        VIDEO_CANVAS_WIDTH
        - resized_width
    ) // 2

    offset_y = (
        VIDEO_CANVAS_HEIGHT
        - resized_height
    ) // 2

    canvas[
        offset_y:
        offset_y
        + resized_height,
        offset_x:
        offset_x
        + resized_width,
    ] = resized

    transform = {
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }

    return canvas, transform


def transform_box(
    box,
    transform,
):
    """Map one source xyxy box to the rendered video canvas."""
    scale = transform["scale"]

    offset_x = transform[
        "offset_x"
    ]

    offset_y = transform[
        "offset_y"
    ]

    return (
        int(
            round(
                box[0]
                * scale
                + offset_x
            )
        ),
        int(
            round(
                box[1]
                * scale
                + offset_y
            )
        ),
        int(
            round(
                box[2]
                * scale
                + offset_x
            )
        ),
        int(
            round(
                box[3]
                * scale
                + offset_y
            )
        ),
    )


def boxes_overlap(
    first,
    second,
    margin=4,
):
    """Return True when two label rectangles overlap."""
    return not (
        first[2] + margin
        <= second[0]
        or second[2] + margin
        <= first[0]
        or first[3] + margin
        <= second[1]
        or second[3] + margin
        <= first[1]
    )


def find_label_rectangle(
    image,
    box,
    label_width,
    label_height,
    occupied_labels,
):
    """Find a readable label position without covering another label."""
    x1, y1, x2, y2 = box

    image_height, image_width = (
        image.shape[:2]
    )

    horizontal_positions = [
        x1,
        x2 - label_width,
        int(
            round(
                (
                    x1
                    + x2
                    - label_width
                )
                / 2.0
            )
        ),
    ]

    horizontal_positions = [
        max(
            0,
            min(
                image_width
                - label_width
                - 1,
                value,
            ),
        )
        for value in horizontal_positions
    ]

    vertical_candidates = []

    for offset_index in range(8):
        offset = (
            offset_index
            * (
                label_height
                + 4
            )
        )

        vertical_candidates.append(
            y1
            - label_height
            - offset
        )

        vertical_candidates.append(
            y2
            + 1
            + offset
        )

    for top in vertical_candidates:
        if (
            top < 0
            or top
            + label_height
            >= image_height
        ):
            continue

        for left in horizontal_positions:
            rectangle = (
                left,
                top,
                left + label_width,
                top + label_height,
            )

            if all(
                not boxes_overlap(
                    rectangle,
                    occupied,
                )
                for occupied
                in occupied_labels
            ):
                return rectangle

    left = horizontal_positions[0]

    top = max(
        0,
        min(
            image_height
            - label_height
            - 1,
            y1
            - label_height,
        ),
    )

    return (
        left,
        top,
        left + label_width,
        top + label_height,
    )


def draw_box_label(
    image,
    box,
    label,
    color,
    occupied_labels,
    thickness=3,
):
    """Draw one labeled box with collision-aware label placement."""
    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
        cv2.LINE_AA,
    )

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.52
    text_thickness = 1

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )

    label_width = (
        text_width
        + 10
    )

    label_height = (
        text_height
        + baseline
        + 8
    )

    (
        left,
        top,
        right,
        bottom,
    ) = find_label_rectangle(
        image,
        box,
        label_width,
        label_height,
        occupied_labels,
    )

    occupied_labels.append(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    if (
        bottom < y1 - 2
        or top > y2 + 2
    ):
        label_center = (
            int(
                round(
                    (
                        left
                        + right
                    )
                    / 2.0
                )
            ),
            (
                bottom
                if bottom < y1
                else top
            ),
        )

        box_anchor = (
            int(
                round(
                    (
                        x1
                        + x2
                    )
                    / 2.0
                )
            ),
            (
                y1
                if bottom < y1
                else y2
            ),
        )

        cv2.line(
            image,
            box_anchor,
            label_center,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        color,
        -1,
    )

    text_y = (
        bottom
        - baseline
        - 3
    )

    cv2.putText(
        image,
        label,
        (
            left + 5,
            text_y,
        ),
        font,
        font_scale,
        (
            0,
            0,
            0,
        ),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_text(
    image,
    text,
    x,
    y,
    scale=0.48,
    color=COLOR_WHITE,
    thickness=1,
):
    """Draw one anti-aliased text line."""
    cv2.putText(
        image,
        str(text),
        (
            int(x),
            int(y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_section_box(
    panel,
    top,
    bottom,
):
    """Draw a subtle panel section background."""
    cv2.rectangle(
        panel,
        (12, top),
        (
            PANEL_WIDTH - 12,
            bottom,
        ),
        COLOR_SECTION,
        -1,
    )

    cv2.rectangle(
        panel,
        (12, top),
        (
            PANEL_WIDTH - 12,
            bottom,
        ),
        COLOR_LINE,
        1,
    )


def event_rows_for_frame(
    events,
    frame_no,
):
    """Return exact MOTMetrics events for one source frame."""
    return events[
        events[
            "source_frame"
        ]
        == frame_no
    ]


def render_tracking_frame(
    frame,
    frame_no,
    gt_entries,
    track_entries,
    frame_events,
    clip_totals,
    accepted_row,
    selection,
    fps,
):
    """
    Render one frame using exact MOTMetrics event labels.

    MATCH, SWITCH, MISS, and FP are taken directly from the same
    MOTAccumulator protocol used in the accepted quantitative evaluator.
    """
    canvas, transform = (
        fit_frame_to_canvas(
            frame
        )
    )

    gt_box_map = {
        int(gt_id): box
        for gt_id, box
        in gt_entries
    }

    track_box_map = {
        int(track_id): box
        for track_id, box
        in track_entries
    }

    for gt_id, gt_box in gt_entries:
        transformed = transform_box(
            gt_box,
            transform,
        )

        cv2.rectangle(
            canvas,
            (
                transformed[0],
                transformed[1],
            ),
            (
                transformed[2],
                transformed[3],
            ),
            COLOR_GT,
            1,
            cv2.LINE_AA,
        )

    occupied_labels = []

    current_counts = {
        "MATCH": 0,
        "SWITCH": 0,
        "MISS": 0,
        "FP": 0,
    }

    for _, event in frame_events.iterrows():
        event_type = str(
            event["Type"]
        )

        if event_type not in current_counts:
            continue

        current_counts[
            event_type
        ] += 1

        if event_type in {
            "MATCH",
            "SWITCH",
        }:
            gt_id = int(
                event["OId"]
            )

            track_id = int(
                event["HId"]
            )

            if (
                gt_id
                not in gt_box_map
                or track_id
                not in track_box_map
            ):
                raise RuntimeError(
                    "Pass-2 tracking output does not match the exact "
                    "MOTMetrics event record at "
                    f"source frame {frame_no}."
                )

            transformed = transform_box(
                track_box_map[
                    track_id
                ],
                transform,
            )

            if event_type == "SWITCH":
                previous = event[
                    "PreviousHId"
                ]

                if pd.isna(
                    previous
                ):
                    label = (
                        f"SWITCH GT{gt_id} -> T{track_id}"
                    )
                else:
                    label = (
                        f"SWITCH GT{gt_id} "
                        f"T{int(previous)} -> T{track_id}"
                    )

                color = COLOR_SWITCH

            else:
                label = (
                    f"T{track_id} | GT{gt_id}"
                )

                color = COLOR_MATCH

            draw_box_label(
                canvas,
                transformed,
                label,
                color,
                occupied_labels,
                thickness=3,
            )

        elif event_type == "MISS":
            gt_id = int(
                event["OId"]
            )

            if gt_id not in gt_box_map:
                raise RuntimeError(
                    "MISS event GT identity is absent from "
                    f"source frame {frame_no}."
                )

            transformed = transform_box(
                gt_box_map[
                    gt_id
                ],
                transform,
            )

            draw_box_label(
                canvas,
                transformed,
                f"MISS GT{gt_id}",
                COLOR_MISS,
                occupied_labels,
                thickness=3,
            )

        elif event_type == "FP":
            track_id = int(
                event["HId"]
            )

            if track_id not in track_box_map:
                raise RuntimeError(
                    "FP event track identity is absent from "
                    f"source frame {frame_no}."
                )

            transformed = transform_box(
                track_box_map[
                    track_id
                ],
                transform,
            )

            draw_box_label(
                canvas,
                transformed,
                f"FP T{track_id}",
                COLOR_FP,
                occupied_labels,
                thickness=3,
            )

    expected_track_ids = set(
        track_box_map.keys()
    )

    represented_track_ids = set()

    for _, event in frame_events.iterrows():
        if (
            str(event["Type"])
            in {
                "MATCH",
                "SWITCH",
                "FP",
            }
            and not pd.isna(
                event["HId"]
            )
        ):
            represented_track_ids.add(
                int(event["HId"])
            )

    if (
        represented_track_ids
        != expected_track_ids
    ):
        raise RuntimeError(
            "The pass-2 track set differs from the exact pass-1 "
            "MOTMetrics event record at "
            f"source frame {frame_no}."
        )

    panel = np.full(
        (
            OUTPUT_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        COLOR_PANEL,
        dtype=np.uint8,
    )

    draw_text(
        panel,
        "ECCV 2016 Face Tracking",
        24,
        40,
        scale=0.67,
        thickness=2,
    )

    draw_text(
        panel,
        "PhysioTrack qualitative benchmark",
        24,
        70,
        scale=0.43,
        color=COLOR_LIGHT,
    )

    draw_section_box(
        panel,
        96,
        186,
    )

    draw_text(
        panel,
        "SEQUENCE",
        26,
        120,
        scale=0.41,
        color=COLOR_MUTED,
        thickness=2,
    )

    draw_text(
        panel,
        "T-ara",
        26,
        149,
        scale=0.58,
        thickness=2,
    )

    clip_time = (
        frame_no
        - selection[
            "start_frame"
        ]
    ) / fps

    clip_duration = (
        selection[
            "window_frames"
        ]
        / fps
    )

    draw_text(
        panel,
        (
            f"Time {clip_time:05.1f} / "
            f"{clip_duration:05.1f} s   |   "
            f"Frame {frame_no}"
        ),
        26,
        175,
        scale=0.40,
        color=COLOR_LIGHT,
    )

    draw_section_box(
        panel,
        200,
        395,
    )

    draw_text(
        panel,
        "CURRENT FRAME - EXACT MOTMETRICS EVENTS",
        26,
        226,
        scale=0.39,
        color=COLOR_MUTED,
        thickness=2,
    )

    current_lines = [
        (
            "GT faces",
            len(gt_entries),
        ),
        (
            "Active tracks",
            len(track_entries),
        ),
        (
            "MATCH",
            current_counts["MATCH"],
        ),
        (
            "SWITCH",
            current_counts["SWITCH"],
        ),
        (
            "MISS",
            current_counts["MISS"],
        ),
        (
            "FP",
            current_counts["FP"],
        ),
    ]

    y = 258

    for label, value in current_lines:
        draw_text(
            panel,
            f"{label}:",
            28,
            y,
            scale=0.45,
            color=COLOR_LIGHT,
        )

        draw_text(
            panel,
            str(value),
            250,
            y,
            scale=0.48,
            thickness=2,
        )

        y += 23

    matched_rows = frame_events[
        frame_events["Type"].isin(
            [
                "MATCH",
                "SWITCH",
            ]
        )
    ]

    mean_iou = 0.0

    if len(matched_rows) > 0:
        mean_iou = float(
            (
                1.0
                - matched_rows[
                    "D"
                ].astype(float)
            ).mean()
        )

    draw_text(
        panel,
        "Mean matched IoU:",
        28,
        387,
        scale=0.44,
        color=COLOR_LIGHT,
    )

    draw_text(
        panel,
        f"{mean_iou:.3f}",
        250,
        387,
        scale=0.47,
        thickness=2,
    )

    draw_section_box(
        panel,
        410,
        560,
    )

    draw_text(
        panel,
        "CUMULATIVE EVENTS IN THIS 60-S CLIP",
        26,
        436,
        scale=0.39,
        color=COLOR_MUTED,
        thickness=2,
    )

    clip_lines = [
        (
            "MATCH",
            clip_totals["MATCH"],
        ),
        (
            "SWITCH",
            clip_totals["SWITCH"],
        ),
        (
            "MISS",
            clip_totals["MISS"],
        ),
        (
            "FP",
            clip_totals["FP"],
        ),
    ]

    y = 470

    for label, value in clip_lines:
        draw_text(
            panel,
            f"{label}:",
            28,
            y,
            scale=0.45,
            color=COLOR_LIGHT,
        )

        draw_text(
            panel,
            str(value),
            250,
            y,
            scale=0.48,
            thickness=2,
        )

        y += 25

    draw_section_box(
        panel,
        575,
        850,
    )

    draw_text(
        panel,
        "ACCEPTED FULL-SEQUENCE RESULTS",
        26,
        601,
        scale=0.40,
        color=COLOR_MUTED,
        thickness=2,
    )

    metric_lines = [
        (
            "Recall",
            f"{float(accepted_row['Recall_percent']):.2f}%",
        ),
        (
            "Precision",
            f"{float(accepted_row['Precision_percent']):.2f}%",
        ),
        (
            "F1",
            f"{float(accepted_row['F1_percent']):.2f}%",
        ),
        (
            "FAF",
            f"{float(accepted_row['FAF']):.4f}",
        ),
        (
            "IDS",
            str(int(accepted_row["IDS"])),
        ),
        (
            "Fragmentations",
            str(int(accepted_row["Frag"])),
        ),
        (
            "MOTA",
            f"{float(accepted_row['MOTA_percent']):.2f}%",
        ),
        (
            "MOTP",
            f"{float(accepted_row['MOTP_IoU_percent']):.2f}%",
        ),
        (
            "IDF1",
            f"{float(accepted_row['IDF1_percent']):.2f}%",
        ),
    ]

    y = 635

    for label, value in metric_lines:
        draw_text(
            panel,
            f"{label}:",
            28,
            y,
            scale=0.44,
            color=COLOR_LIGHT,
        )

        draw_text(
            panel,
            value,
            250,
            y,
            scale=0.47,
            thickness=2,
        )

        y += 23

    draw_text(
        panel,
        "Sequence: complete T-ara benchmark video",
        26,
        832,
        scale=0.36,
        color=COLOR_MUTED,
    )

    draw_section_box(
        panel,
        865,
        970,
    )

    draw_text(
        panel,
        "LEGEND",
        26,
        891,
        scale=0.40,
        color=COLOR_MUTED,
        thickness=2,
    )

    legend = [
        (
            COLOR_MATCH,
            "MATCH",
        ),
        (
            COLOR_SWITCH,
            "SWITCH",
        ),
        (
            COLOR_MISS,
            "MISS",
        ),
        (
            COLOR_FP,
            "FP",
        ),
        (
            COLOR_GT,
            "GT reference",
        ),
    ]

    y = 921

    for index, (
        color,
        label,
    ) in enumerate(legend):
        column = (
            0
            if index < 3
            else 1
        )

        row = (
            index
            if index < 3
            else index - 3
        )

        x = (
            28
            if column == 0
            else 250
        )

        yy = (
            y
            + row * 22
        )

        cv2.rectangle(
            panel,
            (
                x,
                yy - 11,
            ),
            (
                x + 15,
                yy + 3,
            ),
            color,
            -1,
        )

        draw_text(
            panel,
            label,
            x + 22,
            yy + 2,
            scale=0.35,
            color=COLOR_LIGHT,
        )

    draw_section_box(
        panel,
        985,
        1065,
    )

    draw_text(
        panel,
        "METHOD NOTE",
        26,
        1009,
        scale=0.38,
        color=COLOR_MUTED,
        thickness=2,
    )

    note_lines = [
        "Frame events are exact MOTMetrics events from",
        "the accepted evaluator protocol (IoU = 0.50).",
        "Full-sequence metrics are read from the final",
        "accepted quantitative result CSV.",
    ]

    y = 1030

    for line in note_lines:
        draw_text(
            panel,
            line,
            26,
            y,
            scale=0.32,
            color=COLOR_LIGHT,
        )

        y += 15

    return np.hstack(
        (
            canvas,
            panel,
        )
    )



def choose_representative_frames(
    frame_stats,
    selection,
):
    """
    Choose six complementary frames from the selected 60-second window.

    Roles:
    1. stable multi-face tracking
    2. exact identity switch
    3. false-negative challenge
    4. false-positive challenge
    5. crowded tracking
    6. localization challenge

    The frame choices are deterministic and use exact MOTMetrics event counts.
    """
    start = selection[
        "start_frame"
    ]

    end = selection[
        "end_frame"
    ]

    candidates = [
        row
        for row in frame_stats
        if (
            start
            <= row[
                "frame_no"
            ]
            <= end
        )
    ]

    used = set()
    selected = []

    def choose(
        role,
        filtered,
        key,
        reverse=True,
    ):
        pool = [
            row
            for row in filtered
            if row[
                "frame_no"
            ]
            not in used
        ]

        if not pool:
            return False

        row = sorted(
            pool,
            key=key,
            reverse=reverse,
        )[0]

        used.add(
            row[
                "frame_no"
            ]
        )

        selected.append(
            {
                "role": role,
                **row,
            }
        )

        return True

    stable = [
        row
        for row in candidates
        if (
            row["switches"] == 0
            and row["misses"] == 0
            and row["false_positives"] == 0
            and row["detections"] >= 2
        )
    ]

    choose(
        "stable_multi_face",
        stable,
        key=lambda row: (
            row["detections"],
            row["mean_iou"],
        ),
    )

    switch_frames = [
        row
        for row in candidates
        if row["switches"] > 0
    ]

    choose(
        "identity_switch",
        switch_frames,
        key=lambda row: (
            row["switches"],
            row["detections"],
            -row["misses"],
        ),
    )

    miss_frames = [
        row
        for row in candidates
        if row["misses"] > 0
    ]

    choose(
        "false_negative",
        miss_frames,
        key=lambda row: (
            row["misses"],
            row["detections"],
        ),
    )

    fp_frames = [
        row
        for row in candidates
        if row["false_positives"] > 0
    ]

    choose(
        "false_positive",
        fp_frames,
        key=lambda row: (
            row["false_positives"],
            row["detections"],
        ),
    )

    choose(
        "crowded_tracking",
        candidates,
        key=lambda row: (
            row["gt_faces"],
            row["detections"],
            -row["misses"],
        ),
    )

    localization = [
        row
        for row in candidates
        if (
            row["detections"] > 0
            and row["mean_iou"] > 0.0
        )
    ]

    choose(
        "localization_challenge",
        localization,
        key=lambda row: (
            row["mean_iou"],
            -row["detections"],
        ),
        reverse=False,
    )

    fallback_order = sorted(
        candidates,
        key=lambda row: (
            row["switches"],
            row["detections"],
            row["gt_faces"],
            -row["misses"],
            -row["false_positives"],
        ),
        reverse=True,
    )

    while (
        len(selected)
        < REPRESENTATIVE_FRAME_COUNT
    ):
        added = False

        for row in fallback_order:
            if (
                row["frame_no"]
                in used
            ):
                continue

            used.add(
                row[
                    "frame_no"
                ]
            )

            selected.append(
                {
                    "role": (
                        "additional_tracking_evidence"
                    ),
                    **row,
                }
            )

            added = True
            break

        if not added:
            break

    return selected[
        :REPRESENTATIVE_FRAME_COUNT
    ]



def write_events_csv(
    events,
    selection,
):
    """Write exact MOTMetrics events for the selected clip."""
    clip_events = events[
        (
            events[
                "source_frame"
            ]
            >= selection[
                "start_frame"
            ]
        )
        &
        (
            events[
                "source_frame"
            ]
            <= selection[
                "end_frame"
            ]
        )
    ].copy()

    columns = [
        "source_frame",
        "Type",
        "OId",
        "HId",
        "PreviousHId",
        "D",
    ]

    clip_events[
        columns
    ].to_csv(
        EVENTS_CSV,
        index=False,
        encoding="utf-8",
    )


def write_selection_csv(
    selection,
    representative_rows,
    accepted_row,
    fps,
):
    """Write a machine-readable record of clip and frame selection."""
    with open(
        SELECTION_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "sequence",
                "source_video",
                "clip_start_frame",
                "clip_end_frame",
                "clip_start_seconds",
                "clip_end_seconds",
                "clip_duration_seconds",
                "clip_exact_switches",
                "clip_detected_observations",
                "clip_misses",
                "clip_false_positives",
                "selection_rule",
                "sequence_recall_percent",
                "sequence_precision_percent",
                "sequence_f1_percent",
                "sequence_faf",
                "sequence_ids",
                "sequence_fragmentations",
                "sequence_mota_percent",
                "sequence_motp_percent",
                "sequence_idf1_percent",
                "annotated_video",
            ]
        )

        writer.writerow(
            [
                SEQUENCE_NAME,
                VIDEO_NAME,
                selection[
                    "start_frame"
                ],
                selection[
                    "end_frame"
                ],
                (
                    f"{(selection['start_frame'] - 1) / fps:.3f}"
                ),
                (
                    f"{selection['end_frame'] / fps:.3f}"
                ),
                (
                    f"{selection['window_frames'] / fps:.3f}"
                ),
                selection[
                    "switches"
                ],
                selection[
                    "detections"
                ],
                selection[
                    "misses"
                ],
                selection[
                    "false_positives"
                ],
                selection[
                    "selection_rule"
                ],
                (
                    f"{float(accepted_row['Recall_percent']):.4f}"
                ),
                (
                    f"{float(accepted_row['Precision_percent']):.4f}"
                ),
                (
                    f"{float(accepted_row['F1_percent']):.4f}"
                ),
                (
                    f"{float(accepted_row['FAF']):.6f}"
                ),
                int(
                    accepted_row[
                        "IDS"
                    ]
                ),
                int(
                    accepted_row[
                        "Frag"
                    ]
                ),
                (
                    f"{float(accepted_row['MOTA_percent']):.4f}"
                ),
                (
                    f"{float(accepted_row['MOTP_IoU_percent']):.4f}"
                ),
                (
                    f"{float(accepted_row['IDF1_percent']):.4f}"
                ),
                str(
                    OUTPUT_VIDEO.relative_to(
                        VALIDATION_DIR
                    )
                ).replace(
                    "\\",
                    "/",
                ),
            ]
        )

        writer.writerow(
            []
        )

        writer.writerow(
            [
                "representative_role",
                "frame_no",
                "gt_faces",
                "matches",
                "switches",
                "misses",
                "false_positives",
                "detections",
                "mean_iou",
                "representative_frame",
            ]
        )

        for row in representative_rows:
            writer.writerow(
                [
                    row[
                        "role"
                    ],
                    row[
                        "frame_no"
                    ],
                    row[
                        "gt_faces"
                    ],
                    row[
                        "matches"
                    ],
                    row[
                        "switches"
                    ],
                    row[
                        "misses"
                    ],
                    row[
                        "false_positives"
                    ],
                    row[
                        "detections"
                    ],
                    (
                        f"{row['mean_iou']:.6f}"
                    ),
                    row[
                        "output_path"
                    ],
                ]
            )


def create_summary_figure(
    representative_rows,
):
    """Create a clean 2x3 qualitative overview for README/thesis use."""
    tiles = []

    target_width = 960
    caption_height = 76

    for row in representative_rows:
        image_path = (
            VALIDATION_DIR
            / row[
                "output_path"
            ]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                "Could not read representative frame:\n"
                f"{image_path}"
            )

        scale = (
            target_width
            / image.shape[1]
        )

        target_height = int(
            round(
                image.shape[0]
                * scale
            )
        )

        resized = cv2.resize(
            image,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tile = np.full(
            (
                caption_height
                + target_height,
                target_width,
                3,
            ),
            (
                250,
                250,
                250,
            ),
            dtype=np.uint8,
        )

        tile[
            caption_height:,
            :,
        ] = resized

        title = (
            f"{row['role']} | "
            f"frame {row['frame_no']}"
        )

        details = (
            f"MATCH {row['matches']}  "
            f"SWITCH {row['switches']}  "
            f"MISS {row['misses']}  "
            f"FP {row['false_positives']}  "
            f"IoU {row['mean_iou']:.3f}"
        )

        cv2.putText(
            tile,
            title,
            (
                18,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (
                25,
                25,
                25,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            tile,
            details,
            (
                18,
                59,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                70,
                70,
                70,
            ),
            1,
            cv2.LINE_AA,
        )

        tiles.append(
            tile
        )

    if len(tiles) != 6:
        raise RuntimeError(
            "Expected exactly six representative frames."
        )

    row_1 = np.hstack(
        tiles[0:2]
    )

    row_2 = np.hstack(
        tiles[2:4]
    )

    row_3 = np.hstack(
        tiles[4:6]
    )

    figure = np.vstack(
        (
            row_1,
            row_2,
            row_3,
        )
    )

    if not cv2.imwrite(
        str(SUMMARY_FIGURE),
        figure,
    ):
        raise RuntimeError(
            "Could not save qualitative summary figure."
        )


def render_selected_clip(
    video_path,
    gt_by_frame,
    analysis,
    accepted_row,
    representative_targets,
):
    """
    Re-run T-ara from frame one to preserve OC-SORT temporal state,
    then render only the selected 60-second window.

    Pass-2 track identities are checked against the exact pass-1
    MOTMetrics event record before a frame is written.
    """
    detector, tracker = create_models()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not reopen T-ara for rendering."
        )

    fps = analysis[
        "fps"
    ]

    selection = analysis[
        "selection"
    ]

    start_frame = selection[
        "start_frame"
    ]

    end_frame = selection[
        "end_frame"
    ]

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
        ),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Could not create the Full-HD qualitative MP4."
        )

    target_by_frame = {
        int(row["frame_no"]): row
        for row in representative_targets
    }

    saved_representatives = []

    clip_totals = {
        "MATCH": 0,
        "SWITCH": 0,
        "MISS": 0,
        "FP": 0,
    }

    frame_no = 0

    print(
        "\nPass 2/2: rendering the verified 60-second tracking clip"
    )

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_no += 1

        detections = detector.predict(
            frame
        )

        tracks = tracker.track(
            frame,
            detections,
        )

        gt_entries = gt_by_frame.get(
            frame_no,
            [],
        )

        track_entries = extract_tracks(
            tracks
        )

        if frame_no < start_frame:
            if (
                frame_no
                % 500
                == 0
            ):
                print(
                    "Prepared tracker state through "
                    f"frame {frame_no}"
                )

            continue

        if frame_no > end_frame:
            break

        frame_events = event_rows_for_frame(
            analysis[
                "events"
            ],
            frame_no,
        )

        for event_type in clip_totals:
            clip_totals[
                event_type
            ] += int(
                (
                    frame_events[
                        "Type"
                    ]
                    == event_type
                ).sum()
            )

        rendered = render_tracking_frame(
            frame=frame,
            frame_no=frame_no,
            gt_entries=gt_entries,
            track_entries=track_entries,
            frame_events=frame_events,
            clip_totals=clip_totals,
            accepted_row=accepted_row,
            selection=selection,
            fps=fps,
        )

        writer.write(
            rendered
        )

        if frame_no in target_by_frame:
            target = dict(
                target_by_frame[
                    frame_no
                ]
            )

            output_path = (
                FRAMES_DIR
                / (
                    f"T-ara_"
                    f"{target['role']}_"
                    f"frame_{frame_no}.png"
                )
            )

            if not cv2.imwrite(
                str(output_path),
                rendered,
            ):
                writer.release()
                cap.release()

                raise RuntimeError(
                    "Could not save representative frame."
                )

            target[
                "output_path"
            ] = str(
                output_path.relative_to(
                    VALIDATION_DIR
                )
            ).replace(
                "\\",
                "/",
            )

            saved_representatives.append(
                target
            )

        rendered_count = (
            frame_no
            - start_frame
            + 1
        )

        if (
            rendered_count
            % 300
            == 0
        ):
            print(
                f"Rendered "
                f"{rendered_count}/"
                f"{selection['window_frames']} clip frames"
            )

    cap.release()
    writer.release()

    expected_clip_totals = {
        "MATCH": sum(
            row["matches"]
            for row in analysis[
                "frame_stats"
            ]
            if (
                start_frame
                <= row[
                    "frame_no"
                ]
                <= end_frame
            )
        ),
        "SWITCH": selection[
            "switches"
        ],
        "MISS": selection[
            "misses"
        ],
        "FP": selection[
            "false_positives"
        ],
    }

    if (
        clip_totals
        != expected_clip_totals
    ):
        raise RuntimeError(
            "Rendered clip event totals do not match the "
            "verified pass-1 MOTMetrics event record."
        )

    if (
        len(
            saved_representatives
        )
        != REPRESENTATIVE_FRAME_COUNT
    ):
        raise RuntimeError(
            "Not all representative frames were saved."
        )

    role_order = {
        row[
            "role"
        ]: index
        for index, row
        in enumerate(
            representative_targets
        )
    }

    saved_representatives.sort(
        key=lambda row: role_order[
            row[
                "role"
            ]
        ]
    )

    return (
        clip_totals,
        saved_representatives,
    )



def print_preflight(
    gt_end_frame,
    accepted_row,
):
    """Print the exact configuration before expensive processing."""
    print(
        "ECCV 2016 T-ara Face Tracking Qualitative Benchmark"
    )

    print(
        "=================================================="
    )

    print(
        "\nDataset root:"
    )

    print(
        DATASET_ROOT
    )

    print(
        "\nSequence:"
    )

    print(
        SEQUENCE_NAME
    )

    print(
        f"Ground-truth end frame: "
        f"{gt_end_frame}"
    )

    print(
        f"Target qualitative duration: "
        f"{TARGET_CLIP_SECONDS:.1f} seconds"
    )

    print(
        "\nTracking configuration:"
    )

    print(
        f"Detector confidence: "
        f"{DETECTOR_CONFIDENCE}"
    )

    print(
        f"Detector IoU: "
        f"{DETECTOR_IOU}"
    )

    print(
        "Tracker: OC-SORT"
    )

    print(
        f"Device: "
        f"{DEVICE}"
    )

    print(
        f"GT/prediction matching IoU: "
        f"{MATCHING_IOU}"
    )

    print(
        "Frame event source: exact MOTMetrics accumulator events"
    )

    print(
        "\nAccepted T-ara metrics:"
    )

    metric_lines = [
        (
            "Recall",
            f"{float(accepted_row['Recall_percent']):.2f}%",
        ),
        (
            "Precision",
            f"{float(accepted_row['Precision_percent']):.2f}%",
        ),
        (
            "F1",
            f"{float(accepted_row['F1_percent']):.2f}%",
        ),
        (
            "FAF",
            f"{float(accepted_row['FAF']):.4f}",
        ),
        (
            "IDS",
            str(int(accepted_row["IDS"])),
        ),
        (
            "Fragmentations",
            str(int(accepted_row["Frag"])),
        ),
        (
            "MOTA",
            f"{float(accepted_row['MOTA_percent']):.2f}%",
        ),
        (
            "MOTP",
            f"{float(accepted_row['MOTP_IoU_percent']):.2f}%",
        ),
        (
            "IDF1",
            f"{float(accepted_row['IDF1_percent']):.2f}%",
        ),
    ]

    for label, value in metric_lines:
        print(
            f"{label}: {value}"
        )

    print(
        "\nThe script will first reproduce T-ara quantitatively."
    )

    print(
        "Qualitative output is generated only if that reproduction "
        "matches the accepted result."
    )

    print(
        "Only previous qualitative outputs will be cleaned."
    )

    print(
        "Accepted quantitative artifacts will not be modified."
    )


def main():
    """Generate final, verified, tracking-specific T-ara qualitative evidence."""
    video_path, gt_path = (
        verify_inputs()
    )

    accepted_row = (
        load_quantitative_row()
    )

    (
        gt_by_frame,
        gt_end_frame,
    ) = load_ground_truth(
        gt_path
    )

    print_preflight(
        gt_end_frame,
        accepted_row,
    )

    # Verify the scientific run before replacing qualitative outputs.
    analysis = (
        run_full_sequence_analysis(
            video_path,
            gt_by_frame,
            gt_end_frame,
            accepted_row,
        )
    )

    representative_targets = (
        choose_representative_frames(
            analysis[
                "frame_stats"
            ],
            analysis[
                "selection"
            ],
        )
    )

    # Clean only after the accepted result has been reproduced.
    clean_previous_qualitative_outputs()

    write_events_csv(
        analysis[
            "events"
        ],
        analysis[
            "selection"
        ],
    )

    (
        clip_totals,
        representative_rows,
    ) = render_selected_clip(
        video_path,
        gt_by_frame,
        analysis,
        accepted_row,
        representative_targets,
    )

    write_selection_csv(
        analysis[
            "selection"
        ],
        representative_rows,
        accepted_row,
        analysis[
            "fps"
        ],
    )

    create_summary_figure(
        representative_rows
    )

    print(
        "\n=================================================="
    )

    print(
        "Qualitative tracking generation completed successfully."
    )

    print(
        "\nScientific verification:"
    )

    print(
        "T-ara was re-evaluated with the same MOTMetrics protocol "
        "and reproduced the accepted quantitative result."
    )

    print(
        "\nSelected clip:"
    )

    print(
        f"Frames "
        f"{analysis['selection']['start_frame']}-"
        f"{analysis['selection']['end_frame']}"
    )

    print(
        f"Duration: "
        f"{analysis['selection']['window_frames'] / analysis['fps']:.2f} s"
    )

    print(
        "\nExact MOTMetrics events inside rendered clip:"
    )

    for event_type in [
        "MATCH",
        "SWITCH",
        "MISS",
        "FP",
    ]:
        print(
            f"{event_type}: "
            f"{clip_totals[event_type]}"
        )

    print(
        "\nAnnotated video:"
    )

    print(
        OUTPUT_VIDEO
    )

    print(
        "\nRepresentative frames:"
    )

    for row in representative_rows:
        print(
            f"- {row['role']}: "
            f"{VALIDATION_DIR / row['output_path']}"
        )

    print(
        "\nExact clip event CSV:"
    )

    print(
        EVENTS_CSV
    )

    print(
        "\nSelection record:"
    )

    print(
        SELECTION_CSV
    )

    print(
        "\nSummary figure:"
    )

    print(
        SUMMARY_FIGURE
    )

    print(
        "\nAccepted quantitative artifacts were not modified."
    )


if __name__ == "__main__":
    main()
