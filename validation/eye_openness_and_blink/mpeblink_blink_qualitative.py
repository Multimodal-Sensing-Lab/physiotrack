import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from physiotrack.face.eyes import EyeOpenness
from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.models import Models

import mpeblink_blink_eval as benchmark


SCRIPT_DIR = Path(
    __file__
).resolve().parent

REPOSITORY_ROOT = (
    SCRIPT_DIR.parents[
        1
    ]
)

WORKSPACE_ROOT = (
    REPOSITORY_ROOT.parent
)

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

SEQUENCE_RESULTS_PATH = (
    RESULTS_DIR
    / "mpeblink_test_sequence_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "mpeblink_test_summary.txt"
)

QUALITATIVE_DIR = (
    RESULTS_DIR
    / "qualitative"
)

VIDEO_DIR = (
    QUALITATIVE_DIR
    / "annotated_videos"
)

IMAGE_DIR = (
    QUALITATIVE_DIR
    / "annotated_images"
)

SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "mpeblink_qualitative_selection.csv"
)

COMBINED_IMAGE_PATH = (
    RESULTS_DIR
    / "figures"
    / "mpeblink_qualitative_examples.png"
)

BLINK_THRESHOLD = (
    benchmark.SELECTED_THRESHOLD
)

MIN_CLOSED_FRAMES = (
    benchmark.SELECTED_MIN_CLOSED_FRAMES
)

EVENT_IOU_THRESHOLD = (
    benchmark.EVENT_IOU_THRESHOLD
)

PANEL_WIDTH = 520
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
WINDOW_SECONDS = 4.0
CLIP_CONTEXT_SECONDS = 3.0

QUALITATIVE_ROLES = [
    "strong_detection",
    "representative",
    "challenging_false_positive",
    "challenging_false_negative",
    "mixed_detection",
    "accurate_count",
    "high_blink_activity",
    "low_blink_activity",
]


def sequence_f1(row):
    """Calculate event F1 for one accepted person sequence."""
    true_positive = int(
        row[
            "true_positive"
        ]
    )

    false_positive = int(
        row[
            "false_positive"
        ]
    )

    false_negative = int(
        row[
            "false_negative"
        ]
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if denominator == 0:
        return 0.0

    return (
        2.0
        * true_positive
        / denominator
    )


def select_one(
    table,
    role,
    used_sequences,
    used_videos,
):
    """Select one deterministic accepted sequence for one qualitative role."""
    candidates = table.copy()

    candidates[
        "sequence_f1"
    ] = candidates.apply(
        sequence_f1,
        axis=1,
    )

    if role == "strong_detection":
        candidates = candidates[
            (
                candidates[
                    "true_positive"
                ] >= 1
            )
            & (
                candidates[
                    "false_positive"
                ] == 0
            )
            & (
                candidates[
                    "false_negative"
                ] == 0
            )
        ]

        candidates = candidates.sort_values(
            [
                "true_positive",
                "gt_blinks",
                "video_id",
                "person_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

    elif role == "representative":
        candidates = candidates[
            candidates[
                "gt_blinks"
            ] > 0
        ].copy()

        candidates[
            "distance_to_global_f1"
        ] = np.abs(
            candidates[
                "sequence_f1"
            ]
            - 0.269191
        )

        candidates = candidates.sort_values(
            [
                "distance_to_global_f1",
                "gt_blinks",
                "video_id",
                "person_id",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )

    elif role == "challenging_false_positive":
        candidates = candidates[
            candidates[
                "false_positive"
            ] > 0
        ]

        candidates = candidates.sort_values(
            [
                "false_positive",
                "predicted_blinks",
                "video_id",
                "person_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

    elif role == "challenging_false_negative":
        candidates = candidates[
            candidates[
                "false_negative"
            ] > 0
        ]

        candidates = candidates.sort_values(
            [
                "false_negative",
                "gt_blinks",
                "video_id",
                "person_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

    elif role == "mixed_detection":
        candidates = candidates[
            (
                candidates[
                    "true_positive"
                ] > 0
            )
            & (
                candidates[
                    "false_positive"
                ] > 0
            )
            & (
                candidates[
                    "false_negative"
                ] > 0
            )
        ].copy()

        candidates[
            "mixed_total"
        ] = (
            candidates[
                "true_positive"
            ]
            + candidates[
                "false_positive"
            ]
            + candidates[
                "false_negative"
            ]
        )

        candidates = candidates.sort_values(
            [
                "mixed_total",
                "true_positive",
                "video_id",
                "person_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

    elif role == "accurate_count":
        candidates = candidates[
            (
                candidates[
                    "gt_blinks"
                ] >= 2
            )
            & (
                candidates[
                    "true_positive"
                ] > 0
            )
        ]

        candidates = candidates.sort_values(
            [
                "absolute_count_error",
                "false_positive",
                "false_negative",
                "gt_blinks",
                "video_id",
                "person_id",
            ],
            ascending=[
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        )

    elif role == "high_blink_activity":
        candidates = candidates[
            candidates[
                "gt_blinks"
            ] > 0
        ]

        candidates = candidates.sort_values(
            [
                "gt_blinks",
                "true_positive",
                "video_id",
                "person_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )

    elif role == "low_blink_activity":
        candidates = candidates[
            (
                candidates[
                    "gt_blinks"
                ] >= 1
            )
            & (
                candidates[
                    "gt_blinks"
                ] <= 2
            )
        ]

        candidates = candidates.sort_values(
            [
                "gt_blinks",
                "absolute_count_error",
                "video_id",
                "person_id",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )

    else:
        raise RuntimeError(
            f"Unknown qualitative role: {role}"
        )

    if candidates.empty:
        raise RuntimeError(
            f"No candidate sequence found for role: {role}"
        )

    preferred = candidates[
        ~candidates[
            "video_id"
        ].astype(
            int
        ).isin(
            used_videos
        )
    ]

    if not preferred.empty:
        candidates = preferred

    for _, row in candidates.iterrows():
        key = (
            int(
                row[
                    "video_id"
                ]
            ),
            str(
                row[
                    "person_id"
                ]
            ),
        )

        if key in used_sequences:
            continue

        return row

    raise RuntimeError(
        f"No unused sequence found for role: {role}"
    )


def select_cases(table):
    """Select eight deterministic qualitative benchmark sequences."""
    selected = []

    used_sequences = set()
    used_videos = set()

    for role in QUALITATIVE_ROLES:
        row = select_one(
            table,
            role,
            used_sequences,
            used_videos,
        )

        video_id = int(
            row[
                "video_id"
            ]
        )

        person_id = str(
            row[
                "person_id"
            ]
        )

        used_sequences.add(
            (
                video_id,
                person_id,
            )
        )

        used_videos.add(
            video_id
        )

        selected.append(
            (
                role,
                row,
            )
        )

    return selected


def load_annotation(video_id):
    """Load one MPEBlink test annotation and video path."""
    video_dir = (
        DATASET_ROOT
        / "test"
        / str(
            video_id
        )
    )

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

    return (
        video_path,
        annotation,
    )


def process_sequence(
    video_id,
    person_id,
    landmarks_model,
    eye_model,
):
    """Re-run one accepted sequence using the quantitative benchmark protocol."""
    (
        video_path,
        annotation,
    ) = load_annotation(
        video_id
    )

    expected_frames = int(
        annotation[
            "length"
        ]
    )

    person_annotation = annotation[
        person_id
    ]

    openness = np.full(
        expected_frames,
        np.nan,
        dtype=np.float32,
    )

    boxes = [
        None
        for _ in range(
            expected_frames
        )
    ]

    landmark_points = [
        None
        for _ in range(
            expected_frames
        )
    ]

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_index = 0

    while frame_index < expected_frames:
        success, frame = capture.read()

        if not success:
            break

        bbox = person_annotation[
            "bbox"
        ][
            frame_index
        ]

        if (
            bbox is not None
            and isinstance(
                bbox,
                (list, tuple),
            )
            and len(
                bbox
            ) == 4
        ):
            x, y, width, height = bbox

            image_height, image_width = (
                frame.shape[:2]
            )

            if (
                width > 0
                and height > 0
            ):
                x1 = max(
                    0.0,
                    float(
                        x
                    ),
                )

                y1 = max(
                    0.0,
                    float(
                        y
                    ),
                )

                x2 = min(
                    float(
                        image_width
                    ),
                    float(
                        x + width
                    ),
                )

                y2 = min(
                    float(
                        image_height
                    ),
                    float(
                        y + height
                    ),
                )

                if (
                    x2 > x1
                    and y2 > y1
                ):
                    box = np.asarray(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        dtype=float,
                    )

                    boxes[
                        frame_index
                    ] = box

                    landmarks = (
                        landmarks_model.predict_face(
                            frame,
                            box,
                        )
                    )

                    if landmarks is not None:
                        landmark_points[
                            frame_index
                        ] = np.asarray(
                            landmarks
                        ).copy()

                        eye_result = (
                            eye_model.predict(
                                landmarks
                            )
                        )

                        value = (
                            eye_result.get(
                                "mean_openness"
                            )
                        )

                        if (
                            value is not None
                            and np.isfinite(
                                value
                            )
                        ):
                            openness[
                                frame_index
                            ] = float(
                                value
                            )

        frame_index += 1

    capture.release()

    if frame_index != expected_frames:
        raise RuntimeError(
            f"Video frame mismatch for test/{video_id}: "
            f"read {frame_index}, expected {expected_frames}"
        )

    ground_truth = [
        (
            int(
                event[
                    0
                ]
            ),
            int(
                event[
                    1
                ]
            ),
        )
        for event in person_annotation[
            "blink"
        ]
    ]

    predicted = benchmark.build_events(
        openness,
        BLINK_THRESHOLD,
        MIN_CLOSED_FRAMES,
    )

    (
        true_positive,
        false_positive,
        false_negative,
        matches,
    ) = benchmark.match_events(
        predicted,
        ground_truth,
        EVENT_IOU_THRESHOLD,
    )

    return {
        "video_path": video_path,
        "fps": fps,
        "frames": expected_frames,
        "openness": openness,
        "boxes": boxes,
        "landmarks": landmark_points,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "matches": matches,
    }


def verify_against_accepted(
    role,
    accepted_row,
    result,
):
    """Verify selected-sequence results against the accepted quantitative CSV."""
    checks = {
        "gt_blinks": len(
            result[
                "ground_truth"
            ]
        ),
        "predicted_blinks": len(
            result[
                "predicted"
            ]
        ),
        "true_positive": int(
            result[
                "true_positive"
            ]
        ),
        "false_positive": int(
            result[
                "false_positive"
            ]
        ),
        "false_negative": int(
            result[
                "false_negative"
            ]
        ),
    }

    for column, verified_value in checks.items():
        accepted_value = int(
            accepted_row[
                column
            ]
        )

        if accepted_value != verified_value:
            raise RuntimeError(
                f"Qualitative verification failed for {role}: "
                f"{column} accepted={accepted_value}, "
                f"verified={verified_value}"
            )


def event_membership(frame_index, events):
    """Return whether one frame lies inside any event interval."""
    for start, end in events:
        if (
            frame_index >= start
            and frame_index <= end
        ):
            return True

    return False


def matched_event_sets(result):
    """Return matched prediction and ground-truth event indices."""
    matched_predictions = {
        int(
            item[
                0
            ]
        )
        for item in result[
            "matches"
        ]
    }

    matched_ground_truth = {
        int(
            item[
                1
            ]
        )
        for item in result[
            "matches"
        ]
    }

    return (
        matched_predictions,
        matched_ground_truth,
    )


def diagnostic_event(role, result):
    """Choose one event that best explains the qualitative role."""
    predicted = result[
        "predicted"
    ]

    ground_truth = result[
        "ground_truth"
    ]

    (
        matched_predictions,
        matched_ground_truth,
    ) = matched_event_sets(
        result
    )

    if role == "challenging_false_positive":
        for index, event in enumerate(
            predicted
        ):
            if index not in matched_predictions:
                return (
                    "FP",
                    event,
                )

    if role == "challenging_false_negative":
        for index, event in enumerate(
            ground_truth
        ):
            if index not in matched_ground_truth:
                return (
                    "FN",
                    event,
                )

    if result[
        "matches"
    ]:
        best_match = max(
            result[
                "matches"
            ],
            key=lambda item: float(
                item[
                    2
                ]
            ),
        )

        return (
            "TP",
            predicted[
                int(
                    best_match[
                        0
                    ]
                )
            ],
        )

    if ground_truth:
        return (
            "GT",
            ground_truth[
                0
            ],
        )

    if predicted:
        return (
            "Pred",
            predicted[
                0
            ],
        )

    return (
        "None",
        (
            0,
            max(
                0,
                result[
                    "frames"
                ]
                - 1,
            ),
        ),
    )


def clean_owned_outputs():
    """Remove only qualitative outputs owned by this generator."""
    if QUALITATIVE_DIR.is_dir():
        shutil.rmtree(
            QUALITATIVE_DIR
        )

    if COMBINED_IMAGE_PATH.is_file():
        COMBINED_IMAGE_PATH.unlink()

    VIDEO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMBINED_IMAGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def fit_frame(frame, width, height):
    """Resize a video frame to fit a fixed display area."""
    image_height, image_width = (
        frame.shape[:2]
    )

    scale = min(
        width / image_width,
        height / image_height,
    )

    target_width = max(
        1,
        int(
            round(
                image_width
                * scale
            )
        ),
    )

    target_height = max(
        1,
        int(
            round(
                image_height
                * scale
            )
        ),
    )

    resized = cv2.resize(
        frame,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        width
        - target_width
    ) // 2

    offset_y = (
        height
        - target_height
    ) // 2

    canvas[
        offset_y:offset_y + target_height,
        offset_x:offset_x + target_width,
    ] = resized

    return (
        canvas,
        scale,
        offset_x,
        offset_y,
    )


def project_box(
    box,
    scale,
    offset_x,
    offset_y,
):
    """Project an original-frame bounding box into the display frame."""
    if box is None:
        return None

    x1, y1, x2, y2 = [
        int(
            round(
                float(
                    value
                )
                * scale
            )
        )
        for value in box
    ]

    return (
        x1 + offset_x,
        y1 + offset_y,
        x2 + offset_x,
        y2 + offset_y,
    )


def draw_landmarks(
    canvas,
    landmarks,
    scale,
    offset_x,
    offset_y,
):
    """Draw available facial landmark points as supporting visual evidence."""
    if landmarks is None:
        return

    points = np.asarray(
        landmarks
    )

    if (
        points.ndim != 2
        or points.shape[
            1
        ] < 2
    ):
        return

    for point in points:
        x = float(
            point[
                0
            ]
        )

        y = float(
            point[
                1
            ]
        )

        if not (
            np.isfinite(
                x
            )
            and np.isfinite(
                y
            )
        ):
            continue

        draw_x = int(
            round(
                x
                * scale
            )
        ) + offset_x

        draw_y = int(
            round(
                y
                * scale
            )
        ) + offset_y

        cv2.circle(
            canvas,
            (
                draw_x,
                draw_y,
            ),
            2,
            (
                255,
                220,
                80,
            ),
            -1,
            cv2.LINE_AA,
        )


def text_line(
    panel,
    text,
    y,
    scale=0.62,
    thickness=1,
):
    """Draw one panel text line."""
    cv2.putText(
        panel,
        text,
        (
            24,
            y,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (
            238,
            238,
            238,
        ),
        thickness,
        cv2.LINE_AA,
    )


def draw_signal_graph(
    panel,
    result,
    frame_index,
    x1,
    y1,
    x2,
    y2,
    diagnostic_event=None,
    event_type=None,
):
    """Draw a rolling EyeOpenness signal with GT and predicted blink state."""
    cv2.rectangle(
        panel,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            70,
            70,
            70,
        ),
        1,
    )

    fps = max(
        1.0,
        float(
            result[
                "fps"
            ]
        ),
    )

    radius = max(
        10,
        int(
            round(
                WINDOW_SECONDS
                * fps
            )
        ),
    )

    start = max(
        0,
        frame_index - radius,
    )

    end = min(
        result[
            "frames"
        ]
        - 1,
        frame_index + radius,
    )

    values = result[
        "openness"
    ][
        start:end + 1
    ]

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size:
        lower = min(
            0.0,
            float(
                np.min(
                    finite
                )
            )
            - 0.05,
        )

        upper = max(
            1.0,
            float(
                np.max(
                    finite
                )
            )
            + 0.05,
        )

    else:
        lower = 0.0
        upper = 1.0

    width = max(
        1,
        x2 - x1
    )

    height = max(
        1,
        y2 - y1
    )

    def map_x(index):
        if end == start:
            return x1

        fraction = (
            index - start
        ) / (
            end - start
        )

        return int(
            round(
                x1
                + fraction
                * width
            )
        )

    def map_y(value):
        fraction = (
            value - lower
        ) / (
            upper - lower
        )

        fraction = min(
            1.0,
            max(
                0.0,
                fraction,
            ),
        )

        return int(
            round(
                y2
                - fraction
                * height
            )
        )

    for gt_start, gt_end in result[
        "ground_truth"
    ]:
        overlap_start = max(
            start,
            gt_start,
        )

        overlap_end = min(
            end,
            gt_end,
        )

        if overlap_end >= overlap_start:
            cv2.rectangle(
                panel,
                (
                    map_x(
                        overlap_start
                    ),
                    y1,
                ),
                (
                    map_x(
                        overlap_end
                    ),
                    y2,
                ),
                (
                    70,
                    45,
                    45,
                ),
                -1,
            )

    for pred_start, pred_end in result[
        "predicted"
    ]:
        overlap_start = max(
            start,
            pred_start,
        )

        overlap_end = min(
            end,
            pred_end,
        )

        if overlap_end >= overlap_start:
            cv2.rectangle(
                panel,
                (
                    map_x(
                        overlap_start
                    ),
                    y1,
                ),
                (
                    map_x(
                        overlap_end
                    ),
                    y1 + 9,
                ),
                (
                    70,
                    130,
                    70,
                ),
                -1,
            )

    threshold_y = map_y(
        BLINK_THRESHOLD
    )

    cv2.line(
        panel,
        (
            x1,
            threshold_y,
        ),
        (
            x2,
            threshold_y,
        ),
        (
            0,
            190,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    previous = None

    for index in range(
        start,
        end + 1,
    ):
        value = result[
            "openness"
        ][
            index
        ]

        if not np.isfinite(
            value
        ):
            previous = None
            continue

        point = (
            map_x(
                index
            ),
            map_y(
                float(
                    value
                )
            ),
        )

        if previous is not None:
            cv2.line(
                panel,
                previous,
                point,
                (
                    230,
                    230,
                    230,
                ),
                2,
                cv2.LINE_AA,
            )

        previous = point

    if (
        diagnostic_event is not None
        and event_type is not None
    ):
        diagnostic_start = max(
            start,
            int(
                diagnostic_event[
                    0
                ]
            ),
        )

        diagnostic_end = min(
            end,
            int(
                diagnostic_event[
                    1
                ]
            ),
        )

        if diagnostic_end >= diagnostic_start:
            if event_type == "TP":
                outline_color = (
                    70,
                    220,
                    70,
                )
            elif event_type == "FP":
                outline_color = (
                    60,
                    150,
                    255,
                )
            elif event_type == "FN":
                outline_color = (
                    80,
                    80,
                    255,
                )
            else:
                outline_color = (
                    180,
                    180,
                    180,
                )

            cv2.rectangle(
                panel,
                (
                    map_x(
                        diagnostic_start
                    ),
                    y1 + 1,
                ),
                (
                    map_x(
                        diagnostic_end
                    ),
                    y2 - 1,
                ),
                outline_color,
                2,
            )

    current_x = map_x(
        frame_index
    )

    cv2.line(
        panel,
        (
            current_x,
            y1,
        ),
        (
            current_x,
            y2,
        ),
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "Rolling EyeOpenness signal",
        (
            x1,
            y1 - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )


def diagnostic_status(
    frame_index,
    event_type,
    diagnostic_event,
):
    """Return a clear frame-level diagnostic label for the selected event."""
    start = int(
        diagnostic_event[
            0
        ]
    )

    end = int(
        diagnostic_event[
            1
        ]
    )

    active = (
        frame_index >= start
        and frame_index <= end
    )

    if event_type == "TP":
        label = "TRUE POSITIVE"
    elif event_type == "FP":
        label = "FALSE POSITIVE"
    elif event_type == "FN":
        label = "FALSE NEGATIVE"
    elif event_type == "GT":
        label = "GROUND-TRUTH EVENT"
    elif event_type == "Pred":
        label = "PREDICTED EVENT"
    else:
        label = "NO EVENT"

    return (
        label,
        active,
    )


def render_frame(
    frame,
    result,
    accepted_row,
    role,
    frame_index,
    event_type,
    diagnostic_event,
):
    """Render one professional benchmark frame with a live side panel."""
    video_width = (
        DISPLAY_WIDTH
        - PANEL_WIDTH
    )

    (
        display,
        scale,
        offset_x,
        offset_y,
    ) = fit_frame(
        frame,
        video_width,
        DISPLAY_HEIGHT,
    )

    draw_landmarks(
        display,
        result[
            "landmarks"
        ][
            frame_index
        ],
        scale,
        offset_x,
        offset_y,
    )

    projected_box = project_box(
        result[
            "boxes"
        ][
            frame_index
        ],
        scale,
        offset_x,
        offset_y,
    )

    if projected_box is not None:
        cv2.rectangle(
            display,
            (
                projected_box[
                    0
                ],
                projected_box[
                    1
                ],
            ),
            (
                projected_box[
                    2
                ],
                projected_box[
                    3
                ],
            ),
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "GT target face (evaluated)",
            (
                projected_box[
                    0
                ],
                max(
                    24,
                    projected_box[
                        1
                    ]
                    - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                220,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    openness = result[
        "openness"
    ][
        frame_index
    ]

    if np.isfinite(
        openness
    ):
        eye_state = (
            "CLOSED"
            if float(
                openness
            )
            < BLINK_THRESHOLD
            else "OPEN"
        )

        openness_text = (
            f"{float(openness):.3f}"
        )

    else:
        eye_state = "UNAVAILABLE"
        openness_text = "N/A"

    gt_active = event_membership(
        frame_index,
        result[
            "ground_truth"
        ],
    )

    pred_active = event_membership(
        frame_index,
        result[
            "predicted"
        ],
    )

    (
        diagnostic_label,
        diagnostic_active,
    ) = diagnostic_status(
        frame_index,
        event_type,
        diagnostic_event,
    )

    panel = np.full(
        (
            DISPLAY_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        28,
        dtype=np.uint8,
    )

    text_line(
        panel,
        "PhysioTrack",
        42,
        scale=0.86,
        thickness=2,
    )

    text_line(
        panel,
        "Eye Openness + Blink",
        76,
        scale=0.70,
        thickness=2,
    )

    cv2.line(
        panel,
        (
            24,
            94,
        ),
        (
            PANEL_WIDTH - 24,
            94,
        ),
        (
            80,
            80,
            80,
        ),
        1,
    )

    video_id = int(
        accepted_row[
            "video_id"
        ]
    )

    person_id = str(
        accepted_row[
            "person_id"
        ]
    )

    text_line(
        panel,
        f"Benchmark: MPEBlink 2.0 / test",
        126,
    )

    text_line(
        panel,
        f"Case: {role}",
        154,
    )

    text_line(
        panel,
        f"Video {video_id} | {person_id}",
        182,
    )

    text_line(
        panel,
        (
            f"Frame {frame_index} / "
            f"{result['frames'] - 1}  |  "
            f"{frame_index / max(result['fps'], 1.0):.2f} s"
        ),
        222,
    )

    text_line(
        panel,
        f"Mean eye openness: {openness_text}",
        258,
        scale=0.68,
        thickness=2,
    )

    text_line(
        panel,
        (
            f"Threshold: {BLINK_THRESHOLD:.2f}  |  "
            f"State: {eye_state}"
        ),
        290,
        scale=0.62,
        thickness=2,
    )

    text_line(
        panel,
        (
            f"GT blink: {'YES' if gt_active else 'NO'}  |  "
            f"Predicted blink: {'YES' if pred_active else 'NO'}"
        ),
        326,
        scale=0.61,
        thickness=2,
    )

    badge_text = (
        f"Diagnostic event: {diagnostic_label}"
    )

    badge_color = (
        (70, 170, 70)
        if event_type == "TP"
        else (
            (60, 120, 220)
            if event_type == "FP"
            else (
                (80, 80, 220)
                if event_type == "FN"
                else (
                    130,
                    130,
                    130,
                )
            )
        )
    )

    cv2.rectangle(
        panel,
        (
            20,
            344,
        ),
        (
            PANEL_WIDTH - 20,
            382,
        ),
        badge_color,
        -1,
    )

    cv2.putText(
        panel,
        (
            badge_text
            + (
                "  |  ACTIVE"
                if diagnostic_active
                else "  |  CONTEXT"
            )
        ),
        (
            30,
            370,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    text_line(
        panel,
        "Evaluation target: annotated person sequence only",
        410,
        scale=0.50,
    )

    text_line(
        panel,
        (
            f"GT events: {int(accepted_row['gt_blinks'])}  |  "
            f"Pred: {int(accepted_row['predicted_blinks'])}"
        ),
        436,
    )

    text_line(
        panel,
        (
            f"TP {int(accepted_row['true_positive'])}  "
            f"FP {int(accepted_row['false_positive'])}  "
            f"FN {int(accepted_row['false_negative'])}"
        ),
        464,
    )

    landmark_status = (
        "AVAILABLE"
        if result[
            "landmarks"
        ][
            frame_index
        ] is not None
        else "UNAVAILABLE"
    )

    text_line(
        panel,
        (
            f"Face landmarks: {landmark_status}"
        ),
        492,
        scale=0.54,
    )

    draw_signal_graph(
        panel,
        result,
        frame_index,
        24,
        542,
        PANEL_WIDTH - 24,
        665,
        diagnostic_event=diagnostic_event,
        event_type=event_type,
    )

    text_line(
        panel,
        "GT intervals: shaded background",
        684,
        scale=0.43,
    )

    text_line(
        panel,
        "Predicted intervals: green strip",
        702,
        scale=0.43,
    )

    text_line(
        panel,
        "Selected diagnostic event: colored outline",
        718,
        scale=0.43,
    )

    canvas = np.concatenate(
        [
            display,
            panel,
        ],
        axis=1,
    )

    return canvas


def clip_bounds(event, fps, frame_count):
    """Return a compact qualitative clip centered on the diagnostic event."""
    context = max(
        1,
        int(
            round(
                CLIP_CONTEXT_SECONDS
                * max(
                    fps,
                    1.0,
                )
            )
        ),
    )

    start = max(
        0,
        int(
            event[
                0
            ]
        )
        - context,
    )

    end = min(
        frame_count - 1,
        int(
            event[
                1
            ]
        )
        + context,
    )

    return (
        start,
        end,
    )


def write_case_outputs(
    role,
    accepted_row,
    result,
):
    """Write one dynamic annotated MP4 and three supporting PNG frames."""
    (
        event_type,
        event,
    ) = diagnostic_event(
        role,
        result,
    )

    (
        clip_start,
        clip_end,
    ) = clip_bounds(
        event,
        result[
            "fps"
        ],
        result[
            "frames"
        ],
    )

    video_id = int(
        accepted_row[
            "video_id"
        ]
    )

    person_id = str(
        accepted_row[
            "person_id"
        ]
    )

    stem = (
        f"{role}_"
        f"video_{video_id}_"
        f"{person_id}"
    )

    video_output_path = (
        VIDEO_DIR
        / f"{stem}.mp4"
    )

    image_output_path = (
        IMAGE_DIR
        / f"{stem}.png"
    )

    capture = cv2.VideoCapture(
        str(
            result[
                "video_path"
            ]
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {result['video_path']}"
        )

    fps = float(
        result[
            "fps"
        ]
    )

    if (
        not np.isfinite(
            fps
        )
        or fps <= 0
    ):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            video_output_path
        ),
        fourcc,
        fps,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        ),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not create video: {video_output_path}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        clip_start,
    )

    center_frame = int(
        round(
            (
                max(
                    0,
                    int(
                        event[
                            0
                        ]
                    ),
                )
                + min(
                    result[
                        "frames"
                    ]
                    - 1,
                    int(
                        event[
                            1
                        ]
                    ),
                )
            )
            / 2.0
        )
    )

    image_saved = False

    for frame_index in range(
        clip_start,
        clip_end + 1,
    ):
        success, frame = capture.read()

        if not success:
            writer.release()
            capture.release()
            raise RuntimeError(
                f"Could not read frame {frame_index} "
                f"from {result['video_path']}"
            )

        rendered = render_frame(
            frame,
            result,
            accepted_row,
            role,
            frame_index,
            event_type,
            event,
        )

        writer.write(
            rendered
        )

        if (
            frame_index == center_frame
            and not image_saved
        ):
            cv2.imwrite(
                str(
                    image_output_path
                ),
                rendered,
            )

            image_saved = True

    writer.release()
    capture.release()

    if not image_saved:
        raise RuntimeError(
            f"Could not save representative frame for {role}"
        )

    return {
        "role": role,
        "video_id": video_id,
        "person_id": person_id,
        "event_type": event_type,
        "diagnostic_label": (
            "TRUE POSITIVE"
            if event_type == "TP"
            else (
                "FALSE POSITIVE"
                if event_type == "FP"
                else (
                    "FALSE NEGATIVE"
                    if event_type == "FN"
                    else event_type
                )
            )
        ),
        "event_start": int(
            event[
                0
            ]
        ),
        "event_end": int(
            event[
                1
            ]
        ),
        "clip_start": clip_start,
        "clip_end": clip_end,
        "video_output": str(
            video_output_path.relative_to(
                SCRIPT_DIR
            )
        ),
        "image_output": str(
            image_output_path.relative_to(
                SCRIPT_DIR
            )
        ),
    }


def save_combined_image(rows):
    """Create one compact contact sheet from the eight representative PNGs."""
    images = []

    for row in rows:
        image_path = (
            SCRIPT_DIR
            / row[
                "image_output"
            ]
        )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None:
            raise RuntimeError(
                f"Could not read generated image: {image_path}"
            )

        thumbnail = cv2.resize(
            image,
            (
                640,
                360,
            ),
            interpolation=cv2.INTER_AREA,
        )

        cv2.putText(
            thumbnail,
            row[
                "role"
            ],
            (
                18,
                32,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        images.append(
            thumbnail
        )

    rows_of_images = []

    for start in range(
        0,
        len(
            images
        ),
        2,
    ):
        rows_of_images.append(
            np.concatenate(
                images[
                    start:start + 2
                ],
                axis=1,
            )
        )

    combined = np.concatenate(
        rows_of_images,
        axis=0,
    )

    cv2.imwrite(
        str(
            COMBINED_IMAGE_PATH
        ),
        combined,
    )


def main():
    """Generate verified professional video and image benchmark evidence."""
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(
            f"MPEBlink 2.0 dataset not found: {DATASET_ROOT}"
        )

    if not SEQUENCE_RESULTS_PATH.is_file():
        raise RuntimeError(
            "Accepted quantitative sequence results are required: "
            f"{SEQUENCE_RESULTS_PATH}"
        )

    if not SUMMARY_PATH.is_file():
        raise RuntimeError(
            "Accepted quantitative summary is required: "
            f"{SUMMARY_PATH}"
        )

    table = pd.read_csv(
        SEQUENCE_RESULTS_PATH
    )

    required_columns = {
        "video_id",
        "person_id",
        "fps",
        "frames",
        "gt_blinks",
        "predicted_blinks",
        "true_positive",
        "false_positive",
        "false_negative",
        "absolute_count_error",
    }

    missing_columns = (
        required_columns
        - set(
            table.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Accepted sequence-results CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    selected = select_cases(
        table
    )

    model_path = Models.resolve(
        Models.Face.MediaPipe.Landmarks.face_landmarker
    )

    landmarks_model = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    eye_model = EyeOpenness()

    verified = []

    print(
        "Selected qualitative benchmark cases:"
    )

    try:
        for role, accepted_row in selected:
            video_id = int(
                accepted_row[
                    "video_id"
                ]
            )

            person_id = str(
                accepted_row[
                    "person_id"
                ]
            )

            print(
                f"  {role}: "
                f"test/{video_id}/{person_id}"
            )

            result = process_sequence(
                video_id,
                person_id,
                landmarks_model,
                eye_model,
            )

            verify_against_accepted(
                role,
                accepted_row,
                result,
            )

            verified.append(
                (
                    role,
                    accepted_row,
                    result,
                )
            )

    finally:
        landmarks_model.close()

    print(
        "Selected-case re-run verification: PASS"
    )

    clean_owned_outputs()

    output_rows = []

    for (
        role,
        accepted_row,
        result,
    ) in verified:
        output = write_case_outputs(
            role,
            accepted_row,
            result,
        )

        output[
            "gt_blinks"
        ] = int(
            accepted_row[
                "gt_blinks"
            ]
        )

        output[
            "predicted_blinks"
        ] = int(
            accepted_row[
                "predicted_blinks"
            ]
        )

        output[
            "true_positive"
        ] = int(
            accepted_row[
                "true_positive"
            ]
        )

        output[
            "false_positive"
        ] = int(
            accepted_row[
                "false_positive"
            ]
        )

        output[
            "false_negative"
        ] = int(
            accepted_row[
                "false_negative"
            ]
        )

        output_rows.append(
            output
        )

    with SELECTION_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                output_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    save_combined_image(
        output_rows
    )

    print(
        "\nQualitative benchmark generation: PASS"
    )

    print(
        "Generated professional dynamic benchmark videos and images."
    )

    print(
        "Saved:"
    )

    print(
        SELECTION_CSV_PATH
    )

    print(
        VIDEO_DIR
    )

    print(
        IMAGE_DIR
    )

    print(
        COMBINED_IMAGE_PATH
    )


if __name__ == "__main__":
    main()
