from pathlib import Path
import csv
import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd

from physiotrack.face.face_orientation import FaceOrientation

from aflw_head_pose_eval import (
    IMAGE_ROOT,
    RESULTS_CSV,
    SUMMARY_TXT,
    angular_error_degrees,
    convert_aflw_pose,
    load_annotations,
    make_face_box,
)


SCRIPT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = SCRIPT_DIR / "results"

QUALITATIVE_DIR = (
    RESULTS_DIR
    / "qualitative"
)

ANNOTATED_DIR = (
    QUALITATIVE_DIR
    / "annotated_images"
)

SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "aflw_head_pose_qualitative_selection.csv"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "aflw_head_pose_qualitative_examples.png"
)


DEVICE = "cpu"

ACCEPTED_SUCCESSFUL = 23407
ACCEPTED_FAILED_ELIGIBLE = 1
ACCEPTED_SUCCESS_RATE = 99.9957

ACCEPTED_YAW_MAE = 11.1036
ACCEPTED_PITCH_MAE = 13.7046
ACCEPTED_ROLL_MAE = 13.7529
ACCEPTED_OVERALL_MAE = 12.8537

ACCEPTED_YAW_MEDIAN = 7.7820
ACCEPTED_PITCH_MEDIAN = 7.1886
ACCEPTED_ROLL_MEDIAN = 4.9559

RERUN_TOLERANCE_DEGREES = 1e-3

CANDIDATE_COUNT_PER_ROLE = 40
MIN_FACE_AREA_RATIO = 0.025

BOX_COLOR = (
    0,
    220,
    255,
)

GT_COLOR = (
    90,
    220,
    90,
)

PRED_COLOR = (
    255,
    170,
    70,
)

TEXT_COLOR = (
    245,
    245,
    245,
)

BACKGROUND_COLOR = (
    24,
    24,
    24,
)

PANEL_COLOR = (
    52,
    52,
    52,
)


def load_accepted_summary():
    """Load and verify the accepted quantitative summary."""
    if not SUMMARY_TXT.is_file():
        raise FileNotFoundError(
            f"Accepted summary file not found: {SUMMARY_TXT}"
        )

    values = {}

    with SUMMARY_TXT.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith(
                "Successful predictions:"
            ):
                values[
                    "successful"
                ] = int(
                    line.split(
                        ":",
                        1,
                    )[1].strip()
                )

            elif line.startswith(
                "Failed eligible samples:"
            ):
                values[
                    "failed_eligible"
                ] = int(
                    line.split(
                        ":",
                        1,
                    )[1].strip()
                )

            elif line.startswith(
                "Success rate:"
            ):
                values[
                    "success_rate"
                ] = float(
                    line.split(
                        ":",
                        1,
                    )[1]
                    .strip()
                    .rstrip("%")
                )

            elif line.startswith(
                "Yaw MAE:"
            ):
                values[
                    "yaw_mae"
                ] = float(
                    line.split(
                        ":",
                        1,
                    )[1]
                    .replace(
                        "degrees",
                        "",
                    )
                    .strip()
                )

            elif line.startswith(
                "Pitch MAE:"
            ):
                values[
                    "pitch_mae"
                ] = float(
                    line.split(
                        ":",
                        1,
                    )[1]
                    .replace(
                        "degrees",
                        "",
                    )
                    .strip()
                )

            elif line.startswith(
                "Roll MAE:"
            ):
                values[
                    "roll_mae"
                ] = float(
                    line.split(
                        ":",
                        1,
                    )[1]
                    .replace(
                        "degrees",
                        "",
                    )
                    .strip()
                )

            elif line.startswith(
                "Overall MAE:"
            ):
                values[
                    "overall_mae"
                ] = float(
                    line.split(
                        ":",
                        1,
                    )[1]
                    .replace(
                        "degrees",
                        "",
                    )
                    .strip()
                )

    required = {
        "successful",
        "failed_eligible",
        "success_rate",
        "yaw_mae",
        "pitch_mae",
        "roll_mae",
        "overall_mae",
    }

    missing = (
        required
        - set(values)
    )

    if missing:
        raise RuntimeError(
            "Accepted summary is missing required values: "
            + ", ".join(
                sorted(missing)
            )
        )

    expected = {
        "successful": ACCEPTED_SUCCESSFUL,
        "failed_eligible": ACCEPTED_FAILED_ELIGIBLE,
        "success_rate": ACCEPTED_SUCCESS_RATE,
        "yaw_mae": ACCEPTED_YAW_MAE,
        "pitch_mae": ACCEPTED_PITCH_MAE,
        "roll_mae": ACCEPTED_ROLL_MAE,
        "overall_mae": ACCEPTED_OVERALL_MAE,
    }

    for key, expected_value in expected.items():
        actual_value = values[
            key
        ]

        if isinstance(
            expected_value,
            int,
        ):
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Accepted summary mismatch for "
                    f"{key}: {actual_value} "
                    f"!= {expected_value}"
                )

        else:
            if not np.isclose(
                actual_value,
                expected_value,
                atol=5e-5,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"Accepted summary mismatch for "
                    f"{key}: {actual_value} "
                    f"!= {expected_value}"
                )

    return values


def load_accepted_results():
    """Load and validate the accepted per-face quantitative results."""
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            f"Accepted result CSV not found: {RESULTS_CSV}"
        )

    data = pd.read_csv(
        RESULTS_CSV
    )

    required_columns = {
        "face_id",
        "filepath",
        "protocol_eligible",
        "status",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
    }

    missing_columns = sorted(
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Accepted result CSV is missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    successful = data[
        data[
            "status"
        ] == "ok"
    ].copy()

    if len(
        successful
    ) != ACCEPTED_SUCCESSFUL:
        raise RuntimeError(
            f"Expected "
            f"{ACCEPTED_SUCCESSFUL} "
            f"successful rows, found "
            f"{len(successful)}."
        )

    numeric_columns = [
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
    ]

    numeric_values = successful[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        np.isfinite(
            numeric_values
        )
    ):
        raise RuntimeError(
            "Successful accepted rows contain "
            "non-finite pose values."
        )

    successful[
        "mean_axis_error"
    ] = successful[
        [
            "yaw_error",
            "pitch_error",
            "roll_error",
        ]
    ].mean(
        axis=1
    )

    return (
        data,
        successful,
    )


def build_annotation_lookup():
    """Build a face-ID lookup from the same AFLW join as the evaluator."""
    annotations = (
        load_annotations()
    )

    lookup = {}

    for row in annotations:
        face_id = int(
            row[
                0
            ]
        )

        if face_id in lookup:
            raise RuntimeError(
                f"Duplicate AFLW face ID "
                f"in annotation lookup: "
                f"{face_id}"
            )

        lookup[
            face_id
        ] = row

    return lookup


def ranked_candidate_ids(
    successful,
):
    """Create deterministic numerical candidate rankings by role."""
    rankings = {}

    frontal = successful[
        (
            successful[
                "gt_yaw"
            ].abs() <= 15.0
        )
        & (
            successful[
                "gt_pitch"
            ].abs() <= 15.0
        )
        & (
            successful[
                "gt_roll"
            ].abs() <= 15.0
        )
    ].copy()

    rankings[
        "strong_frontal"
    ] = frontal.sort_values(
        [
            "mean_axis_error",
            "face_id",
        ],
        ascending=[
            True,
            True,
        ],
    )[
        "face_id"
    ].head(
        CANDIDATE_COUNT_PER_ROLE
    ).astype(
        int
    ).tolist()

    successful = successful.copy()

    successful[
        "representative_distance"
    ] = (
        successful[
            "mean_axis_error"
        ]
        - ACCEPTED_OVERALL_MAE
    ).abs()

    rankings[
        "representative"
    ] = successful.sort_values(
        [
            "representative_distance",
            "face_id",
        ],
        ascending=[
            True,
            True,
        ],
    )[
        "face_id"
    ].head(
        CANDIDATE_COUNT_PER_ROLE
    ).astype(
        int
    ).tolist()

    challenging_target = float(
        successful[
            "mean_axis_error"
        ].quantile(
            0.95
        )
    )

    successful[
        "challenging_distance"
    ] = (
        successful[
            "mean_axis_error"
        ]
        - challenging_target
    ).abs()

    rankings[
        "challenging"
    ] = successful.sort_values(
        [
            "challenging_distance",
            "face_id",
        ],
        ascending=[
            True,
            True,
        ],
    )[
        "face_id"
    ].head(
        CANDIDATE_COUNT_PER_ROLE
    ).astype(
        int
    ).tolist()

    directional_roles = [
        (
            "negative_yaw",
            "gt_yaw",
            -45.0,
        ),
        (
            "positive_yaw",
            "gt_yaw",
            45.0,
        ),
        (
            "negative_pitch",
            "gt_pitch",
            -30.0,
        ),
        (
            "positive_pitch",
            "gt_pitch",
            30.0,
        ),
    ]

    for (
        role,
        column,
        target,
    ) in directional_roles:
        if target < 0.0:
            subset = successful[
                successful[
                    column
                ] <= target
            ].copy()
        else:
            subset = successful[
                successful[
                    column
                ] >= target
            ].copy()

        subset[
            "direction_distance"
        ] = (
            subset[
                column
            ]
            - target
        ).abs()

        subset[
            "direction_error"
        ] = subset[
            [
                "yaw_error",
                "pitch_error",
                "roll_error",
            ]
        ].mean(
            axis=1
        )

        rankings[
            role
        ] = subset.sort_values(
            [
                "direction_error",
                "direction_distance",
                "face_id",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )[
            "face_id"
        ].head(
            CANDIDATE_COUNT_PER_ROLE
        ).astype(
            int
        ).tolist()

    successful[
        "abs_roll"
    ] = successful[
        "gt_roll"
    ].abs()

    roll_subset = successful[
        successful[
            "abs_roll"
        ] >= 35.0
    ].copy()

    rankings[
        "high_roll"
    ] = roll_subset.sort_values(
        [
            "roll_error",
            "abs_roll",
            "face_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )[
        "face_id"
    ].head(
        CANDIDATE_COUNT_PER_ROLE
    ).astype(
        int
    ).tolist()

    for role, face_ids in rankings.items():
        if not face_ids:
            raise RuntimeError(
                f"No candidates were found "
                f"for qualitative role: "
                f"{role}"
            )

    return rankings


def calculate_prominence(
    face_box,
    image_width,
    image_height,
):
    """Calculate deterministic target-face prominence."""
    x1, y1, x2, y2 = (
        face_box[
            0
        ].astype(
            float
        )
    )

    box_width = (
        x2
        - x1
    )

    box_height = (
        y2
        - y1
    )

    image_area = float(
        image_width
        * image_height
    )

    face_area_ratio = (
        box_width
        * box_height
        / image_area
    )

    center_x = (
        x1
        + x2
    ) / 2.0

    center_y = (
        y1
        + y2
    ) / 2.0

    normalized_dx = abs(
        center_x
        - image_width / 2.0
    ) / (
        image_width / 2.0
    )

    normalized_dy = abs(
        center_y
        - image_height / 2.0
    ) / (
        image_height / 2.0
    )

    center_distance = min(
        1.0,
        float(
            np.hypot(
                normalized_dx,
                normalized_dy,
            )
            / np.sqrt(
                2.0
            )
        ),
    )

    center_score = (
        1.0
        - center_distance
    )

    prominence = (
        0.85
        * face_area_ratio
        + 0.15
        * center_score
    )

    return {
        "face_area_ratio": face_area_ratio,
        "center_score": center_score,
        "prominence": prominence,
    }


def inspect_candidate(
    face_id,
    annotation_lookup,
):
    """Load one candidate and calculate visual prominence."""
    if face_id not in annotation_lookup:
        raise RuntimeError(
            f"Face ID {face_id} "
            f"is missing from the "
            f"evaluation annotation join."
        )

    row = annotation_lookup[
        face_id
    ]

    (
        _,
        filepath,
        x,
        y,
        w,
        h,
        _,
        _,
        _,
    ) = row

    image_path = (
        IMAGE_ROOT
        / filepath
    )

    frame = cv2.imread(
        str(
            image_path
        )
    )

    if frame is None:
        return None

    image_height, image_width = (
        frame.shape[
            :2
        ]
    )

    face_box = make_face_box(
        x,
        y,
        w,
        h,
        image_width,
        image_height,
    )

    if face_box is None:
        return None

    prominence = (
        calculate_prominence(
            face_box,
            image_width,
            image_height,
        )
    )

    return {
        "face_id": face_id,
        "filepath": filepath,
        "frame": frame,
        "face_box": face_box,
        **prominence,
    }


def select_qualitative_cases(
    rankings,
    accepted_by_face,
    annotation_lookup,
):
    """Select eight unique, visually interpretable benchmark faces."""
    role_order = [
        "strong_frontal",
        "representative",
        "challenging",
        "negative_yaw",
        "positive_yaw",
        "negative_pitch",
        "positive_pitch",
        "high_roll",
    ]

    used_face_ids = set()
    selections = []

    for role in role_order:
        inspected = []

        for face_id in rankings[
            role
        ]:
            candidate = inspect_candidate(
                face_id,
                annotation_lookup,
            )

            if candidate is None:
                continue

            if candidate[
                "face_area_ratio"
            ] < MIN_FACE_AREA_RATIO:
                continue

            inspected.append(
                candidate
            )

        if not inspected:
            for face_id in rankings[
                role
            ]:
                candidate = inspect_candidate(
                    face_id,
                    annotation_lookup,
                )

                if candidate is not None:
                    inspected.append(
                        candidate
                    )

        if not inspected:
            raise RuntimeError(
                f"No readable visual candidates "
                f"were found for role: "
                f"{role}"
            )

        inspected.sort(
            key=lambda item: (
                -item[
                    "prominence"
                ],
                item[
                    "face_id"
                ],
            )
        )

        selected = None

        for candidate in inspected:
            if candidate[
                "face_id"
            ] not in used_face_ids:
                selected = candidate
                break

        if selected is None:
            raise RuntimeError(
                f"Could not select a unique "
                f"face for role: "
                f"{role}"
            )

        face_id = selected[
            "face_id"
        ]

        used_face_ids.add(
            face_id
        )

        accepted_row = (
            accepted_by_face.loc[
                face_id
            ]
        )

        selections.append(
            {
                "role": role,
                "face_id": face_id,
                "filepath": selected[
                    "filepath"
                ],
                "frame": selected[
                    "frame"
                ],
                "face_box": selected[
                    "face_box"
                ],
                "face_area_ratio": selected[
                    "face_area_ratio"
                ],
                "center_score": selected[
                    "center_score"
                ],
                "prominence": selected[
                    "prominence"
                ],
                "accepted_row": accepted_row,
            }
        )

    return selections


def rerun_and_verify(
    selections,
    annotation_lookup,
):
    """Rerun selected faces and verify them against accepted CSV values."""
    estimator = FaceOrientation(
        device=DEVICE,
        verbose=False,
    )

    for selection in selections:
        face_id = selection[
            "face_id"
        ]

        row = annotation_lookup[
            face_id
        ]

        (
            _,
            _,
            _,
            _,
            _,
            _,
            raw_roll,
            raw_pitch,
            raw_yaw,
        ) = row

        gt_yaw, gt_pitch, gt_roll = (
            convert_aflw_pose(
                raw_roll,
                raw_pitch,
                raw_yaw,
            )
        )

        frame = selection[
            "frame"
        ]

        face_box = selection[
            "face_box"
        ]

        prediction = estimator.predict(
            frame,
            face_box,
        )

        if not prediction.instances:
            raise RuntimeError(
                f"Qualitative verification "
                f"failed for face ID "
                f"{face_id}: no prediction."
            )

        orientation = (
            prediction.instances[
                0
            ].orientation
        )

        if orientation is None:
            raise RuntimeError(
                f"Qualitative verification "
                f"failed for face ID "
                f"{face_id}: orientation "
                f"is None."
            )

        pred_yaw = float(
            orientation[
                "yaw"
            ]
        )

        pred_pitch = float(
            orientation[
                "pitch"
            ]
        )

        pred_roll = float(
            orientation[
                "roll"
            ]
        )

        rerun = {
            "gt_yaw": gt_yaw,
            "gt_pitch": gt_pitch,
            "gt_roll": gt_roll,
            "pred_yaw": pred_yaw,
            "pred_pitch": pred_pitch,
            "pred_roll": pred_roll,
            "yaw_error": angular_error_degrees(
                pred_yaw,
                gt_yaw,
            ),
            "pitch_error": angular_error_degrees(
                pred_pitch,
                gt_pitch,
            ),
            "roll_error": angular_error_degrees(
                pred_roll,
                gt_roll,
            ),
        }

        accepted_row = selection[
            "accepted_row"
        ]

        compare_keys = [
            "gt_yaw",
            "gt_pitch",
            "gt_roll",
            "pred_yaw",
            "pred_pitch",
            "pred_roll",
            "yaw_error",
            "pitch_error",
            "roll_error",
        ]

        for key in compare_keys:
            accepted_value = float(
                accepted_row[
                    key
                ]
            )

            rerun_value = float(
                rerun[
                    key
                ]
            )

            if not np.isclose(
                accepted_value,
                rerun_value,
                atol=RERUN_TOLERANCE_DEGREES,
                rtol=0.0,
            ):
                raise RuntimeError(
                    f"Qualitative verification "
                    f"mismatch for face ID "
                    f"{face_id}, {key}: "
                    f"rerun={rerun_value:.6f}, "
                    f"accepted="
                    f"{accepted_value:.6f}"
                )

        selection[
            "rerun"
        ] = rerun


def add_text(
    canvas,
    text,
    origin,
    scale,
    thickness=1,
    color=TEXT_COLOR,
):
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def resize_with_padding(
    image,
    width,
    height,
):
    """Resize an image while preserving aspect ratio."""
    image_height, image_width = (
        image.shape[
            :2
        ]
    )

    scale = min(
        width / image_width,
        height / image_height,
    )

    resized_width = max(
        1,
        int(
            round(
                image_width
                * scale
            )
        ),
    )

    resized_height = max(
        1,
        int(
            round(
                image_height
                * scale
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :
    ] = BACKGROUND_COLOR

    left = (
        width
        - resized_width
    ) // 2

    top = (
        height
        - resized_height
    ) // 2

    canvas[
        top:
        top + resized_height,
        left:
        left + resized_width,
    ] = resized

    return canvas


def create_face_crop(
    frame,
    face_box,
):
    """Create a padded crop around the target AFLW face."""
    x1, y1, x2, y2 = (
        face_box[
            0
        ].astype(
            float
        )
    )

    width = (
        x2
        - x1
    )

    height = (
        y2
        - y1
    )

    padding_x = (
        0.35
        * width
    )

    padding_y = (
        0.35
        * height
    )

    image_height, image_width = (
        frame.shape[
            :2
        ]
    )

    crop_x1 = max(
        0,
        int(
            np.floor(
                x1
                - padding_x
            )
        ),
    )

    crop_y1 = max(
        0,
        int(
            np.floor(
                y1
                - padding_y
            )
        ),
    )

    crop_x2 = min(
        image_width,
        int(
            np.ceil(
                x2
                + padding_x
            )
        ),
    )

    crop_y2 = min(
        image_height,
        int(
            np.ceil(
                y2
                + padding_y
            )
        ),
    )

    return frame[
        crop_y1:
        crop_y2,
        crop_x1:
        crop_x2,
    ].copy()


def draw_angle_gauge(
    canvas,
    label,
    gt_value,
    pred_value,
    error_value,
    left,
    top,
    width,
):
    """Draw a signed-angle comparison gauge for one pose axis."""
    add_text(
        canvas,
        label,
        (
            left,
            top,
        ),
        0.55,
        2,
    )

    add_text(
        canvas,
        (
            f"GT {gt_value:+.2f} deg   "
            f"Pred {pred_value:+.2f} deg   "
            f"Error {error_value:.2f} deg"
        ),
        (
            left,
            top + 30,
        ),
        0.42,
        1,
    )

    line_y = (
        top
        + 65
    )

    cv2.line(
        canvas,
        (
            left,
            line_y,
        ),
        (
            left + width,
            line_y,
        ),
        (
            130,
            130,
            130,
        ),
        2,
        cv2.LINE_AA,
    )

    center_x = (
        left
        + width // 2
    )

    cv2.line(
        canvas,
        (
            center_x,
            line_y - 10,
        ),
        (
            center_x,
            line_y + 10,
        ),
        (
            180,
            180,
            180,
        ),
        1,
        cv2.LINE_AA,
    )

    add_text(
        canvas,
        "-90",
        (
            left - 4,
            line_y + 25,
        ),
        0.33,
        1,
    )

    add_text(
        canvas,
        "0",
        (
            center_x - 5,
            line_y + 25,
        ),
        0.33,
        1,
    )

    add_text(
        canvas,
        "+90",
        (
            left + width - 28,
            line_y + 25,
        ),
        0.33,
        1,
    )

    def angle_to_x(
        angle,
    ):
        clipped = float(
            np.clip(
                angle,
                -90.0,
                90.0,
            )
        )

        return int(
            round(
                left
                + (
                    clipped
                    + 90.0
                )
                / 180.0
                * width
            )
        )

    gt_x = angle_to_x(
        gt_value
    )

    pred_x = angle_to_x(
        pred_value
    )

    cv2.circle(
        canvas,
        (
            gt_x,
            line_y - 9,
        ),
        7,
        GT_COLOR,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        canvas,
        (
            pred_x,
            line_y + 9,
        ),
        7,
        PRED_COLOR,
        -1,
        cv2.LINE_AA,
    )


def render_qualitative_image(
    selection,
    accepted_summary,
    output_path,
):
    """Render one 1920 x 1080 qualitative evidence figure."""
    frame = selection[
        "frame"
    ]

    face_box = selection[
        "face_box"
    ]

    rerun = selection[
        "rerun"
    ]

    annotated = frame.copy()

    x1, y1, x2, y2 = (
        face_box[
            0
        ].astype(
            int
        )
    )

    cv2.rectangle(
        annotated,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        BOX_COLOR,
        3,
        cv2.LINE_AA,
    )

    add_text(
        annotated,
        (
            f"AFLW target "
            f"face {selection['face_id']}"
        ),
        (
            max(
                5,
                x1,
            ),
            max(
                25,
                y1 - 10,
            ),
        ),
        0.55,
        2,
        BOX_COLOR,
    )

    crop = create_face_crop(
        frame,
        face_box,
    )

    canvas = np.zeros(
        (
            1080,
            1920,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :
    ] = BACKGROUND_COLOR

    add_text(
        canvas,
        "AFLW Head Pose Qualitative Validation",
        (
            25,
            50,
        ),
        0.95,
        2,
    )

    add_text(
        canvas,
        "Original image with evaluated AFLW face",
        (
            25,
            95,
        ),
        0.58,
        1,
    )

    add_text(
        canvas,
        "Target face crop",
        (
            705,
            95,
        ),
        0.58,
        1,
    )

    original_panel = resize_with_padding(
        annotated,
        650,
        650,
    )

    crop_panel = resize_with_padding(
        crop,
        650,
        650,
    )

    canvas[
        115:
        765,
        20:
        670,
    ] = original_panel

    canvas[
        115:
        765,
        700:
        1350,
    ] = crop_panel

    panel_left = 1370

    cv2.rectangle(
        canvas,
        (
            panel_left,
            20,
        ),
        (
            1900,
            1060,
        ),
        PANEL_COLOR,
        -1,
    )

    add_text(
        canvas,
        "QUALITATIVE CASE",
        (
            panel_left + 20,
            60,
        ),
        0.48,
        1,
    )

    add_text(
        canvas,
        selection[
            "role"
        ],
        (
            panel_left + 20,
            95,
        ),
        0.62,
        2,
    )

    add_text(
        canvas,
        (
            f"Face ID: "
            f"{selection['face_id']}"
        ),
        (
            panel_left + 20,
            130,
        ),
        0.45,
        1,
    )

    add_text(
        canvas,
        (
            f"Image: "
            f"{selection['filepath']}"
        ),
        (
            panel_left + 20,
            158,
        ),
        0.40,
        1,
    )

    add_text(
        canvas,
        "POSE COMPARISON",
        (
            panel_left + 20,
            215,
        ),
        0.48,
        1,
    )

    draw_angle_gauge(
        canvas,
        "Yaw",
        rerun[
            "gt_yaw"
        ],
        rerun[
            "pred_yaw"
        ],
        rerun[
            "yaw_error"
        ],
        panel_left + 20,
        255,
        470,
    )

    draw_angle_gauge(
        canvas,
        "Pitch",
        rerun[
            "gt_pitch"
        ],
        rerun[
            "pred_pitch"
        ],
        rerun[
            "pitch_error"
        ],
        panel_left + 20,
        375,
        470,
    )

    draw_angle_gauge(
        canvas,
        "Roll",
        rerun[
            "gt_roll"
        ],
        rerun[
            "pred_roll"
        ],
        rerun[
            "roll_error"
        ],
        panel_left + 20,
        495,
        470,
    )

    mean_error = (
        rerun[
            "yaw_error"
        ]
        + rerun[
            "pitch_error"
        ]
        + rerun[
            "roll_error"
        ]
    ) / 3.0

    add_text(
        canvas,
        (
            f"Mean axis error: "
            f"{mean_error:.2f} deg"
        ),
        (
            panel_left + 20,
            625,
        ),
        0.50,
        2,
    )

    add_text(
        canvas,
        "ACCEPTED FULL BENCHMARK",
        (
            panel_left + 20,
            690,
        ),
        0.48,
        1,
    )

    benchmark_lines = [
        (
            f"Successful: "
            f"{accepted_summary['successful']}"
        ),
        (
            f"Failed eligible: "
            f"{accepted_summary['failed_eligible']}"
        ),
        (
            f"Success rate: "
            f"{accepted_summary['success_rate']:.4f}%"
        ),
        (
            f"Yaw MAE: "
            f"{accepted_summary['yaw_mae']:.4f} deg"
        ),
        (
            f"Pitch MAE: "
            f"{accepted_summary['pitch_mae']:.4f} deg"
        ),
        (
            f"Roll MAE: "
            f"{accepted_summary['roll_mae']:.4f} deg"
        ),
        (
            f"Overall MAE: "
            f"{accepted_summary['overall_mae']:.4f} deg"
        ),
    ]

    for index, line in enumerate(
        benchmark_lines
    ):
        add_text(
            canvas,
            line,
            (
                panel_left + 20,
                730 + index * 28,
            ),
            0.42,
            1,
        )

    add_text(
        canvas,
        "PROTOCOL",
        (
            panel_left + 20,
            950,
        ),
        0.48,
        1,
    )

    add_text(
        canvas,
        "GT: AFLW FacePose, sign-converted",
        (
            panel_left + 20,
            985,
        ),
        0.39,
        1,
    )

    add_text(
        canvas,
        "Initialization: AFLW GT face rectangle",
        (
            panel_left + 20,
            1013,
        ),
        0.39,
        1,
    )

    add_text(
        canvas,
        "Backend: 6DRepNet360 | Device: CPU",
        (
            panel_left + 20,
            1041,
        ),
        0.39,
        1,
    )

    add_text(
        canvas,
        "GT marker",
        (
            25,
            820,
        ),
        0.47,
        1,
        GT_COLOR,
    )

    add_text(
        canvas,
        "PhysioTrack marker",
        (
            160,
            820,
        ),
        0.47,
        1,
        PRED_COLOR,
    )

    add_text(
        canvas,
        (
            "The visible box identifies the exact AFLW face evaluated in "
            "images that may contain multiple faces."
        ),
        (
            25,
            870,
        ),
        0.48,
        1,
    )

    add_text(
        canvas,
        (
            "Per-image pose values and errors are verified against the "
            "accepted quantitative CSV before this figure is written."
        ),
        (
            25,
            905,
        ),
        0.48,
        1,
    )

    add_text(
        canvas,
        (
            "The accepted benchmark metrics remain the aggregate results "
            "from all successful primary-protocol samples."
        ),
        (
            25,
            940,
        ),
        0.48,
        1,
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        canvas,
    ):
        raise RuntimeError(
            f"Could not save qualitative "
            f"figure: {output_path}"
        )


def create_combined_figure(
    selections,
    output_path,
):
    """Create a compact 2 x 4 target-face summary figure."""
    tile_width = 480
    tile_height = 540

    canvas = np.zeros(
        (
            tile_height * 2,
            tile_width * 4,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :
    ] = BACKGROUND_COLOR

    for index, selection in enumerate(
        selections
    ):
        crop = create_face_crop(
            selection[
                "frame"
            ],
            selection[
                "face_box"
            ],
        )

        crop_panel = resize_with_padding(
            crop,
            460,
            430,
        )

        row = (
            index // 4
        )

        column = (
            index % 4
        )

        left = (
            column
            * tile_width
            + 10
        )

        top = (
            row
            * tile_height
            + 10
        )

        canvas[
            top:
            top + 430,
            left:
            left + 460,
        ] = crop_panel

        rerun = selection[
            "rerun"
        ]

        mean_error = (
            rerun[
                "yaw_error"
            ]
            + rerun[
                "pitch_error"
            ]
            + rerun[
                "roll_error"
            ]
        ) / 3.0

        add_text(
            canvas,
            selection[
                "role"
            ],
            (
                left,
                top + 465,
            ),
            0.46,
            1,
        )

        add_text(
            canvas,
            (
                f"Face {selection['face_id']} | "
                f"mean error {mean_error:.2f} deg"
            ),
            (
                left,
                top + 495,
            ),
            0.40,
            1,
        )

        add_text(
            canvas,
            (
                f"GT Y/P/R "
                f"{rerun['gt_yaw']:+.1f}/"
                f"{rerun['gt_pitch']:+.1f}/"
                f"{rerun['gt_roll']:+.1f}"
            ),
            (
                left,
                top + 520,
            ),
            0.36,
            1,
            GT_COLOR,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(
            output_path
        ),
        canvas,
    ):
        raise RuntimeError(
            f"Could not save combined "
            f"qualitative figure: "
            f"{output_path}"
        )


def save_selection_csv(
    selections,
    output_path=SELECTION_CSV_PATH,
):
    fieldnames = [
        "role",
        "face_id",
        "filepath",
        "face_area_ratio",
        "center_score",
        "prominence",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
        "mean_axis_error",
        "annotated_image",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for selection in selections:
            rerun = selection[
                "rerun"
            ]

            mean_axis_error = (
                rerun[
                    "yaw_error"
                ]
                + rerun[
                    "pitch_error"
                ]
                + rerun[
                    "roll_error"
                ]
            ) / 3.0

            annotated_name = (
                f"{selection['role']}_"
                f"face_{selection['face_id']}.png"
            )

            writer.writerow(
                {
                    "role": selection[
                        "role"
                    ],
                    "face_id": selection[
                        "face_id"
                    ],
                    "filepath": selection[
                        "filepath"
                    ],
                    "face_area_ratio": selection[
                        "face_area_ratio"
                    ],
                    "center_score": selection[
                        "center_score"
                    ],
                    "prominence": selection[
                        "prominence"
                    ],
                    "gt_yaw": rerun[
                        "gt_yaw"
                    ],
                    "gt_pitch": rerun[
                        "gt_pitch"
                    ],
                    "gt_roll": rerun[
                        "gt_roll"
                    ],
                    "pred_yaw": rerun[
                        "pred_yaw"
                    ],
                    "pred_pitch": rerun[
                        "pred_pitch"
                    ],
                    "pred_roll": rerun[
                        "pred_roll"
                    ],
                    "yaw_error": rerun[
                        "yaw_error"
                    ],
                    "pitch_error": rerun[
                        "pitch_error"
                    ],
                    "roll_error": rerun[
                        "roll_error"
                    ],
                    "mean_axis_error": (
                        mean_axis_error
                    ),
                    "annotated_image": (
                        "results/qualitative/"
                        "annotated_images/"
                        f"{annotated_name}"
                    ),
                }
            )


def validate_staged_qualitative_outputs(
    staged_qualitative_dir,
    staged_selection_csv,
    staged_combined_figure,
    selections,
):
    """Validate staged qualitative evidence before replacing final outputs."""
    expected_roles = [
        selection["role"]
        for selection in selections
    ]

    expected_face_ids = [
        int(selection["face_id"])
        for selection in selections
    ]

    if len(expected_roles) != 8:
        raise RuntimeError(
            f"Expected 8 qualitative cases, found {len(expected_roles)}."
        )

    if len(set(expected_roles)) != len(expected_roles):
        raise RuntimeError(
            "Qualitative case roles are not unique."
        )

    if len(set(expected_face_ids)) != len(expected_face_ids):
        raise RuntimeError(
            "Qualitative face IDs are not unique."
        )

    if not staged_selection_csv.is_file():
        raise RuntimeError(
            "Staged qualitative selection CSV is missing."
        )

    data = pd.read_csv(
        staged_selection_csv
    )

    required_columns = [
        "role",
        "face_id",
        "filepath",
        "face_area_ratio",
        "center_score",
        "prominence",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
        "mean_axis_error",
        "annotated_image",
    ]

    if data.columns.tolist() != required_columns:
        raise RuntimeError(
            "Staged qualitative selection CSV schema is incorrect."
        )

    if data["role"].tolist() != expected_roles:
        raise RuntimeError(
            "Staged qualitative case order or roles changed unexpectedly."
        )

    if data["face_id"].astype(int).tolist() != expected_face_ids:
        raise RuntimeError(
            "Staged qualitative face IDs do not match the verified selections."
        )

    numeric_columns = [
        "face_area_ratio",
        "center_score",
        "prominence",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
        "mean_axis_error",
    ]

    numeric_values = data[
        numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError(
            "Staged qualitative selection CSV contains non-finite values."
        )

    for row_index, selection in enumerate(
        selections
    ):
        rerun = selection["rerun"]

        expected_values = {
            "face_area_ratio": float(
                selection["face_area_ratio"]
            ),
            "center_score": float(
                selection["center_score"]
            ),
            "prominence": float(
                selection["prominence"]
            ),
            "gt_yaw": float(
                rerun["gt_yaw"]
            ),
            "gt_pitch": float(
                rerun["gt_pitch"]
            ),
            "gt_roll": float(
                rerun["gt_roll"]
            ),
            "pred_yaw": float(
                rerun["pred_yaw"]
            ),
            "pred_pitch": float(
                rerun["pred_pitch"]
            ),
            "pred_roll": float(
                rerun["pred_roll"]
            ),
            "yaw_error": float(
                rerun["yaw_error"]
            ),
            "pitch_error": float(
                rerun["pitch_error"]
            ),
            "roll_error": float(
                rerun["roll_error"]
            ),
            "mean_axis_error": float(
                (
                    rerun["yaw_error"]
                    + rerun["pitch_error"]
                    + rerun["roll_error"]
                )
                / 3.0
            ),
        }

        for column, expected_value in expected_values.items():
            actual_value = float(
                data.iloc[row_index][column]
            )

            if not np.isclose(
                actual_value,
                expected_value,
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(
                    f"Staged qualitative value mismatch for "
                    f"{selection['role']}, {column}."
                )

    annotated_dir = (
        staged_qualitative_dir
        / "annotated_images"
    )

    expected_names = [
        f"{role}_face_{face_id}.png"
        for role, face_id in zip(
            expected_roles,
            expected_face_ids,
        )
    ]

    actual_names = sorted(
        path.name
        for path in annotated_dir.glob(
            "*.png"
        )
    )

    if actual_names != sorted(expected_names):
        raise RuntimeError(
            "Staged annotated qualitative image set is incomplete or unexpected."
        )

    for image_name in expected_names:
        image_path = (
            annotated_dir
            / image_name
        )

        image = cv2.imread(
            str(
                image_path
            )
        )

        if image is None or image.size == 0:
            raise RuntimeError(
                f"Staged annotated image could not be read: {image_path}"
            )

        if image.shape[:2] != (
            1080,
            1920,
        ):
            raise RuntimeError(
                f"Staged annotated image has unexpected dimensions: "
                f"{image_path}, {image.shape[:2]}"
            )

    combined = cv2.imread(
        str(
            staged_combined_figure
        )
    )

    if combined is None or combined.size == 0:
        raise RuntimeError(
            "Staged combined qualitative figure could not be read."
        )

    if combined.shape[:2] != (
        1080,
        1920,
    ):
        raise RuntimeError(
            "Staged combined qualitative figure has unexpected dimensions."
        )


def replace_owned_qualitative_outputs(
    staging_root,
    staged_qualitative_dir,
    staged_combined_figure,
):
    """Replace only qualitative-script-owned outputs with rollback protection."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    backup_dir = staging_root / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    qualitative_backup = backup_dir / "qualitative"
    figure_backup = backup_dir / COMBINED_FIGURE_PATH.name

    qualitative_installed = False
    figure_installed = False

    try:
        if QUALITATIVE_DIR.exists():
            os.replace(QUALITATIVE_DIR, qualitative_backup)

        if COMBINED_FIGURE_PATH.exists():
            os.replace(COMBINED_FIGURE_PATH, figure_backup)

        os.replace(staged_qualitative_dir, QUALITATIVE_DIR)
        qualitative_installed = True

        os.replace(staged_combined_figure, COMBINED_FIGURE_PATH)
        figure_installed = True

    except Exception:
        if figure_installed and COMBINED_FIGURE_PATH.exists():
            COMBINED_FIGURE_PATH.unlink()

        if qualitative_installed and QUALITATIVE_DIR.exists():
            shutil.rmtree(QUALITATIVE_DIR)

        if qualitative_backup.exists():
            os.replace(qualitative_backup, QUALITATIVE_DIR)

        if figure_backup.exists():
            os.replace(figure_backup, COMBINED_FIGURE_PATH)

        raise


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".aflw_head_pose_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    staged_qualitative_dir = (
        staging_root
        / "qualitative"
    )

    staged_annotated_dir = (
        staged_qualitative_dir
        / "annotated_images"
    )

    staged_selection_csv = (
        staged_qualitative_dir
        / SELECTION_CSV_PATH.name
    )

    staged_combined_figure = (
        staging_root
        / COMBINED_FIGURE_PATH.name
    )

    staged_annotated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Staging directory:",
        staging_root,
    )

    try:
        accepted_summary = (
            load_accepted_summary()
        )

        _, successful = (
            load_accepted_results()
        )

        annotation_lookup = (
            build_annotation_lookup()
        )

        accepted_by_face = (
            successful
            .set_index(
                "face_id"
            )
        )

        rankings = (
            ranked_candidate_ids(
                successful
            )
        )

        selections = (
            select_qualitative_cases(
                rankings,
                accepted_by_face,
                annotation_lookup,
            )
        )

        print(
            "Selected qualitative cases:"
        )

        for selection in selections:
            accepted_row = selection[
                "accepted_row"
            ]

            print(
                f"- {selection['role']}: "
                f"face_id={selection['face_id']}, "
                f"filepath={selection['filepath']}, "
                f"accepted mean axis error="
                f"{float(accepted_row['mean_axis_error']):.4f} deg, "
                f"face area="
                f"{selection['face_area_ratio'] * 100.0:.2f}%"
            )

        print()
        print(
            "Rerunning selected faces and "
            "verifying accepted per-face results..."
        )

        rerun_and_verify(
            selections,
            annotation_lookup,
        )

        print(
            "Per-face quantitative verification: PASS"
        )

        for selection in selections:
            output_path = (
                staged_annotated_dir
                / (
                    f"{selection['role']}_"
                    f"face_{selection['face_id']}.png"
                )
            )

            render_qualitative_image(
                selection,
                accepted_summary,
                output_path,
            )

        save_selection_csv(
            selections,
            staged_selection_csv,
        )

        create_combined_figure(
            selections,
            staged_combined_figure,
        )

        print()
        print(
            "Validating staged qualitative outputs..."
        )

        validate_staged_qualitative_outputs(
            staged_qualitative_dir,
            staged_selection_csv,
            staged_combined_figure,
            selections,
        )

        replace_owned_qualitative_outputs(
            staging_root,
            staged_qualitative_dir,
            staged_combined_figure,
        )

        print(
            "Committed final qualitative outputs."
        )


    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


    print()
    print(
        "AFLW qualitative benchmark evidence "
        "completed successfully."
    )

    print(
        "Annotated images:",
        ANNOTATED_DIR,
    )

    print(
        "Selection CSV:",
        SELECTION_CSV_PATH,
    )

    print(
        "Combined figure:",
        COMBINED_FIGURE_PATH,
    )


if __name__ == "__main__":
    main()
