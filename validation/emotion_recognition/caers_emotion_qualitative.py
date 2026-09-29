from pathlib import Path
import csv
import math
import os
import shutil
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np

from physiotrack.face.emotion import FaceEmotion


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "CAER-S"
TEST_ROOT = DATASET_ROOT / "test"

RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "caers_emotion_results.csv"
SUMMARY_PATH = RESULTS_DIR / "caers_emotion_summary.txt"

QUALITATIVE_DIR = RESULTS_DIR / "qualitative"
ANNOTATED_DIR = QUALITATIVE_DIR / "annotated_images"
SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "caers_emotion_qualitative_selection.csv"
)

FIGURES_DIR = RESULTS_DIR / "figures"
COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "caers_emotion_qualitative_examples.png"
)

MODEL_NAME = "enet_b0_8_best_afew"
ENGINE = "onnx"

GT_CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

MODEL_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

SCORE_COLUMNS = {
    "Anger": "score_anger",
    "Contempt": "score_contempt",
    "Disgust": "score_disgust",
    "Fear": "score_fear",
    "Happiness": "score_happiness",
    "Neutral": "score_neutral",
    "Sadness": "score_sadness",
    "Surprise": "score_surprise",
}

CONFIDENCE_TOLERANCE = 1e-6
SCORE_TOLERANCE = 1e-6

EXPECTED_RESULT_COLUMNS = [
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


def parse_bool(value):
    """Parse a CSV boolean value."""
    normalized = value.strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    raise RuntimeError(
        f"Unexpected boolean value: {value}"
    )


def load_accepted_results():
    """Load and validate the accepted quantitative result table."""
    if not RESULTS_PATH.is_file():
        raise RuntimeError(
            f"Quantitative results not found: {RESULTS_PATH}"
        )

    if not SUMMARY_PATH.is_file():
        raise RuntimeError(
            f"Quantitative summary not found: {SUMMARY_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != EXPECTED_RESULT_COLUMNS:
            raise RuntimeError(
                "Quantitative result schema does not match the "
                "expected accepted schema."
            )

        rows = list(
            reader
        )

    if len(rows) != 13942:
        raise RuntimeError(
            f"Expected 13,942 accepted quantitative rows, found "
            f"{len(rows):,}."
        )

    seen_paths = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        relative_path = row[
            "relative_path"
        ]

        if relative_path in seen_paths:
            raise RuntimeError(
                f"Duplicate quantitative path at row {row_number}: "
                f"{relative_path}"
            )

        seen_paths.add(
            relative_path
        )

        if row[
            "status"
        ] != "ok":
            raise RuntimeError(
                "Qualitative generation requires the accepted zero-failure "
                f"quantitative package; found status={row['status']!r} "
                f"at row {row_number}."
            )

        expected_label = row[
            "expected_model_label"
        ]
        predicted_label = row[
            "predicted_label"
        ]

        if expected_label not in GT_CLASSES:
            raise RuntimeError(
                f"Unexpected ground-truth label at row {row_number}."
            )

        if predicted_label not in MODEL_CLASSES:
            raise RuntimeError(
                f"Unexpected predicted label at row {row_number}."
            )

        recorded_correct = parse_bool(
            row[
                "correct"
            ]
        )

        if recorded_correct != (
            predicted_label
            == expected_label
        ):
            raise RuntimeError(
                f"Correctness mismatch at row {row_number}."
            )

        scores = np.asarray(
            [
                float(
                    row[
                        SCORE_COLUMNS[
                            class_name
                        ]
                    ]
                )
                for class_name in MODEL_CLASSES
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                scores
            )
        ):
            raise RuntimeError(
                f"Non-finite score at row {row_number}."
            )

        if not math.isclose(
            float(
                scores.sum()
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise RuntimeError(
                f"Scores do not sum to one at row {row_number}."
            )

        if MODEL_CLASSES[
            int(
                np.argmax(
                    scores
                )
            )
        ] != predicted_label:
            raise RuntimeError(
                f"Prediction is not score argmax at row {row_number}."
            )

    return rows


def median_representative(
    candidates,
    value_key="confidence",
):
    """Select the deterministic candidate nearest the median value."""
    if not candidates:
        raise RuntimeError(
            "No candidates available for qualitative role."
        )

    values = np.asarray(
        [
            float(
                row[
                    value_key
                ]
            )
            for row in candidates
        ],
        dtype=np.float64,
    )

    target = float(
        np.median(
            values
        )
    )

    ranked = sorted(
        candidates,
        key=lambda row: (
            abs(
                float(
                    row[
                        value_key
                    ]
                )
                - target
            ),
            row[
                "relative_path"
            ],
        ),
    )

    return ranked[0]


def select_qualitative_cases(
    rows,
):
    """Select deterministic representative and failure-oriented examples."""
    selections = []
    used_paths = set()

    for class_name in GT_CLASSES:
        candidates = [
            row
            for row in rows
            if (
                row[
                    "expected_model_label"
                ]
                == class_name
                and parse_bool(
                    row[
                        "correct"
                    ]
                )
            )
        ]

        selected = median_representative(
            candidates
        )

        selections.append(
            {
                "role": (
                    "representative_correct_"
                    + class_name.lower()
                ),
                "selection_reason": (
                    "Correct prediction with confidence nearest the "
                    f"median among correct {class_name} samples."
                ),
                "row": selected,
            }
        )

        used_paths.add(
            selected[
                "relative_path"
            ]
        )

    incorrect = [
        row
        for row in rows
        if (
            not parse_bool(
                row[
                    "correct"
                ]
            )
            and row[
                "relative_path"
            ]
            not in used_paths
        )
    ]

    representative_error = median_representative(
        incorrect
    )

    selections.append(
        {
            "role": "representative_incorrect",
            "selection_reason": (
                "Incorrect prediction with confidence nearest the "
                "median among all incorrect samples."
            ),
            "row": representative_error,
        }
    )

    used_paths.add(
        representative_error[
            "relative_path"
        ]
    )

    high_confidence_errors = sorted(
        [
            row
            for row in incorrect
            if row[
                "relative_path"
            ]
            not in used_paths
        ],
        key=lambda row: (
            -float(
                row[
                    "confidence"
                ]
            ),
            row[
                "relative_path"
            ],
        ),
    )

    if not high_confidence_errors:
        raise RuntimeError(
            "No high-confidence incorrect candidate is available."
        )

    high_confidence_error = high_confidence_errors[0]

    selections.append(
        {
            "role": "high_confidence_incorrect",
            "selection_reason": (
                "Highest-confidence incorrect prediction after "
                "excluding previously selected cases."
            ),
            "row": high_confidence_error,
        }
    )

    used_paths.add(
        high_confidence_error[
            "relative_path"
        ]
    )

    contempt_candidates = [
        row
        for row in rows
        if (
            row[
                "predicted_label"
            ]
            == "Contempt"
            and row[
                "relative_path"
            ]
            not in used_paths
        )
    ]

    contempt_case = median_representative(
        contempt_candidates
    )

    selections.append(
        {
            "role": "representative_contempt_prediction",
            "selection_reason": (
                "Contempt prediction with confidence nearest the median "
                "among all Contempt predictions."
            ),
            "row": contempt_case,
        }
    )

    return selections


def load_and_crop(
    row,
):
    """Load the source frame and reproduce the accepted clipped target crop."""
    image_path = (
        TEST_ROOT
        / row[
            "relative_path"
        ]
    )

    if not image_path.is_file():
        raise RuntimeError(
            f"Qualitative source image not found: {image_path}"
        )

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise RuntimeError(
            f"Could not read qualitative source image: {image_path}"
        )

    height, width = image.shape[:2]

    original_box = (
        int(
            row[
                "original_x1"
            ]
        ),
        int(
            row[
                "original_y1"
            ]
        ),
        int(
            row[
                "original_x2"
            ]
        ),
        int(
            row[
                "original_y2"
            ]
        ),
    )

    clipped_box = (
        max(
            0,
            min(
                original_box[0],
                width,
            ),
        ),
        max(
            0,
            min(
                original_box[1],
                height,
            ),
        ),
        max(
            0,
            min(
                original_box[2],
                width,
            ),
        ),
        max(
            0,
            min(
                original_box[3],
                height,
            ),
        ),
    )

    accepted_clipped_box = (
        int(
            row[
                "clipped_x1"
            ]
        ),
        int(
            row[
                "clipped_y1"
            ]
        ),
        int(
            row[
                "clipped_x2"
            ]
        ),
        int(
            row[
                "clipped_y2"
            ]
        ),
    )

    if clipped_box != accepted_clipped_box:
        raise RuntimeError(
            "Recomputed clipped face box does not match accepted "
            f"quantitative result for {row['relative_path']}."
        )

    x1, y1, x2, y2 = clipped_box

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise RuntimeError(
            f"Invalid qualitative crop: {row['relative_path']}"
        )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise RuntimeError(
            f"Empty qualitative crop: {row['relative_path']}"
        )

    return (
        image,
        crop,
        clipped_box,
    )


def rerun_and_verify(
    selections,
):
    """Rerun FaceEmotion and verify every selected case against accepted results."""
    model = FaceEmotion(
        model_name=MODEL_NAME,
        engine=ENGINE,
    )

    for selection in selections:
        row = selection[
            "row"
        ]

        (
            image,
            crop,
            clipped_box,
        ) = load_and_crop(
            row
        )

        prediction = model.predict(
            crop
        )

        if prediction[
            "emotion"
        ] != row[
            "predicted_label"
        ]:
            raise RuntimeError(
                "Qualitative rerun prediction mismatch for "
                f"{row['relative_path']}."
            )

        rerun_confidence = float(
            prediction[
                "confidence"
            ]
        )

        accepted_confidence = float(
            row[
                "confidence"
            ]
        )

        if not math.isclose(
            rerun_confidence,
            accepted_confidence,
            rel_tol=0.0,
            abs_tol=CONFIDENCE_TOLERANCE,
        ):
            raise RuntimeError(
                "Qualitative rerun confidence mismatch for "
                f"{row['relative_path']}."
            )

        rerun_scores = prediction[
            "scores"
        ]

        if set(
            rerun_scores
        ) != set(
            MODEL_CLASSES
        ):
            raise RuntimeError(
                "Unexpected qualitative rerun score schema for "
                f"{row['relative_path']}."
            )

        for class_name in MODEL_CLASSES:
            accepted_score = float(
                row[
                    SCORE_COLUMNS[
                        class_name
                    ]
                ]
            )

            rerun_score = float(
                rerun_scores[
                    class_name
                ]
            )

            if not math.isclose(
                rerun_score,
                accepted_score,
                rel_tol=0.0,
                abs_tol=SCORE_TOLERANCE,
            ):
                raise RuntimeError(
                    "Qualitative rerun score mismatch for "
                    f"{row['relative_path']} ({class_name})."
                )

        selection[
            "image"
        ] = image
        selection[
            "crop"
        ] = crop
        selection[
            "clipped_box"
        ] = clipped_box
        selection[
            "rerun_confidence"
        ] = rerun_confidence


def draw_case_panel(
    selection,
):
    """Create one annotated scene-plus-crop qualitative panel."""
    row = selection[
        "row"
    ]

    image = selection[
        "image"
    ].copy()

    crop = selection[
        "crop"
    ]

    x1, y1, x2, y2 = selection[
        "clipped_box"
    ]

    cv2.rectangle(
        image,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        (
            0,
            255,
            0,
        ),
        3,
    )

    scene_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )
    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )

    figure = plt.figure(
        figsize=(
            10.0,
            5.2,
        )
    )

    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[
            2.3,
            1.0,
        ],
    )

    scene_axis = figure.add_subplot(
        grid[
            0,
            0,
        ]
    )
    crop_axis = figure.add_subplot(
        grid[
            0,
            1,
        ]
    )

    scene_axis.imshow(
        scene_rgb
    )
    scene_axis.set_title(
        "CAER-S scene and target-face box"
    )
    scene_axis.axis(
        "off"
    )

    crop_axis.imshow(
        crop_rgb
    )
    crop_axis.set_title(
        "FaceEmotion input crop"
    )
    crop_axis.axis(
        "off"
    )

    correctness = (
        "Correct"
        if parse_bool(
            row[
                "correct"
            ]
        )
        else "Incorrect"
    )

    figure.suptitle(
        (
            f"{selection['role']} | {row['relative_path']}\n"
            f"GT={row['expected_model_label']} | "
            f"Prediction={row['predicted_label']} | "
            f"Confidence={float(row['confidence']):.3f} | "
            f"{correctness}"
        ),
        fontsize=11,
    )

    figure.tight_layout()

    return figure


def save_selection_csv(
    selections,
    output_path,
):
    """Save a machine-readable record of qualitative selection."""
    fieldnames = [
        "role",
        "selection_reason",
        "relative_path",
        "dataset_label",
        "ground_truth",
        "predicted_label",
        "confidence",
        "correct",
        "clipped_x1",
        "clipped_y1",
        "clipped_x2",
        "clipped_y2",
        "score_anger",
        "score_contempt",
        "score_disgust",
        "score_fear",
        "score_happiness",
        "score_neutral",
        "score_sadness",
        "score_surprise",
        "annotated_image",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for selection in selections:
            row = selection[
                "row"
            ]

            annotated_name = (
                f"{selection['role']}.png"
            )

            writer.writerow(
                {
                    "role": selection[
                        "role"
                    ],
                    "selection_reason": selection[
                        "selection_reason"
                    ],
                    "relative_path": row[
                        "relative_path"
                    ],
                    "dataset_label": row[
                        "dataset_label"
                    ],
                    "ground_truth": row[
                        "expected_model_label"
                    ],
                    "predicted_label": row[
                        "predicted_label"
                    ],
                    "confidence": row[
                        "confidence"
                    ],
                    "correct": row[
                        "correct"
                    ],
                    "clipped_x1": row[
                        "clipped_x1"
                    ],
                    "clipped_y1": row[
                        "clipped_y1"
                    ],
                    "clipped_x2": row[
                        "clipped_x2"
                    ],
                    "clipped_y2": row[
                        "clipped_y2"
                    ],
                    "score_anger": row[
                        "score_anger"
                    ],
                    "score_contempt": row[
                        "score_contempt"
                    ],
                    "score_disgust": row[
                        "score_disgust"
                    ],
                    "score_fear": row[
                        "score_fear"
                    ],
                    "score_happiness": row[
                        "score_happiness"
                    ],
                    "score_neutral": row[
                        "score_neutral"
                    ],
                    "score_sadness": row[
                        "score_sadness"
                    ],
                    "score_surprise": row[
                        "score_surprise"
                    ],
                    "annotated_image": (
                        "results/qualitative/annotated_images/"
                        f"{annotated_name}"
                    ),
                }
            )


def create_combined_figure(
    selections,
    output_path,
):
    """Create the combined 2-by-5 qualitative summary figure."""
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(
            20.0,
            8.5,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(
        -1
    )

    for axis, selection in zip(
        axes,
        selections,
    ):
        row = selection[
            "row"
        ]

        image = selection[
            "image"
        ].copy()

        x1, y1, x2, y2 = selection[
            "clipped_box"
        ]

        cv2.rectangle(
            image,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            3,
        )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(
            image_rgb
        )

        correctness = (
            "Correct"
            if parse_bool(
                row[
                    "correct"
                ]
            )
            else "Incorrect"
        )

        axis.set_title(
            (
                f"{selection['role']}\n"
                f"GT: {row['expected_model_label']} | "
                f"Pred: {row['predicted_label']}\n"
                f"Conf: {float(row['confidence']):.3f} | "
                f"{correctness}"
            ),
            fontsize=9,
        )

        axis.axis(
            "off"
        )

    figure.suptitle(
        (
            "CAER-S Emotion Recognition Qualitative Examples\n"
            "Green box: externally supplied target-face crop used by "
            "PhysioTrack FaceEmotion"
        ),
        fontsize=14,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.94,
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def validate_staged_outputs(
    staging_qualitative_dir,
    staging_combined_figure,
    selections,
):
    """Validate staged qualitative outputs before final replacement."""
    staging_annotated_dir = (
        staging_qualitative_dir
        / "annotated_images"
    )

    staging_selection_csv = (
        staging_qualitative_dir
        / SELECTION_CSV_PATH.name
    )

    if not staging_annotated_dir.is_dir():
        raise RuntimeError(
            "Staged annotated-image directory was not created."
        )

    annotated_files = sorted(
        staging_annotated_dir.glob(
            "*.png"
        )
    )

    if len(
        annotated_files
    ) != len(
        selections
    ):
        raise RuntimeError(
            "Staged annotated-image count does not match the "
            "qualitative selection count."
        )

    if not staging_selection_csv.is_file():
        raise RuntimeError(
            "Staged qualitative selection CSV was not created."
        )

    with staging_selection_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        selection_rows = list(
            csv.DictReader(
                handle
            )
        )

    if len(
        selection_rows
    ) != len(
        selections
    ):
        raise RuntimeError(
            "Staged qualitative selection CSV row count is invalid."
        )

    if not staging_combined_figure.is_file():
        raise RuntimeError(
            "Staged combined qualitative figure was not created."
        )

    if staging_combined_figure.stat().st_size == 0:
        raise RuntimeError(
            "Staged combined qualitative figure is empty."
        )

    for path in annotated_files:
        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Staged annotated image is empty: {path.name}"
            )


def replace_qualitative_outputs(
    staging_qualitative_dir,
    staging_combined_figure,
):
    """Replace only qualitative-script-owned outputs with rollback support."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_qualitative_dir = (
        RESULTS_DIR
        / ".caers_emotion_qualitative_backup"
    )
    backup_combined_figure = (
        FIGURES_DIR
        / ".caers_emotion_qualitative_examples_backup.png"
    )

    if backup_qualitative_dir.exists():
        shutil.rmtree(
            backup_qualitative_dir
        )

    if backup_combined_figure.exists():
        backup_combined_figure.unlink()

    qualitative_backed_up = False
    figure_backed_up = False

    try:
        if QUALITATIVE_DIR.exists():
            os.replace(
                QUALITATIVE_DIR,
                backup_qualitative_dir,
            )
            qualitative_backed_up = True

        if COMBINED_FIGURE_PATH.exists():
            os.replace(
                COMBINED_FIGURE_PATH,
                backup_combined_figure,
            )
            figure_backed_up = True

        os.replace(
            staging_qualitative_dir,
            QUALITATIVE_DIR,
        )

        os.replace(
            staging_combined_figure,
            COMBINED_FIGURE_PATH,
        )

    except Exception:
        if QUALITATIVE_DIR.exists():
            shutil.rmtree(
                QUALITATIVE_DIR,
                ignore_errors=True,
            )

        if COMBINED_FIGURE_PATH.exists():
            COMBINED_FIGURE_PATH.unlink(
                missing_ok=True
            )

        if qualitative_backed_up:
            os.replace(
                backup_qualitative_dir,
                QUALITATIVE_DIR,
            )

        if figure_backed_up:
            os.replace(
                backup_combined_figure,
                COMBINED_FIGURE_PATH,
            )

        raise

    else:
        if backup_qualitative_dir.exists():
            shutil.rmtree(
                backup_qualitative_dir
            )

        if backup_combined_figure.exists():
            backup_combined_figure.unlink()


def main():
    """Generate verified CAER-S qualitative benchmark evidence."""
    print(
        "CAER-S Emotion Recognition Qualitative Validation"
    )
    print(
        "=" * 49
    )

    rows = load_accepted_results()

    selections = select_qualitative_cases(
        rows
    )

    print()
    print(
        "Selected qualitative cases:"
    )

    for selection in selections:
        row = selection[
            "row"
        ]

        print(
            f"- {selection['role']}: "
            f"{row['relative_path']}, "
            f"GT={row['expected_model_label']}, "
            f"prediction={row['predicted_label']}, "
            f"confidence={float(row['confidence']):.4f}"
        )

    print()
    print(
        "Rerunning selected target-face crops and verifying "
        "accepted predictions..."
    )

    rerun_and_verify(
        selections
    )

    print(
        "Per-image quantitative verification: PASS"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".caers_emotion_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    staging_qualitative_dir = (
        staging_root
        / "qualitative"
    )
    staging_annotated_dir = (
        staging_qualitative_dir
        / "annotated_images"
    )
    staging_combined_figure = (
        staging_root
        / COMBINED_FIGURE_PATH.name
    )

    try:
        staging_annotated_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for selection in selections:
            figure = draw_case_panel(
                selection
            )

            output_path = (
                staging_annotated_dir
                / f"{selection['role']}.png"
            )

            figure.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )

            plt.close(
                figure
            )

        save_selection_csv(
            selections,
            staging_qualitative_dir
            / SELECTION_CSV_PATH.name,
        )

        create_combined_figure(
            selections,
            staging_combined_figure,
        )

        validate_staged_outputs(
            staging_qualitative_dir,
            staging_combined_figure,
            selections,
        )

        replace_qualitative_outputs(
            staging_qualitative_dir,
            staging_combined_figure,
        )

    finally:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )

    print()
    print(
        "CAER-S qualitative validation completed successfully."
    )
    print(
        f"Annotated images: {ANNOTATED_DIR}"
    )
    print(
        f"Selection CSV: {SELECTION_CSV_PATH}"
    )
    print(
        f"Combined figure: {COMBINED_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()
