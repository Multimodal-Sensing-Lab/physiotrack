from pathlib import Path
import csv
import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd

from physiotrack.face.landmarks import FaceLandmarks
from mediapipe_300w_mapping import get_mediapipe_300w_51


VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "300W"

RESULTS_DIR = VALIDATION_DIR / "results"

RESULTS_CSV = (
    RESULTS_DIR
    / "300w_landmark_results.csv"
)

QUALITATIVE_DIR = (
    RESULTS_DIR
    / "qualitative"
)

ANNOTATED_DIR = (
    QUALITATIVE_DIR
    / "annotated_images"
)

SELECTION_CSV = (
    QUALITATIVE_DIR
    / "300w_landmark_qualitative_selection.csv"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

SUMMARY_FIGURE = (
    FIGURES_DIR
    / "300w_landmark_qualitative_examples.png"
)

FINAL_QUALITATIVE_DIR = QUALITATIVE_DIR
FINAL_ANNOTATED_DIR = ANNOTATED_DIR
FINAL_SELECTION_CSV = SELECTION_CSV
FINAL_SUMMARY_FIGURE = SUMMARY_FIGURE

DATASETS = {
    "Indoor": DATASET_ROOT / "01_Indoor",
    "Outdoor": DATASET_ROOT / "02_Outdoor",
}

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

IMAGE_CANVAS_WIDTH = 1400
PANEL_WIDTH = (
    OUTPUT_WIDTH
    - IMAGE_CANVAS_WIDTH
)

NME_TOLERANCE_PERCENT = 0.01

COLOR_GT = (0, 215, 255)
COLOR_PREDICTED = (0, 210, 0)
COLOR_VECTOR = (190, 190, 190)
COLOR_FACE_BOX = (255, 210, 0)
COLOR_WHITE = (245, 245, 245)
COLOR_LIGHT = (205, 205, 205)
COLOR_MUTED = (155, 155, 155)
COLOR_PANEL = (25, 25, 25)
COLOR_SECTION = (34, 34, 34)
COLOR_LINE = (70, 70, 70)


def resolve_model_path():
    """Resolve the MediaPipe face landmarker model without a user-specific path."""
    candidates = []

    env_model = os.environ.get(
        "PHYSIOTRACK_FACE_LANDMARKER_MODEL"
    )

    if env_model:
        candidates.append(
            Path(env_model)
        )

    local_app_data = os.environ.get(
        "LOCALAPPDATA"
    )

    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "physiotrack"
            / "weights"
            / "mediapipe"
            / "face_landmarker.task"
        )

    candidates.append(
        PROJECT_ROOT
        / "weights"
        / "mediapipe"
        / "face_landmarker.task"
    )

    candidates.append(
        Path.home()
        / ".cache"
        / "physiotrack"
        / "weights"
        / "mediapipe"
        / "face_landmarker.task"
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(
        f"  - {candidate}"
        for candidate in candidates
    )

    raise FileNotFoundError(
        "Could not locate face_landmarker.task.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Set PHYSIOTRACK_FACE_LANDMARKER_MODEL "
        "to an explicit model path if needed."
    )


def verify_dataset():
    """Verify the expected 300-W evaluation package."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_ROOT}"
        )

    for split, folder in DATASETS.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset split: {folder}"
            )

        images = sorted(
            folder.glob("*.png")
        )

        annotations = sorted(
            folder.glob("*.pts")
        )

        image_stems = {
            path.stem
            for path in images
        }

        annotation_stems = {
            path.stem
            for path in annotations
        }

        if image_stems != annotation_stems:
            raise RuntimeError(
                f"{split}: image/annotation stems do not match."
            )

        if len(images) != 300:
            raise RuntimeError(
                f"{split}: expected 300 PNG images, "
                f"found {len(images)}"
            )

        if len(annotations) != 300:
            raise RuntimeError(
                f"{split}: expected 300 PTS files, "
                f"found {len(annotations)}"
            )


def load_results():
    """Load and verify the accepted quantitative result CSV."""
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            "Accepted quantitative result file not found:\n"
            f"{RESULTS_CSV}"
        )

    df = pd.read_csv(
        RESULTS_CSV
    )

    required_columns = {
        "split",
        "image",
        "status",
        "nme",
        "nme_percent",
        "mean_pixel_error",
        "interocular_px",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "The accepted result CSV is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if len(df) != 600:
        raise RuntimeError(
            f"Expected 600 result rows, found {len(df)}"
        )

    for split in [
        "Indoor",
        "Outdoor",
    ]:
        split_df = df[
            df["split"]
            == split
        ]

        if len(split_df) != 300:
            raise RuntimeError(
                f"{split}: expected 300 result rows, "
                f"found {len(split_df)}"
            )

    successful = df[
        df["status"]
        == "ok"
    ]

    failed = df[
        df["status"]
        == "failed_detection"
    ]

    if len(successful) != 588:
        raise RuntimeError(
            "Accepted result verification failed: "
            f"expected 588 successful predictions, "
            f"found {len(successful)}"
        )

    if len(failed) != 12:
        raise RuntimeError(
            "Accepted result verification failed: "
            f"expected 12 failed predictions, "
            f"found {len(failed)}"
        )

    mean_nme = float(
        successful[
            "nme_percent"
        ].mean()
    )

    median_nme = float(
        successful[
            "nme_percent"
        ].median()
    )

    std_nme = float(
        successful[
            "nme_percent"
        ].std(
            ddof=0
        )
    )

    expected = {
        "mean": 4.6503,
        "median": 4.3178,
        "std": 1.8048,
    }

    actual = {
        "mean": mean_nme,
        "median": median_nme,
        "std": std_nme,
    }

    for name in expected:
        if abs(
            actual[name]
            - expected[name]
        ) > 0.0001:
            raise RuntimeError(
                "Accepted result verification failed for "
                f"{name} NME: expected {expected[name]:.4f}%, "
                f"found {actual[name]:.4f}%"
            )

    return (
        df,
        {
            "images": len(df),
            "successful": len(successful),
            "failed": len(failed),
            "detection_rate": (
                len(successful)
                / len(df)
                * 100.0
            ),
            "mean_nme": mean_nme,
            "median_nme": median_nme,
            "std_nme": std_nme,
        },
    )


def load_300w_points(path):
    """Load the 68 landmark coordinates from a 300-W .pts file."""
    points = []
    inside = False

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if line == "{":
                inside = True
                continue

            if line == "}":
                break

            if inside and line:
                x, y = map(
                    float,
                    line.split(),
                )

                points.append(
                    (x, y)
                )

    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) != 68:
        raise RuntimeError(
            f"Expected 68 points in {path.name}, "
            f"found {len(points)}"
        )

    points -= 1.0

    return points


def make_gt_face_box(
    points,
    image_width,
    image_height,
):
    """Create the same padded face box used by the quantitative evaluator."""
    x_min = float(
        points[:, 0].min()
    )

    y_min = float(
        points[:, 1].min()
    )

    x_max = float(
        points[:, 0].max()
    )

    y_max = float(
        points[:, 1].max()
    )

    face_width = (
        x_max
        - x_min
    )

    face_height = (
        y_max
        - y_min
    )

    pad_x = (
        0.20
        * face_width
    )

    pad_y = (
        0.20
        * face_height
    )

    return [
        max(
            0.0,
            x_min - pad_x,
        ),
        max(
            0.0,
            y_min - pad_y,
        ),
        min(
            float(
                image_width
                - 1
            ),
            x_max + pad_x,
        ),
        min(
            float(
                image_height
                - 1
            ),
            y_max + pad_y,
        ),
    ]


def compute_face_prominence(
    row,
):
    """Estimate how large and central the annotated face is in the source image."""
    split = str(
        row[
            "split"
        ]
    )

    image_path = (
        DATASETS[
            split
        ]
        / str(
            row[
                "image"
            ]
        )
    )

    pts_path = (
        image_path
        .with_suffix(".pts")
    )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = (
        frame.shape[:2]
    )

    points = load_300w_points(
        pts_path
    )

    x_min = float(
        points[:, 0].min()
    )

    y_min = float(
        points[:, 1].min()
    )

    x_max = float(
        points[:, 0].max()
    )

    y_max = float(
        points[:, 1].max()
    )

    face_width = max(
        1.0,
        x_max
        - x_min,
    )

    face_height = max(
        1.0,
        y_max
        - y_min,
    )

    face_area_ratio = (
        face_width
        * face_height
        / (
            float(width)
            * float(height)
        )
    )

    face_center_x = (
        x_min
        + x_max
    ) / 2.0

    face_center_y = (
        y_min
        + y_max
    ) / 2.0

    dx = (
        face_center_x
        - width / 2.0
    ) / max(
        width / 2.0,
        1.0,
    )

    dy = (
        face_center_y
        - height / 2.0
    ) / max(
        height / 2.0,
        1.0,
    )

    center_distance = float(
        np.sqrt(
            dx * dx
            + dy * dy
        )
    )

    center_score = max(
        0.0,
        1.0
        - min(
            center_distance,
            1.0,
        )
    )

    prominence = (
        0.85
        * face_area_ratio
        + 0.15
        * center_score
    )

    return {
        "face_area_ratio": float(
            face_area_ratio
        ),
        "center_score": float(
            center_score
        ),
        "prominence": float(
            prominence
        ),
    }


def add_prominence_columns(
    df,
):
    """Add deterministic face-size and centrality measures to result rows."""
    rows = []

    for _, row in df.iterrows():
        values = compute_face_prominence(
            row
        )

        record = row.to_dict()

        record.update(
            values
        )

        rows.append(
            record
        )

    return pd.DataFrame(
        rows
    )


def select_cases(df):
    """Select clear qualitative cases while preserving accepted result categories."""
    df = add_prominence_columns(
        df
    )

    cases = []

    for split in [
        "Indoor",
        "Outdoor",
    ]:
        split_df = df[
            df["split"]
            == split
        ].copy()

        successful = split_df[
            split_df["status"]
            == "ok"
        ].copy()

        failed = split_df[
            split_df["status"]
            == "failed_detection"
        ].copy()

        if successful.empty:
            raise RuntimeError(
                f"{split}: no successful predictions are available."
            )

        if failed.empty:
            raise RuntimeError(
                f"{split}: no failed detection is available "
                "for qualitative evidence."
            )

        top_prominent = successful.sort_values(
            [
                "prominence",
                "face_area_ratio",
                "image",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        prominent_count = max(
            20,
            int(
                round(
                    len(top_prominent)
                    * 0.20
                )
            ),
        )

        prominent_pool = top_prominent.head(
            prominent_count
        ).copy()

        strong = prominent_pool.sort_values(
            [
                "nme_percent",
                "prominence",
                "image",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        ).iloc[0]

        median_nme = float(
            successful[
                "nme_percent"
            ].median()
        )

        representative = (
            prominent_pool.assign(
                median_distance=(
                    prominent_pool[
                        "nme_percent"
                    ]
                    - median_nme
                ).abs()
            )
            .sort_values(
                [
                    "median_distance",
                    "prominence",
                    "image",
                ],
                ascending=[
                    True,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        challenging_pool = successful[
            successful[
                "face_area_ratio"
            ]
            >= float(
                successful[
                    "face_area_ratio"
                ].quantile(
                    0.50
                )
            )
        ].copy()

        challenging = challenging_pool.sort_values(
            [
                "nme_percent",
                "prominence",
                "image",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

        if split == "Indoor":
            preferred_failed = failed[
                failed["image"]
                == "indoor_008.png"
            ]

            if len(preferred_failed) != 1:
                raise RuntimeError(
                    "Indoor qualitative failure case "
                    "indoor_008.png was not found exactly once "
                    "in the accepted result CSV."
                )

            failed_case = preferred_failed.iloc[0]

            failed_selection = (
                "Curated accepted failed detection selected for "
                "clear single-face qualitative interpretation"
            )

        else:
            failed_case = failed.sort_values(
                [
                    "prominence",
                    "face_area_ratio",
                    "image",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            ).iloc[0]

            failed_selection = (
                "Most visually prominent accepted failed detection "
                f"in the {split} split"
            )

        prefix = split.lower()

        cases.extend(
            [
                {
                    "role": (
                        f"strong_{prefix}"
                    ),
                    "selection": (
                        "Lowest NME among the most visually prominent "
                        f"successful faces in the {split} split"
                    ),
                    "row": strong,
                },
                {
                    "role": (
                        f"representative_{prefix}"
                    ),
                    "selection": (
                        "NME nearest the split median among the most "
                        f"visually prominent successful faces in the {split} split"
                    ),
                    "row": representative,
                },
                {
                    "role": (
                        f"challenging_{prefix}"
                    ),
                    "selection": (
                        "Highest NME among successful faces with at least "
                        f"median annotated face size in the {split} split"
                    ),
                    "row": challenging,
                },
                {
                    "role": (
                        f"failed_{prefix}"
                    ),
                    "selection": (
                        failed_selection
                    ),
                    "row": failed_case,
                },
            ]
        )

    return cases


def evaluate_case(
    case,
    landmarker,
    mapping,
):
    """Re-run one selected image with the accepted landmark protocol."""
    row = case[
        "row"
    ]

    split = str(
        row[
            "split"
        ]
    )

    image_name = str(
        row[
            "image"
        ]
    )

    folder = DATASETS[
        split
    ]

    image_path = (
        folder
        / image_name
    )

    pts_path = (
        image_path
        .with_suffix(".pts")
    )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = (
        frame.shape[:2]
    )

    gt_68 = load_300w_points(
        pts_path
    )

    gt_51 = (
        gt_68[
            17:
        ]
    )

    face_box = make_gt_face_box(
        gt_68,
        width,
        height,
    )

    landmarks = landmarker.predict_face(
        frame,
        face_box,
    )

    expected_status = str(
        row[
            "status"
        ]
    )

    if expected_status == "failed_detection":
        if landmarks is not None:
            raise RuntimeError(
                "Qualitative verification failed for "
                f"{image_name}: accepted status is "
                "failed_detection, but the selected rerun "
                "returned landmarks."
            )

        return {
            "frame": frame,
            "gt_51": gt_51,
            "predicted": None,
            "face_box": face_box,
            "status": "failed_detection",
            "nme_percent": np.nan,
            "mean_pixel_error": np.nan,
            "interocular_px": np.nan,
        }

    if expected_status != "ok":
        raise RuntimeError(
            f"Unsupported accepted status for {image_name}: "
            f"{expected_status}"
        )

    if landmarks is None:
        raise RuntimeError(
            "Qualitative verification failed for "
            f"{image_name}: accepted status is ok, "
            "but the selected rerun failed."
        )

    predicted = []

    for index in mapping:
        landmark = landmarks[
            index
        ]

        predicted.append(
            (
                landmark.x
                * width,
                landmark.y
                * height,
            )
        )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    if len(predicted) != 51:
        raise RuntimeError(
            f"Expected 51 predicted points, "
            f"found {len(predicted)}"
        )

    point_errors = np.linalg.norm(
        gt_51
        - predicted,
        axis=1,
    )

    interocular = np.linalg.norm(
        gt_51[19]
        - gt_51[28]
    )

    if interocular <= 0:
        raise RuntimeError(
            f"Invalid interocular distance for "
            f"{image_name}"
        )

    mean_pixel_error = float(
        point_errors.mean()
    )

    nme_percent = float(
        mean_pixel_error
        / interocular
        * 100.0
    )

    accepted_nme = float(
        row[
            "nme_percent"
        ]
    )

    if abs(
        nme_percent
        - accepted_nme
    ) > NME_TOLERANCE_PERCENT:
        raise RuntimeError(
            "Qualitative verification failed for "
            f"{image_name}: accepted NME is "
            f"{accepted_nme:.6f}%, selected rerun is "
            f"{nme_percent:.6f}%."
        )

    return {
        "frame": frame,
        "gt_51": gt_51,
        "predicted": predicted,
        "face_box": face_box,
        "status": "ok",
        "nme_percent": nme_percent,
        "mean_pixel_error": mean_pixel_error,
        "interocular_px": float(
            interocular
        ),
    }


def fit_image_to_canvas(frame):
    """Letterbox one source image into the fixed qualitative canvas."""
    source_height, source_width = (
        frame.shape[:2]
    )

    scale = min(
        IMAGE_CANVAS_WIDTH
        / source_width,
        OUTPUT_HEIGHT
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
            OUTPUT_HEIGHT,
            IMAGE_CANVAS_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    offset_x = (
        IMAGE_CANVAS_WIDTH
        - resized_width
    ) // 2

    offset_y = (
        OUTPUT_HEIGHT
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

    return (
        canvas,
        {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
        },
    )


def transform_point(
    point,
    transform,
):
    """Map one source-image point to the qualitative canvas."""
    return (
        int(
            round(
                point[0]
                * transform[
                    "scale"
                ]
                + transform[
                    "offset_x"
                ]
            )
        ),
        int(
            round(
                point[1]
                * transform[
                    "scale"
                ]
                + transform[
                    "offset_y"
                ]
            )
        ),
    )


def transform_box(
    box,
    transform,
):
    """Map one source-image box to the qualitative canvas."""
    point_1 = transform_point(
        (
            box[0],
            box[1],
        ),
        transform,
    )

    point_2 = transform_point(
        (
            box[2],
            box[3],
        ),
        transform,
    )

    return (
        point_1[0],
        point_1[1],
        point_2[0],
        point_2[1],
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
    """Draw one panel section."""
    cv2.rectangle(
        panel,
        (
            12,
            top,
        ),
        (
            PANEL_WIDTH
            - 12,
            bottom,
        ),
        COLOR_SECTION,
        -1,
    )

    cv2.rectangle(
        panel,
        (
            12,
            top,
        ),
        (
            PANEL_WIDTH
            - 12,
            bottom,
        ),
        COLOR_LINE,
        1,
    )


def render_case(
    case,
    result,
    benchmark,
):
    """Render one verified qualitative landmark example."""
    row = case[
        "row"
    ]

    canvas, transform = (
        fit_image_to_canvas(
            result[
                "frame"
            ]
        )
    )

    face_box = transform_box(
        result[
            "face_box"
        ],
        transform,
    )

    cv2.rectangle(
        canvas,
        (
            face_box[0],
            face_box[1],
        ),
        (
            face_box[2],
            face_box[3],
        ),
        COLOR_FACE_BOX,
        2,
        cv2.LINE_AA,
    )

    gt_points = [
        transform_point(
            point,
            transform,
        )
        for point
        in result[
            "gt_51"
        ]
    ]

    predicted_points = None

    if (
        result[
            "predicted"
        ]
        is not None
    ):
        predicted_points = [
            transform_point(
                point,
                transform,
            )
            for point
            in result[
                "predicted"
            ]
        ]

        for gt_point, predicted_point in zip(
            gt_points,
            predicted_points,
        ):
            cv2.line(
                canvas,
                gt_point,
                predicted_point,
                COLOR_VECTOR,
                1,
                cv2.LINE_AA,
            )

    for point in gt_points:
        cv2.circle(
            canvas,
            point,
            4,
            COLOR_GT,
            -1,
            cv2.LINE_AA,
        )

    if predicted_points is not None:
        for point in predicted_points:
            cv2.circle(
                canvas,
                point,
                5,
                COLOR_PREDICTED,
                2,
                cv2.LINE_AA,
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
        "300-W Facial Landmarks",
        24,
        42,
        scale=0.68,
        thickness=2,
    )

    draw_text(
        panel,
        "PhysioTrack qualitative benchmark",
        24,
        72,
        scale=0.43,
        color=COLOR_LIGHT,
    )

    draw_section_box(
        panel,
        98,
        220,
    )

    draw_text(
        panel,
        "QUALITATIVE CASE",
        26,
        124,
        scale=0.40,
        color=COLOR_MUTED,
        thickness=2,
    )

    draw_text(
        panel,
        case[
            "role"
        ],
        26,
        154,
        scale=0.52,
        thickness=2,
    )

    draw_text(
        panel,
        (
            f"Split: "
            f"{row['split']}"
        ),
        26,
        182,
        scale=0.43,
        color=COLOR_LIGHT,
    )

    draw_text(
        panel,
        (
            f"Image: "
            f"{row['image']}"
        ),
        26,
        207,
        scale=0.43,
        color=COLOR_LIGHT,
    )

    draw_section_box(
        panel,
        236,
        428,
    )

    draw_text(
        panel,
        "CONTROLLED LANDMARK PROTOCOL",
        26,
        262,
        scale=0.39,
        color=COLOR_MUTED,
        thickness=2,
    )

    protocol_lines = [
        "Evaluation points: 51",
        "Face border points excluded: 17",
        "Prediction model: MediaPipe 478",
        "Face box: GT-derived + 20% padding",
        "Normalization: outer-eye interocular",
        "Matching: fixed anatomical mapping",
    ]

    y = 294

    for line in protocol_lines:
        draw_text(
            panel,
            line,
            30,
            y,
            scale=0.40,
            color=COLOR_LIGHT,
        )

        y += 24

    draw_section_box(
        panel,
        444,
        674,
    )

    draw_text(
        panel,
        "SELECTED IMAGE RESULT",
        26,
        470,
        scale=0.40,
        color=COLOR_MUTED,
        thickness=2,
    )

    if (
        result[
            "status"
        ]
        == "ok"
    ):
        image_lines = [
            (
                "Status",
                "ok",
            ),
            (
                "Accepted NME",
                (
                    f"{float(row['nme_percent']):.4f}%"
                ),
            ),
            (
                "Verified NME",
                (
                    f"{result['nme_percent']:.4f}%"
                ),
            ),
            (
                "Mean pixel error",
                (
                    f"{result['mean_pixel_error']:.2f} px"
                ),
            ),
            (
                "Interocular distance",
                (
                    f"{result['interocular_px']:.2f} px"
                ),
            ),
            (
                "Face area ratio",
                (
                    f"{float(row['face_area_ratio']) * 100.0:.2f}%"
                ),
            ),
        ]

    else:
        image_lines = [
            (
                "Status",
                "failed_detection",
            ),
            (
                "Accepted NME",
                "N/A",
            ),
            (
                "Verified NME",
                "N/A",
            ),
            (
                "Prediction landmarks",
                "not returned",
            ),
            (
                "Ground-truth points",
                "51 displayed",
            ),
            (
                "Face area ratio",
                (
                    f"{float(row['face_area_ratio']) * 100.0:.2f}%"
                ),
            ),
        ]

    y = 506

    for label, value in image_lines:
        draw_text(
            panel,
            f"{label}:",
            30,
            y,
            scale=0.42,
            color=COLOR_LIGHT,
        )

        draw_text(
            panel,
            value,
            278,
            y,
            scale=0.43,
            thickness=2,
        )

        y += 28

    draw_section_box(
        panel,
        690,
        882,
    )

    draw_text(
        panel,
        "ACCEPTED FULL BENCHMARK RESULTS",
        26,
        716,
        scale=0.39,
        color=COLOR_MUTED,
        thickness=2,
    )

    benchmark_lines = [
        (
            "Images",
            benchmark[
                "images"
            ],
        ),
        (
            "Successful",
            benchmark[
                "successful"
            ],
        ),
        (
            "Failed",
            benchmark[
                "failed"
            ],
        ),
        (
            "Detection rate",
            (
                f"{benchmark['detection_rate']:.2f}%"
            ),
        ),
        (
            "Mean NME",
            (
                f"{benchmark['mean_nme']:.4f}%"
            ),
        ),
        (
            "Median NME",
            (
                f"{benchmark['median_nme']:.4f}%"
            ),
        ),
        (
            "Std NME",
            (
                f"{benchmark['std_nme']:.4f}%"
            ),
        ),
    ]

    y = 752

    for label, value in benchmark_lines:
        draw_text(
            panel,
            f"{label}:",
            30,
            y,
            scale=0.42,
            color=COLOR_LIGHT,
        )

        draw_text(
            panel,
            value,
            278,
            y,
            scale=0.43,
            thickness=2,
        )

        y += 24

    draw_section_box(
        panel,
        898,
        1002,
    )

    draw_text(
        panel,
        "LEGEND",
        26,
        924,
        scale=0.40,
        color=COLOR_MUTED,
        thickness=2,
    )

    legend = [
        (
            COLOR_GT,
            "GT 51-point landmark",
        ),
        (
            COLOR_PREDICTED,
            "Predicted mapped landmark",
        ),
        (
            COLOR_VECTOR,
            "Point-wise error vector",
        ),
        (
            COLOR_FACE_BOX,
            "Evaluation face box",
        ),
    ]

    y = 954

    for color, label in legend:
        cv2.circle(
            panel,
            (
                34,
                y - 4,
            ),
            6,
            color,
            -1,
            cv2.LINE_AA,
        )

        draw_text(
            panel,
            label,
            52,
            y,
            scale=0.35,
            color=COLOR_LIGHT,
        )

        y += 20

    draw_text(
        panel,
        "Controlled component validation; not an official",
        24,
        1038,
        scale=0.34,
        color=COLOR_LIGHT,
    )

    draw_text(
        panel,
        "300-W competition or leaderboard result.",
        24,
        1058,
        scale=0.34,
        color=COLOR_LIGHT,
    )

    return np.hstack(
        (
            canvas,
            panel,
        )
    )


def verify_selected_cases(
    cases,
    benchmark,
):
    """Re-run and verify all selected images before replacing old outputs."""
    model_path = resolve_model_path()

    mapping = (
        get_mediapipe_300w_51()
    )

    if len(mapping) != 51:
        raise RuntimeError(
            f"Expected 51 mapped landmarks, "
            f"found {len(mapping)}"
        )

    if len(set(mapping)) != 51:
        raise RuntimeError(
            "The 51-point mapping contains "
            "duplicate MediaPipe indices."
        )

    print("Model verification:")
    print(
        f"MediaPipe model: "
        f"{model_path.name}"
    )

    print(
        "Selected qualitative cases:"
    )

    landmarker = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    verified = []

    try:
        for case in cases:
            row = case[
                "row"
            ]

            print(
                f"- {case['role']}: "
                f"{row['image']} "
                f"({row['status']})"
            )

            result = evaluate_case(
                case,
                landmarker,
                mapping,
            )

            rendered = render_case(
                case,
                result,
                benchmark,
            )

            verified.append(
                {
                    "case": case,
                    "result": result,
                    "rendered": rendered,
                }
            )

    finally:
        landmarker.close()

    print(
        "\nSelected-case verification passed."
    )

    return verified


def clean_previous_qualitative_outputs():
    """Remove only outputs owned by this qualitative generator."""
    if QUALITATIVE_DIR.exists():
        shutil.rmtree(
            QUALITATIVE_DIR
        )

    QUALITATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ANNOTATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SUMMARY_FIGURE.exists():
        SUMMARY_FIGURE.unlink()


def save_annotated_images(
    verified,
):
    """Save all verified annotated qualitative images."""
    rows = []

    for item in verified:
        case = item[
            "case"
        ]

        result = item[
            "result"
        ]

        row = case[
            "row"
        ]

        output_path = (
            ANNOTATED_DIR
            / (
                f"{case['role']}_"
                f"{Path(str(row['image'])).stem}.png"
            )
        )

        if not cv2.imwrite(
            str(output_path),
            item[
                "rendered"
            ],
        ):
            raise RuntimeError(
                "Could not save qualitative image:\n"
                f"{output_path}"
            )

        rows.append(
            {
                "role": case[
                    "role"
                ],
                "selection": case[
                    "selection"
                ],
                "split": str(
                    row[
                        "split"
                    ]
                ),
                "image": str(
                    row[
                        "image"
                    ]
                ),
                "accepted_status": str(
                    row[
                        "status"
                    ]
                ),
                "verified_status": result[
                    "status"
                ],
                "accepted_nme_percent": (
                    ""
                    if pd.isna(
                        row[
                            "nme_percent"
                        ]
                    )
                    else (
                        f"{float(row['nme_percent']):.6f}"
                    )
                ),
                "verified_nme_percent": (
                    ""
                    if not np.isfinite(
                        result[
                            "nme_percent"
                        ]
                    )
                    else (
                        f"{result['nme_percent']:.6f}"
                    )
                ),
                "mean_pixel_error": (
                    ""
                    if not np.isfinite(
                        result[
                            "mean_pixel_error"
                        ]
                    )
                    else (
                        f"{result['mean_pixel_error']:.6f}"
                    )
                ),
                "interocular_px": (
                    ""
                    if not np.isfinite(
                        result[
                            "interocular_px"
                        ]
                    )
                    else (
                        f"{result['interocular_px']:.6f}"
                    )
                ),
                "face_area_ratio": (
                    f"{float(row['face_area_ratio']):.6f}"
                ),
                "center_score": (
                    f"{float(row['center_score']):.6f}"
                ),
                "prominence": (
                    f"{float(row['prominence']):.6f}"
                ),
                "annotated_image": str(
                    (
                        FINAL_ANNOTATED_DIR
                        / output_path.name
                    ).relative_to(
                        VALIDATION_DIR
                    )
                ).replace(
                    "\\",
                    "/",
                ),
            }
        )

    return rows


def write_selection_csv(
    rows,
):
    """Write the deterministic qualitative selection record."""
    fieldnames = [
        "role",
        "selection",
        "split",
        "image",
        "accepted_status",
        "verified_status",
        "accepted_nme_percent",
        "verified_nme_percent",
        "mean_pixel_error",
        "interocular_px",
        "face_area_ratio",
        "center_score",
        "prominence",
        "annotated_image",
    ]

    with open(
        SELECTION_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def create_summary_figure(
    rows,
):
    """Create a compact 2x4 overview of the annotated qualitative cases."""
    tiles = []

    target_width = 900
    caption_height = 74

    for row in rows:
        image_path = (
            ANNOTATED_DIR
            / Path(
                row[
                    "annotated_image"
                ]
            ).name
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                "Could not read qualitative image:\n"
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
            f"{row['image']}"
        )

        if (
            row[
                "verified_status"
            ]
            == "ok"
        ):
            details = (
                f"NME "
                f"{float(row['verified_nme_percent']):.3f}%"
            )

        else:
            details = (
                "failed_detection"
            )

        cv2.putText(
            tile,
            title,
            (
                18,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
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
                58,
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

    if len(tiles) != 8:
        raise RuntimeError(
            "Expected exactly eight qualitative cases."
        )

    grid_rows = []

    for index in range(
        0,
        8,
        2,
    ):
        grid_rows.append(
            np.hstack(
                (
                    tiles[index],
                    tiles[
                        index + 1
                    ],
                )
            )
        )

    figure = np.vstack(
        grid_rows
    )

    if not cv2.imwrite(
        str(SUMMARY_FIGURE),
        figure,
    ):
        raise RuntimeError(
            "Could not save qualitative summary figure."
        )


def print_preflight(
    benchmark,
    cases,
):
    """Print the verified inputs and selected qualitative roles."""
    print(
        "300-W Facial Landmark Qualitative Benchmark"
    )

    print(
        "==========================================="
    )

    print(
        "\nDataset:"
    )

    print(
        DATASET_ROOT
    )

    print(
        "\nAccepted quantitative results:"
    )

    print(
        f"Images: "
        f"{benchmark['images']}"
    )

    print(
        f"Successful: "
        f"{benchmark['successful']}"
    )

    print(
        f"Failed detections: "
        f"{benchmark['failed']}"
    )

    print(
        f"Detection rate: "
        f"{benchmark['detection_rate']:.2f}%"
    )

    print(
        f"Mean NME: "
        f"{benchmark['mean_nme']:.4f}%"
    )

    print(
        f"Median NME: "
        f"{benchmark['median_nme']:.4f}%"
    )

    print(
        f"Std NME: "
        f"{benchmark['std_nme']:.4f}%"
    )

    print(
        "\nQualitative selection:"
    )

    for case in cases:
        row = case[
            "row"
        ]

        nme = (
            "N/A"
            if pd.isna(
                row[
                    "nme_percent"
                ]
            )
            else (
                f"{float(row['nme_percent']):.4f}%"
            )
        )

        print(
            f"- {case['role']}: "
            f"{row['image']} | "
            f"{row['status']} | "
            f"NME {nme}"
        )

    print(
        "\nThe selected cases will be re-run and verified "
        "before previous qualitative outputs are replaced."
    )

    print(
        "Accepted quantitative artifacts will not be modified."
    )



def validate_staged_qualitative_outputs():
    """Validate newly generated qualitative artifacts before final replacement."""
    if not QUALITATIVE_DIR.is_dir():
        raise RuntimeError(
            "Staged qualitative directory was not created."
        )

    if not ANNOTATED_DIR.is_dir():
        raise RuntimeError(
            "Staged annotated-image directory was not created."
        )

    annotated = sorted(
        ANNOTATED_DIR.glob(
            "*.png"
        )
    )

    if len(annotated) != 8:
        raise RuntimeError(
            "Expected exactly eight staged annotated images."
        )

    for path in annotated:
        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read staged qualitative image: {path}"
            )

        if (
            image.shape[1] != OUTPUT_WIDTH
            or image.shape[0] != OUTPUT_HEIGHT
        ):
            raise RuntimeError(
                "Staged qualitative image has unexpected dimensions: "
                f"{path.name}"
            )

    if not SELECTION_CSV.is_file():
        raise RuntimeError(
            "Staged qualitative selection CSV was not created."
        )

    selection = pd.read_csv(
        SELECTION_CSV
    )

    expected_columns = [
        "role",
        "selection",
        "split",
        "image",
        "accepted_status",
        "verified_status",
        "accepted_nme_percent",
        "verified_nme_percent",
        "mean_pixel_error",
        "interocular_px",
        "face_area_ratio",
        "center_score",
        "prominence",
        "annotated_image",
    ]

    if list(selection.columns) != expected_columns:
        raise RuntimeError(
            "Staged qualitative selection CSV schema is incorrect."
        )

    if len(selection) != 8:
        raise RuntimeError(
            "Expected exactly eight staged qualitative selection rows."
        )

    if selection["role"].duplicated().any():
        raise RuntimeError(
            "Staged qualitative selection contains duplicate roles."
        )

    for _, row in selection.iterrows():
        path = (
            ANNOTATED_DIR
            / Path(
                str(row["annotated_image"])
            ).name
        )

        if not path.is_file():
            raise RuntimeError(
                "Staged qualitative selection references a missing image: "
                f"{path.name}"
            )

        if str(row["accepted_status"]) != str(row["verified_status"]):
            raise RuntimeError(
                "Staged qualitative verification status does not match "
                "the accepted quantitative status."
            )

        if str(row["verified_status"]) == "ok":
            accepted_nme = float(row["accepted_nme_percent"])
            verified_nme = float(row["verified_nme_percent"])
            if abs(accepted_nme - verified_nme) > NME_TOLERANCE_PERCENT:
                raise RuntimeError(
                    "Staged qualitative NME does not match the accepted "
                    "quantitative value."
                )

    if not SUMMARY_FIGURE.is_file():
        raise RuntimeError(
            "Staged qualitative summary figure was not created."
        )

    figure = cv2.imread(
        str(SUMMARY_FIGURE)
    )

    if figure is None:
        raise RuntimeError(
            "Staged qualitative summary figure is unreadable."
        )


def replace_owned_qualitative_outputs(
    staging_dir,
):
    """Replace qualitative-owned outputs with rollback on commit failure."""
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_qualitative = backup_dir / FINAL_QUALITATIVE_DIR.name
    backup_figure = backup_dir / FINAL_SUMMARY_FIGURE.name
    qualitative_backed_up = False
    figure_backed_up = False
    qualitative_installed = False
    figure_installed = False

    try:
        if FINAL_QUALITATIVE_DIR.exists():
            os.replace(FINAL_QUALITATIVE_DIR, backup_qualitative)
            qualitative_backed_up = True

        if FINAL_SUMMARY_FIGURE.exists():
            os.replace(FINAL_SUMMARY_FIGURE, backup_figure)
            figure_backed_up = True

        os.replace(QUALITATIVE_DIR, FINAL_QUALITATIVE_DIR)
        qualitative_installed = True

        os.replace(SUMMARY_FIGURE, FINAL_SUMMARY_FIGURE)
        figure_installed = True

    except Exception:
        if figure_installed and FINAL_SUMMARY_FIGURE.exists():
            FINAL_SUMMARY_FIGURE.unlink()

        if qualitative_installed and FINAL_QUALITATIVE_DIR.exists():
            shutil.rmtree(FINAL_QUALITATIVE_DIR)

        if figure_backed_up and backup_figure.exists():
            os.replace(backup_figure, FINAL_SUMMARY_FIGURE)

        if qualitative_backed_up and backup_qualitative.exists():
            os.replace(backup_qualitative, FINAL_QUALITATIVE_DIR)

        raise


def main():
    global QUALITATIVE_DIR
    global ANNOTATED_DIR
    global SELECTION_CSV
    global SUMMARY_FIGURE

    verify_dataset()

    (
        results,
        benchmark,
    ) = load_results()

    cases = select_cases(
        results
    )

    print_preflight(
        benchmark,
        cases,
    )

    verified = verify_selected_cases(
        cases,
        benchmark,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".300w_landmark_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    QUALITATIVE_DIR = staging_dir / "qualitative"
    ANNOTATED_DIR = QUALITATIVE_DIR / "annotated_images"
    SELECTION_CSV = QUALITATIVE_DIR / FINAL_SELECTION_CSV.name
    staged_figures_dir = staging_dir / "figures"
    SUMMARY_FIGURE = staged_figures_dir / FINAL_SUMMARY_FIGURE.name

    try:
        clean_previous_qualitative_outputs()

        rows = save_annotated_images(
            verified
        )

        write_selection_csv(
            rows
        )

        create_summary_figure(
            rows
        )

        print(
            "\nValidating staged qualitative outputs..."
        )

        validate_staged_qualitative_outputs()

        replace_owned_qualitative_outputs(
            staging_dir
        )

    finally:
        QUALITATIVE_DIR = FINAL_QUALITATIVE_DIR
        ANNOTATED_DIR = FINAL_ANNOTATED_DIR
        SELECTION_CSV = FINAL_SELECTION_CSV
        SUMMARY_FIGURE = FINAL_SUMMARY_FIGURE

        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(
        "\n==========================================="
    )

    print(
        "Qualitative landmark generation completed successfully."
    )

    print(
        "\nAnnotated images:"
    )

    for row in rows:
        print(
            VALIDATION_DIR
            / row[
                "annotated_image"
            ]
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

    print(
        "Committed final qualitative outputs."
    )


if __name__ == "__main__":
    main()
