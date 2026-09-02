from __future__ import annotations

import csv
import math
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.is_dir():
    sys.path.insert(
        0,
        str(SRC_DIR),
    )

from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.face.mouth import MouthOpenness
from physiotrack.models import Models


FELT_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "FELT"
    / "raw_motion_speech"
)

RAVDESS_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "RAVDESS"
    / "Video_Speech"
)

RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
QUALITATIVE_DIR = RESULTS_DIR / "qualitative"
ANNOTATED_DIR = QUALITATIVE_DIR / "annotated_images"

PER_FRAME_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_openness_per_frame.csv"
)

PER_ACTOR_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_openness_per_actor.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_openness_summary.txt"
)

SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "felt_ravdess_mouth_openness_qualitative_selection.csv"
)

COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_openness_qualitative_examples.png"
)

EXPECTED_FRAMES = 158286
EXPECTED_ACTORS = 24
EXPECTED_MAE = 0.032706
EXPECTED_PEARSON = 0.932273
EXPECTED_CCC = 0.931997
SUMMARY_TOLERANCE = 5e-7
RERUN_TOLERANCE = 1e-5

ROLES = [
    "closed_representative",
    "low_opening",
    "medium_opening",
    "high_opening",
    "very_high_opening",
    "representative_error",
    "challenging_underestimate",
    "challenging_overestimate",
]


FELT_COLOR = (
    0,
    215,
    255,
)

PHYSIOTRACK_COLOR = (
    255,
    200,
    0,
)

FACE_BOX_COLOR = (
    80,
    220,
    80,
)

TEXT_COLOR = (
    255,
    255,
    255,
)

TEXT_BACKGROUND = (
    20,
    20,
    20,
)


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def dataset_inventory(
    root: Path,
) -> dict[str, tuple[int, int]]:
    inventory = {}

    for path in sorted(
        root.rglob("*")
    ):
        if not path.is_file():
            continue

        stat = path.stat()

        inventory[
            path.relative_to(root).as_posix()
        ] = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    return inventory


def clean_owned_outputs() -> None:
    if QUALITATIVE_DIR.exists():
        shutil.rmtree(
            QUALITATIVE_DIR
        )

    if COMBINED_FIGURE_PATH.is_file():
        COMBINED_FIGURE_PATH.unlink()

    ANNOTATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def parse_summary_value(
    label: str,
) -> float:
    require_file(
        SUMMARY_PATH
    )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith(
                f"{label}:"
            ):
                return float(
                    line.split(
                        ":",
                        1,
                    )[1].strip()
                )

    raise RuntimeError(
        f"Required summary value was not found: {label}"
    )


def concordance_correlation_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_mean = float(
        reference.mean()
    )

    prediction_mean = float(
        prediction.mean()
    )

    reference_variance = float(
        reference.var()
    )

    prediction_variance = float(
        prediction.var()
    )

    covariance = float(
        np.mean(
            (
                reference
                - reference_mean
            )
            * (
                prediction
                - prediction_mean
            )
        )
    )

    denominator = (
        reference_variance
        + prediction_variance
        + (
            reference_mean
            - prediction_mean
        ) ** 2
    )

    if denominator <= 0:
        return math.nan

    return float(
        2.0
        * covariance
        / denominator
    )


def load_and_validate_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    for path in (
        PER_FRAME_PATH,
        PER_ACTOR_PATH,
        SUMMARY_PATH,
    ):
        require_file(
            path
        )

    frame_table = pd.read_csv(
        PER_FRAME_PATH
    )

    actor_table = pd.read_csv(
        PER_ACTOR_PATH
    )

    required_columns = {
        "actor",
        "trial",
        "frame",
        "status",
        "felt_reference",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "FaceRectX",
        "FaceRectY",
        "FaceRectWidth",
        "FaceRectHeight",
    }

    missing_columns = sorted(
        required_columns
        - set(
            frame_table.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Per-frame results are missing required columns: "
            f"{missing_columns}"
        )

    if len(frame_table) != EXPECTED_FRAMES:
        raise RuntimeError(
            "Unexpected per-frame result count. "
            f"Expected {EXPECTED_FRAMES}, found {len(frame_table)}."
        )

    if len(actor_table) != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected per-actor result count. "
            f"Expected {EXPECTED_ACTORS}, found {len(actor_table)}."
        )

    if frame_table.duplicated(
        [
            "actor",
            "trial",
            "frame",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in accepted results."
        )

    status_counts = frame_table[
        "status"
    ].value_counts(
        dropna=False
    )

    if (
        len(status_counts) != 1
        or "success" not in status_counts
        or int(
            status_counts[
                "success"
            ]
        )
        != EXPECTED_FRAMES
    ):
        raise RuntimeError(
            "Qualitative generation requires the accepted all-success "
            f"quantitative result set. Found: {status_counts.to_dict()}"
        )

    numeric_columns = [
        "felt_reference",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "FaceRectX",
        "FaceRectY",
        "FaceRectWidth",
        "FaceRectHeight",
    ]

    numeric_values = frame_table[
        numeric_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            numeric_values
        )
    ):
        raise RuntimeError(
            "Accepted per-frame results contain non-finite values."
        )

    reference = frame_table[
        "felt_reference"
    ].to_numpy(
        dtype=np.float64
    )

    prediction = frame_table[
        "physiotrack_openness"
    ].to_numpy(
        dtype=np.float64
    )

    mae = float(
        np.mean(
            np.abs(
                prediction
                - reference
            )
        )
    )

    pearson = float(
        np.corrcoef(
            reference,
            prediction,
        )[0, 1]
    )

    ccc = (
        concordance_correlation_coefficient(
            reference,
            prediction,
        )
    )

    checks = {
        "MAE": (
            mae,
            EXPECTED_MAE,
        ),
        "Pearson r": (
            pearson,
            EXPECTED_PEARSON,
        ),
        "Lin CCC": (
            ccc,
            EXPECTED_CCC,
        ),
    }

    for label, (
        computed,
        expected,
    ) in checks.items():
        summary_value = (
            parse_summary_value(
                label
            )
        )

        if not math.isclose(
            computed,
            summary_value,
            rel_tol=0.0,
            abs_tol=SUMMARY_TOLERANCE,
        ):
            raise RuntimeError(
                "Accepted summary does not match per-frame results for "
                f"{label}: computed={computed}, summary={summary_value}."
            )

        if not math.isclose(
            summary_value,
            expected,
            rel_tol=0.0,
            abs_tol=SUMMARY_TOLERANCE,
        ):
            raise RuntimeError(
                "Accepted benchmark value changed unexpectedly for "
                f"{label}: expected={expected}, found={summary_value}."
            )

    benchmark_metrics = {
        "mae": mae,
        "pearson_r": pearson,
        "ccc": ccc,
    }

    return (
        frame_table,
        actor_table,
        benchmark_metrics,
    )


def normalized_face_area(
    table: pd.DataFrame,
) -> pd.Series:
    area = (
        table[
            "FaceRectWidth"
        ]
        * table[
            "FaceRectHeight"
        ]
    ).astype(
        np.float64
    )

    maximum = float(
        area.max()
    )

    if maximum <= 0:
        return pd.Series(
            np.zeros(
                len(table),
                dtype=np.float64,
            ),
            index=table.index,
        )

    return area / maximum


def choose_candidate(
    pool: pd.DataFrame,
    score: pd.Series,
    used_actors: set[str],
    used_trials: set[str],
) -> pd.Series:
    candidates = pool.copy()

    if candidates.empty:
        raise RuntimeError(
            "Qualitative candidate pool is empty."
        )

    candidates["_score"] = score.loc[
        candidates.index
    ]

    candidates = candidates.sort_values(
        [
            "_score",
            "actor",
            "trial",
            "frame",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    distinct = candidates[
        ~candidates[
            "actor"
        ].isin(
            used_actors
        )
        & ~candidates[
            "trial"
        ].isin(
            used_trials
        )
    ]

    if not distinct.empty:
        return distinct.iloc[
            0
        ]

    trial_distinct = candidates[
        ~candidates[
            "trial"
        ].isin(
            used_trials
        )
    ]

    if not trial_distinct.empty:
        return trial_distinct.iloc[
            0
        ]

    return candidates.iloc[
        0
    ]


def select_examples(
    frame_table: pd.DataFrame,
    benchmark_metrics: dict[str, float],
) -> list[tuple[str, pd.Series]]:
    table = frame_table.copy()

    reference = table[
        "felt_reference"
    ]

    absolute_error = table[
        "absolute_error"
    ]

    signed_error = table[
        "signed_error"
    ]

    reference_std = max(
        float(
            reference.std()
        ),
        1e-12,
    )

    error_std = max(
        float(
            absolute_error.std()
        ),
        1e-12,
    )

    face_prominence = (
        normalized_face_area(
            table
        )
    )

    q05 = float(
        reference.quantile(
            0.05
        )
    )

    q25 = float(
        reference.quantile(
            0.25
        )
    )

    q50 = float(
        reference.quantile(
            0.50
        )
    )

    q90 = float(
        reference.quantile(
            0.90
        )
    )

    q99 = float(
        reference.quantile(
            0.99
        )
    )

    signed_q01 = float(
        signed_error.quantile(
            0.01
        )
    )

    signed_q99 = float(
        signed_error.quantile(
            0.99
        )
    )

    median_error = float(
        absolute_error.median()
    )

    definitions = [
        (
            "closed_representative",
            table[
                reference
                <= q05
            ],
            (
                np.abs(
                    reference
                    - q05
                )
                / reference_std
                + 0.25
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "low_opening",
            table,
            (
                np.abs(
                    reference
                    - q25
                )
                / reference_std
                + 0.25
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "medium_opening",
            table,
            (
                np.abs(
                    reference
                    - q50
                )
                / reference_std
                + 0.25
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "high_opening",
            table,
            (
                np.abs(
                    reference
                    - q90
                )
                / reference_std
                + 0.25
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "very_high_opening",
            table,
            (
                np.abs(
                    reference
                    - q99
                )
                / reference_std
                + 0.25
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "representative_error",
            table,
            (
                np.abs(
                    absolute_error
                    - benchmark_metrics[
                        "mae"
                    ]
                )
                / error_std
                + 0.20
                * np.abs(
                    reference
                    - q50
                )
                / reference_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "challenging_underestimate",
            table,
            (
                np.abs(
                    signed_error
                    - signed_q01
                )
                / error_std
                + 0.10
                * np.abs(
                    reference
                    - q50
                )
                / reference_std
                - 0.05
                * face_prominence
            ),
        ),
        (
            "challenging_overestimate",
            table,
            (
                np.abs(
                    signed_error
                    - signed_q99
                )
                / error_std
                + 0.10
                * np.abs(
                    reference
                    - q50
                )
                / reference_std
                - 0.05
                * face_prominence
            ),
        ),
    ]

    selections = []
    used_actors: set[str] = set()
    used_trials: set[str] = set()

    for role, pool, score in definitions:
        row = choose_candidate(
            pool,
            score,
            used_actors,
            used_trials,
        )

        selections.append(
            (
                role,
                row,
            )
        )

        used_actors.add(
            str(
                row[
                    "actor"
                ]
            )
        )

        used_trials.add(
            str(
                row[
                    "trial"
                ]
            )
        )

    return selections


def resolve_annotation_row(
    actor: str,
    trial: str,
    frame_id: int,
) -> tuple[pd.Series, Path, Path]:
    csv_path = (
        FELT_ROOT
        / actor
        / f"{trial}.csv"
    )

    video_path = (
        RAVDESS_ROOT
        / actor
        / f"{trial}.mp4"
    )

    require_file(
        csv_path
    )

    require_file(
        video_path
    )

    table = pd.read_csv(
        csv_path
    )

    table["frame"] = pd.to_numeric(
        table[
            "frame"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    rows = table[
        table[
            "frame"
        ]
        == frame_id
    ].copy()

    if rows.empty:
        raise RuntimeError(
            "Selected frame is missing from FELT annotation: "
            f"{actor}/{trial}, frame={frame_id}."
        )

    if len(rows) > 1:
        rows["_face_area"] = (
            rows[
                "FaceRectWidth"
            ]
            * rows[
                "FaceRectHeight"
            ]
        )

        rows = rows.sort_values(
            [
                "_face_area",
                "FaceScore",
            ],
            ascending=[
                False,
                False,
            ],
        )

    return (
        rows.iloc[
            0
        ],
        csv_path,
        video_path,
    )


def felt_distance(
    row: pd.Series,
    point_a: int,
    point_b: int,
) -> float:
    return float(
        math.hypot(
            float(
                row[
                    f"x_{point_a}"
                ]
                - row[
                    f"x_{point_b}"
                ]
            ),
            float(
                row[
                    f"y_{point_a}"
                ]
                - row[
                    f"y_{point_b}"
                ]
            ),
        )
    )


def felt_reference(
    row: pd.Series,
) -> float:
    width = felt_distance(
        row,
        48,
        54,
    )

    height = felt_distance(
        row,
        62,
        66,
    )

    if width <= 0:
        raise RuntimeError(
            "Selected FELT mouth width is non-positive."
        )

    return height / width


def read_video_frame(
    video_path: Path,
    frame_id: int,
) -> np.ndarray:
    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not open selected RAVDESS video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_id,
    )

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            "Could not read selected RAVDESS frame: "
            f"{video_path.name}, frame={frame_id}."
        )

    return frame


def landmark_pixel(
    landmark,
    frame: np.ndarray,
) -> tuple[int, int]:
    height, width = frame.shape[:2]

    x = int(
        round(
            float(
                landmark.x
            )
            * width
        )
    )

    y = int(
        round(
            float(
                landmark.y
            )
            * height
        )
    )

    return (
        x,
        y,
    )


def felt_pixel(
    row: pd.Series,
    index: int,
) -> tuple[int, int]:
    return (
        int(
            round(
                float(
                    row[
                        f"x_{index}"
                    ]
                )
            )
        ),
        int(
            round(
                float(
                    row[
                        f"y_{index}"
                    ]
                )
            )
        ),
    )


def draw_point(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.circle(
        image,
        point,
        5,
        color,
        -1,
        lineType=cv2.LINE_AA,
    )


def draw_line(
    image: np.ndarray,
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.line(
        image,
        point_a,
        point_b,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def draw_text_panel(
    image: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    line_height = 25
    padding = 12

    panel_height = (
        padding
        * 2
        + line_height
        * len(
            lines
        )
    )

    panel = np.full(
        (
            panel_height,
            image.shape[1],
            3,
        ),
        TEXT_BACKGROUND,
        dtype=np.uint8,
    )

    for index, line in enumerate(
        lines
    ):
        y = (
            padding
            + 18
            + index
            * line_height
        )

        cv2.putText(
            panel,
            line,
            (
                padding,
                y,
            ),
            font,
            font_scale,
            TEXT_COLOR,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return np.vstack(
        [
            panel,
            image,
        ]
    )


def annotate_example(
    role: str,
    accepted_row: pd.Series,
    felt_row: pd.Series,
    frame: np.ndarray,
    landmarks,
    rerun_prediction: float,
    benchmark_metrics: dict[str, float],
) -> np.ndarray:
    image = frame.copy()

    x1 = int(
        round(
            float(
                felt_row[
                    "FaceRectX"
                ]
            )
        )
    )

    y1 = int(
        round(
            float(
                felt_row[
                    "FaceRectY"
                ]
            )
        )
    )

    x2 = int(
        round(
            float(
                felt_row[
                    "FaceRectX"
                ]
                + felt_row[
                    "FaceRectWidth"
                ]
            )
        )
    )

    y2 = int(
        round(
            float(
                felt_row[
                    "FaceRectY"
                ]
                + felt_row[
                    "FaceRectHeight"
                ]
            )
        )
    )

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
        FACE_BOX_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    felt_points = {
        48: felt_pixel(
            felt_row,
            48,
        ),
        54: felt_pixel(
            felt_row,
            54,
        ),
        62: felt_pixel(
            felt_row,
            62,
        ),
        66: felt_pixel(
            felt_row,
            66,
        ),
    }

    draw_line(
        image,
        felt_points[
            48
        ],
        felt_points[
            54
        ],
        FELT_COLOR,
    )

    draw_line(
        image,
        felt_points[
            62
        ],
        felt_points[
            66
        ],
        FELT_COLOR,
    )

    for point in felt_points.values():
        draw_point(
            image,
            point,
            FELT_COLOR,
        )

    physiotrack_points = {
        61: landmark_pixel(
            landmarks[
                61
            ],
            frame,
        ),
        291: landmark_pixel(
            landmarks[
                291
            ],
            frame,
        ),
        13: landmark_pixel(
            landmarks[
                13
            ],
            frame,
        ),
        14: landmark_pixel(
            landmarks[
                14
            ],
            frame,
        ),
    }

    draw_line(
        image,
        physiotrack_points[
            61
        ],
        physiotrack_points[
            291
        ],
        PHYSIOTRACK_COLOR,
    )

    draw_line(
        image,
        physiotrack_points[
            13
        ],
        physiotrack_points[
            14
        ],
        PHYSIOTRACK_COLOR,
    )

    for point in physiotrack_points.values():
        draw_point(
            image,
            point,
            PHYSIOTRACK_COLOR,
        )

    lines = [
        (
            f"Role: {role} | {accepted_row['actor']} | "
            f"{accepted_row['trial']} | frame {int(accepted_row['frame'])}"
        ),
        (
            "FELT reference: "
            f"{float(accepted_row['felt_reference']):.4f} | "
            "PhysioTrack: "
            f"{rerun_prediction:.4f} | "
            "absolute error: "
            f"{float(accepted_row['absolute_error']):.4f}"
        ),
        (
            "FELT geometry: yellow | PhysioTrack geometry: blue | "
            "FELT FaceRect initialization: green"
        ),
        (
            "Full benchmark: "
            f"MAE={benchmark_metrics['mae']:.4f}, "
            f"Pearson r={benchmark_metrics['pearson_r']:.4f}, "
            f"Lin CCC={benchmark_metrics['ccc']:.4f}"
        ),
    ]

    return draw_text_panel(
        image,
        lines,
    )


def create_combined_figure(
    annotated_paths: list[tuple[str, Path]],
    selection_rows: list[dict[str, object]],
) -> None:
    if len(annotated_paths) != len(selection_rows):
        raise RuntimeError(
            "Qualitative combined-figure inputs have inconsistent lengths."
        )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16.0, 8.5),
    )

    text_panel_height = (
        12
        * 2
        + 25
        * 4
    )

    for axis, (
        role,
        path,
    ), selection_row in zip(
        axes.flat,
        annotated_paths,
        selection_rows,
    ):
        image = cv2.imread(
            str(
                path
            )
        )

        if image is None:
            raise RuntimeError(
                f"Could not read generated qualitative image: {path}"
            )

        if image.shape[0] <= text_panel_height:
            raise RuntimeError(
                "Generated qualitative image is too short for the expected "
                f"metadata panel: {path}"
            )

        image = image[
            text_panel_height:,
            :,
        ]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(
            image
        )

        role_title = role.replace(
            "_",
            " "
        ).title()

        reference = float(
            selection_row[
                "felt_reference"
            ]
        )

        prediction = float(
            selection_row[
                "rerun_physiotrack_openness"
            ]
        )

        absolute_error = float(
            selection_row[
                "rerun_absolute_error"
            ]
        )

        axis.set_title(
            f"{role_title}\n"
            f"{selection_row['actor']} | "
            f"Ref={reference:.3f} | "
            f"Pred={prediction:.3f} | "
            f"|e|={absolute_error:.3f}",
            fontsize=9,
        )

        axis.axis(
            "off"
        )

    figure.suptitle(
        "FELT/RAVDESS Mouth-Openness Qualitative Benchmark Examples",
        fontsize=14,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.955,
        )
    )

    figure.savefig(
        COMBINED_FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    print(
        "=== FELT/RAVDESS Mouth Openness Qualitative Validation ==="
    )

    if not FELT_ROOT.is_dir():
        raise FileNotFoundError(
            "FELT speech annotations were not found at the expected "
            "project-relative location: datasets/FELT/raw_motion_speech"
        )

    if not RAVDESS_ROOT.is_dir():
        raise FileNotFoundError(
            "RAVDESS speech videos were not found at the expected "
            "project-relative location: datasets/RAVDESS/Video_Speech"
        )

    (
        frame_table,
        _,
        benchmark_metrics,
    ) = load_and_validate_results()

    selections = select_examples(
        frame_table,
        benchmark_metrics,
    )

    if len(selections) != len(
        ROLES
    ):
        raise RuntimeError(
            "Unexpected qualitative selection count."
        )

    selected_roles = [
        role
        for role, _ in selections
    ]

    if selected_roles != ROLES:
        raise RuntimeError(
            "Qualitative selection roles do not match the locked role order."
        )

    felt_before = dataset_inventory(
        FELT_ROOT
    )

    ravdess_before = dataset_inventory(
        RAVDESS_ROOT
    )

    model_path = Path(
        Models.resolve(
            Models.Face.MediaPipe.Landmarks.face_landmarker
        )
    ).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            "PhysioTrack MediaPipe face-landmarker model could not be resolved."
        )

    clean_owned_outputs()

    landmarker = FaceLandmarks(
        model_path=model_path,
        num_faces=1,
    )

    mouth = MouthOpenness()

    selection_rows = []
    annotated_paths = []

    try:
        for index, (
            role,
            accepted_row,
        ) in enumerate(
            selections,
            start=1,
        ):
            actor = str(
                accepted_row[
                    "actor"
                ]
            )

            trial = str(
                accepted_row[
                    "trial"
                ]
            )

            frame_id = int(
                accepted_row[
                    "frame"
                ]
            )

            (
                felt_row,
                csv_path,
                video_path,
            ) = resolve_annotation_row(
                actor,
                trial,
                frame_id,
            )

            recomputed_reference = felt_reference(
                felt_row
            )

            if not math.isclose(
                recomputed_reference,
                float(
                    accepted_row[
                        "felt_reference"
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Selected FELT reference does not match accepted CSV for "
                    f"{actor}/{trial}, frame={frame_id}."
                )

            frame = read_video_frame(
                video_path,
                frame_id,
            )

            box = (
                float(
                    felt_row[
                        "FaceRectX"
                    ]
                ),
                float(
                    felt_row[
                        "FaceRectY"
                    ]
                ),
                float(
                    felt_row[
                        "FaceRectX"
                    ]
                    + felt_row[
                        "FaceRectWidth"
                    ]
                ),
                float(
                    felt_row[
                        "FaceRectY"
                    ]
                    + felt_row[
                        "FaceRectHeight"
                    ]
                ),
            )

            landmarks = landmarker.predict_face(
                frame,
                box,
            )

            if landmarks is None:
                raise RuntimeError(
                    "Qualitative verification failed because PhysioTrack "
                    "FaceLandmarks returned None for selected frame "
                    f"{actor}/{trial}, frame={frame_id}."
                )

            result = mouth.predict(
                landmarks,
                image_size=(
                    frame.shape[1],
                    frame.shape[0],
                ),
            )

            prediction = result[
                "mouth_openness"
            ]

            if prediction is None or not math.isfinite(
                float(
                    prediction
                )
            ):
                raise RuntimeError(
                    "Qualitative verification produced invalid mouth openness for "
                    f"{actor}/{trial}, frame={frame_id}."
                )

            rerun_prediction = float(
                prediction
            )

            accepted_prediction = float(
                accepted_row[
                    "physiotrack_openness"
                ]
            )

            if not math.isclose(
                rerun_prediction,
                accepted_prediction,
                rel_tol=0.0,
                abs_tol=RERUN_TOLERANCE,
            ):
                raise RuntimeError(
                    "Qualitative rerun does not match accepted quantitative "
                    f"prediction for {actor}/{trial}, frame={frame_id}: "
                    f"accepted={accepted_prediction}, rerun={rerun_prediction}."
                )

            annotated = annotate_example(
                role,
                accepted_row,
                felt_row,
                frame,
                landmarks,
                rerun_prediction,
                benchmark_metrics,
            )

            output_name = (
                f"{index:02d}_{role}_{actor}_{trial}_frame_{frame_id:04d}.png"
            )

            output_path = (
                ANNOTATED_DIR
                / output_name
            )

            if not cv2.imwrite(
                str(
                    output_path
                ),
                annotated,
            ):
                raise RuntimeError(
                    f"Could not write qualitative image: {output_path}"
                )

            generated_absolute_error = abs(
                rerun_prediction
                - recomputed_reference
            )

            selection_rows.append(
                {
                    "role": role,
                    "actor": actor,
                    "trial": trial,
                    "frame": frame_id,
                    "felt_reference": recomputed_reference,
                    "accepted_physiotrack_openness": accepted_prediction,
                    "rerun_physiotrack_openness": rerun_prediction,
                    "accepted_signed_error": float(
                        accepted_row[
                            "signed_error"
                        ]
                    ),
                    "accepted_absolute_error": float(
                        accepted_row[
                            "absolute_error"
                        ]
                    ),
                    "rerun_absolute_error": generated_absolute_error,
                    "FaceRectX": float(
                        felt_row[
                            "FaceRectX"
                        ]
                    ),
                    "FaceRectY": float(
                        felt_row[
                            "FaceRectY"
                        ]
                    ),
                    "FaceRectWidth": float(
                        felt_row[
                            "FaceRectWidth"
                        ]
                    ),
                    "FaceRectHeight": float(
                        felt_row[
                            "FaceRectHeight"
                        ]
                    ),
                    "source_csv": csv_path.relative_to(
                        WORKSPACE_ROOT
                    ).as_posix(),
                    "source_video": video_path.relative_to(
                        WORKSPACE_ROOT
                    ).as_posix(),
                    "output_image": output_path.relative_to(
                        SCRIPT_DIR
                    ).as_posix(),
                }
            )

            annotated_paths.append(
                (
                    role,
                    output_path,
                )
            )

            print(
                f"{role}: {actor}/{trial}, frame={frame_id}, "
                f"reference={recomputed_reference:.4f}, "
                f"prediction={rerun_prediction:.4f}, "
                f"abs_error={generated_absolute_error:.4f}"
            )

    finally:
        landmarker.close()

    with SELECTION_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                selection_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            selection_rows
        )

    create_combined_figure(
        annotated_paths,
        selection_rows,
    )

    felt_after = dataset_inventory(
        FELT_ROOT
    )

    ravdess_after = dataset_inventory(
        RAVDESS_ROOT
    )

    if felt_before != felt_after:
        raise RuntimeError(
            "FELT dataset integrity check failed during qualitative generation."
        )

    if ravdess_before != ravdess_after:
        raise RuntimeError(
            "RAVDESS dataset integrity check failed during qualitative generation."
        )

    print()
    print(
        "Qualitative verification: PASS"
    )

    print(
        f"Selected examples: {len(selection_rows)}"
    )

    print(
        "FELT dataset integrity: PASS"
    )

    print(
        "RAVDESS dataset integrity: PASS"
    )

    print(
        f"Saved: {SELECTION_CSV_PATH}"
    )

    print(
        f"Saved: {COMBINED_FIGURE_PATH}"
    )

    print(
        f"Annotated images: {ANNOTATED_DIR}"
    )


if __name__ == "__main__":
    main()
