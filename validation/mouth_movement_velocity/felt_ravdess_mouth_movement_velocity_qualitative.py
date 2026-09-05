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
from physiotrack.face.mouth_motion import MouthMovement
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
ANNOTATED_DIR = QUALITATIVE_DIR / "annotated_transitions"

PER_FRAME_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_movement_velocity_per_frame.csv"
)

PER_ACTOR_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_movement_velocity_per_actor.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_movement_velocity_summary.txt"
)

SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "felt_ravdess_mouth_movement_velocity_qualitative_selection.csv"
)

COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_movement_velocity_qualitative_examples.png"
)

EXPECTED_FRAMES = 158286
EXPECTED_TRANSITIONS = 156846
EXPECTED_ACTORS = 24
EXPECTED_FPS = 30000.0 / 1001.0

RERUN_OPENNESS_TOLERANCE = 1e-5
RERUN_TEMPORAL_TOLERANCE = 1e-10

ROLES = [
    "low_movement",
    "medium_movement",
    "high_movement",
    "very_high_movement",
    "representative_error",
    "challenging_underestimate",
    "challenging_overestimate",
    "high_error_actor_representative",
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


def load_and_validate_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
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

    required_frame_columns = {
        "actor",
        "trial",
        "frame",
        "temporal_status",
        "felt_openness_reference",
        "physiotrack_openness",
        "frame_gap",
        "elapsed_time_sec",
        "felt_mouth_movement",
        "physiotrack_mouth_movement",
        "movement_signed_error",
        "movement_absolute_error",
        "felt_mouth_velocity",
        "physiotrack_mouth_velocity",
        "velocity_signed_error",
        "velocity_absolute_error",
    }

    missing_frame_columns = sorted(
        required_frame_columns
        - set(
            frame_table.columns
        )
    )

    if missing_frame_columns:
        raise RuntimeError(
            "Per-frame results are missing required columns: "
            f"{missing_frame_columns}"
        )

    required_actor_columns = {
        "actor",
        "evaluated_transitions",
        "movement_mae",
    }

    missing_actor_columns = sorted(
        required_actor_columns
        - set(
            actor_table.columns
        )
    )

    if missing_actor_columns:
        raise RuntimeError(
            "Per-actor results are missing required columns: "
            f"{missing_actor_columns}"
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

    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ].copy()

    if len(transitions) != EXPECTED_TRANSITIONS:
        raise RuntimeError(
            "Unexpected evaluated-transition count. "
            f"Expected {EXPECTED_TRANSITIONS}, found {len(transitions)}."
        )

    numeric_columns = [
        "felt_openness_reference",
        "physiotrack_openness",
        "frame_gap",
        "elapsed_time_sec",
        "felt_mouth_movement",
        "physiotrack_mouth_movement",
        "movement_signed_error",
        "movement_absolute_error",
        "felt_mouth_velocity",
        "physiotrack_mouth_velocity",
        "velocity_signed_error",
        "velocity_absolute_error",
    ]

    numeric_values = transitions[
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
            "Accepted temporal transitions contain non-finite values."
        )

    if not np.allclose(
        transitions[
            "frame_gap"
        ].to_numpy(
            dtype=np.float64
        ),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Qualitative validation requires contiguous frame transitions."
        )

    return (
        frame_table,
        actor_table,
    )


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
    actor_table: pd.DataFrame,
) -> list[tuple[str, pd.Series]]:
    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ].copy()

    reference = transitions[
        "felt_mouth_movement"
    ]

    absolute_error = transitions[
        "movement_absolute_error"
    ]

    signed_error = transitions[
        "movement_signed_error"
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

    q10 = float(
        reference.quantile(
            0.10
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

    median_error = float(
        absolute_error.median()
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

    highest_mae_actor = str(
        actor_table.sort_values(
            [
                "movement_mae",
                "actor",
            ],
            ascending=[
                False,
                True,
            ],
        ).iloc[
            0
        ][
            "actor"
        ]
    )

    high_error_actor_pool = transitions[
        transitions[
            "actor"
        ]
        == highest_mae_actor
    ]

    definitions = [
        (
            "low_movement",
            transitions,
            (
                np.abs(
                    reference
                    - q10
                )
                / reference_std
                + 0.20
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
            ),
        ),
        (
            "medium_movement",
            transitions,
            (
                np.abs(
                    reference
                    - q50
                )
                / reference_std
                + 0.20
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
            ),
        ),
        (
            "high_movement",
            transitions,
            (
                np.abs(
                    reference
                    - q90
                )
                / reference_std
                + 0.20
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
            ),
        ),
        (
            "very_high_movement",
            transitions,
            (
                np.abs(
                    reference
                    - q99
                )
                / reference_std
                + 0.20
                * np.abs(
                    absolute_error
                    - median_error
                )
                / error_std
            ),
        ),
        (
            "representative_error",
            transitions,
            (
                np.abs(
                    absolute_error
                    - float(
                        absolute_error.mean()
                    )
                )
                / error_std
                + 0.10
                * np.abs(
                    reference
                    - q50
                )
                / reference_std
            ),
        ),
        (
            "challenging_underestimate",
            transitions,
            (
                np.abs(
                    signed_error
                    - signed_q01
                )
                / error_std
            ),
        ),
        (
            "challenging_overestimate",
            transitions,
            (
                np.abs(
                    signed_error
                    - signed_q99
                )
                / error_std
            ),
        ),
        (
            "high_error_actor_representative",
            high_error_actor_pool,
            (
                np.abs(
                    absolute_error
                    - float(
                        high_error_actor_pool[
                            "movement_absolute_error"
                        ].median()
                    )
                )
                / error_std
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
) -> tuple[
    pd.Series,
    Path,
    Path,
]:
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

    return (
        height
        / width
    )


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

    if (
        not ok
        or frame is None
    ):
        raise RuntimeError(
            "Could not read selected RAVDESS frame: "
            f"{video_path.name}, frame={frame_id}."
        )

    return frame


def landmark_pixel(
    landmark,
    frame: np.ndarray,
) -> tuple[
    int,
    int,
]:
    height, width = frame.shape[
        :2
    ]

    return (
        int(
            round(
                float(
                    landmark.x
                )
                * width
            )
        ),
        int(
            round(
                float(
                    landmark.y
                )
                * height
            )
        ),
    )


def felt_pixel(
    row: pd.Series,
    index: int,
) -> tuple[
    int,
    int,
]:
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


def annotate_frame(
    frame: np.ndarray,
    felt_row: pd.Series,
    landmarks,
    frame_label: str,
    felt_openness: float,
    physiotrack_openness: float,
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

    cv2.line(
        image,
        felt_points[
            48
        ],
        felt_points[
            54
        ],
        FELT_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(
        image,
        felt_points[
            62
        ],
        felt_points[
            66
        ],
        FELT_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    for point in felt_points.values():
        cv2.circle(
            image,
            point,
            5,
            FELT_COLOR,
            -1,
            lineType=cv2.LINE_AA,
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

    cv2.line(
        image,
        physiotrack_points[
            61
        ],
        physiotrack_points[
            291
        ],
        PHYSIOTRACK_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    cv2.line(
        image,
        physiotrack_points[
            13
        ],
        physiotrack_points[
            14
        ],
        PHYSIOTRACK_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    for point in physiotrack_points.values():
        cv2.circle(
            image,
            point,
            5,
            PHYSIOTRACK_COLOR,
            -1,
            lineType=cv2.LINE_AA,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    label_lines = [
        frame_label,
        (
            "FELT openness="
            f"{felt_openness:.4f}"
        ),
        (
            "PhysioTrack openness="
            f"{physiotrack_openness:.4f}"
        ),
    ]

    panel_height = 92

    panel = np.full(
        (
            panel_height,
            image.shape[
                1
            ],
            3,
        ),
        TEXT_BACKGROUND,
        dtype=np.uint8,
    )

    for index, line in enumerate(
        label_lines
    ):
        cv2.putText(
            panel,
            line,
            (
                12,
                24
                + index
                * 27,
            ),
            font,
            0.58,
            TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )

    return np.vstack(
        [
            panel,
            image,
        ]
    )


def combine_transition_pair(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    role: str,
    actor: str,
    trial: str,
    previous_frame_id: int,
    current_frame_id: int,
    felt_movement: float,
    predicted_movement: float,
    felt_velocity: float,
    predicted_velocity: float,
    movement_absolute_error: float,
) -> np.ndarray:
    target_height = min(
        previous_image.shape[
            0
        ],
        current_image.shape[
            0
        ],
    )

    previous_image = previous_image[
        :target_height,
        :,
    ]

    current_image = current_image[
        :target_height,
        :,
    ]

    pair = np.hstack(
        [
            previous_image,
            current_image,
        ]
    )

    lines = [
        (
            f"Role: {role} | {actor} | {trial} | "
            f"transition {previous_frame_id}->{current_frame_id}"
        ),
        (
            "Movement | FELT="
            f"{felt_movement:.4f} | "
            "PhysioTrack="
            f"{predicted_movement:.4f} | "
            "|e|="
            f"{movement_absolute_error:.4f}"
        ),
        (
            "Velocity | FELT="
            f"{felt_velocity:.4f} | "
            "PhysioTrack="
            f"{predicted_velocity:.4f}"
        ),
        (
            "FELT geometry: yellow | PhysioTrack geometry: blue | "
            "FELT FaceRect: green"
        ),
    ]

    panel_height = 118

    panel = np.full(
        (
            panel_height,
            pair.shape[
                1
            ],
            3,
        ),
        TEXT_BACKGROUND,
        dtype=np.uint8,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, line in enumerate(
        lines
    ):
        cv2.putText(
            panel,
            line,
            (
                12,
                24
                + index
                * 27,
            ),
            font,
            0.56,
            TEXT_COLOR,
            1,
            lineType=cv2.LINE_AA,
        )

    return np.vstack(
        [
            panel,
            pair,
        ]
    )


def create_combined_figure(
    annotated_paths: list[
        tuple[
            str,
            Path,
        ]
    ],
    selection_rows: list[
        dict[
            str,
            object,
        ]
    ],
) -> None:
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(
            20.0,
            10.0,
        ),
    )

    for axis, (
        role,
        path,
    ), row in zip(
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

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(
            image
        )

        axis.set_title(
            role.replace(
                "_",
                " "
            ).title()
            + "\n"
            + (
                f"{row['actor']} | "
                f"Ref={float(row['felt_mouth_movement']):.3f} | "
                f"Pred={float(row['rerun_mouth_movement']):.3f} | "
                f"|e|={float(row['rerun_movement_absolute_error']):.3f}"
            ),
            fontsize=9,
        )

        axis.axis(
            "off"
        )

    figure.suptitle(
        "FELT/RAVDESS Mouth Movement and Velocity Qualitative Examples",
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
        "=== FELT/RAVDESS Mouth Movement and Velocity Qualitative Validation ==="
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
        actor_table,
    ) = load_and_validate_results()

    selections = select_examples(
        frame_table,
        actor_table,
    )

    if len(
        selections
    ) != len(
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

    accepted_lookup = frame_table.set_index(
        [
            "actor",
            "trial",
            "frame",
        ],
        verify_integrity=True,
    )

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

            current_frame_id = int(
                accepted_row[
                    "frame"
                ]
            )

            previous_frame_id = (
                current_frame_id
                - 1
            )

            previous_key = (
                actor,
                trial,
                previous_frame_id,
            )

            if previous_key not in accepted_lookup.index:
                raise RuntimeError(
                    "Selected transition is missing its previous frame in "
                    f"{actor}/{trial}: {previous_frame_id}->{current_frame_id}."
                )

            previous_result_row = accepted_lookup.loc[
                previous_key
            ]

            (
                previous_felt_row,
                csv_path,
                video_path,
            ) = resolve_annotation_row(
                actor,
                trial,
                previous_frame_id,
            )

            (
                current_felt_row,
                _,
                _,
            ) = resolve_annotation_row(
                actor,
                trial,
                current_frame_id,
            )

            previous_reference = felt_reference(
                previous_felt_row
            )

            current_reference = felt_reference(
                current_felt_row
            )

            accepted_previous_reference = float(
                previous_result_row[
                    "felt_openness_reference"
                ]
            )

            accepted_current_reference = float(
                accepted_row[
                    "felt_openness_reference"
                ]
            )

            if not math.isclose(
                previous_reference,
                accepted_previous_reference,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Previous-frame FELT reference does not match accepted results."
                )

            if not math.isclose(
                current_reference,
                accepted_current_reference,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "Current-frame FELT reference does not match accepted results."
                )

            previous_frame = read_video_frame(
                video_path,
                previous_frame_id,
            )

            current_frame = read_video_frame(
                video_path,
                current_frame_id,
            )

            previous_box = (
                float(
                    previous_felt_row[
                        "FaceRectX"
                    ]
                ),
                float(
                    previous_felt_row[
                        "FaceRectY"
                    ]
                ),
                float(
                    previous_felt_row[
                        "FaceRectX"
                    ]
                    + previous_felt_row[
                        "FaceRectWidth"
                    ]
                ),
                float(
                    previous_felt_row[
                        "FaceRectY"
                    ]
                    + previous_felt_row[
                        "FaceRectHeight"
                    ]
                ),
            )

            current_box = (
                float(
                    current_felt_row[
                        "FaceRectX"
                    ]
                ),
                float(
                    current_felt_row[
                        "FaceRectY"
                    ]
                ),
                float(
                    current_felt_row[
                        "FaceRectX"
                    ]
                    + current_felt_row[
                        "FaceRectWidth"
                    ]
                ),
                float(
                    current_felt_row[
                        "FaceRectY"
                    ]
                    + current_felt_row[
                        "FaceRectHeight"
                    ]
                ),
            )

            previous_landmarks = landmarker.predict_face(
                previous_frame,
                previous_box,
            )

            current_landmarks = landmarker.predict_face(
                current_frame,
                current_box,
            )

            if previous_landmarks is None:
                raise RuntimeError(
                    "PhysioTrack FaceLandmarks returned None for selected "
                    f"previous frame {actor}/{trial}/{previous_frame_id}."
                )

            if current_landmarks is None:
                raise RuntimeError(
                    "PhysioTrack FaceLandmarks returned None for selected "
                    f"current frame {actor}/{trial}/{current_frame_id}."
                )

            previous_mouth = mouth.predict(
                previous_landmarks,
                image_size=(
                    previous_frame.shape[
                        1
                    ],
                    previous_frame.shape[
                        0
                    ],
                ),
            )

            current_mouth = mouth.predict(
                current_landmarks,
                image_size=(
                    current_frame.shape[
                        1
                    ],
                    current_frame.shape[
                        0
                    ],
                ),
            )

            previous_prediction = previous_mouth[
                "mouth_openness"
            ]

            current_prediction = current_mouth[
                "mouth_openness"
            ]

            if (
                previous_prediction is None
                or current_prediction is None
            ):
                raise RuntimeError(
                    "Selected qualitative transition produced missing "
                    "mouth-openness output."
                )

            previous_prediction = float(
                previous_prediction
            )

            current_prediction = float(
                current_prediction
            )

            accepted_previous_prediction = float(
                previous_result_row[
                    "physiotrack_openness"
                ]
            )

            accepted_current_prediction = float(
                accepted_row[
                    "physiotrack_openness"
                ]
            )

            if not math.isclose(
                previous_prediction,
                accepted_previous_prediction,
                rel_tol=0.0,
                abs_tol=RERUN_OPENNESS_TOLERANCE,
            ):
                raise RuntimeError(
                    "Previous-frame qualitative rerun does not match accepted "
                    f"mouth-openness result for {actor}/{trial}, "
                    f"frame={previous_frame_id}."
                )

            if not math.isclose(
                current_prediction,
                accepted_current_prediction,
                rel_tol=0.0,
                abs_tol=RERUN_OPENNESS_TOLERANCE,
            ):
                raise RuntimeError(
                    "Current-frame qualitative rerun does not match accepted "
                    f"mouth-openness result for {actor}/{trial}, "
                    f"frame={current_frame_id}."
                )

            motion = MouthMovement(
                fps=EXPECTED_FPS
            )

            first_motion = motion.update(
                openness=previous_prediction,
                person_id=0,
            )

            second_motion = motion.update(
                openness=current_prediction,
                person_id=0,
            )

            if (
                abs(
                    float(
                        first_motion[
                            "mouth_movement"
                        ]
                    )
                )
                > RERUN_TEMPORAL_TOLERANCE
                or abs(
                    float(
                        first_motion[
                            "mouth_velocity"
                        ]
                    )
                )
                > RERUN_TEMPORAL_TOLERANCE
            ):
                raise RuntimeError(
                    "MouthMovement initialization semantics changed during "
                    "qualitative verification."
                )

            rerun_movement = float(
                second_motion[
                    "mouth_movement"
                ]
            )

            rerun_velocity = float(
                second_motion[
                    "mouth_velocity"
                ]
            )

            accepted_movement = float(
                accepted_row[
                    "physiotrack_mouth_movement"
                ]
            )

            accepted_velocity = float(
                accepted_row[
                    "physiotrack_mouth_velocity"
                ]
            )

            if not math.isclose(
                rerun_movement,
                accepted_movement,
                rel_tol=0.0,
                abs_tol=RERUN_TEMPORAL_TOLERANCE,
            ):
                raise RuntimeError(
                    "Qualitative MouthMovement rerun does not match accepted "
                    f"movement for {actor}/{trial}, frame={current_frame_id}."
                )

            if not math.isclose(
                rerun_velocity,
                accepted_velocity,
                rel_tol=0.0,
                abs_tol=RERUN_TEMPORAL_TOLERANCE,
            ):
                raise RuntimeError(
                    "Qualitative MouthMovement rerun does not match accepted "
                    f"velocity for {actor}/{trial}, frame={current_frame_id}."
                )

            felt_movement = abs(
                current_reference
                - previous_reference
            )

            felt_velocity = (
                felt_movement
                * EXPECTED_FPS
            )

            accepted_felt_movement = float(
                accepted_row[
                    "felt_mouth_movement"
                ]
            )

            accepted_felt_velocity = float(
                accepted_row[
                    "felt_mouth_velocity"
                ]
            )

            if not math.isclose(
                felt_movement,
                accepted_felt_movement,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Recomputed FELT movement does not match accepted result."
                )

            if not math.isclose(
                felt_velocity,
                accepted_felt_velocity,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    "Recomputed FELT velocity does not match accepted result."
                )

            movement_absolute_error = abs(
                rerun_movement
                - felt_movement
            )

            previous_annotated = annotate_frame(
                previous_frame,
                previous_felt_row,
                previous_landmarks,
                f"Previous frame: {previous_frame_id}",
                previous_reference,
                previous_prediction,
            )

            current_annotated = annotate_frame(
                current_frame,
                current_felt_row,
                current_landmarks,
                f"Current frame: {current_frame_id}",
                current_reference,
                current_prediction,
            )

            combined = combine_transition_pair(
                previous_annotated,
                current_annotated,
                role,
                actor,
                trial,
                previous_frame_id,
                current_frame_id,
                felt_movement,
                rerun_movement,
                felt_velocity,
                rerun_velocity,
                movement_absolute_error,
            )

            output_name = (
                f"{index:02d}_{role}_{actor}_{trial}_"
                f"frames_{previous_frame_id:04d}_{current_frame_id:04d}.png"
            )

            output_path = (
                ANNOTATED_DIR
                / output_name
            )

            if not cv2.imwrite(
                str(
                    output_path
                ),
                combined,
            ):
                raise RuntimeError(
                    f"Could not write qualitative image: {output_path}"
                )

            selection_rows.append(
                {
                    "role": role,
                    "actor": actor,
                    "trial": trial,
                    "previous_frame": previous_frame_id,
                    "current_frame": current_frame_id,
                    "previous_felt_openness": previous_reference,
                    "current_felt_openness": current_reference,
                    "previous_accepted_physiotrack_openness": accepted_previous_prediction,
                    "current_accepted_physiotrack_openness": accepted_current_prediction,
                    "previous_rerun_physiotrack_openness": previous_prediction,
                    "current_rerun_physiotrack_openness": current_prediction,
                    "felt_mouth_movement": felt_movement,
                    "accepted_mouth_movement": accepted_movement,
                    "rerun_mouth_movement": rerun_movement,
                    "accepted_movement_absolute_error": float(
                        accepted_row[
                            "movement_absolute_error"
                        ]
                    ),
                    "rerun_movement_absolute_error": movement_absolute_error,
                    "felt_mouth_velocity": felt_velocity,
                    "accepted_mouth_velocity": accepted_velocity,
                    "rerun_mouth_velocity": rerun_velocity,
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
                f"{role}: {actor}/{trial}, "
                f"frames={previous_frame_id}->{current_frame_id}, "
                f"movement_ref={felt_movement:.4f}, "
                f"movement_pred={rerun_movement:.4f}, "
                f"velocity_ref={felt_velocity:.4f}, "
                f"velocity_pred={rerun_velocity:.4f}, "
                f"abs_error={movement_absolute_error:.4f}"
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
        "Qualitative temporal verification: PASS"
    )

    print(
        f"Selected transitions: {len(selection_rows)}"
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
        f"Annotated transitions: {ANNOTATED_DIR}"
    )


if __name__ == "__main__":
    main()
