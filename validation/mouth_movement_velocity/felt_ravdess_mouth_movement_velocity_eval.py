from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

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

from physiotrack.face.mouth_motion import MouthMovement


FELT_ROOT = (
    WORKSPACE_ROOT
    / "datasets"
    / "FELT"
    / "raw_motion_speech"
)

MOUTH_OPENNESS_RESULTS_DIR = (
    REPO_ROOT
    / "validation"
    / "mouth_openness"
    / "results"
)

MOUTH_OPENNESS_PER_FRAME_PATH = (
    MOUTH_OPENNESS_RESULTS_DIR
    / "felt_ravdess_mouth_openness_per_frame.csv"
)

OUTPUT_DIR = SCRIPT_DIR / "results"

PER_FRAME_OUTPUT_PATH = (
    OUTPUT_DIR
    / "felt_ravdess_mouth_movement_velocity_per_frame.csv"
)

PER_ACTOR_OUTPUT_PATH = (
    OUTPUT_DIR
    / "felt_ravdess_mouth_movement_velocity_per_actor.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "felt_ravdess_mouth_movement_velocity_summary.txt"
)

EXPECTED_ACTORS = [
    f"Actor_{index:02d}"
    for index in range(1, 25)
]
EXPECTED_TRIALS_PER_ACTOR = 60
EXPECTED_TOTAL_TRIALS = 1440
EXPECTED_UNIQUE_ANNOTATED_FRAMES = 158286
EXPECTED_DUPLICATE_ROWS_RESOLVED = 2
EXPECTED_FPS = 30000.0 / 1001.0

REFERENCE_TOLERANCE = 1e-9
FIRST_FRAME_ZERO_TOLERANCE = 1e-12

REQUIRED_FELT_COLUMNS = {
    "frame",
    "FaceRectWidth",
    "FaceRectHeight",
    "FaceScore",
    "x_48",
    "y_48",
    "x_54",
    "y_54",
    "x_62",
    "y_62",
}

REQUIRED_OPENNESS_RESULT_COLUMNS = {
    "actor",
    "trial",
    "frame",
    "status",
    "felt_reference",
    "physiotrack_openness",
}


def distance_from_row(
    row: pd.Series,
    point_a: int,
    point_b: int,
) -> float:
    dx = float(
        row[f"x_{point_a}"]
        - row[f"x_{point_b}"]
    )

    dy = float(
        row[f"y_{point_a}"]
        - row[f"y_{point_b}"]
    )

    return math.hypot(
        dx,
        dy,
    )


def resolve_duplicate_frames(
    frame_table: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    table = frame_table.copy()

    table["_face_area"] = (
        table["FaceRectWidth"]
        * table["FaceRectHeight"]
    )

    unique_frame_count = int(
        table["frame"].nunique()
    )

    duplicate_rows_resolved = int(
        len(table)
        - unique_frame_count
    )

    table = (
        table.sort_values(
            [
                "frame",
                "_face_area",
                "FaceScore",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=["frame"],
            keep="first",
        )
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )

    return (
        table,
        duplicate_rows_resolved,
    )


def felt_reference(
    row: pd.Series,
) -> float:
    mouth_width = distance_from_row(
        row,
        48,
        54,
    )

    mouth_height = distance_from_row(
        row,
        62,
        66,
    )

    if (
        not math.isfinite(
            mouth_width
        )
        or mouth_width <= 0
    ):
        raise ValueError(
            "FELT mouth-corner width is invalid."
        )

    reference = (
        mouth_height
        / mouth_width
    )

    if not math.isfinite(
        reference
    ):
        raise ValueError(
            "FELT mouth-openness reference is non-finite."
        )

    return float(
        reference
    )


def pearson_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if (
        reference.size < 2
        or prediction.size < 2
        or np.std(reference) <= 0
        or np.std(prediction) <= 0
    ):
        return math.nan

    return float(
        np.corrcoef(
            reference,
            prediction,
        )[0, 1]
    )


def spearman_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_ranks = (
        pd.Series(
            reference,
            dtype="float64",
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    prediction_ranks = (
        pd.Series(
            prediction,
            dtype="float64",
        )
        .rank(
            method="average"
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    return pearson_correlation(
        reference_ranks,
        prediction_ranks,
    )


def concordance_correlation_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if (
        reference.size < 2
        or prediction.size < 2
    ):
        return math.nan

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


def regression_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if (
        reference.size == 0
        or prediction.size == 0
        or reference.size != prediction.size
    ):
        raise ValueError(
            "Regression metrics require non-empty paired arrays."
        )

    if (
        not np.all(
            np.isfinite(
                reference
            )
        )
        or not np.all(
            np.isfinite(
                prediction
            )
        )
    ):
        raise ValueError(
            "Regression metrics require finite paired arrays."
        )

    signed_error = (
        prediction
        - reference
    )

    absolute_error = np.abs(
        signed_error
    )

    return {
        "mae": float(
            absolute_error.mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    signed_error
                    ** 2
                )
            )
        ),
        "median_absolute_error": float(
            np.median(
                absolute_error
            )
        ),
        "std_absolute_error": float(
            absolute_error.std()
        ),
        "p90_absolute_error": float(
            np.percentile(
                absolute_error,
                90,
            )
        ),
        "p95_absolute_error": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),
        "mean_signed_error": float(
            signed_error.mean()
        ),
        "pearson_r": pearson_correlation(
            reference,
            prediction,
        ),
        "spearman_rho": spearman_correlation(
            reference,
            prediction,
        ),
        "ccc": concordance_correlation_coefficient(
            reference,
            prediction,
        ),
    }


def format_metric(
    value: float,
) -> str:
    if math.isfinite(
        float(value)
    ):
        return f"{float(value):.6f}"

    return "nan"


def validate_dataset_structure() -> tuple[
    list[dict[str, object]],
    int,
]:
    if not FELT_ROOT.is_dir():
        raise FileNotFoundError(
            "FELT speech annotations were not found at the expected "
            "project-relative location:\n"
            "datasets/FELT/raw_motion_speech"
        )

    actor_dirs = sorted(
        path.name
        for path in FELT_ROOT.iterdir()
        if path.is_dir()
    )

    if actor_dirs != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected FELT actor structure. "
            f"Expected {EXPECTED_ACTORS}, found {actor_dirs}."
        )

    trials = []
    total_unique_frames = 0
    total_duplicate_rows_resolved = 0

    for actor in EXPECTED_ACTORS:
        actor_dir = (
            FELT_ROOT
            / actor
        )

        csv_paths = sorted(
            actor_dir.glob(
                "*.csv"
            )
        )

        if (
            len(csv_paths)
            != EXPECTED_TRIALS_PER_ACTOR
        ):
            raise RuntimeError(
                f"Unexpected FELT trial count for {actor}. "
                f"Expected {EXPECTED_TRIALS_PER_ACTOR}, "
                f"found {len(csv_paths)}."
            )

        for csv_path in csv_paths:
            table = pd.read_csv(
                csv_path
            )

            missing_columns = sorted(
                REQUIRED_FELT_COLUMNS
                - set(
                    table.columns
                )
            )

            if missing_columns:
                raise RuntimeError(
                    "FELT annotation is missing required columns in "
                    f"{actor}/{csv_path.name}: {missing_columns}"
                )

            frame_values = pd.to_numeric(
                table["frame"],
                errors="coerce",
            ).to_numpy(
                dtype=np.float64
            )

            if not np.all(
                np.isfinite(
                    frame_values
                )
            ):
                raise RuntimeError(
                    "FELT frame IDs contain non-finite values in "
                    f"{actor}/{csv_path.name}."
                )

            if not np.all(
                frame_values
                == np.floor(
                    frame_values
                )
            ):
                raise RuntimeError(
                    "FELT frame IDs must be integers in "
                    f"{actor}/{csv_path.name}."
                )

            table["frame"] = (
                frame_values.astype(
                    np.int64
                )
            )

            (
                table,
                duplicate_rows_resolved,
            ) = resolve_duplicate_frames(
                table
            )

            frame_ids = table[
                "frame"
            ].to_numpy(
                dtype=np.int64
            )

            expected_frame_ids = np.arange(
                len(table),
                dtype=np.int64,
            )

            if not np.array_equal(
                frame_ids,
                expected_frame_ids,
            ):
                raise RuntimeError(
                    "FELT frame IDs must be contiguous and zero-based "
                    "after duplicate resolution in "
                    f"{actor}/{csv_path.name}."
                )

            references = np.asarray(
                [
                    felt_reference(
                        row
                    )
                    for _, row in table.iterrows()
                ],
                dtype=np.float64,
            )

            trials.append(
                {
                    "actor": actor,
                    "trial": csv_path.stem,
                    "csv_path": csv_path,
                    "frame_ids": frame_ids,
                    "references": references,
                    "annotated_frames": len(
                        table
                    ),
                }
            )

            total_unique_frames += len(
                table
            )

            total_duplicate_rows_resolved += (
                duplicate_rows_resolved
            )

    if len(trials) != EXPECTED_TOTAL_TRIALS:
        raise RuntimeError(
            "Unexpected FELT trial count. "
            f"Expected {EXPECTED_TOTAL_TRIALS}, found {len(trials)}."
        )

    if (
        total_unique_frames
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Unexpected unique FELT frame count. "
            f"Expected {EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found {total_unique_frames}."
        )

    if (
        total_duplicate_rows_resolved
        != EXPECTED_DUPLICATE_ROWS_RESOLVED
    ):
        raise RuntimeError(
            "Unexpected duplicate-row count. "
            f"Expected {EXPECTED_DUPLICATE_ROWS_RESOLVED}, "
            f"found {total_duplicate_rows_resolved}."
        )

    return (
        trials,
        total_duplicate_rows_resolved,
    )


def load_mouth_openness_results() -> pd.DataFrame:
    if not MOUTH_OPENNESS_PER_FRAME_PATH.is_file():
        raise FileNotFoundError(
            "Accepted mouth-openness per-frame results were not found at:\n"
            "validation/mouth_openness/results/"
            "felt_ravdess_mouth_openness_per_frame.csv\n"
            "Run the accepted mouth-openness evaluator first."
        )

    table = pd.read_csv(
        MOUTH_OPENNESS_PER_FRAME_PATH
    )

    missing_columns = sorted(
        REQUIRED_OPENNESS_RESULT_COLUMNS
        - set(
            table.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Mouth-openness per-frame results are missing required columns: "
            f"{missing_columns}"
        )

    if len(table) != EXPECTED_UNIQUE_ANNOTATED_FRAMES:
        raise RuntimeError(
            "Unexpected mouth-openness per-frame result count. "
            f"Expected {EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found {len(table)}."
        )

    if table.duplicated(
        [
            "actor",
            "trial",
            "frame",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in the "
            "mouth-openness result file."
        )

    status_counts = table[
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
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Mouth-movement validation requires the accepted all-success "
            "mouth-openness result set. Found: "
            f"{status_counts.to_dict()}"
        )

    numeric_columns = [
        "felt_reference",
        "physiotrack_openness",
    ]

    numeric_values = table[
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
            "Mouth-openness per-frame results contain non-finite required values."
        )

    table["frame"] = pd.to_numeric(
        table["frame"],
        errors="raise",
    ).astype(
        np.int64
    )

    return table


def main() -> None:
    print(
        "=== FELT/RAVDESS Mouth Movement and Velocity Evaluation ==="
    )

    print(
        "Validation target: PhysioTrack MouthMovement"
    )

    print(
        "Prediction input: accepted PhysioTrack mouth-openness outputs"
    )

    print(
        "Ground-truth openness: FELT d(62,66) / d(48,54)"
    )

    print(
        "Ground-truth movement: absolute frame-to-frame openness change"
    )

    print(
        "Ground-truth velocity: movement divided by elapsed frame time"
    )

    print(
        f"FPS: {EXPECTED_FPS:.12f}"
    )

    print()

    print(
        "Running dataset and source-result preflight..."
    )

    (
        trials,
        duplicate_rows_resolved,
    ) = validate_dataset_structure()

    openness_table = (
        load_mouth_openness_results()
    )

    openness_lookup = (
        openness_table.set_index(
            [
                "actor",
                "trial",
                "frame",
            ],
            verify_integrity=True,
        )
    )

    expected_keys = {
        (
            str(
                row.actor
            ),
            str(
                row.trial
            ),
            int(
                row.frame
            ),
        )
        for row in openness_table.itertuples(
            index=False
        )
    }

    felt_keys = set()

    for trial in trials:
        actor = str(
            trial["actor"]
        )

        trial_name = str(
            trial["trial"]
        )

        for frame_id in np.asarray(
            trial["frame_ids"],
            dtype=np.int64,
        ):
            felt_keys.add(
                (
                    actor,
                    trial_name,
                    int(
                        frame_id
                    ),
                )
            )

    if felt_keys != expected_keys:
        missing = sorted(
            felt_keys
            - expected_keys
        )

        extra = sorted(
            expected_keys
            - felt_keys
        )

        raise RuntimeError(
            "FELT annotations and accepted mouth-openness results do not "
            "contain identical actor/trial/frame keys. "
            f"Missing result keys={missing[:5]}, "
            f"extra result keys={extra[:5]}."
        )

    print(
        "Preflight: PASS"
    )

    print(
        f"Actors: {len(EXPECTED_ACTORS)}"
    )

    print(
        f"Trials: {len(trials)}"
    )

    print(
        "Unique annotated frames: "
        f"{EXPECTED_UNIQUE_ANNOTATED_FRAMES}"
    )

    print(
        "Duplicate annotation rows resolved: "
        f"{duplicate_rows_resolved}"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    owned_outputs = (
        PER_FRAME_OUTPUT_PATH,
        PER_ACTOR_OUTPUT_PATH,
        SUMMARY_OUTPUT_PATH,
    )

    removed_outputs = []

    for output_path in owned_outputs:
        if output_path.is_file():
            output_path.unlink()
            removed_outputs.append(
                output_path.name
            )

    if removed_outputs:
        print(
            "Removed previous quantitative outputs: "
            + ", ".join(
                removed_outputs
            )
        )
    else:
        print(
            "Previous quantitative outputs: none"
        )

    print()

    frame_results = []

    movement_reference_values = []
    movement_prediction_values = []
    velocity_reference_values = []
    velocity_prediction_values = []

    actor_accumulators = {
        actor: {
            "frames": 0,
            "initialization_frames": 0,
            "evaluated_transitions": 0,
            "movement_reference": [],
            "movement_prediction": [],
            "velocity_reference": [],
            "velocity_prediction": [],
        }
        for actor in EXPECTED_ACTORS
    }

    total_initialization_frames = 0
    total_evaluated_transitions = 0

    start_time = time.time()

    for trial_index, trial in enumerate(
        trials,
        start=1,
    ):
        actor = str(
            trial["actor"]
        )

        trial_name = str(
            trial["trial"]
        )

        frame_ids = np.asarray(
            trial["frame_ids"],
            dtype=np.int64,
        )

        references = np.asarray(
            trial["references"],
            dtype=np.float64,
        )

        motion = MouthMovement(
            fps=EXPECTED_FPS
        )

        previous_reference = None
        previous_frame_id = None

        for sequence_index, (
            frame_id,
            reference_openness,
        ) in enumerate(
            zip(
                frame_ids,
                references,
            )
        ):
            key = (
                actor,
                trial_name,
                int(
                    frame_id
                ),
            )

            source_row = openness_lookup.loc[
                key
            ]

            stored_reference = float(
                source_row[
                    "felt_reference"
                ]
            )

            prediction_openness = float(
                source_row[
                    "physiotrack_openness"
                ]
            )

            if not math.isclose(
                float(
                    reference_openness
                ),
                stored_reference,
                rel_tol=0.0,
                abs_tol=REFERENCE_TOLERANCE,
            ):
                raise RuntimeError(
                    "Independent FELT reference does not match the accepted "
                    "mouth-openness result for "
                    f"{actor}/{trial_name}, frame={frame_id}: "
                    f"independent={reference_openness}, "
                    f"stored={stored_reference}."
                )

            motion_result = motion.update(
                openness=prediction_openness,
                person_id=0,
            )

            prediction_movement = float(
                motion_result[
                    "mouth_movement"
                ]
            )

            prediction_velocity = float(
                motion_result[
                    "mouth_velocity"
                ]
            )

            actor_accumulators[
                actor
            ]["frames"] += 1

            if sequence_index == 0:
                temporal_status = (
                    "initialization"
                )

                reference_movement = math.nan
                reference_velocity = math.nan
                movement_signed_error = math.nan
                movement_absolute_error = math.nan
                velocity_signed_error = math.nan
                velocity_absolute_error = math.nan
                frame_gap = math.nan
                elapsed_time = math.nan

                total_initialization_frames += 1
                actor_accumulators[
                    actor
                ]["initialization_frames"] += 1

                if (
                    abs(
                        prediction_movement
                    )
                    > FIRST_FRAME_ZERO_TOLERANCE
                    or abs(
                        prediction_velocity
                    )
                    > FIRST_FRAME_ZERO_TOLERANCE
                ):
                    raise RuntimeError(
                        "MouthMovement initialization semantics changed for "
                        f"{actor}/{trial_name}: "
                        f"movement={prediction_movement}, "
                        f"velocity={prediction_velocity}."
                    )

            else:
                temporal_status = (
                    "evaluated_transition"
                )

                frame_gap = int(
                    frame_id
                    - previous_frame_id
                )

                if frame_gap <= 0:
                    raise RuntimeError(
                        "Non-positive frame gap in "
                        f"{actor}/{trial_name}, frame={frame_id}."
                    )

                elapsed_time = (
                    frame_gap
                    / EXPECTED_FPS
                )

                reference_movement = abs(
                    float(
                        reference_openness
                    )
                    - float(
                        previous_reference
                    )
                )

                reference_velocity = (
                    reference_movement
                    / elapsed_time
                )

                movement_signed_error = (
                    prediction_movement
                    - reference_movement
                )

                movement_absolute_error = abs(
                    movement_signed_error
                )

                velocity_signed_error = (
                    prediction_velocity
                    - reference_velocity
                )

                velocity_absolute_error = abs(
                    velocity_signed_error
                )

                total_evaluated_transitions += 1
                actor_accumulators[
                    actor
                ]["evaluated_transitions"] += 1

                movement_reference_values.append(
                    reference_movement
                )

                movement_prediction_values.append(
                    prediction_movement
                )

                velocity_reference_values.append(
                    reference_velocity
                )

                velocity_prediction_values.append(
                    prediction_velocity
                )

                actor_accumulators[
                    actor
                ]["movement_reference"].append(
                    reference_movement
                )

                actor_accumulators[
                    actor
                ]["movement_prediction"].append(
                    prediction_movement
                )

                actor_accumulators[
                    actor
                ]["velocity_reference"].append(
                    reference_velocity
                )

                actor_accumulators[
                    actor
                ]["velocity_prediction"].append(
                    prediction_velocity
                )

            frame_results.append(
                {
                    "actor": actor,
                    "trial": trial_name,
                    "frame": int(
                        frame_id
                    ),
                    "temporal_status": temporal_status,
                    "felt_openness_reference": float(
                        reference_openness
                    ),
                    "physiotrack_openness": prediction_openness,
                    "frame_gap": frame_gap,
                    "elapsed_time_sec": elapsed_time,
                    "felt_mouth_movement": reference_movement,
                    "physiotrack_mouth_movement": prediction_movement,
                    "movement_signed_error": movement_signed_error,
                    "movement_absolute_error": movement_absolute_error,
                    "felt_mouth_velocity": reference_velocity,
                    "physiotrack_mouth_velocity": prediction_velocity,
                    "velocity_signed_error": velocity_signed_error,
                    "velocity_absolute_error": velocity_absolute_error,
                }
            )

            previous_reference = float(
                reference_openness
            )

            previous_frame_id = int(
                frame_id
            )

        if (
            trial_index % 100 == 0
            or trial_index == len(trials)
        ):
            print(
                f"Trials: {trial_index}/{len(trials)} | "
                f"frames={len(frame_results)} | "
                f"evaluated_transitions={total_evaluated_transitions}"
            )

    runtime = (
        time.time()
        - start_time
    )

    expected_initialization_frames = (
        EXPECTED_TOTAL_TRIALS
    )

    expected_transitions = (
        EXPECTED_UNIQUE_ANNOTATED_FRAMES
        - EXPECTED_TOTAL_TRIALS
    )

    if (
        total_initialization_frames
        != expected_initialization_frames
    ):
        raise RuntimeError(
            "Unexpected initialization-frame count. "
            f"Expected {expected_initialization_frames}, "
            f"found {total_initialization_frames}."
        )

    if (
        total_evaluated_transitions
        != expected_transitions
    ):
        raise RuntimeError(
            "Unexpected evaluated-transition count. "
            f"Expected {expected_transitions}, "
            f"found {total_evaluated_transitions}."
        )

    if (
        len(frame_results)
        != EXPECTED_UNIQUE_ANNOTATED_FRAMES
    ):
        raise RuntimeError(
            "Unexpected per-frame output count. "
            f"Expected {EXPECTED_UNIQUE_ANNOTATED_FRAMES}, "
            f"found {len(frame_results)}."
        )

    movement_reference_array = np.asarray(
        movement_reference_values,
        dtype=np.float64,
    )

    movement_prediction_array = np.asarray(
        movement_prediction_values,
        dtype=np.float64,
    )

    velocity_reference_array = np.asarray(
        velocity_reference_values,
        dtype=np.float64,
    )

    velocity_prediction_array = np.asarray(
        velocity_prediction_values,
        dtype=np.float64,
    )

    movement_metrics = regression_metrics(
        movement_reference_array,
        movement_prediction_array,
    )

    velocity_metrics = regression_metrics(
        velocity_reference_array,
        velocity_prediction_array,
    )

    actor_results = []

    for actor in EXPECTED_ACTORS:
        accumulator = actor_accumulators[
            actor
        ]

        actor_movement_reference = np.asarray(
            accumulator[
                "movement_reference"
            ],
            dtype=np.float64,
        )

        actor_movement_prediction = np.asarray(
            accumulator[
                "movement_prediction"
            ],
            dtype=np.float64,
        )

        actor_velocity_reference = np.asarray(
            accumulator[
                "velocity_reference"
            ],
            dtype=np.float64,
        )

        actor_velocity_prediction = np.asarray(
            accumulator[
                "velocity_prediction"
            ],
            dtype=np.float64,
        )

        actor_movement_metrics = regression_metrics(
            actor_movement_reference,
            actor_movement_prediction,
        )

        actor_velocity_metrics = regression_metrics(
            actor_velocity_reference,
            actor_velocity_prediction,
        )

        actor_results.append(
            {
                "actor": actor,
                "frames": int(
                    accumulator[
                        "frames"
                    ]
                ),
                "initialization_frames": int(
                    accumulator[
                        "initialization_frames"
                    ]
                ),
                "evaluated_transitions": int(
                    accumulator[
                        "evaluated_transitions"
                    ]
                ),
                **{
                    f"movement_{key}": value
                    for key, value in actor_movement_metrics.items()
                },
                **{
                    f"velocity_{key}": value
                    for key, value in actor_velocity_metrics.items()
                },
            }
        )

    per_frame_fieldnames = list(
        frame_results[
            0
        ].keys()
    )

    with PER_FRAME_OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=per_frame_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            frame_results
        )

    per_actor_fieldnames = list(
        actor_results[
            0
        ].keys()
    )

    with PER_ACTOR_OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=per_actor_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            actor_results
        )

    summary_lines = [
        "FELT/RAVDESS Mouth Movement and Velocity Evaluation",
        "",
        "Validation target: PhysioTrack MouthMovement",
        "Prediction input: accepted PhysioTrack mouth-openness per-frame results",
        "Source result: validation/mouth_openness/results/"
        "felt_ravdess_mouth_openness_per_frame.csv",
        "FELT dataset: datasets/FELT/raw_motion_speech",
        "Dataset scope: paired speech subset",
        "Ground-truth openness: d(62,66) / d(48,54)",
        "Ground-truth movement: absolute change in FELT openness between consecutive frames",
        "Ground-truth velocity: ground-truth movement / elapsed frame time",
        "PhysioTrack movement: MouthMovement output from accepted PhysioTrack openness sequence",
        "PhysioTrack velocity: MouthMovement output using the locked video FPS",
        "Initialization rule: first frame of every trial is verified as zero output and excluded from regression metrics",
        "Duplicate-frame rule: largest FaceRect area, FaceScore tie-breaker",
        "Temporal alignment: exact zero-based FELT frame order",
        f"FPS: {EXPECTED_FPS:.12f}",
        f"Actors: {len(EXPECTED_ACTORS)}",
        f"Trials: {len(trials)}",
        f"Total annotated frames: {len(frame_results)}",
        f"Initialization frames: {total_initialization_frames}",
        f"Evaluated frame transitions: {total_evaluated_transitions}",
        f"Duplicate annotation rows resolved: {duplicate_rows_resolved}",
        "",
        "Mouth Movement Metrics",
        f"MAE: {format_metric(movement_metrics['mae'])}",
        f"RMSE: {format_metric(movement_metrics['rmse'])}",
        "Median absolute error: "
        f"{format_metric(movement_metrics['median_absolute_error'])}",
        "Std absolute error: "
        f"{format_metric(movement_metrics['std_absolute_error'])}",
        "90th percentile absolute error: "
        f"{format_metric(movement_metrics['p90_absolute_error'])}",
        "95th percentile absolute error: "
        f"{format_metric(movement_metrics['p95_absolute_error'])}",
        "Mean signed error (prediction - reference): "
        f"{format_metric(movement_metrics['mean_signed_error'])}",
        f"Pearson r: {format_metric(movement_metrics['pearson_r'])}",
        f"Spearman rho: {format_metric(movement_metrics['spearman_rho'])}",
        f"Lin CCC: {format_metric(movement_metrics['ccc'])}",
        "",
        "Mouth Velocity Metrics",
        f"MAE: {format_metric(velocity_metrics['mae'])}",
        f"RMSE: {format_metric(velocity_metrics['rmse'])}",
        "Median absolute error: "
        f"{format_metric(velocity_metrics['median_absolute_error'])}",
        "Std absolute error: "
        f"{format_metric(velocity_metrics['std_absolute_error'])}",
        "90th percentile absolute error: "
        f"{format_metric(velocity_metrics['p90_absolute_error'])}",
        "95th percentile absolute error: "
        f"{format_metric(velocity_metrics['p95_absolute_error'])}",
        "Mean signed error (prediction - reference): "
        f"{format_metric(velocity_metrics['mean_signed_error'])}",
        f"Pearson r: {format_metric(velocity_metrics['pearson_r'])}",
        f"Spearman rho: {format_metric(velocity_metrics['spearman_rho'])}",
        f"Lin CCC: {format_metric(velocity_metrics['ccc'])}",
        "",
        f"Runtime: {runtime:.2f} seconds",
    ]

    with SUMMARY_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(
                summary_lines
            )
        )

    print()
    print(
        "=== Final Summary ==="
    )

    for line in summary_lines:
        print(
            line
        )

    print()
    print(
        f"Saved: {PER_FRAME_OUTPUT_PATH}"
    )

    print(
        f"Saved: {PER_ACTOR_OUTPUT_PATH}"
    )

    print(
        f"Saved: {SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
