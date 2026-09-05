from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

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

OVERALL_TABLE_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_movement_velocity_thesis_table.csv"
)

PER_ACTOR_TABLE_PATH = (
    RESULTS_DIR
    / "felt_ravdess_mouth_movement_velocity_per_actor_thesis_table.csv"
)

MOVEMENT_AGREEMENT_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_movement_agreement.png"
)

VELOCITY_AGREEMENT_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_velocity_agreement.png"
)

ERROR_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_movement_velocity_error_distribution.png"
)

PER_ACTOR_FIGURE_PATH = (
    FIGURES_DIR
    / "felt_ravdess_mouth_movement_velocity_per_actor.png"
)

OWNED_OUTPUTS = [
    OVERALL_TABLE_PATH,
    PER_ACTOR_TABLE_PATH,
    MOVEMENT_AGREEMENT_FIGURE_PATH,
    VELOCITY_AGREEMENT_FIGURE_PATH,
    ERROR_FIGURE_PATH,
    PER_ACTOR_FIGURE_PATH,
]

EXPECTED_FRAMES = 158286
EXPECTED_INITIALIZATION_FRAMES = 1440
EXPECTED_TRANSITIONS = 156846
EXPECTED_ACTORS = 24
EXPECTED_FPS = 30000.0 / 1001.0

SUMMARY_TOLERANCE = 5e-7
PER_ACTOR_TOLERANCE = 5e-12
IDENTITY_TOLERANCE = 1e-10


def clean_owned_outputs() -> None:
    for path in OWNED_OUTPUTS:
        if path.is_file():
            path.unlink()


def require_input(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required quantitative result file not found: {path}"
        )


def pearson_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

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

    signed_error = (
        prediction
        - reference
    )

    absolute_error = np.abs(
        signed_error
    )

    return {
        "MAE": float(
            absolute_error.mean()
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    signed_error
                    ** 2
                )
            )
        ),
        "Median absolute error": float(
            np.median(
                absolute_error
            )
        ),
        "Std absolute error": float(
            absolute_error.std()
        ),
        "90th percentile absolute error": float(
            np.percentile(
                absolute_error,
                90,
            )
        ),
        "95th percentile absolute error": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),
        "Mean signed error": float(
            signed_error.mean()
        ),
        "Pearson r": pearson_correlation(
            reference,
            prediction,
        ),
        "Spearman rho": spearman_correlation(
            reference,
            prediction,
        ),
        "Lin CCC": concordance_correlation_coefficient(
            reference,
            prediction,
        ),
    }


def load_quantitative_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    for path in (
        PER_FRAME_PATH,
        PER_ACTOR_PATH,
        SUMMARY_PATH,
    ):
        require_input(
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
        "frames",
        "initialization_frames",
        "evaluated_transitions",
        "movement_mae",
        "movement_rmse",
        "movement_mean_signed_error",
        "movement_pearson_r",
        "movement_spearman_rho",
        "movement_ccc",
        "velocity_mae",
        "velocity_rmse",
        "velocity_mean_signed_error",
        "velocity_pearson_r",
        "velocity_spearman_rho",
        "velocity_ccc",
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
            "Unexpected per-frame result count: "
            f"expected {EXPECTED_FRAMES}, found {len(frame_table)}."
        )

    if len(actor_table) != EXPECTED_ACTORS:
        raise RuntimeError(
            "Unexpected per-actor result count: "
            f"expected {EXPECTED_ACTORS}, found {len(actor_table)}."
        )

    if frame_table.duplicated(
        [
            "actor",
            "trial",
            "frame",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in per-frame results."
        )

    status_counts = frame_table[
        "temporal_status"
    ].value_counts(
        dropna=False
    )

    expected_status_counts = {
        "initialization": EXPECTED_INITIALIZATION_FRAMES,
        "evaluated_transition": EXPECTED_TRANSITIONS,
    }

    if status_counts.to_dict() != expected_status_counts:
        raise RuntimeError(
            "Unexpected temporal-status accounting. "
            f"Expected {expected_status_counts}, "
            f"found {status_counts.to_dict()}."
        )

    initialization = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "initialization"
    ].copy()

    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ].copy()

    if len(initialization) != EXPECTED_INITIALIZATION_FRAMES:
        raise RuntimeError(
            "Unexpected initialization-frame count."
        )

    if len(transitions) != EXPECTED_TRANSITIONS:
        raise RuntimeError(
            "Unexpected evaluated-transition count."
        )

    initialization_numeric = initialization[
        [
            "felt_openness_reference",
            "physiotrack_openness",
            "physiotrack_mouth_movement",
            "physiotrack_mouth_velocity",
        ]
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            initialization_numeric
        )
    ):
        raise RuntimeError(
            "Initialization rows contain non-finite required values."
        )

    if not np.allclose(
        initialization[
            "physiotrack_mouth_movement"
        ].to_numpy(
            dtype=np.float64
        ),
        0.0,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Initialization mouth-movement values are not all zero."
        )

    if not np.allclose(
        initialization[
            "physiotrack_mouth_velocity"
        ].to_numpy(
            dtype=np.float64
        ),
        0.0,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Initialization mouth-velocity values are not all zero."
        )

    transition_numeric_columns = [
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

    transition_numeric = transitions[
        transition_numeric_columns
    ].to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(
            transition_numeric
        )
    ):
        raise RuntimeError(
            "Evaluated transitions contain non-finite quantitative values."
        )

    frame_gaps = transitions[
        "frame_gap"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.allclose(
        frame_gaps,
        1.0,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Accepted temporal benchmark expects frame_gap=1 "
            "for every evaluated transition."
        )

    elapsed_time = transitions[
        "elapsed_time_sec"
    ].to_numpy(
        dtype=np.float64
    )

    expected_elapsed_time = (
        1.0
        / EXPECTED_FPS
    )

    if not np.allclose(
        elapsed_time,
        expected_elapsed_time,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Elapsed-time values do not match the locked benchmark FPS."
        )

    movement_reference = transitions[
        "felt_mouth_movement"
    ].to_numpy(
        dtype=np.float64
    )

    movement_prediction = transitions[
        "physiotrack_mouth_movement"
    ].to_numpy(
        dtype=np.float64
    )

    velocity_reference = transitions[
        "felt_mouth_velocity"
    ].to_numpy(
        dtype=np.float64
    )

    velocity_prediction = transitions[
        "physiotrack_mouth_velocity"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.allclose(
        velocity_reference,
        movement_reference
        * EXPECTED_FPS,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Ground-truth velocity is inconsistent with movement and FPS."
        )

    if not np.allclose(
        velocity_prediction,
        movement_prediction
        * EXPECTED_FPS,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "PhysioTrack velocity is inconsistent with movement and FPS."
        )

    if not np.allclose(
        transitions[
            "movement_signed_error"
        ].to_numpy(
            dtype=np.float64
        ),
        movement_prediction
        - movement_reference,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Stored movement signed errors are inconsistent."
        )

    if not np.allclose(
        transitions[
            "movement_absolute_error"
        ].to_numpy(
            dtype=np.float64
        ),
        np.abs(
            movement_prediction
            - movement_reference
        ),
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Stored movement absolute errors are inconsistent."
        )

    if not np.allclose(
        transitions[
            "velocity_signed_error"
        ].to_numpy(
            dtype=np.float64
        ),
        velocity_prediction
        - velocity_reference,
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Stored velocity signed errors are inconsistent."
        )

    if not np.allclose(
        transitions[
            "velocity_absolute_error"
        ].to_numpy(
            dtype=np.float64
        ),
        np.abs(
            velocity_prediction
            - velocity_reference
        ),
        rtol=0.0,
        atol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Stored velocity absolute errors are inconsistent."
        )

    return (
        frame_table,
        actor_table,
    )


def parse_summary_metrics() -> dict[
    str,
    float,
]:
    metrics = {}
    section = None

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "Mouth Movement Metrics":
                section = "movement"
                continue

            if line == "Mouth Velocity Metrics":
                section = "velocity"
                continue

            if (
                not line
                or section is None
            ):
                continue

            if line.startswith(
                "Runtime:"
            ):
                section = None
                continue

            if ":" not in line:
                continue

            key, value = line.split(
                ":",
                1,
            )

            normalized_key = (
                "Mean signed error"
                if key.startswith(
                    "Mean signed error"
                )
                else key
            )

            try:
                numeric_value = float(
                    value.strip()
                )
            except ValueError:
                continue

            metrics[
                f"{section}:{normalized_key}"
            ] = numeric_value

    return metrics


def validate_summary_consistency(
    movement_metrics: dict[
        str,
        float,
    ],
    velocity_metrics: dict[
        str,
        float,
    ],
) -> None:
    summary_metrics = (
        parse_summary_metrics()
    )

    for section, metrics in (
        (
            "movement",
            movement_metrics,
        ),
        (
            "velocity",
            velocity_metrics,
        ),
    ):
        for key, value in metrics.items():
            summary_key = (
                f"{section}:{key}"
            )

            if summary_key not in summary_metrics:
                raise RuntimeError(
                    "Required metric is missing from summary: "
                    f"{summary_key}"
                )

            if not math.isclose(
                value,
                summary_metrics[
                    summary_key
                ],
                rel_tol=0.0,
                abs_tol=SUMMARY_TOLERANCE,
            ):
                raise RuntimeError(
                    "Summary/per-frame consistency check failed for "
                    f"{summary_key}: "
                    f"recomputed={value}, "
                    f"summary={summary_metrics[summary_key]}."
                )


def validate_per_actor_consistency(
    frame_table: pd.DataFrame,
    actor_table: pd.DataFrame,
) -> None:
    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ]

    actor_lookup = actor_table.set_index(
        "actor"
    )

    for actor, group in transitions.groupby(
        "actor",
        sort=True,
    ):
        if actor not in actor_lookup.index:
            raise RuntimeError(
                f"Actor missing from per-actor results: {actor}"
            )

        movement_reference = group[
            "felt_mouth_movement"
        ].to_numpy(
            dtype=np.float64
        )

        movement_prediction = group[
            "physiotrack_mouth_movement"
        ].to_numpy(
            dtype=np.float64
        )

        velocity_reference = group[
            "felt_mouth_velocity"
        ].to_numpy(
            dtype=np.float64
        )

        velocity_prediction = group[
            "physiotrack_mouth_velocity"
        ].to_numpy(
            dtype=np.float64
        )

        movement_metrics = regression_metrics(
            movement_reference,
            movement_prediction,
        )

        velocity_metrics = regression_metrics(
            velocity_reference,
            velocity_prediction,
        )

        expected = actor_lookup.loc[
            actor
        ]

        if int(
            expected[
                "evaluated_transitions"
            ]
        ) != len(
            group
        ):
            raise RuntimeError(
                "Per-actor transition count mismatch for "
                f"{actor}."
            )

        checks = {
            "movement_mae": movement_metrics[
                "MAE"
            ],
            "movement_rmse": movement_metrics[
                "RMSE"
            ],
            "movement_mean_signed_error": movement_metrics[
                "Mean signed error"
            ],
            "movement_pearson_r": movement_metrics[
                "Pearson r"
            ],
            "movement_spearman_rho": movement_metrics[
                "Spearman rho"
            ],
            "movement_ccc": movement_metrics[
                "Lin CCC"
            ],
            "velocity_mae": velocity_metrics[
                "MAE"
            ],
            "velocity_rmse": velocity_metrics[
                "RMSE"
            ],
            "velocity_mean_signed_error": velocity_metrics[
                "Mean signed error"
            ],
            "velocity_pearson_r": velocity_metrics[
                "Pearson r"
            ],
            "velocity_spearman_rho": velocity_metrics[
                "Spearman rho"
            ],
            "velocity_ccc": velocity_metrics[
                "Lin CCC"
            ],
        }

        for column, value in checks.items():
            if not math.isclose(
                float(
                    expected[
                        column
                    ]
                ),
                value,
                rel_tol=0.0,
                abs_tol=PER_ACTOR_TOLERANCE,
            ):
                raise RuntimeError(
                    "Per-actor consistency check failed for "
                    f"{actor}/{column}."
                )


def create_tables(
    frame_table: pd.DataFrame,
    actor_table: pd.DataFrame,
) -> None:
    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ]

    movement_metrics = regression_metrics(
        transitions[
            "felt_mouth_movement"
        ].to_numpy(
            dtype=np.float64
        ),
        transitions[
            "physiotrack_mouth_movement"
        ].to_numpy(
            dtype=np.float64
        ),
    )

    velocity_metrics = regression_metrics(
        transitions[
            "felt_mouth_velocity"
        ].to_numpy(
            dtype=np.float64
        ),
        transitions[
            "physiotrack_mouth_velocity"
        ].to_numpy(
            dtype=np.float64
        ),
    )

    overall_table = pd.DataFrame(
        [
            {
                "Measure": "Mouth movement",
                "Transitions": len(
                    transitions
                ),
                **movement_metrics,
            },
            {
                "Measure": "Mouth velocity",
                "Transitions": len(
                    transitions
                ),
                **velocity_metrics,
            },
        ]
    )

    overall_table.to_csv(
        OVERALL_TABLE_PATH,
        index=False,
    )

    per_actor_table = pd.DataFrame(
        {
            "Actor": actor_table[
                "actor"
            ],
            "Frames": actor_table[
                "frames"
            ].astype(
                int
            ),
            "Initialization frames": actor_table[
                "initialization_frames"
            ].astype(
                int
            ),
            "Evaluated transitions": actor_table[
                "evaluated_transitions"
            ].astype(
                int
            ),
            "Movement MAE": actor_table[
                "movement_mae"
            ],
            "Movement RMSE": actor_table[
                "movement_rmse"
            ],
            "Movement Pearson r": actor_table[
                "movement_pearson_r"
            ],
            "Movement Spearman rho": actor_table[
                "movement_spearman_rho"
            ],
            "Movement Lin CCC": actor_table[
                "movement_ccc"
            ],
            "Velocity MAE": actor_table[
                "velocity_mae"
            ],
            "Velocity RMSE": actor_table[
                "velocity_rmse"
            ],
            "Velocity Pearson r": actor_table[
                "velocity_pearson_r"
            ],
            "Velocity Spearman rho": actor_table[
                "velocity_spearman_rho"
            ],
            "Velocity Lin CCC": actor_table[
                "velocity_ccc"
            ],
        }
    )

    per_actor_table.to_csv(
        PER_ACTOR_TABLE_PATH,
        index=False,
    )


def create_agreement_figure(
    reference: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    upper_limit = float(
        max(
            reference.max(),
            prediction.max(),
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            8.5,
            7.0,
        )
    )

    hexbin = axis.hexbin(
        reference,
        prediction,
        gridsize=85,
        mincnt=1,
        bins="log",
    )

    axis.plot(
        [
            0.0,
            upper_limit,
        ],
        [
            0.0,
            upper_limit,
        ],
        linestyle="--",
        linewidth=1.2,
        label="Identity",
    )

    axis.set_xlim(
        0.0,
        upper_limit,
    )

    axis.set_ylim(
        0.0,
        upper_limit,
    )

    axis.set_xlabel(
        x_label
    )

    axis.set_ylabel(
        y_label
    )

    axis.set_title(
        title
    )

    axis.legend()

    colorbar = figure.colorbar(
        hexbin,
        ax=axis,
    )

    colorbar.set_label(
        "Transition density (log scale)"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_error_figure(
    transitions: pd.DataFrame,
) -> None:
    movement_error = transitions[
        "movement_signed_error"
    ].to_numpy(
        dtype=np.float64
    )

    velocity_error = transitions[
        "velocity_signed_error"
    ].to_numpy(
        dtype=np.float64
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            8.5,
            10.0,
        ),
    )

    axes[
        0
    ].hist(
        movement_error,
        bins=100,
    )

    axes[
        0
    ].axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Zero error",
    )

    axes[
        0
    ].axvline(
        float(
            movement_error.mean()
        ),
        linestyle=":",
        linewidth=1.2,
        label="Mean signed error",
    )

    axes[
        0
    ].set_xlabel(
        "Predicted movement - FELT movement"
    )

    axes[
        0
    ].set_ylabel(
        "Transitions"
    )

    axes[
        0
    ].set_title(
        "Mouth-Movement Signed-Error Distribution"
    )

    axes[
        0
    ].legend()

    axes[
        1
    ].hist(
        velocity_error,
        bins=100,
    )

    axes[
        1
    ].axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="Zero error",
    )

    axes[
        1
    ].axvline(
        float(
            velocity_error.mean()
        ),
        linestyle=":",
        linewidth=1.2,
        label="Mean signed error",
    )

    axes[
        1
    ].set_xlabel(
        "Predicted velocity - FELT velocity"
    )

    axes[
        1
    ].set_ylabel(
        "Transitions"
    )

    axes[
        1
    ].set_title(
        "Mouth-Velocity Signed-Error Distribution"
    )

    axes[
        1
    ].legend()

    figure.tight_layout()

    figure.savefig(
        ERROR_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def create_per_actor_figure(
    actor_table: pd.DataFrame,
) -> None:
    actors = actor_table[
        "actor"
    ].str.replace(
        "Actor_",
        "",
        regex=False,
    )

    x_positions = np.arange(
        len(
            actor_table
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            12.0,
            6.0,
        )
    )

    axis.bar(
        x_positions,
        actor_table[
            "movement_mae"
        ],
    )

    weighted_mae = float(
        np.average(
            actor_table[
                "movement_mae"
            ],
            weights=actor_table[
                "evaluated_transitions"
            ],
        )
    )

    axis.axhline(
        weighted_mae,
        linestyle="--",
        linewidth=1.2,
        label="Transition-weighted overall MAE",
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        actors,
        rotation=0,
    )

    axis.set_xlabel(
        "Actor"
    )

    axis.set_ylabel(
        "Movement MAE"
    )

    axis.set_title(
        "Mouth-Movement Error by Actor"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        PER_ACTOR_FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    print(
        "=== FELT/RAVDESS Mouth Movement and Velocity Plot and Table Generation ==="
    )

    (
        frame_table,
        actor_table,
    ) = load_quantitative_results()

    transitions = frame_table[
        frame_table[
            "temporal_status"
        ]
        == "evaluated_transition"
    ].copy()

    movement_reference = transitions[
        "felt_mouth_movement"
    ].to_numpy(
        dtype=np.float64
    )

    movement_prediction = transitions[
        "physiotrack_mouth_movement"
    ].to_numpy(
        dtype=np.float64
    )

    velocity_reference = transitions[
        "felt_mouth_velocity"
    ].to_numpy(
        dtype=np.float64
    )

    velocity_prediction = transitions[
        "physiotrack_mouth_velocity"
    ].to_numpy(
        dtype=np.float64
    )

    movement_metrics = regression_metrics(
        movement_reference,
        movement_prediction,
    )

    velocity_metrics = regression_metrics(
        velocity_reference,
        velocity_prediction,
    )

    validate_summary_consistency(
        movement_metrics,
        velocity_metrics,
    )

    validate_per_actor_consistency(
        frame_table,
        actor_table,
    )

    if not math.isclose(
        movement_metrics[
            "Pearson r"
        ],
        velocity_metrics[
            "Pearson r"
        ],
        rel_tol=0.0,
        abs_tol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Movement and velocity Pearson correlations should match "
            "under constant-FPS scalar transformation."
        )

    if not math.isclose(
        movement_metrics[
            "Spearman rho"
        ],
        velocity_metrics[
            "Spearman rho"
        ],
        rel_tol=0.0,
        abs_tol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Movement and velocity Spearman correlations should match "
            "under constant-FPS scalar transformation."
        )

    if not math.isclose(
        movement_metrics[
            "Lin CCC"
        ],
        velocity_metrics[
            "Lin CCC"
        ],
        rel_tol=0.0,
        abs_tol=IDENTITY_TOLERANCE,
    ):
        raise RuntimeError(
            "Movement and velocity Lin CCC should match "
            "under equal positive scalar transformation."
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_owned_outputs()

    create_tables(
        frame_table,
        actor_table,
    )

    create_agreement_figure(
        movement_reference,
        movement_prediction,
        MOVEMENT_AGREEMENT_FIGURE_PATH,
        "FELT landmark-derived mouth movement",
        "PhysioTrack mouth movement",
        "FELT/RAVDESS Mouth-Movement Agreement",
    )

    create_agreement_figure(
        velocity_reference,
        velocity_prediction,
        VELOCITY_AGREEMENT_FIGURE_PATH,
        "FELT landmark-derived mouth velocity",
        "PhysioTrack mouth velocity",
        "FELT/RAVDESS Mouth-Velocity Agreement",
    )

    create_error_figure(
        transitions
    )

    create_per_actor_figure(
        actor_table
    )

    print(
        "Quantitative result consistency: PASS"
    )

    print(
        f"Frames: {len(frame_table)}"
    )

    print(
        f"Initialization frames: {EXPECTED_INITIALIZATION_FRAMES}"
    )

    print(
        f"Evaluated transitions: {len(transitions)}"
    )

    print(
        f"Actors: {len(actor_table)}"
    )

    print(
        "Movement MAE: "
        f"{movement_metrics['MAE']:.6f}"
    )

    print(
        "Movement Pearson r: "
        f"{movement_metrics['Pearson r']:.6f}"
    )

    print(
        "Movement Lin CCC: "
        f"{movement_metrics['Lin CCC']:.6f}"
    )

    print(
        "Velocity MAE: "
        f"{velocity_metrics['MAE']:.6f}"
    )

    print(
        "Velocity Pearson r: "
        f"{velocity_metrics['Pearson r']:.6f}"
    )

    print(
        "Velocity Lin CCC: "
        f"{velocity_metrics['Lin CCC']:.6f}"
    )

    print()

    for path in OWNED_OUTPUTS:
        if path.is_file():
            print(
                f"Saved: {path}"
            )


if __name__ == "__main__":
    main()
