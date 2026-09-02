from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

PER_FRAME_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_per_frame.csv"
PER_ACTOR_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_per_actor.csv"
SUMMARY_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_summary.txt"

OVERALL_TABLE_CSV_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_thesis_table.csv"
PER_ACTOR_TABLE_CSV_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_per_actor_thesis_table.csv"

# Legacy outputs from an earlier draft. They are not part of the accepted final layout,
# but the plot script owns them and should remove them during clean reruns if they remain.
LEGACY_OVERALL_TABLE_MD_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_thesis_table.md"
LEGACY_PER_ACTOR_TABLE_MD_PATH = RESULTS_DIR / "felt_ravdess_mouth_openness_per_actor_thesis_table.md"

AGREEMENT_FIGURE_PATH = FIGURES_DIR / "felt_ravdess_mouth_openness_agreement.png"
ERROR_FIGURE_PATH = FIGURES_DIR / "felt_ravdess_mouth_openness_error_distribution.png"
PER_ACTOR_FIGURE_PATH = FIGURES_DIR / "felt_ravdess_mouth_openness_per_actor.png"

OWNED_OUTPUTS = [
    OVERALL_TABLE_CSV_PATH,
    PER_ACTOR_TABLE_CSV_PATH,
    AGREEMENT_FIGURE_PATH,
    ERROR_FIGURE_PATH,
    PER_ACTOR_FIGURE_PATH,
    LEGACY_OVERALL_TABLE_MD_PATH,
    LEGACY_PER_ACTOR_TABLE_MD_PATH,
]

EXPECTED_FRAMES = 158286
EXPECTED_ACTORS = 24
SUMMARY_TOLERANCE = 5e-7
PER_ACTOR_TOLERANCE = 5e-12


def clean_owned_outputs() -> None:
    for path in OWNED_OUTPUTS:
        if path.is_file():
            path.unlink()


def require_input(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required quantitative result file not found: {path}"
        )


def pearson_correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.corrcoef(reference, prediction)[0, 1])


def spearman_correlation(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        pd.Series(reference, dtype="float64").corr(
            pd.Series(prediction, dtype="float64"),
            method="spearman",
        )
    )


def concordance_correlation_coefficient(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    reference_mean = float(reference.mean())
    prediction_mean = float(prediction.mean())
    reference_variance = float(reference.var())
    prediction_variance = float(prediction.var())

    covariance = float(
        np.mean((reference - reference_mean) * (prediction - prediction_mean))
    )

    denominator = (
        reference_variance
        + prediction_variance
        + (reference_mean - prediction_mean) ** 2
    )

    if denominator <= 0:
        return math.nan

    return float(2.0 * covariance / denominator)


def regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    signed_error = prediction - reference
    absolute_error = np.abs(signed_error)

    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(signed_error ** 2))),
        "Median absolute error": float(np.median(absolute_error)),
        "90th percentile absolute error": float(np.percentile(absolute_error, 90)),
        "95th percentile absolute error": float(np.percentile(absolute_error, 95)),
        "Mean signed error": float(signed_error.mean()),
        "Pearson r": pearson_correlation(reference, prediction),
        "Spearman rho": spearman_correlation(reference, prediction),
        "Lin CCC": concordance_correlation_coefficient(reference, prediction),
    }


def load_quantitative_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (PER_FRAME_PATH, PER_ACTOR_PATH, SUMMARY_PATH):
        require_input(path)

    frame_table = pd.read_csv(PER_FRAME_PATH)
    actor_table = pd.read_csv(PER_ACTOR_PATH)

    required_frame_columns = {
        "actor",
        "trial",
        "frame",
        "status",
        "felt_reference",
        "felt_reference_three_pair",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "felt_reference_facebox",
        "physiotrack_openness_facebox",
        "facebox_signed_error",
        "facebox_absolute_error",
    }
    missing_frame_columns = sorted(required_frame_columns - set(frame_table.columns))
    if missing_frame_columns:
        raise RuntimeError(
            "Per-frame results are missing required columns: "
            f"{missing_frame_columns}"
        )

    required_actor_columns = {
        "actor",
        "annotations",
        "successful_predictions",
        "availability",
        "primary_mae",
        "primary_rmse",
        "primary_mean_signed_error",
        "primary_pearson_r",
        "primary_spearman_rho",
        "primary_ccc",
        "facebox_mae",
        "facebox_rmse",
        "facebox_mean_signed_error",
        "facebox_pearson_r",
        "facebox_spearman_rho",
        "facebox_ccc",
    }
    missing_actor_columns = sorted(required_actor_columns - set(actor_table.columns))
    if missing_actor_columns:
        raise RuntimeError(
            "Per-actor results are missing required columns: "
            f"{missing_actor_columns}"
        )

    if len(frame_table) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Unexpected per-frame result count: expected {EXPECTED_FRAMES}, found {len(frame_table)}."
        )

    if len(actor_table) != EXPECTED_ACTORS:
        raise RuntimeError(
            f"Unexpected per-actor result count: expected {EXPECTED_ACTORS}, found {len(actor_table)}."
        )

    if frame_table.duplicated(["actor", "trial", "frame"]).any():
        raise RuntimeError(
            "Duplicate actor/trial/frame rows were found in per-frame results."
        )

    status_counts = frame_table["status"].value_counts(dropna=False)
    if (
        len(status_counts) != 1
        or "success" not in status_counts
        or int(status_counts["success"]) != EXPECTED_FRAMES
    ):
        raise RuntimeError(
            "Accepted quantitative results must contain exactly "
            f"{EXPECTED_FRAMES} successful frames. Found: {status_counts.to_dict()}"
        )

    numeric_columns = [
        "felt_reference",
        "felt_reference_three_pair",
        "physiotrack_openness",
        "signed_error",
        "absolute_error",
        "felt_reference_facebox",
        "physiotrack_openness_facebox",
        "facebox_signed_error",
        "facebox_absolute_error",
    ]
    numeric_values = frame_table[numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError(
            "Accepted per-frame quantitative results contain non-finite values."
        )

    return frame_table, actor_table


def parse_summary_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    section = None

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "Primary Reference Metrics":
                section = "primary"
                continue
            if line == "Secondary Three-Pair Reference Metrics":
                section = "secondary"
                continue
            if line == "Face-Box-Normalized Sensitivity Metrics":
                section = "facebox"
                continue
            if not line or section is None:
                continue
            if line.startswith("Runtime:"):
                section = None
                continue
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            value = value.strip()

            try:
                numeric_value = float(value)
            except ValueError:
                continue

            normalized_key = "Mean signed error" if key.startswith("Mean signed error") else key
            metrics[f"{section}:{normalized_key}"] = numeric_value

    return metrics


def validate_summary_consistency(
    primary_metrics: dict[str, float],
    secondary_metrics: dict[str, float],
    facebox_metrics: dict[str, float],
) -> None:
    summary_metrics = parse_summary_metrics()

    for section, metrics in (
        ("primary", primary_metrics),
        ("secondary", secondary_metrics),
        ("facebox", facebox_metrics),
    ):
        for key, value in metrics.items():
            if key == "Std absolute error":
                continue

            summary_key = f"{section}:{key}"
            if summary_key not in summary_metrics:
                continue

            if not math.isclose(
                value,
                summary_metrics[summary_key],
                rel_tol=0.0,
                abs_tol=SUMMARY_TOLERANCE,
            ):
                raise RuntimeError(
                    "Summary/per-frame consistency check failed for "
                    f"{summary_key}: recomputed={value}, summary={summary_metrics[summary_key]}."
                )


def validate_per_actor_consistency(
    frame_table: pd.DataFrame,
    actor_table: pd.DataFrame,
) -> None:
    actor_lookup = actor_table.set_index("actor")

    for actor, group in frame_table.groupby("actor", sort=True):
        if actor not in actor_lookup.index:
            raise RuntimeError(f"Actor missing from per-actor results: {actor}")

        reference = group["felt_reference"].to_numpy(dtype=np.float64)
        prediction = group["physiotrack_openness"].to_numpy(dtype=np.float64)
        facebox_reference = group["felt_reference_facebox"].to_numpy(
            dtype=np.float64
        )
        facebox_prediction = group["physiotrack_openness_facebox"].to_numpy(
            dtype=np.float64
        )
        metrics = regression_metrics(reference, prediction)
        facebox_metrics = regression_metrics(
            facebox_reference,
            facebox_prediction,
        )
        expected = actor_lookup.loc[actor]

        checks = {
            "primary_mae": metrics["MAE"],
            "primary_rmse": metrics["RMSE"],
            "primary_mean_signed_error": metrics["Mean signed error"],
            "primary_pearson_r": metrics["Pearson r"],
            "primary_spearman_rho": metrics["Spearman rho"],
            "primary_ccc": metrics["Lin CCC"],
            "facebox_mae": facebox_metrics["MAE"],
            "facebox_rmse": facebox_metrics["RMSE"],
            "facebox_mean_signed_error": facebox_metrics["Mean signed error"],
            "facebox_pearson_r": facebox_metrics["Pearson r"],
            "facebox_spearman_rho": facebox_metrics["Spearman rho"],
            "facebox_ccc": facebox_metrics["Lin CCC"],
        }

        for column, value in checks.items():
            if not math.isclose(
                float(expected[column]),
                value,
                rel_tol=0.0,
                abs_tol=PER_ACTOR_TOLERANCE,
            ):
                raise RuntimeError(
                    "Per-actor consistency check failed for "
                    f"{actor}/{column}."
                )


def create_tables(frame_table: pd.DataFrame, actor_table: pd.DataFrame) -> None:
    reference = frame_table["felt_reference"].to_numpy(dtype=np.float64)
    prediction = frame_table["physiotrack_openness"].to_numpy(dtype=np.float64)
    primary_metrics = regression_metrics(reference, prediction)

    facebox_reference = frame_table["felt_reference_facebox"].to_numpy(
        dtype=np.float64
    )
    facebox_prediction = frame_table["physiotrack_openness_facebox"].to_numpy(
        dtype=np.float64
    )
    facebox_metrics = regression_metrics(
        facebox_reference,
        facebox_prediction,
    )

    overall_table = pd.DataFrame(
        [
            {
                "Frames": len(frame_table),
                "Availability (%)": 100.0,
                "MAE": primary_metrics["MAE"],
                "RMSE": primary_metrics["RMSE"],
                "Median absolute error": primary_metrics["Median absolute error"],
                "P90 absolute error": primary_metrics["90th percentile absolute error"],
                "P95 absolute error": primary_metrics["95th percentile absolute error"],
                "Mean signed error": primary_metrics["Mean signed error"],
                "Pearson r": primary_metrics["Pearson r"],
                "Spearman rho": primary_metrics["Spearman rho"],
                "Lin CCC": primary_metrics["Lin CCC"],
                "Face-box MAE": facebox_metrics["MAE"],
                "Face-box RMSE": facebox_metrics["RMSE"],
                "Face-box mean signed error": facebox_metrics["Mean signed error"],
                "Face-box Pearson r": facebox_metrics["Pearson r"],
                "Face-box Spearman rho": facebox_metrics["Spearman rho"],
                "Face-box Lin CCC": facebox_metrics["Lin CCC"],
            }
        ]
    )
    overall_table.to_csv(OVERALL_TABLE_CSV_PATH, index=False)

    per_actor_table = pd.DataFrame(
        {
            "Actor": actor_table["actor"],
            "Frames": actor_table["annotations"].astype(int),
            "Availability (%)": actor_table["availability"] * 100.0,
            "MAE": actor_table["primary_mae"],
            "RMSE": actor_table["primary_rmse"],
            "Mean signed error": actor_table["primary_mean_signed_error"],
            "Pearson r": actor_table["primary_pearson_r"],
            "Spearman rho": actor_table["primary_spearman_rho"],
            "Lin CCC": actor_table["primary_ccc"],
            "Face-box MAE": actor_table["facebox_mae"],
            "Face-box RMSE": actor_table["facebox_rmse"],
            "Face-box Pearson r": actor_table["facebox_pearson_r"],
            "Face-box Spearman rho": actor_table["facebox_spearman_rho"],
            "Face-box Lin CCC": actor_table["facebox_ccc"],
        }
    )
    per_actor_table.to_csv(PER_ACTOR_TABLE_CSV_PATH, index=False)


def create_agreement_figure(frame_table: pd.DataFrame) -> None:
    reference = frame_table["felt_reference"].to_numpy(dtype=np.float64)
    prediction = frame_table["physiotrack_openness"].to_numpy(dtype=np.float64)
    upper_limit = float(max(reference.max(), prediction.max()))

    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    hexbin = axis.hexbin(reference, prediction, gridsize=85, mincnt=1, bins="log")

    axis.plot(
        [0.0, upper_limit],
        [0.0, upper_limit],
        linestyle="--",
        linewidth=1.2,
        label="Identity",
    )
    axis.set_xlim(0.0, upper_limit)
    axis.set_ylim(0.0, upper_limit)
    axis.set_xlabel("FELT landmark-derived mouth openness")
    axis.set_ylabel("PhysioTrack mouth openness")
    axis.set_title("FELT/RAVDESS Mouth-Openness Agreement")
    axis.legend()

    colorbar = figure.colorbar(hexbin, ax=axis)
    colorbar.set_label("Frame density (log scale)")

    figure.tight_layout()
    figure.savefig(AGREEMENT_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_error_figure(frame_table: pd.DataFrame) -> None:
    signed_error = frame_table["signed_error"].to_numpy(dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.hist(signed_error, bins=100)
    axis.axvline(0.0, linestyle="--", linewidth=1.2, label="Zero error")
    axis.axvline(
        float(signed_error.mean()),
        linestyle=":",
        linewidth=1.2,
        label="Mean signed error",
    )
    axis.set_xlabel("Prediction - FELT reference")
    axis.set_ylabel("Frames")
    axis.set_title("Mouth-Openness Signed-Error Distribution")
    axis.legend()

    figure.tight_layout()
    figure.savefig(ERROR_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def create_per_actor_figure(actor_table: pd.DataFrame) -> None:
    actors = actor_table["actor"].str.replace("Actor_", "", regex=False)
    x_positions = np.arange(len(actor_table))

    figure, axis = plt.subplots(figsize=(12.0, 6.0))
    axis.bar(x_positions, actor_table["primary_mae"])
    axis.axhline(
        float(
            np.average(
                actor_table["primary_mae"],
                weights=actor_table["successful_predictions"],
            )
        ),
        linestyle="--",
        linewidth=1.2,
        label="Frame-weighted overall MAE",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(actors, rotation=0)
    axis.set_xlabel("Actor")
    axis.set_ylabel("MAE")
    axis.set_title("Mouth-Openness Error by Actor")
    axis.legend()

    figure.tight_layout()
    figure.savefig(PER_ACTOR_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    print("=== FELT/RAVDESS Mouth Openness Plot and Table Generation ===")

    frame_table, actor_table = load_quantitative_results()

    reference = frame_table["felt_reference"].to_numpy(dtype=np.float64)
    secondary_reference = frame_table["felt_reference_three_pair"].to_numpy(dtype=np.float64)
    prediction = frame_table["physiotrack_openness"].to_numpy(dtype=np.float64)

    primary_metrics = regression_metrics(reference, prediction)
    secondary_metrics = regression_metrics(secondary_reference, prediction)

    facebox_reference = frame_table["felt_reference_facebox"].to_numpy(
        dtype=np.float64
    )
    facebox_prediction = frame_table["physiotrack_openness_facebox"].to_numpy(
        dtype=np.float64
    )
    facebox_metrics = regression_metrics(
        facebox_reference,
        facebox_prediction,
    )

    validate_summary_consistency(
        primary_metrics,
        secondary_metrics,
        facebox_metrics,
    )
    validate_per_actor_consistency(frame_table, actor_table)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_owned_outputs()

    create_tables(frame_table, actor_table)
    create_agreement_figure(frame_table)
    create_error_figure(frame_table)
    create_per_actor_figure(actor_table)

    print("Quantitative result consistency: PASS")
    print(f"Frames: {len(frame_table)}")
    print(f"Actors: {len(actor_table)}")
    print(f"MAE: {primary_metrics['MAE']:.6f}")
    print(f"Pearson r: {primary_metrics['Pearson r']:.6f}")
    print(f"Lin CCC: {primary_metrics['Lin CCC']:.6f}")
    print(
        "Face-box sensitivity MAE: "
        f"{facebox_metrics['MAE']:.6f}"
    )
    print(
        "Face-box sensitivity Lin CCC: "
        f"{facebox_metrics['Lin CCC']:.6f}"
    )
    print()

    for path in OWNED_OUTPUTS:
        if path.exists():
            print(f"Saved: {path}")
        else:
            print(f"Removed legacy or stale output: {path}")


if __name__ == "__main__":
    main()
