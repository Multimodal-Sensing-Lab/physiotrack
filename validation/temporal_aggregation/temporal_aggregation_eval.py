from collections import Counter
from pathlib import Path
import csv
import inspect
import math
import os
import shutil
import sys
import tempfile

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "temporal_aggregation_results.csv"
METRICS_PATH = RESULTS_DIR / "temporal_aggregation_metrics.csv"
SUMMARY_PATH = RESULTS_DIR / "temporal_aggregation_summary.txt"

EXPECTED_TEMPORAL_SOURCE = (
    SRC_ROOT
    / "physiotrack"
    / "face"
    / "temporal.py"
)

NUMERIC_TOLERANCE = 1e-12

RESULT_FIELDS = (
    "case",
    "check",
    "expected",
    "observed",
    "absolute_error",
    "status",
    "notes",
)

METRIC_FIELDS = (
    "metric",
    "value",
)


def import_physio_track():
    """Import the current PhysioTrack temporal implementation."""
    if not SRC_ROOT.is_dir():
        raise FileNotFoundError(
            f"PhysioTrack source directory not found: {SRC_ROOT}"
        )

    src_string = str(SRC_ROOT)
    if src_string not in sys.path:
        sys.path.insert(0, src_string)

    from physiotrack.face.temporal import FaceTemporalAggregator
    from physiotrack.results import Instance

    implementation_path = Path(
        inspect.getfile(FaceTemporalAggregator)
    ).resolve()

    if implementation_path != EXPECTED_TEMPORAL_SOURCE.resolve():
        raise RuntimeError(
            "Imported FaceTemporalAggregator does not match the "
            "current repository source. "
            f"Expected {EXPECTED_TEMPORAL_SOURCE.resolve()}, "
            f"got {implementation_path}"
        )

    return (
        FaceTemporalAggregator,
        Instance,
        implementation_path,
    )


def create_instance(
    Instance,
    person_id,
    *,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    eye=0.3,
    gaze_x=0.5,
    gaze_y=0.5,
    mouth=0.2,
    movement=0.05,
    brightness=0.6,
    sharpness=100.0,
    face_area_ratio=0.1,
    blink=False,
    emotion="Neutral",
    eyes_available=True,
    gaze_available=True,
    mouth_available=True,
    mouth_motion_available=True,
    quality_available=True,
    emotion_available=True,
    face_features=True,
):
    """Create a deterministic tracked face instance."""
    if not face_features:
        return Instance(
            id=person_id,
            orientation={
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
            },
            face_features=None,
        )

    return Instance(
        id=person_id,
        orientation={
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        },
        face_features={
            "eyes": {
                "available": eyes_available,
                "mean_openness": eye,
            },
            "gaze": {
                "available": gaze_available,
                "mean_iris_x": gaze_x,
                "mean_iris_y": gaze_y,
            },
            "mouth": {
                "available": mouth_available,
                "mouth_openness": mouth,
            },
            "mouth_motion": {
                "available": mouth_motion_available,
                "mouth_movement": movement,
            },
            "quality": {
                "available": quality_available,
                "brightness": brightness,
                "sharpness": sharpness,
                "face_area_ratio": face_area_ratio,
            },
            "blink": {
                "available": True,
                "blink": blink,
            },
            "emotion": {
                "available": emotion_available,
                "emotion": emotion,
            },
        },
    )


def independent_numeric_summary(values):
    """Independently calculate the expected population statistics."""
    valid_values = []

    for value in values:
        if value is None:
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(numeric_value):
            continue

        valid_values.append(numeric_value)

    if not valid_values:
        return None

    count = len(valid_values)
    mean = sum(valid_values) / count
    variance = sum(
        (value - mean) ** 2
        for value in valid_values
    ) / count

    return {
        "mean": float(mean),
        "std": float(math.sqrt(variance)),
        "min": float(min(valid_values)),
        "max": float(max(valid_values)),
    }


def format_value(value):
    """Return a compact serializable representation."""
    if isinstance(value, float):
        return f"{value:.16g}"

    if value is None:
        return "None"

    return str(value)


def add_check(
    rows,
    *,
    case,
    check,
    expected,
    observed,
    notes="",
    tolerance=NUMERIC_TOLERANCE,
):
    """Append one validation check and return whether it passed."""
    absolute_error = ""

    if (
        isinstance(expected, (int, float, np.integer, np.floating))
        and not isinstance(expected, bool)
        and isinstance(observed, (int, float, np.integer, np.floating))
        and not isinstance(observed, bool)
    ):
        expected_float = float(expected)
        observed_float = float(observed)

        if (
            math.isfinite(expected_float)
            and math.isfinite(observed_float)
        ):
            absolute_error = abs(observed_float - expected_float)
            passed = absolute_error <= tolerance
        else:
            passed = expected_float == observed_float
    else:
        passed = observed == expected

    rows.append(
        {
            "case": case,
            "check": check,
            "expected": format_value(expected),
            "observed": format_value(observed),
            "absolute_error": (
                ""
                if absolute_error == ""
                else format_value(absolute_error)
            ),
            "status": "PASS" if passed else "FAIL",
            "notes": notes,
        }
    )

    return passed


def add_summary_checks(
    rows,
    *,
    case,
    prefix,
    expected,
    observed,
):
    """Compare one numerical summary field-by-field."""
    passed = True

    if expected is None:
        passed &= add_check(
            rows,
            case=case,
            check=f"{prefix}.summary",
            expected=None,
            observed=observed,
        )
        return passed

    passed &= add_check(
        rows,
        case=case,
        check=f"{prefix}.present",
        expected=True,
        observed=observed is not None,
    )

    if observed is None:
        return False

    for statistic in (
        "mean",
        "std",
        "min",
        "max",
    ):
        passed &= add_check(
            rows,
            case=case,
            check=f"{prefix}.{statistic}",
            expected=expected[statistic],
            observed=observed[statistic],
        )

    return passed


def run_constructor_validation(
    FaceTemporalAggregator,
    rows,
):
    """Validate constructor constraints and frame-window derivation."""
    case = "constructor_and_window"

    aggregator = FaceTemporalAggregator(
        fps=29.97,
        window_sec=5.0,
    )

    expected_window_frames = max(
        1,
        int(round(29.97 * 5.0)),
    )

    add_check(
        rows,
        case=case,
        check="window_frames",
        expected=expected_window_frames,
        observed=aggregator.window_frames,
    )

    for check_name, kwargs in (
        (
            "fps_zero_rejected",
            {"fps": 0, "window_sec": 1.0},
        ),
        (
            "fps_negative_rejected",
            {"fps": -1, "window_sec": 1.0},
        ),
        (
            "window_zero_rejected",
            {"fps": 25, "window_sec": 0},
        ),
        (
            "window_negative_rejected",
            {"fps": 25, "window_sec": -1},
        ),
    ):
        rejected = False

        try:
            FaceTemporalAggregator(**kwargs)
        except ValueError:
            rejected = True

        add_check(
            rows,
            case=case,
            check=check_name,
            expected=True,
            observed=rejected,
        )


def run_full_numeric_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate all current numerical aggregation fields independently."""
    case = "full_numeric_summary"

    aggregator = FaceTemporalAggregator(
        fps=4,
        window_sec=2.0,
    )

    samples = [
        {
            "yaw": -20.0,
            "pitch": 5.0,
            "roll": 1.0,
            "eye": 0.10,
            "gaze_x": 0.20,
            "gaze_y": 0.70,
            "mouth": 0.05,
            "movement": 0.00,
            "brightness": 0.20,
            "sharpness": 50.0,
            "face_area_ratio": 0.08,
            "blink": False,
            "emotion": "Neutral",
        },
        {
            "yaw": -10.0,
            "pitch": 10.0,
            "roll": 2.0,
            "eye": 0.20,
            "gaze_x": 0.30,
            "gaze_y": 0.60,
            "mouth": 0.15,
            "movement": 0.10,
            "brightness": 0.40,
            "sharpness": 100.0,
            "face_area_ratio": 0.10,
            "blink": True,
            "emotion": "Happiness",
        },
        {
            "yaw": 10.0,
            "pitch": 15.0,
            "roll": 3.0,
            "eye": 0.30,
            "gaze_x": 0.40,
            "gaze_y": 0.50,
            "mouth": 0.25,
            "movement": 0.20,
            "brightness": 0.60,
            "sharpness": 150.0,
            "face_area_ratio": 0.12,
            "blink": False,
            "emotion": "Happiness",
        },
        {
            "yaw": 20.0,
            "pitch": 20.0,
            "roll": 4.0,
            "eye": 0.40,
            "gaze_x": 0.50,
            "gaze_y": 0.40,
            "mouth": 0.35,
            "movement": 0.30,
            "brightness": 0.80,
            "sharpness": 200.0,
            "face_area_ratio": 0.14,
            "blink": True,
            "emotion": "Happiness",
        },
    ]

    summary = None

    for sample in samples:
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                **sample,
            )
        )

    add_check(
        rows,
        case=case,
        check="person_id",
        expected=1,
        observed=summary["person_id"],
    )
    add_check(
        rows,
        case=case,
        check="window_frames",
        expected=4,
        observed=summary["window_frames"],
    )
    add_check(
        rows,
        case=case,
        check="window_sec",
        expected=1.0,
        observed=summary["window_sec"],
    )

    mappings = (
        (
            "head_pose.yaw",
            [sample["yaw"] for sample in samples],
            summary["head_pose"]["yaw"],
        ),
        (
            "head_pose.pitch",
            [sample["pitch"] for sample in samples],
            summary["head_pose"]["pitch"],
        ),
        (
            "head_pose.roll",
            [sample["roll"] for sample in samples],
            summary["head_pose"]["roll"],
        ),
        (
            "eyes.mean_openness",
            [sample["eye"] for sample in samples],
            summary["eyes"]["mean_openness"],
        ),
        (
            "gaze.mean_iris_x",
            [sample["gaze_x"] for sample in samples],
            summary["gaze"]["mean_iris_x"],
        ),
        (
            "gaze.mean_iris_y",
            [sample["gaze_y"] for sample in samples],
            summary["gaze"]["mean_iris_y"],
        ),
        (
            "mouth.openness",
            [sample["mouth"] for sample in samples],
            summary["mouth"]["openness"],
        ),
        (
            "mouth.movement",
            [sample["movement"] for sample in samples],
            summary["mouth"]["movement"],
        ),
        (
            "quality.brightness",
            [sample["brightness"] for sample in samples],
            summary["quality"]["brightness"],
        ),
        (
            "quality.sharpness",
            [sample["sharpness"] for sample in samples],
            summary["quality"]["sharpness"],
        ),
        (
            "quality.face_area_ratio",
            [sample["face_area_ratio"] for sample in samples],
            summary["quality"]["face_area_ratio"],
        ),
    )

    for prefix, values, observed in mappings:
        add_summary_checks(
            rows,
            case=case,
            prefix=prefix,
            expected=independent_numeric_summary(values),
            observed=observed,
        )

    expected_blinks = sum(
        1
        for sample in samples
        if sample["blink"]
    )

    add_check(
        rows,
        case=case,
        check="blink.events",
        expected=expected_blinks,
        observed=summary["blink"]["events"],
    )

    expected_emotion = Counter(
        sample["emotion"]
        for sample in samples
    ).most_common(1)[0][0]

    add_check(
        rows,
        case=case,
        check="emotion.dominant",
        expected=expected_emotion,
        observed=summary["emotion"]["dominant"],
    )


def run_availability_and_finite_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate availability filtering and non-finite numerical handling."""
    case = "availability_and_non_finite"

    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    samples = [
        create_instance(
            Instance,
            1,
            eye=0.2,
            gaze_x=0.1,
            gaze_y=0.9,
            mouth=0.1,
            movement=0.01,
            brightness=0.3,
            sharpness=30.0,
            face_area_ratio=0.05,
            emotion="Neutral",
        ),
        create_instance(
            Instance,
            1,
            eye=0.9,
            gaze_x=0.9,
            gaze_y=0.1,
            mouth=0.9,
            movement=0.9,
            brightness=0.9,
            sharpness=900.0,
            face_area_ratio=0.9,
            emotion="Sadness",
            eyes_available=False,
            gaze_available=False,
            mouth_available=False,
            mouth_motion_available=False,
            quality_available=False,
            emotion_available=False,
        ),
        create_instance(
            Instance,
            1,
            eye=np.nan,
            gaze_x=np.inf,
            gaze_y=-np.inf,
            mouth=np.nan,
            movement=np.inf,
            brightness=np.nan,
            sharpness=np.inf,
            face_area_ratio=-np.inf,
            emotion="Neutral",
        ),
        create_instance(
            Instance,
            1,
            eye=0.4,
            gaze_x=0.3,
            gaze_y=0.7,
            mouth=0.3,
            movement=0.03,
            brightness=0.5,
            sharpness=50.0,
            face_area_ratio=0.15,
            emotion="Neutral",
        ),
    ]

    summary = None

    for instance in samples:
        summary = aggregator.update(instance)

    expected_values = {
        "eyes.mean_openness": (
            [0.2, np.nan, 0.4],
            summary["eyes"]["mean_openness"],
        ),
        "gaze.mean_iris_x": (
            [0.1, np.inf, 0.3],
            summary["gaze"]["mean_iris_x"],
        ),
        "gaze.mean_iris_y": (
            [0.9, -np.inf, 0.7],
            summary["gaze"]["mean_iris_y"],
        ),
        "mouth.openness": (
            [0.1, np.nan, 0.3],
            summary["mouth"]["openness"],
        ),
        "mouth.movement": (
            [0.01, np.inf, 0.03],
            summary["mouth"]["movement"],
        ),
        "quality.brightness": (
            [0.3, np.nan, 0.5],
            summary["quality"]["brightness"],
        ),
        "quality.sharpness": (
            [30.0, np.inf, 50.0],
            summary["quality"]["sharpness"],
        ),
        "quality.face_area_ratio": (
            [0.05, -np.inf, 0.15],
            summary["quality"]["face_area_ratio"],
        ),
    }

    for prefix, (values, observed) in expected_values.items():
        add_summary_checks(
            rows,
            case=case,
            prefix=prefix,
            expected=independent_numeric_summary(values),
            observed=observed,
        )

    add_check(
        rows,
        case=case,
        check="emotion.dominant",
        expected="Neutral",
        observed=summary["emotion"]["dominant"],
    )


def run_sliding_window_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate bounded rolling-window truncation."""
    case = "sliding_window"

    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    summaries = []

    for frame_index in range(10):
        summaries.append(
            aggregator.update(
                create_instance(
                    Instance,
                    1,
                    yaw=float(frame_index),
                    eye=float(frame_index) / 10.0,
                    brightness=float(frame_index) / 10.0,
                    emotion=(
                        "Neutral"
                        if frame_index < 5
                        else "Happiness"
                    ),
                )
            )
        )

    final_summary = summaries[-1]
    expected_values = list(range(4, 10))

    add_check(
        rows,
        case=case,
        check="configured_window_frames",
        expected=6,
        observed=aggregator.window_frames,
    )
    add_check(
        rows,
        case=case,
        check="final_window_frames",
        expected=6,
        observed=final_summary["window_frames"],
    )
    add_check(
        rows,
        case=case,
        check="final_window_sec",
        expected=3.0,
        observed=final_summary["window_sec"],
    )

    add_summary_checks(
        rows,
        case=case,
        prefix="head_pose.yaw",
        expected=independent_numeric_summary(expected_values),
        observed=final_summary["head_pose"]["yaw"],
    )

    add_summary_checks(
        rows,
        case=case,
        prefix="eyes.mean_openness",
        expected=independent_numeric_summary(
            [value / 10.0 for value in expected_values]
        ),
        observed=final_summary["eyes"]["mean_openness"],
    )

    add_check(
        rows,
        case=case,
        check="emotion.dominant",
        expected="Happiness",
        observed=final_summary["emotion"]["dominant"],
    )


def run_person_isolation_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate independent rolling state for different tracked identities."""
    case = "person_isolation"

    aggregator = FaceTemporalAggregator(
        fps=5,
        window_sec=2.0,
    )

    person_1_values = (1.0, 2.0, 3.0)
    person_2_values = (-10.0, -20.0)

    summary_1 = None
    summary_2 = None

    for value in person_1_values:
        summary_1 = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=value,
            )
        )

    for value in person_2_values:
        summary_2 = aggregator.update(
            create_instance(
                Instance,
                2,
                yaw=value,
            )
        )

    add_check(
        rows,
        case=case,
        check="person_1_id",
        expected=1,
        observed=summary_1["person_id"],
    )
    add_check(
        rows,
        case=case,
        check="person_2_id",
        expected=2,
        observed=summary_2["person_id"],
    )

    add_summary_checks(
        rows,
        case=case,
        prefix="person_1.head_pose.yaw",
        expected=independent_numeric_summary(person_1_values),
        observed=aggregator.summary(1)["head_pose"]["yaw"],
    )

    add_summary_checks(
        rows,
        case=case,
        prefix="person_2.head_pose.yaw",
        expected=independent_numeric_summary(person_2_values),
        observed=aggregator.summary(2)["head_pose"]["yaw"],
    )

    add_check(
        rows,
        case=case,
        check="buffer_count",
        expected=2,
        observed=len(aggregator.buffers),
    )


def run_reset_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate per-person and global temporal reset behavior."""
    case = "reset_behavior"

    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    aggregator.update(
        create_instance(
            Instance,
            1,
            yaw=10.0,
        )
    )
    aggregator.update(
        create_instance(
            Instance,
            2,
            yaw=20.0,
        )
    )

    aggregator.reset(person_id=1)

    add_check(
        rows,
        case=case,
        check="person_1_reset",
        expected=None,
        observed=aggregator.summary(1),
    )
    add_check(
        rows,
        case=case,
        check="person_2_preserved",
        expected=True,
        observed=aggregator.summary(2) is not None,
    )

    aggregator.reset()

    add_check(
        rows,
        case=case,
        check="global_reset_person_2",
        expected=None,
        observed=aggregator.summary(2),
    )
    add_check(
        rows,
        case=case,
        check="global_reset_buffer_count",
        expected=0,
        observed=len(aggregator.buffers),
    )


def run_rejected_update_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate rejection of untracked or featureless instances."""
    case = "rejected_updates"

    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    no_id_result = aggregator.update(
        create_instance(
            Instance,
            None,
        )
    )

    add_check(
        rows,
        case=case,
        check="no_person_id_returns_none",
        expected=None,
        observed=no_id_result,
    )
    add_check(
        rows,
        case=case,
        check="no_person_id_does_not_create_buffer",
        expected=0,
        observed=len(aggregator.buffers),
    )

    no_features_result = aggregator.update(
        create_instance(
            Instance,
            1,
            face_features=False,
        )
    )

    add_check(
        rows,
        case=case,
        check="no_face_features_returns_none",
        expected=None,
        observed=no_features_result,
    )
    add_check(
        rows,
        case=case,
        check="no_face_features_does_not_create_buffer",
        expected=0,
        observed=len(aggregator.buffers),
    )


def run_empty_available_feature_validation(
    FaceTemporalAggregator,
    Instance,
    rows,
):
    """Validate None summaries when no finite available numerical value exists."""
    case = "empty_numeric_feature"

    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    summary = aggregator.update(
        create_instance(
            Instance,
            1,
            eye=np.nan,
            gaze_x=np.nan,
            gaze_y=np.nan,
            mouth=np.nan,
            movement=np.nan,
            brightness=np.nan,
            sharpness=np.nan,
            face_area_ratio=np.nan,
            emotion=None,
        )
    )

    for prefix, observed in (
        (
            "eyes.mean_openness",
            summary["eyes"]["mean_openness"],
        ),
        (
            "gaze.mean_iris_x",
            summary["gaze"]["mean_iris_x"],
        ),
        (
            "gaze.mean_iris_y",
            summary["gaze"]["mean_iris_y"],
        ),
        (
            "mouth.openness",
            summary["mouth"]["openness"],
        ),
        (
            "mouth.movement",
            summary["mouth"]["movement"],
        ),
        (
            "quality.brightness",
            summary["quality"]["brightness"],
        ),
        (
            "quality.sharpness",
            summary["quality"]["sharpness"],
        ),
        (
            "quality.face_area_ratio",
            summary["quality"]["face_area_ratio"],
        ),
    ):
        add_check(
            rows,
            case=case,
            check=prefix,
            expected=None,
            observed=observed,
        )

    add_check(
        rows,
        case=case,
        check="emotion.dominant",
        expected=None,
        observed=summary["emotion"]["dominant"],
    )


def validate_rows(rows):
    """Validate result-table integrity."""
    if not rows:
        raise RuntimeError(
            "No validation checks were generated"
        )

    duplicate_keys = set()
    seen = set()

    for row in rows:
        key = (
            row["case"],
            row["check"],
        )

        if key in seen:
            duplicate_keys.add(key)

        seen.add(key)

    if duplicate_keys:
        raise RuntimeError(
            "Duplicate validation check keys found: "
            f"{sorted(duplicate_keys)[:10]}"
        )

    invalid_statuses = {
        row["status"]
        for row in rows
        if row["status"] not in {"PASS", "FAIL"}
    }

    if invalid_statuses:
        raise RuntimeError(
            "Unexpected validation status values: "
            f"{sorted(invalid_statuses)}"
        )


def build_metrics(rows):
    """Create compact aggregate correctness metrics."""
    total_checks = len(rows)
    passed_checks = sum(
        1
        for row in rows
        if row["status"] == "PASS"
    )
    failed_checks = total_checks - passed_checks

    numeric_errors = [
        float(row["absolute_error"])
        for row in rows
        if row["absolute_error"] != ""
    ]

    max_absolute_error = (
        max(numeric_errors)
        if numeric_errors
        else 0.0
    )

    case_names = sorted(
        {row["case"] for row in rows}
    )

    return [
        {
            "metric": "validation_cases",
            "value": len(case_names),
        },
        {
            "metric": "total_checks",
            "value": total_checks,
        },
        {
            "metric": "passed_checks",
            "value": passed_checks,
        },
        {
            "metric": "failed_checks",
            "value": failed_checks,
        },
        {
            "metric": "pass_rate",
            "value": passed_checks / total_checks,
        },
        {
            "metric": "max_numeric_absolute_error",
            "value": max_absolute_error,
        },
        {
            "metric": "overall_status",
            "value": "PASS" if failed_checks == 0 else "FAIL",
        },
    ]


def write_csv(
    path,
    rows,
    fields,
):
    """Write a CSV file with a stable schema."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path,
    *,
    implementation_path,
    rows,
    metrics,
):
    """Write the textual scientific validation record."""
    metric_map = {
        row["metric"]: row["value"]
        for row in metrics
    }

    case_status = {}

    for row in rows:
        case_status.setdefault(
            row["case"],
            True,
        )
        case_status[row["case"]] &= row["status"] == "PASS"

    lines = [
        "PhysioTrack Temporal Aggregation Correctness Validation",
        "=======================================================",
        "",
        "Purpose",
        "-------",
        (
            "Validate the deterministic mathematical and state-management "
            "behavior of the current PhysioTrack FaceTemporalAggregator "
            "implementation using controlled sequences with independently "
            "computed expected results."
        ),
        "",
        "Implementation",
        "--------------",
        str(
            implementation_path.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "",
        "Validation Scope",
        "----------------",
        (
            "The validation covers constructor constraints, temporal-window "
            "derivation, population mean/std/min/max statistics, availability "
            "filtering, non-finite numerical filtering, bounded sliding-window "
            "behavior, per-person state isolation, per-person and global reset "
            "behavior, tracked-instance requirements, blink-event aggregation, "
            "dominant-emotion aggregation, and all numerical feature groups "
            "currently summarized by the component."
        ),
        "",
        (
            "This is a deterministic correctness validation, not an external "
            "predictive-accuracy benchmark. It does not establish the scientific "
            "accuracy of upstream predictive components."
        ),
        "",
        "Results",
        "-------",
        f"Validation cases: {metric_map['validation_cases']}",
        f"Total checks: {metric_map['total_checks']}",
        f"Passed checks: {metric_map['passed_checks']}",
        f"Failed checks: {metric_map['failed_checks']}",
        (
            "Pass rate: "
            f"{float(metric_map['pass_rate']):.6f}"
        ),
        (
            "Maximum numerical absolute error: "
            f"{float(metric_map['max_numeric_absolute_error']):.16g}"
        ),
        "",
        "Case Status",
        "-----------",
    ]

    for case in sorted(case_status):
        lines.append(
            f"{case}: {'PASS' if case_status[case] else 'FAIL'}"
        )

    lines.extend(
        [
            "",
            "Overall Status",
            "--------------",
            str(metric_map["overall_status"]),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def validate_staged_outputs(staging_dir):
    """Validate staged artifacts before final promotion."""
    required_paths = (
        staging_dir / RESULTS_PATH.name,
        staging_dir / METRICS_PATH.name,
        staging_dir / SUMMARY_PATH.name,
    )

    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(
                f"Expected staged output is missing: {path.name}"
            )

        if path.stat().st_size <= 0:
            raise RuntimeError(
                f"Expected staged output is empty: {path.name}"
            )

    with (
        staging_dir / RESULTS_PATH.name
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise RuntimeError(
                "Staged result CSV header does not match the expected schema"
            )

        result_rows = list(reader)

    if not result_rows:
        raise RuntimeError(
            "Staged result CSV contains no data rows"
        )

    with (
        staging_dir / METRICS_PATH.name
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise RuntimeError(
                "Staged metrics CSV header does not match the expected schema"
            )

        metric_rows = list(reader)

    metric_map = {
        row["metric"]: row["value"]
        for row in metric_rows
    }

    if metric_map.get("overall_status") != "PASS":
        raise RuntimeError(
            "Staged validation did not achieve overall PASS"
        )

    if any(
        row["status"] != "PASS"
        for row in result_rows
    ):
        raise RuntimeError(
            "Staged result CSV contains failed checks"
        )


def promote_outputs(staging_dir):
    """Replace only script-owned final outputs with rollback protection."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_paths = (
        RESULTS_PATH,
        METRICS_PATH,
        SUMMARY_PATH,
    )

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=".temporal_aggregation_rollback_",
            dir=str(RESULTS_DIR),
        )
    )

    replaced_paths = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                shutil.copy2(
                    final_path,
                    backup_dir / final_path.name,
                )

        for final_path in final_paths:
            staged_path = staging_dir / final_path.name
            os.replace(
                staged_path,
                final_path,
            )
            replaced_paths.append(final_path)

    except Exception:
        for final_path in replaced_paths:
            if final_path.exists():
                final_path.unlink()

        for backup_path in backup_dir.iterdir():
            os.replace(
                backup_path,
                RESULTS_DIR / backup_path.name,
            )

        raise

    finally:
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def run_validation(
    FaceTemporalAggregator,
    Instance,
):
    """Run the complete controlled correctness validation."""
    rows = []

    run_constructor_validation(
        FaceTemporalAggregator,
        rows,
    )
    run_full_numeric_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_availability_and_finite_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_sliding_window_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_person_isolation_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_reset_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_rejected_update_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )
    run_empty_available_feature_validation(
        FaceTemporalAggregator,
        Instance,
        rows,
    )

    validate_rows(rows)
    metrics = build_metrics(rows)

    return rows, metrics


def main():
    """Run preflight, correctness validation, staging, and safe promotion."""
    print(
        "PhysioTrack Temporal Aggregation Correctness Validation"
    )
    print(
        "======================================================="
    )

    (
        FaceTemporalAggregator,
        Instance,
        implementation_path,
    ) = import_physio_track()

    print("Preflight: PASS")
    print(
        "Implementation: "
        f"{implementation_path.relative_to(REPO_ROOT)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".temporal_aggregation_staging_",
            dir=str(RESULTS_DIR),
        )
    )

    try:
        rows, metrics = run_validation(
            FaceTemporalAggregator,
            Instance,
        )

        metric_map = {
            row["metric"]: row["value"]
            for row in metrics
        }

        write_csv(
            staging_dir / RESULTS_PATH.name,
            rows,
            RESULT_FIELDS,
        )
        write_csv(
            staging_dir / METRICS_PATH.name,
            metrics,
            METRIC_FIELDS,
        )
        write_summary(
            staging_dir / SUMMARY_PATH.name,
            implementation_path=implementation_path,
            rows=rows,
            metrics=metrics,
        )

        validate_staged_outputs(staging_dir)

        print(
            f"Validation cases: {metric_map['validation_cases']}"
        )
        print(
            f"Total checks: {metric_map['total_checks']}"
        )
        print(
            f"Passed checks: {metric_map['passed_checks']}"
        )
        print(
            f"Failed checks: {metric_map['failed_checks']}"
        )
        print(
            "Maximum numerical absolute error: "
            f"{float(metric_map['max_numeric_absolute_error']):.16g}"
        )
        print("Staged outputs: PASS")

        if metric_map["overall_status"] != "PASS":
            raise RuntimeError(
                "Temporal Aggregation correctness validation failed"
            )

        promote_outputs(staging_dir)

        print("Final output promotion: PASS")
        print("Overall status: PASS")

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
