from collections import Counter, deque
from pathlib import Path
import csv
import json
import math
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from temporal_aggregation_eval import (
    create_instance,
    import_physio_track,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
QUALITATIVE_DIR = RESULTS_DIR / "qualitative"
FIGURES_DIR = RESULTS_DIR / "figures"

MANIFEST_PATH = (
    QUALITATIVE_DIR
    / "temporal_aggregation_qualitative_manifest.csv"
)
CONTACT_SHEET_PATH = (
    FIGURES_DIR
    / "temporal_aggregation_qualitative_contact_sheet.png"
)

CASE_FIELDS = (
    "case_id",
    "case_name",
    "figure_file",
    "description",
    "status",
)

NUMERIC_TOLERANCE = 1e-12


def independent_summary(
    values,
):
    """Calculate population statistics independently."""
    valid = []

    for value in values:
        if value is None:
            continue

        numeric = float(
            value
        )

        if not math.isfinite(
            numeric
        ):
            continue

        valid.append(
            numeric
        )

    if not valid:
        return None

    mean = sum(
        valid
    ) / len(
        valid
    )

    variance = sum(
        (
            value
            - mean
        ) ** 2
        for value in valid
    ) / len(
        valid
    )

    return {
        "mean": mean,
        "std": math.sqrt(
            variance
        ),
        "min": min(
            valid
        ),
        "max": max(
            valid
        ),
    }


def assert_close(
    expected,
    observed,
    *,
    label,
):
    """Raise when one numerical value differs beyond tolerance."""
    if expected is None:
        if observed is not None:
            raise RuntimeError(
                f"{label}: expected None, observed {observed}"
            )
        return

    if observed is None:
        raise RuntimeError(
            f"{label}: observed value is missing"
        )

    error = abs(
        float(
            observed
        )
        - float(
            expected
        )
    )

    if error > NUMERIC_TOLERANCE:
        raise RuntimeError(
            f"{label}: absolute error {error} exceeds tolerance"
        )


def make_case_window_growth(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize growth of a temporal window before saturation."""
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    raw_yaw = [
        5.0,
        10.0,
        20.0,
        15.0,
        30.0,
        25.0,
    ]

    observed_means = []
    expected_means = []
    observed_sizes = []
    expected_sizes = []

    for index, yaw in enumerate(
        raw_yaw,
        start=1,
    ):
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=yaw,
            )
        )

        window_values = raw_yaw[
            :index
        ]
        expected = independent_summary(
            window_values
        )

        observed_means.append(
            summary[
                "head_pose"
            ][
                "yaw"
            ][
                "mean"
            ]
        )
        expected_means.append(
            expected[
                "mean"
            ]
        )
        observed_sizes.append(
            summary[
                "window_frames"
            ]
        )
        expected_sizes.append(
            index
        )

        assert_close(
            expected[
                "mean"
            ],
            observed_means[
                -1
            ],
            label=(
                "window_growth.yaw.mean"
            ),
        )

        if observed_sizes[
            -1
        ] != expected_sizes[
            -1
        ]:
            raise RuntimeError(
                "Window growth frame count mismatch"
            )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    frames = list(
        range(
            1,
            len(
                raw_yaw
            )
            + 1,
        )
    )

    axis.plot(
        frames,
        raw_yaw,
        marker="o",
        label="Raw yaw",
    )
    axis.plot(
        frames,
        observed_means,
        marker="s",
        label="Observed rolling mean",
    )
    axis.plot(
        frames,
        expected_means,
        linestyle="--",
        label="Independent expected mean",
    )

    axis.set_xlabel(
        "Frame index"
    )
    axis.set_ylabel(
        "Yaw (degrees)"
    )
    axis.set_title(
        "Temporal Window Growth Before Saturation"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
    )

    second_axis = axis.twinx()
    second_axis.plot(
        frames,
        observed_sizes,
        marker="x",
        label="Window frames",
    )
    second_axis.set_ylabel(
        "Window frames"
    )
    second_axis.set_ylim(
        0,
        max(
            expected_sizes
        )
        + 1,
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


def make_case_sliding_window(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize rolling replacement after the window is full."""
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    raw_values = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]

    expected_means = []
    observed_means = []

    for index, value in enumerate(
        raw_values,
    ):
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=value,
            )
        )

        start = max(
            0,
            index
            - aggregator.window_frames
            + 1,
        )

        window = raw_values[
            start:
            index
            + 1
        ]

        expected = independent_summary(
            window
        )[
            "mean"
        ]

        observed = summary[
            "head_pose"
        ][
            "yaw"
        ][
            "mean"
        ]

        assert_close(
            expected,
            observed,
            label=(
                "sliding_window.yaw.mean"
            ),
        )

        expected_means.append(
            expected
        )
        observed_means.append(
            observed
        )

    frames = list(
        range(
            1,
            len(
                raw_values
            )
            + 1,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    axis.plot(
        frames,
        raw_values,
        marker="o",
        label="Raw yaw",
    )
    axis.plot(
        frames,
        observed_means,
        marker="s",
        label="Observed rolling mean",
    )
    axis.plot(
        frames,
        expected_means,
        linestyle="--",
        label="Independent expected mean",
    )

    axis.axvline(
        aggregator.window_frames,
        linestyle=":",
        label="Window reaches full size",
    )

    axis.set_xlabel(
        "Frame index"
    )
    axis.set_ylabel(
        "Yaw (degrees)"
    )
    axis.set_title(
        "Sliding-Window Replacement After Saturation"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def make_case_person_isolation(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize independent per-person temporal state."""
    aggregator = FaceTemporalAggregator(
        fps=5,
        window_sec=2.0,
    )

    person_1 = [
        5.0,
        10.0,
        15.0,
        20.0,
    ]
    person_2 = [
        -5.0,
        -10.0,
        -15.0,
        -20.0,
    ]

    means_1 = []
    means_2 = []

    for value_1, value_2 in zip(
        person_1,
        person_2,
    ):
        summary_1 = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=value_1,
            )
        )
        summary_2 = aggregator.update(
            create_instance(
                Instance,
                2,
                yaw=value_2,
            )
        )

        means_1.append(
            summary_1[
                "head_pose"
            ][
                "yaw"
            ][
                "mean"
            ]
        )
        means_2.append(
            summary_2[
                "head_pose"
            ][
                "yaw"
            ][
                "mean"
            ]
        )

    expected_1 = [
        independent_summary(
            person_1[
                :index
                + 1
            ]
        )[
            "mean"
        ]
        for index in range(
            len(
                person_1
            )
        )
    ]

    expected_2 = [
        independent_summary(
            person_2[
                :index
                + 1
            ]
        )[
            "mean"
        ]
        for index in range(
            len(
                person_2
            )
        )
    ]

    for expected, observed in zip(
        expected_1,
        means_1,
    ):
        assert_close(
            expected,
            observed,
            label=(
                "person_isolation.person_1"
            ),
        )

    for expected, observed in zip(
        expected_2,
        means_2,
    ):
        assert_close(
            expected,
            observed,
            label=(
                "person_isolation.person_2"
            ),
        )

    frames = list(
        range(
            1,
            len(
                person_1
            )
            + 1,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    axis.plot(
        frames,
        means_1,
        marker="o",
        label="Person 1 rolling yaw mean",
    )
    axis.plot(
        frames,
        means_2,
        marker="s",
        label="Person 2 rolling yaw mean",
    )
    axis.plot(
        frames,
        expected_1,
        linestyle="--",
        label="Person 1 expected mean",
    )
    axis.plot(
        frames,
        expected_2,
        linestyle="--",
        label="Person 2 expected mean",
    )

    axis.set_xlabel(
        "Temporal update"
    )
    axis.set_ylabel(
        "Yaw mean (degrees)"
    )
    axis.set_title(
        "Independent Temporal State for Two Tracked Persons"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def make_case_reset(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize per-person reset and subsequent clean restart."""
    aggregator = FaceTemporalAggregator(
        fps=10,
        window_sec=1.0,
    )

    before_reset = [
        10.0,
        20.0,
        30.0,
    ]

    means_before = []

    for value in before_reset:
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=value,
            )
        )

        means_before.append(
            summary[
                "head_pose"
            ][
                "yaw"
            ][
                "mean"
            ]
        )

    aggregator.reset(
        person_id=1
    )

    after_reset = [
        100.0,
        120.0,
    ]

    means_after = []

    for value in after_reset:
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                yaw=value,
            )
        )

        means_after.append(
            summary[
                "head_pose"
            ][
                "yaw"
            ][
                "mean"
            ]
        )

    expected_after = [
        100.0,
        110.0,
    ]

    for expected, observed in zip(
        expected_after,
        means_after,
    ):
        assert_close(
            expected,
            observed,
            label=(
                "reset_behavior.restart"
            ),
        )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    x_before = [
        1,
        2,
        3,
    ]
    x_after = [
        5,
        6,
    ]

    axis.plot(
        x_before,
        means_before,
        marker="o",
        label="Rolling mean before reset",
    )
    axis.plot(
        x_after,
        means_after,
        marker="s",
        label="Rolling mean after reset",
    )

    axis.axvline(
        4,
        linestyle="--",
        label="Temporal state reset",
    )

    axis.set_xlabel(
        "Temporal update"
    )
    axis.set_ylabel(
        "Yaw mean (degrees)"
    )
    axis.set_title(
        "Per-Person Temporal Reset and Clean Restart"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def make_case_blink_events(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize blink-event count within a sliding temporal window."""
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    blink_sequence = [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
    ]

    observed_counts = []
    expected_counts = []

    for index, blink in enumerate(
        blink_sequence,
    ):
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                blink=blink,
            )
        )

        start = max(
            0,
            index
            - aggregator.window_frames
            + 1,
        )

        window = blink_sequence[
            start:
            index
            + 1
        ]

        expected = sum(
            1
            for item in window
            if item
        )

        observed = summary[
            "blink"
        ][
            "events"
        ]

        if observed != expected:
            raise RuntimeError(
                "Blink event aggregation mismatch"
            )

        expected_counts.append(
            expected
        )
        observed_counts.append(
            observed
        )

    frames = list(
        range(
            1,
            len(
                blink_sequence
            )
            + 1,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    axis.step(
        frames,
        observed_counts,
        where="mid",
        label="Observed blink events in window",
    )
    axis.plot(
        frames,
        expected_counts,
        linestyle="--",
        label="Independent expected count",
    )

    blink_frames = [
        frame
        for frame, blink in zip(
            frames,
            blink_sequence,
        )
        if blink
    ]

    axis.scatter(
        blink_frames,
        [
            0
        ]
        * len(
            blink_frames
        ),
        marker="x",
        label="Blink-event frames",
    )

    axis.set_xlabel(
        "Frame index"
    )
    axis.set_ylabel(
        "Blink events in current window"
    )
    axis.set_title(
        "Blink-Event Aggregation Across the Sliding Window"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def make_case_dominant_emotion(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize dominant-emotion changes inside a rolling window."""
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    emotions = [
        "Neutral",
        "Neutral",
        "Happiness",
        "Happiness",
        "Happiness",
        "Neutral",
        "Sadness",
        "Sadness",
        "Sadness",
        "Sadness",
    ]

    observed = []
    expected = []

    for index, emotion in enumerate(
        emotions,
    ):
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                emotion=emotion,
            )
        )

        start = max(
            0,
            index
            - aggregator.window_frames
            + 1,
        )

        window = emotions[
            start:
            index
            + 1
        ]

        expected_emotion = Counter(
            window
        ).most_common(
            1
        )[0][0]

        observed_emotion = summary[
            "emotion"
        ][
            "dominant"
        ]

        if observed_emotion != expected_emotion:
            raise RuntimeError(
                "Dominant emotion aggregation mismatch"
            )

        observed.append(
            observed_emotion
        )
        expected.append(
            expected_emotion
        )

    categories = [
        "Neutral",
        "Happiness",
        "Sadness",
    ]

    mapping = {
        emotion: index
        for index, emotion in enumerate(
            categories
        )
    }

    frames = list(
        range(
            1,
            len(
                emotions
            )
            + 1,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    axis.scatter(
        frames,
        [
            mapping[
                emotion
            ]
            for emotion in emotions
        ],
        label="Raw frame emotion",
    )
    axis.step(
        frames,
        [
            mapping[
                emotion
            ]
            for emotion in observed
        ],
        where="mid",
        label="Observed dominant emotion",
    )
    axis.plot(
        frames,
        [
            mapping[
                emotion
            ]
            for emotion in expected
        ],
        linestyle="--",
        label="Independent expected dominant emotion",
    )

    axis.set_yticks(
        list(
            mapping.values()
        )
    )
    axis.set_yticklabels(
        categories
    )
    axis.set_xlabel(
        "Frame index"
    )
    axis.set_ylabel(
        "Emotion"
    )
    axis.set_title(
        "Dominant Emotion Within the Sliding Temporal Window"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def make_case_quality_summary(
    FaceTemporalAggregator,
    Instance,
    output_path,
):
    """Visualize temporal aggregation of Face Quality brightness."""
    aggregator = FaceTemporalAggregator(
        fps=2,
        window_sec=3.0,
    )

    brightness = [
        0.20,
        0.25,
        0.30,
        0.55,
        0.65,
        0.75,
        0.50,
        0.40,
    ]

    observed_means = []
    expected_means = []

    for index, value in enumerate(
        brightness,
    ):
        summary = aggregator.update(
            create_instance(
                Instance,
                1,
                brightness=value,
            )
        )

        start = max(
            0,
            index
            - aggregator.window_frames
            + 1,
        )

        window = brightness[
            start:
            index
            + 1
        ]

        expected = independent_summary(
            window
        )[
            "mean"
        ]

        observed = summary[
            "quality"
        ][
            "brightness"
        ][
            "mean"
        ]

        assert_close(
            expected,
            observed,
            label=(
                "quality.brightness.mean"
            ),
        )

        observed_means.append(
            observed
        )
        expected_means.append(
            expected
        )

    frames = list(
        range(
            1,
            len(
                brightness
            )
            + 1,
        )
    )

    figure, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    axis.plot(
        frames,
        brightness,
        marker="o",
        label="Raw brightness",
    )
    axis.plot(
        frames,
        observed_means,
        marker="s",
        label="Observed rolling mean",
    )
    axis.plot(
        frames,
        expected_means,
        linestyle="--",
        label="Independent expected mean",
    )

    axis.set_xlabel(
        "Frame index"
    )
    axis.set_ylabel(
        "Normalized brightness"
    )
    axis.set_title(
        "Temporal Aggregation of Face Quality Brightness"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
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


def build_contact_sheet(
    figure_paths,
    output_path,
):
    """Combine representative temporal-validation figures in one sheet."""
    images = [
        plt.imread(
            path
        )
        for path in figure_paths
    ]

    columns = 2
    rows = math.ceil(
        len(
            images
        )
        / columns
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            16,
            5.5
            * rows,
        ),
    )

    axes = np.array(
        axes
    ).reshape(
        -1
    )

    for axis, image, path in zip(
        axes,
        images,
        figure_paths,
    ):
        axis.imshow(
            image
        )
        axis.set_title(
            path.stem.replace(
                "_",
                " ",
            ).title()
        )
        axis.axis(
            "off"
        )

    for axis in axes[
        len(
            images
        ):
    ]:
        axis.axis(
            "off"
        )

    figure.suptitle(
        "Temporal Aggregation Representative Correctness Evidence",
        fontsize=16,
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.98,
        )
    )
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def validate_image(
    path,
):
    """Verify that a generated figure can be read and is non-empty."""
    if not path.is_file():
        raise RuntimeError(
            f"Missing generated figure: {path.name}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Generated figure is empty: {path.name}"
        )

    image = plt.imread(
        path
    )

    if image.size == 0:
        raise RuntimeError(
            f"Unable to read generated figure: {path.name}"
        )


def write_manifest(
    path,
    rows,
):
    """Write the qualitative evidence manifest."""
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CASE_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def validate_staged_outputs(
    staging_root,
    expected_case_count,
):
    """Validate all staged qualitative evidence before promotion."""
    staged_qualitative = (
        staging_root
        / "qualitative"
    )
    staged_figures = (
        staging_root
        / "figures"
    )

    manifest_path = (
        staged_qualitative
        / MANIFEST_PATH.name
    )
    contact_sheet_path = (
        staged_figures
        / CONTACT_SHEET_PATH.name
    )

    if not manifest_path.is_file():
        raise RuntimeError(
            "Staged qualitative manifest is missing"
        )

    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        if tuple(
            reader.fieldnames
            or ()
        ) != CASE_FIELDS:
            raise RuntimeError(
                "Staged qualitative manifest schema mismatch"
            )

        rows = list(
            reader
        )

    if len(
        rows
    ) != expected_case_count:
        raise RuntimeError(
            "Staged qualitative manifest case count mismatch"
        )

    if any(
        row[
            "status"
        ]
        != "PASS"
        for row in rows
    ):
        raise RuntimeError(
            "Staged qualitative manifest contains failed cases"
        )

    for row in rows:
        validate_image(
            staged_qualitative
            / row[
                "figure_file"
            ]
        )

    validate_image(
        contact_sheet_path
    )


def promote_outputs(
    staging_root,
):
    """Transactionally replace outputs owned by the qualitative script."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    QUALITATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rollback_root = Path(
        tempfile.mkdtemp(
            prefix=".temporal_qualitative_rollback_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    final_qualitative = (
        QUALITATIVE_DIR
    )
    final_contact_sheet = (
        CONTACT_SHEET_PATH
    )

    staging_qualitative = (
        staging_root
        / "qualitative"
    )
    staging_contact_sheet = (
        staging_root
        / "figures"
        / CONTACT_SHEET_PATH.name
    )

    backup_qualitative = (
        rollback_root
        / "qualitative"
    )
    backup_contact_sheet = (
        rollback_root
        / "figures"
        / CONTACT_SHEET_PATH.name
    )

    try:
        if final_qualitative.exists():
            shutil.copytree(
                final_qualitative,
                backup_qualitative,
            )

        if final_contact_sheet.exists():
            backup_contact_sheet.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                final_contact_sheet,
                backup_contact_sheet,
            )

        if final_qualitative.exists():
            shutil.rmtree(
                final_qualitative
            )

        os.replace(
            staging_qualitative,
            final_qualitative,
        )

        final_contact_sheet.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if final_contact_sheet.exists():
            final_contact_sheet.unlink()

        os.replace(
            staging_contact_sheet,
            final_contact_sheet,
        )

    except Exception:
        if final_qualitative.exists():
            shutil.rmtree(
                final_qualitative
            )

        if backup_qualitative.exists():
            shutil.copytree(
                backup_qualitative,
                final_qualitative,
            )

        if final_contact_sheet.exists():
            final_contact_sheet.unlink()

        if backup_contact_sheet.exists():
            final_contact_sheet.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copy2(
                backup_contact_sheet,
                final_contact_sheet,
            )

        raise

    finally:
        shutil.rmtree(
            rollback_root,
            ignore_errors=True,
        )


def main():
    """Generate representative temporal-behavior evidence safely."""
    print(
        "PhysioTrack Temporal Aggregation Qualitative Validation"
    )
    print(
        "======================================================="
    )

    (
        FaceTemporalAggregator,
        Instance,
        implementation_path,
    ) = import_physio_track()

    print(
        "Preflight: PASS"
    )
    print(
        "Implementation: "
        f"{implementation_path.relative_to(SCRIPT_DIR.parents[1])}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".temporal_qualitative_staging_",
            dir=str(
                RESULTS_DIR
            ),
        )
    )

    staged_qualitative = (
        staging_root
        / "qualitative"
    )
    staged_figures = (
        staging_root
        / "figures"
    )

    staged_qualitative.mkdir(
        parents=True,
        exist_ok=True,
    )
    staged_figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = [
        (
            "case_01",
            "Window growth",
            "case_01_window_growth.png",
            (
                "Window-frame growth and rolling yaw mean before the "
                "configured temporal window reaches full capacity."
            ),
            make_case_window_growth,
        ),
        (
            "case_02",
            "Sliding-window replacement",
            "case_02_sliding_window.png",
            (
                "Replacement of old samples after the temporal window "
                "reaches its configured maximum size."
            ),
            make_case_sliding_window,
        ),
        (
            "case_03",
            "Per-person isolation",
            "case_03_person_isolation.png",
            (
                "Independent rolling state for two tracked person IDs "
                "with deliberately opposite yaw trajectories."
            ),
            make_case_person_isolation,
        ),
        (
            "case_04",
            "Temporal reset",
            "case_04_reset_behavior.png",
            (
                "Per-person reset followed by a clean temporal restart "
                "without leakage from pre-reset values."
            ),
            make_case_reset,
        ),
        (
            "case_05",
            "Blink-event aggregation",
            "case_05_blink_events.png",
            (
                "Blink-event counting within the current sliding temporal "
                "window against independently derived event counts."
            ),
            make_case_blink_events,
        ),
        (
            "case_06",
            "Dominant emotion",
            "case_06_dominant_emotion.png",
            (
                "Rolling dominant-emotion behavior compared with an "
                "independently derived categorical majority."
            ),
            make_case_dominant_emotion,
        ),
        (
            "case_07",
            "Face Quality aggregation",
            "case_07_quality_brightness.png",
            (
                "Temporal aggregation of Face Quality brightness using "
                "the current rolling-window implementation."
            ),
            make_case_quality_summary,
        ),
    ]

    manifest_rows = []
    figure_paths = []

    try:
        for (
            case_id,
            case_name,
            filename,
            description,
            function,
        ) in cases:
            output_path = (
                staged_qualitative
                / filename
            )

            function(
                FaceTemporalAggregator,
                Instance,
                output_path,
            )

            validate_image(
                output_path
            )

            manifest_rows.append(
                {
                    "case_id": case_id,
                    "case_name": case_name,
                    "figure_file": filename,
                    "description": description,
                    "status": "PASS",
                }
            )
            figure_paths.append(
                output_path
            )

            print(
                f"{case_id}: PASS"
            )

        write_manifest(
            staged_qualitative
            / MANIFEST_PATH.name,
            manifest_rows,
        )

        build_contact_sheet(
            figure_paths,
            staged_figures
            / CONTACT_SHEET_PATH.name,
        )

        validate_staged_outputs(
            staging_root,
            len(
                cases
            ),
        )

        print(
            f"Representative cases: {len(cases)}"
        )
        print(
            "Staged outputs: PASS"
        )

        promote_outputs(
            staging_root
        )

        print(
            "Final output promotion: PASS"
        )
        print(
            "Overall status: PASS"
        )

    finally:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()
