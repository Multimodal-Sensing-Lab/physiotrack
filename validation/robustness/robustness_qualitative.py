from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "generated_controlled"
)
RESULTS_DIR = (
    SCRIPT_DIR
    / "results"
)
FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)
QUALITATIVE_DIR = (
    RESULTS_DIR
    / "qualitative"
)
ANNOTATED_DIR = (
    QUALITATIVE_DIR
    / "annotated_images"
)

RESULTS_CSV = (
    RESULTS_DIR
    / "robustness_results.csv"
)
METADATA_CSV = (
    TEST_DATA_DIR
    / "robustness_generated_cases.csv"
)

SELECTION_CSV = (
    QUALITATIVE_DIR
    / "robustness_qualitative_selection.csv"
)
COMBINED_FIGURE = (
    FIGURES_DIR
    / "robustness_qualitative_examples.png"
)

EXPECTED_CASES = [
    "baseline",
    "dim_lighting",
    "overexposed_lighting",
    "gaussian_blur",
    "motion_blur",
    "small_face",
    "partial_occlusion",
]

SELECTION_FIELDS = [
    "case_id",
    "condition",
    "image_file",
    "status",
    "detected_faces",
    "available_static_modules",
    "detector_confidence",
    "quality_sharpness",
    "quality_face_area_ratio",
    "head_pose_yaw_abs_delta_vs_baseline",
    "gaze_estimation_yaw_abs_delta_vs_baseline",
    "mouth_openness_abs_delta_vs_baseline",
    "emotion_label",
    "emotion_confidence",
    "annotated_image",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes"}:
            return True

        if normalized in {"false", "0", "no", ""}:
            return False

    if isinstance(value, (int, np.integer)):
        return bool(value)

    raise ValueError(
        f"Cannot parse boolean value: {value!r}"
    )


def load_rows() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    if not RESULTS_CSV.is_file():
        raise FileNotFoundError(
            f"Accepted robustness results not found: {RESULTS_CSV}"
        )

    if not METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"Generated robustness metadata not found: {METADATA_CSV}"
        )

    with RESULTS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        results_rows = list(
            csv.DictReader(file)
        )

    with METADATA_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        metadata_rows = {
            row["case_id"]: row
            for row in csv.DictReader(file)
        }

    if len(results_rows) != len(EXPECTED_CASES):
        raise RuntimeError(
            "Robustness qualitative input row count mismatch"
        )

    if [
        row["case_id"]
        for row in results_rows
    ] != EXPECTED_CASES:
        raise RuntimeError(
            "Robustness qualitative input case order mismatch"
        )

    if set(
        metadata_rows
    ) != set(
        EXPECTED_CASES
    ):
        raise RuntimeError(
            "Robustness metadata cases do not match accepted protocol"
        )

    return (
        results_rows,
        metadata_rows,
    )


def count_available_modules(
    row: dict[str, str],
) -> int:
    fields = [
        "detection_available",
        "landmarks_available",
        "quality_available",
        "head_pose_available",
        "eyes_available",
        "gaze_available",
        "gaze_estimation_available",
        "mouth_available",
        "emotion_available",
        "regions_available",
    ]

    return sum(
        parse_bool(
            row[field]
        )
        for field in fields
    )


def format_float(
    row: dict[str, str],
    key: str,
    digits: int = 4,
) -> str:
    value = row.get(
        key,
        ""
    )

    if value in {
        "",
        None,
    }:
        return "n/a"

    return f"{float(value):.{digits}f}"


def annotate_case(
    image: np.ndarray,
    row: dict[str, str],
) -> np.ndarray:
    output = image.copy()

    if row[
        "status"
    ] == "DETECTED":
        x1 = int(
            round(
                float(
                    row[
                        "box_x1"
                    ]
                )
            )
        )
        y1 = int(
            round(
                float(
                    row[
                        "box_y1"
                    ]
                )
            )
        )
        x2 = int(
            round(
                float(
                    row[
                        "box_x2"
                    ]
                )
            )
        )
        y2 = int(
            round(
                float(
                    row[
                        "box_y2"
                    ]
                )
            )
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

    return output


def build_selection_record(
    row: dict[str, str],
    annotated_filename: str,
) -> dict[str, Any]:
    return {
        "case_id":
            row[
                "case_id"
            ],
        "condition":
            row[
                "condition"
            ],
        "image_file":
            row[
                "image_file"
            ],
        "status":
            row[
                "status"
            ],
        "detected_faces":
            int(
                row[
                    "detected_faces"
                ]
            ),
        "available_static_modules":
            count_available_modules(
                row
            ),
        "detector_confidence":
            row.get(
                "detector_confidence",
                "",
            ),
        "quality_sharpness":
            row.get(
                "quality_sharpness",
                "",
            ),
        "quality_face_area_ratio":
            row.get(
                "quality_face_area_ratio",
                "",
            ),
        "head_pose_yaw_abs_delta_vs_baseline":
            row.get(
                "head_pose_yaw_abs_delta_vs_baseline",
                "",
            ),
        "gaze_estimation_yaw_abs_delta_vs_baseline":
            row.get(
                "gaze_estimation_yaw_abs_delta_vs_baseline",
                "",
            ),
        "mouth_openness_abs_delta_vs_baseline":
            row.get(
                "mouth_openness_abs_delta_vs_baseline",
                "",
            ),
        "emotion_label":
            row.get(
                "emotion_label",
                "",
            ),
        "emotion_confidence":
            row.get(
                "emotion_confidence",
                "",
            ),
        "annotated_image":
            annotated_filename,
    }


def write_selection_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SELECTION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )


def create_combined_figure(
    rows: list[dict[str, str]],
    image_paths: dict[str, Path],
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 8.2),
    )

    flat_axes = axes.ravel()

    for index, row in enumerate(
        rows
    ):
        axis = flat_axes[
            index
        ]

        image = cv2.imread(
            str(
                image_paths[
                    row[
                        "case_id"
                    ]
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read qualitative image for case {row['case_id']}"
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(
            image_rgb
        )
        axis.axis(
            "off"
        )

        title = (
            f"{row['condition']}\n"
            f"{row['status']} | modules {count_available_modules(row)}/10\n"
            f"det {format_float(row, 'detector_confidence', 3)} | "
            f"sharp {format_float(row, 'quality_sharpness', 2)}\n"
            f"|Δ head yaw| {format_float(row, 'head_pose_yaw_abs_delta_vs_baseline', 2)} | "
            f"|Δ gaze yaw| {format_float(row, 'gaze_estimation_yaw_abs_delta_vs_baseline', 2)}"
        )

        axis.set_title(
            title,
            fontsize=9,
        )

    for index in range(
        len(rows),
        len(flat_axes),
    ):
        flat_axes[
            index
        ].axis(
            "off"
        )

    figure.suptitle(
        "Controlled Robustness Qualitative Examples",
        fontsize=16,
    )
    figure.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
            0.95,
        ]
    )
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def validate_staging(
    staging_dir: Path,
) -> None:
    selection_path = (
        staging_dir
        / "qualitative"
        / SELECTION_CSV.name
    )
    annotated_dir = (
        staging_dir
        / "qualitative"
        / "annotated_images"
    )
    combined_figure = (
        staging_dir
        / "figures"
        / COMBINED_FIGURE.name
    )

    if not selection_path.is_file():
        raise FileNotFoundError(
            "Staged robustness qualitative selection CSV missing"
        )

    with selection_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        selection_rows = list(
            csv.DictReader(file)
        )

    if len(
        selection_rows
    ) != len(
        EXPECTED_CASES
    ):
        raise RuntimeError(
            "Robustness qualitative selection row count mismatch"
        )

    if [
        row[
            "case_id"
        ]
        for row in selection_rows
    ] != EXPECTED_CASES:
        raise RuntimeError(
            "Robustness qualitative selection ordering mismatch"
        )

    expected_images = {
        f"{index:02d}_{case_id}_annotated.png"
        for index, case_id in enumerate(
            EXPECTED_CASES,
            start=1,
        )
    }

    actual_images = {
        path.name
        for path in annotated_dir.iterdir()
        if path.is_file()
    }

    if actual_images != expected_images:
        raise RuntimeError(
            "Robustness qualitative annotated-image set mismatch"
        )

    if not combined_figure.is_file():
        raise FileNotFoundError(
            "Robustness qualitative combined figure missing"
        )

    if combined_figure.stat().st_size <= 0:
        raise RuntimeError(
            "Robustness qualitative combined figure is empty"
        )


def promote_staging(
    staging_dir: Path,
) -> None:
    staging_qualitative = (
        staging_dir
        / "qualitative"
    )
    staging_figure = (
        staging_dir
        / "figures"
        / COMBINED_FIGURE.name
    )

    backup_dir = (
        staging_dir
        / ".backup"
    )
    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qualitative_backup = (
        backup_dir
        / "qualitative"
    )
    figure_backup = (
        backup_dir
        / COMBINED_FIGURE.name
    )

    promoted_qualitative = False
    promoted_figure = False
    backed_up_qualitative = False
    backed_up_figure = False

    try:
        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if QUALITATIVE_DIR.exists():
            os.replace(
                QUALITATIVE_DIR,
                qualitative_backup,
            )
            backed_up_qualitative = True

        if COMBINED_FIGURE.exists():
            os.replace(
                COMBINED_FIGURE,
                figure_backup,
            )
            backed_up_figure = True

        os.replace(
            staging_qualitative,
            QUALITATIVE_DIR,
        )
        promoted_qualitative = True

        os.replace(
            staging_figure,
            COMBINED_FIGURE,
        )
        promoted_figure = True

    except Exception:
        if (
            promoted_qualitative
            and QUALITATIVE_DIR.exists()
        ):
            shutil.rmtree(
                QUALITATIVE_DIR,
                ignore_errors=True,
            )

        if (
            promoted_figure
            and COMBINED_FIGURE.exists()
        ):
            COMBINED_FIGURE.unlink()

        if (
            backed_up_qualitative
            and qualitative_backup.exists()
        ):
            os.replace(
                qualitative_backup,
                QUALITATIVE_DIR,
            )

        if (
            backed_up_figure
            and figure_backup.exists()
        ):
            os.replace(
                figure_backup,
                COMBINED_FIGURE,
            )

        raise


def main() -> None:
    results_rows, metadata_rows = load_rows()

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".robustness_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    staging_qualitative = (
        staging_dir
        / "qualitative"
    )
    staging_annotated = (
        staging_qualitative
        / "annotated_images"
    )
    staging_figures = (
        staging_dir
        / "figures"
    )

    staging_annotated.mkdir(
        parents=True,
        exist_ok=True,
    )
    staging_figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_records = []
    annotated_paths: dict[str, Path] = {}

    try:
        for index, row in enumerate(
            results_rows,
            start=1,
        ):
            case_id = row[
                "case_id"
            ]
            metadata = metadata_rows[
                case_id
            ]
            source_path = (
                TEST_DATA_DIR
                / metadata[
                    "output_file"
                ]
            )

            image = cv2.imread(
                str(
                    source_path
                ),
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise RuntimeError(
                    f"Could not read robustness qualitative source: {source_path}"
                )

            annotated = annotate_case(
                image,
                row,
            )

            annotated_filename = (
                f"{index:02d}_{case_id}_annotated.png"
            )
            annotated_path = (
                staging_annotated
                / annotated_filename
            )

            if not cv2.imwrite(
                str(
                    annotated_path
                ),
                annotated,
            ):
                raise RuntimeError(
                    f"Could not write annotated robustness image: {annotated_path}"
                )

            reread = cv2.imread(
                str(
                    annotated_path
                ),
                cv2.IMREAD_COLOR,
            )

            if reread is None:
                raise RuntimeError(
                    f"Could not reread annotated robustness image: {annotated_path}"
                )

            if reread.shape != image.shape:
                raise RuntimeError(
                    f"Annotated robustness image shape mismatch for {case_id}"
                )

            selection_records.append(
                build_selection_record(
                    row,
                    annotated_filename,
                )
            )

            annotated_paths[
                case_id
            ] = annotated_path

        write_selection_csv(
            staging_qualitative
            / SELECTION_CSV.name,
            selection_records,
        )

        create_combined_figure(
            results_rows,
            annotated_paths,
            staging_figures
            / COMBINED_FIGURE.name,
        )

        validate_staging(
            staging_dir
        )
        promote_staging(
            staging_dir
        )

    finally:
        shutil.rmtree(
            staging_dir,
            ignore_errors=True,
        )

    print(
        "Controlled Robustness Qualitative Validation"
    )
    print(
        "=" * 43
    )
    print(
        f"Cases: {len(results_rows)}"
    )
    print(
        "Annotated images: "
        f"{len(results_rows)}"
    )
    print(
        "Selection consistency: PASS"
    )
    print(
        "Image integrity checks: PASS"
    )
    print(
        f"Selection CSV: {SELECTION_CSV}"
    )
    print(
        f"Annotated images: {ANNOTATED_DIR}"
    )
    print(
        f"Combined figure: {COMBINED_FIGURE}"
    )


if __name__ == "__main__":
    main()
