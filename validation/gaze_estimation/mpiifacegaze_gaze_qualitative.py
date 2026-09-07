from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]

DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "MPIIFaceGaze" / "Data"
RESULTS_DIR = SCRIPT_DIR / "results"
PER_SAMPLE_CSV_PATH = RESULTS_DIR / "mpiifacegaze_ethxgaze_per_sample.csv"
QUALITATIVE_DIR = RESULTS_DIR / "qualitative"
FIGURES_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = QUALITATIVE_DIR / "mpiifacegaze_qualitative_manifest.csv"
CONTACT_SHEET_PATH = FIGURES_DIR / "mpiifacegaze_qualitative_contact_sheet.png"

CASE_COUNT = 8
TARGET_PERCENTILES = [5, 25, 50, 75, 90, 95, 99, 100]
EXPECTED_SUCCESSFUL_SAMPLES = 37629

DISPLAY_MAX_ABS_ANGLE_DEG = 35.0
DISPLAY_AXIS_SCALE_RATIO = 0.18
DISPLAY_MAX_VECTOR_RATIO = 0.20
DISPLAY_MIN_VECTOR_PX = 10.0
DISPLAY_CLIP_MARGIN_PX = 14


def validate_vector(values, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)

    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Invalid {label} gaze vector.")

    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"Zero-length {label} gaze vector.")

    return vector / norm


def vector_to_yaw_pitch(vector: np.ndarray) -> tuple[float, float]:
    x, y, z = vector

    yaw = math.degrees(math.atan2(x, -z))
    pitch = math.degrees(math.atan2(y, math.sqrt(x * x + z * z)))

    return float(yaw), float(pitch)


def select_cases(successful: pd.DataFrame) -> list[dict]:
    errors = successful["angular_error_deg"].astype(float).to_numpy()

    selections = []
    used_indices = set()
    used_participants = set()

    for percentile in TARGET_PERCENTILES:
        target = float(np.percentile(errors, percentile))

        candidates = successful.assign(
            distance=successful["angular_error_deg"].astype(float).sub(target).abs()
        ).sort_values(
            ["distance", "participant", "image_relative_path"],
            kind="mergesort",
        )

        selected_index = None

        for index, row in candidates.iterrows():
            if index in used_indices:
                continue
            if row["participant"] in used_participants and percentile != 100:
                continue
            selected_index = index
            break

        if selected_index is None:
            for index in candidates.index:
                if index not in used_indices:
                    selected_index = index
                    break

        if selected_index is None:
            raise RuntimeError("Could not select enough qualitative cases.")

        row = successful.loc[selected_index]
        used_indices.add(selected_index)
        used_participants.add(str(row["participant"]))

        selections.append(
            {
                "percentile_label": "Maximum" if percentile == 100 else f"P{percentile:02d}",
                "target_percentile": percentile,
                "target_error_deg": target,
                "row_index": int(selected_index),
                "row": row,
            }
        )

    if len(selections) != CASE_COUNT:
        raise RuntimeError("Unexpected qualitative case count.")

    return selections


def angle_to_display_endpoint(
    yaw_deg: float,
    pitch_deg: float,
    origin: tuple[int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """
    Convert yaw/pitch to a bounded 2D directional display.

    This display is qualitative only. The scientific comparison remains the
    benchmark 3D angular error already reported by the quantitative evaluator.
    """
    yaw_display = float(
        np.clip(
            yaw_deg,
            -DISPLAY_MAX_ABS_ANGLE_DEG,
            DISPLAY_MAX_ABS_ANGLE_DEG,
        )
    )

    pitch_display = float(
        np.clip(
            pitch_deg,
            -DISPLAY_MAX_ABS_ANGLE_DEG,
            DISPLAY_MAX_ABS_ANGLE_DEG,
        )
    )

    dx = (
        yaw_display / DISPLAY_MAX_ABS_ANGLE_DEG
    ) * (
        DISPLAY_AXIS_SCALE_RATIO * image_width
    )

    dy = (
        pitch_display / DISPLAY_MAX_ABS_ANGLE_DEG
    ) * (
        DISPLAY_AXIS_SCALE_RATIO * image_height
    )

    vector_length = float(math.hypot(dx, dy))
    max_length = float(min(image_width, image_height) * DISPLAY_MAX_VECTOR_RATIO)

    if vector_length > max_length and vector_length > 0:
        scale = max_length / vector_length
        dx *= scale
        dy *= scale
        vector_length = max_length

    if 0 < vector_length < DISPLAY_MIN_VECTOR_PX:
        scale = DISPLAY_MIN_VECTOR_PX / vector_length
        dx *= scale
        dy *= scale

    endpoint_x = int(round(origin[0] + dx))
    endpoint_y = int(round(origin[1] + dy))

    endpoint_x = max(
        DISPLAY_CLIP_MARGIN_PX,
        min(image_width - DISPLAY_CLIP_MARGIN_PX, endpoint_x),
    )

    endpoint_y = max(
        DISPLAY_CLIP_MARGIN_PX,
        min(image_height - DISPLAY_CLIP_MARGIN_PX, endpoint_y),
    )

    return endpoint_x, endpoint_y


def draw_gaze_arrow(
    canvas: np.ndarray,
    origin: tuple[int, int],
    endpoint: tuple[int, int],
    bgr_color: tuple[int, int, int],
) -> None:
    thickness = max(
        2,
        int(round(min(canvas.shape[:2]) / 220.0)),
    )

    cv2.arrowedLine(
        canvas,
        origin,
        endpoint,
        bgr_color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.16,
    )


def add_title_band(
    image: np.ndarray,
    title: str,
    subtitle: str,
) -> np.ndarray:
    height, width = image.shape[:2]

    band_height = max(
        72,
        int(round(0.14 * height)),
    )

    canvas = cv2.copyMakeBorder(
        image,
        band_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = max(
        0.60,
        min(0.92, width / 700.0),
    )

    subtitle_scale = max(
        0.50,
        min(0.76, width / 850.0),
    )

    cv2.putText(
        canvas,
        title,
        (14, int(round(band_height * 0.42))),
        font,
        title_scale,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        subtitle,
        (14, int(round(band_height * 0.80))),
        font,
        subtitle_scale,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )

    return canvas


def annotate_case(
    image: np.ndarray,
    gt_vector: np.ndarray,
    pred_vector: np.ndarray,
    participant: str,
    image_relative_path: str,
    angular_error_deg: float,
    percentile_label: str,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]

    gt_yaw, gt_pitch = vector_to_yaw_pitch(gt_vector)
    pred_yaw, pred_pitch = vector_to_yaw_pitch(pred_vector)

    origin = (
        int(round(0.50 * width)),
        int(round(0.42 * height)),
    )

    gt_endpoint = angle_to_display_endpoint(
        gt_yaw,
        gt_pitch,
        origin,
        width,
        height,
    )

    pred_endpoint = angle_to_display_endpoint(
        pred_yaw,
        pred_pitch,
        origin,
        width,
        height,
    )

    draw_gaze_arrow(
        canvas,
        origin,
        gt_endpoint,
        (0, 180, 0),
    )

    draw_gaze_arrow(
        canvas,
        origin,
        pred_endpoint,
        (0, 0, 220),
    )

    cv2.circle(
        canvas,
        origin,
        max(4, int(round(min(width, height) / 150.0))),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    title = (
        f"{percentile_label} | "
        f"{participant}/{image_relative_path} | "
        f"Angular error = {angular_error_deg:.2f} deg"
    )

    subtitle = (
        f"GT: yaw={gt_yaw:.1f}, pitch={gt_pitch:.1f} deg   |   "
        f"Pred: yaw={pred_yaw:.1f}, pitch={pred_pitch:.1f} deg"
    )

    return add_title_band(
        canvas,
        title,
        subtitle,
    )


def create_staging_directory() -> tuple[Path, Path, Path]:
    """Create staging before qualitative artifact generation."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".mpiifacegaze_gaze_qualitative_",
            dir=RESULTS_DIR,
        )
    )

    staged_qualitative_dir = (
        staging_dir
        / "qualitative"
    )

    staged_figures_dir = (
        staging_dir
        / "figures"
    )

    staged_qualitative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        staging_dir,
        staged_qualitative_dir,
        staged_figures_dir,
    )


def validate_staged_outputs(
    staged_qualitative_dir: Path,
    staged_manifest_path: Path,
    staged_contact_sheet_path: Path,
    expected_manifest_rows: list[dict],
    successful: pd.DataFrame,
) -> None:
    """Verify qualitative files and their quantitative provenance."""
    if not staged_manifest_path.is_file():
        raise RuntimeError(
            "Missing staged qualitative manifest."
        )

    if not staged_contact_sheet_path.is_file():
        raise RuntimeError(
            "Missing staged qualitative contact sheet."
        )

    manifest = pd.read_csv(
        staged_manifest_path
    )

    if len(
        manifest
    ) != CASE_COUNT:
        raise RuntimeError(
            "Unexpected staged qualitative manifest row count."
        )

    if len(
        expected_manifest_rows
    ) != CASE_COUNT:
        raise RuntimeError(
            "Unexpected in-memory qualitative manifest row count."
        )

    expected_output_files = {
        row[
            "output_file"
        ]
        for row in expected_manifest_rows
    }

    actual_output_files = {
        path.name
        for path in staged_qualitative_dir.glob(
            "case_*.png"
        )
    }

    if actual_output_files != expected_output_files:
        raise RuntimeError(
            "Staged qualitative image set does not match the manifest."
        )

    successful_lookup = successful.set_index(
        [
            "participant",
            "image_relative_path",
        ],
        drop=False,
    )

    for index, expected_row in enumerate(
        expected_manifest_rows
    ):
        stored_row = manifest.iloc[
            index
        ]

        string_columns = [
            "selection_label",
            "participant",
            "image_relative_path",
            "output_file",
        ]

        for column in string_columns:
            if str(
                stored_row[
                    column
                ]
            ) != str(
                expected_row[
                    column
                ]
            ):
                raise RuntimeError(
                    f"Staged qualitative manifest mismatch in {column}."
                )

        numeric_columns = [
            "case",
            "target_percentile",
            "target_error_deg",
            "angular_error_deg",
            "gt_x",
            "gt_y",
            "gt_z",
            "pred_x",
            "pred_y",
            "pred_z",
            "gt_yaw_deg",
            "gt_pitch_deg",
            "pred_yaw_deg",
            "pred_pitch_deg",
        ]

        for column in numeric_columns:
            if not math.isclose(
                float(
                    stored_row[
                        column
                    ]
                ),
                float(
                    expected_row[
                        column
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"Staged qualitative manifest mismatch in {column}."
                )

        key = (
            str(
                expected_row[
                    "participant"
                ]
            ),
            str(
                expected_row[
                    "image_relative_path"
                ]
            ),
        )

        if key not in successful_lookup.index:
            raise RuntimeError(
                "Qualitative selection is absent from successful raw results."
            )

        raw_row = successful_lookup.loc[
            key
        ]

        if isinstance(
            raw_row,
            pd.DataFrame,
        ):
            raise RuntimeError(
                "Duplicate successful quantitative key used by qualitative evidence."
            )

        if not math.isclose(
            float(
                raw_row[
                    "angular_error_deg"
                ]
            ),
            float(
                expected_row[
                    "angular_error_deg"
                ]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Qualitative angular error differs from raw quantitative evidence."
            )

        output_path = (
            staged_qualitative_dir
            / expected_row[
                "output_file"
            ]
        )

        image = cv2.imread(
            str(
                output_path
            )
        )

        if image is None or image.size == 0:
            raise RuntimeError(
                f"Staged qualitative image is unreadable: {output_path.name}"
            )

    contact_sheet = cv2.imread(
        str(
            staged_contact_sheet_path
        )
    )

    if (
        contact_sheet is None
        or contact_sheet.size == 0
    ):
        raise RuntimeError(
            "Staged qualitative contact sheet is unreadable."
        )


def atomic_copy_file(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Atomically install one validated qualitative-owned file."""
    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )

    os.close(
        descriptor
    )

    temporary_path = Path(
        temporary_name
    )

    try:
        shutil.copy2(
            source_path,
            temporary_path,
        )

        os.replace(
            temporary_path,
            destination_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def commit_outputs(
    staged_qualitative_dir: Path,
    staged_contact_sheet_path: Path,
    staging_dir: Path,
) -> None:
    """Replace qualitative-owned outputs transactionally with rollback."""
    QUALITATIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_qualitative_files = sorted(
        path
        for path in staged_qualitative_dir.iterdir()
        if path.is_file()
    )

    output_pairs = [
        (
            staged_path,
            QUALITATIVE_DIR
            / staged_path.name,
        )
        for staged_path in staged_qualitative_files
    ]

    output_pairs.append(
        (
            staged_contact_sheet_path,
            CONTACT_SHEET_PATH,
        )
    )

    desired_qualitative_names = {
        path.name
        for path in staged_qualitative_files
    }

    previous_qualitative_files = sorted(
        path
        for path in QUALITATIVE_DIR.iterdir()
        if path.is_file()
    )

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_qualitative_dir = (
        backup_dir
        / "qualitative"
    )

    backup_qualitative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for previous_path in previous_qualitative_files:
        shutil.copy2(
            previous_path,
            backup_qualitative_dir
            / previous_path.name,
        )

    backup_contact_sheet = (
        backup_dir
        / CONTACT_SHEET_PATH.name
    )

    if CONTACT_SHEET_PATH.is_file():
        shutil.copy2(
            CONTACT_SHEET_PATH,
            backup_contact_sheet,
        )

    installed_paths = []

    try:
        for staged_path, final_path in output_pairs:
            atomic_copy_file(
                staged_path,
                final_path,
            )

            installed_paths.append(
                final_path
            )

        for previous_path in previous_qualitative_files:
            if (
                previous_path.name
                not in desired_qualitative_names
                and previous_path.exists()
            ):
                previous_path.unlink()

    except Exception:
        for installed_path in installed_paths:
            if installed_path.exists():
                installed_path.unlink()

        for backup_path in backup_qualitative_dir.iterdir():
            atomic_copy_file(
                backup_path,
                QUALITATIVE_DIR
                / backup_path.name,
            )

        if backup_contact_sheet.is_file():
            atomic_copy_file(
                backup_contact_sheet,
                CONTACT_SHEET_PATH,
            )

        raise


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "MPIIFaceGaze dataset not found at datasets/MPIIFaceGaze/Data"
        )

    if not PER_SAMPLE_CSV_PATH.is_file():
        raise FileNotFoundError(
            "Missing quantitative per-sample result file. "
            "Run the quantitative evaluator first."
        )

    generated_camera_files = list(DATASET_ROOT.rglob("ptgaze_camera.yaml"))
    if generated_camera_files:
        raise RuntimeError(
            "Dataset cleanliness check failed: "
            "ptgaze_camera.yaml exists inside the dataset."
        )

    data = pd.read_csv(PER_SAMPLE_CSV_PATH)

    required_columns = [
        "participant",
        "image_relative_path",
        "status",
        "gt_x",
        "gt_y",
        "gt_z",
        "pred_x",
        "pred_y",
        "pred_z",
        "angular_error_deg",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required per-sample columns: "
            + ", ".join(missing_columns)
        )

    successful = data[data["status"] == "success"].copy()

    if len(successful) != EXPECTED_SUCCESSFUL_SAMPLES:
        raise RuntimeError(
            "Unexpected successful sample count. "
            f"Expected {EXPECTED_SUCCESSFUL_SAMPLES}, "
            f"found {len(successful)}."
        )

    if not np.all(
        np.isfinite(
            successful["angular_error_deg"].astype(float).to_numpy()
        )
    ):
        raise RuntimeError(
            "Successful samples contain invalid angular errors."
        )

    selections = select_cases(successful)

    (
        staging_dir,
        staged_qualitative_dir,
        staged_figures_dir,
    ) = create_staging_directory()


    staged_manifest_path = (
        staged_qualitative_dir
        / MANIFEST_PATH.name
    )

    staged_contact_sheet_path = (
        staged_figures_dir
        / CONTACT_SHEET_PATH.name
    )

    print(
        f"Staging directory: {staging_dir}"
    )

    try:
        manifest_rows = []
        rendered_paths = []

        for case_number, selection in enumerate(selections, start=1):
            row = selection["row"]

            participant = str(row["participant"])
            image_relative_path = str(row["image_relative_path"])

            image_path = DATASET_ROOT / participant / image_relative_path
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(f"Could not read selected image: {image_path}")

            gt_vector = validate_vector(
                [row["gt_x"], row["gt_y"], row["gt_z"]],
                "ground-truth",
            )

            pred_vector = validate_vector(
                [row["pred_x"], row["pred_y"], row["pred_z"]],
                "predicted",
            )

            angular_error = float(row["angular_error_deg"])

            annotated = annotate_case(
                image=image,
                gt_vector=gt_vector,
                pred_vector=pred_vector,
                participant=participant,
                image_relative_path=image_relative_path,
                angular_error_deg=angular_error,
                percentile_label=selection["percentile_label"],
            )

            output_name = (
                f"case_{case_number:02d}_"
                f"{selection['percentile_label'].lower()}_"
                f"{participant}.png"
            )

            output_path = staged_qualitative_dir / output_name

            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Failed to write {output_path.name}")

            rendered_paths.append(output_path)

            gt_yaw, gt_pitch = vector_to_yaw_pitch(gt_vector)
            pred_yaw, pred_pitch = vector_to_yaw_pitch(pred_vector)

            manifest_rows.append(
                {
                    "case": case_number,
                    "selection_label": selection["percentile_label"],
                    "target_percentile": selection["target_percentile"],
                    "target_error_deg": selection["target_error_deg"],
                    "participant": participant,
                    "image_relative_path": image_relative_path,
                    "angular_error_deg": angular_error,
                    "gt_x": gt_vector[0],
                    "gt_y": gt_vector[1],
                    "gt_z": gt_vector[2],
                    "pred_x": pred_vector[0],
                    "pred_y": pred_vector[1],
                    "pred_z": pred_vector[2],
                    "gt_yaw_deg": gt_yaw,
                    "gt_pitch_deg": gt_pitch,
                    "pred_yaw_deg": pred_yaw,
                    "pred_pitch_deg": pred_pitch,
                    "output_file": output_name,
                }
            )

        with open(
            staged_manifest_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            fieldnames = list(manifest_rows[0].keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        figure, axes = plt.subplots(
            2,
            4,
            figsize=(20, 10.8),
        )

        for axis, output_path, manifest_row in zip(axes.ravel(), rendered_paths, manifest_rows):
            image_bgr = cv2.imread(str(output_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            axis.imshow(image_rgb)
            axis.set_title(
                (
                    f"{manifest_row['selection_label']} | "
                    f"{manifest_row['participant']} | "
                    f"{manifest_row['angular_error_deg']:.2f}°"
                ),
                fontsize=13,
                pad=9,
            )
            axis.axis("off")

        figure.suptitle(
            "MPIIFaceGaze Qualitative Evidence: "
            "Ground Truth vs PhysioTrack GazeEstimator",
            fontsize=18,
            y=0.985,
        )

        figure.text(
            0.5,
            0.016,
            (
                "Green = ground truth, red = prediction. "
                "Arrows are a bounded 2D directional visualization; "
                "scientific comparison uses the reported 3D angular error."
            ),
            ha="center",
            va="bottom",
            fontsize=11,
        )

        figure.tight_layout(rect=(0, 0.04, 1, 0.95))
        figure.savefig(
            staged_contact_sheet_path,
            dpi=260,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(
            "Validating staged qualitative outputs..."
        )

        validate_staged_outputs(
            staged_qualitative_dir,
            staged_manifest_path,
            staged_contact_sheet_path,
            manifest_rows,
            successful,
        )

        commit_outputs(
            staged_qualitative_dir,
            staged_contact_sheet_path,
            staging_dir,
        )

        print(
            "Committed final qualitative outputs."
        )

        print("=== MPIIFaceGaze Qualitative Evidence ===")
        print(f"Successful quantitative samples: {len(successful)}")
        print(f"Selected cases: {len(manifest_rows)}")
        print(
            "Qualitative outputs were generated and validated in staging "
            "before final replacement."
        )
        print()

        for row in manifest_rows:
            print(
                f"Case {row['case']:02d} | "
                f"{row['selection_label']} | "
                f"{row['participant']} | "
                f"{row['image_relative_path']} | "
                f"error={row['angular_error_deg']:.4f} deg"
            )

        print()
        print("Dataset write operations: NONE")
        print(f"Saved manifest: {MANIFEST_PATH}")
        print(f"Saved contact sheet: {CONTACT_SHEET_PATH}")


    finally:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()