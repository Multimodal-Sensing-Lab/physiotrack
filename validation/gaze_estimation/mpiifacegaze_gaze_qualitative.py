from __future__ import annotations

import csv
import math
import shutil
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
MANIFEST_PATH = QUALITATIVE_DIR / "mpiifacegaze_qualitative_manifest.csv"
CONTACT_SHEET_PATH = QUALITATIVE_DIR / "mpiifacegaze_qualitative_contact_sheet.png"

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


def clean_output_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


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

    clean_output_directory(QUALITATIVE_DIR)

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

        output_path = QUALITATIVE_DIR / output_name

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

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as file:
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
        CONTACT_SHEET_PATH,
        dpi=260,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("=== MPIIFaceGaze Qualitative Evidence ===")
    print(f"Successful quantitative samples: {len(successful)}")
    print(f"Selected cases: {len(manifest_rows)}")
    print(f"Cleaned output directory before writing: {QUALITATIVE_DIR}")
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


if __name__ == "__main__":
    main()