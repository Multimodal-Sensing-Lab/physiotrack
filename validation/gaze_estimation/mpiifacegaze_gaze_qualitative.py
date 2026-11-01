from __future__ import annotations

import csv
import filecmp
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

# Images may be placed within large black borders by MPIIFaceGaze.
# Image-region detection affects drawing only; it never changes GT/predictions.
DISPLAY_REGION_MAX_SIDE = 320
DISPLAY_REGION_NONBLACK_THRESHOLD = 15



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


def visible_image_region(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the photographic area inside black-padded MPIIFaceGaze images.

    The largest connected non-black region is selected after downsampling and
    closing tiny gaps.  The result is an illustrative drawing region, *not* a
    face/eye detection or a ground-truth landmark measurement.  Entirely visible
    photographs use the full image. No inference or model state is touched.
    """
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Cannot position gaze arrows on an empty image")

    scale = min(1.0, DISPLAY_REGION_MAX_SIDE / max(height, width))
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    small = cv2.resize(image, (small_width, small_height),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    visible = (np.max(small, axis=2) > DISPLAY_REGION_NONBLACK_THRESHOLD).astype(np.uint8)
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE,
                               np.ones((5, 5), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(visible, connectivity=8)
    if count <= 1:
        return (0, 0, width, height)

    largest = max(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    x, y, region_w, region_h, area = (int(n) for n in stats[largest])
    # Suppress accidental tiny bright spots against mostly black images.
    if area < 0.01 * small_width * small_height or region_w < 6 or region_h < 6:
        return (0, 0, width, height)

    left = max(0, int(math.floor(x * width / small_width)))
    top = max(0, int(math.floor(y * height / small_height)))
    right = min(width, int(math.ceil((x + region_w) * width / small_width)))
    bottom = min(height, int(math.ceil((y + region_h) * height / small_height)))
    if right - left < 8 or bottom - top < 8:
        return (0, 0, width, height)
    return (left, top, right, bottom)


def angle_to_display_endpoint(
    yaw_deg: float,
    pitch_deg: float,
    origin: tuple[int, int],
    image_width: int,
    image_height: int,
    display_region: tuple[int, int, int, int] | None = None,
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

    # Use the photographic region (not black canvas padding) for arrow scale.
    left, top, right, bottom = display_region or (0, 0, image_width, image_height)
    region_width = right - left
    region_height = bottom - top

    dx = (yaw_display / DISPLAY_MAX_ABS_ANGLE_DEG) * (DISPLAY_AXIS_SCALE_RATIO * region_width)
    dy = (pitch_display / DISPLAY_MAX_ABS_ANGLE_DEG) * (DISPLAY_AXIS_SCALE_RATIO * region_height)

    vector_length = float(math.hypot(dx, dy))
    max_length = float(min(region_width, region_height) * DISPLAY_MAX_VECTOR_RATIO)

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

    # Keep both arrows inside the photographic area (not the black padding).
    margin = min(DISPLAY_CLIP_MARGIN_PX, max(0, (region_width - 1) // 4),
                 max(0, (region_height - 1) // 4))
    endpoint_x = max(left + margin, min(right - 1 - margin, endpoint_x))
    endpoint_y = max(top + margin, min(bottom - 1 - margin, endpoint_y))

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

    # The source image can contain black margins. Anchor the qualitative arrows
    # near the upper middle of the visible face photograph, never in the margin.
    # This is a display anchor, not a measured anatomical landmark position.
    region = visible_image_region(image)
    left, top, right, bottom = region
    origin = (
        max(left, min(right - 1, int(round(left + 0.50 * (right - left))))),
        max(top, min(bottom - 1, int(round(top + 0.44 * (bottom - top))))),
    )

    gt_endpoint = angle_to_display_endpoint(
        gt_yaw,
        gt_pitch,
        origin,
        width,
        height,
        display_region=region,
    )

    pred_endpoint = angle_to_display_endpoint(
        pred_yaw,
        pred_pitch,
        origin,
        width,
        height,
        display_region=region,
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


def _is_owned_qualitative_file(path: Path) -> bool:
    """Only this script's manifest and numbered case PNGs are replaceable."""
    return (
        path.name == MANIFEST_PATH.name
        or (path.suffix.lower() == ".png"
            and path.name.startswith("case_")
            and len(path.name) > 8
            and path.name[5:7].isdigit()
            and path.name[7] == "_")
    )


def commit_outputs(
    staged_qualitative_dir: Path,
    staged_contact_sheet_path: Path,
    staging_dir: Path,
) -> None:
    """Install just this script's files, verifying publication and restoring on failure.

    Unrelated files under qualitative/ and figures/ are never deleted.
    The prior owned outputs are preserved until the new outputs pass verification.
    """
    if QUALITATIVE_DIR.is_symlink() or FIGURES_DIR.is_symlink():
        raise RuntimeError("Refusing a symlinked qualitative output directory")
    QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    staged_files = sorted(p for p in staged_qualitative_dir.iterdir() if p.is_file())
    if (not staged_files or
            not all(_is_owned_qualitative_file(p) for p in staged_files) or
            not any(p.name == MANIFEST_PATH.name for p in staged_files)):
        raise RuntimeError("Staged qualitative outputs do not match script ownership")
    if (not staged_contact_sheet_path.is_file() or
            staged_contact_sheet_path.stat().st_size == 0):
        raise RuntimeError("Staged qualitative contact sheet is missing")

    expected_names = {p.name for p in staged_files}
    if len(expected_names) != len(staged_files):
        raise RuntimeError("Duplicate staged qualitative filenames")

    previous_files = sorted(
        p for p in QUALITATIVE_DIR.iterdir()
        if p.is_file() and _is_owned_qualitative_file(p)
    )
    if (any(p.is_symlink() for p in previous_files) or
            (CONTACT_SHEET_PATH.is_symlink())):
        raise RuntimeError("Refusing to replace symlinked qualitative outputs")

    backup_dir = staging_dir / "backup"
    backup_qualitative = backup_dir / "qualitative"
    backup_qualitative.mkdir(parents=True, exist_ok=True)
    for old in previous_files:
        shutil.copy2(old, backup_qualitative / old.name)
    backup_contact = backup_dir / CONTACT_SHEET_PATH.name
    had_contact = CONTACT_SHEET_PATH.is_file()
    if had_contact:
        shutil.copy2(CONTACT_SHEET_PATH, backup_contact)

    try:
        for staged in staged_files:
            atomic_copy_file(staged, QUALITATIVE_DIR / staged.name)
        atomic_copy_file(staged_contact_sheet_path, CONTACT_SHEET_PATH)

        # Remove obsolete outputs belonging to this script, never other files.
        for old in previous_files:
            if old.name not in expected_names and old.exists():
                old.unlink()

        published_files = {
            p.name for p in QUALITATIVE_DIR.iterdir()
            if p.is_file() and _is_owned_qualitative_file(p)
        }
        if published_files != expected_names:
            raise RuntimeError("Published qualitative file set differs from staging")
        for staged in staged_files:
            if not filecmp.cmp(staged, QUALITATIVE_DIR / staged.name, shallow=False):
                raise RuntimeError(f"Published qualitative file differs: {staged.name}")
        if not filecmp.cmp(staged_contact_sheet_path, CONTACT_SHEET_PATH,
                            shallow=False):
            raise RuntimeError("Published qualitative contact sheet differs")

    except Exception:
        # Restore precisely the previous owned set, including its obsolete cases.
        for current in QUALITATIVE_DIR.iterdir():
            if current.is_file() and _is_owned_qualitative_file(current):
                current.unlink()
        for old in previous_files:
            atomic_copy_file(backup_qualitative / old.name,
                             QUALITATIVE_DIR / old.name)
        if had_contact:
            atomic_copy_file(backup_contact, CONTACT_SHEET_PATH)
        elif CONTACT_SHEET_PATH.exists():
            CONTACT_SHEET_PATH.unlink()
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