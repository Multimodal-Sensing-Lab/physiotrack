from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = SCRIPT_DIR.parent

SOURCE_IMAGE = (
    VALIDATION_DIR
    / "integration"
    / "test_data"
    / "images"
    / "frontal_1.png"
)

FINAL_OUTPUT_DIR = (
    SCRIPT_DIR
    / "test_data"
    / "generated_controlled"
)

METADATA_CSV = "robustness_generated_cases.csv"
SUMMARY_JSON = "robustness_generated_cases_summary.json"

CASE_ORDER = [
    "baseline",
    "dim_lighting",
    "overexposed_lighting",
    "gaussian_blur",
    "motion_blur",
    "small_face",
    "partial_occlusion",
]

CSV_FIELDS = [
    "case_id",
    "condition",
    "output_file",
    "source_file",
    "width",
    "height",
    "source_sha256",
    "output_sha256",
    "parameters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic controlled robustness images from the "
            "accepted integration frontal-face reference image."
        )
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the source image and output location without writing files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def source_inventory() -> dict[str, int]:
    stat = SOURCE_IMAGE.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def preflight() -> tuple[np.ndarray, dict[str, Any]]:
    if not SOURCE_IMAGE.is_file():
        raise FileNotFoundError(
            f"Required source image not found: {SOURCE_IMAGE}"
        )

    image = cv2.imread(str(SOURCE_IMAGE), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read source image: {SOURCE_IMAGE}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(
            "Source robustness image must be a three-channel color image"
        )

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid source image dimensions")

    output_parent = FINAL_OUTPUT_DIR.parent

    if output_parent.exists() and not output_parent.is_dir():
        raise RuntimeError(
            f"Robustness test-data root is not a directory: {output_parent}"
        )

    output_parent.mkdir(parents=True, exist_ok=True)

    info = {
        "source_file": str(
            SOURCE_IMAGE.relative_to(VALIDATION_DIR.parent)
        ).replace("\\", "/"),
        "width": int(width),
        "height": int(height),
        "source_sha256": sha256_file(SOURCE_IMAGE),
    }

    return image, info


def dim_lighting(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    factor = 0.35
    output = np.clip(
        image.astype(np.float32) * factor,
        0.0,
        255.0,
    ).astype(np.uint8)

    return output, {
        "type": "global_intensity_scale",
        "factor": factor,
    }


def overexposed_lighting(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha = 1.60
    beta = 55.0
    output = np.clip(
        image.astype(np.float32) * alpha + beta,
        0.0,
        255.0,
    ).astype(np.uint8)

    return output, {
        "type": "linear_overexposure",
        "alpha": alpha,
        "beta": beta,
    }


def gaussian_blur(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = 31
    sigma = 7.0
    output = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return output, {
        "type": "gaussian_blur",
        "kernel_size": kernel_size,
        "sigma": sigma,
    }


def motion_blur(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    kernel_size = 31
    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32,
    )
    kernel[
        kernel_size // 2,
        :
    ] = 1.0 / kernel_size

    output = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
    )

    return output, {
        "type": "horizontal_motion_blur",
        "kernel_size": kernel_size,
    }


def estimate_border_background(
    image: np.ndarray,
) -> tuple[int, int, int]:
    height, width = image.shape[:2]
    border_y = max(1, int(round(height * 0.10)))
    border_x = max(1, int(round(width * 0.10)))

    border_pixels = np.concatenate(
        [
            image[:border_y, :, :].reshape(-1, 3),
            image[-border_y:, :, :].reshape(-1, 3),
            image[:, :border_x, :].reshape(-1, 3),
            image[:, -border_x:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    median = np.median(
        border_pixels,
        axis=0,
    )

    return tuple(
        int(round(value))
        for value in median
    )


def small_face(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = 0.35
    height, width = image.shape[:2]

    resized_width = max(
        1,
        int(round(width * scale)),
    )
    resized_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    background = estimate_border_background(image)

    canvas = np.empty_like(image)
    canvas[:, :] = background

    x1 = (width - resized_width) // 2
    y1 = (height - resized_height) // 2
    x2 = x1 + resized_width
    y2 = y1 + resized_height

    canvas[
        y1:y2,
        x1:x2,
    ] = resized

    return canvas, {
        "type": "scene_downscale_centered",
        "scale": scale,
        "paste_box": [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ],
        "background_bgr": list(background),
    }


def partial_occlusion(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]

    normalized_box = [
        0.31,
        0.53,
        0.69,
        0.70,
    ]

    x1 = int(round(normalized_box[0] * width))
    y1 = int(round(normalized_box[1] * height))
    x2 = int(round(normalized_box[2] * width))
    y2 = int(round(normalized_box[3] * height))

    output = image.copy()

    fill_value = int(
        round(
            float(
                np.median(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY,
                    )
                )
            )
        )
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (fill_value, fill_value, fill_value),
        thickness=-1,
    )

    return output, {
        "type": "fixed_normalized_lower_face_occlusion",
        "normalized_box": normalized_box,
        "pixel_box": [
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        ],
        "fill_gray": fill_value,
    }


def generate_cases(
    source: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    return {
        "baseline": (
            source.copy(),
            {
                "type": "unmodified_reference",
            },
        ),
        "dim_lighting": dim_lighting(source),
        "overexposed_lighting": overexposed_lighting(source),
        "gaussian_blur": gaussian_blur(source),
        "motion_blur": motion_blur(source),
        "small_face": small_face(source),
        "partial_occlusion": partial_occlusion(source),
    }


def validate_generated_image(
    source: np.ndarray,
    generated: np.ndarray,
    case_id: str,
) -> None:
    if generated is None:
        raise RuntimeError(
            f"Generated image is missing for case: {case_id}"
        )

    if generated.shape != source.shape:
        raise RuntimeError(
            f"Generated image shape mismatch for case: {case_id}"
        )

    if generated.dtype != np.uint8:
        raise RuntimeError(
            f"Generated image dtype mismatch for case: {case_id}"
        )

    if not np.isfinite(generated).all():
        raise RuntimeError(
            f"Generated image contains non-finite values: {case_id}"
        )


def write_staged_outputs(
    staging_dir: Path,
    source: np.ndarray,
    source_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = generate_cases(source)

    if list(cases) != CASE_ORDER:
        raise RuntimeError(
            "Generated robustness case order differs from the accepted protocol"
        )

    records = []

    for case_index, case_id in enumerate(CASE_ORDER, start=1):
        generated, parameters = cases[case_id]

        validate_generated_image(
            source,
            generated,
            case_id,
        )

        filename = f"{case_index:02d}_{case_id}.png"
        output_path = staging_dir / filename

        success = cv2.imwrite(
            str(output_path),
            generated,
        )

        if not success:
            raise RuntimeError(
                f"Could not write generated robustness image: {output_path}"
            )

        reread = cv2.imread(
            str(output_path),
            cv2.IMREAD_COLOR,
        )

        if reread is None:
            raise RuntimeError(
                f"Could not reread generated robustness image: {output_path}"
            )

        validate_generated_image(
            source,
            reread,
            case_id,
        )

        height, width = reread.shape[:2]

        records.append(
            {
                "case_id": case_id,
                "condition": case_id.replace("_", " "),
                "output_file": filename,
                "source_file": source_info["source_file"],
                "width": width,
                "height": height,
                "source_sha256": source_info["source_sha256"],
                "output_sha256": sha256_file(output_path),
                "parameters": json.dumps(
                    parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )

    csv_path = staging_dir / METADATA_CSV

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "generator": "robustness_dataset_generator",
        "evidence_type": "controlled_robustness_test_input_generation",
        "source": source_info,
        "cases": len(records),
        "case_order": CASE_ORDER,
        "conditions": {
            row["case_id"]: json.loads(
                row["parameters"]
            )
            for row in records
        },
        "dataset_source_modified": False,
        "interpretation": (
            "These images are deterministic controlled stress-test inputs "
            "derived from a single accepted frontal reference image. They are "
            "not additional natural benchmark samples and must not be used as "
            "independent accuracy observations."
        ),
        "status": "PASS",
    }

    with (
        staging_dir
        / SUMMARY_JSON
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return records, summary


def validate_staging(
    staging_dir: Path,
    source_info: dict[str, Any],
) -> None:
    expected_files = {
        f"{index:02d}_{case_id}.png"
        for index, case_id in enumerate(CASE_ORDER, start=1)
    }
    expected_files.update(
        {
            METADATA_CSV,
            SUMMARY_JSON,
        }
    )

    actual_files = {
        path.name
        for path in staging_dir.iterdir()
        if path.is_file()
    }

    if actual_files != expected_files:
        raise RuntimeError(
            "Generated robustness staging contents do not match the accepted protocol"
        )

    with (
        staging_dir
        / METADATA_CSV
    ).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if len(rows) != len(CASE_ORDER):
        raise RuntimeError(
            "Generated robustness metadata row count mismatch"
        )

    if [
        row["case_id"]
        for row in rows
    ] != CASE_ORDER:
        raise RuntimeError(
            "Generated robustness metadata case ordering mismatch"
        )

    for row in rows:
        output_path = staging_dir / row["output_file"]

        if not output_path.is_file():
            raise RuntimeError(
                f"Generated robustness output is missing: {output_path}"
            )

        if sha256_file(output_path) != row["output_sha256"]:
            raise RuntimeError(
                f"Generated robustness SHA256 mismatch: {output_path}"
            )

        if row["source_sha256"] != source_info["source_sha256"]:
            raise RuntimeError(
                "Generated metadata source SHA256 mismatch"
            )

    with (
        staging_dir
        / SUMMARY_JSON
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(file)

    if summary.get("status") != "PASS":
        raise RuntimeError(
            "Generated robustness summary did not pass validation"
        )

    if int(summary.get("cases", -1)) != len(CASE_ORDER):
        raise RuntimeError(
            "Generated robustness summary case count mismatch"
        )


def promote_staging(staging_dir: Path) -> None:
    parent = FINAL_OUTPUT_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)

    backup_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{FINAL_OUTPUT_DIR.name}_backup_",
            dir=parent,
        )
    )
    backup_dir.rmdir()

    backed_up = False
    promoted = False

    try:
        if FINAL_OUTPUT_DIR.exists():
            os.replace(
                FINAL_OUTPUT_DIR,
                backup_dir,
            )
            backed_up = True

        os.replace(
            staging_dir,
            FINAL_OUTPUT_DIR,
        )
        promoted = True

    except Exception:
        if promoted and FINAL_OUTPUT_DIR.exists():
            shutil.rmtree(
                FINAL_OUTPUT_DIR,
                ignore_errors=True,
            )

        if backed_up and backup_dir.exists():
            os.replace(
                backup_dir,
                FINAL_OUTPUT_DIR,
            )

        raise

    if backup_dir.exists():
        shutil.rmtree(
            backup_dir,
            ignore_errors=True,
        )


def main() -> None:
    args = parse_args()

    source, source_info = preflight()

    if args.preflight_only:
        print("Robustness controlled-input generator preflight: PASS")
        print(f"Source image: {source_info['source_file']}")
        print(
            f"Resolution: {source_info['width']}x{source_info['height']}"
        )
        print(f"Source SHA256: {source_info['source_sha256']}")
        print(f"Planned cases: {len(CASE_ORDER)}")
        for case_id in CASE_ORDER:
            print(f"- {case_id}")
        return

    before_inventory = source_inventory()

    parent = FINAL_OUTPUT_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".generated_controlled_",
            dir=parent,
        )
    )

    try:
        records, summary = write_staged_outputs(
            staging_dir,
            source,
            source_info,
        )

        validate_staging(
            staging_dir,
            source_info,
        )

        after_inventory = source_inventory()

        if before_inventory != after_inventory:
            raise RuntimeError(
                "Source integration image metadata changed during generation"
            )

        promote_staging(
            staging_dir
        )

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(
                staging_dir,
                ignore_errors=True,
            )
        raise

    print("Robustness controlled-input generation")
    print("=" * 39)
    print(f"Source image: {source_info['source_file']}")
    print(f"Generated cases: {len(records)}")
    print("Source image unchanged: PASS")
    print("Staged-output validation: PASS")
    print("Status: PASS")
    print(f"Final outputs: {FINAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
