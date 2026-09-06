from pathlib import Path
import argparse
import csv
import os
import shutil
import sqlite3
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face.face_orientation import FaceOrientation


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "AFLW"

DATA_ROOT = DATASET_ROOT / "aflw" / "data"
IMAGE_ROOT = DATA_ROOT / "flickr"
DATABASE_PATH = DATA_ROOT / "aflw.sqlite"

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "aflw_head_pose_results.csv"
SUMMARY_TXT = RESULTS_DIR / "aflw_head_pose_summary.txt"

DEVICE = "cpu"

MAX_ABS_YAW = 90.0
MAX_ABS_ROLL = 90.0
MAX_ABS_PITCH = 90.0

REQUIRED_DATABASE_TABLES = {
    "FaceImages",
    "Faces",
    "FaceRect",
    "FacePose",
}


def angular_error_degrees(predicted, reference):
    """Return the smallest absolute angular difference in degrees."""
    difference = (predicted - reference + 180.0) % 360.0 - 180.0
    return abs(float(difference))


def validate_dataset_layout():
    """Validate the AFLW files required by this evaluation."""
    required_paths = [
        DATABASE_PATH,
        IMAGE_ROOT / "0",
        IMAGE_ROOT / "2",
        IMAGE_ROOT / "3",
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "AFLW dataset layout is incomplete. Missing required paths:\n"
            f"{formatted}\n\n"
            "See validation/head_pose/README_AFLW_HEAD_POSE.txt for the "
            "required AFLW archive extraction and directory layout."
        )

    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        available_tables = {row[0] for row in cursor.fetchall()}
    finally:
        connection.close()

    missing_tables = sorted(
        REQUIRED_DATABASE_TABLES - available_tables
    )

    if missing_tables:
        raise RuntimeError(
            "AFLW SQLite database is missing required tables: "
            + ", ".join(missing_tables)
        )


def load_database_accounting():
    """Return database counts that describe the evaluation population."""
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    try:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM FacePose")
        face_pose_records = int(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM Faces AS f
            INNER JOIN FaceImages AS fi
                ON f.file_id = fi.file_id
               AND f.db_id = fi.db_id
            INNER JOIN FaceRect AS fr
                ON f.face_id = fr.face_id
            INNER JOIN FacePose AS fp
                ON f.face_id = fp.face_id
            """
        )
        joined_records = int(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT COUNT(DISTINCT f.face_id)
            FROM Faces AS f
            INNER JOIN FaceImages AS fi
                ON f.file_id = fi.file_id
               AND f.db_id = fi.db_id
            INNER JOIN FaceRect AS fr
                ON f.face_id = fr.face_id
            INNER JOIN FacePose AS fp
                ON f.face_id = fp.face_id
            """
        )
        distinct_joined_faces = int(cursor.fetchone()[0])
    finally:
        connection.close()

    if joined_records != distinct_joined_faces:
        raise RuntimeError(
            "The AFLW evaluation join produced duplicate face IDs. "
            "This evaluator expects one joined record per face."
        )

    return {
        "face_pose_records": face_pose_records,
        "joined_records": joined_records,
        "distinct_joined_faces": distinct_joined_faces,
    }


def load_annotations():
    """Load AFLW face boxes and head-pose annotations."""
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            f.face_id,
            fi.filepath,
            fr.x,
            fr.y,
            fr.w,
            fr.h,
            fp.roll,
            fp.pitch,
            fp.yaw
        FROM Faces AS f
        INNER JOIN FaceImages AS fi
            ON f.file_id = fi.file_id
           AND f.db_id = fi.db_id
        INNER JOIN FaceRect AS fr
            ON f.face_id = fr.face_id
        INNER JOIN FacePose AS fp
            ON f.face_id = fp.face_id
        ORDER BY f.face_id
        """
    )

    rows = cursor.fetchall()
    connection.close()

    face_ids = [row[0] for row in rows]
    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError(
            "Duplicate face IDs were found in the AFLW evaluation join."
        )

    return rows


def convert_aflw_pose(roll, pitch, yaw):
    """Convert AFLW pose values to the PhysioTrack convention."""
    gt_roll = -float(np.degrees(roll))
    gt_pitch = -float(np.degrees(pitch))
    gt_yaw = -float(np.degrees(yaw))

    return gt_yaw, gt_pitch, gt_roll


def is_primary_protocol_sample(yaw, pitch, roll):
    """Return whether a sample belongs to the primary evaluation range."""
    return (
        abs(yaw) <= MAX_ABS_YAW
        and abs(roll) <= MAX_ABS_ROLL
        and abs(pitch) < MAX_ABS_PITCH
    )


def make_face_box(x, y, w, h, image_width, image_height):
    """Convert an AFLW face rectangle to a valid xyxy box."""
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))

    x2 = min(float(image_width), float(x + w))
    y2 = min(float(image_height), float(y + h))

    if x2 <= x1 or y2 <= y1:
        return None

    return np.asarray([[x1, y1, x2, y2]], dtype=np.float32)


def evaluate_sample(row, estimator):
    """Evaluate one AFLW face with the PhysioTrack head-pose estimator."""
    (
        face_id,
        filepath,
        x,
        y,
        w,
        h,
        raw_roll,
        raw_pitch,
        raw_yaw,
    ) = row

    gt_yaw, gt_pitch, gt_roll = convert_aflw_pose(
        raw_roll,
        raw_pitch,
        raw_yaw,
    )

    protocol_eligible = is_primary_protocol_sample(
        gt_yaw,
        gt_pitch,
        gt_roll,
    )

    result = {
        "face_id": face_id,
        "filepath": filepath,
        "protocol_eligible": protocol_eligible,
        "status": "",
        "gt_yaw": gt_yaw,
        "gt_pitch": gt_pitch,
        "gt_roll": gt_roll,
        "pred_yaw": np.nan,
        "pred_pitch": np.nan,
        "pred_roll": np.nan,
        "yaw_error": np.nan,
        "pitch_error": np.nan,
        "roll_error": np.nan,
    }

    image_path = IMAGE_ROOT / filepath

    if not image_path.is_file():
        result["status"] = "missing_image"
        return result

    if not protocol_eligible:
        result["status"] = "outside_primary_range"
        return result

    frame = cv2.imread(str(image_path))

    if frame is None:
        result["status"] = "image_read_failed"
        return result

    image_height, image_width = frame.shape[:2]

    face_box = make_face_box(
        x,
        y,
        w,
        h,
        image_width,
        image_height,
    )

    if face_box is None:
        result["status"] = "invalid_face_box"
        return result

    prediction = estimator.predict(frame, face_box)

    if not prediction.instances:
        result["status"] = "prediction_failed"
        return result

    orientation = prediction.instances[0].orientation

    if orientation is None:
        result["status"] = "prediction_failed"
        return result

    pred_yaw = float(orientation["yaw"])
    pred_pitch = float(orientation["pitch"])
    pred_roll = float(orientation["roll"])

    if not np.all(np.isfinite([pred_yaw, pred_pitch, pred_roll])):
        result["status"] = "nonfinite_prediction"
        return result

    result["pred_yaw"] = pred_yaw
    result["pred_pitch"] = pred_pitch
    result["pred_roll"] = pred_roll

    result["yaw_error"] = angular_error_degrees(pred_yaw, gt_yaw)
    result["pitch_error"] = angular_error_degrees(pred_pitch, gt_pitch)
    result["roll_error"] = angular_error_degrees(pred_roll, gt_roll)

    result["status"] = "ok"
    return result


def summarize_errors(rows):
    """Calculate summary statistics for successful predictions."""
    successful = [row for row in rows if row["status"] == "ok"]

    if not successful:
        raise RuntimeError(
            "No successful head-pose predictions were produced."
        )

    yaw_errors = np.asarray(
        [row["yaw_error"] for row in successful],
        dtype=float,
    )
    pitch_errors = np.asarray(
        [row["pitch_error"] for row in successful],
        dtype=float,
    )
    roll_errors = np.asarray(
        [row["roll_error"] for row in successful],
        dtype=float,
    )

    axis_mae = {
        "yaw": float(np.mean(yaw_errors)),
        "pitch": float(np.mean(pitch_errors)),
        "roll": float(np.mean(roll_errors)),
    }

    overall_mae = float(
        np.mean(
            [
                axis_mae["yaw"],
                axis_mae["pitch"],
                axis_mae["roll"],
            ]
        )
    )

    return {
        "successful": len(successful),
        "yaw_mae": axis_mae["yaw"],
        "pitch_mae": axis_mae["pitch"],
        "roll_mae": axis_mae["roll"],
        "overall_mae": overall_mae,
        "yaw_median": float(np.median(yaw_errors)),
        "pitch_median": float(np.median(pitch_errors)),
        "roll_median": float(np.median(roll_errors)),
        "yaw_std": float(np.std(yaw_errors)),
        "pitch_std": float(np.std(pitch_errors)),
        "roll_std": float(np.std(roll_errors)),
    }


def save_results(rows, output_path=RESULTS_CSV):
    """Save per-face evaluation results."""
    fieldnames = [
        "face_id",
        "filepath",
        "protocol_eligible",
        "status",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    output_path,
    database_accounting,
    annotations,
    status_counts,
    summary,
    eligible_samples,
    failed_eligible_samples,
    success_rate,
    elapsed,
):
    """Save the textual scientific record for the completed evaluation."""
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("AFLW Head Pose Validation\n\n")

        file.write("Dataset:\n")
        file.write("Annotated Facial Landmarks in the Wild (AFLW)\n")
        file.write("Expected workspace location: datasets/AFLW\n\n")

        file.write("Evaluated component:\n")
        file.write("PhysioTrack FaceOrientation using 6DRepNet360\n\n")

        file.write("Ground truth:\n")
        file.write("AFLW FacePose roll, pitch, and yaw annotations\n")
        file.write("AFLW angles converted from radians to degrees\n")
        file.write(
            "AFLW angle signs inverted to match the PhysioTrack output "
            "convention\n\n"
        )

        file.write("Evaluation protocol:\n")
        file.write(
            "AFLW provides approximate coarse head-pose annotations "
            "derived from facial landmarks using POSIT. The annotations "
            "are not manually verified for every face.\n"
        )
        file.write(
            "The primary evaluation therefore reports controlled "
            "coarse-pose accuracy rather than fine-grained ground-truth "
            "pose accuracy.\n"
        )
        file.write(
            "AFLW ground-truth face rectangles were used to isolate "
            "head-pose estimation from face detection.\n"
        )
        file.write(
            "Primary pose range: |yaw| <= 90 degrees, |pitch| < 90 "
            "degrees, |roll| <= 90 degrees\n"
        )
        file.write(
            "The controlled range avoids Euler-angle boundary and "
            "extreme-orientation cases where axis-wise angle comparison "
            "becomes ambiguous.\n"
        )
        file.write(
            "Angular errors use wrapped absolute angular difference in "
            "degrees.\n"
        )
        file.write(f"Device: {DEVICE}\n\n")

        file.write("Dataset accounting:\n")
        file.write(
            f"FacePose records in database: "
            f"{database_accounting['face_pose_records']}\n"
        )
        file.write(f"Joined evaluation records: {len(annotations)}\n")
        file.write(
            f"Primary-protocol eligible samples: {eligible_samples}\n"
        )

        for status, count in sorted(status_counts.items()):
            file.write(f"{status}: {count}\n")

        file.write("\nResults:\n")
        file.write(f"Successful predictions: {summary['successful']}\n")
        file.write(f"Failed eligible samples: {failed_eligible_samples}\n")
        file.write(f"Success rate: {success_rate:.4f}%\n")
        file.write(f"Yaw MAE: {summary['yaw_mae']:.4f} degrees\n")
        file.write(f"Pitch MAE: {summary['pitch_mae']:.4f} degrees\n")
        file.write(f"Roll MAE: {summary['roll_mae']:.4f} degrees\n")
        file.write(
            f"Overall MAE: {summary['overall_mae']:.4f} degrees\n"
        )

        file.write("\nMedian absolute error:\n")
        file.write(f"Yaw: {summary['yaw_median']:.4f} degrees\n")
        file.write(f"Pitch: {summary['pitch_median']:.4f} degrees\n")
        file.write(f"Roll: {summary['roll_median']:.4f} degrees\n")

        file.write("\nStandard deviation of absolute error:\n")
        file.write(f"Yaw: {summary['yaw_std']:.4f} degrees\n")
        file.write(f"Pitch: {summary['pitch_std']:.4f} degrees\n")
        file.write(f"Roll: {summary['roll_std']:.4f} degrees\n")

        file.write("\nRuntime:\n")
        file.write(f"{elapsed / 60.0:.2f} minutes\n")


def validate_staged_evaluator_outputs(
    results_path,
    summary_path,
    database_accounting,
):
    """Validate staged scientific outputs before replacing accepted evidence."""
    expected_columns = [
        "face_id",
        "filepath",
        "protocol_eligible",
        "status",
        "gt_yaw",
        "gt_pitch",
        "gt_roll",
        "pred_yaw",
        "pred_pitch",
        "pred_roll",
        "yaw_error",
        "pitch_error",
        "roll_error",
    ]

    with open(
        results_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != expected_columns:
            raise RuntimeError(
                "Staged evaluator CSV schema does not match the accepted "
                "AFLW result schema."
            )

        rows = list(reader)

    if len(rows) != database_accounting["joined_records"]:
        raise RuntimeError(
            "Staged evaluator CSV row count does not match the AFLW join."
        )

    if len(rows) != 24384:
        raise RuntimeError(
            f"Expected 24384 joined AFLW rows, found {len(rows)}."
        )

    face_ids = [
        int(
            row["face_id"]
        )
        for row in rows
    ]

    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError(
            "Staged evaluator CSV contains duplicate face IDs."
        )

    status_counts = {}
    successful = []
    eligible_count = 0
    outside_count = 0

    for row in rows:
        gt_values = np.asarray(
            [
                float(row["gt_yaw"]),
                float(row["gt_pitch"]),
                float(row["gt_roll"]),
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(gt_values)):
            raise RuntimeError(
                f"Face {row['face_id']} contains non-finite ground-truth pose."
            )

        recomputed_eligible = is_primary_protocol_sample(
            gt_values[0],
            gt_values[1],
            gt_values[2],
        )

        eligible = row["protocol_eligible"].strip().lower() in {
            "true",
            "1",
        }

        if eligible != recomputed_eligible:
            raise RuntimeError(
                f"Face {row['face_id']} has inconsistent protocol eligibility."
            )

        if eligible:
            eligible_count += 1
        else:
            outside_count += 1

        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == "outside_primary_range":
            if eligible:
                raise RuntimeError(
                    f"Face {row['face_id']} is marked outside the primary "
                    "range but is protocol-eligible."
                )

            continue

        if not eligible:
            raise RuntimeError(
                f"Face {row['face_id']} has status {status!r} outside the "
                "primary protocol."
            )

        if status != "ok":
            continue

        values = np.asarray(
            [
                float(row["pred_yaw"]),
                float(row["pred_pitch"]),
                float(row["pred_roll"]),
                float(row["yaw_error"]),
                float(row["pitch_error"]),
                float(row["roll_error"]),
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Face {row['face_id']} contains non-finite prediction data."
            )

        pred_yaw, pred_pitch, pred_roll = values[:3]
        stored_errors = values[3:6]

        recomputed_errors = np.asarray(
            [
                angular_error_degrees(
                    pred_yaw,
                    gt_values[0],
                ),
                angular_error_degrees(
                    pred_pitch,
                    gt_values[1],
                ),
                angular_error_degrees(
                    pred_roll,
                    gt_values[2],
                ),
            ],
            dtype=float,
        )

        if not np.allclose(
            stored_errors,
            recomputed_errors,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Face {row['face_id']} contains an inconsistent angular error."
            )

        successful.append(
            {
                "status": "ok",
                "yaw_error": float(stored_errors[0]),
                "pitch_error": float(stored_errors[1]),
                "roll_error": float(stored_errors[2]),
            }
        )

    expected_status_counts = {
        "image_read_failed": 1,
        "ok": 23407,
        "outside_primary_range": 976,
    }

    if status_counts != expected_status_counts:
        raise RuntimeError(
            "Staged evaluator status accounting differs from the accepted "
            "AFLW protocol population. "
            f"Expected {expected_status_counts}, found {status_counts}. "
            "Final outputs will not be committed."
        )

    if eligible_count != 23408:
        raise RuntimeError(
            f"Expected 23408 primary-protocol samples, found {eligible_count}."
        )

    if outside_count != 976:
        raise RuntimeError(
            f"Expected 976 outside-primary-range samples, found {outside_count}."
        )

    failed_rows = [
        row
        for row in rows
        if row["status"] == "image_read_failed"
    ]

    if len(failed_rows) != 1:
        raise RuntimeError(
            "Expected exactly one AFLW image-read failure."
        )

    failed_row = failed_rows[0]

    if (
        int(failed_row["face_id"]) != 47825
        or failed_row["filepath"] != "2/image09437.jpg"
    ):
        raise RuntimeError(
            "The staged image-read failure does not match the accepted "
            "AFLW dataset failure."
        )

    if len(successful) != 23407:
        raise RuntimeError(
            f"Expected 23407 successful predictions, found {len(successful)}."
        )

    summary = summarize_errors(successful)
    failed_eligible = eligible_count - summary["successful"]
    success_rate = summary["successful"] / eligible_count * 100.0

    summary_text = summary_path.read_text(
        encoding="utf-8"
    )

    required_lines = [
        "FacePose records in database: 24396",
        "Joined evaluation records: 24384",
        "Primary-protocol eligible samples: 23408",
        "image_read_failed: 1",
        "ok: 23407",
        "outside_primary_range: 976",
        f"Successful predictions: {summary['successful']}",
        f"Failed eligible samples: {failed_eligible}",
        f"Success rate: {success_rate:.4f}%",
        f"Yaw MAE: {summary['yaw_mae']:.4f} degrees",
        f"Pitch MAE: {summary['pitch_mae']:.4f} degrees",
        f"Roll MAE: {summary['roll_mae']:.4f} degrees",
        f"Overall MAE: {summary['overall_mae']:.4f} degrees",
        f"Yaw: {summary['yaw_median']:.4f} degrees",
        f"Pitch: {summary['pitch_median']:.4f} degrees",
        f"Roll: {summary['roll_median']:.4f} degrees",
        f"Yaw: {summary['yaw_std']:.4f} degrees",
        f"Pitch: {summary['pitch_std']:.4f} degrees",
        f"Roll: {summary['roll_std']:.4f} degrees",
    ]

    missing_lines = [
        line
        for line in required_lines
        if line not in summary_text
    ]

    if missing_lines:
        raise RuntimeError(
            "Staged evaluator summary is inconsistent with the staged CSV: "
            + "; ".join(missing_lines)
        )


def replace_owned_outputs(staging_dir):
    """Replace only evaluator-owned outputs with rollback protection."""
    final_paths = [
        RESULTS_CSV,
        SUMMARY_TXT,
    ]
    staged_paths = [
        staging_dir / RESULTS_CSV.name,
        staging_dir / SUMMARY_TXT.name,
    ]

    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backups = []
    installed = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                backup_path = backup_dir / final_path.name
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))

        for staged_path, final_path in zip(staged_paths, final_paths):
            os.replace(staged_path, final_path)
            installed.append(final_path)

    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()

        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, final_path)

        raise



def parse_args():
    """Parse optional reproducibility/preflight arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PhysioTrack head pose on AFLW."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate the AFLW layout and database accounting, then exit "
            "without loading the model or running inference."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    validate_dataset_layout()
    database_accounting = load_database_accounting()
    annotations = load_annotations()

    if len(annotations) != database_accounting["joined_records"]:
        raise RuntimeError(
            "AFLW annotation query count changed between preflight and load."
        )

    print("AFLW dataset preflight: PASS")
    print("Dataset root:", DATASET_ROOT)
    print(
        "FacePose records in database:",
        database_accounting["face_pose_records"],
    )
    print("Joined evaluation records:", len(annotations))
    print(
        "Primary protocol:",
        "controlled coarse-pose evaluation within "
        "|yaw| <= 90, |pitch| < 90, |roll| <= 90 degrees",
    )
    print("Device:", DEVICE)

    if args.preflight_only:
        print("Preflight-only mode: no model inference was run.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".aflw_head_pose_eval_",
            dir=RESULTS_DIR,
        )
    )

    staged_results = staging_dir / RESULTS_CSV.name
    staged_summary = staging_dir / SUMMARY_TXT.name

    print("Staging directory:", staging_dir)

    try:
        estimator = FaceOrientation(device=DEVICE, verbose=False)

        rows = []
        start_time = time.perf_counter()

        for index, row in enumerate(annotations, start=1):
            result = evaluate_sample(row, estimator)
            rows.append(result)

            if index % 500 == 0:
                print(f"Processed {index}/{len(annotations)} records")

        elapsed = time.perf_counter() - start_time

        status_counts = {}
        for row in rows:
            status = row["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        summary = summarize_errors(rows)

        eligible_samples = sum(
            1 for row in rows if row["protocol_eligible"]
        )
        failed_eligible_samples = eligible_samples - summary["successful"]

        success_rate = (
            summary["successful"] / eligible_samples * 100.0
            if eligible_samples > 0
            else 0.0
        )

        print("\n=== AFLW Head Pose Validation ===\n")
        print(
            "Database FacePose records:",
            database_accounting["face_pose_records"],
        )
        print("Joined evaluation records:", len(annotations))
        print("Primary-protocol eligible samples:", eligible_samples)

        for status, count in sorted(status_counts.items()):
            print(f"{status}: {count}")

        print("\nPrimary evaluation results:")
        print(f"Successful predictions: {summary['successful']}")
        print(f"Failed eligible samples: {failed_eligible_samples}")
        print(f"Success rate: {success_rate:.4f}%")
        print(f"Yaw MAE: {summary['yaw_mae']:.4f} degrees")
        print(f"Pitch MAE: {summary['pitch_mae']:.4f} degrees")
        print(f"Roll MAE: {summary['roll_mae']:.4f} degrees")
        print(f"Overall MAE: {summary['overall_mae']:.4f} degrees")

        print("\nMedian absolute error:")
        print(f"Yaw: {summary['yaw_median']:.4f} degrees")
        print(f"Pitch: {summary['pitch_median']:.4f} degrees")
        print(f"Roll: {summary['roll_median']:.4f} degrees")

        print("\nStandard deviation of absolute error:")
        print(f"Yaw: {summary['yaw_std']:.4f} degrees")
        print(f"Pitch: {summary['pitch_std']:.4f} degrees")
        print(f"Roll: {summary['roll_std']:.4f} degrees")

        print(f"\nRuntime: {elapsed / 60.0:.2f} minutes")

        save_results(rows, staged_results)
        save_summary(
            staged_summary,
            database_accounting,
            annotations,
            status_counts,
            summary,
            eligible_samples,
            failed_eligible_samples,
            success_rate,
            elapsed,
        )

        print("\nValidating staged evaluator outputs...")
        validate_staged_evaluator_outputs(
            staged_results,
            staged_summary,
            database_accounting,
        )

        replace_owned_outputs(staging_dir)

        print("Committed final evaluator outputs.")

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nSaved:")
    print(RESULTS_CSV)
    print(SUMMARY_TXT)


if __name__ == "__main__":
    main()
