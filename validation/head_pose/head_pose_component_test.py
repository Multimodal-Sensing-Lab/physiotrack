from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import sqlite3
import tempfile
import time

import cv2
import numpy as np

from physiotrack.face.analysis import FaceAnalysis
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.face_orientation import FaceOrientation
from physiotrack.results import Instance, Result


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "AFLW"

DATA_ROOT = DATASET_ROOT / "aflw" / "data"
IMAGE_ROOT = DATA_ROOT / "flickr"
DATABASE_PATH = DATA_ROOT / "aflw.sqlite"

RESULTS_DIR = SCRIPT_DIR / "results"
COMPONENT_OUTPUT_DIR = RESULTS_DIR / "component_execution"

RESULTS_CSV = (
    COMPONENT_OUTPUT_DIR
    / "head_pose_component_results.csv"
)

SUMMARY_JSON = (
    COMPONENT_OUTPUT_DIR
    / "head_pose_component_summary.json"
)

DEVICE = "cpu"

MAX_ABS_YAW = 90.0
MAX_ABS_ROLL = 90.0
MAX_ABS_PITCH = 90.0

EXPECTED_FACE_POSE_RECORDS = 24396
EXPECTED_JOINED_RECORDS = 24384
EXPECTED_PRIMARY_RECORDS = 23408
EXPECTED_OK_RECORDS = 23407

EXPECTED_INPUT_FAILURE_FACE_ID = 47825
EXPECTED_INPUT_FAILURE_FILEPATH = "2/image09437.jpg"

REQUIRED_DATABASE_TABLES = {
    "FaceImages",
    "Faces",
    "FaceRect",
    "FacePose",
}


class ControlledFaceDetector:
    """Return exactly one externally controlled AFLW face rectangle."""

    def __init__(self):
        self.current_box = None

    def set_box(self, box):
        """Set the one face rectangle returned by the detector."""
        box = np.asarray(
            box,
            dtype=float,
        ).reshape(-1)

        if box.size != 4:
            raise ValueError(
                "Controlled face box must contain four coordinates."
            )

        if not np.all(
            np.isfinite(
                box
            )
        ):
            raise ValueError(
                "Controlled face box contains non-finite values."
            )

        self.current_box = box.copy()

    def predict(self, frame):
        """Return one PhysioTrack face Result using the controlled box."""
        if self.current_box is None:
            raise RuntimeError(
                "Controlled face box was not set before prediction."
            )

        return Result(
            orig_img=frame,
            instances=[
                Instance(
                    id=None,
                    box=self.current_box.copy(),
                    confidence=1.0,
                    cls=0,
                    cls_name="face",
                )
            ],
            task="face",
        )


def parse_args():
    """Parse component-test arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the real PhysioTrack Head Pose component through "
            "FaceAnalysis on the accepted AFLW primary-protocol population."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate AFLW layout and protocol accounting without "
            "loading the model."
        ),
    )

    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small number of real FaceAnalysis/FaceOrientation "
            "inferences without writing final outputs."
        ),
    )

    parser.add_argument(
        "--smoke-count",
        type=int,
        default=3,
        help="Number of successful faces required in smoke-test mode.",
    )

    return parser.parse_args()


def validate_dataset_layout():
    """Validate the AFLW files required by this component test."""
    required_paths = [
        DATABASE_PATH,
        IMAGE_ROOT / "0",
        IMAGE_ROOT / "2",
        IMAGE_ROOT / "3",
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "AFLW dataset layout is incomplete. Missing required paths:\n"
            f"{formatted}\n\n"
            "See validation/head_pose/README_AFLW_HEAD_POSE.txt "
            "for the required AFLW directory layout."
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

        available_tables = {
            row[0]
            for row in cursor.fetchall()
        }

    finally:
        connection.close()

    missing_tables = sorted(
        REQUIRED_DATABASE_TABLES
        - available_tables
    )

    if missing_tables:
        raise RuntimeError(
            "AFLW SQLite database is missing required tables: "
            + ", ".join(
                missing_tables
            )
        )


def load_database_accounting():
    """Return and validate AFLW database counts."""
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM FacePose"
        )
        face_pose_records = int(
            cursor.fetchone()[0]
        )

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
        joined_records = int(
            cursor.fetchone()[0]
        )

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
        distinct_joined_faces = int(
            cursor.fetchone()[0]
        )

    finally:
        connection.close()

    if face_pose_records != EXPECTED_FACE_POSE_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_FACE_POSE_RECORDS} FacePose records, "
            f"found {face_pose_records}."
        )

    if joined_records != EXPECTED_JOINED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOINED_RECORDS} joined records, "
            f"found {joined_records}."
        )

    if joined_records != distinct_joined_faces:
        raise RuntimeError(
            "The AFLW evaluation join produced duplicate face IDs."
        )

    return {
        "face_pose_records": face_pose_records,
        "joined_records": joined_records,
        "distinct_joined_faces": distinct_joined_faces,
    }


def load_annotations():
    """Load AFLW face rectangles and pose annotations in read-only mode."""
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH}?mode=ro",
        uri=True,
    )

    try:
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

    finally:
        connection.close()

    face_ids = [
        row[0]
        for row in rows
    ]

    if len(
        face_ids
    ) != len(
        set(
            face_ids
        )
    ):
        raise RuntimeError(
            "Duplicate face IDs were found in the AFLW join."
        )

    return rows


def convert_aflw_pose(
    roll,
    pitch,
    yaw,
):
    """Convert AFLW pose values to the accepted PhysioTrack convention."""
    gt_roll = -float(
        np.degrees(
            roll
        )
    )

    gt_pitch = -float(
        np.degrees(
            pitch
        )
    )

    gt_yaw = -float(
        np.degrees(
            yaw
        )
    )

    return (
        gt_yaw,
        gt_pitch,
        gt_roll,
    )


def is_primary_protocol_sample(
    yaw,
    pitch,
    roll,
):
    """Return whether a sample belongs to the accepted primary range."""
    return (
        abs(
            yaw
        )
        <= MAX_ABS_YAW
        and abs(
            roll
        )
        <= MAX_ABS_ROLL
        and abs(
            pitch
        )
        < MAX_ABS_PITCH
    )


def load_primary_annotations():
    """Return exactly the accepted AFLW primary-protocol population."""
    annotations = load_annotations()

    primary = []

    for row in annotations:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            raw_roll,
            raw_pitch,
            raw_yaw,
        ) = row

        (
            gt_yaw,
            gt_pitch,
            gt_roll,
        ) = convert_aflw_pose(
            raw_roll,
            raw_pitch,
            raw_yaw,
        )

        if is_primary_protocol_sample(
            gt_yaw,
            gt_pitch,
            gt_roll,
        ):
            primary.append(
                row
            )

    if len(
        primary
    ) != EXPECTED_PRIMARY_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY_RECORDS} primary-protocol "
            f"records, found {len(primary)}."
        )

    return (
        annotations,
        primary,
    )


def make_face_box(
    x,
    y,
    w,
    h,
    image_width,
    image_height,
):
    """Convert an AFLW face rectangle to a valid 1D xyxy box."""
    x1 = max(
        0.0,
        float(
            x
        ),
    )

    y1 = max(
        0.0,
        float(
            y
        ),
    )

    x2 = min(
        float(
            image_width
        ),
        float(
            x
            + w
        ),
    )

    y2 = min(
        float(
            image_height
        ),
        float(
            y
            + h
        ),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return np.asarray(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dtype=np.float32,
    )


def make_config():
    """Enable Head Pose and disable unrelated face-analysis components."""
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=True,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )

    config.validate()

    return config


def validate_pipeline_configuration(
    pipeline,
):
    """Verify that the real pipeline contains only the intended component."""
    if not isinstance(
        pipeline.orientation,
        FaceOrientation,
    ):
        raise RuntimeError(
            "FaceAnalysis did not initialize the real FaceOrientation component."
        )

    unrelated_components = {
        "tracker": pipeline.tracker,
        "landmarks": pipeline.landmarks,
        "quality": pipeline.quality,
        "eyes": pipeline.eyes,
        "blink": pipeline.blink,
        "gaze": pipeline.gaze,
        "gaze_estimation": pipeline.gaze_estimation,
        "mouth": pipeline.mouth,
        "mouth_motion": pipeline.mouth_motion,
        "emotion": pipeline.emotion,
        "regions": pipeline.regions,
        "temporal": pipeline.temporal,
    }

    enabled_unrelated = [
        name
        for (
            name,
            component,
        ) in unrelated_components.items()
        if component is not None
    ]

    if enabled_unrelated:
        raise RuntimeError(
            "Unexpected face-analysis components are enabled: "
            + ", ".join(
                enabled_unrelated
            )
        )


def make_empty_result(
    face_id,
    filepath,
    status,
    failure_reason,
):
    """Return one structured non-success row."""
    return {
        "face_id": int(
            face_id
        ),
        "filepath": filepath,
        "input_box_x1": "",
        "input_box_y1": "",
        "input_box_x2": "",
        "input_box_y2": "",
        "yaw": "",
        "pitch": "",
        "roll": "",
        "status": status,
        "failure_reason": failure_reason,
    }


def execute_sample(
    row,
    pipeline,
    detector,
):
    """Run one accepted AFLW face through the real PhysioTrack pipeline."""
    (
        face_id,
        filepath,
        x,
        y,
        w,
        h,
        _,
        _,
        _,
    ) = row

    image_path = (
        IMAGE_ROOT
        / filepath
    )

    if not image_path.is_file():
        return make_empty_result(
            face_id,
            filepath,
            "INPUT_MISSING",
            "Image file does not exist.",
        )

    frame = cv2.imread(
        str(
            image_path
        )
    )

    if frame is None:
        return make_empty_result(
            face_id,
            filepath,
            "IMAGE_READ_FAILED",
            "OpenCV could not decode the image.",
        )

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
        return make_empty_result(
            face_id,
            filepath,
            "INVALID_FACE_BOX",
            "AFLW face rectangle is invalid after image-bound clipping.",
        )

    detector.set_box(
        face_box
    )

    try:
        result = pipeline.predict(
            frame
        )

    except Exception as error:
        return make_empty_result(
            face_id,
            filepath,
            "EXECUTION_FAILED",
            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    if len(
        result
    ) != 1:
        return make_empty_result(
            face_id,
            filepath,
            "RESULT_COUNT_MISMATCH",
            (
                "Expected exactly one controlled FaceAnalysis result, "
                f"found {len(result)}."
            ),
        )

    instance = result[0]

    returned_box = np.asarray(
        instance.box,
        dtype=float,
    ).reshape(-1)

    if (
        returned_box.size != 4
        or not np.all(
            np.isfinite(
                returned_box
            )
        )
    ):
        return make_empty_result(
            face_id,
            filepath,
            "INVALID_RETURNED_BOX",
            "FaceAnalysis returned an invalid face rectangle.",
        )

    if not np.allclose(
        returned_box,
        face_box,
        rtol=0.0,
        atol=1e-9,
    ):
        return make_empty_result(
            face_id,
            filepath,
            "RETURNED_BOX_MISMATCH",
            "FaceAnalysis did not preserve the controlled AFLW face rectangle.",
        )

    orientation = instance.orientation

    if orientation is None:
        return make_empty_result(
            face_id,
            filepath,
            "NO_ORIENTATION",
            "FaceAnalysis returned no Head Pose orientation.",
        )

    try:
        yaw = float(
            orientation[
                "yaw"
            ]
        )
        pitch = float(
            orientation[
                "pitch"
            ]
        )
        roll = float(
            orientation[
                "roll"
            ]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return make_empty_result(
            face_id,
            filepath,
            "INVALID_ORIENTATION_STRUCTURE",
            str(
                error
            ),
        )

    if not np.all(
        np.isfinite(
            [
                yaw,
                pitch,
                roll,
            ]
        )
    ):
        return make_empty_result(
            face_id,
            filepath,
            "NONFINITE_ORIENTATION",
            "Head Pose output contains a non-finite value.",
        )

    return {
        "face_id": int(
            face_id
        ),
        "filepath": filepath,
        "input_box_x1": float(
            face_box[0]
        ),
        "input_box_y1": float(
            face_box[1]
        ),
        "input_box_x2": float(
            face_box[2]
        ),
        "input_box_y2": float(
            face_box[3]
        ),
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "status": "OK",
        "failure_reason": "",
    }


def write_results_csv(
    output_path,
    rows,
):
    """Write structured numerical Head Pose component outputs."""
    fieldnames = [
        "face_id",
        "filepath",
        "input_box_x1",
        "input_box_y1",
        "input_box_x2",
        "input_box_y2",
        "yaw",
        "pitch",
        "roll",
        "status",
        "failure_reason",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def summarize_rows(
    rows,
):
    """Summarize execution statuses without computing accuracy metrics."""
    status_counts = {}

    for row in rows:
        status = row[
            "status"
        ]

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    successful = status_counts.get(
        "OK",
        0,
    )

    input_failures = sum(
        status_counts.get(
            name,
            0,
        )
        for name in (
            "INPUT_MISSING",
            "IMAGE_READ_FAILED",
            "INVALID_FACE_BOX",
        )
    )

    component_failures = (
        len(
            rows
        )
        - successful
        - input_failures
    )

    if component_failures > 0:
        overall_status = "FAIL"
    elif input_failures > 0:
        overall_status = "PASS_WITH_INPUT_FAILURES"
    else:
        overall_status = "PASS"

    return {
        "status_counts": status_counts,
        "successful_component_outputs": successful,
        "input_failures": input_failures,
        "component_execution_failures": component_failures,
        "status": overall_status,
    }


def validate_staged_outputs(
    results_path,
    summary_path,
):
    """Validate staged isolated-component outputs before final replacement."""
    with results_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    if len(
        rows
    ) != EXPECTED_PRIMARY_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_PRIMARY_RECORDS} rows, found {len(rows)}."
        )

    face_ids = [
        int(
            row[
                "face_id"
            ]
        )
        for row in rows
    ]

    if len(
        face_ids
    ) != len(
        set(
            face_ids
        )
    ):
        raise RuntimeError(
            "Duplicate face IDs were found in staged component results."
        )

    status_counts = {}

    for row in rows:
        status = row[
            "status"
        ]

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

        if status != "OK":
            continue

        numeric_fields = [
            "input_box_x1",
            "input_box_y1",
            "input_box_x2",
            "input_box_y2",
            "yaw",
            "pitch",
            "roll",
        ]

        values = np.asarray(
            [
                float(
                    row[
                        field
                    ]
                )
                for field in numeric_fields
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(
                f"Face {row['face_id']} contains non-finite numerical output."
            )

        if (
            float(
                row[
                    "input_box_x2"
                ]
            )
            <= float(
                row[
                    "input_box_x1"
                ]
            )
            or float(
                row[
                    "input_box_y2"
                ]
            )
            <= float(
                row[
                    "input_box_y1"
                ]
            )
        ):
            raise RuntimeError(
                f"Face {row['face_id']} contains an invalid input box."
            )

    expected_status_counts = {
        "OK": EXPECTED_OK_RECORDS,
        "IMAGE_READ_FAILED": 1,
    }

    if status_counts != expected_status_counts:
        raise RuntimeError(
            "Staged component status accounting does not match the accepted "
            "AFLW primary-protocol input population. "
            f"Expected {expected_status_counts}, found {status_counts}. "
            "Final outputs will not be committed."
        )

    failure_rows = [
        row
        for row in rows
        if row[
            "status"
        ] == "IMAGE_READ_FAILED"
    ]

    failure = failure_rows[0]

    if (
        int(
            failure[
                "face_id"
            ]
        )
        != EXPECTED_INPUT_FAILURE_FACE_ID
        or failure[
            "filepath"
        ]
        != EXPECTED_INPUT_FAILURE_FILEPATH
    ):
        raise RuntimeError(
            "The isolated input failure does not match the accepted AFLW "
            "benchmark input failure."
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    expected_summary = {
        "expected_primary_protocol_records": EXPECTED_PRIMARY_RECORDS,
        "processed_primary_protocol_records": EXPECTED_PRIMARY_RECORDS,
        "successful_component_outputs": EXPECTED_OK_RECORDS,
        "input_failures": 1,
        "component_execution_failures": 0,
        "status": "PASS_WITH_INPUT_FAILURES",
    }

    for (
        key,
        expected_value,
    ) in expected_summary.items():
        if summary.get(
            key
        ) != expected_value:
            raise RuntimeError(
                f"Summary field {key!r} does not match staged CSV accounting."
            )

    if summary.get(
        "status_counts"
    ) != expected_status_counts:
        raise RuntimeError(
            "Summary status_counts do not match staged CSV accounting."
        )


def replace_owned_outputs(
    staging_dir,
):
    """Replace only isolated-component outputs with rollback protection."""
    COMPONENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_results = (
        staging_dir
        / RESULTS_CSV.name
    )

    staged_summary = (
        staging_dir
        / SUMMARY_JSON.name
    )

    final_paths = [
        RESULTS_CSV,
        SUMMARY_JSON,
    ]

    staged_paths = [
        staged_results,
        staged_summary,
    ]

    backup_dir = (
        staging_dir
        / "backup"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backups = []
    installed = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                backup_path = (
                    backup_dir
                    / final_path.name
                )

                os.replace(
                    final_path,
                    backup_path,
                )

                backups.append(
                    (
                        backup_path,
                        final_path,
                    )
                )

        for (
            staged_path,
            final_path,
        ) in zip(
            staged_paths,
            final_paths,
        ):
            os.replace(
                staged_path,
                final_path,
            )

            installed.append(
                final_path
            )

    except Exception:
        for final_path in installed:
            if final_path.exists():
                final_path.unlink()

        for (
            backup_path,
            final_path,
        ) in reversed(
            backups
        ):
            if backup_path.exists():
                os.replace(
                    backup_path,
                    final_path,
                )

        raise


def run_smoke_test(
    primary_annotations,
    pipeline,
    detector,
    smoke_count,
):
    """Run a small real-inference check without writing final outputs."""
    if smoke_count < 1:
        raise ValueError(
            "--smoke-count must be at least 1."
        )

    successful = 0
    skipped_input_failures = 0

    for row in primary_annotations:
        result = execute_sample(
            row,
            pipeline,
            detector,
        )

        if result[
            "status"
        ] == "OK":
            successful += 1

            print(
                f"Smoke test OK: face_id={result['face_id']}, "
                f"yaw={result['yaw']:.6f}, "
                f"pitch={result['pitch']:.6f}, "
                f"roll={result['roll']:.6f}"
            )

            if successful >= smoke_count:
                break

        elif result[
            "status"
        ] in {
            "INPUT_MISSING",
            "IMAGE_READ_FAILED",
            "INVALID_FACE_BOX",
        }:
            skipped_input_failures += 1

        else:
            raise RuntimeError(
                "Smoke test detected a component execution failure: "
                f"face_id={result['face_id']}, "
                f"status={result['status']}, "
                f"reason={result['failure_reason']}"
            )

    if successful != smoke_count:
        raise RuntimeError(
            f"Smoke test requested {smoke_count} successful faces but "
            f"obtained {successful}."
        )

    print()
    print(
        "Smoke-test component execution: PASS"
    )
    print(
        "Successful real-inference faces:",
        successful,
    )
    print(
        "Input failures skipped during smoke test:",
        skipped_input_failures,
    )
    print(
        "No final result files were written."
    )


def main():
    """Run isolated real PhysioTrack Head Pose execution on AFLW."""
    args = parse_args()

    validate_dataset_layout()
    accounting = load_database_accounting()

    (
        annotations,
        primary_annotations,
    ) = load_primary_annotations()

    if len(
        annotations
    ) != accounting[
        "joined_records"
    ]:
        raise RuntimeError(
            "AFLW annotation count changed between preflight checks."
        )

    print("=" * 78)
    print(
        "PhysioTrack Head Pose Isolated Component Execution"
    )
    print("=" * 78)
    print(
        "Dataset root:",
        DATASET_ROOT,
    )
    print(
        "FacePose records:",
        accounting[
            "face_pose_records"
        ],
    )
    print(
        "Joined records:",
        accounting[
            "joined_records"
        ],
    )
    print(
        "Primary-protocol records:",
        len(
            primary_annotations
        ),
    )
    print(
        "Primary protocol:",
        "|yaw| <= 90, |pitch| < 90, |roll| <= 90 degrees",
    )
    print(
        "Pipeline: PhysioTrack FaceAnalysis"
    )
    print(
        "Target component: FaceOrientation"
    )
    print(
        "Backend: 6DRepNet360"
    )
    print(
        "Device:",
        DEVICE,
    )
    print(
        "Input face boxes: AFLW ground-truth rectangles"
    )
    print(
        "Tracking: disabled"
    )
    print(
        "Unrelated face-analysis components: disabled"
    )
    print(
        "Accuracy metrics: not computed by this component-execution test"
    )

    if args.preflight_only:
        print()
        print(
            "Preflight-only mode: no model inference was run."
        )
        return

    COMPONENT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".head_pose_component_test_",
            dir=COMPONENT_OUTPUT_DIR,
        )
    )

    print(
        "Staging directory:",
        staging_dir,
    )

    detector = ControlledFaceDetector()
    config = make_config()

    pipeline = None

    try:
        pipeline = FaceAnalysis(
            detector=detector,
            config=config,
            device=DEVICE,
            verbose=False,
        )

        validate_pipeline_configuration(
            pipeline
        )

        if args.smoke_test:
            run_smoke_test(
                primary_annotations,
                pipeline,
                detector,
                args.smoke_count,
            )
            return

        staged_results = (
            staging_dir
            / RESULTS_CSV.name
        )

        staged_summary = (
            staging_dir
            / SUMMARY_JSON.name
        )

        rows = []

        start_time = (
            time.perf_counter()
        )

        for (
            index,
            row,
        ) in enumerate(
            primary_annotations,
            start=1,
        ):
            result = execute_sample(
                row,
                pipeline,
                detector,
            )

            rows.append(
                result
            )

            if index % 500 == 0:
                print(
                    f"Processed {index}/"
                    f"{len(primary_annotations)} records"
                )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        execution_summary = summarize_rows(
            rows
        )

        write_results_csv(
            staged_results,
            rows,
        )

        summary = {
            "component": "Head Pose",
            "execution_type": "isolated_component_execution",
            "dataset": "AFLW",
            "dataset_root": "datasets/AFLW",
            "pipeline": "PhysioTrack FaceAnalysis",
            "target_component": "FaceOrientation",
            "backend": "6DRepNet360",
            "device": DEVICE,
            "input_face_box": "AFLW ground-truth face rectangle",
            "primary_protocol": (
                "|yaw| <= 90 degrees, |pitch| < 90 degrees, "
                "|roll| <= 90 degrees"
            ),
            "ground_truth_use": (
                "AFLW pose annotations are used only to select the same "
                "primary-protocol population as the accepted scientific "
                "benchmark. No angular accuracy metric is computed here."
            ),
            "tracking_enabled": False,
            "unrelated_components_disabled": True,
            "expected_primary_protocol_records": EXPECTED_PRIMARY_RECORDS,
            "processed_primary_protocol_records": len(
                rows
            ),
            "successful_component_outputs": execution_summary[
                "successful_component_outputs"
            ],
            "input_failures": execution_summary[
                "input_failures"
            ],
            "component_execution_failures": execution_summary[
                "component_execution_failures"
            ],
            "status_counts": execution_summary[
                "status_counts"
            ],
            "runtime_seconds": elapsed,
            "runtime_minutes": (
                elapsed
                / 60.0
            ),
            "records_per_second": (
                len(
                    rows
                )
                / elapsed
                if elapsed > 0
                else None
            ),
            "status": execution_summary[
                "status"
            ],
            "interpretation": (
                "This is software execution evidence for the real PhysioTrack "
                "FaceOrientation component through FaceAnalysis. It does not "
                "replace the accepted AFLW angular-accuracy benchmark."
            ),
        }

        staged_summary.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            "Validating staged component-execution outputs..."
        )

        validate_staged_outputs(
            staged_results,
            staged_summary,
        )

        replace_owned_outputs(
            staging_dir
        )

        print(
            "Committed final component-execution outputs."
        )
        print()
        print(
            "Execution summary:"
        )
        print(
            "Primary-protocol records:",
            len(
                rows
            ),
        )
        print(
            "Successful component outputs:",
            execution_summary[
                "successful_component_outputs"
            ],
        )
        print(
            "Input failures:",
            execution_summary[
                "input_failures"
            ],
        )
        print(
            "Component execution failures:",
            execution_summary[
                "component_execution_failures"
            ],
        )

        for (
            status,
            count,
        ) in sorted(
            execution_summary[
                "status_counts"
            ].items()
        ):
            print(
                f"{status}: {count}"
            )

        print(
            f"Runtime: {elapsed / 60.0:.2f} minutes"
        )
        print(
            "Overall status:",
            execution_summary[
                "status"
            ],
        )
        print()
        print(
            "Saved results:",
            RESULTS_CSV,
        )
        print(
            "Saved summary:",
            SUMMARY_JSON,
        )

    finally:
        if (
            pipeline is not None
            and hasattr(
                pipeline,
                "close",
            )
        ):
            pipeline.close()

        if staging_dir.exists():
            shutil.rmtree(
                staging_dir
            )


if __name__ == "__main__":
    main()
