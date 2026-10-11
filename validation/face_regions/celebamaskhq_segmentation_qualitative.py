from pathlib import Path
import csv
import os
import shutil
import tempfile

import cv2
import numpy as np

from physiotrack.face import FaceRegions

from celebamaskhq_segmentation_eval import (
    CLASS_NAMES,
    IMAGE_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    build_gt_label_map,
    load_image,
    load_test_hq_indices,
)


SCRIPT_DIR = Path(__file__).resolve().parent

SUMMARY_PATH = (
    OUTPUT_DIR
    / "celebamaskhq_segmentation_summary.txt"
)

QUALITATIVE_DIR = (
    OUTPUT_DIR
    / "qualitative"
)

ANNOTATED_DIR = (
    QUALITATIVE_DIR
    / "annotated_images"
)

SELECTION_CSV_PATH = (
    QUALITATIVE_DIR
    / "celebamaskhq_qualitative_selection.csv"
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

COMBINED_FIGURE_PATH = (
    FIGURES_DIR
    / "celebamaskhq_qualitative_examples.png"
)


ACCEPTED_TEST_IMAGES = 2824
ACCEPTED_SUCCESSFUL = 2824
ACCEPTED_FAILED = 0
ACCEPTED_PIXEL_ACCURACY = 0.955849
ACCEPTED_ALL_CLASS_MIOU = 0.815328
ACCEPTED_FOREGROUND_MIOU = 0.808718
ACCEPTED_ALL_CLASS_MDICE = 0.893567
ACCEPTED_FOREGROUND_MDICE = 0.889541

GENERAL_SAMPLE_COUNT = 24
SPECIAL_TOP_COUNT = 10
MIN_GENERAL_FOREGROUND_CLASSES = 6


CLASS_COLORS_BGR = np.array(
    [
        [35, 35, 35],
        [140, 75, 35],
        [180, 160, 120],
        [95, 70, 155],
        [55, 190, 230],
        [40, 150, 245],
        [180, 90, 210],
        [210, 115, 195],
        [215, 150, 80],
        [235, 175, 85],
        [45, 210, 245],
        [75, 85, 220],
        [130, 80, 230],
        [165, 105, 245],
        [60, 120, 60],
        [220, 210, 80],
        [80, 190, 115],
        [170, 120, 80],
        [120, 175, 150],
    ],
    dtype=np.uint8,
)


def load_accepted_summary():
    """Load and verify the accepted full-benchmark summary."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary file not found: {SUMMARY_PATH}"
        )

    values = {}

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for raw_line in file:
            line = raw_line.strip()

            if line.startswith("Official test split size:"):
                values["test_images"] = int(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("Successful images:"):
                values["successful"] = int(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("Failed images:"):
                values["failed"] = int(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("Pixel accuracy:"):
                values["pixel_accuracy"] = float(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("All-class mIoU:"):
                values["all_class_miou"] = float(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("Foreground mIoU:"):
                values["foreground_miou"] = float(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("All-class mean Dice:"):
                values["all_class_mdice"] = float(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("Foreground mean Dice:"):
                values["foreground_mdice"] = float(
                    line.split(":", 1)[1].strip()
                )

    expected = {
        "test_images": ACCEPTED_TEST_IMAGES,
        "successful": ACCEPTED_SUCCESSFUL,
        "failed": ACCEPTED_FAILED,
        "pixel_accuracy": ACCEPTED_PIXEL_ACCURACY,
        "all_class_miou": ACCEPTED_ALL_CLASS_MIOU,
        "foreground_miou": ACCEPTED_FOREGROUND_MIOU,
        "all_class_mdice": ACCEPTED_ALL_CLASS_MDICE,
        "foreground_mdice": ACCEPTED_FOREGROUND_MDICE,
    }

    missing = set(expected) - set(values)

    if missing:
        raise ValueError(
            "Missing accepted summary values: "
            + ", ".join(sorted(missing))
        )

    for key, expected_value in expected.items():
        actual_value = values[key]

        if isinstance(expected_value, int):
            if actual_value != expected_value:
                raise RuntimeError(
                    f"Accepted summary mismatch for {key}: "
                    f"{actual_value} != {expected_value}"
                )
        elif not np.isclose(
            actual_value,
            expected_value,
            atol=5e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Accepted summary mismatch for {key}: "
                f"{actual_value} != {expected_value}"
            )

    return values


def calculate_image_metrics(
    gt_map,
    pred_map,
):
    """Calculate image-level diagnostics for qualitative selection."""
    encoded = (
        NUM_CLASSES
        * gt_map.astype(np.int64).ravel()
        + pred_map.astype(np.int64).ravel()
    )

    confusion_matrix = np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )

    true_positive = np.diag(
        confusion_matrix
    ).astype(np.float64)

    gt_pixels = confusion_matrix.sum(
        axis=1
    ).astype(np.float64)

    pred_pixels = confusion_matrix.sum(
        axis=0
    ).astype(np.float64)

    union = gt_pixels + pred_pixels - true_positive
    dice_denominator = gt_pixels + pred_pixels

    iou = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64,
    )

    dice = np.full(
        NUM_CLASSES,
        np.nan,
        dtype=np.float64,
    )

    valid_iou = union > 0
    valid_dice = dice_denominator > 0

    iou[valid_iou] = (
        true_positive[valid_iou]
        / union[valid_iou]
    )

    dice[valid_dice] = (
        2.0
        * true_positive[valid_dice]
        / dice_denominator[valid_dice]
    )

    gt_present = gt_pixels > 0
    foreground_present = gt_present.copy()
    foreground_present[0] = False

    return {
        "pixel_accuracy": (
            true_positive.sum()
            / confusion_matrix.sum()
        ),
        "all_class_miou": float(
            np.nanmean(iou[gt_present])
        ),
        "foreground_miou": float(
            np.nanmean(iou[foreground_present])
        ),
        "all_class_mdice": float(
            np.nanmean(dice[gt_present])
        ),
        "foreground_mdice": float(
            np.nanmean(dice[foreground_present])
        ),
    }


def profile_ground_truth(image_id):
    """Profile semantic content using ground truth only."""
    gt_map = build_gt_label_map(image_id)

    class_pixels = np.bincount(
        gt_map.ravel(),
        minlength=NUM_CLASSES,
    ).astype(np.int64)

    foreground_present = class_pixels[1:] > 0
    foreground_pixels = int(class_pixels[1:].sum())

    return {
        "image_id": image_id,
        "foreground_classes": int(
            foreground_present.sum()
        ),
        "foreground_ratio": (
            foreground_pixels
            / float(gt_map.size)
        ),
        "class_pixels": class_pixels,
    }


def build_candidate_pool(test_indices):
    """Build a deterministic and semantically diverse candidate pool."""
    profiles = []

    print()
    print("Profiling test-set ground-truth masks...")

    for position, image_id in enumerate(
        test_indices,
        start=1,
    ):
        profiles.append(
            profile_ground_truth(image_id)
        )

        if (
            position % 250 == 0
            or position == len(test_indices)
        ):
            print(
                f"Profiled {position}/"
                f"{len(test_indices)}"
            )

    profile_by_id = {
        profile["image_id"]: profile
        for profile in profiles
    }

    candidate_ids = set()

    general_positions = np.linspace(
        0,
        len(test_indices) - 1,
        GENERAL_SAMPLE_COUNT,
        dtype=int,
    )

    for position in general_positions:
        candidate_ids.add(
            test_indices[int(position)]
        )

    diversity_ranked = sorted(
        profiles,
        key=lambda item: (
            -item["foreground_classes"],
            -item["foreground_ratio"],
            item["image_id"],
        ),
    )

    for profile in diversity_ranked[:SPECIAL_TOP_COUNT]:
        candidate_ids.add(
            profile["image_id"]
        )

    for class_name in [
        "eye_g",
        "hat",
        "ear_r",
        "neck_l",
    ]:
        class_id = CLASS_NAMES.index(class_name)

        ranked = sorted(
            profiles,
            key=lambda item: (
                -int(item["class_pixels"][class_id]),
                -item["foreground_classes"],
                item["image_id"],
            ),
        )

        positive = [
            profile
            for profile in ranked
            if profile["class_pixels"][class_id] > 0
        ]

        if not positive:
            raise RuntimeError(
                f"No test image contains class {class_name}."
            )

        for profile in positive[:SPECIAL_TOP_COUNT]:
            candidate_ids.add(
                profile["image_id"]
            )

    return (
        sorted(candidate_ids),
        profiles,
        profile_by_id,
    )


def run_candidate_inference(
    candidate_ids,
    profile_by_id,
):
    """Run the accepted FaceRegions protocol on candidate images."""
    face_regions = FaceRegions(
        device="cpu",
        verbose=False,
    )

    records = {}

    print()
    print("Running candidate inference...")

    for position, image_id in enumerate(
        candidate_ids,
        start=1,
    ):
        image = load_image(image_id)
        gt_map = build_gt_label_map(image_id)

        height, width = image.shape[:2]

        full_image_box = np.array(
            [
                [
                    0,
                    0,
                    width,
                    height,
                ]
            ],
            dtype=np.float32,
        )

        output = face_regions.predict(
            image,
            boxes=full_image_box,
        )

        result = output["result"]

        if result.seg_map is None:
            raise RuntimeError(
                f"Segmentation map is None for HQ {image_id}."
            )

        pred_map = np.asarray(
            result.seg_map,
            dtype=np.uint8,
        )

        pred_map = np.squeeze(pred_map)

        if pred_map.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape for HQ {image_id}: "
                f"{pred_map.shape}"
            )

        if pred_map.shape != gt_map.shape:
            raise ValueError(
                f"Prediction shape mismatch for HQ {image_id}: "
                f"{pred_map.shape} != {gt_map.shape}"
            )

        records[image_id] = {
            "image": image,
            "gt_map": gt_map,
            "pred_map": pred_map,
            "metrics": calculate_image_metrics(
                gt_map,
                pred_map,
            ),
            "profile": profile_by_id[image_id],
        }

        print(
            f"[{position:03d}/{len(candidate_ids):03d}] "
            f"HQ {image_id}: foreground mIoU="
            f"{records[image_id]['metrics']['foreground_miou']:.4f}"
        )

    return records


def select_first_unique(
    ranked_ids,
    used_ids,
):
    for image_id in ranked_ids:
        if image_id not in used_ids:
            used_ids.add(image_id)
            return image_id

    raise RuntimeError(
        "Could not select a unique qualitative image."
    )


def select_qualitative_cases(
    candidate_records,
    profiles,
):
    """Select eight deterministic qualitative benchmark cases."""
    used_ids = set()
    candidate_ids = list(candidate_records)

    eligible_general = [
        image_id
        for image_id in candidate_ids
        if candidate_records[image_id]["profile"][
            "foreground_classes"
        ] >= MIN_GENERAL_FOREGROUND_CLASSES
    ]

    if len(eligible_general) < 3:
        raise RuntimeError(
            "Too few general candidates for qualitative selection."
        )

    strong_ranked = sorted(
        eligible_general,
        key=lambda image_id: (
            -candidate_records[image_id]["metrics"][
                "foreground_miou"
            ],
            image_id,
        ),
    )

    representative_ranked = sorted(
        eligible_general,
        key=lambda image_id: (
            abs(
                candidate_records[image_id]["metrics"][
                    "foreground_miou"
                ]
                - ACCEPTED_FOREGROUND_MIOU
            ),
            image_id,
        ),
    )

    challenging_ranked = sorted(
        eligible_general,
        key=lambda image_id: (
            candidate_records[image_id]["metrics"][
                "foreground_miou"
            ],
            image_id,
        ),
    )

    selections = [
        {
            "role": "strong_candidate",
            "image_id": select_first_unique(
                strong_ranked,
                used_ids,
            ),
            "selection": (
                "Highest foreground mIoU within the deterministic "
                "qualitative candidate pool"
            ),
        },
        {
            "role": "representative_candidate",
            "image_id": select_first_unique(
                representative_ranked,
                used_ids,
            ),
            "selection": (
                "Foreground mIoU closest to the accepted full-test "
                "foreground mIoU within the deterministic candidate pool"
            ),
        },
        {
            "role": "challenging_candidate",
            "image_id": select_first_unique(
                challenging_ranked,
                used_ids,
            ),
            "selection": (
                "Lowest foreground mIoU within the deterministic "
                "qualitative candidate pool"
            ),
        },
    ]

    special_roles = [
        (
            "eye_glasses",
            "eye_g",
            "Large ground-truth eye_g region",
        ),
        (
            "hat",
            "hat",
            "Large ground-truth hat region",
        ),
        (
            "earring",
            "ear_r",
            "Large ground-truth ear_r region",
        ),
        (
            "necklace",
            "neck_l",
            "Large ground-truth neck_l region",
        ),
    ]

    candidate_id_set = set(candidate_ids)

    for role, class_name, rationale in special_roles:
        class_id = CLASS_NAMES.index(class_name)

        ranked_profiles = sorted(
            [
                profile
                for profile in profiles
                if profile["image_id"] in candidate_id_set
                and profile["class_pixels"][class_id] > 0
            ],
            key=lambda profile: (
                -int(profile["class_pixels"][class_id]),
                -profile["foreground_classes"],
                profile["image_id"],
            ),
        )

        ranked_ids = [
            profile["image_id"]
            for profile in ranked_profiles
        ]

        selections.append(
            {
                "role": role,
                "image_id": select_first_unique(
                    ranked_ids,
                    used_ids,
                ),
                "selection": rationale,
            }
        )

    diversity_ranked = sorted(
        candidate_ids,
        key=lambda image_id: (
            -candidate_records[image_id]["profile"][
                "foreground_classes"
            ],
            -candidate_records[image_id]["profile"][
                "foreground_ratio"
            ],
            image_id,
        ),
    )

    selections.append(
        {
            "role": "high_semantic_diversity",
            "image_id": select_first_unique(
                diversity_ranked,
                used_ids,
            ),
            "selection": (
                "Highest semantic-class diversity among remaining "
                "qualitative candidates"
            ),
        }
    )

    if len(selections) != 8:
        raise RuntimeError(
            f"Expected 8 qualitative cases, found {len(selections)}."
        )

    return selections


def label_map_to_color(label_map):
    return CLASS_COLORS_BGR[label_map]


def blend_label_map(
    image,
    label_map,
    alpha=0.48,
):
    color_map = label_map_to_color(label_map)

    return cv2.addWeighted(
        image,
        1.0 - alpha,
        color_map,
        alpha,
        0.0,
    )


def add_text(
    canvas,
    text,
    origin,
    scale=0.48,
    thickness=1,
):
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def render_qualitative_image(
    selection,
    record,
    accepted_summary,
    output_path,
):
    """Render one 1920 x 1080 qualitative evidence image."""
    image = record["image"]
    gt_map = record["gt_map"]
    pred_map = record["pred_map"]
    metrics = record["metrics"]
    profile = record["profile"]

    gt_overlay = blend_label_map(
        image,
        gt_map,
    )

    pred_overlay = blend_label_map(
        image,
        pred_map,
    )

    canvas = np.full(
        (1080, 1920, 3),
        24,
        dtype=np.uint8,
    )

    image_size = 500
    top = 150

    panels = [
        ("Original", image),
        ("Ground Truth", gt_overlay),
        ("PhysioTrack Prediction", pred_overlay),
    ]

    for left, (title, panel) in zip(
        [20, 530, 1040],
        panels,
    ):
        resized = cv2.resize(
            panel,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas[
            top:top + image_size,
            left:left + image_size,
        ] = resized

        add_text(
            canvas,
            title,
            (left, 125),
            scale=0.72,
            thickness=2,
        )

    panel_left = 1560

    cv2.rectangle(
        canvas,
        (panel_left, 20),
        (1900, 1060),
        (52, 52, 52),
        -1,
    )

    info_lines = [
        ("CelebAMask-HQ", 55, 0.72, 2),
        ("Face Region Segmentation", 85, 0.50, 1),
        ("QUALITATIVE CASE", 135, 0.48, 1),
        (selection["role"], 165, 0.53, 2),
        (f"HQ image: {selection['image_id']}.jpg", 195, 0.43, 1),
        ("IMAGE-LEVEL DIAGNOSTICS", 255, 0.46, 1),
        (f"Pixel accuracy: {metrics['pixel_accuracy'] * 100.0:.2f}%", 290, 0.43, 1),
        (f"All-class mIoU: {metrics['all_class_miou'] * 100.0:.2f}%", 320, 0.43, 1),
        (f"Foreground mIoU: {metrics['foreground_miou'] * 100.0:.2f}%", 350, 0.43, 1),
        (f"All-class Dice: {metrics['all_class_mdice'] * 100.0:.2f}%", 380, 0.43, 1),
        (f"Foreground Dice: {metrics['foreground_mdice'] * 100.0:.2f}%", 410, 0.43, 1),
        (f"GT foreground classes: {profile['foreground_classes']}", 440, 0.43, 1),
        (f"GT foreground area: {profile['foreground_ratio'] * 100.0:.2f}%", 470, 0.43, 1),
        ("ACCEPTED FULL BENCHMARK", 535, 0.46, 1),
        (f"Images: {accepted_summary['test_images']}", 570, 0.43, 1),
        (f"Successful: {accepted_summary['successful']}", 600, 0.43, 1),
        (f"Failed: {accepted_summary['failed']}", 630, 0.43, 1),
        (f"Pixel accuracy: {accepted_summary['pixel_accuracy'] * 100.0:.4f}%", 660, 0.43, 1),
        (f"Foreground mIoU: {accepted_summary['foreground_miou'] * 100.0:.4f}%", 690, 0.43, 1),
        (f"Foreground Dice: {accepted_summary['foreground_mdice'] * 100.0:.4f}%", 720, 0.43, 1),
        ("PROTOCOL", 785, 0.46, 1),
        ("Input / GT: 512 x 512", 820, 0.43, 1),
        ("Classes: 19", 850, 0.43, 1),
        ("Initialization: full image", 880, 0.43, 1),
        ("Backend: SegFace", 910, 0.43, 1),
        ("Device: CPU", 940, 0.43, 1),
    ]

    for text, y_position, scale, thickness in info_lines:
        add_text(
            canvas,
            text,
            (panel_left + 20, y_position),
            scale=scale,
            thickness=thickness,
        )

    add_text(
        canvas,
        "Image-level metrics are qualitative diagnostics only.",
        (20, 710),
        scale=0.46,
    )

    add_text(
        canvas,
        "Accepted benchmark metrics come from the full 2,824-image",
        (20, 740),
        scale=0.46,
    )

    add_text(
        canvas,
        "dataset-level confusion matrix.",
        (20, 770),
        scale=0.46,
    )

    legend_y = 830

    for class_id, class_name in enumerate(CLASS_NAMES):
        column = class_id // 7
        row = class_id % 7

        x_position = 20 + column * 500
        y_position = legend_y + row * 30

        color = tuple(
            int(value)
            for value in CLASS_COLORS_BGR[class_id]
        )

        cv2.rectangle(
            canvas,
            (x_position, y_position - 14),
            (x_position + 18, y_position + 4),
            color,
            -1,
        )

        add_text(
            canvas,
            class_name,
            (x_position + 28, y_position),
            scale=0.42,
        )

    if not cv2.imwrite(
        str(output_path),
        canvas,
    ):
        raise RuntimeError(
            f"Could not save qualitative image: {output_path}"
        )


def create_combined_figure(
    selections,
    candidate_records,
):
    """Create a compact 2 x 4 prediction-overlay summary figure."""
    tile_width = 480
    tile_height = 540

    canvas = np.full(
        (
            tile_height * 2,
            tile_width * 4,
            3,
        ),
        24,
        dtype=np.uint8,
    )

    for index, selection in enumerate(selections):
        record = candidate_records[
            selection["image_id"]
        ]

        overlay = blend_label_map(
            record["image"],
            record["pred_map"],
        )

        overlay = cv2.resize(
            overlay,
            (460, 460),
            interpolation=cv2.INTER_LINEAR,
        )

        row = index // 4
        column = index % 4

        left = column * tile_width + 10
        top = row * tile_height + 10

        canvas[
            top:top + 460,
            left:left + 460,
        ] = overlay

        foreground_miou = (
            record["metrics"]["foreground_miou"]
            * 100.0
        )

        add_text(
            canvas,
            selection["role"],
            (left, top + 490),
            scale=0.46,
        )

        add_text(
            canvas,
            f"HQ {selection['image_id']} | fg mIoU {foreground_miou:.2f}%",
            (left, top + 520),
            scale=0.40,
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(COMBINED_FIGURE_PATH),
        canvas,
    ):
        raise RuntimeError(
            f"Could not save combined figure: {COMBINED_FIGURE_PATH}"
        )


def save_selection_csv(
    selections,
    candidate_records,
):
    fieldnames = [
        "role",
        "selection",
        "image_id",
        "source_image",
        "pixel_accuracy",
        "all_class_miou",
        "foreground_miou",
        "all_class_mdice",
        "foreground_mdice",
        "foreground_classes",
        "foreground_ratio",
        "present_classes",
        "annotated_image",
    ]

    with SELECTION_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for selection in selections:
            image_id = selection["image_id"]
            record = candidate_records[image_id]
            metrics = record["metrics"]
            profile = record["profile"]

            present_classes = [
                CLASS_NAMES[class_id]
                for class_id in range(1, NUM_CLASSES)
                if profile["class_pixels"][class_id] > 0
            ]

            annotated_name = (
                f"{selection['role']}_hq_{image_id}.png"
            )

            writer.writerow(
                {
                    "role": selection["role"],
                    "selection": selection["selection"],
                    "image_id": image_id,
                    "source_image": f"{image_id}.jpg",
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "all_class_miou": metrics["all_class_miou"],
                    "foreground_miou": metrics["foreground_miou"],
                    "all_class_mdice": metrics["all_class_mdice"],
                    "foreground_mdice": metrics["foreground_mdice"],
                    "foreground_classes": profile["foreground_classes"],
                    "foreground_ratio": profile["foreground_ratio"],
                    "present_classes": ";".join(present_classes),
                    "annotated_image": (
                        "results/qualitative/annotated_images/"
                        f"{annotated_name}"
                    ),
                }
            )


def clean_previous_qualitative_outputs():
    """Remove only outputs owned by this qualitative generator."""
    if QUALITATIVE_DIR.exists():
        shutil.rmtree(QUALITATIVE_DIR)

    if COMBINED_FIGURE_PATH.exists():
        COMBINED_FIGURE_PATH.unlink()


def run_qualitative_generation():
    accepted_summary = load_accepted_summary()
    test_indices = load_test_hq_indices()

    if len(test_indices) != ACCEPTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {ACCEPTED_TEST_IMAGES} test images, "
            f"found {len(test_indices)}."
        )

    (
        candidate_ids,
        profiles,
        profile_by_id,
    ) = build_candidate_pool(test_indices)

    print()
    print(
        f"Qualitative candidate pool: "
        f"{len(candidate_ids)} images"
    )

    candidate_records = run_candidate_inference(
        candidate_ids,
        profile_by_id,
    )

    selections = select_qualitative_cases(
        candidate_records,
        profiles,
    )

    print()
    print("Selected qualitative cases:")

    for selection in selections:
        image_id = selection["image_id"]
        metrics = candidate_records[image_id]["metrics"]

        print(
            f"- {selection['role']}: HQ {image_id}, "
            f"foreground mIoU="
            f"{metrics['foreground_miou']:.4f}"
        )

    # All profiling, inference, validation, and selection complete before
    # previous qualitative evidence is replaced.
    clean_previous_qualitative_outputs()

    ANNOTATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for selection in selections:
        image_id = selection["image_id"]

        output_path = (
            ANNOTATED_DIR
            / f"{selection['role']}_hq_{image_id}.png"
        )

        render_qualitative_image(
            selection=selection,
            record=candidate_records[image_id],
            accepted_summary=accepted_summary,
            output_path=output_path,
        )

    save_selection_csv(
        selections,
        candidate_records,
    )

    create_combined_figure(
        selections,
        candidate_records,
    )

    print()
    print(
        "Qualitative benchmark evidence completed successfully."
    )

    print(
        f"Selection CSV: {SELECTION_CSV_PATH}"
    )

    print(
        f"Annotated images: {ANNOTATED_DIR}"
    )

    print(
        f"Combined figure: {COMBINED_FIGURE_PATH}"
    )



def validate_staged_qualitative_outputs():
    """Validate newly generated qualitative artifacts before final replacement."""
    if not QUALITATIVE_DIR.is_dir() or not ANNOTATED_DIR.is_dir():
        raise RuntimeError("Staged qualitative directories were not created.")

    images = sorted(ANNOTATED_DIR.glob("*.png"))
    if len(images) != 8:
        raise RuntimeError(f"Expected 8 staged annotated images, found {len(images)}.")
    for path in images:
        if cv2.imread(str(path)) is None:
            raise RuntimeError(f"Could not read staged qualitative image: {path.name}")

    if not SELECTION_CSV_PATH.is_file():
        raise RuntimeError("Staged qualitative selection CSV was not created.")
    with SELECTION_CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 8:
        raise RuntimeError("Staged qualitative selection CSV must contain 8 rows.")

    expected_roles = {
        "strong_candidate", "representative_candidate", "challenging_candidate",
        "high_semantic_diversity", "eye_glasses", "hat", "earring", "necklace",
    }
    if {row["role"] for row in rows} != expected_roles:
        raise RuntimeError("Staged qualitative selection roles are incorrect.")

    image_names = {path.name for path in images}
    for row in rows:
        if Path(row["annotated_image"]).name not in image_names:
            raise RuntimeError("Staged qualitative selection references a missing image.")
        for field in (
            "pixel_accuracy", "all_class_miou", "foreground_miou",
            "all_class_mdice", "foreground_mdice", "foreground_ratio",
        ):
            if not np.isfinite(float(row[field])):
                raise RuntimeError(f"Staged qualitative selection contains non-finite {field}.")

    if not COMBINED_FIGURE_PATH.is_file() or cv2.imread(str(COMBINED_FIGURE_PATH)) is None:
        raise RuntimeError("Staged qualitative combined figure is unreadable.")


def replace_owned_qualitative_outputs(final_qualitative_dir, final_combined_figure_path, staging_dir):
    """Replace qualitative-owned outputs with rollback on commit failure."""
    backup_dir = staging_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_qualitative = backup_dir / final_qualitative_dir.name
    backup_figure = backup_dir / final_combined_figure_path.name
    q_backup = f_backup = q_installed = f_installed = False
    try:
        if final_qualitative_dir.exists():
            os.replace(final_qualitative_dir, backup_qualitative); q_backup = True
        if final_combined_figure_path.exists():
            os.replace(final_combined_figure_path, backup_figure); f_backup = True
        os.replace(QUALITATIVE_DIR, final_qualitative_dir); q_installed = True
        final_combined_figure_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(COMBINED_FIGURE_PATH, final_combined_figure_path); f_installed = True
    except Exception:
        if f_installed and final_combined_figure_path.exists(): final_combined_figure_path.unlink()
        if q_installed and final_qualitative_dir.exists(): shutil.rmtree(final_qualitative_dir)
        if f_backup and backup_figure.exists(): os.replace(backup_figure, final_combined_figure_path)
        if q_backup and backup_qualitative.exists(): os.replace(backup_qualitative, final_qualitative_dir)
        raise


def main():
    """Generate qualitative benchmark evidence transactionally."""
    global QUALITATIVE_DIR, ANNOTATED_DIR, SELECTION_CSV_PATH, FIGURES_DIR, COMBINED_FIGURE_PATH

    final_qualitative_dir = QUALITATIVE_DIR
    final_annotated_dir = ANNOTATED_DIR
    final_selection_csv_path = SELECTION_CSV_PATH
    final_figures_dir = FIGURES_DIR
    final_combined_figure_path = COMBINED_FIGURE_PATH

    staging_dir = Path(tempfile.mkdtemp(prefix=".celebamaskhq_segmentation_qualitative_", dir=OUTPUT_DIR))
    QUALITATIVE_DIR = staging_dir / "qualitative"
    ANNOTATED_DIR = QUALITATIVE_DIR / "annotated_images"
    SELECTION_CSV_PATH = QUALITATIVE_DIR / final_selection_csv_path.name
    FIGURES_DIR = staging_dir / "figures"
    COMBINED_FIGURE_PATH = FIGURES_DIR / final_combined_figure_path.name
    try:
        run_qualitative_generation()
        print()
        print("Validating staged qualitative outputs...")
        validate_staged_qualitative_outputs()
        replace_owned_qualitative_outputs(final_qualitative_dir, final_combined_figure_path, staging_dir)
    finally:
        QUALITATIVE_DIR = final_qualitative_dir
        ANNOTATED_DIR = final_annotated_dir
        SELECTION_CSV_PATH = final_selection_csv_path
        FIGURES_DIR = final_figures_dir
        COMBINED_FIGURE_PATH = final_combined_figure_path
        if staging_dir.exists(): shutil.rmtree(staging_dir)
    print()
    print("Committed final qualitative outputs.")


if __name__ == "__main__":
    main()
