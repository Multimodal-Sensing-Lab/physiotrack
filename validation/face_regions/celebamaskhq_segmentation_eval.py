from pathlib import Path
import csv
import time

import cv2
import numpy as np

from physiotrack.face import FaceRegions


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "CelebAMask-HQ"

IMAGE_DIR = DATASET_ROOT / "CelebA-HQ-img"
MASK_ROOT = DATASET_ROOT / "CelebAMask-HQ-mask-anno"
MAPPING_PATH = DATASET_ROOT / "CelebA-HQ-to-CelebA-mapping.txt"

# The official CelebA partition file is stored with the validation code so
# the downloaded CelebAMask-HQ dataset directory remains unchanged.
PARTITION_PATH = SCRIPT_DIR / "list_eval_partition.txt"

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 19
TEST_PARTITION = 2
EVALUATION_LIMIT = None


LABEL_SUFFIX_ORDER = [
    "neck",
    "skin",
    "cloth",
    "l_ear",
    "r_ear",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "nose",
    "mouth",
    "l_lip",
    "u_lip",
    "hair",
    "eye_g",
    "hat",
    "ear_r",
    "neck_l",
]


CLASS_NAMES = [
    "background",
    *LABEL_SUFFIX_ORDER,
]



def validate_required_paths():
    """Validate the required dataset and protocol files before evaluation."""
    required_paths = {
        "image directory": IMAGE_DIR,
        "mask directory": MASK_ROOT,
        "HQ-to-CelebA mapping": MAPPING_PATH,
        "CelebA partition file": PARTITION_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required CelebAMask-HQ evaluation inputs:\n"
            + "\n".join(missing)
        )


def load_partition_file():
    partitions = {}

    with PARTITION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid partition line {line_number}: {line}"
                )

            filename, partition = parts
            partition = int(partition)

            if partition not in {0, 1, 2}:
                raise ValueError(
                    f"Invalid partition value at line "
                    f"{line_number}: {partition}"
                )

            if filename in partitions:
                raise ValueError(
                    f"Duplicate partition entry for {filename}"
                )

            partitions[filename] = partition

    if len(partitions) != 202599:
        raise RuntimeError(
            f"Expected 202599 CelebA partition entries, "
            f"found {len(partitions)}."
        )

    return partitions


def load_test_hq_indices():
    partitions = load_partition_file()
    test_indices = []

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        file.readline()

        for line_number, line in enumerate(
            file,
            start=2,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid mapping line {line_number}: {line}"
                )

            hq_index = int(parts[0])
            original_filename = parts[2]

            if original_filename not in partitions:
                raise KeyError(
                    f"Missing partition entry for "
                    f"{original_filename}"
                )

            if partitions[original_filename] == TEST_PARTITION:
                test_indices.append(hq_index)

    if len(set(test_indices)) != len(test_indices):
        raise RuntimeError(
            "Duplicate CelebAMask-HQ indices were found in the test split."
        )

    return test_indices


def get_mask_subfolder(image_id):
    return MASK_ROOT / str(image_id // 2000)


def load_image(image_id):
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return cv2.resize(
        image,
        (512, 512),
        interpolation=cv2.INTER_LINEAR,
    )


def build_gt_label_map(image_id):
    gt_map = np.zeros(
        (512, 512),
        dtype=np.uint8,
    )

    mask_dir = get_mask_subfolder(
        image_id
    )

    prefix = f"{image_id:05d}"

    for class_id, class_name in enumerate(
        LABEL_SUFFIX_ORDER,
        start=1,
    ):
        mask_path = (
            mask_dir
            / f"{prefix}_{class_name}.png"
        )

        if not mask_path.exists():
            continue

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"Could not load mask: {mask_path}"
            )

        mask = np.squeeze(
            mask
        )

        if mask.ndim != 2:
            raise ValueError(
                f"Unexpected mask shape for "
                f"{mask_path.name}: {mask.shape}"
            )

        if mask.shape != gt_map.shape:
            raise ValueError(
                f"Mask shape mismatch for "
                f"{mask_path.name}: "
                f"{mask.shape} != {gt_map.shape}"
            )

        gt_map[
            mask > 0
        ] = class_id

    return gt_map


def update_confusion_matrix(
    confusion_matrix,
    gt_map,
    pred_map,
):
    valid = (
        (gt_map >= 0)
        & (gt_map < NUM_CLASSES)
        & (pred_map >= 0)
        & (pred_map < NUM_CLASSES)
    )

    encoded = (
        NUM_CLASSES
        * gt_map[valid].astype(
            np.int64
        )
        + pred_map[valid].astype(
            np.int64
        )
    )

    counts = np.bincount(
        encoded,
        minlength=(
            NUM_CLASSES
            * NUM_CLASSES
        ),
    )

    confusion_matrix += counts.reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )


def calculate_metrics(
    confusion_matrix,
):
    true_positive = np.diag(
        confusion_matrix
    ).astype(
        np.float64
    )

    gt_pixels = confusion_matrix.sum(
        axis=1
    ).astype(
        np.float64
    )

    pred_pixels = confusion_matrix.sum(
        axis=0
    ).astype(
        np.float64
    )

    union = (
        gt_pixels
        + pred_pixels
        - true_positive
    )

    dice_denominator = (
        gt_pixels
        + pred_pixels
    )

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
    valid_dice = (
        dice_denominator > 0
    )

    iou[
        valid_iou
    ] = (
        true_positive[
            valid_iou
        ]
        / union[
            valid_iou
        ]
    )

    dice[
        valid_dice
    ] = (
        2.0
        * true_positive[
            valid_dice
        ]
        / dice_denominator[
            valid_dice
        ]
    )

    total_pixels = (
        confusion_matrix.sum()
    )

    if total_pixels > 0:
        pixel_accuracy = (
            true_positive.sum()
            / total_pixels
        )
    else:
        pixel_accuracy = np.nan

    gt_present = (
        gt_pixels > 0
    )

    foreground_present = (
        gt_present.copy()
    )

    foreground_present[
        0
    ] = False

    all_class_miou = float(
        np.nanmean(
            iou[
                gt_present
            ]
        )
    )

    foreground_miou = float(
        np.nanmean(
            iou[
                foreground_present
            ]
        )
    )

    all_class_mdice = float(
        np.nanmean(
            dice[
                gt_present
            ]
        )
    )

    foreground_mdice = float(
        np.nanmean(
            dice[
                foreground_present
            ]
        )
    )

    return {
        "iou": iou,
        "dice": dice,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "pixel_accuracy": pixel_accuracy,
        "all_class_miou": all_class_miou,
        "foreground_miou": foreground_miou,
        "all_class_mdice": all_class_mdice,
        "foreground_mdice": foreground_mdice,
    }


def save_class_metrics(
    metrics,
):
    output_path = (
        OUTPUT_DIR
        / "celebamaskhq_class_metrics.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "class_id",
                "class_name",
                "gt_pixels",
                "pred_pixels",
                "iou",
                "dice",
            ]
        )

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):
            writer.writerow(
                [
                    class_id,
                    class_name,
                    int(
                        metrics[
                            "gt_pixels"
                        ][
                            class_id
                        ]
                    ),
                    int(
                        metrics[
                            "pred_pixels"
                        ][
                            class_id
                        ]
                    ),
                    metrics[
                        "iou"
                    ][
                        class_id
                    ],
                    metrics[
                        "dice"
                    ][
                        class_id
                    ],
                ]
            )

    return output_path


def save_confusion_matrix(
    confusion_matrix,
):
    output_path = (
        OUTPUT_DIR
        / "celebamaskhq_confusion_matrix.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "gt_class",
                *CLASS_NAMES,
            ]
        )

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):
            writer.writerow(
                [
                    class_name,
                    *confusion_matrix[
                        class_id
                    ].tolist(),
                ]
            )

    return output_path


def save_summary(
    metrics,
    total_test_images,
    evaluated_images,
    successful,
    failed,
    elapsed,
):
    output_path = (
        OUTPUT_DIR
        / "celebamaskhq_segmentation_summary.txt"
    )

    throughput = (
        successful / elapsed
        if elapsed > 0
        else np.nan
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "CelebAMask-HQ Face Region Segmentation Evaluation\n"
        )
        file.write(
            "===============================================\n\n"
        )

        file.write(
            f"Official test split size: "
            f"{total_test_images}\n"
        )

        file.write(
            f"Evaluated images: "
            f"{evaluated_images}\n"
        )

        file.write(
            f"Successful images: "
            f"{successful}\n"
        )

        file.write(
            f"Failed images: "
            f"{failed}\n"
        )

        file.write(
            f"Elapsed time: "
            f"{elapsed:.2f} s\n"
        )

        file.write(
            f"Throughput: "
            f"{throughput:.4f} images/s\n"
        )

        file.write(
            "\nEvaluation protocol:\n"
        )

        file.write(
            "- Dataset: CelebAMask-HQ\n"
        )

        file.write(
            "- Split: Official CelebA test partition\n"
        )

        file.write(
            "- Input resolution: 512 x 512\n"
        )

        file.write(
            "- Ground-truth resolution: 512 x 512\n"
        )

        file.write(
            "- Initialization: Full aligned-image box\n"
        )

        file.write(
            "- Number of semantic classes: 19\n"
        )

        file.write(
            "- Aggregation: Dataset-level confusion matrix\n"
        )

        file.write(
            "\nOverall metrics:\n"
        )

        file.write(
            f"Pixel accuracy: "
            f"{metrics['pixel_accuracy']:.6f}\n"
        )

        file.write(
            f"All-class mIoU: "
            f"{metrics['all_class_miou']:.6f}\n"
        )

        file.write(
            f"Foreground mIoU: "
            f"{metrics['foreground_miou']:.6f}\n"
        )

        file.write(
            f"All-class mean Dice: "
            f"{metrics['all_class_mdice']:.6f}\n"
        )

        file.write(
            f"Foreground mean Dice: "
            f"{metrics['foreground_mdice']:.6f}\n"
        )

        file.write(
            "\nPer-class metrics:\n"
        )

        for class_id, class_name in enumerate(
            CLASS_NAMES
        ):
            gt_pixels = metrics[
                "gt_pixels"
            ][
                class_id
            ]

            if gt_pixels == 0:
                continue

            file.write(
                f"{class_id:2d} "
                f"{class_name:10s} "
                f"IoU="
                f"{metrics['iou'][class_id]:.6f} "
                f"Dice="
                f"{metrics['dice'][class_id]:.6f}\n"
            )

    return output_path


def main():
    validate_required_paths()

    test_indices = (
        load_test_hq_indices()
    )

    if len(
        test_indices
    ) != 2824:
        raise RuntimeError(
            f"Expected 2824 test images, "
            f"found {len(test_indices)}."
        )

    selected_indices = (
        test_indices
        if EVALUATION_LIMIT is None
        else test_indices[
            :EVALUATION_LIMIT
        ]
    )

    print(
        f"Full test split size: "
        f"{len(test_indices)}"
    )

    print(
        f"Evaluation images: "
        f"{len(selected_indices)}"
    )

    print(
        f"First 10 test HQ indices: "
        f"{selected_indices[:10]}"
    )

    confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        dtype=np.int64,
    )

    face_regions = FaceRegions(
        device="cpu",
        verbose=False,
    )

    successful = 0
    failed = 0

    start_time = (
        time.perf_counter()
    )

    for position, image_id in enumerate(
        selected_indices,
        start=1,
    ):
        try:
            image = load_image(
                image_id
            )

            gt_map = (
                build_gt_label_map(
                    image_id
                )
            )

            height, width = (
                image.shape[:2]
            )

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

            output = (
                face_regions.predict(
                    image,
                    boxes=full_image_box,
                )
            )

            result = output[
                "result"
            ]

            if result.seg_map is None:
                raise RuntimeError(
                    "Segmentation map is None."
                )

            pred_map = np.asarray(
                result.seg_map,
                dtype=np.uint8,
            )

            pred_map = np.squeeze(
                pred_map
            )

            if pred_map.ndim != 2:
                raise ValueError(
                    f"Unexpected prediction shape: "
                    f"{pred_map.shape}"
                )

            if (
                pred_map.shape
                != gt_map.shape
            ):
                raise ValueError(
                    f"Prediction shape mismatch: "
                    f"{pred_map.shape} "
                    f"!= {gt_map.shape}"
                )

            update_confusion_matrix(
                confusion_matrix,
                gt_map,
                pred_map,
            )

            successful += 1

            print(
                f"[{position:04d}/"
                f"{len(selected_indices):04d}] "
                f"HQ {image_id}: OK"
            )

        except Exception as error:
            failed += 1

            print(
                f"[{position:04d}/"
                f"{len(selected_indices):04d}] "
                f"HQ {image_id}: FAILED - "
                f"{error}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    metrics = calculate_metrics(
        confusion_matrix
    )

    metrics_path = (
        save_class_metrics(
            metrics
        )
    )

    confusion_path = (
        save_confusion_matrix(
            confusion_matrix
        )
    )

    summary_path = (
        save_summary(
            metrics=metrics,
            total_test_images=len(
                test_indices
            ),
            evaluated_images=len(
                selected_indices
            ),
            successful=successful,
            failed=failed,
            elapsed=elapsed,
        )
    )

    print()
    print(
        "Evaluation summary:"
    )

    print(
        f"Successful images: "
        f"{successful}"
    )

    print(
        f"Failed images: "
        f"{failed}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed:.2f} s"
    )

    if elapsed > 0:
        print(
            f"Throughput: "
            f"{successful / elapsed:.4f} "
            f"images/s"
        )

    print(
        f"Pixel accuracy: "
        f"{metrics['pixel_accuracy']:.4f}"
    )

    print(
        f"All-class mIoU: "
        f"{metrics['all_class_miou']:.4f}"
    )

    print(
        f"Foreground mIoU: "
        f"{metrics['foreground_miou']:.4f}"
    )

    print(
        f"All-class mean Dice: "
        f"{metrics['all_class_mdice']:.4f}"
    )

    print(
        f"Foreground mean Dice: "
        f"{metrics['foreground_mdice']:.4f}"
    )

    print()
    print(
        "Per-class metrics:"
    )

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        gt_pixels = metrics[
            "gt_pixels"
        ][
            class_id
        ]

        if gt_pixels == 0:
            continue

        print(
            f"{class_id:2d} "
            f"{class_name:10s} "
            f"IoU="
            f"{metrics['iou'][class_id]:.4f} "
            f"Dice="
            f"{metrics['dice'][class_id]:.4f}"
        )

    print()
    print(
        f"Saved class metrics: "
        f"{metrics_path}"
    )

    print(
        f"Saved confusion matrix: "
        f"{confusion_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()