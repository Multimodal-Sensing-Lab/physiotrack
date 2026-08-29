from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat

from wider_face_eval import (
    load_predictions,
    normalize_scores,
    evaluate_setting,
)


def main():
    project_dir = Path(__file__).resolve().parent
    project_root = project_dir.parents[2]

    wider_root = (
        project_root
        / "datasets"
        / "WIDER_FACE"
    )

    pred_dir = (
        project_dir
        / "results"
        / "predictions"
    )

    gt_dir = (
        wider_root
        / "eval_tools"
        / "eval_tools"
        / "ground_truth"
    )

    results_dir = (
        project_dir
        / "results"
    )

    figures_dir = (
        results_dir
        / "figures"
    )

    inference_summary_path = (
        results_dir
        / "wider_face_inference_summary.json"
    )

    summary_path = (
        results_dir
        / "wider_face_summary.txt"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not inference_summary_path.is_file():
        raise FileNotFoundError(
            "Inference summary was not found. "
            "Run wider_face_inference.py first: "
            f"{inference_summary_path}"
        )

    with open(
        inference_summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        inference_summary = json.load(
            file
        )

    val_data = loadmat(
        gt_dir
        / "wider_face_val.mat"
    )

    event_list = (
        val_data["event_list"]
    )

    file_list = (
        val_data["file_list"]
    )

    face_bbx_list = (
        val_data["face_bbx_list"]
    )

    print("Reading predictions...")

    predictions = load_predictions(
        pred_dir,
        event_list,
        file_list,
    )

    predictions = normalize_scores(
        predictions
    )

    settings = {
        "Easy": "wider_easy_val.mat",
        "Medium": "wider_medium_val.mat",
        "Hard": "wider_hard_val.mat",
    }

    results = {}

    for name, filename in settings.items():
        setting_data = loadmat(
            gt_dir
            / filename
        )

        (
            ap,
            precision,
            recall,
            total_faces,
        ) = evaluate_setting(
            predictions,
            face_bbx_list,
            setting_data["gt_list"],
        )

        results[name] = {
            "ap": float(ap),
            "precision": precision,
            "recall": recall,
            "total_faces": int(
                total_faces
            ),
        }

        print(
            f"{name}: "
            f"AP={ap:.6f}, "
            f"faces={total_faces}"
        )

    plt.figure(
        figsize=(8, 6)
    )

    for name, values in results.items():
        plt.plot(
            values["recall"],
            values["precision"],
            label=(
                f"{name} "
                f"(AP={values['ap']:.4f})"
            ),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "WIDER FACE Validation "
        "Precision-Recall"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    figure_path = (
        figures_dir
        / "wider_face_precision_recall.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    table_rows = []

    for name, values in results.items():
        table_rows.append(
            {
                "Difficulty": name,
                "Average Precision": round(
                    values["ap"],
                    6,
                ),
                "Evaluated Faces": values[
                    "total_faces"
                ],
            }
        )

    table = pd.DataFrame(
        table_rows
    )

    table_csv_path = (
        results_dir
        / "wider_face_thesis_table.csv"
    )

    table_md_path = (
        results_dir
        / "wider_face_thesis_table.md"
    )

    table.to_csv(
        table_csv_path,
        index=False,
    )

    table.to_markdown(
        table_md_path,
        index=False,
    )

    summary_lines = [
        "WIDER FACE Face Detection Validation",
        "",
        "Dataset:",
        inference_summary["dataset"],
        "",
        "Validation images:",
        str(
            inference_summary[
                "validation_images"
            ]
        ),
        "",
        "Evaluation protocol:",
        (
            "Official WIDER FACE "
            "Easy / Medium / Hard splits"
        ),
        "IoU threshold: 0.5",
        (
            "Confidence threshold used "
            f"for inference: "
            f"{inference_summary['confidence_threshold']}"
        ),
        (
            "max_det: "
            f"{inference_summary['max_det']}"
        ),
        (
            "Device: "
            f"{inference_summary['device']}"
        ),
        "",
        "Results:",
        (
            "Easy AP: "
            f"{results['Easy']['ap']:.6f}"
        ),
        (
            "Medium AP: "
            f"{results['Medium']['ap']:.6f}"
        ),
        (
            "Hard AP: "
            f"{results['Hard']['ap']:.6f}"
        ),
        "",
        "Evaluated faces:",
        (
            "Easy: "
            f"{results['Easy']['total_faces']}"
        ),
        (
            "Medium: "
            f"{results['Medium']['total_faces']}"
        ),
        (
            "Hard: "
            f"{results['Hard']['total_faces']}"
        ),
        "",
        "Inference summary:",
        (
            "Images processed: "
            f"{inference_summary['images_processed']}"
        ),
        (
            "Failed images: "
            f"{inference_summary['failed_images']}"
        ),
        (
            "Prediction files: "
            f"{inference_summary['prediction_files']}"
        ),
        (
            "Total detections: "
            f"{inference_summary['total_detections']}"
        ),
        (
            "Maximum detections in one image: "
            f"{inference_summary['maximum_detections_in_one_image']}"
        ),
        (
            "Image with maximum detections: "
            f"{inference_summary['image_with_maximum_detections']}"
        ),
        (
            "Inference runtime: "
            f"{inference_summary['runtime_minutes']:.2f} minutes"
        ),
        "",
        "Generated outputs:",
        (
            "Prediction directory: "
            "results/predictions"
        ),
        (
            "Thesis table CSV: "
            "results/wider_face_thesis_table.csv"
        ),
        (
            "Thesis table Markdown: "
            "results/wider_face_thesis_table.md"
        ),
        (
            "Precision-recall figure: "
            "results/figures/"
            "wider_face_precision_recall.png"
        ),
    ]

    summary_path.write_text(
        "\n".join(summary_lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== WIDER FACE Thesis Table ==="
    )

    print(
        table.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved summary: {summary_path}"
    )

    print(
        f"Saved figure: {figure_path}"
    )

    print(
        f"Saved table CSV: {table_csv_path}"
    )

    print(
        f"Saved table Markdown: {table_md_path}"
    )

    if inference_summary_path.exists():
        inference_summary_path.unlink()

        print(
            "Removed temporary inference summary:",
            inference_summary_path,
        )


if __name__ == "__main__":
    main()