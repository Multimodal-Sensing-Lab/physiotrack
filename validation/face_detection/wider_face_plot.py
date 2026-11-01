from pathlib import Path
import json
import shutil
import tempfile

import matplotlib.pyplot as plt
import pandas as pd


def main():
    project_dir = Path(__file__).resolve().parent

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

    benchmark_results_path = (
        results_dir
        / "wider_face_results.csv"
    )

    summary_path = (
        results_dir
        / "wider_face_summary.txt"
    )

    table_csv_path = (
        results_dir
        / "wider_face_thesis_table.csv"
    )

    table_md_path = (
        results_dir
        / "wider_face_thesis_table.md"
    )

    figure_path = (
        figures_dir
        / "wider_face_precision_recall.png"
    )

    required_paths = [
        inference_summary_path,
        benchmark_results_path,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                "Required quantitative input was not found. "
                "Run wider_face_inference.py and "
                "wider_face_eval.py first: "
                f"{path}"
            )

    final_paths = [
        summary_path,
        table_csv_path,
        table_md_path,
        figure_path,
    ]

    for path in final_paths:
        if path.exists() and not path.is_file():
            raise RuntimeError(
                "Owned output path exists but is not a file: "
                f"{path}"
            )

    with open(
        inference_summary_path,
        "r",
        encoding="utf-8",
    ) as file:
        inference_summary = json.load(
            file
        )

    required_summary_keys = {
        "dataset",
        "validation_images",
        "images_processed",
        "failed_images",
        "prediction_files",
        "total_detections",
        "maximum_detections_in_one_image",
        "image_with_maximum_detections",
        "confidence_threshold",
        "max_det",
        "device",
        "runtime_minutes",
    }

    if not required_summary_keys.issubset(
        inference_summary
    ):
        missing = sorted(
            required_summary_keys
            - set(inference_summary)
        )
        raise RuntimeError(
            "Missing fields in inference summary: "
            + ", ".join(missing)
        )

    benchmark = pd.read_csv(
        benchmark_results_path
    )

    required_columns = {
        "Difficulty",
        "Threshold Index",
        "Score Threshold",
        "Precision",
        "Recall",
        "Average Precision",
        "Evaluated Faces",
    }

    if not required_columns.issubset(
        benchmark.columns
    ):
        raise RuntimeError(
            "Unexpected columns in benchmark results: "
            f"{benchmark.columns.tolist()}"
        )

    settings = [
        "Easy",
        "Medium",
        "Hard",
    ]

    results = {}

    for name in settings:
        subset = (
            benchmark[
                benchmark["Difficulty"]
                == name
            ]
            .sort_values(
                "Threshold Index"
            )
            .reset_index(
                drop=True
            )
        )

        if len(subset) != 1000:
            raise RuntimeError(
                f"Unexpected number of "
                f"{name} threshold rows: "
                f"{len(subset)}"
            )

        ap_values = (
            subset[
                "Average Precision"
            ]
            .dropna()
            .unique()
        )

        face_values = (
            subset[
                "Evaluated Faces"
            ]
            .dropna()
            .unique()
        )

        if len(ap_values) != 1:
            raise RuntimeError(
                f"Inconsistent AP values "
                f"for {name}."
            )

        if len(face_values) != 1:
            raise RuntimeError(
                f"Inconsistent evaluated-face "
                f"counts for {name}."
            )

        results[name] = {
            "ap": float(
                ap_values[0]
            ),
            "precision": subset[
                "Precision"
            ].to_numpy(),
            "recall": subset[
                "Recall"
            ].to_numpy(),
            "total_faces": int(
                face_values[0]
            ),
        }

        print(
            f"{name}: "
            f"AP={results[name]['ap']:.6f}, "
            f"faces={results[name]['total_faces']}"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("WIDER FACE plot preflight: PASS")

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".wider_face_plot_",
            dir=results_dir,
        )
    )

    staged_summary_path = (
        staging_root
        / "wider_face_summary.txt"
    )

    staged_table_csv_path = (
        staging_root
        / "wider_face_thesis_table.csv"
    )

    staged_table_md_path = (
        staging_root
        / "wider_face_thesis_table.md"
    )

    staged_figure_path = (
        staging_root
        / "wider_face_precision_recall.png"
    )

    staged_paths = [
        staged_summary_path,
        staged_table_csv_path,
        staged_table_md_path,
        staged_figure_path,
    ]

    try:
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

        plt.savefig(
            staged_figure_path,
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

        table.to_csv(
            staged_table_csv_path,
            index=False,
        )

        table.to_markdown(
            staged_table_md_path,
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
                "Benchmark results: "
                "results/wider_face_results.csv"
            ),
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

        staged_summary_path.write_text(
            "\n".join(summary_lines)
            + "\n",
            encoding="utf-8",
        )

        for path in staged_paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(
                    "Staged plot output is missing or empty: "
                    f"{path}"
                )

        validated_table = pd.read_csv(
            staged_table_csv_path
        )

        if len(validated_table) != len(settings):
            raise RuntimeError(
                "Unexpected number of rows in staged thesis table: "
                f"{len(validated_table)}"
            )

        if list(validated_table["Difficulty"]) != settings:
            raise RuntimeError(
                "Unexpected difficulty ordering in staged thesis table."
            )

        plt.imread(
            staged_figure_path
        )

        backup_paths = [
            staging_root / f"previous_{path.name}"
            for path in final_paths
        ]

        committed = []

        try:
            for final_path, backup_path in zip(
                final_paths,
                backup_paths,
            ):
                if final_path.exists():
                    final_path.replace(
                        backup_path
                    )

            for staged_path, final_path in zip(
                staged_paths,
                final_paths,
            ):
                staged_path.replace(
                    final_path
                )
                committed.append(
                    final_path
                )

        except Exception:
            for final_path in committed:
                if final_path.exists():
                    final_path.unlink()

            for final_path, backup_path in zip(
                final_paths,
                backup_paths,
            ):
                if backup_path.exists():
                    backup_path.replace(
                        final_path
                    )

            raise

    finally:
        plt.close("all")

        if staging_root.exists():
            shutil.rmtree(
                staging_root
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

if __name__ == "__main__":
    main()
