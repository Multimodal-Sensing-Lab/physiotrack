WIDER FACE Face Detection Validation
====================================

This directory contains the scripts used to evaluate the PhysioTrack face detector on the WIDER FACE validation split and to generate reproducible qualitative benchmark examples from the final quantitative predictions.

Dataset
-------
WIDER FACE

Official page:
http://shuoyang1213.me/WIDERFACE/

Required downloads:
- WIDER_val.zip
- wider_face_split.zip
- eval_tools.zip

Dataset setup
-------------
Create the following directory at the project root:

datasets/WIDER_FACE

Extract the downloaded archives so that the relevant structure is:

datasets/
└── WIDER_FACE/
    ├── WIDER_val/
    │   └── images/
    ├── wider_face_split/
    └── eval_tools/
        └── eval_tools/
            └── ground_truth/
                ├── wider_face_val.mat
                ├── wider_easy_val.mat
                ├── wider_medium_val.mat
                └── wider_hard_val.mat

The dataset location is resolved relative to the project structure using the documented directory layout.

Validation files
----------------
wider_face_inference.py
    Runs face detection on all images in the WIDER FACE validation split and writes the prediction files and inference summary required by the later validation stages.

wider_face_eval.py
    Evaluates the generated predictions using the official WIDER FACE Easy, Medium, and Hard validation ground truth and writes the detailed benchmark results to wider_face_results.csv.

wider_face_plot.py
    Reads the saved benchmark results and generates the final precision-recall figure, thesis result tables, and validation summary.

wider_face_qualitative.py
    Generates reproducible qualitative benchmark examples from the real saved WIDER FACE predictions and the official WIDER FACE ground truth. It does not rerun the detector and does not modify the final quantitative predictions or AP results.

Validation protocol
-------------------
Validation split:
WIDER FACE validation set

Evaluation subsets:
- Easy
- Medium
- Hard

IoU threshold:
0.5

Inference confidence threshold:
0.001

Maximum detections per image:
10000

Device:
CPU

Number of score thresholds used by the evaluator:
1000

Running the validation
----------------------
Activate the PhysioTrack environment:

conda activate PhysioTrack-Thesis

Open the face detection validation directory:

physiotrack/validation/face_detection

Run the quantitative validation scripts in this order:

python wider_face_inference.py

python wider_face_eval.py

python wider_face_plot.py

After the quantitative validation has completed successfully, generate the qualitative benchmark outputs:

python wider_face_qualitative.py

The inference stage processes the complete WIDER FACE validation split and stores the generated prediction files in:

validation/face_detection/results/predictions

The evaluator then computes Average Precision for the Easy, Medium, and Hard subsets and stores the detailed precision-recall results in:

validation/face_detection/results/wider_face_results.csv

The plotting script reads the saved benchmark results and inference summary to create the final quantitative validation artifacts.

The qualitative script reads the existing prediction files, official WIDER FACE ground truth, and the final quantitative result table. It verifies that the expected validation image count, prediction file count, and final AP values are present before generating qualitative outputs.

Expected quantitative results
-----------------------------
The validated configuration produces:

Easy AP:
0.958883

Medium AP:
0.948828

Hard AP:
0.871830

Evaluated faces:

Easy:
7211

Medium:
13319

Hard:
31958

Validation images:
3226

Failed images:
0

Prediction files:
3226

Maximum detections in one image:
3134

Minor differences in runtime are expected between machines. The scientific results should remain unchanged when the same code, model, dataset, and evaluation protocol are used.

Qualitative benchmark outputs
-----------------------------
The qualitative stage produces ten deterministic WIDER FACE examples that cover clear detections, scale variation, a readable Hard example, a readable failure case, and larger group scenes.

The ten output roles are:

- easy_clear_01
- easy_clear_02
- medium_scale_01
- medium_scale_02
- hard_readable_01
- hard_challenge_01
- group_medium_01
- group_medium_02
- group_hard_01
- group_hard_02

The Hard challenge example is intentionally selected to show a real, readable limitation rather than only successful cases. The group examples provide additional evidence on images containing larger numbers of faces.

The annotated images are stored in:

validation/face_detection/results/qualitative/annotated_images

The qualitative selection table is stored in:

validation/face_detection/results/qualitative/wider_face_qualitative_selection.csv

The combined qualitative figure is stored in:

validation/face_detection/results/figures/wider_face_qualitative_examples.png

Each annotated image contains:
- WIDER FACE benchmark image
- true-positive detections
- false-positive detections when present
- false-negative eligible ground-truth boxes when present
- ignored/non-subset predictions when present
- per-image display metrics
- matching IoU
- raw visualization threshold
- full benchmark AP for the corresponding WIDER FACE subset
- a legend explaining the visualization

Qualitative display protocol
----------------------------
Matching IoU:
0.50

Raw display threshold:
0.25

The raw display threshold is used only to keep the qualitative figures readable. It does not define or modify the benchmark AP calculation.

The full benchmark AP displayed in each qualitative panel is read from:

validation/face_detection/results/wider_face_thesis_table.csv

The qualitative script verifies these values against the final results before producing the figures.

Per-image precision, recall, F1, matched IoU, and confidence values shown in the qualitative panels are display-level measurements for the selected example at the raw display threshold. They must not be interpreted as replacements for the official full-dataset WIDER FACE AP metrics.

Qualitative output cleanup
--------------------------
Before generating a new qualitative result set, wider_face_qualitative.py removes only its own existing qualitative outputs:

- files inside results/qualitative/annotated_images
- results/qualitative/wider_face_qualitative_selection.csv
- results/figures/wider_face_qualitative_examples.png

It then regenerates the complete qualitative output set.

The script does not delete or modify:

results/predictions

or any other quantitative validation artifact.

This separation ensures that repeated qualitative runs do not leave obsolete images while preserving the final quantitative predictions required for reproducibility.

Generated results
-----------------
After a complete quantitative and qualitative run, the main outputs are stored under:

validation/face_detection/results

Expected contents:

results/
├── predictions/
├── qualitative/
│   ├── annotated_images/
│   │   ├── easy_clear_01.png
│   │   ├── easy_clear_02.png
│   │   ├── medium_scale_01.png
│   │   ├── medium_scale_02.png
│   │   ├── hard_readable_01.png
│   │   ├── hard_challenge_01.png
│   │   ├── group_medium_01.png
│   │   ├── group_medium_02.png
│   │   ├── group_hard_01.png
│   │   └── group_hard_02.png
│   └── wider_face_qualitative_selection.csv
├── wider_face_inference_summary.json
├── wider_face_results.csv
├── wider_face_summary.txt
├── wider_face_thesis_table.csv
├── wider_face_thesis_table.md
└── figures/
    ├── wider_face_precision_recall.png
    └── wider_face_qualitative_examples.png

wider_face_inference_summary.json
    Inference metadata including image counts, failure counts, detector settings, and runtime information.

wider_face_results.csv
    Detailed benchmark results containing the saved precision-recall data for the Easy, Medium, and Hard subsets.

wider_face_summary.txt
    Detailed record of the quantitative validation setup, inference statistics, and final benchmark results.

wider_face_thesis_table.csv
    CSV version of the final Easy, Medium, and Hard quantitative results.

wider_face_thesis_table.md
    Markdown version of the same quantitative results table.

figures/wider_face_precision_recall.png
    Precision-recall curves for the three WIDER FACE validation subsets.

predictions/
    Prediction files generated from the WIDER FACE validation images and consumed by the evaluator and qualitative script.

qualitative/annotated_images/
    Ten annotated WIDER FACE benchmark examples showing detector outputs and per-image display metrics.

qualitative/wider_face_qualitative_selection.csv
    Machine-readable record of the selected qualitative examples and their per-image display metrics.

figures/wider_face_qualitative_examples.png
    Combined overview figure containing the ten qualitative examples.

Reproducing the validation
--------------------------
A complete reproduction consists of:

1. Downloading the three required WIDER FACE packages.
2. Extracting them into datasets/WIDER_FACE using the directory structure shown above.
3. Running wider_face_inference.py.
4. Running wider_face_eval.py.
5. Running wider_face_plot.py.
6. Confirming that the reported AP values and evaluated-face counts match the expected quantitative results above.
7. Running wider_face_qualitative.py.
8. Confirming that ten annotated qualitative images, the qualitative selection CSV, and the combined qualitative figure are generated.

If the reported quantitative metrics differ materially from the expected values, verify the dataset structure, model version, environment, and validation configuration before using the results.

If the qualitative script fails its preflight checks, verify that the quantitative validation has been completed successfully and that results/predictions and results/wider_face_thesis_table.csv contain the final outputs.
