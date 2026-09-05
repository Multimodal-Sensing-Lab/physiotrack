WIDER FACE Face Detection Validation
====================================

Overview
--------
This directory contains the reproducible Face Detection validation package for
PhysioTrack.

The package provides two complementary evidence layers:

1. Scientific benchmark validation on the official WIDER FACE validation split.
2. Isolated PhysioTrack Face Detection component execution using the real
   project implementation with unrelated face-analysis components disabled.

The scientific benchmark measures predictive accuracy against WIDER FACE
ground-truth annotations. The isolated component execution verifies that the
actual PhysioTrack Face Detection path runs correctly and exports real numerical
outputs. The isolated component execution is not an accuracy benchmark and must
not be interpreted as a substitute for WIDER FACE Average Precision (AP).

Dataset
-------
Dataset:
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

Extract the downloaded packages so that the relevant structure is:

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

All validation paths are project-relative. The dataset is treated as read-only.

Validation scripts
------------------
wider_face_inference.py
    Runs Face Detection on all WIDER FACE validation images and writes the
    WIDER-format prediction files and inference summary.

wider_face_eval.py
    Evaluates the saved predictions against the official WIDER FACE Easy,
    Medium, and Hard validation ground truth and writes the complete
    precision-recall result table.

wider_face_plot.py
    Reads the final benchmark results and inference summary and generates the
    quantitative summary, thesis tables, and precision-recall figure.

wider_face_qualitative.py
    Generates deterministic qualitative examples from the saved predictions and
    official WIDER FACE ground truth. It does not rerun the detector and does
    not modify the benchmark AP calculation.

face_detection_component_test.py
    Runs the real PhysioTrack Face Detection implementation through the
    FaceAnalysis project path with unrelated face-analysis components disabled.
    It processes the full WIDER FACE validation image set and exports numerical
    detector outputs to a structured table.

Scientific benchmark protocol
-----------------------------
Validation split:
WIDER FACE validation set

Validation images:
3226

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

Evaluator score thresholds:
1000

Running the scientific benchmark
--------------------------------
Activate the PhysioTrack environment:

conda activate PhysioTrack-Thesis

Open:

physiotrack/validation/face_detection

Run:

python wider_face_inference.py
python wider_face_eval.py
python wider_face_plot.py
python wider_face_qualitative.py

The scripts should be run in the order shown above because the later stages
consume outputs created by earlier stages.

Validated quantitative results
------------------------------
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

Total detections:
193618

Maximum detections in one image:
3134

Image with maximum detections:
0--Parade/0_Parade_Parade_0_275.jpg

Runtime varies between machines and is not a scientific metric.

The observed AP ordering:

Easy > Medium > Hard

is consistent with the intended difficulty structure of the WIDER FACE
benchmark.

Qualitative benchmark
---------------------
The qualitative stage produces ten deterministic examples:

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

The examples include clear detections, scale variation, larger group scenes,
and a readable challenging case. The challenging example is retained to show a
real limitation rather than presenting only successful cases.

Annotated images are stored in:

results/qualitative/annotated_images

The qualitative selection table is stored in:

results/qualitative/wider_face_qualitative_selection.csv

The combined qualitative figure is stored in:

results/figures/wider_face_qualitative_examples.png

Qualitative display protocol
----------------------------
Matching IoU:
0.50

Raw display threshold:
0.25

The raw display threshold is used only for visualization readability. It does
not define or modify the benchmark AP calculation.

Per-image precision, recall, F1, matched IoU, and confidence values shown in the
qualitative panels are example-level display measurements. They must not be
interpreted as replacements for the official full-dataset WIDER FACE AP metrics.

Isolated PhysioTrack Face Detection execution
---------------------------------------------
Run:

python face_detection_component_test.py

Purpose:
Verify software-level operation of the real PhysioTrack Face Detection path and
preserve its numerical outputs in a reproducible structured table.

The script:
- uses the real PhysioTrack FaceAnalysis detection path
- processes all 3226 WIDER FACE validation images
- disables unrelated optional face-analysis components
- uses CPU execution
- uses confidence threshold 0.001
- uses max_det 10000
- preserves raw detector outputs
- records explicit status and failure information
- uses project-relative paths
- leaves the source dataset unchanged
- uses staged output generation and validated replacement

The following unrelated face-analysis components are disabled:
- tracking
- head pose
- landmarks
- quality
- eyes
- blink
- gaze
- gaze estimation
- mouth
- mouth motion
- emotion
- face regions
- temporal analysis

Isolated component outputs
--------------------------
Main numerical result table:

results/component_execution/face_detection_component_results.csv

Execution summary:

results/component_execution/face_detection_component_summary.json

The result table contains:
- input_identifier
- input_type
- image_index
- detection_index
- detections_in_image
- image_width
- image_height
- person_id
- class_id
- class_name
- box_x1
- box_y1
- box_x2
- box_y2
- box_width
- box_height
- box_area
- confidence
- status
- failure_reason

For valid detections:

box_width = box_x2 - box_x1

box_height = box_y2 - box_y1

box_area = box_width × box_height

Because tracking is disabled, person_id is expected to be empty.

Validated isolated execution results
------------------------------------
Validation images:
3226

Processed images:
3226

Failed images:
0

Read failures:
0

Prediction failures:
0

Images with detections:
3226

Images without detections:
0

Total detections:
193618

Result rows:
193618

Invalid-box diagnostic detections:
18

Overall execution status:
PASS

The result table contains 3226 unique input identifiers corresponding to the
3226 validation images. Image indices cover the complete range 1 to 3226.
Within each image, detection indices are unique and consecutive, and the
reported detections_in_image value agrees with the number of rows belonging to
that image.

The 18 invalid-box diagnostic rows represent approximately 0.0093% of all
193618 detector outputs. They are retained explicitly with status:

DETECTED_INVALID_BOX

The raw detector coordinates are preserved without silent correction or
removal. Derived width, height, and area values are intentionally left empty
for these diagnostic rows because the raw box geometry has a non-positive
dimension.

These rows provide auditable diagnostic evidence and do not replace or modify
the scientific WIDER FACE benchmark results.

Safe rerun design
-----------------
The validation package follows the safe-rerun pattern:

preflight
-> temporary/staging generation
-> validation of newly generated outputs
-> replacement of script-owned final outputs

The preflight stage occurs before previously valid final outputs are replaced.

If required dataset files, annotations, prediction dependencies, or other
critical inputs are unavailable or invalid, the affected script stops without
deleting previously valid evidence.

Output ownership
----------------
wider_face_inference.py
    Owns:
    - results/predictions
    - results/wider_face_inference_summary.json

wider_face_eval.py
    Owns:
    - results/wider_face_results.csv

wider_face_plot.py
    Owns:
    - results/wider_face_summary.txt
    - results/wider_face_thesis_table.csv
    - results/wider_face_thesis_table.md
    - results/figures/wider_face_precision_recall.png

wider_face_qualitative.py
    Owns:
    - results/qualitative/annotated_images
    - results/qualitative/wider_face_qualitative_selection.csv
    - results/figures/wider_face_qualitative_examples.png

face_detection_component_test.py
    Owns:
    - results/component_execution/face_detection_component_results.csv
    - results/component_execution/face_detection_component_summary.json

No script should delete another script's final outputs.

Expected result structure
-------------------------
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
├── component_execution/
│   ├── face_detection_component_results.csv
│   └── face_detection_component_summary.json
├── wider_face_inference_summary.json
├── wider_face_results.csv
├── wider_face_summary.txt
├── wider_face_thesis_table.csv
├── wider_face_thesis_table.md
└── figures/
    ├── wider_face_precision_recall.png
    └── wider_face_qualitative_examples.png

Output descriptions
-------------------
results/predictions
    WIDER FACE prediction files generated from all validation images.

results/wider_face_inference_summary.json
    Inference metadata, detector settings, image accounting, detection counts,
    and runtime information.

results/wider_face_results.csv
    Complete saved precision-recall data for the Easy, Medium, and Hard subsets.

results/wider_face_summary.txt
    Text record of the benchmark configuration and final quantitative results.

results/wider_face_thesis_table.csv
    Compact CSV table of final AP values and evaluated-face counts.

results/wider_face_thesis_table.md
    Markdown version of the same benchmark table.

results/figures/wider_face_precision_recall.png
    Precision-recall curves for the three WIDER FACE validation subsets.

results/qualitative/annotated_images
    Ten annotated benchmark examples.

results/qualitative/wider_face_qualitative_selection.csv
    Machine-readable record of the selected qualitative examples and their
    display-level metrics.

results/figures/wider_face_qualitative_examples.png
    Combined qualitative figure.

results/component_execution/face_detection_component_results.csv
    Numerical outputs from isolated execution of the real PhysioTrack Face
    Detection component.

results/component_execution/face_detection_component_summary.json
    Execution metadata and accounting summary for the isolated component test.

Complete reproduction procedure
-------------------------------
1. Install the PhysioTrack project environment and dependencies.
2. Download the required official WIDER FACE packages.
3. Extract them into datasets/WIDER_FACE using the documented structure.
4. Activate:

   conda activate PhysioTrack-Thesis

5. Open:

   physiotrack/validation/face_detection

6. Run the scientific benchmark:

   python wider_face_inference.py
   python wider_face_eval.py
   python wider_face_plot.py
   python wider_face_qualitative.py

7. Verify the quantitative and qualitative outputs.

8. Run the isolated Face Detection component execution:

   python face_detection_component_test.py

9. Verify the isolated component CSV and JSON summary.

10. Confirm image accounting, failure accounting, output schemas, and benchmark
    metrics before using the package as reproducibility evidence.

Interpretation and limitations
------------------------------
The scientific WIDER FACE benchmark is the evidence used to characterize Face
Detection predictive accuracy.

The isolated component execution is software-level evidence that the actual
PhysioTrack Face Detection implementation operates independently and exports
real numerical detector outputs.

Integration or isolated execution PASS must not be interpreted as scientific
predictive accuracy.

Qualitative examples are illustrative and intentionally limited in number.
They do not define dataset-level accuracy.

Runtime is environment-dependent and is not used as an accuracy measure.

Final files to preserve
-----------------------
Preserve:
- wider_face_inference.py
- wider_face_eval.py
- wider_face_plot.py
- wider_face_qualitative.py
- face_detection_component_test.py
- README_WIDER_FACE.txt
- results/predictions
- results/wider_face_inference_summary.json
- results/wider_face_results.csv
- results/wider_face_summary.txt
- results/wider_face_thesis_table.csv
- results/wider_face_thesis_table.md
- results/figures/wider_face_precision_recall.png
- results/qualitative/annotated_images
- results/qualitative/wider_face_qualitative_selection.csv
- results/figures/wider_face_qualitative_examples.png
- results/component_execution/face_detection_component_results.csv
- results/component_execution/face_detection_component_summary.json

Together, these files preserve the scientific benchmark evidence, qualitative
evidence, and isolated PhysioTrack Face Detection execution record required for
reproducibility.
