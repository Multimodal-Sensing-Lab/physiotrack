WIDER FACE Face Detection Validation
====================================

Overview
--------
This directory documents the reproducible PhysioTrack Face Detection validation
package and its independent synthetic qualitative demonstration.

The scientific validation comprises two complementary evidence layers:

1. Scientific benchmark validation on the official WIDER FACE validation split.
2. Isolated PhysioTrack Face Detection component execution using the real
   project implementation with unrelated face-analysis components disabled.

The scientific benchmark measures predictive accuracy against WIDER FACE
ground-truth annotations. Isolated component execution verifies the software
path and exports numerical detector outputs; it does not measure benchmark
accuracy. A separate synthetic qualitative demonstration illustrates detector
behaviour on generated images without affecting WIDER FACE Average Precision
(AP) or other benchmark results.

Dataset
-------
Dataset:
WIDER FACE

Official page:
http://shuoyang1213.me/WIDERFACE/

Original dataset paper:
Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A Face Detection
Benchmark. Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 5525-5533.
DOI: https://doi.org/10.1109/CVPR.2016.596

Required downloads:
- WIDER_val.zip
- wider_face_split.zip
- eval_tools.zip

Dataset setup
-------------
The WIDER FACE dataset is located in the parent directory containing
the physiotrack-v2 repository:

datasets/WIDER_FACE

The extracted dataset directory structure is:

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

Validation paths are derived from the scripts' locations. The datasets and
physiotrack-v2 directories share a common parent directory. The dataset is
treated as read-only.

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
Conda environment with the PhysioTrack dependencies installed:

conda activate PhysioTrack-v2

Working directory:

physiotrack-v2/validation/face_detection

Scientific benchmark execution order:

python wider_face_inference.py
python wider_face_eval.py
python wider_face_plot.py
python wider_face_qualitative.py

The execution order reflects the dependency of later stages on earlier outputs.

Validated quantitative results
------------------------------
Easy AP:
0.959001

Medium AP:
0.949087

Hard AP:
0.872082

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
193617

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

The raw display threshold determines which predictions are visualized and
therefore affects the displayed per-image TP, FP, FN, precision, recall, and F1.
It does not define or modify the full-dataset benchmark AP calculation.

Per-image precision, recall, F1, matched IoU, and confidence values shown in the
qualitative panels are example-level display measurements, distinct from the
official full-dataset WIDER FACE AP metrics.

Isolated PhysioTrack Face Detection execution
---------------------------------------------
Execution command:

python face_detection_component_test.py

Purpose:
Software-level verification of the real PhysioTrack Face Detection path, with
numerical outputs retained in a reproducible structured table.

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
193617

Result rows:
193617

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
193617 detector outputs. They are retained explicitly with status:

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

For face_detection_synthetic_demo.py, every discovered image and video must
complete successfully. New annotated media and both summaries are generated in
a temporary staging directory, then checked for output completeness, media
readability, record consistency, and SHA-256 integrity. If any input or check
fails, the previously published synthetic_demo outputs remain unchanged.
Publication uses a backup and restores the previous directory if an ordinary
replacement error occurs. Other validation outputs are not replaced.

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

face_detection_synthetic_demo.py
    Owns:
    - results/synthetic_demo/images (only with image inputs)
    - results/synthetic_demo/videos (only with video inputs)
    - results/synthetic_demo/synthetic_demo_summary.json
    - results/synthetic_demo/synthetic_demo_summary.txt

Output ownership is restricted to each script's designated results. Existing
outputs belonging to other stages remain unchanged.

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
├── synthetic_demo/
│   ├── images/
│   ├── synthetic_demo_summary.json
│   └── synthetic_demo_summary.txt
├── wider_face_inference_summary.json
├── wider_face_results.csv
├── wider_face_summary.txt
├── wider_face_thesis_table.csv
├── wider_face_thesis_table.md
└── figures/
    ├── wider_face_precision_recall.png
    └── wider_face_qualitative_examples.png

The synthetic_demo tree above reflects image-only input. A videos/ directory
is generated only if video inputs are processed successfully.

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

results/synthetic_demo/images
    Annotated synthetic images generated when image inputs are present.

results/synthetic_demo/videos
    Annotated synthetic videos generated when video inputs are present.
    Media-type directories are omitted when no outputs of that type exist.

results/synthetic_demo/synthetic_demo_summary.json
    Synthetic media inventory, raw and displayed detection counts, processing
    status, and input-file hashes.

results/synthetic_demo/synthetic_demo_summary.txt
    Human-readable rendering of the same complete synthetic demonstration
    summary, including per-file records, detection totals, settings, source
    hashes, and software environment information.

Complete reproduction procedure
-------------------------------
1. Software environment: PhysioTrack project installation and dependencies.
2. Dataset packages: WIDER_val.zip, wider_face_split.zip, and eval_tools.zip.
3. Dataset location: ../datasets/WIDER_FACE relative to physiotrack-v2,
   with the directory structure specified under Dataset setup.
4. Conda environment activation:

   conda activate PhysioTrack-v2

5. Working directory:

   physiotrack-v2/validation/face_detection

6. Scientific benchmark execution order:

   python wider_face_inference.py
   python wider_face_eval.py
   python wider_face_plot.py
   python wider_face_qualitative.py

7. Scientific outputs: Easy, Medium, and Hard AP; precision-recall data;
   benchmark qualitative figures and selection table.

8. Isolated component execution:

   python face_detection_component_test.py

9. Isolated component outputs: detection-level CSV and execution-summary JSON.

10. Reproducibility records: image and detection accounting, processing-failure
    counts, output schemas, and benchmark metrics.

11. Independent synthetic demonstration, using media from
    validation/synthetic_media/face_detection_media/:

    python face_detection_synthetic_demo.py

12. Synthetic demonstration outputs: annotated media and
    results/synthetic_demo/synthetic_demo_summary.json.
    A matching synthetic_demo_summary.txt is also generated.

Interpretation and limitations
------------------------------
The scientific WIDER FACE benchmark is the evidence used to characterize Face
Detection predictive accuracy.

The isolated component execution is software-level evidence that the actual
PhysioTrack Face Detection implementation operates independently and exports
real numerical detector outputs.

An integration or isolated execution PASS indicates successful software
execution, not a measurement of scientific predictive accuracy.

Qualitative examples are illustrative and intentionally limited in number.
They do not define dataset-level accuracy.

Runtime is environment-dependent and is not used as an accuracy measure.

Reproducibility artifact inventory
---------------------------------
- wider_face_inference.py
- wider_face_eval.py
- wider_face_plot.py
- wider_face_qualitative.py
- face_detection_component_test.py
- face_detection_synthetic_demo.py
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
- validation/synthetic_media/face_detection_media/
- results/synthetic_demo/images (when image inputs exist)
- results/synthetic_demo/videos (when video inputs exist)
- results/synthetic_demo/synthetic_demo_summary.json
- results/synthetic_demo/synthetic_demo_summary.txt

Together, these artifacts document the scientific benchmark, benchmark
qualitative results, isolated component execution, and independent synthetic
qualitative demonstration.

Synthetic Qualitative Demonstration
-----------------------------------

Purpose
~~~~~~~
This demonstration illustrates the qualitative behaviour of the PhysioTrack Face
Detection component using independently generated synthetic media. It is
conducted separately from the WIDER FACE benchmark validation and serves as a
visual demonstration of the component's detection behaviour rather than a
quantitative accuracy evaluation.

Synthetic media
~~~~~~~~~~~~~~~
The demonstration uses ten synthetic 3D-style images generated using OpenAI
ChatGPT. The images depict artificial human faces and scenes representing
frontal views, profile views, varying head orientations, multiple faces,
partial occlusion, lighting variations, partially cropped faces, and a scene
without faces.

The synthetic input media are stored in:

validation/synthetic_media/face_detection_media/

These images are independent of the WIDER FACE dataset and are not used in its
scientific benchmark evaluation. The demonstration script discovers supported
images and videos added to this directory without changes to its source code.

Execution
~~~~~~~~~
The demonstration is executed using the independent script:

validation/face_detection/face_detection_synthetic_demo.py

The script automatically discovers supported images and videos in the input
directory and its subdirectories, without manually specified filenames. It
processes each image or video frame with the actual PhysioTrack Face detector
and FaceAnalysis pipeline, with unrelated face-analysis components disabled.
The media files are not modified.

Face detections are illustrated using bounding boxes and available model
confidence scores. The detector confidence threshold is 0.001, while a
separate visualization threshold of 0.25 determines which detections appear
in the annotated outputs. This display threshold does not alter detector
outputs or WIDER FACE benchmark metrics.

The qualitative outputs are stored in:

validation/face_detection/results/synthetic_demo/

Annotated images and videos are organized in separate subdirectories only
when that media type is present; empty media folders are not published. The
synthetic_demo_summary.json file records the processed media, detection counts,
processing status, and any reading or processing failures. Output generation
uses staging and validation before results are committed.

A second file, synthetic_demo_summary.txt, contains a human-readable rendering
of the same summary data. Both summary files are verified together with the
annotated outputs before the new synthetic_demo directory is published. An
incomplete run is rejected rather than published as a partial demonstration;
failed-run diagnostics are printed to the terminal while the previous accepted
results remain unchanged.

All script-controlled input and output locations are derived from the location
of face_detection_synthetic_demo.py, rather than a machine-specific absolute
path. The project checkout can therefore be relocated while retaining the
expected repository structure and the required installed dependencies and
model weights. For traceability, the summary includes input and output file
hashes, component-source and script hashes, detector settings, and relevant
software versions. Reproducing identical numerical outputs requires a
consistent model and software environment; path portability alone does not
guarantee identical outputs across different devices.

Interpretation
~~~~~~~~~~~~~~
The demonstration illustrates detection under the represented synthetic
conditions, including different facial orientations, multiple faces,
occlusion, lighting variations, and partially visible faces. A scene without
faces illustrates detector behaviour when no facial subject is present.

Bounding boxes indicate regions identified as faces by the model, and their
confidence scores describe individual detector outputs. These scores do not
measure overall detection accuracy. The ten-image demonstration produced 19
raw detections, of which 14 met the display threshold; all ten image files
were processed successfully.

The synthetic examples do not use WIDER FACE benchmark ground-truth annotations
and do not calculate Average Precision (AP), Precision, Recall, F1-score, or
other benchmark evaluation metrics. Their visual results do not establish
full-dataset predictive performance. The WIDER FACE methodology, validation
scripts, and quantitative results are independent of this demonstration.

Media provenance
~~~~~~~~~~~~~~~~
The synthetic input images were generated using OpenAI ChatGPT. The project's
media provenance and usage documentation is maintained in
VALIDATION_MEDIA_LICENSES.md. The demonstration summary records the SHA-256
hash of each processed source file for traceability.

The ten independently generated input images are named 01.png through 10.png
and are stored in validation/synthetic_media/face_detection_media/. Their
generation and use are documented under the applicable OpenAI Terms of Use:
https://openai.com/policies/terms-of-use/




Downloading and Preparing Benchmark Media
-----------------------------------------
WIDER FACE benchmark images and derived annotated qualitative images
are not distributed with the public GitHub repository.

Download WIDER_val.zip, wider_face_split.zip, and eval_tools.zip from:
http://shuoyang1213.me/WIDERFACE/

Extract them into the workspace's datasets/WIDER_FACE/ directory,
alongside the physiotrack-v2 repository, preserving the WIDER_val,
wider_face_split, and eval_tools directory structure documented above.

After preparing the dataset, run the validation scripts in the order
specified in this README. Benchmark qualitative media are generated
locally and must not be committed to GitHub.

Numerical results, tables, and independent quantitative plots may be
preserved in the repository. Independently generated synthetic demo
media are separate from the restricted benchmark images.

See VALIDATION_MEDIA_LICENSES.md for media distribution rules.
