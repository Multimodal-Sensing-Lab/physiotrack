CAER-S Emotion Recognition Validation
=====================================

Overview
--------
This directory contains the reproducible Emotion Recognition validation package
for PhysioTrack.

The package provides two complementary evidence layers:

1. Scientific benchmark validation on a defined CAER-S target-face test subset.
2. Isolated PhysioTrack Emotion Recognition component execution through the real
   FaceAnalysis project path with unrelated optional face-analysis components
   disabled.

The scientific benchmark measures categorical emotion-classification
performance against CAER-S labels under a controlled target-face crop protocol.
The isolated component execution verifies that the current PhysioTrack
FaceEmotion implementation runs correctly through FaceAnalysis and exports real
numerical emotion outputs.

These procedures answer different questions and must be interpreted separately.
The isolated component execution is software execution evidence and does not
replace the scientific CAER-S classification benchmark.

Dataset
-------
Dataset:

CAER-S

Official project page:

https://caer-dataset.github.io/

CAER-S is the static-image version of the Context-Aware Emotion Recognition
dataset. It contains scene frames organized into seven categorical emotion
classes.

The local dataset used by this validation contains:

Training images:
49007

Test images:
20992

Total images:
69999

Test classes:

- Anger
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset package is expected at:

datasets/CAER-S/

with the structure:

datasets/
└── CAER-S/
    ├── train/
    ├── test/
    └── test.txt

The dataset directory is treated as read-only validation input.

Why CAER-S Was Selected
-----------------------
CAER-S was selected because it provides a large, labeled categorical
emotion-recognition test population with seven emotion classes that have direct
semantic overlap with the fixed PhysioTrack FaceEmotion classifier.

The dataset is suitable for evaluating a fixed pretrained classifier without
training or fine-tuning the PhysioTrack model.

CAER-S also contains realistic scene-level frames rather than only tightly
cropped faces. This is useful for evaluating a practical limitation of the
current component: PhysioTrack FaceEmotion is a face-crop classifier, whereas
CAER-S was designed for context-aware emotion recognition.

For this reason, the validation does not claim to reproduce the original
context-aware CAER-S benchmark. It defines and documents a face-only protocol
that evaluates the current PhysioTrack component as implemented.

Target-Face Detector Result File
--------------------------------
The downloaded CAER-S image package does not contain facial bounding-box
annotations.

A separate published detector-result file is therefore used:

datasets/CAER-S/test.txt

Source implementation:

https://github.com/ndkhanh360/CAER

The referenced CAER-S implementation documents downloadable results from
dlib's CNN face detector for its training, validation, and test loaders. Its
test configuration uses the CAER-S test image directory together with a
test.txt detector-result file.

For reproducibility in this validation package, the downloaded test.txt file is
stored directly inside:

datasets/CAER-S/

The file is not treated as official CAER-S face ground truth.

It is treated as an external published face-detector result file that provides a
deterministic target-face box for a defined subset of the CAER-S test images.

Target-Face File Preflight
--------------------------
The accepted preflight established:

CAER-S test images:
20992

Annotation rows:
13942

Unique annotated image paths:
13942

Duplicate image paths:
0

Missing referenced images:
0

Malformed rows:
0

Class-ID mismatches:
0

Invalid raw boxes:
0

Partially out-of-bounds boxes:
67

Invalid boxes after clipping:
0

Annotated subset coverage:
66.42%

Unannotated test images:
7050

Annotated images per class:

Anger:
1994

Disgust:
1966

Fear:
1954

Happy:
2019

Neutral:
2003

Sad:
2043

Surprise:
1963

The scientific benchmark therefore evaluates exactly 13942 CAER-S test images.

It must not be reported as classification performance on the complete
20992-image CAER-S test split.

The 67 partially out-of-bounds boxes are deterministically clipped to valid
image boundaries before the target-face crop is evaluated.

Validation Files
----------------
The validation implementation is located under:

validation/emotion_recognition/

Files:

- caers_emotion_eval.py
- caers_emotion_plot.py
- caers_emotion_qualitative.py
- emotion_recognition_component_test.py
- README_EMOTION_RECOGNITION.txt

caers_emotion_eval.py
    Runs the scientific CAER-S target-face classification benchmark. It performs
    dataset and protocol preflight checks, evaluates every valid target-face
    crop using the current PhysioTrack FaceEmotion implementation, and writes
    the detailed benchmark result CSV and scientific summary.

caers_emotion_plot.py
    Reads the accepted detailed benchmark CSV, independently verifies its
    numerical integrity, recomputes the classification metrics, checks
    consistency with the evaluator summary, and generates compact tables and
    quantitative figures.

caers_emotion_qualitative.py
    Selects deterministic representative correct and incorrect benchmark cases,
    reproduces the accepted target-face crops, reruns FaceEmotion on those
    selected cases, verifies label, confidence, and all eight model scores, and
    then generates qualitative evidence.

emotion_recognition_component_test.py
    Runs the real PhysioTrack FaceAnalysis path with Emotion Recognition enabled
    and unrelated optional face-analysis components disabled. It exports real
    numerical FaceEmotion outputs for all detected faces and independently
    verifies those outputs against direct FaceEmotion execution on the exact same
    detector crop.

PhysioTrack Emotion Model
-------------------------
The current component uses:

Model:
enet_b0_8_best_afew

Engine:
ONNX

The fixed model outputs eight classes:

- Anger
- Contempt
- Disgust
- Fear
- Happiness
- Neutral
- Sadness
- Surprise

The component returns:

- predicted categorical emotion
- confidence
- complete eight-class score vector

No model training or fine-tuning is performed during validation.

Class Mapping
-------------
CAER-S and the fixed PhysioTrack classifier use slightly different class names
and different output-space sizes.

The primary benchmark uses the following semantic mapping:

CAER-S Anger
-> Anger

CAER-S Disgust
-> Disgust

CAER-S Fear
-> Fear

CAER-S Happy
-> Happiness

CAER-S Neutral
-> Neutral

CAER-S Sad
-> Sadness

CAER-S Surprise
-> Surprise

PhysioTrack also outputs:

Contempt

CAER-S has no corresponding Contempt ground-truth class.

Contempt predictions are retained exactly as produced by the fixed model and
are counted as incorrect in the primary benchmark. They are not removed,
remapped, or renormalized into the seven shared classes.

Scientific Benchmark Protocol
-----------------------------
The benchmark is a controlled face-only categorical emotion-classification
evaluation.

For every row in test.txt:

1. Load the referenced CAER-S test image.
2. Read the supplied target-face bounding box.
3. Clip the box to valid image boundaries when required.
4. Extract exactly one target-face crop.
5. Pass that crop to the current PhysioTrack FaceEmotion implementation.
6. Store the predicted emotion label.
7. Store the predicted-class confidence.
8. Store all eight model class scores.
9. Compare the predicted label with the mapped CAER-S ground-truth class.
10. Preserve explicit failure information if a sample cannot be evaluated.

The PhysioTrack face detector is not used in the scientific benchmark.

This isolates Emotion Recognition from face-selection ambiguity and face
detection errors.

The scientific benchmark therefore measures:

FaceEmotion(target face crop)
-> categorical emotion prediction

rather than:

full-scene face detection
-> face selection
-> emotion prediction

Failure Accounting
------------------
Every annotated target-face sample remains in the benchmark denominator.

The accepted run produced:

Annotated samples:
13942

Successful predictions:
13942

Failed predictions:
0

Availability:
100.00%

No failed sample is silently removed from the benchmark denominator.

Benchmark Metrics
-----------------
The benchmark reports:

- accuracy
- per-class precision
- per-class recall
- per-class F1
- macro F1
- weighted F1
- raw confusion matrix
- prediction distribution
- Contempt prediction count
- Contempt prediction rate
- prediction availability

The confusion matrix has:

7 ground-truth rows
x
8 prediction columns

because Contempt remains a valid fixed-model prediction even though it is not a
CAER-S ground-truth category.

Validated Quantitative Results
------------------------------
Annotated samples:
13942

Successful predictions:
13942

Failed predictions:
0

Availability:
100.00%

Correct predictions:
3692

Accuracy:
0.264811

Macro F1:
0.252623

Weighted F1:
0.254100

Contempt predictions:
523

Contempt prediction rate:
0.037513

The benchmark therefore reports approximately:

Accuracy:
26.48%

Macro F1:
25.26%

Weighted F1:
25.41%

Contempt prediction rate:
3.75%

Prediction Distribution
-----------------------
Accepted prediction counts:

Anger:
1378

Contempt:
523

Disgust:
268

Fear:
2022

Happiness:
1416

Neutral:
1933

Sadness:
4826

Surprise:
1576

Total:
13942

The distribution shows a strong tendency toward Sadness and relatively few
Disgust predictions.

This behavior is preserved as measured model output. It is not corrected by
post-hoc thresholding, relabeling, or score renormalization.

Quantitative Verification
-------------------------
The plotting stage independently verifies the detailed benchmark result table.

For every successful row, it checks:

- unique image path
- valid ground-truth class
- valid predicted class
- finite class scores
- class scores inside [0, 1]
- probability sum approximately equal to 1
- predicted label equal to the maximum-scoring class
- confidence equal to the predicted-class score
- correctness flag equal to the label comparison

It then recomputes the classification metrics directly from the detailed CSV.

The recomputed values match the evaluator summary.

This provides an independent consistency layer between:

- detailed benchmark rows
- summary metrics
- compact tables
- confusion matrix
- prediction distribution
- figures

Quantitative Figures
--------------------
The plotting stage generates:

results/figures/caers_emotion_confusion_matrix.png

    Row-normalized 7-by-8 confusion matrix.

results/figures/caers_emotion_per_class_metrics.png

    Per-class precision, recall, and F1.

results/figures/caers_emotion_prediction_distribution.png

    Eight-class model prediction distribution.

The corresponding numerical data are preserved in CSV files.

Qualitative Evidence
--------------------
The qualitative stage selects ten deterministic examples from the accepted
benchmark result set.

The selection contains:

- one representative correct Anger prediction
- one representative correct Disgust prediction
- one representative correct Fear prediction
- one representative correct Happiness prediction
- one representative correct Neutral prediction
- one representative correct Sadness prediction
- one representative correct Surprise prediction
- one representative incorrect prediction
- one high-confidence incorrect prediction
- one representative Contempt prediction

Representative correct examples are selected using confidence nearest the
median confidence among correct predictions of that class.

The representative incorrect example uses confidence nearest the median among
incorrect predictions.

The high-confidence incorrect example is selected from the remaining incorrect
population.

The Contempt example is selected using confidence nearest the median among
Contempt predictions.

For every selected sample, the qualitative script:

1. reloads the original CAER-S frame;
2. reproduces the accepted clipped target-face box;
3. extracts the same FaceEmotion input crop;
4. reruns the current FaceEmotion model;
5. verifies the predicted label;
6. verifies the prediction confidence;
7. verifies all eight class scores;
8. writes the qualitative output only after numerical consistency passes.

Each individual qualitative panel shows:

- original CAER-S scene
- external target-face box
- exact FaceEmotion crop
- ground-truth class
- predicted class
- confidence
- correctness state

The combined qualitative figure is stored at:

results/figures/caers_emotion_qualitative_examples.png

The qualitative evidence intentionally preserves failure cases and must not be
interpreted as a success-only gallery.

Safe Rerun Design
-----------------
The validation package follows the safe-rerun pattern:

preflight
-> temporary/staging generation
-> validation of newly generated outputs
-> replacement of script-owned final outputs

Previously valid final evidence is not intentionally deleted before the newly
generated replacement has passed the relevant validation checks.

If required dataset files, accepted result dependencies, or other critical
inputs are unavailable or invalid, the affected script stops before replacing
its previously valid final outputs.

Each script owns only its own outputs.

Output Ownership
----------------
caers_emotion_eval.py
    Owns:
    - results/caers_emotion_results.csv
    - results/caers_emotion_summary.txt

caers_emotion_plot.py
    Owns:
    - results/caers_emotion_metrics.csv
    - results/caers_emotion_thesis_table.csv
    - results/caers_emotion_thesis_table.md
    - results/caers_emotion_per_class_metrics.csv
    - results/caers_emotion_confusion_matrix.csv
    - results/caers_emotion_prediction_distribution.csv
    - results/figures/caers_emotion_confusion_matrix.png
    - results/figures/caers_emotion_per_class_metrics.png
    - results/figures/caers_emotion_prediction_distribution.png

caers_emotion_qualitative.py
    Owns:
    - results/qualitative/
    - results/figures/caers_emotion_qualitative_examples.png

emotion_recognition_component_test.py
    Owns:
    - results/component_execution/emotion_recognition_component_results.csv
    - results/component_execution/emotion_recognition_component_summary.json

No validation script is intended to replace another script's accepted final
outputs.

Isolated PhysioTrack Emotion Recognition Execution
--------------------------------------------------
The isolated execution verifies the real PhysioTrack path:

Face detector
-> FaceEmotion

inside FaceAnalysis.

The following unrelated optional components are disabled:

- tracking
- head pose
- landmarks
- quality
- eyes
- blink
- geometric gaze
- learned gaze estimation
- mouth openness
- mouth motion
- face regions
- temporal aggregation

The face detector remains active because the normal FaceAnalysis Emotion
Recognition path requires a detected face crop.

The isolated component execution uses the same 13942-image CAER-S annotated
subset image population for software-execution coverage.

The CAER-S emotion labels are not used to compute scientific accuracy in this
stage.

The isolated execution therefore answers:

Does the current real PhysioTrack FaceAnalysis path produce valid numerical
FaceEmotion outputs?

It does not answer:

How accurate is the detector-selected face against CAER-S emotion ground truth?

Validated Isolated Component Results
------------------------------------
Input images:
13942

Total result rows:
23746

Detected face rows:
23745

Emotion-available rows:
23745

Images with at least one Emotion output:
13941

EMOTION_UNAVAILABLE:
0

INVALID_FACE_BOX:
0

NO_FACE:
1

EXECUTION_FAILED:
0

Dataset read-only verification:
PASS

For every detected face, the component test compares the FaceAnalysis Emotion
Recognition output with direct FaceEmotion execution on the exact same
FaceAnalysis detector crop.

Validated numerical consistency:

Maximum confidence absolute error:
0.0

Maximum class-score absolute error:
0.0

Minimum eight-class score sum:
0.9999998093

Maximum eight-class score sum:
1.0000001615

The exact numerical agreement confirms that the current FaceAnalysis
integration exports the same predicted label, confidence, and eight-class score
vector as direct FaceEmotion execution on the identical crop.

The single NO_FACE case is preserved explicitly as a detector outcome.

It is not an Emotion Recognition execution failure and does not modify the
scientific benchmark, which uses the external target-face boxes directly.

Relationship Between Benchmark and Isolated Execution
-----------------------------------------------------
The scientific benchmark and the isolated component execution are deliberately
different.

Scientific benchmark:

- input localization comes from test.txt
- exactly one target face is evaluated per annotated image
- CAER-S labels are used
- classification accuracy and F1 are computed

Isolated component execution:

- input localization comes from the real PhysioTrack face detector
- every detected face is preserved
- CAER-S labels are not used for accuracy
- real FaceAnalysis output is verified numerically
- software execution and export are tested

A PASS result from isolated component execution must therefore not be reported
as scientific CAER-S classification accuracy.

Scientific Benchmark Run Order
------------------------------
Activate the project environment:

conda activate PhysioTrack-Thesis

Open:

physiotrack/validation/emotion_recognition

Optional preflight-only check:

python caers_emotion_eval.py --preflight-only

Optional deterministic evaluator smoke test:

python caers_emotion_eval.py --smoke-test 100

Run the scientific benchmark:

python caers_emotion_eval.py

Generate verified quantitative tables and figures:

python caers_emotion_plot.py

Generate qualitative evidence:

python caers_emotion_qualitative.py

The quantitative benchmark must be accepted before downstream tables, figures,
and qualitative outputs are treated as final evidence.

Isolated Component Run Order
----------------------------
Optional deterministic smoke test:

python emotion_recognition_component_test.py --smoke-test 8

Full isolated component execution:

python emotion_recognition_component_test.py

The component test writes only its component_execution outputs.

Expected Final Result Structure
-------------------------------
results/
├── caers_emotion_results.csv
├── caers_emotion_summary.txt
├── caers_emotion_metrics.csv
├── caers_emotion_thesis_table.csv
├── caers_emotion_thesis_table.md
├── caers_emotion_per_class_metrics.csv
├── caers_emotion_confusion_matrix.csv
├── caers_emotion_prediction_distribution.csv
├── qualitative/
│   ├── annotated_images/
│   │   └── ten annotated PNG examples
│   └── caers_emotion_qualitative_selection.csv
├── component_execution/
│   ├── emotion_recognition_component_results.csv
│   └── emotion_recognition_component_summary.json
└── figures/
    ├── caers_emotion_confusion_matrix.png
    ├── caers_emotion_per_class_metrics.png
    ├── caers_emotion_prediction_distribution.png
    └── caers_emotion_qualitative_examples.png

Output Descriptions
-------------------
results/caers_emotion_results.csv
    Detailed per-image scientific benchmark output containing target-face box
    information, predicted label, confidence, all eight class scores,
    correctness, and failure information.

results/caers_emotion_summary.txt
    Text record of the scientific protocol, sample accounting, aggregate
    metrics, per-class metrics, and interpretation.

results/caers_emotion_metrics.csv
    Compact independently recomputed aggregate metric table.

results/caers_emotion_thesis_table.csv
    Thesis-oriented compact quantitative result table.

results/caers_emotion_thesis_table.md
    Markdown representation of the compact thesis table.

results/caers_emotion_per_class_metrics.csv
    Per-class support, prediction count, confusion accounting, precision,
    recall, and F1.

results/caers_emotion_confusion_matrix.csv
    Raw 7-by-8 integer confusion matrix.

results/caers_emotion_prediction_distribution.csv
    Count and rate of predictions for each of the eight model classes.

results/figures/caers_emotion_confusion_matrix.png
    Row-normalized quantitative confusion-matrix figure.

results/figures/caers_emotion_per_class_metrics.png
    Per-class precision, recall, and F1 figure.

results/figures/caers_emotion_prediction_distribution.png
    Eight-class prediction-distribution figure.

results/qualitative/annotated_images/
    Ten deterministic individual qualitative benchmark examples.

results/qualitative/caers_emotion_qualitative_selection.csv
    Machine-readable record of qualitative roles, source images, target-face
    boxes, accepted predictions, scores, and generated image paths.

results/figures/caers_emotion_qualitative_examples.png
    Combined qualitative overview figure.

results/component_execution/emotion_recognition_component_results.csv
    Structured numerical outputs from isolated real FaceAnalysis Emotion
    Recognition execution.

results/component_execution/emotion_recognition_component_summary.json
    Isolated execution configuration, accounting, numerical-consistency checks,
    dataset read-only verification, and final execution status.

Complete Reproduction Procedure
-------------------------------
1. Install the PhysioTrack project environment and dependencies.

2. Download CAER-S from the dataset source documented above.

3. Extract the dataset to:

   datasets/CAER-S/

4. Confirm that the local image structure contains:

   datasets/CAER-S/train/
   datasets/CAER-S/test/

5. Obtain the published external CAER-S dlib CNN detector result file used by
   the referenced CAER implementation.

6. Store the test detector-result file as:

   datasets/CAER-S/test.txt

7. Do not modify the raw CAER-S images or test.txt during validation.

8. Activate:

   conda activate PhysioTrack-Thesis

9. Open:

   physiotrack/validation/emotion_recognition

10. Run the optional evaluator preflight and smoke test when required.

11. Run:

    python caers_emotion_eval.py
    python caers_emotion_plot.py
    python caers_emotion_qualitative.py

12. Verify the 13942-image benchmark accounting, classification metrics,
    confusion matrix, prediction distribution, and qualitative reproduction.

13. Run the isolated software-execution validation:

    python emotion_recognition_component_test.py

14. Verify the component-execution CSV and JSON summary.

15. Confirm result schemas, failure accounting, numerical consistency, dataset
    read-only handling, output ownership, and expected final artifacts before
    treating the package as reproducibility evidence.

Reproducibility
---------------
All validation paths are project-relative.

No machine-specific absolute path is required by the validation scripts.

The expected top-level relationship is:

<project_root>/
├── datasets/
│   └── CAER-S/
│       ├── train/
│       ├── test/
│       └── test.txt
└── physiotrack/
    └── validation/
        └── emotion_recognition/

The benchmark stores complete per-sample numerical outputs rather than only
aggregate metrics.

The plotting stage independently recomputes the accepted metrics from the
detailed CSV.

The qualitative stage reproduces selected samples numerically before writing
visual evidence.

The isolated component stage compares FaceAnalysis output against direct
FaceEmotion output on the same detector crop.

The dataset is treated as read-only.

Methodological Note
-------------------
The reported scientific benchmark values describe the fixed PhysioTrack
FaceEmotion classifier under the documented CAER-S target-face protocol.

They should not be presented as an official CAER-S leaderboard result.

CAER-S was created for context-aware emotion recognition, whereas the current
PhysioTrack component uses only a face crop.

The external target-face detector file is an evaluation adapter required to
make target-face association deterministic for this face-only component.

It is not official CAER-S facial ground truth.

Limitations
-----------
- The scientific benchmark evaluates 13942 of the 20992 CAER-S test images,
  corresponding to 66.42% of the test split.

- The target-face boxes come from an external published dlib detector-result
  file rather than official CAER-S bounding-box annotations.

- The current classifier does not use the wider scene context for which CAER-S
  was designed.

- The fixed model includes Contempt, while CAER-S does not. Contempt predictions
  are therefore retained and counted as incorrect in the primary benchmark.

- No CAER-S training or fine-tuning is performed.

- The fixed classifier shows a substantial prediction tendency toward Sadness
  and weak performance for some classes.

- Qualitative examples complement the complete quantitative benchmark but do
  not replace dataset-level metrics.

- The isolated component execution uses detector-selected faces and is not a
  scientific accuracy benchmark.

- Runtime is machine-dependent and is not used as a classification metric.

Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- caers_emotion_eval.py
- caers_emotion_plot.py
- caers_emotion_qualitative.py
- emotion_recognition_component_test.py
- README_EMOTION_RECOGNITION.txt
- results/caers_emotion_results.csv
- results/caers_emotion_summary.txt
- results/caers_emotion_metrics.csv
- results/caers_emotion_thesis_table.csv
- results/caers_emotion_thesis_table.md
- results/caers_emotion_per_class_metrics.csv
- results/caers_emotion_confusion_matrix.csv
- results/caers_emotion_prediction_distribution.csv
- results/figures/caers_emotion_confusion_matrix.png
- results/figures/caers_emotion_per_class_metrics.png
- results/figures/caers_emotion_prediction_distribution.png
- results/qualitative/annotated_images/
- results/qualitative/caers_emotion_qualitative_selection.csv
- results/figures/caers_emotion_qualitative_examples.png
- results/component_execution/emotion_recognition_component_results.csv
- results/component_execution/emotion_recognition_component_summary.json

Generated caches, temporary staging directories, smoke-test-only outputs, and
obsolete diagnostic files are not part of the final validation deliverables.

Current Validation Status
-------------------------
Scientific benchmark:
ACCEPTED

Quantitative consistency audit:
PASS

Qualitative evidence:
ACCEPTED

Isolated PhysioTrack component execution:
PASS

The package preserves scientific benchmark evidence, deterministic qualitative
evidence, safe-rerun safeguards, and real isolated component outputs separately
for reproducibility.
