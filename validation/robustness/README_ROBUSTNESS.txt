PhysioTrack Controlled Robustness Validation
===========================================

Overview
--------
This directory contains the reproducible controlled robustness validation package
for the final PhysioTrack face-analysis system.

The purpose of this package is not to measure predictive accuracy against an
external benchmark. Instead, it characterizes how the real PhysioTrack
FaceAnalysis pipeline behaves when a fixed frontal reference image is subjected
to deterministic controlled perturbations.

The validation focuses on robustness behavior under:

- baseline conditions
- dim lighting
- overexposed lighting
- Gaussian blur
- motion blur
- reduced face scale
- partial occlusion

The evaluation records whether the face remains detectable, whether the
static-image-compatible face-analysis components remain available, whether
numerical outputs remain valid, and how selected outputs drift relative to the
unmodified baseline.

Validation Location
-------------------
The validation package is located in:

validation/robustness/

The principal scripts are:

robustness_dataset_generator.py
    Generates the deterministic controlled robustness inputs from the accepted
    frontal integration reference image. It preserves the source image,
    records all transformation parameters, writes image hashes, validates the
    staged outputs, and safely replaces only its own generated test-data
    directory.

robustness_eval.py
    Runs the real PhysioTrack FaceAnalysis pipeline on all seven controlled
    robustness cases. It evaluates face detection, static component
    availability, numerical validity, and output drift relative to the
    baseline.

robustness_plot.py
    Independently validates the accepted robustness result CSV and JSON summary,
    recomputes the tabular view, and generates quantitative robustness figures.

robustness_qualitative.py
    Produces deterministic annotated qualitative evidence for all seven
    conditions using the accepted robustness outputs.

Source Image
------------
The controlled robustness inputs are derived from the accepted integration
reference image:

validation/integration/test_data/images/frontal_1.png

The source image resolution is:

843 x 1004 pixels

The source SHA256 recorded by the generator is:

ee1450aca7b11848fa796f2cc2c17f0f18227d61be29887ff074dda08c0d8f3e

The source image is treated as read-only.

Controlled Input Generation
---------------------------
The generator creates seven deterministic cases:

1. baseline
2. dim_lighting
3. overexposed_lighting
4. gaussian_blur
5. motion_blur
6. small_face
7. partial_occlusion

The accepted transformation parameters are:

baseline
    Unmodified reference image.

dim_lighting
    Global intensity scale factor: 0.35

overexposed_lighting
    Linear intensity transform:
    alpha = 1.60
    beta = 55.0

gaussian_blur
    Kernel size: 31
    Sigma: 7.0

motion_blur
    Horizontal motion-blur kernel size: 31

small_face
    Original scene scaled to 35% and centered in a same-size background canvas.

partial_occlusion
    Fixed normalized lower-face occlusion rectangle:
    x: 0.31 to 0.69
    y: 0.53 to 0.70

The generated images are stored under:

validation/robustness/test_data/generated_controlled/

The generator also stores:

robustness_generated_cases.csv
robustness_generated_cases_summary.json

These metadata files document the source image, exact perturbation parameters,
output filenames, and SHA256 hashes required for reproducibility.

Scientific Scope
----------------
This package is a controlled system-behavior evaluation.

It is not:

- a predictive-accuracy benchmark
- a robustness leaderboard score
- an independent seven-image dataset
- a substitute for the component-specific benchmark validations

All seven robustness cases are derived from the same source image. Therefore,
they must not be interpreted as seven statistically independent observations.

The correct interpretation is:

same subject and same original image
-> one controlled perturbation at a time
-> real PhysioTrack execution
-> descriptive comparison against the baseline

Static-Image-Compatible Pipeline
--------------------------------
The robustness evaluator runs the following real PhysioTrack components:

- face detection
- landmarks
- face quality
- head pose
- eye openness
- geometric gaze
- learned gaze estimation
- mouth openness
- emotion recognition
- face regions

The following components are intentionally excluded from this static-image
controlled evaluation:

- tracking
- blink detection
- mouth movement / velocity
- temporal aggregation

These excluded components require sequential temporal evidence and therefore
cannot be evaluated scientifically from a single static image.

The learned gaze configuration is:

gaze_estimation_mode = eth-xgaze
gaze_estimation_min_iou = 0.10

Primary Face Selection
----------------------
Each generated image contains one intended frontal subject.

If multiple detections are returned, the evaluator selects the face with the
highest detector confidence.

This selection rule is deterministic and is used only within this controlled
single-person robustness protocol.

Numerical Validation
--------------------
For every detected face, the evaluator verifies the numerical contracts of all
available static components.

The checks include:

- finite detector confidence
- valid face box geometry
- landmark availability/count
- valid brightness, sharpness, and face-area ratio
- finite head-pose pitch, yaw, and roll
- valid eye openness values
- finite geometric gaze values
- unit-length learned gaze vector
- valid learned gaze pitch, yaw, and association IoU
- valid mouth-openness values
- valid eight-class emotion probability vector
- emotion label equal to the score argmax
- emotion confidence equal to the predicted-class score
- valid face-region skin fraction and association IoU
- no NaN or Inf values in available numerical outputs

No arbitrary robustness pass/fail threshold is imposed on prediction drift.

A detected case can therefore remain operational while still showing meaningful
numerical degradation relative to the baseline.

Accepted Execution Results
--------------------------
Cases:
7

Detected cases:
7

NO_FACE cases:
0

Execution failures:
0

Invalid numerical cases:
0

Input read-only verification:
PASS

All seven cases produced one detected face and all ten evaluated static modules
remained available.

The accepted detector confidences are:

baseline:
0.909525

dim lighting:
0.910930

overexposed lighting:
0.911792

gaussian blur:
0.901819

motion blur:
0.904775

small face:
0.903526

partial occlusion:
0.898910

The lowest detector confidence occurred under partial occlusion, although face
detection still remained available.

Accepted Robustness Behavior
----------------------------
The controlled perturbations produce different degradation patterns while the
pipeline remains operational.

Face Quality Sharpness
----------------------
The accepted sharpness values are:

baseline:
55.3619

dim lighting:
8.29765

overexposed lighting:
89.7561

gaussian blur:
1.59706

motion blur:
3.82887

small face:
102.38

partial occlusion:
67.148

The blur perturbations produce the strongest sharpness degradation, as expected.

The small-face case yields a higher sharpness value because the image content
was rescaled and embedded into a new canvas. This illustrates that sharpness is
not a universal robustness score and should be interpreted together with face
scale and the underlying transformation.

Face Area Ratio
---------------
The baseline face-area ratio is:

0.290097

The small-face case reduces the face-area ratio to:

0.036265

This confirms that the controlled small-face perturbation materially changes
face scale while preserving successful detection and downstream component
availability.

Selected Output Drift
---------------------
The robustness tables record absolute drift relative to the baseline.

Examples of observed behavior include:

dim lighting
    Absolute head-yaw change: approximately 4.50 degrees
    Absolute learned-gaze yaw change: approximately 0.07 degrees

overexposed lighting
    Absolute head-yaw change: approximately 0.03 degrees
    Absolute learned-gaze yaw change: approximately 4.49 degrees

gaussian blur
    Absolute head-yaw change: approximately 1.88 degrees
    Absolute learned-gaze yaw change: approximately 1.87 degrees

motion blur
    Absolute head-yaw change: approximately 10.01 degrees
    Absolute learned-gaze yaw change: approximately 4.18 degrees

small face
    Absolute head-yaw change: approximately 1.66 degrees
    Absolute learned-gaze yaw change: approximately 1.11 degrees

partial occlusion
    Absolute head-yaw change: approximately 2.06 degrees
    Absolute learned-gaze yaw change: approximately 1.58 degrees

These differences are preserved as descriptive robustness evidence.

Emotion Behavior
----------------
The predicted emotion remained:

Anger

for all seven controlled cases.

The emotion confidence changed across conditions, but the categorical prediction
did not change relative to the baseline.

This stability is reported descriptively and must not be interpreted as
emotion-recognition accuracy because these controlled images do not carry
independent emotion ground truth.

Quantitative Outputs
--------------------
The quantitative robustness stage produces:

results/robustness_results.csv
    Detailed per-condition FaceAnalysis output and baseline-relative numerical
    drift.

results/robustness_summary.json
    Protocol, configuration, execution accounting, baseline information, and
    per-condition summary.

results/robustness_metrics.csv
    Compact verified table containing the principal robustness outputs and
    baseline-relative deltas.

results/robustness_metrics.md
    Markdown representation of the compact robustness table.

results/figures/robustness_availability.png
    Static-component availability matrix across the seven controlled
    conditions.

results/figures/robustness_detector_confidence.png
    Face detector confidence across the seven controlled conditions.

results/figures/robustness_quality_sharpness.png
    Face-quality sharpness across the seven controlled conditions.

Qualitative Evidence
--------------------
The qualitative stage preserves all seven controlled cases.

Each annotated image shows the generated robustness condition together with the
face-detection box.

The combined qualitative figure additionally reports:

- case name
- detection status
- number of available static modules
- detector confidence
- sharpness
- absolute head-yaw drift from baseline
- absolute learned-gaze yaw drift from baseline

The qualitative outputs are:

results/qualitative/robustness_qualitative_selection.csv

results/qualitative/annotated_images/

results/figures/robustness_qualitative_examples.png

The qualitative evidence is deterministic and directly tied to the accepted
quantitative results.

Run Order
---------
Activate the project environment:

conda activate PhysioTrack-Thesis

Open:

physiotrack/validation/robustness

Optional controlled-input generator preflight:

python robustness_dataset_generator.py --preflight-only

Generate the controlled robustness inputs:

python robustness_dataset_generator.py

Optional evaluator preflight:

python robustness_eval.py --preflight-only

Run the real controlled robustness evaluation:

python robustness_eval.py

Generate independently verified tables and quantitative figures:

python robustness_plot.py

Generate qualitative evidence:

python robustness_qualitative.py

Safe Rerun Design
-----------------
The robustness package follows the same safe-rerun methodology used throughout
the PhysioTrack validation suite.

The required sequence is:

preflight
-> temporary/staging generation
-> validation of staged outputs
-> replacement of script-owned outputs
-> cleanup of temporary staging data

Existing accepted evidence is not intentionally removed before replacement
outputs have passed the corresponding validation checks.

Output Ownership
----------------
robustness_dataset_generator.py owns:

- test_data/generated_controlled/

This directory contains the seven generated images and their metadata files.

robustness_eval.py owns:

- results/robustness_results.csv
- results/robustness_summary.json

robustness_plot.py owns:

- results/robustness_metrics.csv
- results/robustness_metrics.md
- results/figures/robustness_availability.png
- results/figures/robustness_detector_confidence.png
- results/figures/robustness_quality_sharpness.png

robustness_qualitative.py owns:

- results/qualitative/
- results/figures/robustness_qualitative_examples.png

No script is intended to delete or replace another script's final accepted
outputs.

Expected Final Structure
------------------------
validation/robustness/
|-- robustness_dataset_generator.py
|-- robustness_eval.py
|-- robustness_plot.py
|-- robustness_qualitative.py
|-- README_ROBUSTNESS.txt
|-- test_data/
|   `-- generated_controlled/
|       |-- 01_baseline.png
|       |-- 02_dim_lighting.png
|       |-- 03_overexposed_lighting.png
|       |-- 04_gaussian_blur.png
|       |-- 05_motion_blur.png
|       |-- 06_small_face.png
|       |-- 07_partial_occlusion.png
|       |-- robustness_generated_cases.csv
|       `-- robustness_generated_cases_summary.json
`-- results/
    |-- robustness_results.csv
    |-- robustness_summary.json
    |-- robustness_metrics.csv
    |-- robustness_metrics.md
    |-- figures/
    |   |-- robustness_availability.png
    |   |-- robustness_detector_confidence.png
    |   |-- robustness_quality_sharpness.png
    |   `-- robustness_qualitative_examples.png
    `-- qualitative/
        |-- robustness_qualitative_selection.csv
        `-- annotated_images/
            |-- 01_baseline_annotated.png
            |-- 02_dim_lighting_annotated.png
            |-- 03_overexposed_lighting_annotated.png
            |-- 04_gaussian_blur_annotated.png
            |-- 05_motion_blur_annotated.png
            |-- 06_small_face_annotated.png
            `-- 07_partial_occlusion_annotated.png

Reproducibility
---------------
All validation paths are derived from the repository structure.

No machine-specific absolute path is required by the final robustness scripts.

The source integration image is read-only.

The generated robustness inputs are deterministic and their transformation
parameters and SHA256 hashes are preserved.

The evaluator runs the real PhysioTrack FaceAnalysis pipeline.

The plotting stage independently verifies the accepted result table against the
summary before generating tables or figures.

The qualitative stage reads the accepted numerical outputs and preserves the
same deterministic seven-case ordering.

Methodological Qualifications
-----------------------------
This robustness package must be interpreted within its intended scope.

First, all seven cases are generated from a single frontal source image.
Therefore, the package characterizes controlled degradation behavior and is not
a population-level robustness benchmark.

Second, the test does not assign a universal robustness score. Different
components respond differently to blur, illumination, scale, and occlusion, so
the result is presented as a collection of interpretable numerical changes.

Third, component availability alone does not imply unchanged prediction
quality. A module can remain available while its numerical output drifts
substantially from the baseline.

Fourth, tracking, blink detection, mouth movement, and temporal aggregation are
not evaluated in this static-image protocol because they require temporal
sequences.

Fifth, emotion output stability is not emotion accuracy. The controlled images
have no additional independent ground-truth emotion labels.

Sixth, the small-face transformation changes the effective spatial scale of the
source content. Derived image-quality quantities must therefore be interpreted
together with the face-area ratio and the known transformation.

Scientific Interpretation
-------------------------
The controlled robustness evaluation shows that the final static
PhysioTrack face-analysis pipeline remained operational across all seven tested
conditions.

Face detection succeeded in every case, all ten evaluated static components
remained available, no execution failures occurred, and no invalid numerical
outputs were produced.

The result nevertheless demonstrates measurable condition-dependent
degradation.

Blur produces severe reductions in the face-quality sharpness output.
Motion blur produces the largest observed head-pose yaw drift among the tested
conditions. Overexposure produces a notable learned-gaze yaw drift. Reduced
face scale strongly reduces face-area ratio while retaining successful
detection and downstream availability. Partial occlusion slightly reduces
detector confidence but does not make the pipeline unavailable.

The scientifically appropriate conclusion is therefore not that the system is
invariant to these perturbations.

The supported conclusion is:

the tested PhysioTrack static face-analysis pipeline preserves operational
availability under the defined controlled perturbations, while several
component outputs exhibit measurable and condition-specific numerical
degradation relative to the baseline.

Final Files to Preserve
-----------------------
The final robustness reproducibility package should preserve:

- robustness_dataset_generator.py
- robustness_eval.py
- robustness_plot.py
- robustness_qualitative.py
- README_ROBUSTNESS.txt
- test_data/generated_controlled/
- results/robustness_results.csv
- results/robustness_summary.json
- results/robustness_metrics.csv
- results/robustness_metrics.md
- results/figures/robustness_availability.png
- results/figures/robustness_detector_confidence.png
- results/figures/robustness_quality_sharpness.png
- results/figures/robustness_qualitative_examples.png
- results/qualitative/robustness_qualitative_selection.csv
- results/qualitative/annotated_images/

Temporary staging directories, caches, and obsolete diagnostic artifacts are
not part of the final package.

Current Validation Status
-------------------------
Controlled input generation:
ACCEPTED

Real PhysioTrack robustness execution:
ACCEPTED

Independent quantitative consistency audit:
PASS

Quantitative tables and figures:
ACCEPTED

Qualitative evidence:
ACCEPTED

README:
COMPLETED

Internal thesis documentation:
PENDING

Final package audit:
PENDING
