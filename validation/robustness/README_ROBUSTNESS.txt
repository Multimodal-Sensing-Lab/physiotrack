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

validation/integration/test_data/images/frontal_1.jpg

Source photograph:
Pexels (photograph by Christoph Sixt)

Image attribution:
Christoph Sixt, "Professional Portrait of a Smiling Man in Business Attire."
Distributed under the Pexels License.

Source photograph page:
https://www.pexels.com/photo/professional-portrait-of-a-smiling-man-in-business-attire-29856511/

Pexels License:
https://www.pexels.com/license/

Project-level media provenance and distribution documentation:
VALIDATION_MEDIA_LICENSES.md

The source image resolution is:

4160 x 6240 pixels

The source SHA256 recorded by the generator is:

31bb53b9a1b8c0d2ed1cf667651363e14e9e4071deb23fbd4f0697072e3896ff

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
    x: 0.36 to 0.64
    y: 0.44 to 0.53

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
For the selected primary face in each case, the evaluator verifies the
numerical contracts of its available static components.

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
A NO_FACE result is recorded as an observed robustness limitation rather than
an execution failure. An execution-status PASS does not imply unchanged model
predictions or universal detection success.

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
0.902942

dim lighting:
0.901430

overexposed lighting:
0.900659

gaussian blur:
0.897063

motion blur:
0.897024

small face:
0.869114

partial occlusion:
0.887956

The lowest detector confidence occurred under reduced face scale, although
face detection still remained available.

Accepted Robustness Behavior
----------------------------
The controlled perturbations produce different degradation patterns while the
pipeline remains operational.

Face Quality Sharpness
----------------------
The accepted sharpness values are:

baseline:
29.967635

dim lighting:
5.104474

overexposed lighting:
31.140665

gaussian blur:
1.355422

motion blur:
3.012965

small face:
151.258311

partial occlusion:
37.850271

The dim-lighting and blur conditions reduce the sharpness output relative to
the baseline. The small-face and partial-occlusion cases produce higher numeric
sharpness values, consistent with changes in the effective face crop and local
image structure. Sharpness is therefore interpreted as a descriptor response
rather than a universal robustness score.

Face Area Ratio
---------------
The baseline face-area ratio is:

0.087215

The small-face case reduces the face-area ratio to:

0.010680

This confirms that the controlled small-face perturbation materially changes
face scale while preserving successful detection and downstream component
availability.

Selected Output Drift
---------------------
The robustness tables record absolute drift relative to the baseline.

Examples of observed behavior include:

dim lighting
    Absolute head-yaw change: approximately 0.15 degrees
    Absolute learned-gaze yaw change: approximately 5.89 degrees

overexposed lighting
    Absolute head-yaw change: approximately 2.34 degrees
    Absolute learned-gaze yaw change: approximately 70.49 degrees

gaussian blur
    Absolute head-yaw change: approximately 0.13 degrees
    Absolute learned-gaze yaw change: approximately 1.32 degrees

motion blur
    Absolute head-yaw change: approximately 0.05 degrees
    Absolute learned-gaze yaw change: approximately 19.90 degrees

small face
    Absolute head-yaw change: approximately 0.06 degrees
    Absolute learned-gaze yaw change: approximately 85.14 degrees

partial occlusion
    Absolute head-yaw change: approximately 0.91 degrees
    Absolute learned-gaze yaw change: approximately 3.94 degrees

Additional largest observed baseline-relative changes were:

- learned-gaze pitch: 38.77 degrees under reduced face scale
- mouth openness: 0.063953 under partial occlusion
- eye mean openness: 0.015715 under overexposed lighting
- Emotion confidence: 0.207208 under partial occlusion
- regions skin fraction: 0.213169 under partial occlusion

These differences are preserved as descriptive robustness evidence.

Emotion Behavior
----------------
The predicted emotion remained:

Happiness

for all seven controlled cases.

The emotion confidence changed across conditions, but the categorical prediction
did not change relative to the baseline.

This stability is reported descriptively and must not be interpreted as
emotion-recognition accuracy because these controlled images do not carry
independent emotion ground truth.

Quantitative Outputs
--------------------
The evaluator produces the detailed results and summary. The plotting stage
verifies these inputs and produces the derived metrics tables and figures:

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

conda activate PhysioTrack-v2

Open:

validation/robustness

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

preflight and input validation
-> computation and creation of script-specific staged outputs
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

The result nevertheless demonstrates measurable condition-dependent changes.

Dim lighting, Gaussian blur, and motion blur reduced the FaceQuality sharpness
output relative to the baseline. Reduced face scale strongly reduced the
face-area ratio while retaining successful detection and downstream
availability. The small-face condition produced the largest learned-gaze yaw
and pitch changes, while overexposed lighting produced the largest head-yaw
change. The learned-gaze yaw also changed substantially under overexposure.
Lower-face occlusion produced the largest mouth-openness, Emotion-confidence,
and skin-fraction changes. Reduced face scale produced the lowest detector
confidence, although detection remained successful.

The scientifically appropriate conclusion is therefore not that the system is
invariant to these perturbations.

The supported conclusion is:

the tested PhysioTrack static face-analysis pipeline preserves operational
availability under the defined controlled perturbations, while several
component outputs exhibit measurable and condition-specific numerical changes
relative to the baseline.

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

Robustness Validation Status
----------------------------
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


Downloading and Preparing Test Media
------------------------------------
The original Pexels reference image is not distributed with the public
GitHub repository. Users must download it separately to reproduce the
documented controlled robustness validation.

Required source image:
frontal_1.jpg

Creator: Christoph Sixt
Source:
https://www.pexels.com/photo/professional-portrait-of-a-smiling-man-in-business-attire-29856511/

Download the photograph and save it under:

validation/integration/test_data/images/frontal_1.jpg

The accepted source image has a resolution of 4160 x 6240 pixels.
For exact source-file identification, verify its SHA256 checksum against
the value documented in the "Source Image" section above.

After preparing the source image, run:

python robustness_dataset_generator.py

This script automatically generates the seven controlled robustness images
under validation/robustness/test_data/generated_controlled/.
Users do not need to download or create these seven images separately.

Then execute the remaining validation scripts in the order documented
in the "Run Order" section.

Media provenance and reuse conditions are documented in
VALIDATION_MEDIA_LICENSES.md.