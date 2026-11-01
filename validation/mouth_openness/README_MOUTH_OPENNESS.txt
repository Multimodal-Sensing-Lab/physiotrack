PhysioTrack Mouth-Openness Validation
=====================================

Purpose
-------
This validation evaluates continuous mouth-openness estimation in the
PhysioTrack face-analysis module using paired FELT facial-motion annotations
and RAVDESS speech videos.

The validation target is the combination of:

- PhysioTrack FaceLandmarks, based on the MediaPipe face landmarker
- PhysioTrack MouthOpenness

The benchmark is a regression evaluation of continuous mouth openness. It is
not an open/closed mouth classification task.

After the benchmark, plotting, and qualitative stages are accepted, a separate
isolated component-execution test verifies the current MouthOpenness component
through the real PhysioTrack FaceAnalysis pipeline. The isolated execution is
not a second accuracy benchmark and does not recompute FELT agreement metrics.

A separate synthetic qualitative demonstration documents native end-to-end
face detection, facial landmarks, and mouth-openness outputs on generated
images. Its illustrative measurements are not ground-truth mouth-openness
accuracy scores and do not replace the FELT/RAVDESS benchmark.

Validation Location
-------------------
The validation package is located in:

validation/mouth_openness/

The principal scripts are:

felt_ravdess_mouth_openness_eval.py
    Runs the quantitative benchmark and writes per-frame, per-actor, and
    summary results.

felt_ravdess_mouth_openness_plot.py
    Recomputes and verifies the stored quantitative metrics, generates the
    thesis-oriented CSV and Markdown result tables, and creates quantitative
    figures from the saved benchmark outputs.

felt_ravdess_mouth_openness_qualitative.py
    Selects deterministic benchmark examples from the accepted quantitative
    result set, re-evaluates only those selected frames, and generates
    annotated qualitative examples.

mouth_openness_component_test.py
    Verifies the current MouthOpenness implementation through the real
    FaceAnalysis pipeline on the complete accepted FELT/RAVDESS speech
    population and stores real per-frame numerical outputs without computing
    a second set of accuracy metrics.

mouth_openness_synthetic_demo.py
    Discovers independently generated synthetic images, executes the real
    PhysioTrack FaceAnalysis path with face detection, FaceLandmarks, and
    MouthOpenness enabled, and exports actual per-face measurements, numerical
    checks, and annotated qualitative examples.

Datasets and Scope
------------------
FELT speech annotations:

Official source:
https://zenodo.org/records/13243600

Original dataset academic citation:
Liao, Z., Livingstone, S. R., & Russo, F. A. (2024).
RAVDESS Facial expression and landmark tracking (Version 1.0.0) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.13243600

Project-relative location:
datasets/FELT/raw_motion_speech/

RAVDESS speech videos:

Official source:
https://zenodo.org/records/1188976

Original dataset research paper:
Livingstone, S. R., & Russo, F. A. (2018).
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS):
A dynamic, multimodal set of facial and vocal expressions in North American English.
PLOS ONE, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

Project-relative location:
datasets/RAVDESS/Video_Speech/


The evaluated scope is the speech subset represented by 24 actors and 60
paired trials per actor:

Actors: 24
Paired speech trials: 1440
Unique annotated frames: 158286

FELT provides per-frame 68-point facial landmarks and face rectangles. RAVDESS
provides the corresponding audiovisual speech frames used as input to
PhysioTrack.

The evaluation scripts treat both dataset directories as read-only benchmark
inputs. Generated validation artifacts are written only under
validation/mouth_openness/results/.

Mouth-Openness Definition
-------------------------
PhysioTrack defines mouth openness as a dimensionless ratio between the
central vertical lip separation and the horizontal mouth-corner width.

For MediaPipe landmarks, PhysioTrack computes:

mouth openness = d(13,14) / d(61,291)

where landmarks 13 and 14 represent the central upper and lower lip locations,
and landmarks 61 and 291 represent the left and right mouth corners.

The current implementation converts normalized MediaPipe landmark coordinates
using the image width and image height before Euclidean geometry is evaluated.
This image-dimension-aware formulation keeps horizontal and vertical distances
physically consistent on non-square frames.

The primary FELT landmark-derived reference uses the corresponding geometry in
the 68-point annotation scheme:

primary FELT reference = d(62,66) / d(48,54)

A secondary robustness reference is also evaluated:

secondary FELT reference =
    mean[d(61,67), d(62,66), d(63,65)] / d(48,54)

The secondary reference averages three inner-lip vertical distances while
using the same mouth-corner normalization. It is used only as a robustness
check; the central-pair definition remains the primary benchmark reference.

Evaluation Protocol
-------------------
The validation is a controlled component-level evaluation designed to measure
mouth-openness estimation without confounding the result with independent face
detection errors.

For every annotated frame:

1. The RAVDESS frame is addressed using the exact FELT frame index.
2. The FELT FaceRect is converted from [x, y, width, height] to
   [x, y, x + width, y + height].
3. The FELT FaceRect is supplied to PhysioTrack FaceLandmarks as controlled
   face initialization.
4. FaceLandmarks produces MediaPipe facial landmarks from the corresponding
   video frame.
5. MouthOpenness computes the PhysioTrack mouth-openness ratio.
6. The prediction is compared with the FELT landmark-derived reference for the
   same annotated frame.

No temporal offset is applied. RAVDESS trailing frames without FELT
annotations are ignored.

Duplicate Annotation Handling
-----------------------------
The FELT speech subset contains two duplicate annotation rows associated with
duplicated frame identifiers in one trial. Duplicate frame identifiers are
resolved deterministically using the following rule:

1. Retain the annotation with the largest FaceRect area.
2. If FaceRect areas are equal, retain the annotation with the highest
   FaceScore.

After duplicate resolution, the benchmark contains 158286 unique annotated
frames.

Quality and Inclusion Rules
---------------------------
The benchmark does not exclude samples using thresholds on:

- FaceScore
- mouth width
- mouth openness
- head pose

All valid annotated frames in the defined speech subset are retained. This
preserves difficult poses and extreme mouth configurations rather than
selectively removing challenging samples.

Failure Accounting
------------------
Every annotated frame is assigned to exactly one result state:

- success
- video_read_failure
- landmark_failure
- invalid_reference
- prediction_failure

The quantitative run produced:

Total annotated frames: 158286
Successful predictions: 158286
Video read failures: 0
Landmark failures: 0
Invalid references: 0
Prediction failures: 0
Availability: 100.0000%

The FELT and RAVDESS dataset integrity checks both passed after evaluation.

Observed complete benchmark runtime: 3374.30 seconds (environment dependent).

Safe Rerun and Dataset Protection
---------------------------------
The final validation scripts follow a staged safe-rerun workflow:

1. Validate the locked dataset and protocol.
2. Create a temporary staging area under validation/mouth_openness/results/
   before generative work begins.
3. Generate only the outputs owned by the active script inside staging.
4. Re-read and validate the staged outputs.
5. Replace final script-owned outputs only after staged validation passes.
6. Preserve or restore prior accepted outputs if final installation fails.
7. Remove temporary staging data after completion.

This transactional replacement behavior provides rollback protection for prior
accepted evidence if a final installation step fails.

The quantitative evaluator records inventories of both FELT and RAVDESS before
and after the full run. Dataset paths, file sizes, and modification timestamps
must remain unchanged. Both datasets are therefore treated as read-only
benchmark inputs.

The evaluator also supports:

python felt_ravdess_mouth_openness_eval.py --preflight-only

for dataset/protocol validation without inference, and:

python felt_ravdess_mouth_openness_eval.py --validate-existing-results-only

for full serialized-result validation without rerunning the 158286-frame
inference.

Quantitative Metrics
--------------------
The primary benchmark reports complementary error, association, and agreement
metrics:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- median absolute error
- standard deviation of absolute error
- 90th percentile absolute error
- 95th percentile absolute error
- mean signed error, defined as prediction minus reference
- Pearson correlation coefficient
- Spearman rank correlation coefficient
- Lin's concordance correlation coefficient (CCC)

The same metrics are also computed against the secondary three-pair FELT
reference as a robustness analysis.

Primary Quantitative Results
----------------------------
Primary reference: d(62,66) / d(48,54)

Frames: 158286
Availability: 100.0000%
MAE: 0.032706
RMSE: 0.046186
Median absolute error: 0.024819
Standard deviation of absolute error: 0.032611
90th percentile absolute error: 0.069262
95th percentile absolute error: 0.087438
Mean signed error: -0.001269
Pearson r: 0.932273
Spearman rho: 0.937550
Lin CCC: 0.931997

The near-zero mean signed error indicates little overall directional bias. The
high Pearson, Spearman, and concordance coefficients indicate strong tracking
of continuous mouth-opening variation and strong numerical agreement with the
FELT landmark-derived reference.

Secondary Robustness Results
----------------------------
Secondary reference:
mean[d(61,67), d(62,66), d(63,65)] / d(48,54)

MAE: 0.032722
RMSE: 0.046108
Median absolute error: 0.025041
Standard deviation of absolute error: 0.032485
90th percentile absolute error: 0.068434
95th percentile absolute error: 0.086617
Mean signed error: -0.001222
Pearson r: 0.931966
Spearman rho: 0.936923
Lin CCC: 0.931328

The primary and secondary results are nearly identical. This demonstrates that
the overall validation conclusion is robust to the reasonable alternative
inner-lip reference formulation.

Face-Box-Normalized Sensitivity Results
---------------------------------------
This complementary geometric sensitivity analysis uses the accepted FELT
FaceRect height to normalize lip separation:

FELT face-box reference = d(62,66) / FaceRectHeight
PhysioTrack face-box prediction = predicted lip height (pixels) / FaceRectHeight

The accepted run reported:

MAE: 0.008651
RMSE: 0.011674
Median absolute error: 0.006740
Standard deviation of absolute error: 0.007838
90th percentile absolute error: 0.018508
95th percentile absolute error: 0.023070
Mean signed error: -0.000475
Pearson r: 0.936960
Spearman rho: 0.939203
Lin CCC: 0.936821

These face-box-normalized values are a separate sensitivity descriptor and
must not be confused with the primary mouth-corner-normalized mouth-openness
regression results.

Per-Actor Analysis
------------------
Per-actor results are stored in:

results/felt_ravdess_mouth_openness_per_actor.csv

Availability is 100% for every actor. Performance varies across individuals,
which is expected for a facial-geometry regression task involving differences
in appearance, expression, and head configuration.

The lowest per-actor MAE is 0.019506 for Actor_02. The highest per-actor MAE is
0.060837 for Actor_21. Actor_21 nevertheless retains a Pearson correlation of
0.950070, indicating that its higher error is dominated by systematic
agreement differences rather than a loss of sensitivity to temporal mouth
opening.

Quantitative Figures
--------------------
The plotting script generates two thesis-ready table formats and three
complementary quantitative figures.

Generated thesis tables:

felt_ravdess_mouth_openness_thesis_table.csv
felt_ravdess_mouth_openness_thesis_table.md
    Overall benchmark metrics, including primary-reference and face-box
    sensitivity results.

felt_ravdess_mouth_openness_per_actor_thesis_table.csv
felt_ravdess_mouth_openness_per_actor_thesis_table.md
    Per-actor benchmark metrics for all 24 actors.

Quantitative figures:

felt_ravdess_mouth_openness_agreement.png
    Density-based agreement plot comparing FELT reference values with
    PhysioTrack predictions against the identity line.

felt_ravdess_mouth_openness_error_distribution.png
    Distribution of signed prediction error, including zero error and the mean
    signed error.

felt_ravdess_mouth_openness_per_actor.png
    Per-actor MAE with the frame-weighted overall MAE shown for comparison.

These figures characterize agreement, error distribution, and inter-subject
variation, respectively.

Qualitative Validation
----------------------
The qualitative script selects eight deterministic examples from the stored
quantitative result set. The selection covers the operating range and includes
challenging disagreement cases:

- closed representative
- low opening
- medium opening
- high opening
- very high opening
- representative error
- challenging underestimate
- challenging overestimate

The selected cases use different actors and illustrate both close agreement and
meaningful failure modes. The individual annotated images show:

- the FELT FaceRect used for controlled face initialization
- FELT mouth geometry
- PhysioTrack mouth geometry
- FELT reference value
- PhysioTrack prediction
- absolute error

For each selected sample, the qualitative script re-evaluates the corresponding
frame and verifies consistency with the stored quantitative prediction before
writing the output image.

The combined qualitative figure is stored as:

results/figures/felt_ravdess_mouth_openness_qualitative_examples.png

Output Ownership
----------------
The evaluator owns only benchmark-result outputs:

- results/felt_ravdess_mouth_openness_per_frame.csv
- results/felt_ravdess_mouth_openness_per_actor.csv
- results/felt_ravdess_mouth_openness_summary.txt

The plotting script owns only derived tables and quantitative figures:

- results/felt_ravdess_mouth_openness_thesis_table.csv
- results/felt_ravdess_mouth_openness_thesis_table.md
- results/felt_ravdess_mouth_openness_per_actor_thesis_table.csv
- results/felt_ravdess_mouth_openness_per_actor_thesis_table.md
- results/figures/felt_ravdess_mouth_openness_agreement.png
- results/figures/felt_ravdess_mouth_openness_error_distribution.png
- results/figures/felt_ravdess_mouth_openness_per_actor.png

The qualitative script owns only qualitative outputs:

- results/qualitative/felt_ravdess_mouth_openness_qualitative_selection.csv
- results/qualitative/annotated_images/
- results/figures/felt_ravdess_mouth_openness_qualitative_examples.png

Each script owns only the outputs listed above. New outputs are generated in
staging and validated before the corresponding accepted files are replaced.
The Markdown thesis tables are final plot-owned outputs and are regenerated
together with their matching CSV tables during a clean plot rerun.

The synthetic demonstration script independently owns only:

- results/synthetic_demo/ (the entire directory, including annotated panels,
  the per-face CSV, contact sheet, and JSON/TXT reports)

A synthetic-demo rerun does not delete or replace outputs owned by the
benchmark evaluator, plotting, benchmark qualitative, or isolated component
execution scripts.

Isolated Component Execution Verification
-----------------------------------------
The accepted isolated component-execution script is:

mouth_openness_component_test.py

It executes the current MouthOpenness component through the real PhysioTrack
FaceAnalysis pipeline using the complete accepted FELT/RAVDESS speech
population.

Configuration:

- Target component: MouthOpenness
- Required prerequisite: FaceLandmarks
- Controlled upstream input: accepted FELT FaceRect bounding boxes
- Device: cpu
- Actors: 24
- Paired speech trials: 1440
- Raw FELT annotation rows: 158288
- Unique annotated frames: 158286
- Duplicate annotation rows resolved: 2

The controlled FELT FaceRect input prevents face-detector behavior from
confounding the isolated MouthOpenness execution. FaceLandmarks remains enabled
because MouthOpenness requires current facial landmarks. All unrelated optional
components are disabled:

- tracking
- head_pose
- quality
- eyes
- blink
- gaze
- gaze_estimation
- mouth_motion
- emotion
- regions
- temporal

The isolated script records real numerical outputs for every accepted frame,
including:

- actor, trial, frame, timestamp, and FPS
- frame dimensions
- accepted FaceRect and FaceScore
- duplicate-candidate count
- landmark availability and landmark count
- mouth-openness availability
- mouth_openness
- mouth_width
- mouth_height
- explicit execution status and failure reason

For every available row, the script verifies finite numerical values and checks
that:

mouth_openness = mouth_height / mouth_width

The accepted full isolated run produced:

Actors: 24
Paired speech trials: 1440
Unique annotated frames: 158286
Landmarks available rows: 158286
MouthOpenness available rows: 158286
Execution failures: 0
Component unavailable rows: 0
Overall status: PASS
Runtime: 1890.51 seconds (approximately 31.51 minutes)

Runtime is environment-dependent and is not a scientific performance metric.

A frame-by-frame audit against the accepted quantitative benchmark confirmed
that the isolated FaceAnalysis MouthOpenness value matched the accepted
PhysioTrack benchmark prediction exactly across all 158286 frames. This
supports that the isolated execution exercises the same current MouthOpenness
implementation rather than a duplicated local formula.

Git-Safe Isolated Result Handling
---------------------------------
The isolated component result writer checks the generated CSV size before
final installation.

If the complete result CSV is at or below 90 MiB, it remains:

results/component_execution/mouth_openness_component_results.csv

If the file would exceed 90 MiB, it is split automatically into sequential
parts:

mouth_openness_component_results_part001.csv
mouth_openness_component_results_part002.csv
...

Splitting is performed only between complete actor groups. An actor is never
split across multiple result files. The summary JSON records every generated
filename, row count, actor boundary, and file size.

The accepted run produced one CSV containing 158286 rows because its size was
approximately 30.33 MiB.

The isolated summary is stored as:

results/component_execution/mouth_openness_component_summary.json

Synthetic Qualitative Demonstration
-----------------------------------
Purpose and scope
~~~~~~~~~~~~~~~~~
The independent, non-benchmark illustration is implemented by:

mouth_openness_synthetic_demo.py

It runs the real PhysioTrack face detector -> FaceAnalysis -> FaceLandmarks ->
MouthOpenness path on generated images. Unlike the controlled FELT/RAVDESS
benchmark and isolated component execution, this stage uses native face
localization rather than supplied FELT FaceRect boxes. It is intended to
illustrate operational behavior, not to compute new scientific accuracy
metrics against independently annotated mouth-opening references.

Input media and case coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The script automatically discovers supported raster images recursively under:

validation/synthetic_media/mouth_openness_media/

The accepted demonstration consists of twelve PNG images:

- 01.png: neutral frontal face, fully closed mouth (baseline).
- 02.png: very slightly separated lips.
- 03.png: slightly open mouth.
- 04.png: moderately open mouth.
- 05.png: widely open mouth.
- 06.png: maximum natural mouth opening.
- 07.png: closed-mouth smile.
- 08.png: open-mouth smile with visible teeth.
- 09.png: rounded, O-shaped lips.
- 10.png: rotated head and moderately open mouth.
- 11.png: partially occluded mouth.
- 12.png: three faces with different apparent mouth openings.

All twelve input images in the accepted execution were 1254 x 1254 pixels.
The script uses their actual dimensions and does not require 1024 x 1024
images. The opening conditions are nominal synthetic descriptions, not
physically calibrated lip-separation ground truth.

Synthetic media provenance and usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The twelve input images (01.png through 12.png) were generated using
OpenAI ChatGPT for independent illustrative testing of MouthOpenness.
They are not derived from FELT or RAVDESS benchmark media and are not
associated with independently measured mouth-opening ground truth.
OpenAI Terms of Use: https://openai.com/policies/terms-of-use/
Project media provenance and redistribution documentation:
VALIDATION_MEDIA_LICENSES.md

Execution configuration and reported fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The demonstration executes on CPU with the native PhysioTrack face detector,
FaceLandmarks, and MouthOpenness enabled, and unrelated optional components,
including tracking, mouth motion, and temporal aggregation, disabled.
The detector confidence threshold is 0.001, the detector IoU threshold is
0.45, and the maximum detection count is 10000. A separate confidence
threshold of 0.25 affects annotations only; all raw detections remain in
the CSV, including any that are omitted from visualization.

The per-face CSV records image identity, original dimensions, face index,
detector confidence, face box, 478-landmark availability/count, numerical
mouth-openness availability, mouth_openness, mouth_width, mouth_height,
original lip-landmark coordinates, independently recomputed geometry, and
absolute numerical discrepancies. The definition is:

mouth_openness = pixel_distance(13, 14) / pixel_distance(61, 291)

The script independently recomputes lip distances with image-width and
image-height-aware geometry and checks recorded output values using an
absolute tolerance of 1e-12. Detector confidence is upstream detection
metadata, not a mouth-opening accuracy score.

Annotated visual presentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each generated annotated image has three separate areas: the original source
image preserved at its native pixel dimensions, a same-size annotated image
showing the detected face boxes and measured lip geometry, and a dedicated
460-pixel-wide dark measurement dashboard. Text labels are not drawn on or
above faces. For each displayed face, the dashboard lists face index, detector
confidence, mouth_openness, mouth_width, mouth_height, and measurement
availability. The panel height expands when needed so that all displayed
faces, including the three-face example, receive separate readable entries.
The contact sheet is generated from these annotated three-area panels.

This layout change affects only visual presentation and corresponding
annotated-image dimensions. It leaves the real FaceAnalysis inference,
mouth-openness equations, per-face CSV schema, benchmark results, and
independent numerical verification unchanged.

Accepted execution results
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Synthetic images discovered and processed: 12 / 12.
- Failed images: 0.
- Raw detected face observations and CSV rows: 14.
- Detections shown in the annotated figures: 14.
- Faces with available 478-point landmarks: 14 / 14.
- Faces with available MouthOpenness measurements: 14 / 14.
- Unavailable mouth measurements: 0.
- Detections hidden by the annotation threshold or invalid boxes: 0.
- Three-face case 12.png: 3 detected faces, all with mouth measurements.
- Absolute errors in the three independent geometry checks: <= 1e-12.
- Overall synthetic demonstration status: COMPLETE.

Representative native mouth_openness outputs, dimensionless:

    Input         Face      Mouth openness
    01.png           1         0.004984
    02.png           1         0.003958
    03.png           1         0.057823
    04.png           1         0.213209
    05.png           1         0.550229
    06.png           1         0.707689
    07.png           1         0.003577
    08.png           1         0.252151
    09.png           1         0.183675
    10.png           1         0.332279
    11.png           1         0.279808
    12.png           1         0.002436
    12.png           2         0.253404
    12.png           3         0.638527

From 03.png through 06.png, the observed estimates increase as nominal
mouth opening increases. The very-slight-opening image 02.png produced a
slightly lower estimate than the fully-closed baseline 01.png (0.003958
versus 0.004984), so a strictly monotonic response over all six opening
images must not be claimed. Small differences near a closed mouth are
illustrative observations, not independently established prediction errors.

The other examples demonstrate expression-dependent lip geometry, yaw,
partial occlusion, and independent per-face results in a multi-face image.
All measurements arise from the actual recorded run; there are no
synthetic ground-truth quality labels or classification accuracy scores.

Outputs, safe reruns, and provenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The synthetic demonstration script owns the following outputs:

results/synthetic_demo/
    mouth_openness_observations.csv
    synthetic_demo_summary.json
    synthetic_demo_summary.txt
    synthetic_demo_examples.png
    annotated_images/*.png

The accepted directory contains one three-area annotated panel per input
image (twelve total) and one contact sheet displaying all twelve cases.
The CSV includes every detected face, and the JSON/TXT summaries document
per-image and total counts, original image dimensions, detection thresholds,
software environment, source and annotated-image SHA-256 digests, component
source-file hashes, and the landmark model digest. The TXT report mirrors
the JSON data.

Run from validation/mouth_openness/:

python mouth_openness_synthetic_demo.py

The script uses preflight checks, staging, staged artifact validation, and
publication with backup/rollback protection for its own synthetic_demo
directory. A successful rerun replaces that directory, including stale
artifacts from an earlier run. Other validation stages' accepted outputs
are left untouched.

The synthetic media are independent of FELT/RAVDESS and should be used for
public-facing demonstrations only when the rights and provenance of those
media permit redistribution. Record their provenance and any applicable
restrictions in VALIDATION_MEDIA_LICENSES.md. Do not publish benchmark
RAVDESS frames or FELT-annotated images without confirming the relevant
source permissions.

Reporting boundaries
~~~~~~~~~~~~~~~~~~~~
This demonstration verifies that the active software path can produce
numerically self-consistent per-face measurements under twelve chosen
synthetic visual conditions. It does not independently measure mouth-opening
accuracy, landmark localization accuracy, detector recall, temporal mouth
motion or velocity, or robustness over a statistically representative sample.
The synthetic case labels are descriptions of the generated appearances,
not measured reference values. The quantitative accuracy claims remain those
of the controlled 158286-frame FELT/RAVDESS benchmark only.


Run Order
---------
Activate the thesis environment:

conda activate PhysioTrack-v2

From the validation directory:

cd /d <project-path>\physiotrack-v2\validation\mouth_openness

Optional syntax check:

python -m py_compile felt_ravdess_mouth_openness_eval.py felt_ravdess_mouth_openness_plot.py felt_ravdess_mouth_openness_qualitative.py mouth_openness_component_test.py mouth_openness_synthetic_demo.py

Recommended clean reproduction order:

1. Quantitative preflight:

   python felt_ravdess_mouth_openness_eval.py --preflight-only

2. Full quantitative benchmark:

   python felt_ravdess_mouth_openness_eval.py

3. Independent quantitative tables and figures:

   python felt_ravdess_mouth_openness_plot.py

4. Deterministic qualitative validation:

   python felt_ravdess_mouth_openness_qualitative.py

5. Isolated component preflight:

   python mouth_openness_component_test.py --preflight-only

6. Real FaceAnalysis smoke test:

   python mouth_openness_component_test.py --smoke-test --smoke-count 3

7. Full isolated component execution:

   python mouth_openness_component_test.py

The benchmark and isolated component execution serve different purposes. The
benchmark reports ground-truth-derived agreement metrics. The isolated run
verifies the current software execution path and stores real numerical outputs
without generating a second accuracy result.

8. Independent synthetic qualitative demonstration:

   python mouth_openness_synthetic_demo.py

The standalone synthetic demonstration may be run or rerun independently
of the FELT/RAVDESS benchmark and isolated component-execution outputs.

Output Structure
----------------
The independently generated synthetic inputs are located at:

validation/synthetic_media/mouth_openness_media/

The final validation artifacts are organized as follows:

validation/mouth_openness/
|-- felt_ravdess_mouth_openness_eval.py
|-- felt_ravdess_mouth_openness_plot.py
|-- felt_ravdess_mouth_openness_qualitative.py
|-- mouth_openness_component_test.py
|-- mouth_openness_synthetic_demo.py
|-- README_MOUTH_OPENNESS.txt
`-- results/
    |-- felt_ravdess_mouth_openness_per_frame.csv
    |-- felt_ravdess_mouth_openness_per_actor.csv
    |-- felt_ravdess_mouth_openness_summary.txt
    |-- felt_ravdess_mouth_openness_thesis_table.csv
    |-- felt_ravdess_mouth_openness_thesis_table.md
    |-- felt_ravdess_mouth_openness_per_actor_thesis_table.csv
    |-- felt_ravdess_mouth_openness_per_actor_thesis_table.md
    |-- figures/
    |   |-- felt_ravdess_mouth_openness_agreement.png
    |   |-- felt_ravdess_mouth_openness_error_distribution.png
    |   |-- felt_ravdess_mouth_openness_per_actor.png
    |   `-- felt_ravdess_mouth_openness_qualitative_examples.png
    |-- qualitative/
    |   |-- felt_ravdess_mouth_openness_qualitative_selection.csv
    |   `-- annotated_images/
    |       `-- eight annotated PNG examples
    |-- component_execution/
    |   |-- mouth_openness_component_results.csv
    |   `-- mouth_openness_component_summary.json
    `-- synthetic_demo/
        |-- mouth_openness_observations.csv
        |-- synthetic_demo_summary.json
        |-- synthetic_demo_summary.txt
        |-- synthetic_demo_examples.png
        `-- annotated_images/
            `-- *.png (12 annotated image panels in the accepted run)

Reproducibility
---------------
Dataset paths are resolved from the project structure rather than from a
machine-specific absolute path. The MediaPipe face-landmarker model is resolved
through the PhysioTrack model registry.

The benchmark summary records the software versions and the SHA256 hash of the
resolved face-landmarker model. The validated run used:

OpenCV: 5.0.0
NumPy: 2.4.6
pandas: 3.0.6
MediaPipe: 1.1.0
Face landmarker SHA256:
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

The quantitative metrics are recomputed from the detailed per-frame CSV before
plot generation, and the plotting script checks consistency with both the
summary and per-actor result files. The CSV and Markdown thesis tables are
generated from the same verified in-memory tables, so their contents represent
the same accepted quantitative results. This provides an independent
consistency check between detailed outputs, aggregate metrics, tables, and
figures.

The isolated component execution additionally verifies the current FaceAnalysis
software path using the same read-only FELT/RAVDESS population, the accepted
FaceLandmarks model hash, controlled FELT FaceRects, staged result validation,
transactional output replacement, and explicit Git-safe result handling.

The separate synthetic-demonstration JSON records the active PhysioTrack
component source hashes and model hash along with the original generated
input hashes. Its accepted execution used Python 3.11.17, OpenCV 5.0.0,
NumPy 2.4.6, MediaPipe 1.1.0, and Ultralytics 8.4.177. Each standalone
stage has independent output ownership, enabling reproducible reruns without
replacing other stages' results.

Methodological Qualifications
-----------------------------
Several qualifications are important when interpreting the result.

First, the FELT reference is landmark-derived rather than a direct physical
measurement of lip separation. The benchmark therefore measures agreement
between two landmark-based geometric estimates of mouth openness.

Second, FELT and MediaPipe use different landmark schemes. Although the primary
reference was selected to match the central-height-to-mouth-width geometry of
PhysioTrack as closely as possible, the landmark coordinates are not identical
anatomical definitions.

Third, FELT FaceRect annotations are used for controlled initialization. The
reported result therefore evaluates the PhysioTrack landmark and mouth-openness
components under supplied face localization. It should not be interpreted as
an end-to-end face-detection benchmark.

Fourth, the benchmark covers the paired FELT/RAVDESS speech subset represented
by 1440 trials. It should be described specifically as validation on this
speech subset rather than as an evaluation of every trial available in the
broader source resources.

Finally, individual frames can show substantial disagreement even when the
aggregate metrics are strong. The qualitative challenging-underestimate and
challenging-overestimate examples explicitly document these failure modes.

The isolated component execution is software-path evidence, not a second
accuracy benchmark. Its availability counts, runtime, and component-result
rows must not be interpreted as additional FELT agreement metrics.

The synthetic demonstration also does not provide independently measured
mouth-opening ground truth. Nominal opening and expression cases can change
face-landmark inference, and the 01.png/02.png values are not strictly
monotonic. Its available numerical observations must not be reported as a
new prediction-accuracy percentage or substituted for the benchmark metrics.

Scientific Interpretation
-------------------------
The results demonstrate that PhysioTrack provides a stable continuous
mouth-openness estimate under the defined controlled protocol. Complete frame
availability, low aggregate error, strong Pearson and Spearman association,
and a Lin CCC above 0.93 collectively support reliable agreement with the FELT
landmark-derived reference across the speech subset.

The near-identical primary and secondary reference results strengthen the
conclusion by showing that performance is not dependent on a single narrow
inner-lip reference formulation. Per-actor and qualitative analyses further
show that the aggregate result is supported across subjects while preserving
visible examples of larger disagreement.

The scientifically appropriate description is:

controlled continuous mouth-openness validation of PhysioTrack FaceLandmarks
and MouthOpenness on the paired FELT/RAVDESS speech subset, complemented by
isolated real-pipeline MouthOpenness execution through FaceAnalysis,
accompanied by an independent synthetic qualitative illustration of the
native detector, landmark, and MouthOpenness pipeline.

Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- felt_ravdess_mouth_openness_eval.py
- felt_ravdess_mouth_openness_plot.py
- felt_ravdess_mouth_openness_qualitative.py
- mouth_openness_component_test.py
- mouth_openness_synthetic_demo.py
- README_MOUTH_OPENNESS.txt
- independently generated input images under
  validation/synthetic_media/mouth_openness_media/ (subject to documented rights)
- VALIDATION_MEDIA_LICENSES.md (source provenance and reuse conditions)
- results/felt_ravdess_mouth_openness_per_frame.csv
- results/felt_ravdess_mouth_openness_per_actor.csv
- results/felt_ravdess_mouth_openness_summary.txt
- results/felt_ravdess_mouth_openness_thesis_table.csv
- results/felt_ravdess_mouth_openness_thesis_table.md
- results/felt_ravdess_mouth_openness_per_actor_thesis_table.csv
- results/felt_ravdess_mouth_openness_per_actor_thesis_table.md
- results/figures/felt_ravdess_mouth_openness_agreement.png
- results/figures/felt_ravdess_mouth_openness_error_distribution.png
- results/figures/felt_ravdess_mouth_openness_per_actor.png
- results/qualitative/felt_ravdess_mouth_openness_qualitative_selection.csv
- results/qualitative/annotated_images/
- results/figures/felt_ravdess_mouth_openness_qualitative_examples.png
- results/component_execution/mouth_openness_component_results.csv
- results/component_execution/mouth_openness_component_summary.json
- results/synthetic_demo/mouth_openness_observations.csv
- results/synthetic_demo/synthetic_demo_summary.json
- results/synthetic_demo/synthetic_demo_summary.txt
- results/synthetic_demo/synthetic_demo_examples.png
- all accepted annotated images under results/synthetic_demo/annotated_images/

If a future isolated result exceeds the configured 90 MiB Git-safe threshold,
the single component-results CSV is replaced by the generated
mouth_openness_component_results_partNNN.csv files recorded in the summary
manifest.

Generated caches and obsolete temporary diagnostic files are not part of the
final validation deliverables.

Validation Status
-----------------
Controlled FELT/RAVDESS quantitative mouth-openness evaluation: PASS

Quantitative result tables and figures: PASS

Benchmark qualitative evidence: PASS

Isolated real PhysioTrack MouthOpenness component execution: PASS

Independent synthetic qualitative demonstration: COMPLETE
