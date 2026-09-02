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
    thesis-oriented result tables, and creates quantitative figures from the
    saved benchmark outputs.

felt_ravdess_mouth_openness_qualitative.py
    Selects deterministic benchmark examples from the accepted quantitative
    result set, re-evaluates only those selected frames, and generates
    annotated qualitative examples.

Datasets and Scope
------------------
The evaluation uses two paired resources:

FELT speech annotations:

datasets/FELT/raw_motion_speech/

RAVDESS speech videos:

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
The plotting script generates three complementary figures:

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

Run Order
---------
Run the scripts from the repository root using the project environment:

python validation/mouth_openness/felt_ravdess_mouth_openness_eval.py
python validation/mouth_openness/felt_ravdess_mouth_openness_plot.py
python validation/mouth_openness/felt_ravdess_mouth_openness_qualitative.py

The evaluator regenerates only its quantitative result files. The plotting
script regenerates only its quantitative tables and figures. The qualitative
script regenerates only its qualitative outputs and combined qualitative
figure. Each script therefore owns and cleans only the artifacts that it
creates.

Output Structure
----------------
The final validation artifacts are organized as follows:

validation/mouth_openness/
|-- felt_ravdess_mouth_openness_eval.py
|-- felt_ravdess_mouth_openness_plot.py
|-- felt_ravdess_mouth_openness_qualitative.py
|-- README_MOUTH_OPENNESS.txt
`-- results/
    |-- felt_ravdess_mouth_openness_per_frame.csv
    |-- felt_ravdess_mouth_openness_per_actor.csv
    |-- felt_ravdess_mouth_openness_summary.txt
    |-- felt_ravdess_mouth_openness_thesis_table.csv
    |-- felt_ravdess_mouth_openness_per_actor_thesis_table.csv
    |-- figures/
    |   |-- felt_ravdess_mouth_openness_agreement.png
    |   |-- felt_ravdess_mouth_openness_error_distribution.png
    |   |-- felt_ravdess_mouth_openness_per_actor.png
    |   `-- felt_ravdess_mouth_openness_qualitative_examples.png
    `-- qualitative/
        |-- felt_ravdess_mouth_openness_qualitative_selection.csv
        `-- annotated_images/
            `-- eight annotated PNG examples

Reproducibility
---------------
Dataset paths are resolved from the project structure rather than from a
machine-specific absolute path. The MediaPipe face-landmarker model is resolved
through the PhysioTrack model registry.

The benchmark summary records the software versions and the SHA256 hash of the
resolved face-landmarker model. The validated run used:

OpenCV: 5.0.0
NumPy: 2.4.4
pandas: 2.3.3
MediaPipe: 1.0.0
Face landmarker SHA256:
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

The quantitative metrics are recomputed from the detailed per-frame CSV before
plot generation, and the plotting script checks consistency with both the
summary and per-actor result files. This provides an independent consistency
check between detailed outputs, aggregate metrics, tables, and figures.

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
and MouthOpenness on the paired FELT/RAVDESS speech subset.
