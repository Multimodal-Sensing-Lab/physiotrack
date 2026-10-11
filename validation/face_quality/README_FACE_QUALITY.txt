PhysioTrack Face Quality Validation
====================================

Overview
--------
This package validates the PhysioTrack FaceQuality component using the 300-W
facial image dataset. The component reports three directly computed numerical
descriptors for each valid face crop:

- brightness
- sharpness
- face_area_ratio

The component also preserves face-detector confidence as an auxiliary field.
Detector confidence is not treated as an independently validated FaceQuality
descriptor because it originates from the upstream face detector.

The validation separates two forms of evidence:

1. Controlled scientific descriptor validation
   The FaceQuality implementation is evaluated under controlled image and
   face-box perturbations using fixed, ground-truth-derived face regions.

2. Isolated component execution
   The real PhysioTrack FaceAnalysis pipeline is executed with FaceQuality
   enabled and all unrelated optional face-analysis components disabled.
   This verifies real software execution, numerical export, and per-face
   association. It is not an additional accuracy benchmark.


Dataset
-------
Dataset:
300-W facial landmark evaluation set

Expected local structure:

datasets/
└── 300W/
    ├── 01_Indoor/
    │   ├── *.png
    │   └── *.pts
    └── 02_Outdoor/
        ├── *.png
        └── *.pts

Dataset coverage:

- Indoor images: 300
- Outdoor images: 300
- Total images: 600
- Annotation format: 68-point PTS files

The validation scripts treat the dataset as read-only.


FaceQuality Definitions
-----------------------
For a valid detected or controlled face box, the current FaceQuality
implementation computes:

Brightness
    The face crop is converted to grayscale and its mean intensity is divided
    by 255:

    brightness = mean(gray) / 255

    The resulting value is in the interval [0, 1].

Sharpness
    Sharpness is defined as the variance of the grayscale Laplacian:

    sharpness = variance(Laplacian(gray))

    Larger values indicate stronger local high-frequency image structure.
    This value is not normalized to a fixed upper bound.

Face area ratio
    The face-box coordinates are converted to integers, clipped to the image
    boundaries, and the clipped face area is divided by the full frame area:

    face_area_ratio =
        ((x2 - x1) * (y2 - y1)) / (image_width * image_height)

Detector confidence
    The upstream face-detector confidence is passed through by FaceQuality.
    It is retained as useful auxiliary metadata but is not presented as a new
    quality score computed by FaceQuality.


Controlled Face Initialization
------------------------------
The controlled scientific validation uses the 300-W ground-truth landmarks to
define the target face region.

For each image:

1. The tight bounding box of all 68 reference landmarks is computed.
2. The box is padded by 20 percent of its width and height.
3. The padded box is clipped to the image boundary.
4. The same controlled box is supplied to the real FaceQuality implementation.

This design isolates the descriptor behavior from face-detection accuracy and
face-selection errors.

Of the 600 controlled baseline boxes:

- 532 are fully inside the image.
- 68 touch at least one image boundary.

The face-area scale experiment uses only the 532 fully inside boxes so that
the scale response is not confounded by image-boundary clipping.


Controlled Scientific Validation
--------------------------------
The scientific evaluation is implemented by:

face_quality_eval.py

The evaluator first performs complete dataset and implementation preflight
checks. New outputs are generated in a temporary staging directory, validated,
and promoted to the final result paths only after validation succeeds.

Baseline
~~~~~~~~
The real FaceQuality component is evaluated on all 600 original images using
the controlled face boxes.

Accepted baseline descriptor distribution:

Brightness
    Mean:   0.398100
    Median: 0.390741
    Std:    0.125265
    Min:    0.076807
    Max:    0.818553

Sharpness
    Mean:   402.147187
    Median: 171.433484
    Std:    685.027773
    Min:    2.222105
    Max:    7964.856704

Face area ratio
    Mean:   0.158229
    Median: 0.072849
    Std:    0.194310
    Min:    0.001568
    Max:    1.000000

Independent recomputation of all three baseline descriptor formulas produced a
maximum absolute discrepancy of 0.0.

Brightness sensitivity
~~~~~~~~~~~~~~~~~~~~~~
Each of the 600 images is evaluated at the following intensity scale factors:

0.40, 0.60, 0.80, 1.00, 1.20, 1.40

Accepted mean brightness values:

- 0.40 -> 0.157693
- 0.60 -> 0.237313
- 0.80 -> 0.316941
- 1.00 -> 0.398100
- 1.20 -> 0.467012
- 1.40 -> 0.525849

All 600/600 images show a strictly increasing brightness response.

Exact linear proportionality is not required at the highest factors because
8-bit pixel values are clipped at 255.

Sharpness sensitivity
~~~~~~~~~~~~~~~~~~~~~
Each of the 600 images is evaluated using Gaussian blur kernel sizes:

1, 3, 5, 9, 15, 25

Kernel size 1 represents the unblurred reference image.

Accepted mean sharpness values:

- 1  -> 402.147187
- 3  -> 62.284649
- 5  -> 30.070450
- 9  -> 10.004285
- 15 -> 4.684518
- 25 -> 2.804621

All 600/600 images show a strictly decreasing sharpness response.

Face-area-ratio sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the 532 fully inside controlled face boxes, the face box is scaled about
its center using:

0.50, 0.75, 1.00

Accepted mean face-area-ratio values:

- 0.50 -> 0.029307
- 0.75 -> 0.065933
- 1.00 -> 0.117215

All 532/532 images show a strictly increasing face-area-ratio response.

The maximum absolute error between the FaceQuality output and the independently
computed integer-clipped geometric formula is 0.0.

The detailed scientific result table contains 9,396 rows.


Aggregate Metrics and Quantitative Figures
------------------------------------------
The aggregate analysis is implemented by:

face_quality_plot.py

This script does not rerun FaceQuality inference. It reads the accepted detailed
result table, independently validates its expected structure and monotonic
responses, and generates compact aggregate metrics and quantitative figures.

Outputs:

results/face_quality_metrics.csv

results/figures/face_quality_brightness_sensitivity.png
results/figures/face_quality_sharpness_sensitivity.png
results/figures/face_quality_face_area_ratio_sensitivity.png

The metrics table contains:

- 3 baseline descriptor rows
- 6 brightness sensitivity rows
- 6 sharpness sensitivity rows
- 3 face-area-ratio sensitivity rows

Total aggregate rows: 18

The accepted monotonic rates are 1.0 for all three controlled sensitivity
experiments.


Qualitative Evidence
--------------------
The qualitative evidence is implemented by:

face_quality_qualitative.py

Three deterministic fully inside samples are selected from the accepted
baseline results:

- Outdoor/outdoor_184.png
- Outdoor/outdoor_284.png
- Outdoor/outdoor_294.png

For each sample, the script produces visual evidence for:

Brightness
    0.40, 1.00, 1.40

Sharpness
    Gaussian kernels 1, 9, 25

Face area ratio
    Box scales 0.50, 0.75, 1.00

The numerical values displayed in the qualitative outputs are generated by the
real FaceQuality component.

The qualitative package contains:

- 27 individual annotated images
- 27 numerical qualitative CSV rows
- one combined qualitative figure

Outputs:

results/qualitative/face_quality_qualitative_results.csv
results/qualitative/sample_*.png
results/figures/face_quality_qualitative_examples.png

The qualitative evidence supports interpretation of the controlled numerical
results but does not replace the complete quantitative evaluation.


Isolated PhysioTrack Component Execution
----------------------------------------
The isolated execution test is implemented by:

face_quality_component_test.py

This test executes the real PhysioTrack FaceAnalysis path.

Active components:

- Face detector
- FaceQuality

Disabled optional components:

- tracking
- head_pose
- landmarks
- eyes
- blink
- gaze
- gaze_estimation
- mouth
- mouth_motion
- emotion
- regions
- temporal

The face detector is retained because FaceQuality operates on detected face
instances in the normal FaceAnalysis pipeline.

The isolated component test supports:

Preflight only:

python face_quality_component_test.py --preflight-only

Deterministic smoke test:

python face_quality_component_test.py --smoke-test

Full execution:

python face_quality_component_test.py

Accepted full execution results:

- Images processed: 600
- Total output rows: 7,789
- Detected face rows: 7,788
- Quality-available rows: 7,788
- Images with quality output: 599
- NO_FACE images: 1
- QUALITY_UNAVAILABLE rows: 0
- INVALID_FACE_BOX rows: 0
- EXECUTION_FAILED rows: 0

All 7,788 detected face instances produced numerical FaceQuality output.

The isolated test independently recomputes the descriptor formulas from the
detected face boxes and input images.

Accepted maximum absolute discrepancies:

- Brightness: 0.0
- Sharpness: 0.0
- Face area ratio: 0.0
- Passed-through detector confidence: 0.0

The 300-W dataset inventory was unchanged by the execution.

The 7,788 detections must be interpreted as detected face instances produced
by the active detector, not as a ground-truth count of true faces in 300-W.
This stage verifies operational FaceQuality execution and export rather than
face-detection accuracy.


Safe Rerun Design
-----------------
The package follows a transactional safe-rerun pattern:

complete preflight
-> temporary/staging generation
-> validation of new outputs
-> replacement of script-owned final outputs

Previously valid outputs are not removed before their replacements have passed
the required validation checks.

If preflight, execution, output validation, or final promotion fails, previously
accepted outputs owned by that script are preserved whenever applicable, and
temporary staging data are cleaned.

No validation script writes to or modifies the 300-W dataset.


Output Ownership
----------------
face_quality_eval.py
    Owns:
    - results/face_quality_results.csv
    - results/face_quality_summary.txt

face_quality_plot.py
    Owns:
    - results/face_quality_metrics.csv
    - results/figures/face_quality_brightness_sensitivity.png
    - results/figures/face_quality_sharpness_sensitivity.png
    - results/figures/face_quality_face_area_ratio_sensitivity.png

face_quality_qualitative.py
    Owns:
    - results/qualitative/
    - results/figures/face_quality_qualitative_examples.png

face_quality_component_test.py
    Owns:
    - results/component_execution/face_quality_component_results.csv
    - results/component_execution/face_quality_component_summary.json

No script is intended to delete or replace outputs owned by another stage.


Recommended Execution Order
---------------------------
From:

validation/face_quality/

run:

1. Controlled scientific evaluation

   python face_quality_eval.py

2. Aggregate table and quantitative figures

   python face_quality_plot.py

3. Qualitative evidence

   python face_quality_qualitative.py

4. Isolated component preflight

   python face_quality_component_test.py --preflight-only

5. Isolated component smoke test

   python face_quality_component_test.py --smoke-test

6. Full isolated component execution

   python face_quality_component_test.py

The smoke test does not replace the accepted full component-execution outputs.


Result Structure
----------------
validation/face_quality/
├── face_quality_eval.py
├── face_quality_plot.py
├── face_quality_qualitative.py
├── face_quality_component_test.py
├── README_FACE_QUALITY.txt
└── results/
    ├── face_quality_results.csv
    ├── face_quality_summary.txt
    ├── face_quality_metrics.csv
    ├── figures/
    │   ├── face_quality_brightness_sensitivity.png
    │   ├── face_quality_sharpness_sensitivity.png
    │   ├── face_quality_face_area_ratio_sensitivity.png
    │   └── face_quality_qualitative_examples.png
    ├── qualitative/
    │   ├── face_quality_qualitative_results.csv
    │   └── sample_*.png
    └── component_execution/
        ├── face_quality_component_results.csv
        └── face_quality_component_summary.json


Interpretation and Reporting Boundaries
---------------------------------------
The controlled study validates the numerical correctness and expected
sensitivity behavior of the current FaceQuality descriptors.

The experiment is not a conventional supervised face-image-quality benchmark
with a single human-annotated quality target. The three validated descriptors
have distinct physical or image-processing definitions and are therefore
evaluated using controlled perturbations appropriate to each definition.

Important reporting boundaries are:

- Brightness validation demonstrates response to controlled intensity changes.
- Sharpness validation demonstrates response to controlled Gaussian blur.
- Face-area-ratio validation demonstrates exact geometric behavior under
  controlled box scaling.
- Detector confidence is auxiliary upstream metadata.
- The controlled ground-truth-derived face box removes detector accuracy from
  the scientific descriptor experiment.
- The isolated component execution separately demonstrates real operation
  through FaceAnalysis with the face detector enabled.
- Qualitative examples complement but do not replace the full quantitative
  evaluation.
- The results should not be described as an official 300-W quality benchmark
  or leaderboard result.
- Isolated component execution PASS is software evidence and must not be
  reported as a predictive-accuracy score.
- The results characterize the current PhysioTrack FaceQuality implementation
  under the documented protocol and do not imply universal performance on all
  imaging conditions or populations.


Final Reproducibility Artifacts
-------------------------------
Preserve:

- face_quality_eval.py
- face_quality_plot.py
- face_quality_qualitative.py
- face_quality_component_test.py
- README_FACE_QUALITY.txt
- results/face_quality_results.csv
- results/face_quality_summary.txt
- results/face_quality_metrics.csv
- results/figures/face_quality_brightness_sensitivity.png
- results/figures/face_quality_sharpness_sensitivity.png
- results/figures/face_quality_face_area_ratio_sensitivity.png
- results/figures/face_quality_qualitative_examples.png
- results/qualitative/face_quality_qualitative_results.csv
- all accepted qualitative sample images under results/qualitative/
- results/component_execution/face_quality_component_results.csv
- results/component_execution/face_quality_component_summary.json

Console evidence worth preserving includes:

- successful full controlled evaluator staging validation and commit
- successful plot-stage validation and commit
- successful qualitative-stage validation and commit
- successful isolated component preflight
- successful isolated component smoke test
- successful full isolated component staging validation and commit


Validation Status
-----------------
Controlled FaceQuality descriptor validation: PASS

Aggregate metric and quantitative-figure generation: PASS

Qualitative validation evidence: PASS

Isolated real PhysioTrack FaceQuality execution: PASS

Final component closure requires the accompanying internal academic
documentation and final package audit.
