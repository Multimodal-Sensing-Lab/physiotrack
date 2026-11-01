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

A separate synthetic qualitative demonstration illustrates FaceQuality outputs on
independently generated media. It is not part of the supervised 300-W evaluation
and is not an additional accuracy or image-quality classification benchmark.


Dataset
-------
Dataset:
300-W facial landmark evaluation set

Original dataset publications (as requested by the official 300-W source):

- Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., &
  Pantic, M. (2016). 300 Faces In-The-Wild Challenge: Database and Results.
  Image and Vision Computing, 47, 3-18.
  https://doi.org/10.1016/j.imavis.2016.01.002

- Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013).
  300 Faces in-the-Wild Challenge: The First Facial Landmark Localization
  Challenge. Proceedings of the IEEE International Conference on Computer
  Vision Workshops (ICCVW), 397-403.
  https://doi.org/10.1109/ICCVW.2013.59

- Sagonas, C., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2013).
  A Semi-automatic Methodology for Facial Landmark Annotation.
  Proceedings of the IEEE Conference on Computer Vision and Pattern
  Recognition Workshops (CVPRW), 896-903.
  https://doi.org/10.1109/CVPRW.2013.132

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

The 300-W dataset is resolved relative to the repository parent directory as
`datasets/300W/`. Its original images and derived annotated examples are not
included in the public source-code distribution. Reproduction requires separately
obtained dataset files under their applicable access and usage conditions.


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

Figure interpretation:

- The plotted points are dataset means at each perturbation level.
- Vertical error bars show the mean plus or minus one sample standard deviation
  (Mean ± Standard Deviation, or mean ± SD), computed across the images.
- These descriptive bars are neither observed min/max ranges nor confidence
  intervals. They should be identified as mean ± SD in thesis figure captions.
- Brightness, sharpness, and face_area_ratio are nonnegative by definition.
  On the sharpness and face-area-ratio figures, a symmetric mean - SD lower
  bound can fall below zero when between-image dispersion exceeds the mean.
  This is a graphical property of the error bars, not a negative measured or
  computed FaceQuality value; the observed descriptor values are nonnegative.


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

Because these figures contain original or derived 300-W benchmark imagery,
they are retained in the local validation results and are not distributed
through the public GitHub repository. Any selected use in a thesis remains
subject to the dataset conditions and appropriate original-paper citations.


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


Synthetic Qualitative Demonstration
-----------------------------------
Purpose and scope
~~~~~~~~~~~~~~~~~
The standalone illustrative run is implemented by:

face_quality_synthetic_demo.py

It executes the current repository's real PhysioTrack Face detector and
FaceQuality through FaceAnalysis on independently generated synthetic images.
It documents per-face software outputs without using 300-W annotations,
quality ground-truth labels, or benchmark quality-accuracy scores.

Inputs and test cases
~~~~~~~~~~~~~~~~~~~~~
The script recursively discovers supported image files in:

validation/synthetic_media/face_quality_media/

The accepted 12-image demonstration includes:

- 01.png: frontal baseline, ordinary illumination.
- 02.png: low illumination.
- 03.png: high illumination.
- 04.png: moderate image blur.
- 05.png: strong image blur.
- 06.png: small apparent face size.
- 07.png: medium apparent face size.
- 08.png: large apparent face size.
- 09.png: partial occlusion by a hand.
- 10.png: rotated head.
- 11.png: three faces at different apparent sizes.
- 12.png: background scene with no face.

Synthetic media provenance and usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The twelve synthetic input images (01.png through 12.png) were generated using
OpenAI ChatGPT for this independent qualitative demonstration. These images
are artificial examples, not photographs from the 300-W dataset, and their
nominal test conditions are not independently verified ground-truth labels.

OpenAI, ChatGPT. Terms of Use:
https://openai.com/policies/terms-of-use/

The provenance and applicable distribution conditions for the generated media
are documented separately in VALIDATION_MEDIA_LICENSES.md. No CC0 or Blender
license is attributed to these ChatGPT-generated images.

The script reads each image at its original resolution and does not require
a common fixed image size. In this run, 11 inputs were 1254 x 1254 pixels;
07.png was 1254 x 1139 pixels. Lighting, blur, and apparent size are nominal
synthetic conditions rather than physically calibrated ground truth.

Execution configuration
~~~~~~~~~~~~~~~~~~~~~~~
The active execution path is the real face detector -> FaceAnalysis ->
FaceQuality, with all unrelated optional FaceAnalysis components disabled,
including tracking, landmarks, head pose, emotion, and temporal aggregation.
The script executes on CPU with detector confidence 0.001, NMS IoU 0.45,
maximum detections 10000, and a visualization-only confidence threshold of
0.25. The lower detector threshold retains raw model observations, including
low-confidence detections; the display threshold does not filter the CSV data.

For every detected face with valid output, the script exports detector
confidence (auxiliary metadata), brightness, Laplacian-variance sharpness,
and face-area ratio. It independently recomputes the descriptor formulas
using the detected, integer-clipped face box and verifies the numerical
values and passed-through detector confidence within a tolerance of 1e-12.
No composite quality score or quality class is predicted or inferred.

Accepted demonstration results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Images discovered and processed: 12/12.
- Image execution failures: 0.
- Raw detector observations and CSV rows: 15.
- Observations with FaceQuality descriptors: 15/15.
- Invalid face crops and unavailable quality observations: 0.
- Detections displayed at confidence >= 0.25: 13.
- Raw detections excluded only from visualization: 2.
- Multiple-face input 11.png: 3 faces with available descriptors.
- No-face input 12.png: 0 detections and 0 FaceQuality observations.
- Maximum recorded absolute difference for all four numerical checks: 0.0.

The two extra low-confidence detections occurred in strong-blur input 05.png
(detector confidences approximately 0.03247 and 0.00323). Both are preserved
in the observations CSV but excluded from the corresponding annotated panel
under the documented visualization threshold. The raw detector output count
for this input is three, not one.

Illustrative observations for the primary displayed face include:

- Brightness (02.png, 01.png, 03.png): 0.28154, 0.80549, 0.94351.
- Sharpness (01.png, 04.png, 05.png): 34.94, 3.59, 3.17.
- Face-area ratio (06.png, 07.png, 08.png): 0.04751, 0.13174, 0.40613.

These observed values illustrate changes under the chosen nominal input
conditions. Unlike the controlled 300-W experiment, the synthetic examples
do not isolate every imaging or geometric factor, and the face box is
predicted rather than supplied from annotated landmarks. They must not be
interpreted as calibrated monotonicity tests, detection recall, or estimates
of human-rated image quality.

Outputs and reproducibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The script exclusively owns:

results/synthetic_demo/
    face_quality_observations.csv
    synthetic_demo_summary.json
    synthetic_demo_summary.txt
    synthetic_demo_examples.png
    annotated_images/*.png

For the accepted run, annotated_images/ contains 12 side-by-side image panels,
and the contact sheet shows all 12 inputs. The CSV includes raw per-face
descriptors and independent numerical checks. JSON and TXT record image
counts, device and thresholds, source/output SHA-256 values, component source
file hashes, package versions, and observed execution status. The text
summary mirrors the machine-readable JSON information.

Run from validation/face_quality/:

python face_quality_synthetic_demo.py

A successful rerun publishes a fully validated replacement of the entire
results/synthetic_demo/ directory, removing stale files from earlier runs.
A failed rerun retains or restores the previously accepted directory
whenever possible. The scientific evaluator, plotting, benchmark qualitative,
and component-execution results are not replaced by this script.

The synthetic inputs and output panels are intended for independently
reproducible qualitative illustration. Record the actual media provenance
and applicable redistribution permissions in VALIDATION_MEDIA_LICENSES.md
before distributing these assets on GitHub or in an academic publication.

Interpretation and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The demonstration shows direct per-face brightness, sharpness, and area-ratio
measurements under varied generated-image conditions, including occlusion,
pose, multi-face scenes, and a no-face input. It is independent of the 300-W
controlled validation and does not establish accuracy, model generalization,
face-detection recall, or a benchmark-quality pass rate. Detector confidence
is upstream detection metadata, not an image-quality score.


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
The standalone synthetic script applies the same pre-publication validation
principle at the directory level. It replaces only results/synthetic_demo/;
accepted scientific and component-execution artifacts remain independent.


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

face_quality_synthetic_demo.py
    Owns:
        - results/synthetic_demo/ (entire directory, including its reports,
          contact sheet, and annotated_images/)

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

7. Independent synthetic qualitative demonstration

   python face_quality_synthetic_demo.py

The synthetic run may be repeated independently of the 300-W benchmark.


Result Structure
----------------
Synthetic source media are provided separately at:

validation/synthetic_media/face_quality_media/

validation/face_quality/
├── face_quality_eval.py
├── face_quality_plot.py
├── face_quality_qualitative.py
├── face_quality_component_test.py
├── face_quality_synthetic_demo.py
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
    ├── component_execution/
    │   ├── face_quality_component_results.csv
    │   └── face_quality_component_summary.json
    └── synthetic_demo/
        ├── face_quality_observations.csv
        ├── synthetic_demo_summary.json
        ├── synthetic_demo_summary.txt
        ├── synthetic_demo_examples.png
        └── annotated_images/
            └── *.png (12 side-by-side panels in the accepted run)


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
- The separately generated synthetic qualitative demonstration supplements
  software-output interpretation but does not provide ground-truth quality
  labels, benchmark accuracy, or an independently controlled image study.
- The synthetic images' descriptive test conditions are nominal; reported
  detector confidence is not a quality-accuracy or quality-confidence score.
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
- face_quality_synthetic_demo.py
- independently generated synthetic input images under
  validation/synthetic_media/face_quality_media/ (subject to documented rights)
- VALIDATION_MEDIA_LICENSES.md (media provenance and distribution conditions)
- results/face_quality_results.csv
- results/face_quality_summary.txt
- results/face_quality_metrics.csv
- results/figures/face_quality_brightness_sensitivity.png
- results/figures/face_quality_sharpness_sensitivity.png
- results/figures/face_quality_face_area_ratio_sensitivity.png
- results/figures/face_quality_qualitative_examples.png
- results/qualitative/face_quality_qualitative_results.csv
- all accepted qualitative sample images under results/qualitative/
- results/synthetic_demo/face_quality_observations.csv
- results/synthetic_demo/synthetic_demo_summary.json
- results/synthetic_demo/synthetic_demo_summary.txt
- results/synthetic_demo/synthetic_demo_examples.png
- all accepted annotated images under results/synthetic_demo/annotated_images/
- results/component_execution/face_quality_component_results.csv
- results/component_execution/face_quality_component_summary.json



Validation Status
-----------------
Controlled FaceQuality descriptor validation: PASS

Aggregate metric and quantitative-figure generation: PASS

Qualitative validation evidence: PASS

Isolated real PhysioTrack FaceQuality execution: PASS

Independent synthetic qualitative demonstration: COMPLETE


Downloading and Preparing Benchmark Media
-----------------------------------------
The 300-W source images and derived annotated benchmark examples
are not distributed with the public GitHub repository.

Obtain the 300-W evaluation dataset from:
https://ibug.doc.ic.ac.uk/resources/300-W/

Place the files in the workspace's datasets/300W/ directory,
alongside the physiotrack-v2 repository:

- 01_Indoor/  (PNG images and matching PTS annotations)
- 02_Outdoor/ (PNG images and matching PTS annotations)

The accepted evaluation uses 600 images in total. After preparing
the dataset, follow the validation commands documented above.

Benchmark qualitative images are generated locally and must not
be uploaded to GitHub. Numerical results, independent quantitative
plots, and separately generated synthetic demonstration media are
not subject to this benchmark-image exclusion.

See VALIDATION_MEDIA_LICENSES.md for media distribution conditions.
