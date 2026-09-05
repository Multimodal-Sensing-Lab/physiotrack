300-W Facial Landmark Validation
================================

Overview
--------
This directory contains the reproducible Face Landmarks validation package for
PhysioTrack.

The package provides two complementary forms of evidence:

1. Scientific benchmark validation on the 300-W facial landmark dataset.
2. Isolated execution of the real PhysioTrack Face Landmarks component through
   the project FaceAnalysis path, with unrelated face-analysis components
   disabled.

The scientific benchmark measures landmark-localization accuracy under a
controlled face-initialization protocol. The isolated component execution
verifies that the real PhysioTrack Face Landmarks path runs reproducibly and
exports its native numerical landmark outputs in a structured table.

These procedures answer different questions and must be interpreted
separately. The isolated component execution is software execution evidence and
does not replace the scientific NME benchmark.

Dataset
-------
Official source:

https://ibug.doc.ic.ac.uk/resources/300-W/

Required dataset structure:

datasets/
└── 300W/
    ├── 01_Indoor/
    │   ├── indoor_001.png
    │   ├── indoor_001.pts
    │   └── ...
    └── 02_Outdoor/
        ├── outdoor_001.png
        ├── outdoor_001.pts
        └── ...

The validation uses:

- Indoor images: 300
- Outdoor images: 300
- Total images: 600
- Annotation format: 68-point PTS files

Each image must have a corresponding PTS annotation file.

The optional 300-W bounding-box MAT files are not required by this evaluation.

The dataset directory is treated as read-only benchmark input. Generated
validation outputs are written only under the Face Landmarks validation
directory.

Validation Files
----------------
The validation implementation is located under:

validation/face_landmarks/

Files:

- 300w_landmark_eval.py
- mediapipe_300w_mapping.py
- 300w_landmark_table.py
- 300w_landmark_plot.py
- 300w_landmark_qualitative.py
- face_landmarks_component_test.py

300w_landmark_eval.py
    Runs the scientific 300-W landmark-localization benchmark using the
    controlled 51-point protocol and writes the detailed result CSV and summary.

mediapipe_300w_mapping.py
    Defines the fixed anatomical correspondence between the evaluated 300-W
    landmarks and the selected MediaPipe Face Landmarker outputs.

300w_landmark_table.py
    Reads the accepted evaluator result CSV and produces compact CSV and
    Markdown thesis tables.

300w_landmark_plot.py
    Reads the accepted evaluator result CSV and produces the cumulative error
    distribution data and final CED figure.

300w_landmark_qualitative.py
    Re-evaluates eight deterministic benchmark cases and produces qualitative
    evidence only after the selected cases reproduce their accepted benchmark
    statuses and NME values.

face_landmarks_component_test.py
    Runs the real PhysioTrack Face Landmarks implementation through the
    FaceAnalysis project path, using the same controlled GT-derived face box for
    input localization while disabling unrelated face-analysis components. The
    script exports all native MediaPipe landmarks numerically.

Model
-----
The evaluation uses the MediaPipe model:

face_landmarker.task

The evaluator resolves the model automatically from supported local project
locations.

An explicit model path can also be provided through:

PHYSIOTRACK_FACE_LANDMARKER_MODEL

Scientific Benchmark Protocol
-----------------------------
The benchmark uses the 51-point 300-W landmark protocol.

The original 68-point 300-W annotations are reduced to 51 evaluated landmarks
by excluding the 17 face-border landmarks.

A fixed anatomical mapping selects the corresponding 51 MediaPipe landmarks
from the native 478-landmark prediction.

For controlled landmark-localization evaluation, the face region is initialized
from the ground-truth 68-point annotation:

1. Compute the tight bounding box around the 68 ground-truth points.
2. Add 20 percent padding relative to the ground-truth landmark bounding-box
   width and height.
3. Pass the resulting controlled face region to:

   FaceLandmarks.predict_face()

This design isolates landmark localization from face detection and face
selection. It is therefore a controlled component benchmark rather than an
end-to-end face-detection-plus-landmark benchmark.

Normalized Mean Error
---------------------
For every successful prediction, the 51 predicted landmarks are compared with
the corresponding 51 ground-truth landmarks.

For landmark i, the Euclidean localization error is:

e_i = sqrt((x_i(pred) - x_i(gt))^2 + (y_i(pred) - y_i(gt))^2)

The mean pixel error is:

Mean Pixel Error = (1 / 51) * sum(e_i)

The normalization distance is the inter-ocular distance between the outer eye
corners of the 51-point ground-truth configuration.

Normalized Mean Error is:

NME = Mean Pixel Error / Inter-ocular Distance

and the reported percentage is:

NME (%) = NME * 100

Failed landmark predictions are retained as failed detections and are not
assigned fabricated NME values.

Benchmark Metrics
-----------------
The benchmark reports:

- Number of images
- Number of successful landmark predictions
- Number of failed landmark predictions
- Detection rate
- Mean NME
- Median NME
- Standard deviation of NME
- Cumulative Error Distribution (CED)

The CED at threshold t is:

CED(t) = number of benchmark images with NME <= t / total benchmark images

The denominator includes all 600 benchmark images. Therefore, failed landmark
predictions remain unsuccessful at every CED threshold and the CED curve is not
required to reach 100 percent.

Scientific Benchmark Run Order
------------------------------
Activate the PhysioTrack environment:

conda activate PhysioTrack-Thesis

Open:

physiotrack/validation/face_landmarks

Run:

python 300w_landmark_eval.py
python 300w_landmark_table.py
python 300w_landmark_plot.py
python 300w_landmark_qualitative.py

The evaluator is the primary scientific computation stage.

The table and plot scripts consume the accepted evaluator result CSV.

The qualitative script verifies selected benchmark cases against the accepted
quantitative result before final qualitative output replacement.

Validated Quantitative Results
------------------------------
The evaluation covers all 600 images.

Indoor:
- Images: 300
- Successful predictions: 292
- Failed predictions: 8
- Detection rate: 97.33%
- Mean NME: 4.6988%
- Median NME: 4.3147%
- NME standard deviation: 2.2056%

Outdoor:
- Images: 300
- Successful predictions: 296
- Failed predictions: 4
- Detection rate: 98.67%
- Mean NME: 4.6024%
- Median NME: 4.3228%
- NME standard deviation: 1.2912%

Overall:
- Images: 600
- Successful predictions: 588
- Failed predictions: 12
- Detection rate: 98.00%
- Mean NME: 4.6503%
- Median NME: 4.3178%
- NME standard deviation: 1.8048%

Validated Overall CED Values
----------------------------
- NME <= 5%: 72.6667%
- NME <= 6%: 88.3333%
- NME <= 8%: 95.8333%
- NME <= 10%: 96.6667%

The CED output uses 1,001 thresholds from 0.00% through 10.00% in 0.01%
increments.

The Indoor, Outdoor, and Overall CED curves are monotonic and all fractions
remain within [0, 1].

Result Interpretation
---------------------
The overall 98.00% successful-prediction rate shows that the controlled
PhysioTrack Face Landmarks path returns landmark predictions for the large
majority of 300-W evaluation images.

The overall Mean NME of 4.6503% and Median NME of 4.3178% indicate that most
successful predictions are relatively close to the reference configuration
under the documented 51-point anatomical mapping and inter-ocular
normalization.

The result distribution contains a small number of difficult outliers. For
example, some challenging images produce substantially larger NME values while
still returning a valid landmark prediction. These cases are retained because
they represent genuine localization limitations rather than execution errors.

The twelve failed predictions are also retained explicitly and are not hidden
or assigned artificial accuracy values.

Qualitative Evidence
--------------------
The qualitative stage complements the full 600-image benchmark with eight
deterministic 300-W examples.

The selected roles are:

- strong_indoor
- representative_indoor
- challenging_indoor
- failed_indoor
- strong_outdoor
- representative_outdoor
- challenging_outdoor
- failed_outdoor

The accepted selected cases are:

strong_indoor
    Image: indoor_261.png
    Accepted NME: 3.1141%

representative_indoor
    Image: indoor_166.png
    Accepted NME: 4.2944%

challenging_indoor
    Image: indoor_045.png
    Accepted NME: 29.3506%

failed_indoor
    Image: indoor_008.png
    Accepted status: failed_detection

strong_outdoor
    Image: outdoor_059.png
    Accepted NME: 2.5983%

representative_outdoor
    Image: outdoor_036.png
    Accepted NME: 4.3683%

challenging_outdoor
    Image: outdoor_274.png
    Accepted NME: 7.7456%

failed_outdoor
    Image: outdoor_057.png
    Accepted status: failed_detection

For successful cases, the qualitative script reproduces the accepted NME before
final output generation.

For failed cases, no prediction landmarks or artificial NME values are created.
Ground-truth points remain visible so that the failed output can be interpreted
without misrepresenting it as a successful prediction.

The qualitative visualizations show:

- Ground-truth 51-point landmarks
- Predicted mapped landmarks for successful cases
- Point-wise error vectors
- Controlled evaluation face box
- Accepted full-benchmark summary values
- Case-specific status and NME information

The combined qualitative figure is written to:

results/figures/300w_landmark_qualitative_examples.png

Safe Rerun Design
-----------------
The validation package follows the safe-rerun pattern:

preflight
-> temporary/staging generation
-> validation of newly generated outputs
-> replacement of script-owned final outputs

Previously valid final evidence is not removed before the newly generated
replacement has passed the relevant validation checks.

If a required dataset file, accepted input result, model, or other critical
dependency is unavailable or invalid, the affected script stops without
deleting previously valid final artifacts.

Each script owns only its own outputs.

Output Ownership
----------------
300w_landmark_eval.py
    Owns:
    - results/300w_landmark_results.csv
    - results/300w_landmark_summary.txt

300w_landmark_table.py
    Owns:
    - results/300w_landmark_thesis_table.csv
    - results/300w_landmark_thesis_table.md

300w_landmark_plot.py
    Owns:
    - results/300w_landmark_ced.csv
    - results/figures/300w_landmark_ced.png

300w_landmark_qualitative.py
    Owns:
    - results/qualitative/
    - results/figures/300w_landmark_qualitative_examples.png

face_landmarks_component_test.py
    Owns:
    - results/component_execution/face_landmarks_component_results.csv
    - results/component_execution/face_landmarks_component_summary.json

No validation script is intended to replace another script's accepted final
outputs.

Isolated PhysioTrack Face Landmarks Execution
---------------------------------------------
Run:

python face_landmarks_component_test.py

Purpose:

Verify that the real PhysioTrack Face Landmarks path runs independently inside
the project and exports real numerical landmark outputs.

The isolated component execution uses:

- Pipeline: PhysioTrack FaceAnalysis
- Target component: FaceLandmarks
- Model: face_landmarker.task
- Device: CPU
- Tracking: disabled
- Unrelated face-analysis components: disabled
- Dataset: complete 300-W evaluation set
- Images: 600
- Controlled input face box: GT-derived 68-point bounding box with 20% padding

The ground-truth annotations are used only to define the controlled face input
box and dataset coverage.

The isolated component test does not compute landmark-accuracy metrics.

Native Landmark Output
----------------------
The real MediaPipe Face Landmarker output contains 478 native landmarks for
each successful image.

The isolated component CSV records each native landmark separately.

Main numerical output:

results/component_execution/face_landmarks_component_results.csv

Execution summary:

results/component_execution/face_landmarks_component_summary.json

The structured result CSV includes, among other execution fields:

- split
- image
- image dimensions
- controlled face-box coordinates
- face-box width, height, and area
- landmark index
- normalized x coordinate
- normalized y coordinate
- normalized z coordinate
- pixel x coordinate
- pixel y coordinate
- pipeline landmark availability
- pipeline landmark count
- within-image-bounds status
- execution status
- failure reason

For each successful image:

- expected native landmark count: 478
- landmark_index range: 0 through 477
- every landmark index occurs once
- pixel coordinates are derived from the raw normalized output using the image
  dimensions

The coordinate conversion is:

x_pixel = x_normalized * image_width

y_pixel = y_normalized * image_height

No clipping is applied to raw normalized landmarks.

Out-of-frame landmark coordinates are therefore retained as real model output
instead of being silently corrected.

Images that complete successfully but return no landmarks are represented by a
single:

status = NO_LANDMARKS

row with no fabricated landmark coordinates.

Execution failures are represented separately from NO_LANDMARKS cases.

Validated Isolated Component Results
------------------------------------
Expected images:
600

Processed images:
600

Images with landmarks:
588

Images without landmarks:
12

Execution-failed images:
0

Expected native landmarks per successful image:
478

Native landmark observations:
281,064

Out-of-frame landmark observations:
74

Result rows:
281,076

Overall status:
PASS

Split-level execution results:

Indoor:
- Expected images: 300
- Processed images: 300
- Images with landmarks: 292
- Images without landmarks: 8
- Execution-failed images: 0
- Native landmark observations: 139,576
- Out-of-frame landmark observations: 51
- Result rows: 139,584

Outdoor:
- Expected images: 300
- Processed images: 300
- Images with landmarks: 296
- Images without landmarks: 4
- Execution-failed images: 0
- Native landmark observations: 141,488
- Out-of-frame landmark observations: 23
- Result rows: 141,492

The total row accounting is:

588 successful images * 478 landmarks
= 281,064 landmark observations

281,064 landmark observations
+ 12 NO_LANDMARKS records
= 281,076 result rows

All 600 expected images are represented in the result CSV.

The successful-image landmark structure was verified as follows:

- every successful image contains exactly 478 landmark rows
- landmark_index is unique within each image
- landmark_index spans 0 through 477
- no successful landmark coordinate contains NaN or Inf
- face-box geometry is internally consistent
- normalized-to-pixel coordinate conversion is internally consistent
- within-image-bounds flags match the raw coordinates
- no duplicate landmark rows are present

The 74 out-of-frame landmark observations are retained as raw model outputs.
They are not execution failures and are not geometrically clipped.

Relationship Between Benchmark and Isolated Execution
-----------------------------------------------------
The scientific benchmark and isolated component test produced the same
successful-image count:

- Benchmark successful predictions: 588
- Isolated images with native landmarks: 588

They also produced the same failed/no-landmark count:

- Benchmark failed predictions: 12
- Isolated NO_LANDMARKS images: 12

This agreement is strong reproducibility evidence that the isolated execution
uses the same real Face Landmarks component behavior under the same controlled
face initialization.

However, the two outputs must not be conflated.

The scientific benchmark establishes landmark-localization accuracy through
NME and CED.

The isolated component execution establishes software-level component
operation and records the native numerical model outputs.

A PASS result from the isolated component execution is therefore not an
accuracy score.

Expected Final Result Structure
-------------------------------
results/
├── 300w_landmark_results.csv
├── 300w_landmark_summary.txt
├── 300w_landmark_ced.csv
├── 300w_landmark_thesis_table.csv
├── 300w_landmark_thesis_table.md
├── qualitative/
│   ├── annotated_images/
│   │   ├── strong_indoor_indoor_261.png
│   │   ├── representative_indoor_indoor_166.png
│   │   ├── challenging_indoor_indoor_045.png
│   │   ├── failed_indoor_indoor_008.png
│   │   ├── strong_outdoor_outdoor_059.png
│   │   ├── representative_outdoor_outdoor_036.png
│   │   ├── challenging_outdoor_outdoor_274.png
│   │   └── failed_outdoor_outdoor_057.png
│   └── 300w_landmark_qualitative_selection.csv
├── component_execution/
│   ├── face_landmarks_component_results.csv
│   └── face_landmarks_component_summary.json
└── figures/
    ├── 300w_landmark_ced.png
    └── 300w_landmark_qualitative_examples.png

Output Descriptions
-------------------
results/300w_landmark_results.csv
    Detailed 600-image scientific benchmark result.

results/300w_landmark_summary.txt
    Text record of dataset coverage, protocol, Indoor/Outdoor/Overall metrics,
    and runtime.

results/300w_landmark_ced.csv
    Cumulative error distribution data for Indoor, Outdoor, and Overall
    benchmark subsets.

results/300w_landmark_thesis_table.csv
    Compact CSV table containing the benchmark summary.

results/300w_landmark_thesis_table.md
    Markdown representation of the compact benchmark table.

results/figures/300w_landmark_ced.png
    Final quantitative CED figure.

results/qualitative/annotated_images/
    Eight individually annotated qualitative benchmark images.

results/qualitative/300w_landmark_qualitative_selection.csv
    Machine-readable record of the qualitative roles, accepted benchmark
    values, reproduced values, and generated image paths.

results/figures/300w_landmark_qualitative_examples.png
    Combined qualitative overview figure containing all eight selected cases.

results/component_execution/face_landmarks_component_results.csv
    Structured numerical output from isolated execution of the real PhysioTrack
    Face Landmarks component.

results/component_execution/face_landmarks_component_summary.json
    Aggregate execution accounting for the isolated component run.

Complete Reproduction Procedure
-------------------------------
1. Install the PhysioTrack project environment and dependencies.

2. Download and arrange the 300-W evaluation images and PTS annotations under:

   datasets/300W

3. Confirm the documented Indoor and Outdoor structure.

4. Activate:

   conda activate PhysioTrack-Thesis

5. Open:

   physiotrack/validation/face_landmarks

6. Run the scientific benchmark:

   python 300w_landmark_eval.py
   python 300w_landmark_table.py
   python 300w_landmark_plot.py
   python 300w_landmark_qualitative.py

7. Confirm that the 600-image quantitative metrics, CED outputs, and selected
   qualitative reproductions match the documented accepted results.

8. Run the isolated component execution:

   python face_landmarks_component_test.py

9. Verify:

   results/component_execution/face_landmarks_component_results.csv

   and:

   results/component_execution/face_landmarks_component_summary.json

10. Confirm complete image accounting, per-image landmark counts, coordinate
    validity, status handling, and absence of execution failures before using
    the package as reproducibility evidence.

Methodological Note
-------------------
The reported benchmark values represent the PhysioTrack Face Landmarks
component under the documented controlled evaluation protocol.

Because MediaPipe and 300-W use different native landmark definitions and the
face region is initialized from ground-truth landmarks, the benchmark should
not be presented as an official 300-W competition or leaderboard result.

The fixed 51-point MediaPipe-to-300-W mapping is an evaluation adapter and is
part of the documented experimental protocol.

The isolated component execution exports all 478 native MediaPipe landmarks and
does not use the 51-point accuracy mapping for its software-level output.

Limitations
-----------
- The benchmark evaluates landmark localization under controlled face
  initialization and therefore does not include face-detector or face-selection
  errors.
- The evaluation relies on a fixed anatomical mapping between two different
  landmark definitions.
- Twelve benchmark images did not return landmarks and remain explicit
  failures.
- A small number of valid predictions exhibit large NME values.
- Out-of-frame native MediaPipe coordinates can occur and are retained
  unmodified in the isolated component record.
- The 300-W evaluation set is a defined benchmark sample and should not be
  interpreted as universal real-world landmark performance.
- Qualitative evidence complements, but does not replace, the complete
  quantitative benchmark.
- Isolated component PASS is software execution evidence and does not replace
  the accepted NME/CED accuracy results.

Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- 300w_landmark_eval.py
- mediapipe_300w_mapping.py
- 300w_landmark_table.py
- 300w_landmark_plot.py
- 300w_landmark_qualitative.py
- face_landmarks_component_test.py
- README_300W_FACIAL_LANDMARKS.txt
- results/300w_landmark_results.csv
- results/300w_landmark_summary.txt
- results/300w_landmark_ced.csv
- results/300w_landmark_thesis_table.csv
- results/300w_landmark_thesis_table.md
- results/figures/300w_landmark_ced.png
- results/qualitative/annotated_images/
- results/qualitative/300w_landmark_qualitative_selection.csv
- results/figures/300w_landmark_qualitative_examples.png
- results/component_execution/face_landmarks_component_results.csv
- results/component_execution/face_landmarks_component_summary.json

Generated caches, temporary staging directories, and obsolete diagnostic files
are not part of the final validation deliverables.

Final Validation Status
-----------------------
Scientific benchmark: ACCEPTED

Qualitative evidence: ACCEPTED

Isolated PhysioTrack component execution: PASS

The Face Landmarks validation package is accepted for the defined thesis scope,
with benchmark accuracy, qualitative evidence, reproducibility safeguards, and
real isolated component outputs preserved separately and documented explicitly.
