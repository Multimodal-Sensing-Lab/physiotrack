Gaze Estimation Validation
==========================

Purpose
-------
This validation evaluates the PhysioTrack GazeEstimator component using the
MPIIFaceGaze benchmark in a reproducible cross-dataset protocol. The evaluated
PhysioTrack component wraps the pretrained ptgaze backend and is exercised
through the public PhysioTrack GazeEstimator implementation rather than by
calling the backend directly.

The validation contains five complementary parts:

1. Quantitative evaluation on all MPIIFaceGaze annotations.
2. Independent result auditing and generation of thesis-ready quantitative
   tables and figures.
3. Deterministic qualitative evidence selected from the quantitative results.
4. Isolated component execution through the real PhysioTrack FaceAnalysis
   pipeline using the current learned GazeEstimator implementation.
5. A separate synthetic qualitative demonstration on locally supplied images
   using the same learned GazeEstimator, without benchmark accuracy claims.

Dataset
-------
Dataset: MPIIFaceGaze
Source:
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-3240

Original dataset research publications:

Zhang, X., Sugano, Y., Fritz, M., and Bulling, A. (2019).
"MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation."
IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(1),
pp. 162-175. DOI: https://doi.org/10.1109/TPAMI.2017.2778103

Zhang, X., Sugano, Y., Fritz, M., and Bulling, A. (2017).
"It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation."
Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition Workshops (CVPRW).
DOI: https://doi.org/10.1109/CVPRW.2017.284

The benchmark contains 15 participants (p00 to p14) and 37,667 annotated face
images.

Expected thesis-workspace-relative dataset location (one directory above
the repository):

datasets/MPIIFaceGaze/Data

Expected top-level structure:

datasets/
└── MPIIFaceGaze/
    └── Data/
        ├── p00/
        ├── p01/
        ├── ...
        └── p14/

Each participant directory contains:

- pXX.txt
- Calibration/Camera.mat
- Calibration/monitorPose.mat
- Calibration/screenSize.mat
- dayXX image directories

The validation treats the dataset as read-only. Generated runtime calibration
files and validation outputs must never be written into the dataset tree.

Evaluated Component
-------------------
PhysioTrack component:

src/physiotrack/face/gaze_estimation.py

Class:

GazeEstimator

Configuration used for the benchmark:

- Mode: eth-xgaze
- Device: cpu
- ptgaze version: 0.3.0
- Face detector used by ptgaze: MediaPipe
- Pretrained checkpoint: ETH-XGaze
- Checkpoint file observed in the final run: model.safetensors
- Checkpoint SHA256:
  d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc

Pretrained checkpoint research publication (ETH-XGaze):
Zhang, X., Park, S., Beeler, T., Bradley, D., Tang, S., and Hilliges, O.
(2020). "ETH-XGaze: A Large Scale Dataset for Gaze Estimation under
Extreme Head Pose and Gaze Variation." European Conference on Computer
Vision (ECCV). DOI: https://doi.org/10.1007/978-3-030-58558-7_22

The benchmark validates the PhysioTrack GazeEstimator wrapper itself. The
underlying pretrained model is not retrained or fine-tuned on MPIIFaceGaze.

Validation Files
----------------
Quantitative evaluator:

mpiifacegaze_ethxgaze_eval.py

Independent audit, table, and plot generator:

mpiifacegaze_gaze_plot.py

Qualitative evidence generator:

mpiifacegaze_gaze_qualitative.py

Isolated component execution verifier:

gaze_estimation_component_test.py

Synthetic qualitative demonstration:

gaze_estimation_synthetic_demo.py

All validation scripts are located under the repository root at:

validation/gaze_estimation

Path and Reproducibility Design
-------------------------------
The validation scripts derive paths from their own locations, not the
current working directory. Benchmark scripts resolve the dataset at
<thesis-workspace>/datasets/MPIIFaceGaze/Data, where the repository is
located at <thesis-workspace>/physiotrack-v2. Thus, the dataset directory
is a sibling of the repository, not inside its source tree.

For benchmark reproduction on another machine:

1. Clone or copy the PhysioTrack repository inside a thesis workspace.
2. Install the project dependencies and gaze backend in a Python environment.
3. Download the correct MPIIFaceGaze dataset and place its Data directory
   at <thesis-workspace>/datasets/MPIIFaceGaze/Data.
4. Run the validation scripts from validation/gaze_estimation within the
   repository (or use absolute script paths from any working directory).

The synthetic demonstration does not require MPIIFaceGaze. It reads
validation/synthetic_media/gaze_estimation_media within the repository.
The quantitative evaluator verifies that ptgaze==0.3.0 is installed and
checks the accepted pretrained checkpoint before benchmark inference.

Dataset Preflight
-----------------
Before the full evaluation, the quantitative evaluator verifies:

- datasets/MPIIFaceGaze/Data exists.
- Participants p00 through p14 are present.
- Each participant annotation file pXX.txt exists.
- Camera.mat, monitorPose.mat, and screenSize.mat exist for each participant.
- Required Camera.mat fields are available.
- Annotation rows follow the expected MPIIFaceGaze format.
- Annotation-referenced images exist.
- No duplicate annotation image paths are present within a participant file.
- The total annotation count is exactly 37,667.
- No generated ptgaze_camera.yaml file exists anywhere inside the dataset.

The final preflight produced:

Participants: 15
Annotations: 37667
ptgaze: 0.3.0
PREFLIGHT: PASS

Ground-Truth Gaze Definition
----------------------------
Each MPIIFaceGaze annotation provides the 3D face center and 3D gaze target.

The ground-truth gaze direction is constructed as:

g_gt = gaze_target - face_center

The vector is then normalized to unit length before evaluation.

Prediction
----------
For each annotated image:

1. The image is read from the participant directory.
2. The PhysioTrack GazeEstimator is initialized with the participant camera
   calibration.
3. PhysioTrack performs face detection through the configured ptgaze backend.
4. If a face is available, the pretrained gaze model estimates a normalized
   3D gaze direction.
5. The predicted vector is validated for dimensionality and finite values.
6. The 3D angular error between the ground-truth and predicted gaze vectors is
   calculated.

Camera Calibration Handling
---------------------------
MPIIFaceGaze provides camera calibration in MATLAB format.

For runtime compatibility with ptgaze, the evaluator converts Camera.mat into
temporary YAML camera configurations.

These runtime calibration files are created only inside the evaluator-owned
staging area under validation/gaze_estimation/results. They are never written
into datasets/MPIIFaceGaze/Data.

The staging area is deleted automatically after a successful commit or after
an exception. The dataset itself remains read-only.

Dataset Integrity Protection
----------------------------
The evaluator records an inventory of every file in the MPIIFaceGaze dataset
before inference and compares it with a second inventory after inference.

The inventory comparison checks file paths, file sizes, and modification
timestamps.

If any dataset file is added, removed, or modified during evaluation, the
evaluator raises an error instead of accepting the run.

Final evaluation run:

Dataset integrity after evaluation: PASS

Safe Rerun and Output Ownership
-------------------------------
The validation package uses script-owned outputs and staged replacement so
that a handled failed rerun does not replace previously accepted
evidence. Abrupt termination during final publication may require manual
recovery from a retained staging backup.

The quantitative evaluator owns only:

- results/mpiifacegaze_ethxgaze_summary.txt
- results/mpiifacegaze_ethxgaze_per_person.csv
- results/mpiifacegaze_ethxgaze_per_sample.csv

The plotting/audit script owns only:

- results/mpiifacegaze_ethxgaze_audit.txt
- results/mpiifacegaze_ethxgaze_thesis_table.csv
- results/mpiifacegaze_ethxgaze_thesis_table.md
- results/figures/mpiifacegaze_ethxgaze_per_person_mae.png
- results/figures/mpiifacegaze_ethxgaze_angular_error_distribution.png

The qualitative script owns only:

- results/qualitative/mpiifacegaze_qualitative_manifest.csv
- results/qualitative/case_01_*.png through case_08_*.png
- results/figures/mpiifacegaze_qualitative_contact_sheet.png

The isolated component execution script owns only:

- results/component_execution/gaze_estimation_component_results.csv
  (or gaze_estimation_component_results_partNNN.csv if size splitting is needed)
- results/component_execution/gaze_estimation_component_summary.json

The synthetic demonstration owns only:

- results/synthetic_demo/ (the entire generated directory)
  containing images/, gaze_observations.csv, synthetic_demo_examples.png,
  synthetic_demo_summary.json, and synthetic_demo_summary.txt
- results/synthetic_demo_last_attempt.json and .txt (failure diagnostics only)

For each generative stage, new outputs are first produced in a temporary
staging directory under results, validated there, and only then installed as
the final script-owned outputs. File-level replacement uses rollback
protection so accepted prior outputs are preserved if final installation
fails. Staging data are removed after a normal successful run. The synthetic
demonstration publishes by replacing the entire synthetic_demo directory,
never merging old images with new ones; it validates output files before and
after publication and restores previous results on a handled failure.
If restoration itself fails, the backup is retained for manual recovery.

Metric
------
The primary metric is 3D angular error in degrees.

For normalized ground-truth vector g and normalized predicted vector p:

Angular Error = arccos(clip(g · p, -1, 1)) × 180 / pi

Lower values indicate better gaze-direction agreement.

The evaluator normalizes both vectors before calculating the dot product and
clips the dot product to [-1, 1] for numerical stability.

Failure Accounting
------------------
Every annotation must belong to exactly one of the following categories:

- Successful prediction
- Image read failure
- Face detection failure
- Prediction failure
- Invalid annotation row

The evaluator enforces the invariant:

Total annotations
=
Successful predictions
+ Image read failures
+ Face detection failures
+ Prediction failures
+ Invalid annotation rows

It also writes one row per annotation to the per-sample CSV so that all
failures and successful predictions are independently auditable.

Quantitative Outputs
--------------------
The quantitative evaluator generates:

results/mpiifacegaze_ethxgaze_summary.txt
results/mpiifacegaze_ethxgaze_per_person.csv
results/mpiifacegaze_ethxgaze_per_sample.csv

The per-sample CSV contains the participant, image path, execution status,
failure reason when applicable, ground-truth 3D gaze vector, predicted 3D gaze
vector, and angular error.

The per-person CSV contains annotation counts, failure counts, angular-error
statistics, and runtime for every participant.

Independent Result Audit
------------------------
The script mpiifacegaze_gaze_plot.py independently checks the quantitative
outputs before generating any thesis table or figure.

The audit verifies:

- Exactly 15 participants are represented.
- Exactly 37,667 per-sample rows are present.
- Per-person annotation counts sum to 37,667.
- Per-person failure totals equal the summary totals.
- Per-sample status totals equal the summary totals.
- Failure accounting is complete.
- Per-person mean, median, and standard deviation are recomputed from
  per-sample results and checked against the per-person CSV.
- Overall mean, median, standard deviation, minimum, maximum, 90th percentile,
  and 95th percentile are recomputed from the successful per-sample results
  and checked against the summary.

The consistency audit reported:

Audit status: PASS
Participants checked: 15
Per-sample rows checked: 37667
Successful predictions checked: 37629
Per-person count consistency: PASS
Summary count consistency: PASS
Failure accounting: PASS
Per-person angular statistics recomputation: PASS
Overall angular statistics recomputation: PASS

Final Quantitative Results
--------------------------
Participants: 15
Total annotations: 37,667
Successful predictions: 37,629
Image read failures: 0
Face detection failures: 38
Prediction failures: 0
Invalid annotation rows: 0
Accounted annotations: 37,667

Mean angular error: 8.241855 degrees
Median angular error: 7.744508 degrees
Standard deviation: 4.579882 degrees
Minimum angular error: 0.084242 degrees
Maximum angular error: 57.029950 degrees
90th percentile angular error: 14.149389 degrees
95th percentile angular error: 16.026888 degrees

Reported full-run runtime from the accepted clean rerun:

2621.19 seconds

The 38 face-detection failures were distributed as follows:

- p07: 9
- p08: 4
- p12: 24
- p13: 1

All other participants had zero face-detection failures.

Participant-Level Results
-------------------------
p00: mean 4.2571 degrees, 2927 / 2927 successful
p01: mean 7.6693 degrees, 2904 / 2904 successful
p02: mean 13.5866 degrees, 2916 / 2916 successful
p03: mean 5.4042 degrees, 2929 / 2929 successful
p04: mean 7.3936 degrees, 2860 / 2860 successful
p05: mean 8.7679 degrees, 2870 / 2870 successful
p06: mean 7.3461 degrees, 2877 / 2877 successful
p07: mean 7.5599 degrees, 2834 / 2843 successful
p08: mean 10.8400 degrees, 2763 / 2767 successful
p09: mean 8.9890 degrees, 2719 / 2719 successful
p10: mean 6.2712 degrees, 2194 / 2194 successful
p11: mean 8.9812 degrees, 2262 / 2262 successful
p12: mean 8.2779 degrees, 1577 / 1601 successful
p13: mean 6.7896 degrees, 1497 / 1498 successful
p14: mean 12.9327 degrees, 1500 / 1500 successful

Thesis Quantitative Artifacts
-----------------------------
The independent audit and plotting script generates:

results/mpiifacegaze_ethxgaze_audit.txt
results/mpiifacegaze_ethxgaze_thesis_table.csv
results/mpiifacegaze_ethxgaze_thesis_table.md
results/figures/mpiifacegaze_ethxgaze_per_person_mae.png
results/figures/mpiifacegaze_ethxgaze_angular_error_distribution.png

The per-person figure shows the mean angular error for every participant and
the overall mean angular error.

The angular-error distribution figure shows the empirical cumulative
distribution of 3D angular error across all 37,629 successful predictions. It
complements the mean, median, percentile, and maximum statistics by showing the
complete observed error distribution and retaining the high-error tail.

Qualitative Evidence
--------------------
The qualitative analysis is generated only from successful predictions already
recorded in the quantitative per-sample CSV. The qualitative script does not
rerun the gaze model and therefore cannot introduce a second inference result
for the selected examples.

Eight cases are selected deterministically near these percentiles of the
successful-sample angular-error distribution:

- P05
- P25
- P50
- P75
- P90
- P95
- P99
- Maximum

When possible, the selector prefers different participants across cases to
reduce subject-specific cherry-picking.

Qualitative angular-error levels:

P05: approximately 1.92 degrees
P25: approximately 4.91 degrees
P50: approximately 7.74 degrees
P75: approximately 10.93 degrees
P90: approximately 14.15 degrees
P95: approximately 16.03 degrees
P99: approximately 20.55 degrees
Maximum: approximately 57.03 degrees

Qualitative Visualization
-------------------------
For each selected image:

- Green represents the ground-truth gaze direction.
- Red represents the PhysioTrack prediction.
- The panel reports the true 3D angular error.
- Ground-truth and predicted yaw/pitch values are included as descriptive
  directional information.

The displayed arrows are deliberately bounded 2D directional visualizations.
Their origins are placed in the detected visible photographic region (not
a measured eye landmark), and their endpoints are kept inside this region
rather than the black image padding. These drawing decisions do not alter
the original gaze vectors or scientific angular errors. The arrows are not
treated as calibrated camera-plane projections or quantitative metrics.

Scientific comparison remains based exclusively on the 3D angular error
computed from the original ground-truth and predicted gaze vectors.

Qualitative Output Handling
---------------------------
The qualitative script generates all eight case images, the manifest, and the
combined contact sheet inside a staging area first. The staged manifest and
images are checked against the accepted quantitative per-sample results before
any final output is replaced.

After staged validation passes, only qualitative-script-owned outputs are
replaced. Stale qualitative artifacts are removed only after the complete new
set has been validated and installed successfully.

The eight individual qualitative images and their manifest are stored under:

results/qualitative

The combined qualitative contact sheet is stored under:

results/figures

The script never deletes or modifies MPIIFaceGaze dataset files.

Qualitative Outputs
-------------------
The qualitative generator produces:

results/qualitative/mpiifacegaze_qualitative_manifest.csv
results/qualitative/case_01_*.png
results/qualitative/case_02_*.png
results/qualitative/case_03_*.png
results/qualitative/case_04_*.png
results/qualitative/case_05_*.png
results/qualitative/case_06_*.png
results/qualitative/case_07_*.png
results/qualitative/case_08_*.png
results/figures/mpiifacegaze_qualitative_contact_sheet.png

The manifest stores the exact selection percentile, participant, image path,
ground-truth and predicted vectors, derived yaw/pitch values, angular error,
and generated case filename.

Isolated Component Execution Verification
-----------------------------------------
After the quantitative evaluator, plotting/audit stage, and qualitative
evidence were accepted, a separate isolated execution test was run to verify
the current learned GazeEstimator through the real PhysioTrack FaceAnalysis
pipeline.

Script:

gaze_estimation_component_test.py

This is not a second accuracy benchmark. It does not recompute MPIIFaceGaze
ground-truth angular-error metrics.

Configuration:

- Dataset population: complete MPIIFaceGaze annotated image population
- Participants: 15
- Annotation images: 37,667
- Pipeline: PhysioTrack FaceAnalysis
- Target component: learned GazeEstimator
- Mode: eth-xgaze
- Device: cpu
- ptgaze version: 0.3.0
- GazeEstimator minimum IoU: 0.10
- Checkpoint: model.safetensors
- Checkpoint SHA256:
  d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc

The current PhysioTrack face detector remains active because FaceAnalysis
passes its face boxes to GazeEstimator.predict_faces(). All unrelated optional
components are disabled:

- tracking
- head_pose
- landmarks
- quality
- eyes
- blink
- legacy gaze
- mouth
- mouth_motion
- emotion
- regions
- temporal

This preserves the current project design in which the legacy geometric
GazeDescriptor and the learned GazeEstimator are separate optional
components.

The current GazeEstimator source performs one-to-one greedy IoU association
between PhysioTrack face boxes and ptgaze detections. The accepted target-
conditioned crop fallback and the configured minimum-IoU requirement remain
inside the current source implementation; the isolated script does not
duplicate this logic locally.

The isolated run records real numerical component outputs including:

- face bounding-box coordinates
- detector confidence
- 3D gaze vector x, y, and z
- gaze-vector norm
- pitch
- yaw
- association IoU
- availability
- explicit status and failure reason

Every successful gaze vector is checked for finite values and unit norm, and
every successful association must satisfy the configured minimum IoU.

Final isolated execution results:

Participants:
15

Annotation images:
37,667

Covered annotation images:
37,667

Images with at least one detected face:
37,666

Images without a detected face:
1

Detected face rows:
37,809

Total component-result rows:
37,810

GazeEstimator available rows:
37,661

NO_GAZE_OUTPUT rows:
148

NO_FACE rows:
1

Execution failures:
0

Overall status:
PASS

The result contains more rows than annotation images because multi-face images
produce one row per detected face. The complete 37,667-image annotation
population remains covered exactly once at image level, while detected-face
rows preserve the multi-person output structure.

The accepted isolated run also confirms the current one-to-one association
behavior: in multi-face images only the matched target face receives an
available gaze-estimation output, while unmatched additional faces remain
explicitly unavailable.

The numerical output audit found all successful gaze vectors finite and
unit-normalized within floating-point precision. All successful association
IoU values were above the configured 0.10 threshold.

The final clean isolated execution runtime was:

135.77 minutes (8145.99 seconds)

Runtime is environment-dependent and is not a scientific reproducibility
target.

Git-Safe Result Handling
------------------------
The isolated component writer checks the generated CSV size before final
installation.

If the complete CSV is at or below the configured Git-safe maximum size of
90 MiB, it remains a single file:

results/component_execution/gaze_estimation_component_results.csv

If the complete CSV exceeds that limit, the script splits it automatically
into sequential part files such as:

gaze_estimation_component_results_part001.csv
gaze_estimation_component_results_part002.csv
...

Splitting is performed only between complete participant groups. A participant
is never divided across multiple result files. The summary JSON records the
generated filenames, row counts, participant boundaries, and file sizes.

The accepted isolated run produced one CSV because its size was only
approximately 9.25 MiB.

Synthetic Qualitative Demonstration
-----------------------------------
A separate synthetic demonstration uses the current PhysioTrack FaceAnalysis
integration with the learned GazeEstimator (eth-xgaze) on locally supplied
non-benchmark images. It is independent of MPIIFaceGaze quantitative evaluation
and of the isolated component-execution test.

Script:

gaze_estimation_synthetic_demo.py

Input directory (relative to the repository root):

validation/synthetic_media/gaze_estimation_media/

The accepted run used 12 PNG images named 01.png through 12.png. Images
01-11 show one synthetic face in varied nominal gaze/head/lighting conditions.
Image 12 contains three faces, including a face with glasses. Input source
hashes are recorded separately for each image in the summary JSON and TXT.

Synthetic Media Provenance and Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The twelve synthetic input images (01.png through 12.png) were generated
using OpenAI ChatGPT as independent illustrative media. Their intended gaze
and head-orientation scenarios are descriptive rather than independently
measured ground-truth gaze angles. These images are separate from the
MPIIFaceGaze scientific benchmark and its reference annotations.

Source: OpenAI, ChatGPT (synthetic image generation).
Terms of Use: https://openai.com/policies/terms-of-use/
The OpenAI terms apply to the generated images; the CC0 licence of Blender
Human Base Meshes is not attributed to this image set. Media provenance and
publication conditions are also documented in VALIDATION_MEDIA_LICENSES.md.

Execution configuration:

- Pipeline: PhysioTrack FaceAnalysis with the real learned GazeEstimator
- Mode: eth-xgaze; device: cpu; gaze-estimation minimum IoU: 0.10
- Display confidence threshold: 0.25; face detection uses the default
  FaceAnalysis detector. The script records DETECTOR_CONFIDENCE = 0.001
  in metadata but does not explicitly pass that constant to the detector.
- Tracking, head pose, landmarks, eye openness, blink, legacy geometric gaze,
  mouth, emotion, quality, regions, and temporal analysis: disabled
- Checkpoint: model.safetensors
- Checkpoint SHA256:
  d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc

No Blender/camera calibration was supplied with these synthetic images. The
script creates an assumed pinhole calibration for the image dimensions:

- focal length in pixels: max(image_width, image_height)
- principal point: (image_width/2, image_height/2)
- distortion coefficients: all zero

This is an explicit runtime assumption, not measured camera calibration. The
12 supplied images were all 1448 by 1086 pixels; the calibration YAML is
created in temporary runtime storage and is not published as a measured camera
model. These assumptions may affect numerical predictions. Ground-truth gaze
vectors or reliable synthetic gaze angles were not supplied, so no accuracy,
angular error, precision/recall, or other benchmark metric is calculated.

Final synthetic demonstration results:

- Status: COMPLETE
- Discovered and successfully processed images: 12 / 12
- Failed images: 0
- Detected and displayed faces: 14
- Faces with available learned gaze output: 13
- Faces without a gaze output: 1
- Gaze arrows drawn: 13
- Detections below display threshold: 0
- Invalid face results: 0
- Face-observation CSV rows: 14

The three-face image 12 produced two available gaze predictions and one
NO_GAZE_OUTPUT record (face 2, detector confidence approximately 0.88;
recorded association IoU 0.0).
That face is correctly shown with an orange box and the label "gaze unavailable"
without inventing or drawing a gaze arrow. The unavailable output does not
imply that face detection failed; the exact reason for non-availability is not
established by this synthetic result alone.

Predicted yaw/pitch values for the single-face images (degrees, rounded to
2 decimals; predictions, NOT ground truth):

Image 01: yaw +2.34,  pitch +15.93 (nominal forward gaze)
Image 02: yaw -32.61, pitch +17.18 (nominal gaze left)
Image 03: yaw +26.29, pitch +16.18 (nominal gaze right)
Image 04: yaw +2.61,  pitch +3.63  (nominal gaze up)
Image 05: yaw +0.98,  pitch +43.38 (nominal gaze down)
Image 06: yaw -28.59, pitch +8.05  (nominal upper-left gaze)
Image 07: yaw +24.67, pitch +10.23 (nominal upper-right gaze)
Image 08: yaw -24.02, pitch +33.53 (nominal lower-left gaze)
Image 09: yaw +29.24, pitch +26.80 (nominal lower-right gaze)
Image 10: yaw -2.83,  pitch +19.65 (head rotated; eyes toward camera)
Image 11: yaw +1.50,  pitch +10.34 (low illumination; nominal forward gaze)
Image 12: face 1 yaw +1.29, pitch +15.54;
          face 2 NO_GAZE_OUTPUT;
          face 3 yaw +13.18, pitch +12.31 (multiple faces)

The predicted directions do not always agree visually with the intended gaze
conditions, notably in some upward and diagonal cases. This observation is
reported as a qualitative model behavior, not as a quantitative accuracy
measurement. The red arrows show the model's predictions, not idealized target
directions. Their 2D length and endpoint are illustrative and bounded; the
origin is obtained from the available face keypoints or an approximate upper
face-box position. No additional eye-landmark component is enabled.

Synthetic outputs:

results/synthetic_demo/
  images/                                  (12 annotated PNG images)
  gaze_observations.csv                    (14 face observations)
  synthetic_demo_examples.png             (one 12-image contact sheet)
  synthetic_demo_summary.json             (full machine-readable report)
  synthetic_demo_summary.txt              (matching readable report)

The script derives input/output paths from __file__, checks that FaceAnalysis
and GazeEstimator come from the current repository, and records dependency,
implementation, script, input, and checkpoint SHA256 information where
available. It validates each annotated PNG, CSV schema and accounting,
contact sheet, and JSON/TXT consistency before publication and rechecks the
published set. Outputs are staged and the entire prior results/synthetic_demo
directory is replaced only after a complete successful run. Results are not
merged with old outputs; an accepted previous demonstration is restored after
a handled publication failure. Temporary staging and the superseded backup are
removed after normal success. If image processing fails, previously accepted results are preserved and
results/synthetic_demo_last_attempt.json and .txt report the rejected attempt.

This demonstration does not modify MPIIFaceGaze data or any benchmark,
qualitative evidence, plotted figures, or component-execution outputs.

Run Order
---------
Activate the thesis environment:

conda activate PhysioTrack-v2

Run from the validation directory:

cd /d "<repository-root>\validation\gaze_estimation"

The example path above shows the expected project layout. The Python scripts
are resolved from the documented project structure rather than a fixed
installation location.

Optional syntax check:

python -m py_compile mpiifacegaze_ethxgaze_eval.py mpiifacegaze_gaze_plot.py mpiifacegaze_gaze_qualitative.py gaze_estimation_component_test.py gaze_estimation_synthetic_demo.py

Quantitative evaluator preflight:

python mpiifacegaze_ethxgaze_eval.py --preflight-only

Run the quantitative evaluation:

python mpiifacegaze_ethxgaze_eval.py

Run the independent audit and generate quantitative thesis artifacts:

python mpiifacegaze_gaze_plot.py

Generate deterministic qualitative evidence:

python mpiifacegaze_gaze_qualitative.py

Isolated component preflight:

python gaze_estimation_component_test.py --preflight-only

Isolated real-pipeline smoke test:

python gaze_estimation_component_test.py --smoke-test --smoke-count 3

Full isolated component execution:

python gaze_estimation_component_test.py

Run the independent synthetic qualitative demonstration (requires only
validation/synthetic_media/gaze_estimation_media, not MPIIFaceGaze):

python gaze_estimation_synthetic_demo.py

Recommended clean reproduction order:

1. python mpiifacegaze_ethxgaze_eval.py --preflight-only
2. python mpiifacegaze_ethxgaze_eval.py
3. python mpiifacegaze_gaze_plot.py
4. python mpiifacegaze_gaze_qualitative.py
5. python gaze_estimation_component_test.py --preflight-only
6. python gaze_estimation_component_test.py --smoke-test --smoke-count 3
7. python gaze_estimation_component_test.py
8. python gaze_estimation_synthetic_demo.py

The eighth step is a separate qualitative illustration; it can also be
executed on its own without the benchmark dataset or preceding stages.

Reproducibility Check
---------------------
The evaluator and isolated component script both provide dedicated preflight
modes. A valid installation should report:

ptgaze: 0.3.0
Participants: 15
Annotations: 37667
PREFLIGHT: PASS

Interpretation
--------------
The cross-dataset evaluation demonstrates that the pretrained
ETH-XGaze-based PhysioTrack GazeEstimator can be executed reproducibly on the
full MPIIFaceGaze benchmark and achieves a mean 3D angular error of
8.241855 degrees across 37,629 successful predictions.

The evaluation reproduces the accepted benchmark result while executing the
PhysioTrack GazeEstimator component directly, using project-relative paths,
read-only dataset handling, explicit failure accounting, checkpoint
provenance, per-sample audit data, independent statistics verification, and
deterministic qualitative evidence.

The subsequent isolated FaceAnalysis execution independently verifies the
current learned GazeEstimator software path over the complete 37,667-image
population and records real per-face numerical outputs without creating a
second set of accuracy claims.

The separate 12-image synthetic demonstration documents model behavior in
controlled illustrative conditions: 14 detected faces, 13 available gaze
outputs, and one unavailable output. Its uncalibrated synthetic camera
assumptions and lack of reference gaze vectors preclude accuracy claims.

The participant-level variation and the observed high-error tail are retained
rather than hidden. The maximum error of 57.029950 degrees and all 38
face-detection failures remain part of the reported validation evidence.

Scope and Limitations
---------------------
This is a cross-dataset evaluation of a pretrained model. MPIIFaceGaze is used
for evaluation only; no benchmark images or annotations are used to train or
fine-tune the ETH-XGaze model.

The reported mean angular error is calculated only for samples with successful
gaze predictions. Failures are reported separately and are never silently
discarded from the denominator accounting.

The qualitative arrows are explanatory visualizations and must not be
interpreted as calibrated screen-point predictions.

The validation evaluates gaze-direction estimation. It does not evaluate
screen-point-of-regard accuracy, temporal gaze tracking, fixation detection,
or saccade detection.

The isolated component execution is a software-path and reproducibility
verification. Its availability counts, face-row counts, and association
diagnostics must not be interpreted as an additional accuracy benchmark.

The synthetic demonstration is qualitative only. Predicted yaw/pitch and
2D arrows must not be treated as ground-truth gaze angles, calibrated
screen-point predictions, or scientific accuracy measurements. Inferences
on the stylized images need not generalize to real-world photographs.

Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- mpiifacegaze_ethxgaze_eval.py
- mpiifacegaze_gaze_plot.py
- mpiifacegaze_gaze_qualitative.py
- gaze_estimation_component_test.py
- gaze_estimation_synthetic_demo.py
- README_GAZE_ESTIMATION.txt
- results/mpiifacegaze_ethxgaze_summary.txt
- results/mpiifacegaze_ethxgaze_per_person.csv
- results/mpiifacegaze_ethxgaze_per_sample.csv
- results/mpiifacegaze_ethxgaze_audit.txt
- results/mpiifacegaze_ethxgaze_thesis_table.csv
- results/mpiifacegaze_ethxgaze_thesis_table.md
- results/figures/mpiifacegaze_ethxgaze_per_person_mae.png
- results/figures/mpiifacegaze_ethxgaze_angular_error_distribution.png
- results/qualitative/mpiifacegaze_qualitative_manifest.csv
- results/qualitative/case_01_*.png through case_08_*.png
- results/figures/mpiifacegaze_qualitative_contact_sheet.png
- results/component_execution/gaze_estimation_component_results.csv
- results/component_execution/gaze_estimation_component_summary.json
- results/synthetic_demo/images/*.png (12 annotated images)
- results/synthetic_demo/gaze_observations.csv
- results/synthetic_demo/synthetic_demo_examples.png
- results/synthetic_demo/synthetic_demo_summary.json
- results/synthetic_demo/synthetic_demo_summary.txt

If a future isolated run exceeds the configured Git-safe size threshold, the
single component-results CSV is replaced by the automatically generated
gaze_estimation_component_results_partNNN.csv files recorded in the summary
manifest.


Generated caches and obsolete temporary diagnostic files are not part of the
final validation deliverables.
