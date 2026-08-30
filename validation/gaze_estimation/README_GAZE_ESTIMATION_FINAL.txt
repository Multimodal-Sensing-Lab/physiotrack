Gaze Estimation Validation
==========================

Purpose
-------
This validation evaluates the PhysioTrack GazeEstimator component using the
MPIIFaceGaze benchmark in a reproducible cross-dataset protocol. The evaluated
PhysioTrack component wraps the pretrained ptgaze backend and is exercised
through the public PhysioTrack GazeEstimator implementation rather than by
calling the backend directly.

The validation contains three complementary parts:

1. Quantitative evaluation on all MPIIFaceGaze annotations.
2. Independent result auditing and generation of thesis-ready quantitative
   tables and figures.
3. Deterministic qualitative evidence selected from the quantitative results.

Dataset
-------
Dataset: MPIIFaceGaze
Source:
https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-3240

The benchmark contains 15 participants (p00 to p14) and 37,667 annotated face
images.

Expected project-relative dataset location:

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
- Checkpoint file observed in the accepted run: model.safetensors
- Checkpoint SHA256:
  d1c91b2aa6a0c73856c16890d337afdecdb05563ed52182dfdb77742f1c856bc

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

All validation scripts are located under:

physiotrack/validation/gaze_estimation

Path and Reproducibility Design
-------------------------------
The validation scripts use project-relative paths derived from their own
location. They do not contain user-specific absolute paths.

A user should be able to reproduce the validation on another machine by:

1. Cloning or copying the PhysioTrack project.
2. Installing the required environment and gaze dependency.
3. Downloading the correct MPIIFaceGaze dataset.
4. Placing the dataset at datasets/MPIIFaceGaze/Data.
5. Running the validation scripts from
   physiotrack/validation/gaze_estimation.

The quantitative evaluator verifies that ptgaze==0.3.0 is installed before
starting the benchmark.

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

The accepted preflight produced:

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
a temporary YAML camera configuration.

These temporary camera files are written only to:

validation/gaze_estimation/_runtime

They are never written into datasets/MPIIFaceGaze/Data.

The runtime directory is deleted automatically after evaluation, including
when an exception occurs.

Dataset Integrity Protection
----------------------------
The evaluator records an inventory of every file in the MPIIFaceGaze dataset
before inference and compares it with a second inventory after inference.

The inventory comparison checks file paths, file sizes, and modification
timestamps.

If any dataset file is added, removed, or modified during evaluation, the
evaluator raises an error instead of accepting the run.

Accepted run:

Dataset integrity after evaluation: PASS

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

The accepted audit reported:

Audit status: PASS
Participants checked: 15
Per-sample rows checked: 37667
Successful predictions checked: 37629
Per-person count consistency: PASS
Summary count consistency: PASS
Failure accounting: PASS
Per-person angular statistics recomputation: PASS
Overall angular statistics recomputation: PASS

Accepted Quantitative Results
-----------------------------
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

Accepted full-run runtime:

2560.09 seconds

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

The per-person figure shows the mean angular error for every participant and
the overall mean angular error.

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

Accepted qualitative angular-error levels:

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
They are not treated as calibrated camera-plane projections and are not used
as quantitative metrics.

Scientific comparison remains based exclusively on the 3D angular error
computed from the original ground-truth and predicted gaze vectors.

Qualitative Output Cleanup
--------------------------
Before generating new qualitative artifacts, the qualitative script deletes
only its own output directory:

results/qualitative

It then recreates the directory and writes the new artifacts.

This prevents stale images or manifests from previous runs from being mixed
with the current evidence.

The script never deletes or modifies MPIIFaceGaze dataset files.

Qualitative Outputs
-------------------
The qualitative generator produces:

results/qualitative/mpiifacegaze_qualitative_manifest.csv
results/qualitative/mpiifacegaze_qualitative_contact_sheet.png
results/qualitative/case_01_*.png
results/qualitative/case_02_*.png
results/qualitative/case_03_*.png
results/qualitative/case_04_*.png
results/qualitative/case_05_*.png
results/qualitative/case_06_*.png
results/qualitative/case_07_*.png
results/qualitative/case_08_*.png

The manifest stores the exact selection percentile, participant, image path,
ground-truth and predicted vectors, derived yaw/pitch values, angular error,
and generated case filename.

Run Order
---------
Activate the thesis environment:

conda activate PhysioTrack-Thesis

Change to the validation directory:

cd /d C:\Users\<USER>\Documents\PhysioTrack_Thesis\physiotrack\validation\gaze_estimation

The example path above shows the expected project layout. The Python scripts
themselves do not depend on the Windows username or absolute installation
location.

Optional syntax check:

python -m py_compile mpiifacegaze_ethxgaze_eval.py mpiifacegaze_gaze_plot.py mpiifacegaze_gaze_qualitative.py

Run the quantitative evaluation:

python mpiifacegaze_ethxgaze_eval.py

Run the independent audit and generate quantitative thesis artifacts:

python mpiifacegaze_gaze_plot.py

Generate deterministic qualitative evidence:

python mpiifacegaze_gaze_qualitative.py

Recommended Reproducibility Check
---------------------------------
Before a full run, the dataset preflight can be executed independently from
the validation directory:

python -c "import importlib.metadata as im; import mpiifacegaze_ethxgaze_eval as m; d=m.preflight_dataset(); print('Dataset root:', m.DATASET_ROOT); print('ptgaze:', im.version('ptgaze')); print('Participants:', len(d)); print('Annotations:', sum(len(x) for x in d.values())); print('PREFLIGHT: PASS')"

A valid installation should report:

ptgaze: 0.3.0
Participants: 15
Annotations: 37667
PREFLIGHT: PASS

Interpretation
--------------
The accepted cross-dataset evaluation demonstrates that the pretrained
ETH-XGaze-based PhysioTrack GazeEstimator can be executed reproducibly on the
full MPIIFaceGaze benchmark and achieves a mean 3D angular error of
8.241855 degrees across 37,629 successful predictions.

The evaluation reproduces the previously obtained benchmark result while now
executing the PhysioTrack GazeEstimator component directly, using
project-relative paths, read-only dataset handling, explicit failure
accounting, checkpoint provenance, per-sample audit data, independent
statistics verification, and deterministic qualitative evidence.

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

Final Files to Preserve
-----------------------
Preserve the following for thesis reproducibility:

- mpiifacegaze_ethxgaze_eval.py
- mpiifacegaze_gaze_plot.py
- mpiifacegaze_gaze_qualitative.py
- README.txt
- results/mpiifacegaze_ethxgaze_summary.txt
- results/mpiifacegaze_ethxgaze_per_person.csv
- results/mpiifacegaze_ethxgaze_per_sample.csv
- results/mpiifacegaze_ethxgaze_audit.txt
- results/mpiifacegaze_ethxgaze_thesis_table.csv
- results/mpiifacegaze_ethxgaze_thesis_table.md
- results/figures/mpiifacegaze_ethxgaze_per_person_mae.png
- results/qualitative/mpiifacegaze_qualitative_manifest.csv
- results/qualitative/mpiifacegaze_qualitative_contact_sheet.png
- results/qualitative/case_01_*.png through case_08_*.png

Also preserve screenshots of the accepted dataset preflight and final
quantitative console summary as validation evidence.

Generated caches and obsolete temporary diagnostic files are not part of the
final validation deliverables.
