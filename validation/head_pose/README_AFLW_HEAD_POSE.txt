AFLW Head Pose Validation
=========================

Purpose
-------
This validation evaluates the PhysioTrack FaceOrientation component, which uses
6DRepNet360, on the Annotated Facial Landmarks in the Wild (AFLW) dataset.

The evaluation is designed as a controlled coarse head-pose benchmark. AFLW
provides approximate pose annotations derived from facial landmarks using POSIT,
so the results should not be interpreted as fine-grained manually verified
head-pose ground truth.

Evaluated component
-------------------
PhysioTrack component: FaceOrientation
Backend: 6DRepNet360
Device used for the reported validation: CPU
Evaluated angles: yaw, pitch, and roll

Dataset
-------
Dataset: Annotated Facial Landmarks in the Wild (AFLW)

Official source:
https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw

Only the following AFLW archives are required for this head-pose validation:

- aflw-db.tar.gz
- aflw-images-0.tar.gz
- aflw-images-2.tar.gz
- aflw-images-3.tar.gz

The following AFLW packages are not required for this benchmark:

- aflw-facedbsql-src.tar.gz
- aflw-facedetector.tar.gz
- aflw-gui-src.tar.gz
- aflw-gui-bin-win32
- aflw-matlab.tar.gz

Other official files such as the paper, example script, checksum files, and
documentation may be retained for reference but are not inputs to the evaluator.

Dataset extraction and final layout
-----------------------------------
The required archives may extract into separate directories. Their relevant
contents must be combined into one AFLW dataset directory.

1. Extract aflw-db.tar.gz.

   Relevant content:

   aflw-db/
   `-- aflw/
       |-- data/
       |   |-- aflw.sqlite
       |   `-- aflw-changelog.txt
       `-- doc/
           |-- README.txt
           `-- aflw-befit.pdf

2. Extract aflw-images-0.tar.gz and copy:

   aflw-images-0/aflw/data/flickr/0

   to:

   datasets/AFLW/aflw/data/flickr/0

3. Extract aflw-images-2.tar.gz and copy:

   aflw-images-2/aflw/data/flickr/2

   to:

   datasets/AFLW/aflw/data/flickr/2

4. Extract aflw-images-3.tar.gz and copy:

   aflw-images-3/aflw/data/flickr/3

   to:

   datasets/AFLW/aflw/data/flickr/3

The final required dataset structure is:

datasets/
`-- AFLW/
    `-- aflw/
        |-- data/
        |   |-- aflw.sqlite
        |   |-- aflw-changelog.txt
        |   `-- flickr/
        |       |-- 0/
        |       |-- 2/
        |       `-- 3/
        `-- doc/
            |-- README.txt
            `-- aflw-befit.pdf

The evaluator resolves this location relative to the project workspace. No
machine-specific dataset path is required.

Ground truth and coordinate convention
--------------------------------------
The evaluator reads AFLW FacePose annotations from aflw.sqlite.

AFLW stores roll, pitch, and yaw in radians. The evaluator converts them to
degrees and applies the fixed convention mapping used by the evaluation:

PhysioTrack yaw   = -AFLW yaw
PhysioTrack pitch = -AFLW pitch
PhysioTrack roll  = -AFLW roll

Evaluation protocol
-------------------
AFLW ground-truth face rectangles are supplied to FaceOrientation. This isolates
head-pose estimation from face-detection performance.

Primary pose range:

|yaw|   <= 90 degrees
|pitch| <  90 degrees
|roll|  <= 90 degrees

The controlled range avoids extreme Euler-angle boundary cases where independent
axis-wise comparison becomes ambiguous.

Absolute angular error is calculated using wrapped angular distance:

error = abs(((prediction - reference + 180) mod 360) - 180)

Reported metrics:

- Yaw MAE
- Pitch MAE
- Roll MAE
- Overall MAE
- Median absolute error
- Standard deviation of absolute error
- Prediction success rate

Dataset accounting
------------------
The AFLW database contains 24,396 FacePose records.

Twelve FacePose records have no corresponding FaceRect annotation. Because the
evaluation protocol requires AFLW ground-truth face rectangles, these 12 records
cannot enter the evaluation join.

Final accounting:

FacePose records in database: 24,396
Joined evaluation records: 24,384
Outside primary pose range: 976
Primary-protocol eligible samples: 23,408
Successful predictions: 23,407
Failed eligible samples: 1
Success rate: 99.9957%

The single failed eligible sample was an image-read failure.

Running the validation
----------------------
Activate the project environment and enter the validation directory.

Windows example:

conda activate PhysioTrack-Thesis
cd /d <project-path>\physiotrack\validation\head_pose

Optional preflight:

python aflw_head_pose_eval.py --preflight-only

Full evaluation:

python aflw_head_pose_eval.py

After the quantitative evaluation is complete, generate the thesis table and
quantitative figures:

python aflw_head_pose_plot.py

Output files
------------
The evaluator creates:

results/
|-- aflw_head_pose_results.csv
`-- aflw_head_pose_summary.txt

The plotting/table script creates:

results/
|-- aflw_head_pose_thesis_table.csv
|-- aflw_head_pose_thesis_table.md
`-- figures/
    |-- aflw_head_pose_error_metrics.png
    `-- aflw_head_pose_error_distribution.png

The isolated component-execution script creates:

results/
`-- component_execution/
    |-- head_pose_component_results.csv
    `-- head_pose_component_summary.json

results/figures/aflw_head_pose_error_metrics.png
    Bar-chart summary of per-axis MAE and median absolute angular error.

results/figures/aflw_head_pose_error_distribution.png
    Empirical cumulative distribution of absolute angular error for yaw,
    pitch, and roll across all successful primary-protocol samples. The figure
    complements the summary statistics by showing the full error distribution
    and the relative prevalence of larger-error cases.


Qualitative benchmark evidence
------------------------------
A separate qualitative script is included to visualize the documented AFLW
head-pose protocol on selected benchmark faces:

aflw_head_pose_qualitative.py

The qualitative analysis does not replace or modify the quantitative
evaluation. It uses the same PhysioTrack FaceOrientation component, 6DRepNet360
backend, CPU inference configuration, AFLW FacePose convention mapping, AFLW
ground-truth face rectangles, and primary pose-range definition used by the
quantitative evaluator.

The per-face quantitative CSV is used as the source for
qualitative selection and verification. Eight interpretable cases are selected:

- strong_frontal
- representative
- challenging
- negative_yaw
- positive_yaw
- negative_pitch
- positive_pitch
- high_roll

The strong-frontal example is selected from near-frontal faces with low per-face
angular error. The representative example is selected to have mean axis error
close to the overall benchmark MAE. The challenging example is selected from
the high-error portion of the result distribution. The directional cases are
selected to provide clear negative and positive yaw, negative and positive
pitch, and substantial roll examples.

Visual prominence is also considered so that the evaluated target face is large
enough and sufficiently central to be interpretable in the final evidence
figures.

Before any qualitative figure is written, the selected AFLW faces are rerun
through FaceOrientation using the same ground-truth face rectangle. The rerun
ground-truth angles, predictions, and wrapped angular errors are compared
numerically with the per-face result CSV. Qualitative output is written only if
this verification passes.

Each qualitative evidence figure contains:

- Original AFLW image.
- Visible AFLW target-face bounding box.
- Enlarged target-face crop.
- Ground-truth yaw, pitch, and roll.
- PhysioTrack-predicted yaw, pitch, and roll.
- Wrapped absolute angular error for each axis.
- Mean axis error for the displayed face.
- Full-benchmark metrics.
- Protocol summary.

The visible face rectangle is intentionally included because AFLW images may
contain more than one face. It identifies the exact ground-truth face rectangle
supplied to FaceOrientation and therefore makes the evaluated target explicit.
The rectangle is not an automatically detected face box and should not be
interpreted as a face-detection result.

The per-image pose values and errors are qualitative diagnostics for the
displayed benchmark face. The benchmark result remains the aggregate
evaluation over all 23,407 successful primary-protocol samples.

Run the qualitative analysis after the quantitative CSV and summary are
present:

python aflw_head_pose_qualitative.py

The qualitative outputs are written under:

results/qualitative/

Expected qualitative outputs include:

results/qualitative/annotated_images/
    Eight individual qualitative evidence figures.

results/qualitative/aflw_head_pose_qualitative_selection.csv
    Selected role, face ID, source image, target-face prominence information,
    verified ground-truth pose, prediction, per-axis errors, mean axis error,
    and generated output path.

results/figures/aflw_head_pose_qualitative_examples.png
    Combined 2 x 4 summary figure containing the eight selected target faces and
    their qualitative role, mean axis error, and ground-truth pose.

The qualitative generator owns and may replace only its own qualitative outputs.
It does not delete or modify the quantitative result CSV, summary, thesis table,
or either quantitative head-pose figure.

Isolated PhysioTrack component execution
----------------------------------------
A dedicated component-execution script is included to verify that the real
PhysioTrack FaceOrientation component operates correctly through the project
FaceAnalysis pipeline:

head_pose_component_test.py

This execution test is separate from the AFLW accuracy benchmark. It does not
calculate MAE, median error, standard deviation, or any other angular-accuracy
metric. Its purpose is to run the real project component on the same accepted
primary-protocol AFLW population and export the actual numerical Head Pose
outputs produced by PhysioTrack.

The test uses:

- PhysioTrack FaceAnalysis as the project pipeline.
- FaceOrientation as the target component.
- 6DRepNet360 as the Head Pose backend.
- CPU inference.
- AFLW ground-truth face rectangles as controlled face inputs.
- Tracking disabled.
- Unrelated face-analysis components disabled.
- The same 23,408 primary-protocol AFLW records used by the quantitative
  benchmark.

AFLW pose annotations are used only to select the same primary-protocol
population. They are not used to calculate a second set of accuracy metrics in
this component-execution test.

Before a full execution, the script supports two preliminary checks.

Preflight only:

python head_pose_component_test.py --preflight-only

This verifies the AFLW dataset layout, database accounting, primary-protocol
population, and isolated FaceAnalysis configuration without loading the model
or running inference.

Smoke test:

python head_pose_component_test.py --smoke-test --smoke-count 3

The smoke test runs a small number of real FaceAnalysis/FaceOrientation
inferences and writes no final result files.

Full isolated execution:

python head_pose_component_test.py

The full execution writes:

results/component_execution/head_pose_component_results.csv
    One structured row for every primary-protocol AFLW record. Successful rows
    contain the face ID, source image path, controlled face rectangle, yaw,
    pitch, roll, execution status, and failure information.

results/component_execution/head_pose_component_summary.json
    Component-execution configuration, population accounting, status counts,
    runtime, throughput, and overall execution status.

Final isolated execution accounting:

Primary-protocol records: 23,408
Successful component outputs: 23,407
Input failures: 1
Component execution failures: 0
OK: 23,407
IMAGE_READ_FAILED: 1
Overall status: PASS_WITH_INPUT_FAILURES

The single input failure is the same unavailable AFLW image encountered by the
quantitative evaluator. No PhysioTrack component-execution failures were
observed.

For all 23,407 successful records, the isolated execution reproduced the same
FaceOrientation yaw, pitch, and roll outputs used by the quantitative
evaluation. The component-execution table therefore provides structured
numerical evidence that the real PhysioTrack Head Pose component is operating
through the project pipeline as documented.

Validation results
------------------
Successful predictions: 23,407 / 23,408 eligible samples
Success rate: 99.9957%

Yaw:
MAE 11.1036 degrees
Median absolute error 7.7820 degrees
Std. absolute error 12.7337 degrees

Pitch:
MAE 13.7046 degrees
Median absolute error 7.1886 degrees
Std. absolute error 25.4115 degrees

Roll:
MAE 13.7529 degrees
Median absolute error 4.9559 degrees
Std. absolute error 27.0779 degrees

Overall:
MAE 12.8537 degrees
Median absolute error 6.6887 degrees
Std. absolute error 22.6988 degrees

Reported CPU runtime for the documented evaluation run: 58.96 minutes.

Interpretation and limitations
------------------------------
The reported values characterize PhysioTrack head-pose estimation under a
controlled AFLW coarse-pose protocol.

AFLW pose annotations are approximate and derived from facial landmarks using
POSIT rather than being manually verified high-precision head-pose measurements.
The evaluation should therefore be interpreted as a reproducible coarse-pose
benchmark rather than a fine-grained head-pose leaderboard result.

The benchmark intentionally uses AFLW ground-truth face rectangles, so the
reported numbers measure the head-pose component rather than the combined
face-detection plus head-pose pipeline.

Reproducibility
---------------
The final evaluator uses project-relative paths and does not require any
machine-specific dataset path.

For reproducible execution:

1. Obtain AFLW from the official source.
2. Arrange the required database and image archives exactly as documented above.
3. Use the documented project environment and dependencies.
4. Run the evaluator preflight and then the full evaluator from
   validation/head_pose.
5. Verify the generated quantitative CSV and summary.
6. Generate the thesis table and both quantitative figures from the quantitative
   CSV.
7. Run aflw_head_pose_qualitative.py and inspect the individual qualitative
   figures, selection CSV, and combined qualitative summary figure.
8. Run the isolated component preflight and smoke test.
9. Run head_pose_component_test.py and inspect the structured component
   results CSV and component summary JSON.
10. Preserve the quantitative results, figures, qualitative evidence, and
    isolated component-execution outputs as the reproducible validation
    artifacts for this component.

Equivalent dataset, model, dependency, and device settings should produce
scientifically equivalent results. Runtime may vary between systems.
