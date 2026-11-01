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

Original dataset paper (requested by the official AFLW source):
Koestinger, M., Wohlhart, P., Roth, P. M., and Bischof, H. (2011).
"Annotated Facial Landmarks in the Wild: A Large-scale, Real-world
Database for Facial Landmark Localization."
2011 IEEE International Conference on Computer Vision Workshops (ICCVW),
pp. 2144-2151. DOI: https://doi.org/10.1109/ICCVW.2011.6130513

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

The single failed eligible sample was an image-read failure. It corresponds
to AFLW face ID 47825 and the missing or unreadable image 2/image09437.jpg;
it is not a 6DRepNet360 inference failure.

Running the validation
----------------------
Activate the project environment and enter the validation directory.

Windows example:

conda activate PhysioTrack-v2
cd /d <project-path>\physiotrack-v2\validation\head_pose

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

The benchmark qualitative script creates:

results/
|-- qualitative/
|   |-- annotated_images/             (8 individual case figures)
|   `-- aflw_head_pose_qualitative_selection.csv
`-- figures/
    `-- aflw_head_pose_qualitative_examples.png

The synthetic qualitative demonstration creates:

results/
`-- synthetic_demo/
    |-- images/                        (13 annotated images)
    |-- head_pose_observations.csv
    |-- synthetic_demo_examples.png
    |-- synthetic_demo_summary.json
    `-- synthetic_demo_summary.txt

If any discovered image fails, the previous accepted synthetic_demo directory
is not replaced. The rejected attempt is documented separately under results/:

- synthetic_demo_last_attempt.json
- synthetic_demo_last_attempt.txt

A subsequent successful run removes these obsolete failed-attempt reports.

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

The eight verified qualitative examples are identified below. Mean axis error
is the average absolute angular error over yaw, pitch, and roll for the
selected face; these individual values are not benchmark-wide metrics.

Role              Face ID    Mean axis error (degrees)
----------------  ---------  -------------------------
strong_frontal    58466      0.62
representative    57765      12.88
challenging       43176      36.98
negative_yaw      49170      1.94
positive_yaw      45797      3.00
negative_pitch    55246      5.93
positive_pitch    42901      5.64
high_roll         42229      8.07

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
Reported CPU runtime: 47.71 minutes

The single input failure is the same unavailable AFLW image encountered by the
quantitative evaluator. No PhysioTrack component-execution failures were
observed.

For all 23,407 successful records, the isolated execution reproduced the same
FaceOrientation yaw, pitch, and roll outputs used by the quantitative
evaluation. The component-execution table therefore provides structured
numerical evidence that the real PhysioTrack Head Pose component is operating
through the project pipeline as documented.

Synthetic qualitative head-pose demonstration
---------------------------------------------
The synthetic qualitative demonstration is a separate, non-benchmark visual
assessment of the actual PhysioTrack Head Pose inference and display pipeline.
It uses 13 artificially generated 3D-rendered mannequin images to exercise
frontal and rotated heads, occlusion, lighting variation, and multi-face
input. These images are synthetic rather than photographs of real AFLW
subjects, and are not part of the AFLW evaluation population.

The source images are located under:

validation/synthetic_media/head_pose_media/

The files 01.png through 13.png were used in the documented demonstration.
All 13 evaluated input images have the actual resolution 1122 x 1402 pixels.

Synthetic media provenance and usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The thirteen synthetic PNG inputs (01.png through 13.png) were generated using
OpenAI ChatGPT as illustrative artificial face/head-pose images. They are not
AFLW photographs and were not sourced from the Blender Human Base Meshes asset.
The applicable OpenAI service terms are available at:
https://openai.com/policies/terms-of-use/

These generated images provide independent qualitative examples only. Their
intended pose descriptions do not constitute measured angular ground truth, and
they are not used in the AFLW quantitative benchmark. Media provenance and
distribution documentation is maintained in VALIDATION_MEDIA_LICENSES.md.

The demonstration script is:

head_pose_synthetic_demo.py

After activating the documented environment, run the script from
validation/head_pose (or invoke it by path from another working directory):

python head_pose_synthetic_demo.py

Input and output directories are resolved from the script's location, without
hard-coded user-specific paths. The script verifies that the imported
PhysioTrack classes belong to the local project source checkout. The project,
its dependencies, and required model weights must be available in the active
environment.

The script automatically discovers supported input images and runs the real
project FaceAnalysis pipeline with FaceOrientation/6DRepNet360 enabled on CPU.
A real PhysioTrack face detector supplies face rectangles to the pose
component. Tracking and unrelated analysis components are disabled.
Consequently, unlike the controlled AFLW benchmark, this exercise uses
automatically detected face boxes rather than AFLW ground-truth rectangles.

The documented demonstration uses a detector confidence threshold of 0.001,
IoU threshold of 0.45, and a separate display confidence threshold of 0.25.
Only detections meeting the display threshold are visualized or exported as
per-face pose observations. A detection below that display threshold is
accounted for separately; it is not a Head Pose inference failure.

Each annotated output includes the detected face rectangle, face index,
detection confidence, and predicted yaw, pitch, and roll in degrees. The
three coloured pose axes are rendered using the native PhysioTrack
Result.plot()/6DRepNet360 axis drawing, rather than a separate pose
calculation or projection. The visualized axis colours are:

- X axis: red.
- Y axis: green.
- Z axis: blue.

The axes represent the displayed three-dimensional orientation projected
onto the two-dimensional image. Yaw, pitch, and roll remain the scalar
estimates recorded by FaceOrientation; no axis length or line direction in
the figure is a separately measured ground-truth angle.

The controlled illustrative scenarios, in source-image order, are:

01.png  Frontal head pose
02.png  Yaw rotation in one horizontal direction
03.png  Yaw rotation in the opposite horizontal direction
04.png  Larger yaw rotation in the first direction
05.png  Larger yaw rotation in the opposite direction
06.png  Head raised (pitch variation)
07.png  Head lowered (pitch variation)
08.png  Head tilt in one roll direction
09.png  Head tilt in the opposite roll direction
10.png  Combined yaw, pitch, and roll variation
11.png  Partial facial occlusion by a hand
12.png  Reduced illumination
13.png  Three faces with different horizontal head orientations

The intended rotations in the synthetic image descriptions are nominal
generation conditions, not calibrated ground-truth labels. The following
values are predictions from the real PhysioTrack model, rounded to one
decimal degree for documentation. Full-precision predictions are preserved
in the observations CSV.

Image     Face index    Yaw (deg)    Pitch (deg)    Roll (deg)
--------  ------------  -----------  -------------  -----------
01.png    1                   +0.7           +6.9         -0.1
02.png    1                  -23.5           +6.7         -2.8
03.png    1                  +23.1           +4.7         +1.0
04.png    1                  -45.5           +1.7         -1.6
05.png    1                  +46.2           +3.3         +2.4
06.png    1                   +0.7           -9.2         -0.7
07.png    1                   -1.0          +19.0         +0.8
08.png    1                   +3.6           +4.4        -19.6
09.png    1                   +1.5           +6.7        +20.0
10.png    1                  -20.2           -4.6        +11.0
11.png    1                   -7.0          +10.0         -2.7
12.png    1                   +0.4           +6.2         +0.2
13.png    1                   +1.8           +5.5         +0.2
13.png    2                  +36.4           +0.5         -0.8
13.png    3                  -32.6           +3.7         -0.4


Validated synthetic demonstration accounting:

Input images discovered: 13
Images processed successfully: 13
Image-processing failures: 0
Initial detector outputs across all images: 16
Face detections excluded from display (confidence < 0.25): 1
Faces displayed with valid Head Pose: 15
Three-axis coordinate triads drawn: 15
Faces without a valid displayed pose: 0
Invalid face results: 0
Structured per-face pose observations: 15
Overall execution status: COMPLETE

Of the 16 raw detector outputs, the extra low-confidence detection occurs
in 04.png and is omitted from the plotted and exported per-face observations.
In 13.png, all three visible mannequin faces have distinct pose observations
and their own three-axis overlays. These cases account for the difference
between the number of input images (13) and the number of displayed pose
observations (15).

The generated demonstration artifacts are:

results/synthetic_demo/images/
    Thirteen individual annotated PNG outputs, with deterministic filename
    suffixes used to prevent source-name collisions. Each annotated image
    shows face rectangles, predicted angles, and the native three-axis pose
    rendering for displayed faces.

results/synthetic_demo/head_pose_observations.csv
    Fifteen rows recording source-image name, source SHA-256 checksum, output
    image path, image dimensions, face index, detection confidence, face box,
    and the real unrounded yaw, pitch, and roll predictions.

results/synthetic_demo/synthetic_demo_summary.json
    Machine-readable configuration, source-image checksums, per-image and
    aggregate accounting, detection and display counts, pose-axis counts,
    execution status, output paths, and reproducibility metadata. The latter
    records the demo script and implementation-source SHA-256 checksums plus
    installed package versions; it does not establish model-weight identity.

results/synthetic_demo/synthetic_demo_summary.txt
    Human-readable summary generated from the same execution data as the JSON,
    including configuration, aggregate totals, individual-image results, source
    hashes, and the list of published artifacts.

results/synthetic_demo/synthetic_demo_examples.png
    Contact-sheet overview of the thirteen annotated qualitative examples.

The script first generates results in a temporary staging directory. It checks
that every discovered image succeeded, that all expected annotated images and
the contact sheet are readable, that CSV pose observations and face-accounting
totals are consistent, and that both JSON and TXT reports agree with the
recorded execution. It also checks for unexpected files and input changes.
Only a fully validated stage can replace the entire script-owned
results/synthetic_demo/ directory. New and old demonstration files are never
merged. Published files are checked again after replacement, with restoration
of the previous directory if ordinary publication or verification fails.
Successful publication removes the obsolete backup and staging directory.
If restoration itself cannot complete, the backup is retained for recovery.

A failed image-processing attempt instead leaves the accepted synthetic_demo
untouched and writes results/synthetic_demo_last_attempt.json and .txt. These
reports are diagnostics and are removed after the next successful publication.
The script does not modify the AFLW benchmark CSV, benchmark summary, thesis
table, AFLW figures, qualitative AFLW evidence, or isolated component-execution
results. The recorded source and software checksums help track reproduction,
but equivalent inputs and settings do not guarantee bitwise-identical pose
predictions across different model-weight or runtime configurations.

This demonstration is qualitative only. Artificially generated head-pose
images do not provide independently measured, reliable yaw/pitch/roll ground
truth. Therefore, the displayed pose numbers cannot be used to calculate
MAE, infer calibrated head-pose accuracy, or claim an accuracy percentage.
The AFLW controlled benchmark remains the separate quantitative evidence.
The synthetic demonstration instead provides evidence of successful real
inference, plotting of the predicted 3D axes, basic pose variation, and
multi-face visual association under the illustrated conditions.


Per-face Head Pose association
------------------------------
FaceOrientation accepts the face rectangles produced by the FaceAnalysis input
stage. The 6DRepNet360 backend retains each successfully processed crop's
original input-box index in its pose-result identifier. This index denotes the
position in the supplied box sequence, not a persistent person or track ID.

FaceAnalysis associates each returned orientation with the face at that input
index and verifies correspondence with the returned bounding box. Coordinate
agreement within one pixel accommodates the backend's integer conversion of
face rectangles. If a crop is not processed, its corresponding face has no
orientation; other results remain associated with their original input boxes
regardless of the number or order of returned pose instances.

For an injected predictor that does not supply original input-box indices,
FaceAnalysis accepts a pose only when the returned rectangle uniquely matches
an unmatched face. Ambiguous or unmatched pose outputs are not assigned.

Deterministic Head Pose association verification
------------------------------------------------
The software association path can be checked with:

head_pose_association_regression_test.py

The regression uses controlled face rectangles and deterministic rotation
matrices instead of AFLW images or trained model weights. It exercises the
HeadPoseEstimator crop/index path, FaceOrientation, and the FaceAnalysis
integration path. The cases include:

- An invalid first face rectangle followed by a valid rectangle.
- A valid rectangle followed by an invalid rectangle.
- An entirely invalid input batch.
- Two valid rectangles separated by an invalid rectangle.
- Direct and integrated single-face orientation consistency.
- Reordered pose outputs retaining their original input-box indices.
- Unindexed pose outputs associated by unique bounding-box correspondence.

From the project root, run:

python validation\head_pose\head_pose_association_regression_test.py

Alternatively, from validation/head_pose, run:

python head_pose_association_regression_test.py

The regression reports PASS when all assertions succeed. It generates no
persistent output files and is a software association check, not an AFLW
head-pose accuracy measurement.

Validated regression status: PASS

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

Reported CPU runtime for the documented evaluation run: 59.51 minutes.

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

The synthetic demonstration complements, but does not replace, the quantitative
AFLW benchmark. It uses automatically detected faces on generated mannequin
images and provides no independently verified angular-accuracy measurements.

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
10. Run head_pose_association_regression_test.py and confirm that the
    deterministic per-face association assertions pass.
11. Place the 13 synthetic PNG images under
    validation/synthetic_media/head_pose_media and run
    head_pose_synthetic_demo.py to reproduce the qualitative pose and
    three-axis visualizations; verify its annotated PNGs, observation CSV,
    contact sheet, and matching synthetic_demo_summary.json and .txt reports.
    Confirm the complete synthetic_demo directory contains no obsolete files.
12. Preserve the quantitative results, figures, qualitative benchmark
    evidence, isolated component-execution outputs, and separately labeled
    synthetic demonstration as the reproducible validation artifacts for
    this component.

Equivalent dataset, model, dependency, and device settings should produce
scientifically equivalent results. Runtime may vary between systems.


Downloading and Preparing Benchmark Media
-----------------------------------------
Original AFLW photographs and derived annotated benchmark images
are not distributed with the public GitHub repository.

Obtain the required AFLW archives from the official source:
https://www.tugraz.at/institute/icg/research/team-bischof/learning-recognition-surveillance/downloads/aflw

Required archives:

- aflw-db.tar.gz
- aflw-images-0.tar.gz
- aflw-images-2.tar.gz
- aflw-images-3.tar.gz

Extract and arrange them under the workspace's datasets/AFLW/
directory, alongside physiotrack-v2, preserving the aflw.sqlite
database and the flickr/0, flickr/2, and flickr/3 image directories
exactly as documented above.

After preparing the dataset, run the benchmark and qualitative
validation scripts in the documented order.

AFLW-derived annotated examples remain local. Numerical results,
independent plots, and synthetic head-pose demonstration images
are separate from the restricted benchmark photographs.

See VALIDATION_MEDIA_LICENSES.md for media distribution conditions.