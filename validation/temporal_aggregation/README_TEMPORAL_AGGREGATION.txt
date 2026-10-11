PhysioTrack Temporal Aggregation Validation
==============================================

Overview
--------
This package validates the PhysioTrack FaceTemporalAggregator component.

FaceTemporalAggregator is a deterministic temporal-analysis component rather
than a predictive machine-learning model. It receives per-frame outputs from
upstream face-analysis components and summarizes them over a bounded rolling
window for each tracked person.

The current implementation summarizes:

- head pose: yaw, pitch, and roll
- eye openness: mean_openness
- geometric gaze: mean_iris_x and mean_iris_y
- mouth openness
- mouth movement
- Face Quality: brightness, sharpness, and face_area_ratio
- blink events
- dominant emotion

For the numerical signals, the component reports rolling:

- mean
- population standard deviation
- minimum
- maximum

The validation is therefore designed as a deterministic correctness
evaluation, not as an external predictive-accuracy benchmark. Its purpose is
to establish that the temporal mathematics, state management, rolling-window
behavior, per-person isolation, categorical aggregation, and real PhysioTrack
execution path are correct and reproducible.


Validation Location
-------------------
The validation package is located in:

validation/temporal_aggregation/

The principal scripts are:

temporal_aggregation_eval.py
    Performs controlled deterministic correctness validation against
    independently computed expected values.

temporal_aggregation_plot.py
    Reads the accepted evaluator outputs, verifies their integrity, produces
    thesis-oriented aggregate tables, and generates quantitative figures.

temporal_aggregation_qualitative.py
    Generates representative temporal-behavior evidence for window growth,
    sliding-window replacement, person-specific state, reset behavior, blink
    aggregation, dominant emotion, and Face Quality aggregation.

temporal_aggregation_component_test.py
    Runs the real PhysioTrack FaceAnalysis pipeline on existing single-person
    and multi-person integration fixtures, captures the actual upstream
    frame-level values and temporal summaries, and independently recomputes
    every current temporal summary field for numerical comparison.


Why No External Benchmark Dataset Is Required
---------------------------------------------
FaceTemporalAggregator does not estimate an unknown physical or semantic
quantity directly from raw media. It deterministically transforms already
computed frame-level measurements into temporal summaries.

Consequently, an external ground-truth dataset is not the appropriate primary
validation instrument for this component. Correctness can be established more
directly and rigorously by:

1. supplying controlled sequences with known expected outcomes,
2. independently recomputing the expected temporal summaries,
3. testing boundary and state-management behavior explicitly, and
4. confirming the same calculations during real FaceAnalysis execution on
   actual video data.

The scientific accuracy of the upstream predictive components remains covered
by their own component-specific validations. In particular, correct temporal
aggregation of an emotion label does not itself establish emotion-recognition
accuracy.


Current Temporal-Aggregation Contract
-------------------------------------
The current FaceTemporalAggregator uses a per-person rolling buffer.

The configured maximum number of retained frames is:

window_frames = max(1, round(FPS * window_sec))

Each tracked person has an independent temporal buffer.

For every available finite numerical signal in the current window, the
component calculates:

mean = arithmetic mean of valid values

std = population standard deviation of valid values

min = minimum valid value

max = maximum valid value

Unavailable values and non-finite numerical values are excluded according to
the implementation's availability and validity rules.

Blink aggregation reports the number of blink-event samples contained in the
current temporal window.

Emotion aggregation reports the dominant available emotion label in the
current temporal window.

Temporal state is reset when continuity for a tracked identity is explicitly
broken by the FaceAnalysis processing path.


Controlled Correctness Validation
---------------------------------
The controlled evaluator exercises the current FaceTemporalAggregator directly
using deterministic sequences whose expected results are computed
independently from the implementation.

The accepted validation covers eight groups of behavior:

1. Constructor and window derivation
   - valid FPS/window handling
   - rejection of zero or negative FPS
   - rejection of zero or negative window duration
   - exact frame-window derivation

2. Full numerical summary
   - head pose yaw, pitch, and roll
   - eye openness
   - geometric gaze x/y
   - mouth openness
   - mouth movement
   - Face Quality brightness
   - Face Quality sharpness
   - Face Quality face_area_ratio
   - mean, population standard deviation, minimum, and maximum
   - blink-event aggregation
   - dominant emotion

3. Availability and non-finite handling
   - unavailable upstream feature records
   - NaN
   - positive infinity
   - negative infinity
   - finite-value filtering

4. Sliding-window behavior
   - bounded window size
   - rolling replacement after saturation
   - correct window duration

5. Per-person isolation
   - independent temporal state for different tracked identities
   - absence of cross-person leakage

6. Reset behavior
   - reset of one tracked identity
   - preservation of another identity
   - complete/global reset

7. Rejected updates
   - instances without a tracked person ID
   - instances without face features

8. Empty numerical feature handling
   - correct None summaries when no finite available numerical value exists

Accepted controlled-validation results:

Validation cases: 8
Total checks: 150
Passed checks: 150
Failed checks: 0
Pass rate: 1.000000
Independent numerical comparisons: 106
Maximum numerical absolute error: 0

All independently checked controlled numerical values matched their expected
values exactly.


Quantitative Tables and Figures
-------------------------------
The plotting script reads only the accepted evaluator outputs.

It produces:

results/temporal_aggregation_thesis_table.csv
results/temporal_aggregation_thesis_table.md

results/figures/
    temporal_aggregation_validation_coverage.png
    temporal_aggregation_numeric_error.png

The accepted thesis table contains:

constructor_and_window:
    total checks: 5
    passed: 5
    failed: 0
    numeric checks: 1
    maximum numerical absolute error: 0

full_numeric_summary:
    total checks: 60
    passed: 60
    failed: 0
    numeric checks: 48
    maximum numerical absolute error: 0

availability_and_non_finite:
    total checks: 41
    passed: 41
    failed: 0
    numeric checks: 32
    maximum numerical absolute error: 0

sliding_window:
    total checks: 14
    passed: 14
    failed: 0
    numeric checks: 11
    maximum numerical absolute error: 0

person_isolation:
    total checks: 13
    passed: 13
    failed: 0
    numeric checks: 11
    maximum numerical absolute error: 0

reset_behavior:
    total checks: 4
    passed: 4
    failed: 0
    numeric checks: 1
    maximum numerical absolute error: 0

rejected_updates:
    total checks: 4
    passed: 4
    failed: 0
    numeric checks: 2
    maximum numerical absolute error: 0

empty_numeric_feature:
    total checks: 9
    passed: 9
    failed: 0
    numeric checks: 0
    maximum numerical absolute error: 0

The numerical-correctness figure reports all 106 independent numerical
comparisons together with the maximum observed absolute error.


Representative Temporal Evidence
--------------------------------
The qualitative/representative script provides visual evidence tailored to a
temporal deterministic component rather than conventional image-prediction
examples.

The accepted representative cases are:

case_01_window_growth.png
    Demonstrates growth of the temporal window before saturation and agreement
    between the observed rolling yaw mean and the independently expected mean.

case_02_sliding_window.png
    Demonstrates replacement of old samples after the configured rolling
    window reaches its maximum size.

case_03_person_isolation.png
    Demonstrates independent temporal state for two tracked identities with
    deliberately different trajectories.

case_04_reset_behavior.png
    Demonstrates a per-person temporal reset followed by a clean restart
    without leakage from pre-reset values.

case_05_blink_events.png
    Demonstrates blink-event aggregation within the current rolling window.

case_06_dominant_emotion.png
    Demonstrates dominant-emotion changes within the rolling window and their
    agreement with independently derived categorical majorities.

case_07_quality_brightness.png
    Demonstrates temporal aggregation of Face Quality brightness.

The corresponding manifest is stored at:

results/qualitative/
    temporal_aggregation_qualitative_manifest.csv

A combined contact sheet is stored at:

results/figures/
    temporal_aggregation_qualitative_contact_sheet.png


Real PhysioTrack Component Execution
------------------------------------
After the controlled validation, quantitative outputs, and representative
evidence were accepted, the real PhysioTrack pipeline was executed through
FaceAnalysis.

This execution verifies the actual project path rather than a local
reimplementation of FaceTemporalAggregator.

The component test enables the upstream analyses currently consumed by
FaceTemporalAggregator:

- tracking
- head pose
- landmarks
- Face Quality
- eye openness
- blink analysis
- geometric gaze
- mouth openness
- mouth motion
- emotion
- temporal aggregation

Learned gaze estimation and face-region segmentation are disabled in this
isolated component execution because the current FaceTemporalAggregator does
not summarize their outputs. Their exclusion therefore reduces unrelated work
without removing any input currently required by the target component.

The validated blink configuration is:

blink_threshold = 0.22
min_closed_frames = 3

The temporal window duration is:

temporal_window_sec = 5.0


Real Video Fixtures
-------------------
The real component execution automatically discovers supported video files
from the existing integration fixture directories:

validation/integration/test_data/single_person/

validation/integration/test_data/multi_person/

The accepted run used:

validation/integration/test_data/single_person/face_blink_pose.mp4

validation/integration/test_data/multi_person/multi_person2.mp4

These are software-validation fixtures rather than external accuracy
benchmarks.


Real Component Execution Results
--------------------------------
Single-person fixture:

FPS: 25.0
Reported frames: 429
Processed frames: 429
Frames with faces: 429
Face records: 429
Temporal window frames: 125
Duplicate person-ID frames: 0
Failed frame rows: 0
Failed checks: 0
Status: PASS

Multi-person fixture:

FPS: 25.0
Reported frames: 344
Processed frames: 344
Frames with faces: 344
Face records: 688
Temporal window frames: 125
Duplicate person-ID frames: 0
Failed frame rows: 0
Failed checks: 0
Status: PASS

Aggregate real-component results:

Videos discovered: 2
Videos passed: 2
Videos failed: 0
Processed frames: 773
Face records: 1117
Temporal-available records: 1117
Failed frame rows: 0
Total independent checks: 55850
Passed checks: 55850
Failed checks: 0
Maximum numerical absolute error: 2.131628207280301e-14
Overall status: PASS

The observed maximum numerical difference is at floating-point precision and
does not indicate a mathematical discrepancy.


Independent Real-Data Audit
---------------------------
The real component execution stores both:

1. the actual upstream frame-level values produced by FaceAnalysis, and
2. the actual FaceTemporalAggregator summaries.

For every temporal record, expected rolling statistics are independently
recomputed from the stored frame-level evidence.

The audit covers all currently supported numerical temporal summaries:

- head_pose.yaw
- head_pose.pitch
- head_pose.roll
- eyes.mean_openness
- gaze.mean_iris_x
- gaze.mean_iris_y
- mouth.openness
- mouth.movement
- quality.brightness
- quality.sharpness
- quality.face_area_ratio

For each numerical signal, the audit checks:

- mean
- population standard deviation
- minimum
- maximum

It additionally checks:

- temporal availability
- person ID
- window_frames
- window_sec
- blink events
- dominant emotion

The accepted run produced 1117 real temporal records and 55850 successful
checks with no failed row, no failed check, and no duplicate tracked-person ID
within a frame.

For the multi-person fixture, the two tracked identities were evaluated
independently throughout the run, providing direct evidence that temporal state
does not leak between tracked persons.


Output Organization
-------------------
The accepted output structure is:

validation/temporal_aggregation/
|-- temporal_aggregation_eval.py
|-- temporal_aggregation_plot.py
|-- temporal_aggregation_qualitative.py
|-- temporal_aggregation_component_test.py
|-- README_TEMPORAL_AGGREGATION.txt
`-- results/
    |-- temporal_aggregation_results.csv
    |-- temporal_aggregation_metrics.csv
    |-- temporal_aggregation_summary.txt
    |-- temporal_aggregation_thesis_table.csv
    |-- temporal_aggregation_thesis_table.md
    |-- figures/
    |   |-- temporal_aggregation_validation_coverage.png
    |   |-- temporal_aggregation_numeric_error.png
    |   `-- temporal_aggregation_qualitative_contact_sheet.png
    |-- qualitative/
    |   |-- temporal_aggregation_qualitative_manifest.csv
    |   |-- case_01_window_growth.png
    |   |-- case_02_sliding_window.png
    |   |-- case_03_person_isolation.png
    |   |-- case_04_reset_behavior.png
    |   |-- case_05_blink_events.png
    |   |-- case_06_dominant_emotion.png
    |   `-- case_07_quality_brightness.png
    `-- component_execution/
        |-- temporal_aggregation_component_results.csv
        |-- temporal_aggregation_component_checks.csv
        |-- temporal_aggregation_component_metrics.csv
        `-- temporal_aggregation_component_summary.json

The component-execution outputs are intentionally isolated under
results/component_execution/ so that their ownership and purpose remain
distinct from the controlled evaluator outputs.


Safe Rerun and Output Ownership
-------------------------------
The validation scripts use project-relative paths derived from their own file
location. No user-specific or machine-specific path is embedded in the
validation package.

The evaluator, plotting, qualitative, and component-execution stages own
different output sets.

The safe-rerun architecture follows the same project methodology used for the
other validated PhysioTrack components:

1. preflight required sources and inputs,
2. create a script-specific staging location before generating replacement
   outputs,
3. write new script-owned outputs to staging,
4. validate the staged artifacts,
5. promote only complete accepted staged outputs,
6. preserve previously accepted outputs if generation or validation fails,
7. use rollback protection during final replacement,
8. remove temporary staging/rollback material after completion.

No script is permitted to delete outputs owned by another validation script.

The component-execution script writes exclusively to:

results/component_execution/

It also removes only obsolete root-level component-execution files from the
earlier layout after a successful replacement run.

The integration fixtures are read-only inputs. The validation does not modify
benchmark datasets, integration media, or PhysioTrack source code.


Running the Validation
----------------------
Activate the PhysioTrack thesis environment:

conda activate PhysioTrack-Thesis

Change to the validation directory:

cd /d <project-path>\physiotrack\validation\temporal_aggregation

Run the controlled deterministic evaluator:

python temporal_aggregation_eval.py

Generate the quantitative tables and figures:

python temporal_aggregation_plot.py

Generate representative temporal evidence:

python temporal_aggregation_qualitative.py

Run the real PhysioTrack component execution:

python temporal_aggregation_component_test.py

For reproducibility, syntax can be checked before execution:

python -m py_compile temporal_aggregation_eval.py
python -m py_compile temporal_aggregation_plot.py
python -m py_compile temporal_aggregation_qualitative.py
python -m py_compile temporal_aggregation_component_test.py


Interpretation
--------------
The accepted results establish that the current PhysioTrack
FaceTemporalAggregator:

- derives its rolling-window size correctly,
- calculates the implemented numerical summaries correctly,
- handles unavailable and non-finite values correctly,
- maintains bounded sliding-window behavior,
- maintains independent temporal state per tracked person,
- supports explicit temporal reset without state leakage,
- aggregates blink events correctly,
- derives the dominant available emotion correctly,
- aggregates Face Quality descriptors correctly,
- produces real temporal outputs inside FaceAnalysis,
- preserves numerical correctness on single-person and multi-person video,
- stores outputs in reproducible structured result files.

The controlled evaluator produced exact agreement for all 106 numerical
comparisons.

The real FaceAnalysis execution produced 1117 temporal records and 55850
successful checks. The largest observed numerical difference was approximately
2.13e-14, which is consistent with ordinary floating-point arithmetic.

These findings support the conclusion that FaceTemporalAggregator is
mathematically and operationally correct for the validated implementation
contract.


Scientific Scope and Limitations
--------------------------------
This validation does not claim predictive accuracy for upstream components.

For example:

- correct aggregation of head-pose values does not independently validate
  head-pose estimation accuracy;
- correct aggregation of eye openness does not independently validate
  eye-openness accuracy;
- correct dominant-emotion aggregation does not establish emotion-recognition
  accuracy.

Those scientific questions belong to the corresponding component-specific
benchmark validations.

The current Temporal Aggregator summarizes the geometric gaze descriptor but
does not summarize learned GazeEstimator outputs. It also does not aggregate
face-region segmentation outputs. These are properties of the current
implementation and are not treated as missing outputs in this validation.

The validation therefore evaluates the complete temporal contract currently
implemented by FaceTemporalAggregator and does not invent aggregation fields
that the component does not provide.

Runtime is environment-dependent and is not treated as a scientific
performance metric in this validation.


Final Status
------------
The controlled deterministic validation, quantitative result audit,
representative temporal evidence, and real PhysioTrack component execution all
completed successfully.

Controlled validation:
8 validation cases
150 / 150 checks passed
106 independent numerical comparisons
maximum numerical absolute error = 0

Real PhysioTrack execution:
2 / 2 videos passed
773 processed frames
1117 real face/temporal records
55850 / 55850 checks passed
0 failed frame rows
0 failed checks
maximum numerical absolute error =
2.131628207280301e-14

Overall validation status:
PASS

The current FaceTemporalAggregator is accepted as mathematically correct,
operationally integrated through the real FaceAnalysis execution path, and
reproducible for the validated thesis scope.


Artifacts to Preserve
---------------------
Preserve the following final validation artifacts:

- temporal_aggregation_eval.py
- temporal_aggregation_plot.py
- temporal_aggregation_qualitative.py
- temporal_aggregation_component_test.py
- README_TEMPORAL_AGGREGATION.txt

- results/temporal_aggregation_results.csv
- results/temporal_aggregation_metrics.csv
- results/temporal_aggregation_summary.txt
- results/temporal_aggregation_thesis_table.csv
- results/temporal_aggregation_thesis_table.md

- results/figures/temporal_aggregation_validation_coverage.png
- results/figures/temporal_aggregation_numeric_error.png
- results/figures/temporal_aggregation_qualitative_contact_sheet.png

- results/qualitative/temporal_aggregation_qualitative_manifest.csv
- results/qualitative/case_01_window_growth.png
- results/qualitative/case_02_sliding_window.png
- results/qualitative/case_03_person_isolation.png
- results/qualitative/case_04_reset_behavior.png
- results/qualitative/case_05_blink_events.png
- results/qualitative/case_06_dominant_emotion.png
- results/qualitative/case_07_quality_brightness.png

- results/component_execution/temporal_aggregation_component_results.csv
- results/component_execution/temporal_aggregation_component_checks.csv
- results/component_execution/temporal_aggregation_component_metrics.csv
- results/component_execution/temporal_aggregation_component_summary.json

Temporary staging directories, rollback directories, obsolete root-level
component-execution outputs, and superseded diagnostic artifacts are not part of
the final validation package.
