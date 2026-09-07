PhysioTrack Mouth Movement and Velocity Validation
=================================================

Purpose
-------
This validation evaluates the temporal mouth-motion outputs produced by the
PhysioTrack face-analysis module:

- mouth_movement
- mouth_velocity

The validation target is the PhysioTrack MouthMovement component operating on
accepted per-frame PhysioTrack mouth-openness predictions.

This is a continuous temporal regression evaluation. It is not a mouth-state
classification task.

After the quantitative benchmark, plotting, and qualitative stages are
accepted, a separate isolated component-execution test verifies the current
MouthMovement and mouth-velocity outputs through the real PhysioTrack
FaceAnalysis pipeline. The isolated execution is software-path and
reproducibility evidence; it is not a second accuracy benchmark.

Validation Location
-------------------
The validation package is located in:

validation/mouth_movement_velocity/

The principal scripts are:

felt_ravdess_mouth_movement_velocity_eval.py
    Runs the quantitative temporal benchmark, verifies the accepted
    mouth-openness source results against FELT references, evaluates
    MouthMovement across every valid consecutive frame transition, and writes
    per-frame, per-actor, and summary outputs.

felt_ravdess_mouth_movement_velocity_plot.py
    Independently recomputes and verifies the stored temporal metrics,
    validates movement/velocity/FPS relationships and per-actor consistency,
    and generates thesis-oriented tables and quantitative figures.

felt_ravdess_mouth_movement_velocity_qualitative.py
    Selects deterministic representative and challenging temporal transitions,
    re-evaluates those frame pairs from the original RAVDESS videos, verifies
    agreement with the accepted quantitative results, and generates annotated
    qualitative examples.

mouth_movement_velocity_component_test.py
    Executes the current MouthMovement and mouth-velocity path through the real
    PhysioTrack FaceAnalysis pipeline over the complete paired FELT/RAVDESS
    speech population, records real numerical component outputs, verifies
    temporal initialization and continuity-reset semantics, and does not
    compute a second set of benchmark accuracy metrics.

Datasets and Scope
------------------
The validation uses the same paired FELT/RAVDESS speech subset used for the
accepted PhysioTrack mouth-openness validation.

FELT speech annotations:

datasets/FELT/raw_motion_speech/

RAVDESS speech videos:

datasets/RAVDESS/Video_Speech/

The evaluated scope contains:

Actors: 24
Paired speech trials: 1440
Unique annotated frames: 158286
Initialization frames: 1440
Evaluated consecutive frame transitions: 156846

FELT provides per-frame 68-point facial landmarks and face rectangles. RAVDESS
provides the corresponding audiovisual speech videos.

Relationship to the Accepted Mouth-Openness Validation
------------------------------------------------------
The temporal benchmark does not rerun full face-landmark and mouth-openness
inference over all RAVDESS videos.

Instead, it uses the already accepted per-frame PhysioTrack mouth-openness
outputs stored in:

validation/mouth_openness/results/
felt_ravdess_mouth_openness_per_frame.csv

Those accepted outputs were produced by running PhysioTrack FaceLandmarks and
MouthOpenness on all 1440 paired RAVDESS speech videos and all 158286 FELT
annotated frames.

The present validation therefore covers the same complete set of 1440 speech
trials and 158286 annotated frames while avoiding unnecessary repetition of the
already accepted full-video mouth-openness inference.

For independence, the temporal evaluator re-reads the FELT annotations,
reconstructs the FELT mouth-openness reference for every frame, and verifies
that it matches the accepted mouth-openness benchmark before deriving temporal
ground truth.

The qualitative validation additionally returns to the original RAVDESS videos
and reruns PhysioTrack FaceLandmarks, MouthOpenness, and MouthMovement on the
selected frame pairs.

The isolated component-execution stage is deliberately separate from the
temporal benchmark. It runs the current real FaceAnalysis path on the same
complete 1440-trial, 158286-frame population with FaceLandmarks, MouthOpenness,
and MouthMovement enabled. Its purpose is to verify the implemented software
path and temporal semantics rather than to produce another FELT accuracy
result.

Mouth-Openness Reference
------------------------
The temporal reference is derived from the same primary FELT mouth-openness
definition used in the accepted mouth-openness benchmark:

FELT openness_t = d_t(62,66) / d_t(48,54)

where d_t(a,b) is the Euclidean landmark distance between points a and b at
frame t.

This geometry corresponds to central vertical lip separation normalized by
horizontal mouth-corner width.

Mouth Movement Definition
-------------------------
PhysioTrack defines mouth movement as the absolute change in mouth openness
between the current valid frame and the previous valid frame.

For consecutive frames:

movement_t = |openness_t - openness_(t-1)|

The FELT temporal reference is therefore:

FELT movement_t =
    |FELT openness_t - FELT openness_(t-1)|

The PhysioTrack prediction is produced by the actual MouthMovement component
using accepted PhysioTrack mouth-openness values in temporal order.

Mouth Velocity Definition
-------------------------
PhysioTrack defines mouth velocity as mouth movement divided by elapsed time.

For a frame gap g:

elapsed_time = g / FPS

velocity_t = movement_t / elapsed_time

All evaluated FELT transitions are consecutive, so:

g = 1

The RAVDESS speech videos use the locked frame rate:

FPS = 30000 / 1001
    = 29.970029970030 frames/second

Therefore, for this benchmark:

velocity_t = movement_t * 29.970029970030

The evaluator uses the actual MouthMovement implementation rather than
replacing the PhysioTrack prediction with an external reimplementation.

Temporal Initialization
-----------------------
MouthMovement has no previous sample at the start of a sequence.

Accordingly, for the first valid frame of every trial PhysioTrack returns:

mouth_movement = 0
mouth_velocity = 0

The validation verifies this initialization behavior for all 1440 trials.

These initialization frames are recorded in the detailed result file but are
excluded from regression metrics because no true frame-to-frame transition
exists at the first frame.

MouthMovement also treats a missing valid mouth input as a temporal boundary.
When continuity is broken, the next valid sample starts a new temporal segment
and therefore returns:

mouth_movement = 0
mouth_velocity = 0

Subsequent valid samples in that segment resume normal frame-to-frame movement
and velocity computation. The isolated component smoke test explicitly verifies
this continuity-reset behavior through the real FaceAnalysis pipeline.

Evaluation Protocol
-------------------
For every one of the 1440 paired speech trials:

1. The FELT annotation CSV is loaded.
2. Duplicate frame annotations are resolved deterministically.
3. FELT frame IDs are verified to be contiguous and zero-based.
4. The FELT primary mouth-openness reference is independently recomputed for
   every frame.
5. The accepted PhysioTrack mouth-openness result for the identical
   actor/trial/frame key is loaded.
6. The independently recomputed FELT reference is checked against the stored
   accepted benchmark reference.
7. A new MouthMovement state is created for the trial.
8. Accepted PhysioTrack mouth-openness values are supplied sequentially to the
   actual MouthMovement component.
9. The first frame is verified as an initialization frame with zero movement
   and velocity.
10. For every later consecutive frame, FELT movement and velocity references
    are derived from the frame-to-frame FELT openness change.
11. PhysioTrack mouth_movement and mouth_velocity are compared with their FELT
    temporal references.
12. Per-transition and per-actor results are stored.
13. Aggregate regression metrics are computed independently for movement and
    velocity.

State is restarted for each trial, preventing temporal information from one
video from leaking into another.

Duplicate Annotation Handling
-----------------------------
The FELT speech subset contains two additional face annotations associated with
duplicated frame identifiers in one trial.

Duplicate frame identifiers are resolved using the same deterministic rule as
the accepted mouth-openness validation:

1. Retain the annotation with the largest FaceRect area.
2. If FaceRect areas are equal, retain the annotation with the highest
   FaceScore.

After duplicate resolution, the dataset contains 158286 unique annotated
frames.

Temporal Coverage
-----------------
The validated dataset structure is:

Raw FELT rows: 158288
Unique annotated frames: 158286
Trials: 1440
Initialization frames: 1440
Evaluated transitions: 156846

Every evaluated transition has:

frame_gap = 1

and:

elapsed_time = 1 / 29.970029970030
             = 0.0333666667 seconds approximately

Safe Rerun and Dataset Protection
---------------------------------
The final validation scripts use a staged safe-rerun workflow:

1. Validate the locked dataset, source-result dependency, and protocol.
2. Create a temporary staging area under
   validation/mouth_movement_velocity/results/ before generative work begins.
3. Generate only the outputs owned by the active script inside staging.
4. Re-read and validate the staged outputs.
5. Replace accepted script-owned outputs only after staged validation passes.
6. Preserve or restore prior accepted outputs if final installation fails.
7. Remove temporary staging artifacts after successful completion or failure.

The quantitative evaluator supports:

python felt_ravdess_mouth_movement_velocity_eval.py --preflight-only

for dataset/source-result verification without temporal evaluation, and:

python felt_ravdess_mouth_movement_velocity_eval.py --validate-existing-results-only

for complete serialized-result validation without regenerating the temporal
benchmark.

FELT and RAVDESS are treated as read-only benchmark inputs. The isolated
component execution additionally records file-size and modification-time
inventories before and after full execution and requires both dataset
inventories to remain unchanged.

Quantitative Metrics
--------------------
The same complementary regression metrics are reported for both mouth movement
and mouth velocity:

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

Because every evaluated transition has the same frame interval, mouth velocity
is a positive constant scaling of mouth movement. Consequently, Pearson,
Spearman, and Lin CCC are numerically identical for movement and velocity in
this benchmark, while absolute-error metrics are scaled by the frame rate.

Mouth Movement Results
----------------------
Evaluated transitions: 156846

MAE: 0.011517
RMSE: 0.019764
Median absolute error: 0.005988
Standard deviation of absolute error: 0.016061
90th percentile absolute error: 0.028694
95th percentile absolute error: 0.041387
Mean signed error: 0.000249
Pearson r: 0.827139
Spearman rho: 0.669786
Lin CCC: 0.826886

The near-zero mean signed error indicates minimal overall directional bias.

The Pearson correlation and Lin CCC above 0.82 indicate strong temporal
agreement between PhysioTrack and the FELT landmark-derived frame-to-frame
movement reference.

The lower correlation relative to the static mouth-openness benchmark is
expected for a temporal difference measure because frame-to-frame
differencing amplifies small landmark-estimation variations and local jitter.

Mouth Velocity Results
----------------------
Evaluated transitions: 156846

MAE: 0.345156
RMSE: 0.592318
Median absolute error: 0.179446
Standard deviation of absolute error: 0.481360
90th percentile absolute error: 0.859954
95th percentile absolute error: 1.240356
Mean signed error: 0.007465
Pearson r: 0.827139
Spearman rho: 0.669786
Lin CCC: 0.826886

The velocity agreement coefficients are identical to the movement agreement
coefficients because the entire benchmark uses one constant frame rate and
velocity is therefore a fixed positive scaling of movement.

Per-Actor Analysis
------------------
Per-actor temporal results are stored in:

results/felt_ravdess_mouth_movement_velocity_per_actor.csv

The thesis-oriented per-actor table is stored in:

results/felt_ravdess_mouth_movement_velocity_per_actor_thesis_table.csv

All 24 actors are represented.

The observed per-actor movement MAE remains within a relatively narrow range,
with the lowest value approximately 0.0087 and the highest approximately
0.0150. This indicates that the aggregate result is not dominated by a single
subject, although measurable inter-subject variation remains.

Quantitative Consistency Checks
-------------------------------
The plotting script independently verifies the quantitative result package
before generating figures or tables.

The checks include:

- exactly 158286 per-frame rows
- exactly 1440 initialization rows
- exactly 156846 evaluated transitions
- exactly 24 actors
- unique actor/trial/frame keys
- finite required numerical values
- zero MouthMovement outputs for every initialization frame
- frame_gap equal to 1 for every evaluated transition
- elapsed time consistent with the locked FPS
- FELT velocity equal to FELT movement multiplied by FPS
- PhysioTrack velocity equal to PhysioTrack movement multiplied by FPS
- stored signed and absolute errors equal independently recomputed errors
- summary metrics equal independently recomputed per-frame metrics
- per-actor metrics equal independently recomputed actor-level metrics
- identical Pearson, Spearman, and Lin CCC values for movement and velocity
  under the constant-FPS scalar transformation

The accepted plotting run reported:

Quantitative result consistency: PASS

Quantitative Figures
--------------------
The plotting script generates four quantitative figures:

felt_ravdess_mouth_movement_agreement.png
    Density-based agreement plot comparing FELT-derived mouth movement and
    PhysioTrack mouth movement against the identity line.

felt_ravdess_mouth_velocity_agreement.png
    Density-based agreement plot comparing FELT-derived mouth velocity and
    PhysioTrack mouth velocity against the identity line.

felt_ravdess_mouth_movement_velocity_error_distribution.png
    Signed-error distributions for movement and velocity, including zero-error
    and mean-error reference lines.

felt_ravdess_mouth_movement_velocity_per_actor.png
    Per-actor mouth-movement MAE with the transition-weighted overall MAE shown
    for comparison.

Qualitative Temporal Validation
-------------------------------
The qualitative validation uses temporal transitions rather than isolated
single frames.

Eight deterministic examples are selected from the accepted quantitative
result set:

- low movement
- medium movement
- high movement
- very high movement
- representative error
- challenging underestimate
- challenging overestimate
- representative transition from the actor with the highest movement MAE

For every selected transition, the qualitative script:

1. loads frame t-1 and frame t from the original RAVDESS video
2. loads the corresponding FELT annotations
3. independently recomputes FELT mouth openness for both frames
4. reruns PhysioTrack FaceLandmarks and MouthOpenness on both frames
5. initializes a fresh MouthMovement instance
6. processes the previous and current PhysioTrack openness values
7. verifies the rerun mouth movement and velocity against the accepted
   quantitative outputs
8. visualizes both temporal frames side by side

Each annotated example displays:

- previous and current RAVDESS frames
- FELT FaceRect initialization
- FELT mouth geometry
- PhysioTrack mouth geometry
- FELT and PhysioTrack mouth openness for each frame
- FELT and PhysioTrack mouth movement
- FELT and PhysioTrack mouth velocity
- movement absolute error

The qualitative rerun reproduced the accepted temporal outputs to numerical
floating-point precision.

The selected examples include both close agreement and meaningful failure
modes, making the qualitative evidence representative rather than
success-only.

The combined qualitative figure is stored as:

results/figures/
felt_ravdess_mouth_movement_velocity_qualitative_examples.png

The detailed selection record is stored as:

results/qualitative/
felt_ravdess_mouth_movement_velocity_qualitative_selection.csv

Individual annotated transition images are stored under:

results/qualitative/annotated_transitions/

Interpretation of Qualitative Examples
--------------------------------------
The qualitative figures document agreement with a landmark-derived temporal
reference.

They should not be interpreted as proof that either FELT or PhysioTrack is a
direct physical measurement of lip motion.

FELT and PhysioTrack use different facial-landmark schemes, so visible
disagreement can result from differences in landmark localization and
anatomical point definitions as well as from temporal estimation error.

The correct interpretation is agreement or disagreement with the defined FELT
landmark-derived temporal benchmark.

Isolated Component Execution Verification
-----------------------------------------
The accepted isolated component-execution script is:

mouth_movement_velocity_component_test.py

It runs the current PhysioTrack FaceAnalysis pipeline with the following
configuration.

Enabled components:

- FaceLandmarks
- MouthOpenness
- MouthMovement

Disabled unrelated optional components:

- tracking
- head pose
- quality
- eye openness
- blink
- legacy gaze
- learned gaze estimation
- emotion
- face regions
- temporal aggregation

The accepted FELT FaceRect is supplied as controlled upstream localization.
This prevents face-detector variability from confounding the isolated
MouthMovement software-path verification while retaining the required real
FaceLandmarks and MouthOpenness prerequisites.

The isolated execution covers:

Actors: 24
Paired speech trials: 1440
Raw FELT annotation rows: 158288
Unique annotated frames: 158286
Duplicate annotation rows resolved: 2
Locked FPS: 29.970029970030

For every accepted frame, the script stores real numerical FaceAnalysis
outputs including:

- actor, trial, frame, timestamp, and FPS
- frame dimensions
- accepted FaceRect and FaceScore
- duplicate-candidate count
- landmark availability and landmark count
- mouth-openness availability
- mouth_openness, mouth_width, and mouth_height
- MouthMovement availability
- mouth_movement
- mouth_velocity
- temporal-segment-start status
- explicit execution status and failure reason

The isolated validator independently checks that:

mouth_openness = mouth_height / mouth_width

The first valid sample of every temporal segment must satisfy:

mouth_movement = 0
mouth_velocity = 0

For subsequent valid samples:

mouth_movement_t =
    |mouth_openness_t - mouth_openness_(t-1)|

and, under the locked contiguous-frame FPS:

mouth_velocity_t =
    mouth_movement_t * 29.970029970030

The dedicated smoke test also introduces a missing-input gap through the real
FaceAnalysis path and verifies that the first valid post-gap sample restarts at
zero movement and zero velocity.

The accepted full isolated run produced:

Landmarks available rows: 158286
MouthOpenness available rows: 158286
MouthMovement available rows: 158286
Temporal segment starts: 1440
Execution failures: 0
Component unavailable rows: 0
Overall status: PASS
Runtime: 2103.66 seconds (approximately 35.06 minutes)

Runtime is environment-dependent and is not a scientific performance metric.

A frame-by-frame audit against the accepted temporal benchmark confirmed that
the isolated real-pipeline MouthOpenness, MouthMovement, and mouth-velocity
outputs reproduce the accepted temporal result sequence to floating-point
precision across the complete 158286-frame population.

Git-Safe Isolated Result Handling
---------------------------------
The isolated result writer checks the generated CSV size before final
installation.

If the complete CSV is at or below 90 MiB, it remains:

results/component_execution/
mouth_movement_velocity_component_results.csv

If the file would exceed 90 MiB, it is split automatically into sequential
parts:

mouth_movement_velocity_component_results_part001.csv
mouth_movement_velocity_component_results_part002.csv
...

Splitting is allowed only between complete actor groups. An actor is never
split across multiple result files. The summary JSON records every generated
filename, row count, actor boundary, and file size.

The accepted run produced one result CSV containing 158286 rows with a size of
approximately 38.06 MiB, so no split was required.

The isolated execution summary is stored as:

results/component_execution/
mouth_movement_velocity_component_summary.json

Run Order
---------
Activate the thesis environment:

conda activate PhysioTrack-Thesis

From the component validation directory:

cd /d <project-path>\physiotrack\validation\mouth_movement_velocity

Optional syntax verification:

python -m py_compile felt_ravdess_mouth_movement_velocity_eval.py felt_ravdess_mouth_movement_velocity_plot.py felt_ravdess_mouth_movement_velocity_qualitative.py mouth_movement_velocity_component_test.py

Recommended clean reproduction order:

1. Quantitative preflight:

   python felt_ravdess_mouth_movement_velocity_eval.py --preflight-only

2. Existing-result validation when accepted outputs are already present:

   python felt_ravdess_mouth_movement_velocity_eval.py --validate-existing-results-only

3. Full temporal benchmark:

   python felt_ravdess_mouth_movement_velocity_eval.py

4. Independent quantitative tables and figures:

   python felt_ravdess_mouth_movement_velocity_plot.py

5. Deterministic temporal qualitative validation:

   python felt_ravdess_mouth_movement_velocity_qualitative.py

6. Isolated component preflight:

   python mouth_movement_velocity_component_test.py --preflight-only

7. Real FaceAnalysis smoke test, including explicit continuity-reset
   verification:

   python mouth_movement_velocity_component_test.py --smoke-test --smoke-count 3

8. Full isolated component execution:

   python mouth_movement_velocity_component_test.py

The quantitative benchmark and isolated component execution serve different
purposes. The benchmark reports FELT-derived temporal agreement metrics. The
isolated execution verifies the current implemented FaceAnalysis software path
and stores real numerical component outputs without generating a second
accuracy result.

Output Ownership
----------------
The scripts have separate output ownership:

- the evaluator owns quantitative per-frame, per-actor, and summary outputs
- the plotting script owns thesis tables and quantitative figures
- the qualitative script owns qualitative selections, annotated transitions,
  and the combined qualitative figure
- the isolated component script owns only component-execution CSV/CSV parts
  and the component summary JSON

Each script owns only its designated outputs. New generative outputs are
created in staging and validated before the corresponding accepted artifacts
are replaced.

Output Structure
----------------
The final package is organized as:

validation/mouth_movement_velocity/
|-- felt_ravdess_mouth_movement_velocity_eval.py
|-- felt_ravdess_mouth_movement_velocity_plot.py
|-- felt_ravdess_mouth_movement_velocity_qualitative.py
|-- mouth_movement_velocity_component_test.py
|-- README_MOUTH_MOVEMENT_VELOCITY.txt
`-- results/
    |-- felt_ravdess_mouth_movement_velocity_per_frame.csv
    |-- felt_ravdess_mouth_movement_velocity_per_actor.csv
    |-- felt_ravdess_mouth_movement_velocity_summary.txt
    |-- felt_ravdess_mouth_movement_velocity_thesis_table.csv
    |-- felt_ravdess_mouth_movement_velocity_per_actor_thesis_table.csv
    |-- figures/
    |   |-- felt_ravdess_mouth_movement_agreement.png
    |   |-- felt_ravdess_mouth_velocity_agreement.png
    |   |-- felt_ravdess_mouth_movement_velocity_error_distribution.png
    |   |-- felt_ravdess_mouth_movement_velocity_per_actor.png
    |   `-- felt_ravdess_mouth_movement_velocity_qualitative_examples.png
    |-- qualitative/
    |   |-- felt_ravdess_mouth_movement_velocity_qualitative_selection.csv
    |   `-- annotated_transitions/
    |       `-- eight annotated temporal-transition PNG examples
    `-- component_execution/
        |-- mouth_movement_velocity_component_results.csv
        `-- mouth_movement_velocity_component_summary.json

Reproducibility
---------------
All dataset paths are resolved from the project structure rather than from
machine-specific absolute paths.

The validation scripts treat FELT and RAVDESS as read-only inputs.

The temporal evaluator depends on the accepted mouth-openness per-frame result
file produced by:

validation/mouth_openness/felt_ravdess_mouth_openness_eval.py

This dependency is intentional. It avoids unnecessary repetition of full-video
landmark and mouth-openness inference while preserving complete coverage of the
same 1440 trials and 158286 annotated frames.

The qualitative script independently reruns the original video frames selected
for qualitative inspection and verifies that the current PhysioTrack outputs
match the accepted quantitative results.

The isolated component execution independently exercises the current real
FaceAnalysis path over the complete paired speech population, verifies the
accepted FaceLandmarks model SHA256, enforces controlled FELT FaceRects,
checks temporal initialization and continuity-reset semantics, validates staged
serialized outputs before installation, and applies Git-safe result handling.

Methodological Qualifications
-----------------------------
Several qualifications are important when interpreting this validation.

First, the temporal ground truth is derived from FELT facial landmarks rather
than from a direct physical measurement of lip displacement or lip velocity.

Second, the benchmark measures the temporal change of a normalized mouth-
openness ratio. The reported mouth velocity is therefore the rate of change of
that dimensionless openness ratio per second, not a physical velocity in
millimeters per second.

Third, FELT and MediaPipe use different facial-landmark schemes. The benchmark
therefore evaluates agreement between geometrically corresponding but not
identical landmark definitions.

Fourth, the quantitative temporal benchmark reuses accepted per-frame
PhysioTrack mouth-openness outputs instead of repeating the full RAVDESS video
inference. The accepted source outputs were produced on the full paired speech
subset, and the temporal evaluator independently verifies the FELT reference
for every actor/trial/frame key before computing temporal quantities.

Fifth, all quantitative benchmark transitions are consecutive and all videos
use the same frame rate. The reported benchmark accuracy therefore represents
the normal contiguous-frame operating condition of this dataset. Irregular
frame gaps are not assigned separate benchmark accuracy metrics. However, the
isolated real-pipeline smoke test explicitly verifies the required continuity
reset after a missing input, with the first valid post-gap sample restarting at
zero movement and zero velocity.

Finally, frame-to-frame differentiation is inherently more sensitive to
landmark jitter than static mouth-openness estimation. The temporal metrics
should therefore be interpreted independently rather than expected to match
the static mouth-openness agreement coefficients.

The isolated component execution is software-path and reproducibility evidence,
not a second ground-truth accuracy benchmark. Its runtime, availability counts,
segment-start counts, and result-file size must not be interpreted as
additional FELT regression metrics.

Scientific Interpretation
-------------------------
The results support the conclusion that PhysioTrack MouthMovement provides a
stable and reproducible estimate of frame-to-frame mouth-motion magnitude on
the paired FELT/RAVDESS speech subset.

Across 156846 consecutive temporal transitions, movement achieved a Pearson
correlation of 0.827139 and a Lin CCC of 0.826886 with the FELT landmark-derived
reference, with a near-zero mean signed error of 0.000249.

Mouth velocity showed the corresponding scaled error values and identical
association and concordance coefficients, as expected under the fixed
29.970029970030 FPS protocol.

The quantitative result is supported by independent result-file consistency
checks, per-actor analysis, temporal qualitative examples spanning low to very
high movement as well as representative underestimation and overestimation
cases, and a separate full-population isolated FaceAnalysis execution with
complete MouthMovement availability and zero execution failures.

The scientifically appropriate description is:

controlled temporal validation of PhysioTrack MouthMovement and its
mouth_movement and mouth_velocity outputs on the paired FELT/RAVDESS speech
subset using a FELT landmark-derived frame-to-frame mouth-openness reference,
complemented by isolated real-pipeline MouthMovement software-path
verification.


Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- felt_ravdess_mouth_movement_velocity_eval.py
- felt_ravdess_mouth_movement_velocity_plot.py
- felt_ravdess_mouth_movement_velocity_qualitative.py
- mouth_movement_velocity_component_test.py
- README_MOUTH_MOVEMENT_VELOCITY.txt
- results/felt_ravdess_mouth_movement_velocity_per_frame.csv
- results/felt_ravdess_mouth_movement_velocity_per_actor.csv
- results/felt_ravdess_mouth_movement_velocity_summary.txt
- results/felt_ravdess_mouth_movement_velocity_thesis_table.csv
- results/felt_ravdess_mouth_movement_velocity_per_actor_thesis_table.csv
- results/figures/felt_ravdess_mouth_movement_agreement.png
- results/figures/felt_ravdess_mouth_velocity_agreement.png
- results/figures/felt_ravdess_mouth_movement_velocity_error_distribution.png
- results/figures/felt_ravdess_mouth_movement_velocity_per_actor.png
- results/figures/felt_ravdess_mouth_movement_velocity_qualitative_examples.png
- results/qualitative/felt_ravdess_mouth_movement_velocity_qualitative_selection.csv
- results/qualitative/annotated_transitions/
- results/component_execution/mouth_movement_velocity_component_results.csv
- results/component_execution/mouth_movement_velocity_component_summary.json

If a future isolated result exceeds the configured 90 MiB Git-safe threshold,
the single component-results CSV is replaced by the generated
mouth_movement_velocity_component_results_partNNN.csv files recorded in the
summary manifest.

Generated caches and obsolete temporary diagnostic files are not part of the
final validation deliverables.
