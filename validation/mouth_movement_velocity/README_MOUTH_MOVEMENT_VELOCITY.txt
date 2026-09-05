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

Run Order
---------
Run the scripts from the repository root using the project environment:

python validation/mouth_movement_velocity/felt_ravdess_mouth_movement_velocity_eval.py
python validation/mouth_movement_velocity/felt_ravdess_mouth_movement_velocity_plot.py
python validation/mouth_movement_velocity/felt_ravdess_mouth_movement_velocity_qualitative.py

The scripts have separate output ownership:

- the evaluator owns quantitative per-frame, per-actor, and summary outputs
- the plotting script owns thesis tables and quantitative figures
- the qualitative script owns qualitative selections, annotated transitions,
  and the combined qualitative figure

Each script cleans only the artifacts that it owns.

Output Structure
----------------
The final package is organized as:

validation/mouth_movement_velocity/
|-- felt_ravdess_mouth_movement_velocity_eval.py
|-- felt_ravdess_mouth_movement_velocity_plot.py
|-- felt_ravdess_mouth_movement_velocity_qualitative.py
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
    `-- qualitative/
        |-- felt_ravdess_mouth_movement_velocity_qualitative_selection.csv
        `-- annotated_transitions/
            `-- eight annotated temporal-transition PNG examples

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

Fifth, all evaluated transitions are consecutive and all videos use the same
frame rate. The current benchmark therefore validates MouthMovement under the
normal contiguous-frame operating condition represented by this dataset. It
does not constitute a separate stress test of irregular frame gaps.

Finally, frame-to-frame differentiation is inherently more sensitive to
landmark jitter than static mouth-openness estimation. The temporal metrics
should therefore be interpreted independently rather than expected to match
the static mouth-openness agreement coefficients.

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
checks, per-actor analysis, and temporal qualitative examples spanning low to
very high movement as well as representative underestimation and
overestimation cases.

The scientifically appropriate description is:

controlled temporal validation of PhysioTrack MouthMovement and its
mouth_movement and mouth_velocity outputs on the paired FELT/RAVDESS speech
subset using a FELT landmark-derived frame-to-frame mouth-openness reference.
