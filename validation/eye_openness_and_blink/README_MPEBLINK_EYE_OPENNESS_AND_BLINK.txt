MPEBlink 2.0 Eye Openness and Blink Validation
================================================

Purpose
-------
This README describes how to reproduce the PhysioTrack EyeOpenness and blink
validation on the MPEBlink 2.0 benchmark.

The validation measures two related aspects:

1. Whether the PhysioTrack EyeOpenness descriptor separates blink from
   non-blink frames.
2. How well the current threshold-based temporal blink logic detects blink
   events and estimates blink count, rate, and duration.

The validation package also generates quantitative thesis artifacts and
qualitative benchmark videos and images using the same accepted evaluation
protocol.

Dataset
-------
Name:
MPEBlink 2.0

Official project:
https://github.com/wenzhengzeng/MPEblink

Dataset download:
https://huggingface.co/datasets/Tao-HUST/MPEblink2.0

Task:
Multi-person eye-blink analysis in unconstrained video

Benchmark splits used:

- Validation split for blink-parameter selection
- Test split for final quantitative evaluation and qualitative evidence

The final reported benchmark uses the official test videos and per-person
annotations. Training data are not used for model training or parameter fitting
in this validation.

Dataset Location
----------------
Place the extracted dataset under the project-level datasets directory:

datasets/MPEBlink2/mpeblink2.0

The expected structure is:

datasets/
└── MPEBlink2/
    └── mpeblink2.0/
        ├── train/
        ├── val/
        │   ├── <video_id>/
        │   │   ├── video.mp4
        │   │   └── annotation_WFLW.json
        │   └── ...
        └── test/
            ├── <video_id>/
            │   ├── video.mp4
            │   └── annotation_WFLW.json
            └── ...

The dataset directory is treated as benchmark input only. Validation scripts do
not write generated results into the dataset directory.

Validation Files
----------------
The validation package is located in:

physiotrack/validation/eye_openness_and_blink/

Required scripts:

1. mpeblink_blink_eval.py

   Runs the quantitative benchmark. It performs dataset integrity checks,
   extracts PhysioTrack EyeOpenness values from the annotated target-person
   sequences, applies the frozen blink-detection configuration, computes the
   event-level and eye-openness metrics, and writes the raw test results.

2. mpeblink_blink_plot.py

   Reads the quantitative outputs, verifies their internal consistency, and
   generates the final thesis table and summary performance figure.

3. mpeblink_blink_qualitative.py

   Generates deterministic qualitative benchmark evidence from selected
   MPEBlink 2.0 test person sequences. It re-runs the selected sequences using
   the same accepted benchmark protocol, verifies the resulting event counts
   against the accepted quantitative sequence-results CSV, and then creates
   annotated MP4 videos, representative PNG images, a selection CSV, and a
   combined qualitative figure.

Path Handling
-------------
The validation scripts do not contain a user-specific absolute dataset path.

They determine the repository and project locations automatically from the
script location and expect the dataset at:

datasets/MPEBlink2/mpeblink2.0

The scripts can therefore be used on another machine without editing the
dataset path as long as the documented project structure is preserved.

Evaluated Components
--------------------
PhysioTrack FaceLandmarks
    Backend used by the validation:
    MediaPipe Face Landmarker

PhysioTrack EyeOpenness
    Output used by the validation:
    mean_openness

Blink detection
    Method:
    Fixed threshold applied to mean_openness with a minimum number of
    consecutive closed frames

Face Initialization
-------------------
MPEBlink 2.0 ground-truth per-person face bounding boxes are supplied directly
to PhysioTrack FaceLandmarks.

This isolates the EyeOpenness descriptor and blink logic from face-detection
errors. The reported metrics therefore evaluate eye-openness and blink behavior
under ground-truth face initialization rather than an end-to-end
face-detection-plus-blink pipeline.

Missing or invalid ground-truth face boxes are not treated as successful
EyeOpenness samples. Landmark failures are also tracked explicitly.

Annotation Handling
-------------------
Each MPEBlink person annotation contains frame-level face bounding boxes and
blink-event annotations.

For blink events, the first two values are interpreted as inclusive start and
end frame indices. Additional event metadata are not used by this evaluator.

The event-level evaluation preserves the annotated event boundaries. For
frame-level blink/non-blink masks used in the EyeOpenness analysis, event
intervals are clipped safely to the valid sequence frame range.

The final test split contains one blink annotation whose event boundary extends
outside the declared sequence frame range. This annotation is retained for the
event-level protocol and reported as a dataset-integrity observation rather than
being silently deleted.

Parameter Selection
-------------------
The MPEBlink validation split is used only for blink-parameter selection.

The calibration search evaluates:

Blink threshold:
0.20 to 0.80 in increments of 0.02

Minimum consecutive closed frames:
1, 2, 3

Event matching threshold:
temporal IoU >= 0.50

The selected configuration is:

Blink threshold:
0.44

Minimum consecutive closed frames:
3

These values are frozen before evaluation of the test split. Test results are
not used to retune the blink parameters.

Evaluation Protocol
-------------------
Final split:
MPEBlink 2.0 test

Test videos:
212

Person sequences:
687

Annotation frames:
219,706

Video frames read:
219,706

Face initialization:
Ground-truth per-person MPEBlink bounding boxes

Landmarks:
PhysioTrack FaceLandmarks using MediaPipe

Eye descriptor:
PhysioTrack EyeOpenness mean_openness

Blink threshold:
0.44

Minimum closed frames:
3

Blink event matching:
One-to-one temporal-event matching with temporal IoU >= 0.50

Each video's measured FPS is used for time-based blink-rate and duration
calculations.

Event Metrics
-------------
For one-to-one matched blink events:

Precision

                  TP
    Precision = --------
                TP + FP

Recall

               TP
    Recall = --------
             TP + FN

F1 score

             2 x Precision x Recall
    F1 = --------------------------------
              Precision + Recall

Temporal intersection over union:

                  temporal intersection
    tIoU = -----------------------------------
                   temporal union

The evaluator also reports:

- Mean and median matched temporal IoU
- Mean onset error in frames
- Mean offset error in frames
- Mean blink-duration error in seconds
- Blink-count MAE per person sequence
- Blink-rate MAE in blinks per minute

Eye-Openness Metrics
--------------------
Eye-openness availability is calculated over valid face-box samples.

The blink-versus-non-blink ROC AUC uses finite PhysioTrack EyeOpenness samples
only and uses negative openness as the blink score because lower openness values
are expected during blinking.

The evaluator additionally reports the mean and median EyeOpenness values for
blink and non-blink frames.

Running the Validation
----------------------
Activate the PhysioTrack thesis environment and move to the validation
directory.

Example:

conda activate PhysioTrack-Thesis
cd /d <project-path>\physiotrack\validation\eye_openness_and_blink

Optional dataset-integrity preflight:

python mpeblink_blink_eval.py --preflight-only

Expected preflight dataset counts:

Validation:
- 169 videos
- 81,810 annotation frames
- 570 person sequences
- 1,871 blink events
- 0 out-of-range blink events

Test:
- 212 videos
- 219,706 annotation frames
- 687 person sequences
- 7,564 blink events
- 1 out-of-range blink event retained

Run the final quantitative test evaluation:

python mpeblink_blink_eval.py

The evaluator uses the frozen validation-selected configuration:

blink_threshold = 0.44
min_closed_frames = 3

After the evaluator completes successfully, generate the final thesis table and
quantitative figure:

python mpeblink_blink_plot.py

After the accepted quantitative outputs are present, generate the qualitative
benchmark evidence:

python mpeblink_blink_qualitative.py

The qualitative script removes and regenerates only the qualitative files that
it owns. It does not delete or modify the accepted quantitative result files or
the MPEBlink dataset.

Generated Quantitative Results
------------------------------
The evaluator writes the following files under:

physiotrack/validation/eye_openness_and_blink/results/

mpeblink_test_summary.txt
    Complete benchmark configuration, processing counts, final blink-event
    metrics, and EyeOpenness analysis.

mpeblink_test_sequence_results.csv
    Per-person-sequence event counts and error information used for auditing,
    consistency checks, and deterministic qualitative case selection.

The plot/table script additionally creates:

mpeblink_test_thesis_table.csv
mpeblink_test_thesis_table.md
    Final thesis-oriented summary table.

results/figures/mpeblink_eye_blink_metrics.png
    Final quantitative performance figure showing EyeOpenness ROC AUC,
    blink precision, recall, F1, and matched temporal IoU.

Final Quantitative Results
--------------------------
Dataset processing:

Videos:
212

Person sequences:
687

Annotation frames:
219,706

Video frames read:
219,706

Valid face-box samples:
495,858

Successful EyeOpenness samples:
389,922

EyeOpenness availability:
78.64%

Missing bounding boxes:
100,341

Invalid bounding boxes:
10

Landmark failures:
105,936

Video frame mismatches:
0

Video read failures:
0

Out-of-range blink annotations retained:
1

Blink event results:

Ground-truth blinks:
7,564

Predicted blinks:
6,010

True positives:
1,827

False positives:
4,183

False negatives:
5,737

Precision:
0.303993
30.40%

Recall:
0.241539
24.15%

F1:
0.269191
26.92%

Mean matched temporal IoU:
0.681126

Median matched temporal IoU:
0.666667

Mean onset error:
0.7849 frames

Mean offset error:
1.9836 frames

Mean blink-duration error:
0.101787 seconds

Blink-count MAE per sequence:
6.119360 blinks

Blink-rate MAE:
10.799823 blinks/minute

Eye-openness results:

Finite EyeOpenness samples:
389,922

Blink-frame EyeOpenness samples:
48,912

Non-blink EyeOpenness samples:
341,010

Blink-frame openness mean:
0.352304

Blink-frame openness median:
0.305653

Non-blink openness mean:
0.491654

Non-blink openness median:
0.476032

Blink-versus-non-blink ROC AUC:
0.663061

Reported Runtime
----------------
The reported complete test evaluation required:

96.48 minutes

Runtime depends on hardware and software environment and is not a scientific
reproducibility target. The benchmark counts and numerical metrics are the
reproducibility targets.

Quantitative Verification
-------------------------
A successful reproduction should satisfy the following core conditions:

1. 212 test videos are found and read successfully.
2. 687 annotated person sequences are evaluated.
3. Annotation frames and video frames read both equal 219,706.
4. Video frame mismatches = 0.
5. Video read failures = 0.
6. Ground-truth blink events = 7,564.
7. Predicted blink events = 6,010.
8. TP = 1,827.
9. FP = 4,183.
10. FN = 5,737.
11. Precision = 0.303993.
12. Recall = 0.241539.
13. F1 = 0.269191.
14. Mean matched temporal IoU = 0.681126.
15. Blink-count MAE per sequence = 6.119360.
16. Blink-rate MAE = 10.799823 blinks/minute.
17. EyeOpenness ROC AUC = 0.663061.
18. The generated sequence-results CSV, text summary, thesis table, and figure
    are internally consistent.

Minor runtime differences across machines are expected.

Qualitative Benchmark Evidence
------------------------------
The qualitative stage complements the complete quantitative test evaluation. It
does not replace, modify, or redefine the full-test metrics.

The qualitative generator selects eight deterministic test person sequences
from the accepted sequence-results CSV and re-runs those sequences using the
same benchmark protocol. Before generating any final qualitative artifact, it
verifies that the re-run ground-truth event count, predicted event count, TP,
FP, and FN values match the accepted quantitative sequence results.

The eight qualitative roles are:

- strong_detection
- representative
- challenging_false_positive
- challenging_false_negative
- mixed_detection
- accurate_count
- high_blink_activity
- low_blink_activity

The accepted qualitative cases are:

strong_detection
    test video 24, person1

representative
    test video 192, person1

challenging_false_positive
    test video 190, person1

challenging_false_negative
    test video 189, person0

mixed_detection
    test video 193, person0

accurate_count
    test video 10, person1

high_blink_activity
    test video 191, person0

low_blink_activity
    test video 20, person0

Each qualitative MP4 contains:

- The original MPEBlink video content
- The evaluated ground-truth target-face bounding box
- Available PhysioTrack facial-landmark evidence
- Current frame index and timestamp
- Current mean EyeOpenness value
- Blink threshold
- Current OPEN, CLOSED, or UNAVAILABLE state
- Current ground-truth blink state
- Current predicted blink state
- A clearly identified diagnostic event
- Sequence-level GT, predicted, TP, FP, and FN event counts
- Landmark availability
- A rolling EyeOpenness signal synchronized with the video
- Ground-truth blink intervals
- Predicted blink intervals
- A colored outline identifying the selected diagnostic event

Only the annotated person sequence being evaluated is highlighted, even when
multiple people are visible in the original video. This is intentional and
matches the per-person benchmark protocol. The overlay explicitly identifies
the box as the evaluated ground-truth target face.

The displayed panel values are derived from the actual frame-level benchmark
outputs for the selected target person. They are not manually assigned
presentation values.

Qualitative Outputs
-------------------
The qualitative outputs are written under:

results/qualitative/

Expected artifacts:

results/qualitative/annotated_videos/
    Eight annotated MP4 benchmark clips.

results/qualitative/annotated_images/
    Eight representative PNG benchmark frames.

results/qualitative/mpeblink_qualitative_selection.csv
    Deterministic case roles, source video/person identifiers, diagnostic-event
    information, accepted event counts, and generated output paths.

results/figures/mpeblink_qualitative_examples.png
    Combined figure containing the eight selected qualitative benchmark cases.

The qualitative generator owns and may replace only these qualitative outputs.
It does not delete or modify the accepted quantitative summary, sequence CSV,
thesis table, quantitative figure, or dataset.

Interpretation
--------------
The PhysioTrack EyeOpenness descriptor contains useful blink-related
information. Blink frames have lower median openness than non-blink frames:

Blink-frame median openness:
0.305653

Non-blink median openness:
0.476032

The ROC AUC of 0.663061 indicates moderate separation between blink and
non-blink EyeOpenness values.

The current threshold-based blink detector has substantially weaker event-level
performance:

Precision:
30.40%

Recall:
24.15%

F1:
26.92%

The matched detections that are found have better temporal localization, with a
mean matched temporal IoU of 0.681126. However, the high false-positive and
false-negative counts show that a fixed EyeOpenness threshold with a short
consecutive-frame rule is limited under challenging unconstrained MPEBlink
videos.

The qualitative examples intentionally include both successful and failure
cases so that the visual evidence reflects the measured benchmark behavior
rather than presenting only favorable examples.

Evaluation Scope
----------------
This validation measures PhysioTrack EyeOpenness and the current temporal blink
logic under MPEBlink ground-truth face initialization.

It does not evaluate face detection.

It does not train or fine-tune a blink-specific model.

It does not claim direct equivalence to every official MPEBlink leaderboard
metric. The reported blink-event results use the documented project protocol
with one-to-one temporal-IoU matching at tIoU >= 0.50.

The results should therefore be described as PhysioTrack EyeOpenness and
threshold-based blink-detection performance under the documented MPEBlink 2.0
per-person evaluation protocol.

Reproducibility
---------------
To reproduce this benchmark on another machine:

1. Obtain MPEBlink 2.0 from the official dataset source.
2. Extract it to datasets/MPEBlink2/mpeblink2.0.
3. Install PhysioTrack and its required dependencies.
4. Activate the PhysioTrack thesis environment.
5. Optionally run mpeblink_blink_eval.py --preflight-only to verify the dataset
   layout and benchmark counts.
6. Run mpeblink_blink_eval.py to generate the final test results using the
   frozen validation-selected parameters.
7. Run mpeblink_blink_plot.py to verify the quantitative outputs and generate
   the final thesis table and performance figure.
8. Run mpeblink_blink_qualitative.py to generate the verified qualitative
   benchmark videos, images, selection CSV, and combined figure.
9. Compare the generated quantitative metrics with the values documented in
   this README and inspect the qualitative evidence.

No user-specific absolute dataset path or undocumented manual data modification
is required.
