ECCV 2016 Music Video Face Tracking Validation
================================================

Overview
--------
This directory contains the reproducible Face Tracking validation package for
PhysioTrack.

The package provides two complementary forms of evidence:

1. Scientific benchmark validation on the ECCV 2016 Music Video Face Tracking
   dataset.
2. Isolated execution of the real PhysioTrack Face Tracking component using the
   project FaceAnalysis path with unrelated face-analysis components disabled.

The scientific benchmark evaluates tracking performance against ground-truth
face trajectories using standard multiple-object-tracking metrics. The isolated
component execution verifies software-level operation of the real PhysioTrack
tracking path and preserves its numerical outputs in a structured table.

These procedures answer different questions and must be interpreted
separately. The isolated component execution is not a tracking-accuracy
benchmark and does not replace the scientific benchmark metrics.

Dataset
-------
Dataset:
ECCV 2016 Music Video Face Tracking Dataset

Reference:
Shun Zhang, Yihong Gong, Jia-Bin Huang, Jongwoo Lim, Jinjun Wang,
Narendra Ahuja, and Ming-Hsuan Yang,
"Tracking Persons-of-Interest via Adaptive Discriminative Features,"
European Conference on Computer Vision (ECCV), 2016.

Official project page:

https://sites.google.com/site/shunzhang876/eccv16-face-tracking

Required dataset content
------------------------
The validation uses eight music-video sequences and their corresponding
ground-truth XML annotations.

Required videos:
- Apink.mp4
- BrunoMars.mp4
- Darling.mp4
- GirlsAloud.mp4
- HelloBubble.mp4
- PussycatDolls.mp4
- T-ara.mov
- Westlife.mp4

Required ground-truth files:
- Apink_gt.xml
- BrunoMars_gt.xml
- Darling_gt.xml
- GirlsAloud_gt.xml
- HelloBubble_gt.xml
- stickwitu_gt.xml
- Tara_gt.xml
- Westlife_gt.xml

Additional XML files contained in the source package are not used by this
eight-video validation.

Dataset layout
--------------
Extract the dataset under the project-level datasets directory using:

datasets/
└── FACE_TRACKING_ECCV2016/
    ├── videos/
    │   ├── Apink.mp4
    │   ├── BrunoMars.mp4
    │   ├── Darling.mp4
    │   ├── GirlsAloud.mp4
    │   ├── HelloBubble.mp4
    │   ├── PussycatDolls.mp4
    │   ├── T-ara.mov
    │   └── Westlife.mp4
    └── ground_truth/
        └── GT/
            ├── Apink_gt.xml
            ├── BrunoMars_gt.xml
            ├── Darling_gt.xml
            ├── GirlsAloud_gt.xml
            ├── HelloBubble_gt.xml
            ├── stickwitu_gt.xml
            ├── Tara_gt.xml
            └── Westlife_gt.xml

All validation paths are resolved relative to the project structure. No
machine-specific absolute path is required.

The dataset directory is treated as read-only benchmark input.

Ground-truth format
-------------------
The annotations are stored as XML trajectories. Each trajectory has a
ground-truth identity and contains frame-level face bounding boxes represented
by:

frame number, x, y, width, height

The XML frame numbering is one-based.

The evaluator converts each annotated box to:

[x1, y1, x2, y2]

and uses the trajectory object identifier as the ground-truth identity.

Validation scripts
------------------
The validation implementation is located under:

physiotrack/validation/face_tracking/

eccv16_tracking_eval.py
    Runs the complete PhysioTrack face detector and OC-SORT tracker on all eight
    benchmark videos, compares predicted tracks with ground truth, computes the
    tracking metrics, and writes the detailed result CSV and validation summary.

eccv16_tracking_table.py
    Reads the detailed result CSV and creates compact CSV and Markdown thesis
    tables.

eccv16_tracking_plot.py
    Reads the detailed result CSV and generates the final quantitative tracking
    figure.

eccv16_tracking_qualitative.py
    Reproduces the complete T-ara quantitative result using the same detector,
    tracker, matching threshold, and MOTMetrics protocol before generating
    verified qualitative tracking evidence.

face_tracking_component_test.py
    Runs the real PhysioTrack Face Tracking implementation through the
    FaceAnalysis project path with Face Detection enabled only as the required
    upstream dependency. Unrelated face-analysis components are disabled. The
    script processes all eight benchmark videos and exports real numerical track
    observations to a structured CSV together with an execution summary.

Environment and dependencies
----------------------------
The validation requires the PhysioTrack project environment and the packages
used by the detector, tracker, video reader, evaluator, and plotting scripts.

Important dependencies include:
- OpenCV
- NumPy
- motmetrics 1.4.0
- pandas
- matplotlib
- the PhysioTrack package and its face-tracking dependencies

The quantitative evaluator and qualitative generator include a compatibility
shim for motmetrics 1.4.0 when used with NumPy versions in which np.asfarray is
no longer available. This compatibility handling does not change the
documented metric definitions or benchmark logic.

Scientific benchmark configuration
----------------------------------
Face detector confidence threshold:
0.25

Face detector IoU threshold:
0.45

Tracker backend:
OC-SORT

Tracking device:
CPU

Ground-truth/prediction matching IoU threshold:
0.50

Evaluation library:
motmetrics 1.4.0

A new tracker instance is created for each video sequence. Track identities are
therefore sequence-local, as expected for independent tracking sequences.

Scientific benchmark protocol
-----------------------------
For each video frame:

1. The PhysioTrack face detector produces face detections.
2. The detections are passed to FaceTracker.
3. FaceTracker forwards them to the OC-SORT backend.
4. OC-SORT returns tracked face boxes and persistent sequence-local track IDs.
5. Predicted boxes and ground-truth boxes are matched at IoU >= 0.50.
6. Frame-level associations are accumulated by motmetrics.

The benchmark evaluates the complete detector-plus-tracker chain.
Consequently, detector misses, false detections, localization errors, tracker
association errors, identity switches, and fragmentations all contribute to the
final results.

Metrics
-------
The benchmark reports:
- Recall
- Precision
- F1
- False Alarms per Frame (FAF)
- Identity Switches (IDS)
- Fragmentations (Frag)
- Multiple Object Tracking Accuracy (MOTA)
- Multiple Object Tracking Precision (MOTP)
- IDF1

motmetrics represents MOTP as an IoU distance in this evaluation. The reported
MOTP value is converted to mean matched IoU as:

MOTP IoU = 1 - raw MOTP distance

Scientific benchmark run order
------------------------------
Activate the PhysioTrack environment:

conda activate PhysioTrack-Thesis

Open:

physiotrack/validation/face_tracking

Run the quantitative stages in this order:

python eccv16_tracking_eval.py
python eccv16_tracking_table.py
python eccv16_tracking_plot.py

After successful quantitative validation, run:

python eccv16_tracking_qualitative.py

The evaluator is the computationally expensive quantitative stage because it
processes all 42,007 frames across the eight sequences.

The table and plot stages consume the accepted evaluator result CSV.

The qualitative stage is also computationally expensive because it reproduces
the complete T-ara sequence before final qualitative rendering. The
qualitative evidence is generated only if this reproduced T-ara result matches
the accepted quantitative benchmark result.

Validated quantitative results
------------------------------
Dataset coverage:
- Videos: 8
- Frames: 42,007
- Ground-truth face observations: 95,302
- Predictions: 120,531
- Matches: 84,461
- False negatives: 8,508
- False positives: 33,737
- Identity switches: 2,333
- Fragmentations: 2,340

Overall metrics:
- Recall: 91.07%
- Precision: 72.01%
- F1: 80.43%
- FAF: 0.8031
- IDS: 2,333
- Fragmentations: 2,340
- MOTA: 53.22%
- MOTP: 87.81%
- IDF1: 6.81%

Per-sequence results:

Apink
- Recall: 89.77%
- Precision: 91.64%
- F1: 90.69%
- FAF: 0.1130
- IDS: 213
- Frag: 212
- MOTA: 78.65%
- MOTP: 83.05%
- IDF1: 6.36%

BrunoMars
- Recall: 90.74%
- Precision: 71.28%
- F1: 79.84%
- FAF: 0.9423
- IDS: 468
- Frag: 487
- MOTA: 51.37%
- MOTP: 85.72%
- IDF1: 6.58%

Darling
- Recall: 85.85%
- Precision: 88.25%
- F1: 87.03%
- FAF: 0.2303
- IDS: 479
- Frag: 453
- MOTA: 69.40%
- MOTP: 86.76%
- IDF1: 5.57%

GirlsAloud
- Recall: 91.74%
- Precision: 89.52%
- F1: 90.62%
- FAF: 0.3182
- IDS: 507
- Frag: 468
- MOTA: 77.90%
- MOTP: 85.73%
- IDF1: 6.78%

HelloBubble
- Recall: 90.62%
- Precision: 91.18%
- F1: 90.90%
- FAF: 0.1215
- IDS: 138
- Frag: 142
- MOTA: 79.21%
- MOTP: 88.20%
- IDF1: 8.43%

PussycatDolls
- Recall: 92.92%
- Precision: 63.23%
- F1: 75.25%
- FAF: 1.2995
- IDS: 164
- Frag: 200
- MOTA: 37.73%
- MOTP: 88.27%
- IDF1: 9.39%

T-ara
- Recall: 87.40%
- Precision: 96.65%
- F1: 91.80%
- FAF: 0.0968
- IDS: 267
- Frag: 311
- MOTA: 82.54%
- MOTP: 94.73%
- IDF1: 5.79%

Westlife
- Recall: 98.37%
- Precision: 41.83%
- F1: 58.70%
- FAF: 2.7144
- IDS: 97
- Frag: 67
- MOTA: -39.30%
- MOTP: 88.41%
- IDF1: 5.77%

Runtime and processing speed depend on hardware and system load and are not
scientific accuracy metrics.

Quantitative interpretation
---------------------------
The benchmark shows strong face coverage and generally accurate spatial
localization. The overall recall is 91.07%, the F1 score is 80.43%, and the
mean matched IoU reported as MOTP is 87.81%.

Performance varies substantially across sequences.

Westlife produces a particularly large number of false positives, which reduces
precision to 41.83% and produces a negative MOTA of -39.30%. This is a valid
benchmark outcome and is retained without suppression.

The overall IDF1 of 6.81% is an important limitation of the current
detector-plus-OC-SORT configuration.

Inspection of the PhysioTrack tracking path confirms that FaceTracker does not
regenerate or remap identities on every frame. OC-SORT assigns persistent
sequence-local track identities and retains them for the lifetime of each
track. The tracker is reset only when a new independent video is started.

The low IDF1 is therefore not explained by an unintended per-frame identifier
reset. OC-SORT in this configuration relies primarily on motion and spatial
association rather than a strong face re-identification embedding. Difficult
conditions in the music videos, including occlusion, shot changes, scene
changes, rapid motion, and appearance variation, can cause a lost person to
reappear under a different track identity.

The identity-switch, fragmentation, and IDF1 results are therefore documented
limitations of the current tracking configuration rather than being replaced
by detection-oriented metrics.

Qualitative benchmark evidence
------------------------------
The qualitative stage uses T-ara as a detailed tracking example because the
sequence contains successful multi-face tracking together with real identity
switches, misses, false positives, and localization variation.

Accepted full-sequence T-ara metrics:
- Recall: 87.40%
- Precision: 96.65%
- F1: 91.80%
- FAF: 0.0968
- IDS: 267
- Fragmentations: 311
- MOTA: 82.54%
- MOTP: 94.73%
- IDF1: 5.79%

Before any final qualitative output is committed,
eccv16_tracking_qualitative.py re-evaluates the complete T-ara sequence using
the same detector, tracker, IoU threshold, and MOTMetrics accumulator protocol
as eccv16_tracking_eval.py.

Qualitative output is generated only after the reproduced quantitative result
matches the accepted T-ara benchmark row.

The selected qualitative window is:
- Source sequence: T-ara
- Frames: 2,689 to 4,486
- Duration: approximately 59.99 seconds
- Exact MOTMetrics SWITCH events: 114
- Detected face observations: 5,134
- MISS events: 706
- FP events: 154

The 5,134 detected face observations consist of:
- MATCH: 5,020
- SWITCH: 114

The event labels in the qualitative video are derived from the exact
MOTMetrics event history produced by the same matching protocol as the
quantitative evaluation.

Six deterministic representative frames are generated:

stable_multi_face
    Successful simultaneous tracking of multiple faces without a SWITCH, MISS,
    or FP event in the selected frame.

identity_switch
    A frame containing exact MOTMetrics SWITCH events.

false_negative
    A frame containing missed ground-truth faces.

false_positive
    A frame containing false-positive tracks together with valid matches.

crowded_tracking
    A frame illustrating tracking behavior when multiple faces are visible.

localization_challenge
    A matched frame with comparatively lower matched IoU, illustrating
    localization difficulty while retaining a valid match.

These examples complement the benchmark metrics and do not replace them.

Validated qualitative outputs
-----------------------------
The final qualitative package contains:
- One annotated 60-second T-ara tracking video
- Six representative 1920x1080 frames
- One exact MOTMetrics event CSV
- One qualitative selection CSV
- One combined qualitative figure

The annotated video contains 1,798 frames at approximately 29.97 FPS and spans
frames 2,689 through 4,486 inclusively.

The event CSV contains the complete selected-window MOTMetrics history,
including:
- MATCH: 5,020
- SWITCH: 114
- MISS: 706
- FP: 154

Safe rerun design
-----------------
The validation package follows the safe-rerun pattern:

preflight
-> temporary/staging generation
-> validation of newly generated outputs
-> replacement of script-owned final outputs

Previously valid final evidence is not removed before the newly generated
replacement has passed the corresponding validation checks.

If required dataset files, ground truth, result dependencies, or other critical
inputs are unavailable or invalid, the affected script stops without deleting
previously valid final outputs.

Each script owns only its own artifacts.

Output ownership
----------------
eccv16_tracking_eval.py
    Owns:
    - results/eccv16_tracking_results.csv
    - results/eccv16_tracking_summary.txt

eccv16_tracking_table.py
    Owns:
    - results/eccv16_tracking_thesis_table.csv
    - results/eccv16_tracking_thesis_table.md

eccv16_tracking_plot.py
    Owns:
    - results/figures/eccv16_tracking_metrics.png

eccv16_tracking_qualitative.py
    Owns:
    - results/qualitative/annotated_videos/
    - results/qualitative/representative_frames/
    - results/qualitative/eccv16_tracking_qualitative_events.csv
    - results/qualitative/eccv16_tracking_qualitative_selection.csv
    - results/figures/eccv16_tracking_qualitative_examples.png

face_tracking_component_test.py
    Owns:
    - results/component_execution/face_tracking_component_results.csv
    - results/component_execution/face_tracking_component_summary.json

No validation script is intended to delete another script's final outputs.

Isolated PhysioTrack Face Tracking execution
--------------------------------------------
Run:

python face_tracking_component_test.py

Purpose:
Verify that the real PhysioTrack Face Tracking path operates independently,
with Face Detection enabled only as the required upstream dependency, and that
its numerical tracking outputs can be exported reproducibly.

The isolated component execution:
- uses the real PhysioTrack FaceAnalysis tracking path
- uses the real PhysioTrack face detector as the required tracking dependency
- uses OC-SORT
- processes all eight ECCV 2016 benchmark videos
- processes all 42,007 frames
- uses detector confidence threshold 0.25
- uses detector IoU threshold 0.45
- uses CPU execution
- disables unrelated optional face-analysis components
- preserves real track IDs and raw tracked boxes
- records explicit frame and observation status
- accounts for frames that produce no tracks
- uses project-relative paths
- treats the dataset as read-only
- uses staged generation, staged validation, and owned-output replacement

Disabled unrelated components:
- head pose
- landmarks
- quality
- eyes
- blink
- gaze
- gaze estimation
- mouth
- mouth motion
- emotion
- face regions
- temporal analysis

The ground-truth XML files are used by the isolated component test only to
verify expected sequence length and complete frame coverage. Ground truth is
not used to compute isolated-component accuracy metrics.

The isolated component test is therefore software execution evidence, not a
replacement tracking benchmark.

Isolated component output schema
--------------------------------
Main numerical result table:

results/component_execution/face_tracking_component_results.csv

Execution summary:

results/component_execution/face_tracking_component_summary.json

The result table contains:
- sequence
- source_video
- frame_number
- timestamp_seconds
- image_width
- image_height
- track_index
- tracks_in_frame
- track_id
- class_id
- class_name
- box_x1
- box_y1
- box_x2
- box_y2
- box_width
- box_height
- box_area
- confidence
- status
- failure_reason

For a valid tracked observation:

box_width = box_x2 - box_x1

box_height = box_y2 - box_y1

box_area = box_width × box_height

Frames that complete successfully but produce no tracks are retained in the CSV
with:

status = NO_TRACKS

and empty track-specific fields.

This preserves complete frame-level execution coverage without fabricating
tracking values for frames in which the tracker produced no output.

Validated isolated component results
------------------------------------
Videos expected:
8

Videos processed:
8

Expected total frames:
42,007

Frames processed:
42,007

Failed frames:
0

Frames with tracks:
36,736

Frames without tracks:
5,271

Track observations:
120,531

Invalid-box observations:
0

Missing-ID observations:
0

Result rows:
125,802

Overall status:
PASS

The row accounting is:

120,531 tracked observations
+ 5,271 NO_TRACKS frame records
= 125,802 result rows

All 42,007 expected frames are represented in the result CSV.

Within every frame:
- track_index values are unique and consecutive from 1 to tracks_in_frame
- tracks_in_frame agrees with the number of tracked observations
- track IDs are present for valid tracked observations
- duplicate track IDs are not present within the same frame
- valid box geometry satisfies the documented width, height, and area formulas
- no invalid tracked boxes were observed
- no missing track identifiers were observed

The isolated execution produced exactly 120,531 tracked observations. This
matches the total prediction count produced by the accepted scientific
benchmark run. The per-sequence tracked-observation counts also match the
benchmark prediction counts:

- Apink: 7,126
- BrunoMars: 21,269
- Darling: 9,272
- GirlsAloud: 16,790
- HelloBubble: 5,191
- PussycatDolls: 20,979
- T-ara: 13,139
- Westlife: 26,765

This agreement provides a strong reproducibility check that the isolated
execution uses the same real detector-plus-OC-SORT tracking path and execution
configuration as the scientific benchmark.

The isolated component output does not establish tracking accuracy. Tracking
accuracy is established by the scientific benchmark metrics reported
separately above.

Expected final result structure
-------------------------------
results/
├── eccv16_tracking_results.csv
├── eccv16_tracking_summary.txt
├── eccv16_tracking_thesis_table.csv
├── eccv16_tracking_thesis_table.md
├── qualitative/
│   ├── annotated_videos/
│   │   └── T-ara_face_tracking_qualitative.mp4
│   ├── representative_frames/
│   │   ├── T-ara_stable_multi_face_frame_3898.png
│   │   ├── T-ara_identity_switch_frame_2747.png
│   │   ├── T-ara_false_negative_frame_2744.png
│   │   ├── T-ara_false_positive_frame_3485.png
│   │   ├── T-ara_crowded_tracking_frame_2697.png
│   │   └── T-ara_localization_challenge_frame_3566.png
│   ├── eccv16_tracking_qualitative_events.csv
│   └── eccv16_tracking_qualitative_selection.csv
├── component_execution/
│   ├── face_tracking_component_results.csv
│   └── face_tracking_component_summary.json
└── figures/
    ├── eccv16_tracking_metrics.png
    └── eccv16_tracking_qualitative_examples.png

Output descriptions
-------------------
results/eccv16_tracking_results.csv
    Detailed benchmark results for all eight sequences and the overall
    evaluation.

results/eccv16_tracking_summary.txt
    Text record of the benchmark configuration, sequence results, overall
    results, counts, and runtime information.

results/eccv16_tracking_thesis_table.csv
    Compact CSV representation of the benchmark results.

results/eccv16_tracking_thesis_table.md
    Markdown representation of the same compact benchmark table.

results/figures/eccv16_tracking_metrics.png
    Quantitative tracking-performance figure.

results/qualitative/annotated_videos/T-ara_face_tracking_qualitative.mp4
    Full-HD annotated 60-second tracking video with exact MOTMetrics events,
    predicted track identities, ground-truth identities, current-frame
    statistics, cumulative clip events, and accepted full-sequence T-ara
    benchmark metrics.

results/qualitative/representative_frames/
    Six tracking-focused representative frames covering successful multi-face
    tracking, identity switches, false negatives, false positives, crowded
    tracking, and localization difficulty.

results/qualitative/eccv16_tracking_qualitative_events.csv
    Exact MOTMetrics event history for the selected T-ara qualitative clip.

results/qualitative/eccv16_tracking_qualitative_selection.csv
    Machine-readable record of the selected qualitative window, selection rule,
    accepted T-ara metrics, and representative-frame roles.

results/figures/eccv16_tracking_qualitative_examples.png
    Combined qualitative overview figure.

results/component_execution/face_tracking_component_results.csv
    Structured numerical output from isolated execution of the real PhysioTrack
    Face Tracking component.

results/component_execution/face_tracking_component_summary.json
    Per-sequence and overall execution accounting for the isolated component
    test.

Complete reproduction procedure
-------------------------------
1. Install the PhysioTrack project environment and dependencies.

2. Download the ECCV 2016 music-video face-tracking dataset and corresponding
   ground-truth XML annotations from the documented source.

3. Arrange the data under:

   datasets/FACE_TRACKING_ECCV2016

   using the documented directory structure.

4. Activate:

   conda activate PhysioTrack-Thesis

5. Open:

   physiotrack/validation/face_tracking

6. Run the scientific benchmark:

   python eccv16_tracking_eval.py
   python eccv16_tracking_table.py
   python eccv16_tracking_plot.py
   python eccv16_tracking_qualitative.py

7. Confirm that the quantitative metrics and qualitative verification match the
   documented accepted results.

8. Run the isolated Face Tracking component execution:

   python face_tracking_component_test.py

9. Verify:

   results/component_execution/face_tracking_component_results.csv

   and:

   results/component_execution/face_tracking_component_summary.json

10. Confirm complete frame accounting, failure accounting, track-ID integrity,
    output schemas, and benchmark result consistency before using the package as
    reproducibility evidence.

Reproducibility notes
---------------------
All validation scripts use project-relative paths.

The source dataset is not modified.

Runtime differences across machines are expected and are not interpreted as
accuracy differences.

Scientific benchmark results should be investigated if they differ materially
when the same source data, PhysioTrack implementation, model, tracker,
dependencies, and evaluation configuration are used.

The isolated component execution should process all eight sequences and all
42,007 frames. Numerical tracking observations are comparable across
environments only when the implementation, detector model, tracker,
dependencies, and execution configuration are the same.

Interpretation and limitations
------------------------------
The ECCV 2016 scientific benchmark is the evidence used to characterize Face
Tracking predictive performance and temporal identity behavior.

The isolated component execution is software-level evidence that the actual
PhysioTrack Face Tracking path runs independently and exports real numerical
tracking outputs.

A PASS result from the isolated component execution must not be interpreted as
a tracking-accuracy score.

The qualitative examples provide interpretable evidence of real benchmark
behavior but do not replace full-sequence metrics.

The benchmark evaluates the complete detector-plus-tracker chain and should not
be interpreted as an evaluation of OC-SORT independently from the detector.

The current configuration shows good coverage and localization but limited
long-term identity consistency, as reflected by the low IDF1 and identity
switch counts.

Final files to preserve
-----------------------
Preserve:
- eccv16_tracking_eval.py
- eccv16_tracking_table.py
- eccv16_tracking_plot.py
- eccv16_tracking_qualitative.py
- face_tracking_component_test.py
- README_ECCV2016_FACE_TRACKING.txt
- results/eccv16_tracking_results.csv
- results/eccv16_tracking_summary.txt
- results/eccv16_tracking_thesis_table.csv
- results/eccv16_tracking_thesis_table.md
- results/figures/eccv16_tracking_metrics.png
- results/qualitative/annotated_videos/T-ara_face_tracking_qualitative.mp4
- results/qualitative/representative_frames/
- results/qualitative/eccv16_tracking_qualitative_events.csv
- results/qualitative/eccv16_tracking_qualitative_selection.csv
- results/figures/eccv16_tracking_qualitative_examples.png
- results/component_execution/face_tracking_component_results.csv
- results/component_execution/face_tracking_component_summary.json

Together, these files preserve the scientific benchmark evidence, qualitative
tracking evidence, and isolated PhysioTrack Face Tracking execution record
required for reproducibility.
