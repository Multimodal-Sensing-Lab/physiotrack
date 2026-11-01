ECCV 2016 Music Video Face Tracking Validation
================================================

Overview
--------
This directory contains the reproducible Face Tracking validation package for
PhysioTrack.

The package provides three complementary and independently interpreted forms
of evidence:

1. Scientific benchmark validation on the ECCV 2016 Music Video Face Tracking
   dataset, including quantitative metrics and benchmark qualitative evidence.
2. Isolated execution of the real PhysioTrack Face Tracking component through
   FaceAnalysis with unrelated face-analysis components disabled.
3. Synthetic qualitative demonstrations of tracking behaviour on independent
   synthetic video sequences without benchmark ground-truth annotations.

The scientific benchmark evaluates detector-plus-tracker performance against
annotated face trajectories. The isolated execution verifies software operation
and preserves numerical track observations. The synthetic demonstration visualizes
actual tracking outputs under controlled illustrative conditions. Neither the
isolated execution status nor the synthetic demonstration is a substitute for
scientific tracking-accuracy metrics.

Dataset
-------
Dataset:
ECCV 2016 Music Video Face Tracking Dataset

Reference:
Shun Zhang, Yihong Gong, Jia-Bin Huang, Jongwoo Lim, Jinjun Wang,
Narendra Ahuja, and Ming-Hsuan Yang,
"Tracking Persons-of-Interest via Adaptive Discriminative Features,"
European Conference on Computer Vision (ECCV), 2016.
Pages: 415-433.
DOI: https://doi.org/10.1007/978-3-319-46454-1_26

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
The benchmark dataset is located in the datasets directory alongside the
physiotrack-v2 repository:

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

The datasets/ and physiotrack-v2/ directories share a common parent directory.
Validation paths are derived from the script locations; no machine-specific
absolute path is used. The benchmark dataset is read-only.

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
Before calling motmetrics.distances.iou_matrix, both ground-truth and
predicted boxes are converted from [x1, y1, x2, y2] to
[x, y, width, height], as required by motmetrics 1.4.0. The FaceTracker
interface and its project-level bounding-box format are unchanged.

Validation scripts
------------------
The validation implementation is located under:

physiotrack-v2/validation/face_tracking/

eccv16_tracking_iou_selftest.py
    Checks the evaluation-boundary xyxy-to-xywh conversion in both the
    quantitative and qualitative evaluators against motmetrics 1.4.0.
    This is a regression test, not an independent tracking benchmark.

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
    FaceAnalysis project path with Face Detection enabled as the required upstream
    dependency. Unrelated face-analysis components are disabled. The script
    processes all eight benchmark videos and exports real track observations to
    a structured CSV and execution summary.

face_tracking_synthetic_demo.py
    Discovers supported synthetic videos in validation/synthetic_media/
    face_tracking_media/ and processes each one with the real FaceAnalysis
    detector-plus-OC-SORT pipeline. Produces annotated videos, per-video track
    observation tables, three illustrative PNG frames per video, and matching
    JSON and TXT execution summaries. All outputs are verified before publication;
    no ECCV 2016 annotations or benchmark metrics are used.

Environment and dependencies
----------------------------
The evaluation runs in a Python environment containing the installed local
PhysioTrack project, its face-detection and tracking dependencies, and the
following packages used by the validation scripts:

- OpenCV
- NumPy
- motmetrics 1.4.0
- pandas
- matplotlib

The completed synthetic demonstration recorded Python 3.11.17, OpenCV 5.0.0,
and NumPy 2.4.6 on Windows. The detector and OC-SORT backend were run on CPU.
The project package is imported from physiotrack-v2/src/physiotrack/.

The quantitative evaluator and qualitative generator provide compatibility for
motmetrics 1.4.0 with NumPy versions in which np.asfarray is unavailable.
This compatibility handling does not change the metric definitions.

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
5. Both sets of boxes are converted from xyxy to xywh at the MOTMetrics
   distance-calculation boundary and matched at IoU >= 0.50.
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
Environment activation for the validated project environment:

conda activate PhysioTrack-v2

Working directory:

physiotrack-v2/validation/face_tracking/

Benchmark execution order:

python eccv16_tracking_iou_selftest.py
python eccv16_tracking_eval.py
python eccv16_tracking_table.py
python eccv16_tracking_plot.py
python eccv16_tracking_qualitative.py

The quantitative evaluator processes 42,007 frames across eight independent
sequences. The table and plot read its saved result CSV. The benchmark
qualitative script re-evaluates the full T-ara sequence and compares the
reproduced tracking metrics with the accepted quantitative T-ara row before
publishing qualitative artifacts.

Validated quantitative results
------------------------------
Dataset coverage:
- Videos: 8
- Frames: 42,007
- Ground-truth face observations: 95,302
- Predictions: 120,531
- Matches: 80,102
- False negatives: 12,898
- False positives: 38,127
- Identity switches: 2,302
- Fragmentations: 4,088

Overall metrics:
- Recall: 86.47%
- Precision: 68.37%
- F1: 76.36%
- FAF: 0.9076
- IDS: 2,302
- Fragmentations: 4,088
- MOTA: 44.04%
- MOTP: 67.89%
- IDF1: 6.53%

Per-sequence results:

Apink
- Recall: 86.31%
- Precision: 88.10%
- F1: 87.19%
- FAF: 0.1608
- IDS: 211
- Frag: 248
- MOTA: 71.75%
- MOTP: 65.77%
- IDF1: 6.03%

BrunoMars
- Recall: 84.66%
- Precision: 66.50%
- F1: 74.49%
- FAF: 1.0990
- IDS: 460
- Frag: 896
- MOTA: 39.26%
- MOTP: 68.11%
- IDF1: 6.40%

Darling
- Recall: 82.62%
- Precision: 84.93%
- F1: 83.76%
- FAF: 0.2954
- IDS: 477
- Frag: 582
- MOTA: 62.96%
- MOTP: 67.04%
- IDF1: 5.50%

GirlsAloud
- Recall: 89.71%
- Precision: 87.53%
- F1: 88.61%
- FAF: 0.3784
- IDS: 504
- Frag: 658
- MOTA: 73.86%
- MOTP: 67.75%
- IDF1: 6.72%

HelloBubble
- Recall: 86.35%
- Precision: 86.88%
- F1: 86.61%
- FAF: 0.1807
- IDS: 138
- Frag: 201
- MOTA: 70.67%
- MOTP: 70.86%
- IDF1: 8.41%

PussycatDolls
- Recall: 82.39%
- Precision: 56.06%
- F1: 66.72%
- FAF: 1.5528
- IDS: 155
- Frag: 782
- MOTA: 16.72%
- MOTP: 62.98%
- IDF1: 8.24%

T-ara
- Recall: 86.03%
- Precision: 95.14%
- F1: 90.36%
- FAF: 0.1405
- IDS: 263
- Frag: 354
- MOTA: 79.83%
- MOTP: 75.31%
- IDF1: 5.78%

Westlife
- Recall: 93.50%
- Precision: 39.75%
- F1: 55.79%
- FAF: 2.8112
- IDS: 94
- Frag: 367
- MOTA: -49.02%
- MOTP: 65.10%
- IDF1: 5.62%

Runtime and processing speed depend on hardware and system load and are not
scientific accuracy metrics.

Quantitative interpretation
---------------------------
The benchmark shows substantial face coverage but more limited spatial
localization among matched observations. The overall recall is 86.47%, the
F1 score is 76.36%, and the mean matched IoU reported as MOTP is 67.89%.

Performance varies substantially across sequences.

Westlife produces a particularly large number of false positives, which reduces
precision to 39.75% and produces a negative MOTA of -49.02%. This is a valid
benchmark outcome and is retained without suppression.

The overall IDF1 of 6.53% is an important limitation of the current
detector-plus-OC-SORT configuration.

Inspection of the PhysioTrack tracking path confirms that FaceTracker does not
regenerate or remap identities on every frame. OC-SORT assigns persistent
sequence-local track identities and retains them for the lifetime of each
track. The tracker is reset only when a new independent video is started.

The low IDF1 is therefore not explained by an unintended per-frame identifier
reset. OC-SORT in the validated configuration relies primarily on motion and spatial
association; a face appearance-based Re-identification model is not used. Video
conditions including occlusion, shot changes, rapid motion, and appearance
variation can result in the same person being represented by a new track ID
after track loss.

The identity-switch, fragmentation, and IDF1 results are therefore documented
limitations of the current tracking configuration rather than being replaced
by detection-oriented metrics.

Qualitative benchmark evidence
------------------------------
The qualitative stage uses T-ara as a detailed tracking example because the
sequence contains successful multi-face tracking together with real identity
switches, misses, false positives, and localization variation.

Accepted full-sequence T-ara metrics:
- Recall: 86.03%
- Precision: 95.14%
- F1: 90.36%
- FAF: 0.1405
- IDS: 263
- Fragmentations: 354
- MOTA: 79.83%
- MOTP: 75.31%
- IDF1: 5.78%

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
- Exact MOTMetrics SWITCH events: 112
- Accepted GT/prediction associations (MATCH + SWITCH): 5,069
- MISS events: 771
- FP events: 219

The 5,069 accepted associations consist of:
- MATCH: 4,957
- SWITCH: 112

The qualitative selection CSV fields clip_detected_observations and
detections count accepted MATCH + SWITCH associations, excluding FP events;
they do not represent all tracker predictions.

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
- MATCH: 4,957
- SWITCH: 112
- MISS: 771
- FP: 219

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

face_tracking_synthetic_demo.py
    Owns:
    - results/synthetic_demo/videos/
    - results/synthetic_demo/tracks/
    - results/synthetic_demo/images/
    - results/synthetic_demo/synthetic_demo_summary.json
    - results/synthetic_demo/synthetic_demo_summary.txt
    - results/synthetic_demo_last_attempt.json (failed-run report only)
    - results/synthetic_demo_last_attempt.txt (failed-run report only)

Each script operates on its own output artifacts. The benchmark result files,
benchmark qualitative evidence, and isolated component artifacts are separate
from the synthetic demonstration outputs.

Isolated PhysioTrack Face Tracking execution
--------------------------------------------
Execution command:

python face_tracking_component_test.py

Purpose:
Document that the real PhysioTrack Face Tracking path operates independently,
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
│   │   ├── T-ara_stable_multi_face_frame_3316.png
│   │   ├── T-ara_identity_switch_frame_2747.png
│   │   ├── T-ara_false_negative_frame_2744.png
│   │   ├── T-ara_false_positive_frame_2982.png
│   │   ├── T-ara_crowded_tracking_frame_2697.png
│   │   └── T-ara_localization_challenge_frame_3402.png
│   ├── eccv16_tracking_qualitative_events.csv
│   └── eccv16_tracking_qualitative_selection.csv
├── component_execution/
│   ├── face_tracking_component_results.csv
│   └── face_tracking_component_summary.json
├── synthetic_demo/
│   ├── videos/
│   ├── tracks/
│   ├── images/
│   ├── synthetic_demo_summary.json
│   └── synthetic_demo_summary.txt
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

results/synthetic_demo/videos/
    Annotated synthetic tracking videos showing the actual track bounding boxes,
    per-frame track IDs, available confidence values, and visual movement trails.

results/synthetic_demo/tracks/
    Per-video CSV observations of source frame, track ID, bounding-box geometry,
    available confidence, and drawing status.

results/synthetic_demo/images/
    Three illustrative PNG frames per processed video, decoded from its saved
    annotated MP4 at approximately 25%, 50%, and 75% of its frame sequence.
    These images require no second tracking-inference pass.

results/synthetic_demo/synthetic_demo_summary.json
    Synthetic media provenance hashes, processing settings, frame accounting,
    per-video track statistics, output paths, and processing status. The record
    also includes SHA-256 hashes of the script, implementation source files,
    source videos, and generated videos, CSV tables, and PNG images, together
    with available software version information.

results/synthetic_demo/synthetic_demo_summary.txt
    Human-readable rendering of the complete JSON summary, including per-video
    results and the same reproducibility metadata.

results/synthetic_demo_last_attempt.json and synthetic_demo_last_attempt.txt
    Separate diagnostic reports produced only when a media-processing run fails.
    These reports do not replace previously accepted synthetic_demo outputs.

Complete reproduction procedure
-------------------------------
The complete package consists of three independent execution paths.

1. Environment: Python with the installed physiotrack-v2 project and the
   validation dependencies listed above. The evaluated Conda environment is
   activated by:

   conda activate PhysioTrack-v2

2. Benchmark dataset: ECCV 2016 videos and XML annotations in the documented
   layout at ../datasets/FACE_TRACKING_ECCV2016 relative to physiotrack-v2/.

3. Working directory for all tracking validation commands:

   physiotrack-v2/validation/face_tracking/

4. Scientific benchmark evaluation and its dependent reporting artifacts:

   python eccv16_tracking_iou_selftest.py
   python eccv16_tracking_eval.py
   python eccv16_tracking_table.py
   python eccv16_tracking_plot.py
   python eccv16_tracking_qualitative.py

5. Isolated execution of the real detector-plus-tracker FaceAnalysis path:

   python face_tracking_component_test.py

6. Synthetic input media are located separately under:

   validation/synthetic_media/face_tracking_media/

   Their independent qualitative processing command is:

   python face_tracking_synthetic_demo.py

   The successful demonstration publishes annotated MP4 videos, track-observation
   CSV files, illustrative PNG frames, and the JSON and TXT summaries together.

The benchmark artifacts, component-execution outputs, and synthetic demonstration
outputs reside in separate result subdirectories. Benchmark annotations are
not inputs to the synthetic demonstration.

Reproducibility notes
---------------------
All validation scripts use paths derived from their own locations, with no
machine-specific source or result paths. The source benchmark and synthetic
input media are not modified by their respective validation scripts.

The benchmark evaluation uses a fresh OC-SORT instance for each independent
video. The synthetic demonstration follows the same sequence-local identity
policy and discovers supported videos recursively, without a fixed input list.
Its output filenames incorporate source-relative paths and deterministic hashes
to avoid collisions. All generated videos, observation tables, illustrative
frames, and both summaries are checked in a temporary staging directory. If
any required input fails or output verification fails, the previous synthetic_demo
results are retained. On successful verification, the full synthetic_demo
folder is replaced rather than merged with older files; the new published
folder is checked again before the obsolete backup and temporary staging
folder are removed. Benchmark and isolated-component outputs are not replaced
by this operation.

The completed scientific validation covers all eight sequences and all 42,007
frames. Equivalent numerical tracking results depend on consistent input media,
model weights, project implementation, tracker parameters, dependencies, and
evaluation protocol. Runtime and processing rate are environment-dependent
rather than measures of tracking accuracy.

The synthetic demonstration retains source frame order and recorded video frame
rate, subject to the available OpenCV codecs. The video renderer does not retain
source audio or variable-frame presentation timestamps. SHA-256 values in the
summary identify the exact processed source files.

Interpretation and limitations
------------------------------
The ECCV 2016 benchmark quantifies detector-plus-OC-SORT tracking performance
and temporal identity association against annotated trajectories. The isolated
component execution demonstrates successful operation and numerical output
integrity of the actual PhysioTrack FaceAnalysis tracking path; its PASS status
is not an accuracy metric.

The benchmark qualitative examples visualize selected exact MOTMetrics events
from T-ara. The independent synthetic demonstrations document tracking behaviour
under controlled illustrative conditions, without ground-truth matching or
benchmark scores.

The validated configuration achieves 86.47% recall and 67.89% matched IoU (MOTP)
but shows limited long-term identity association, including 2,302 identity
switches and an IDF1 of 6.53% across the benchmark sequences. A track ID
persists while the tracker maintains the associated trajectory but is not a
permanent biometric identity. Following sufficiently long disappearance, a
returning person can receive a new sequence-local track ID, as illustrated by
the synthetic enter/exit case. These outcomes are retained as measured behaviour
of the evaluated configuration.

Reproducibility artifact inventory
----------------------------------
- eccv16_tracking_iou_selftest.py
- eccv16_tracking_eval.py
- eccv16_tracking_table.py
- eccv16_tracking_plot.py
- eccv16_tracking_qualitative.py
- face_tracking_component_test.py
- face_tracking_synthetic_demo.py
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
- validation/synthetic_media/face_tracking_media/
- results/synthetic_demo/videos/
- results/synthetic_demo/tracks/
- results/synthetic_demo/images/
- results/synthetic_demo/synthetic_demo_summary.json
- results/synthetic_demo/synthetic_demo_summary.txt

Together, these artifacts document the scientific benchmark, benchmark
qualitative evidence, isolated component execution, and independent synthetic
qualitative demonstration.

Synthetic Qualitative Demonstration
-----------------------------------

Purpose
~~~~~~~
The demonstration illustrates the qualitative behaviour of the actual
PhysioTrack Face Tracking component on independently prepared synthetic 3D-style
video sequences. It is separate from the ECCV 2016 scientific benchmark and
is not an additional quantitative tracking-accuracy evaluation.

Synthetic media
~~~~~~~~~~~~~~~
The five synthetic video sequences represent distinct tracking conditions:

- 01_single_face.mp4: a single face over time.
- 02_multiple_faces.mp4: simultaneous tracking of three faces.
- 03_face_occlusion.mp4: temporary occlusion of a tracked face.
- 04_enter_exit.mp4: a person exits the visible scene and later re-enters.
- 05_crossing_faces.mp4: two face trajectories cross in the same scene.

The input media are stored in:

validation/synthetic_media/face_tracking_media/

These videos are not part of the ECCV 2016 dataset and are not associated with
its ground-truth trajectory annotations. The demonstration discovers supported
videos recursively within the designated input directory; it has no fixed video
name list or fixed number of inputs.

Execution
~~~~~~~~~
The independent demonstration script is:

validation/face_tracking/face_tracking_synthetic_demo.py

Each video frame is processed by the real PhysioTrack Face detector and
FaceAnalysis pipeline with FaceTracker configured for OC-SORT. Unrelated face
analysis modules are disabled. A new OC-SORT instance is created for each video,
so track IDs are local to each video.

The evaluated configuration uses CPU, detector confidence 0.25, and detector IoU
threshold 0.45. Annotated frames display the actual tracking boxes, track IDs,
available confidence scores, and limited-length movement trails derived from
observed track positions. The movement trails connect previously observed
track centers for display; no tracking observations are fabricated for frames
without a track.

The outputs are stored in:

validation/face_tracking/results/synthetic_demo/

The videos/ subdirectory contains annotated MP4 videos; tracks/ contains a
corresponding per-video CSV of actual tracking observations and empty-track
frames. The images/ subdirectory contains three PNG frames per video, extracted
at approximately 25%, 50%, and 75% of the saved annotated video without another
inference pass. Both synthetic_demo_summary.json and synthetic_demo_summary.txt
record source-media hashes, software and model settings, source-code hashes,
video properties, tracking counts, generated-output hashes, paths, and status.
Every input must succeed and all staged outputs must pass validation before
publication. The previous synthetic_demo directory is replaced in full, not
merged; the newly published results are verified before the obsolete backup
and staging artifacts are cleaned. Failed attempts preserve accepted results.

Validated synthetic results
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The five videos were processed successfully, with no reported processing
failures. All video outputs were verified at 1280 x 720 pixels and 30 FPS. The
encoded outputs use the mp4v video codec.

Video                         Frames    Track obs.   Distinct IDs   No-track frames
----------------------------  --------  -----------  -------------  ---------------
01_single_face.mp4                300          300              1                0
02_multiple_faces.mp4             300          900              3                0
03_face_occlusion.mp4             300          272              1               28
04_enter_exit.mp4                 360          481              3               43
05_crossing_faces.mp4             300          600              2                0
----------------------------  --------  -----------  -------------  ---------------
TOTAL                            1560         2553              -               71

The synthetic execution recorded 1,560 processed frames, 2,553 track
observations, 71 frames with no tracks, and zero invalid track observations.
The five track CSV files contain 2,624 rows: one per tracked observation and
one explicit NO_TRACK row for each frame without track output.
The demonstration also exports 15 illustrative PNG frames, three per video,
from the saved annotated video outputs.
Distinct-ID counts represent tracker-assigned sequence-local IDs rather than
ground-truth unique individuals.

Qualitative interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~
The single-face example retained ID 1 throughout its 300 frames. The
multiple-face sequence retained three distinct track identities across all 300
frames. In the occlusion example, ID 1 was not reported during frames 164-191
(28 frames), but the same ID was reported again from frame 192 onward.

In the enter/exit sequence, the first individual's ID 1 was last reported at
frame 191; after leaving the visible scene, that individual returned under the
new ID 3 beginning at frame 284. The other visible individual's ID 2 continued
through the remainder of the sequence. This illustrates a loss of identity
continuity following an extended absence, not an adjustment made by the
visualization script.

In the crossing-faces example, two track IDs were reported in every frame of
the 300-frame sequence. The displayed tracks maintained their associations
through the represented crossing. The synthetic demonstration does not contain
ground-truth identities and therefore does not calculate MOTMetrics identity
switches, MOTA, IDF1, Precision, Recall, or other benchmark metrics.

Media provenance
~~~~~~~~~~~~~~~~
The input assets are five synthetic 3D-style video sequences prepared
independently of the ECCV 2016 benchmark videos. The source files remain in
validation/synthetic_media/face_tracking_media/. Source SHA-256 hashes recorded
in synthetic_demo_summary.json identify the exact media processed in the
reported execution.

The five synthetic videos were created using Blender with a character based on
Blender Studio & Community, Human Base Meshes v1.4.1 (Body Male - Realistic).
The original base mesh is distributed under the Creative Commons CC0 1.0
Universal public-domain dedication. The character was adapted and animated in
Blender with ChatGPT assistance to illustrate single-person, multi-person,
occlusion, scene exit/re-entry, and crossing-face tracking conditions.

Original 3D asset source:
https://www.blender.org/download/demo-files/#assets

Original base-mesh license (CC0 1.0 Universal):
https://creativecommons.org/publicdomain/zero/1.0/

The CC0 designation applies to the original Blender base mesh; it is not
presented as a separate license determination for the finished videos. These
videos are independently created demonstration media, not excerpts from the
ECCV 2016 benchmark, and provide no benchmark tracking ground truth.


Obtaining and Preparing Benchmark Media
---------------------------------------
The ECCV 2016 tracking benchmark videos, extracted frames, and
derived annotated visualizations are not distributed with the
public GitHub repository.

Users must obtain the original benchmark videos and corresponding
XML annotations from the official ECCV 2016 Face Tracking project page
provided in the Dataset section above, following the documented video
and XML file requirements and applicable source usage conditions.

Place the acquired files under:

<workspace>/datasets/FACE_TRACKING_ECCV2016/

The datasets directory must be alongside the physiotrack-v2
repository. Preserve the benchmark directory layout and filenames
specified in the dataset preparation section above.

After preparing the benchmark inputs, execute the tracking validation
scripts in the documented order. Benchmark-derived annotated videos,
representative frames, and qualitative contact sheets remain local.

Numerical tracking results and independent plots may be retained
separately. The independently created synthetic tracking videos are
not excerpts from the ECCV 2016 benchmark.

See VALIDATION_MEDIA_LICENSES.md for distribution restrictions.