ECCV 2016 Music Video Face Tracking Validation
================================================

Overview
--------
This validation evaluates the PhysioTrack multi-face tracking pipeline on the
ECCV 2016 Music Video Face Tracking dataset introduced with:

Shun Zhang, Yihong Gong, Jia-Bin Huang, Jongwoo Lim, Jinjun Wang,
Narendra Ahuja, and Ming-Hsuan Yang,
"Tracking Persons-of-Interest via Adaptive Discriminative Features,"
European Conference on Computer Vision (ECCV), 2016.

The evaluation measures the complete detector-plus-tracker chain. Face
detections are produced by the PhysioTrack face detector and associated over
time by the PhysioTrack FaceTracker using the OC-SORT backend.

Dataset Source
--------------
Official project page:

https://sites.google.com/site/shunzhang876/eccv16-face-tracking

The project page provides the music-video dataset and its ground-truth
annotations through the authors' download links.

Required Dataset Content
------------------------
Download the eight music videos and the corresponding ground-truth XML
annotations.

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

The ground-truth package may contain additional XML files that are not part of
this eight-video music-video validation. They are not used by the evaluator.

Dataset Layout
--------------
Extract the dataset under the project-level datasets directory using the
following structure:

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

The validation scripts automatically derive the dataset location from the
project structure. Ground-Truth Format
-------------------
The dataset annotations are stored as XML trajectories. Each trajectory has an
object identity and contains frame-level face bounding boxes represented by:

frame number, x, y, width, height

The XML frame numbering is one-based. The evaluator converts every annotated
box to:

[x1, y1, x2, y2]

and uses the trajectory object identifier as the ground-truth identity.

Validation Files
----------------
The validation implementation is located under:

validation/face_tracking/

The required scripts are:

- eccv16_tracking_eval.py
  Runs the complete face detector and OC-SORT tracker on all eight videos,
  compares predicted tracks with ground truth, computes tracking metrics, and
  writes the detailed CSV and validation summary.

- eccv16_tracking_table.py
  Creates compact CSV and Markdown tables from the detailed evaluation results.

- eccv16_tracking_plot.py
  Creates the final tracking-performance figure from the detailed CSV results.

- eccv16_tracking_qualitative.py
  Generates verified qualitative face-tracking evidence from the T-ara
  benchmark sequence. The script uses the same detector, OC-SORT tracker,
  matching threshold, and MOTMetrics event protocol as the quantitative
  evaluation.

Environment and Dependencies
----------------------------
The validation requires the PhysioTrack project environment and the packages
used by the face detector, tracker, video reader, evaluator, and plotting
scripts.

Important evaluation dependencies include:

- OpenCV
- NumPy
- motmetrics 1.4.0
- pandas
- matplotlib
- the PhysioTrack package and its face-tracking dependencies

The evaluator and qualitative generator contain a small compatibility shim for
motmetrics 1.4.0 when used with NumPy versions in which np.asfarray is no
longer available. The tracking evaluation uses the dependency-compatible implementation while preserving the documented tracking
metrics or evaluation logic.

Evaluation Configuration
------------------------
The evaluation uses:

- Face detector confidence threshold: 0.25
- Face detector IoU threshold: 0.45
- Tracker backend: OC-SORT
- Tracking device: CPU
- Ground-truth/prediction matching IoU threshold: 0.50
- Evaluation library: motmetrics 1.4.0

A new tracker instance is created for every video sequence. Tracker identities
therefore remain sequence-local, as expected for independent tracking
sequences.

Evaluation Protocol
-------------------
For every video frame:

1. The PhysioTrack face detector produces face detections.
2. The detections are passed to FaceTracker.
3. FaceTracker forwards the detections to the OC-SORT tracker.
4. OC-SORT returns tracked face boxes and persistent sequence-local track IDs.
5. Predicted boxes and ground-truth boxes are matched using an IoU threshold
   of 0.50.
6. The frame-level associations are accumulated by motmetrics.

The full detector-plus-tracker pipeline is evaluated. Consequently, detector
misses, false detections, localization errors, and tracker association errors
all contribute to the reported results.

Metrics
-------
The validation reports:

- Recall
- Precision
- F1
- False Alarms per Frame (FAF)
- Identity Switches (IDS)
- Fragmentations (Frag)
- Multiple Object Tracking Accuracy (MOTA)
- Multiple Object Tracking Precision (MOTP)
- IDF1

motmetrics represents MOTP as an IoU distance for this evaluation. The
reported MOTP value is converted to mean matched IoU as:

MOTP IoU = 1 - raw MOTP distance

Run Order
---------
From the PhysioTrack face-tracking validation directory, run:

python eccv16_tracking_eval.py
python eccv16_tracking_table.py
python eccv16_tracking_plot.py
python eccv16_tracking_qualitative.py

The evaluator is the computationally expensive quantitative stage because it
processes all frames of all eight videos. The table and plot scripts operate
on the generated evaluation CSV.

The qualitative script is also computationally expensive because it re-runs
the complete T-ara sequence before rendering the final qualitative clip. This
verification pass is intentional and is used to confirm that the qualitative
run reproduces the T-ara quantitative result before any final
qualitative outputs are generated.

Quantitative Results
-----------------------------
The eight-video evaluation contains:

- Videos: 8
- Frames: 42,007
- Ground-truth objects loaded by the evaluator: 95,302

The overall metrics are:

- Recall: 91.07%
- Precision: 72.01%
- F1: 80.43%
- FAF: 0.8031
- IDS: 2,333
- Fragmentations: 2,340
- MOTA: 53.22%
- MOTP: 87.81%
- IDF1: 6.81%

Small differences in runtime or processing FPS are expected across executions
and machines. They do not change the benchmark accuracy results.

Result Interpretation
---------------------
The results show that the pipeline provides strong face coverage and generally
accurate spatial localization. The overall recall is 91.07%, the F1 score is
80.43%, and the mean matched IoU reported as MOTP is 87.81%.

Tracking performance varies substantially across the individual videos. In
particular, Westlife produces a large number of false-positive predictions,
which results in a negative MOTA for that sequence and reduces the overall
MOTA.

The low overall IDF1 of 6.81% is an important limitation of the current
detector-plus-OC-SORT configuration. Inspection of the PhysioTrack tracking path
confirms that FaceTracker does not regenerate or remap identities on every
frame. FaceTracker forwards detections to the tracker, while OC-SORT assigns a
persistent ID when a Kalman track is created and retains that ID for the
lifetime of the track.

The tracker is reset only when a new independent video sequence is started,
which is the correct behavior for this benchmark. The low IDF1 therefore is
not explained by an unintended per-frame identifier reset in the inspected
implementation.

OC-SORT in this configuration relies primarily on motion and spatial
association rather than a strong face re-identification embedding. The ECCV
2016 music videos contain difficult conditions such as shot and scene changes,
occlusions, rapid camera motion, and substantial appearance variation.
Consequently, a person whose track is lost may later receive a new track
identity. This explains how the system can retain good detection coverage and
localization while showing weak long-term identity consistency.

The reported IDF1, identity-switch, and fragmentation results should therefore
be treated as a documented limitation of the current tracking configuration rather than
hidden or replaced by detection-oriented metrics.

Qualitative Benchmark Evidence
------------------------------
The qualitative generator uses the T-ara benchmark sequence as a detailed
tracking example because it contains successful multi-face tracking together
with real identity switches, missed detections, false positives, and
localization variation.

The full-sequence T-ara metrics are:

- Recall: 87.40%
- Precision: 96.65%
- F1: 91.80%
- FAF: 0.0968
- IDS: 267
- Fragmentations: 311
- MOTA: 82.54%
- MOTP: 94.73%
- IDF1: 5.79%

Before producing qualitative evidence, eccv16_tracking_qualitative.py runs the
complete T-ara sequence using the same detector, tracker, IoU threshold, and
MOTMetrics accumulator protocol as eccv16_tracking_eval.py. The generated
T-ara metrics are compared with the corresponding row in
results/eccv16_tracking_results.csv.

Qualitative outputs are generated only after this quantitative identity check
passes. This guards against producing visually convincing tracking evidence
from a run that does not match the documented benchmark configuration.

The qualitative clip is 60 seconds long. The selected window is chosen
deterministically from the complete T-ara sequence. The primary selection
criterion is the number of exact MOTMetrics SWITCH events. Ties are resolved
by preferring more successful detections, fewer missed detections, and fewer
false-positive observations.

The qualitative window is:

- Source sequence: T-ara
- Source frames: 2,689 to 4,486
- Duration: approximately 59.99 seconds
- Exact MOTMetrics SWITCH events: 114
- Detected face observations: 5,134
- MISS events: 706
- FP events: 154

The 5,134 detected face observations consist of:

- MATCH events: 5,020
- SWITCH events: 114

The event labels shown in the video are not custom approximations. MATCH,
SWITCH, MISS, and FP are read from the MOTMetrics event history produced using
the same matching protocol as the quantitative evaluation.

The video panel contains three distinct levels of information:

Current frame
    Shows the number of ground-truth faces, active tracks, exact MOTMetrics
    MATCH, SWITCH, MISS, and FP events, and mean matched IoU for the current
    frame.

Cumulative 60-second clip
    Shows the running MATCH, SWITCH, MISS, and FP event totals from the start of
    the selected qualitative clip to the current frame.

Full-sequence results
    Shows the fixed T-ara benchmark metrics read from the quantitative
    result CSV. These values are not recomputed from the 60-second clip.

The frame overlays use:

- Green: matched predicted track
- Magenta: MOTMetrics identity switch
- Red: missed ground-truth face
- Orange: false-positive track
- Cyan: ground-truth reference box

Matched tracks show both the OC-SORT track identity and the corresponding
ground-truth identity. Identity-switch labels show the exact ground-truth
identity together with the preceding and current track identities when the existing
assignment is available.

The qualitative renderer uses collision-aware label placement so that labels
are moved away from one another when faces are close together. This improves
readability without changing any underlying detection, tracking, ground-truth,
or MOTMetrics information.

Representative Qualitative Frames
---------------------------------
Six deterministic representative frames are generated from the selected
60-second T-ara clip:

- stable_multi_face
  Successful simultaneous tracking of multiple faces without a SWITCH, MISS, or
  FP event in the selected frame.

- identity_switch
  A frame containing exact MOTMetrics SWITCH events.

- false_negative
  A frame selected to show missed ground-truth faces.

- false_positive
  A frame selected to show false-positive tracks together with valid matches.

- crowded_tracking
  A frame selected to show tracking behavior when multiple faces are visible.

- localization_challenge
  A matched frame with comparatively lower matched IoU, used to illustrate
  spatial localization difficulty while retaining a valid match.

These frames are evidence examples only. They do not replace or alter the
full-sequence quantitative metrics.

Qualitative Output Cleanup
--------------------------
The qualitative generator owns only its qualitative artifacts.

After the T-ara quantitative consistency check succeeds, the
script removes the existing:

results/qualitative/

directory and recreates its qualitative files from scratch. It also replaces:

results/figures/eccv16_tracking_qualitative_examples.png

The script does not delete or modify:

- results/eccv16_tracking_results.csv
- results/eccv16_tracking_summary.txt
- results/eccv16_tracking_thesis_table.csv
- results/eccv16_tracking_thesis_table.md
- results/figures/eccv16_tracking_metrics.png
- any benchmark dataset file

The scientific verification is deliberately performed before qualitative
cleanup. If the T-ara reproduction does not match the quantitative
result, the script stops before replacing the existing qualitative evidence.

Outputs
-------
All generated outputs are stored outside the dataset under:

validation/face_tracking/results/

Expected final outputs are:

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
└── figures/
    ├── eccv16_tracking_metrics.png
    └── eccv16_tracking_qualitative_examples.png

eccv16_tracking_results.csv
    Detailed quantitative results for the eight benchmark videos and the
    overall evaluation.

eccv16_tracking_summary.txt
    Detailed record of the quantitative validation setup, sequence results,
    overall results, and runtime information.

eccv16_tracking_thesis_table.csv
    Compact CSV version of the final quantitative tracking results.

eccv16_tracking_thesis_table.md
    Markdown version of the final quantitative tracking results.

figures/eccv16_tracking_metrics.png
    Quantitative tracking-performance figure.

qualitative/annotated_videos/T-ara_face_tracking_qualitative.mp4
    Full-HD 60-second qualitative benchmark video with exact MOTMetrics events,
    track identities, ground-truth identities, current-frame statistics,
    cumulative clip events, and full-sequence T-ara metrics.

qualitative/representative_frames/
    Six tracking-focused representative frames covering successful tracking,
    identity switches, false negatives, false positives, crowded tracking, and
    localization difficulty.

qualitative/eccv16_tracking_qualitative_events.csv
    Exact MOTMetrics event records for the selected qualitative clip.

qualitative/eccv16_tracking_qualitative_selection.csv
    Machine-readable record of the selected clip, selection rule, T-ara metrics, and representative-frame roles.

figures/eccv16_tracking_qualitative_examples.png
    Combined qualitative overview figure generated from the six representative
    benchmark frames.

The dataset directory is treated as read-only benchmark input. Validation
outputs are not written into the dataset.

Reproducibility
---------------------
The dataset location is resolved from the documented project structure. Runtime and processing speed may vary between systems because they depend on the execution environment and system load.

The qualitative generator follows the same portability rule. It derives the
project root from the location of the validation script and accesses the
benchmark through:

datasets/FACE_TRACKING_ECCV2016

A complete reproduction consists of:

1. Preparing the dataset using the documented directory structure.
2. Running eccv16_tracking_eval.py.
3. Running eccv16_tracking_table.py.
4. Running eccv16_tracking_plot.py.
5. Confirming that the quantitative results match the values documented above.
6. Running eccv16_tracking_qualitative.py.
7. Confirming that the T-ara quantitative verification succeeds.
8. Confirming that the 60-second annotated video, six representative frames,
   exact clip-event CSV, selection CSV, and qualitative summary figure are
   generated.

The qualitative outputs are intended to complement the quantitative benchmark
with interpretable evidence of temporal identity behavior, successful
tracking, identity switches, misses, false positives, and localization
variation.

The validation demonstrates the behavior of the current PhysioTrack
detector-plus-OC-SORT face-tracking pipeline on this benchmark. It should not
be interpreted as evidence of perfect identity preservation in arbitrary
videos or as an evaluation of OC-SORT independently from the detector.
