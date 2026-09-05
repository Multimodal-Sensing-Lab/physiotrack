300-W Facial Landmark Validation
================================

Overview
--------
This validation evaluates the PhysioTrack FaceLandmarks component on the
300-W facial landmark dataset using the MediaPipe Face Landmarker backend.

Dataset
-------
Official source:

https://ibug.doc.ic.ac.uk/resources/300-W/

Required dataset structure:

datasets/
└── 300W/
    ├── 01_Indoor/
    │   ├── indoor_001.png
    │   ├── indoor_001.pts
    │   └── ...
    └── 02_Outdoor/
        ├── outdoor_001.png
        ├── outdoor_001.pts
        └── ...

The validation uses 300 Indoor and 300 Outdoor images. Each image must have a
corresponding 68-point PTS annotation file.

The optional 300-W bounding-box MAT files are not required by this evaluation.

Validation Files
----------------
The validation implementation is located under:

validation/face_landmarks/

Files:

- 300w_landmark_eval.py
- mediapipe_300w_mapping.py
- 300w_landmark_table.py
- 300w_landmark_plot.py
- 300w_landmark_qualitative.py

Model
-----
The evaluation uses the MediaPipe model:

face_landmarker.task

The evaluator resolves the model automatically from supported local locations.
An explicit path can also be provided through:

PHYSIOTRACK_FACE_LANDMARKER_MODEL

Evaluation Protocol
-------------------
The evaluation uses the 51-point 300-W protocol by excluding the 17 face-border
landmarks from the original 68-point annotations.

A fixed anatomical mapping selects the corresponding 51 MediaPipe landmarks.

Normalized Mean Error (NME) is computed using the inter-ocular distance
between the outer eye corners.

For controlled landmark-localization evaluation, the face region is initialized
from the ground-truth landmarks with 20 percent padding before calling:

FaceLandmarks.predict_face()

This isolates landmark localization from face detection and face selection.

Run
---
From the validation directory:

python 300w_landmark_eval.py
python 300w_landmark_table.py
python 300w_landmark_plot.py
python 300w_landmark_qualitative.py

Quantitative Results
----------------
The evaluation covers 600 images.

Overall results:

- Successful predictions: 588
- Failed predictions: 12
- Detection rate: 98.00%
- Mean NME: 4.6503%
- Median NME: 4.3178%
- NME standard deviation: 1.8048%

Overall CED values:

- NME <= 5%: 72.6667%
- NME <= 6%: 88.3333%
- NME <= 8%: 95.8333%
- NME <= 10%: 96.6667%

Outputs
-------
Generated files are stored under:

validation/face_landmarks/results/

Generated outputs:

results/
├── 300w_landmark_results.csv
├── 300w_landmark_summary.txt
├── 300w_landmark_ced.csv
├── 300w_landmark_thesis_table.csv
├── 300w_landmark_thesis_table.md
├── qualitative/
│   └── qualitative evidence for eight selected cases
└── figures/
    ├── 300w_landmark_ced.png
    └── 300w_landmark_qualitative_examples.png


Qualitative Evidence
--------------------
A separate qualitative script is included to complement the full quantitative
benchmark:

300w_landmark_qualitative.py

The qualitative analysis does not replace or modify the 600-image quantitative
evaluation. It provides visual evidence for eight selected 300-W cases using
the same PhysioTrack FaceLandmarks component and the documented controlled
face-initialization protocol.

The individual qualitative artifacts are written under:

results/qualitative/

A combined thesis-ready summary figure containing the eight qualitative cases
is written to:

results/figures/300w_landmark_qualitative_examples.png

The qualitative evidence is intended to make representative successes and
challenging landmark-localization cases visually interpretable. The reported
benchmark result remains the aggregate quantitative evaluation over all 600
images, with 588 successful predictions and 12 failed predictions.

The qualitative generator owns only its qualitative outputs. It does not
replace the quantitative result CSV, summary, CED data, thesis table, or CED
figure.

Methodological Note
-------------------
The reported values represent the PhysioTrack landmark component under this
controlled evaluation protocol. Because MediaPipe and 300-W use different
landmark definitions and the face region is initialized from ground-truth
landmarks, the results should not be presented as official 300-W competition
or leaderboard results.


Final Files to Preserve
-----------------------
Final reproducibility artifacts:

- 300w_landmark_eval.py
- mediapipe_300w_mapping.py
- 300w_landmark_table.py
- 300w_landmark_plot.py
- 300w_landmark_qualitative.py
- README_300W_FACIAL_LANDMARKS.txt
- results/300w_landmark_results.csv
- results/300w_landmark_summary.txt
- results/300w_landmark_ced.csv
- results/300w_landmark_thesis_table.csv
- results/300w_landmark_thesis_table.md
- results/figures/300w_landmark_ced.png
- results/figures/300w_landmark_qualitative_examples.png
- qualitative artifacts generated under results/qualitative/

Generated caches and obsolete temporary diagnostic files are not part of the
final validation deliverables.
