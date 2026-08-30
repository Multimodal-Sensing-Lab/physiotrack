CelebAMask-HQ Face Region Segmentation Validation
==================================================

Purpose
-------
This README describes how to reproduce the CelebAMask-HQ face-region
segmentation validation used for the PhysioTrack thesis project.

The validation evaluates the PhysioTrack FaceRegions component on the
CelebAMask-HQ benchmark and generates the final numerical results, summary
tables, and per-class performance figure.

Dataset
-------
Name:
CelebAMask-HQ

Official project:
https://github.com/switchablenorms/CelebAMask-HQ

Task:
Semantic face-region segmentation

Dataset size:
30,000 aligned face images with pixel-level semantic annotations.

Original image resolution:
1024 x 1024

Annotation resolution:
512 x 512

Benchmark split used:
The CelebAMask-HQ subset corresponding to the official CelebA test partition.

The split is derived by combining:

- CelebA-HQ-to-CelebA-mapping.txt
- list_eval_partition.txt

The resulting CelebAMask-HQ split sizes are:

- Training: 24,183 images
- Validation: 2,993 images
- Test: 2,824 images

Only the 2,824-image test subset is evaluated.

Dataset Location
----------------
Place the extracted CelebAMask-HQ dataset directly under the project-level
datasets directory:

datasets/CelebAMask-HQ

The expected structure is:

datasets/
└── CelebAMask-HQ/
    ├── CelebA-HQ-img/
    ├── CelebAMask-HQ-mask-anno/
    └── CelebA-HQ-to-CelebA-mapping.txt

No machine-specific dataset path is required.

Validation Files
----------------
The validation package is located in:

physiotrack/validation/face_regions/

Required files:

1. celebamaskhq_segmentation_eval.py

   Runs PhysioTrack FaceRegions on the complete 2,824-image evaluation split,
   constructs the ground-truth semantic maps, accumulates the dataset-level
   confusion matrix, computes all segmentation metrics, and writes the raw
   evaluation results.

2. celebamaskhq_segmentation_plot.py

   Reads the generated evaluation results and creates the final thesis tables
   and per-class IoU/Dice figure.

3. list_eval_partition.txt

   Official CelebA train/validation/test partition metadata used with
   CelebA-HQ-to-CelebA-mapping.txt to identify the CelebAMask-HQ test subset.

The partition file is stored with the validation package so that the downloaded
CelebAMask-HQ directory can remain unchanged.

Path Handling
-------------
The evaluation scripts do not contain a user-specific absolute path.

The project location is determined automatically from the location of the
validation script, and the evaluator expects the dataset at:

datasets/CelebAMask-HQ

Therefore, the evaluation can be reproduced on another machine without editing
dataset paths as long as the documented project structure is preserved.

Evaluated Component
-------------------
PhysioTrack FaceRegions

Segmentation backend:
SegFace

Inference device used for the reported validation:
CPU

Evaluation Protocol
-------------------
Evaluation images:
2,824

Successful images:
2,824

Failed images:
0

Input resolution:
512 x 512

Ground-truth resolution:
512 x 512

Initialization:
Full aligned-image bounding box

Number of semantic classes:
19

Aggregation:
Single dataset-level 19 x 19 confusion matrix over all evaluated pixels

Each original 1024 x 1024 CelebAMask-HQ image is resized to 512 x 512 using
OpenCV linear interpolation. The ground-truth masks remain at their native
512 x 512 resolution.

A bounding box covering the complete aligned image is supplied to FaceRegions.
This isolates semantic segmentation performance from face-detector localization
errors.

Ground-Truth Construction
-------------------------
CelebAMask-HQ provides separate binary masks for the semantic regions. The
evaluation combines them into one 19-class label map using the class order
expected by the SegFace-based PhysioTrack component:

0   background
1   neck
2   skin
3   cloth
4   l_ear
5   r_ear
6   l_brow
7   r_brow
8   l_eye
9   r_eye
10  nose
11  mouth
12  l_lip
13  u_lip
14  hair
15  eye_g
16  hat
17  ear_r
18  neck_l

Masks are merged in this order. If semantic masks overlap, a later class in the
sequence overwrites an earlier class, matching the label-map construction used
by the evaluation implementation.

Evaluation Metrics
------------------
The following metrics are calculated from the accumulated dataset-level
confusion matrix:

Pixel Accuracy

    Sum of correctly classified pixels
    -----------------------------------
         Total evaluated pixels

Per-class Intersection over Union (IoU)

                  TP
    IoU = -------------------
          TP + FP + FN

Per-class Dice score

                 2 TP
    Dice = ----------------
           2 TP + FP + FN

All-class mIoU:
Mean IoU over all ground-truth classes present in the evaluation, including
background.

Foreground mIoU:
Mean IoU over the 18 foreground semantic classes.

All-class Mean Dice:
Mean Dice over all ground-truth classes present in the evaluation, including
background.

Foreground Mean Dice:
Mean Dice over the 18 foreground semantic classes.

Metrics are calculated from one confusion matrix accumulated over the complete
dataset. Per-image IoU values are not averaged.

Running the Evaluation
----------------------
Activate the PhysioTrack thesis environment and move to the validation
directory.

Example:

conda activate PhysioTrack-Thesis
cd /d <project-path>\physiotrack\validation\face_regions

Run the complete evaluation:

python celebamaskhq_segmentation_eval.py

The complete CPU evaluation can take several hours.

After the evaluator finishes successfully, generate the final tables and
figure:

python celebamaskhq_segmentation_plot.py

Generated Results
-----------------
The evaluator writes the following files under:

physiotrack/validation/face_regions/results/

celebamaskhq_class_metrics.csv
    Per-class ground-truth pixel counts, predicted pixel counts, IoU, and Dice.

celebamaskhq_confusion_matrix.csv
    Complete 19 x 19 dataset-level confusion matrix.

celebamaskhq_segmentation_summary.txt
    Evaluation configuration, execution statistics, overall metrics, and
    per-class metrics.

The plot/table script additionally creates:

celebamaskhq_thesis_table.csv
celebamaskhq_thesis_table.md
    Final per-class IoU and Dice table.

celebamaskhq_summary_table.csv
celebamaskhq_summary_table.md
    Final overall performance table.

results/figures/celebamaskhq_per_class_metrics.png
    Per-class IoU and Dice performance figure.


Qualitative Benchmark Evidence
------------------------------
A separate qualitative script is included to visualize the documented
CelebAMask-HQ segmentation protocol on selected benchmark images:

celebamaskhq_segmentation_qualitative.py

The qualitative analysis does not replace or modify the full quantitative
evaluation. It uses the same PhysioTrack FaceRegions component, SegFace
backend, 512 x 512 input size, 19-class label definition, CPU inference
configuration, and full aligned-image initialization used by the quantitative evaluator.

The script first profiles the official 2,824-image test split using only the
ground-truth semantic masks. It then builds a deterministic candidate pool and
runs PhysioTrack inference only on those candidate images.

Eight qualitative cases are selected:

- strong_candidate
- representative_candidate
- challenging_candidate
- eye_glasses
- hat
- earring
- necklace
- high_semantic_diversity

The strong, representative, and challenging cases are selected using
image-level foreground mIoU diagnostics within the deterministic candidate
pool.

The accessory-specific examples are selected to make the corresponding
CelebAMask-HQ semantic classes visually interpretable:

- eye_g
- hat
- ear_r
- neck_l

The high-semantic-diversity case is selected from images containing many
ground-truth foreground classes.

Each qualitative figure contains:

- Original image
- Ground-truth semantic overlay
- PhysioTrack prediction overlay
- Image-level qualitative diagnostics
- Full-benchmark metrics
- Evaluation protocol summary
- Complete semantic-class color legend

The image-level Pixel Accuracy, mIoU, and Dice values shown in the side panel
are diagnostic values for the displayed image only. They are not substituted
for the reported benchmark metrics. The official reported results remain the
metrics calculated from the single dataset-level confusion matrix over all
2,824 test images.

No face-detection rectangle is drawn in the qualitative figures because the
documented protocol initializes FaceRegions with a bounding box covering the
entire aligned 512 x 512 image. Adding a visible face box would therefore not
provide additional localization information and could incorrectly suggest that
face detection is being evaluated.

Run the qualitative analysis after the quantitative result files are
present:

python celebamaskhq_segmentation_qualitative.py

The qualitative outputs are written under:

results/qualitative/

Expected qualitative outputs include:

results/qualitative/annotated_images/
    Eight individual qualitative evidence figures.

results/qualitative/celebamaskhq_qualitative_selection.csv
    Selection rationale, source image, image-level diagnostics, semantic
    content information, and generated output path for each selected case.

results/figures/celebamaskhq_qualitative_examples.png
    Combined 2 x 4 summary figure containing the eight selected PhysioTrack
    prediction overlays.

The qualitative generator owns and may replace only its own qualitative
outputs. It does not delete or modify the quantitative CSV files,
confusion matrix, summary, thesis tables, or per-class performance figure.

Quantitative Results
------------------------
Official test split size:
2,824 images

Successful images:
2,824

Failed images:
0

Total evaluated pixels:
740,294,656

Pixel Accuracy:
95.5849%

All-class mIoU:
81.5328%

Foreground mIoU:
80.8718%

All-class Mean Dice:
89.3567%

Foreground Mean Dice:
88.9541%

Per-Class Results
-----------------
Class          IoU (%)    Dice (%)
background       93.43        96.60
neck             85.30        92.07
skin             93.57        96.68
cloth            81.40        89.75
l_ear            80.71        89.33
r_ear            80.03        88.91
l_brow           77.31        87.20
r_brow           77.11        87.08
l_eye            83.36        90.92
r_eye            83.58        91.05
nose             89.04        94.20
mouth            86.99        93.04
l_lip            84.44        91.56
u_lip            82.35        90.32
hair             92.23        95.96
eye_g            91.21        95.40
hat              80.56        89.24
ear_r            60.11        75.09
neck_l           46.38        63.37

Reported Runtime
-------------------------
Elapsed time:
13,793.32 seconds

Throughput:
0.2047 images/second

Execution time is hardware- and environment-dependent and is not expected to
match across machines. The segmentation metrics are the reproducibility target.

Internal Consistency Verification
-------------------
The evaluation outputs were checked for internal consistency.

The validation completed all 2,824 test images without inference failures.

The accumulated confusion matrix contains exactly:

2,824 x 512 x 512 = 740,294,656 pixels

The overall and per-class metrics recomputed directly from the confusion matrix
match the generated class-metrics CSV, text summary, thesis tables, and figure.

Interpretation
--------------
The FaceRegions component shows strong segmentation performance on the
CelebAMask-HQ test subset. Background, skin, hair, eye_g, and nose obtain the
highest IoU values, while most major facial regions achieve IoU values near or
above 0.80.

The weakest classes are ear_r and neck_l. These are relatively small semantic
regions and are more difficult to segment reliably than the dominant face
regions.

Evaluation Scope
----------------
This evaluation measures semantic face-region segmentation under aligned-image
initialization.

It does not measure an end-to-end face-detection-plus-segmentation pipeline,
because the full aligned image is supplied directly as the segmentation region.

The reported results should therefore be described as PhysioTrack FaceRegions
segmentation performance under the documented CelebAMask-HQ aligned-image
evaluation protocol.

Reproducibility
---------------
To reproduce the validation on another machine:

1. Obtain and extract the CelebAMask-HQ dataset.
2. Place the extracted dataset at datasets/CelebAMask-HQ.
3. Keep list_eval_partition.txt in the face_regions validation directory.
4. Install the PhysioTrack project and its required dependencies.
5. Run celebamaskhq_segmentation_eval.py.
6. Run celebamaskhq_segmentation_plot.py.
7. Run celebamaskhq_segmentation_qualitative.py.
8. Compare the generated quantitative files with the reported metrics above
   and inspect the generated qualitative evidence under results/qualitative/
   and results/figures/.

No user-specific absolute path or undocumented manual split is required.
