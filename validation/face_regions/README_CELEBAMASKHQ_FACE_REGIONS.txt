CelebAMask-HQ Face Region Segmentation Validation
==================================================

Purpose
-------
This README describes how to reproduce the CelebAMask-HQ face-region
segmentation validation used for the PhysioTrack thesis project.

The validation package provides four complementary forms of evidence:

1. Quantitative scientific evaluation of the PhysioTrack FaceRegions component
   on the complete CelebAMask-HQ test subset.
2. Qualitative benchmark evidence showing representative, challenging, and
   accessory-specific segmentation cases.
3. Isolated component-execution evidence confirming that the real PhysioTrack
   FaceRegions path runs through FaceAnalysis and produces internally
   consistent structured numerical outputs.
4. Independent synthetic qualitative demonstration of real SegFace predictions
   on artificial mannequin images, without benchmark ground truth or accuracy
   scores.

The isolated component-execution PASS status is software evidence only. It does
not replace the scientific IoU, Dice, pixel-accuracy, or confusion-matrix
benchmark. The synthetic examples are visual demonstrations, not additional
measurements of segmentation accuracy.


Dataset
-------
Name:
CelebAMask-HQ

Official project:
https://github.com/switchablenorms/CelebAMask-HQ

Original dataset publication:
Lee, C.-H., Liu, Z., Wu, L., & Luo, P. (2020). MaskGAN: Towards Diverse
and Interactive Facial Image Manipulation. Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), 5549-5558.
DOI: https://doi.org/10.1109/CVPR42600.2020.00559

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


Dataset Media Use and Distribution
----------------------------------
CelebAMask-HQ is treated as research-use benchmark imagery. Its original
images, semantic masks, and qualitative figures containing those images are
retained locally and are not included in the public GitHub repository or in
the thesis figures. Quantitative benchmark statistics, numerical tables, and
plots without restricted original imagery remain available for reporting,
with citation of Lee et al. (2020), as listed above.

The benchmark qualitative generation script and its numerical diagnostics
remain part of the reproducible research workflow, but the resulting image
files are local-only. Independently generated synthetic examples are stored
separately and may be used for public qualitative illustration subject to their
own documented media provenance. Media-source and distribution details are maintained in the project-level
VALIDATION_MEDIA_LICENSES.md.

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

The dataset is treated as read-only input. Validation outputs are never written
into the dataset directory.


Validation Files
----------------
The validation package is located in:

validation/face_regions/

Required files:

1. celebamaskhq_segmentation_eval.py

   Runs PhysioTrack FaceRegions on the complete 2,824-image evaluation split,
   constructs the ground-truth semantic maps, accumulates the dataset-level
   confusion matrix, computes all segmentation metrics, and writes the raw
   evaluation results.

2. celebamaskhq_segmentation_plot.py

   Reads the generated evaluation results and creates the final thesis tables,
   per-class IoU/Dice performance figure, and row-normalized confusion-matrix
   figure.

3. celebamaskhq_segmentation_qualitative.py

   Generates deterministic qualitative benchmark evidence from the accepted
   evaluation protocol and result configuration.

4. face_regions_component_test.py

   Runs the real FaceRegions component through PhysioTrack FaceAnalysis with
   only Face Regions enabled, using the same 2,824-image official test subset
   and the same controlled full aligned-image input box. It exports structured
   numerical component outputs without computing benchmark accuracy metrics.

5. list_eval_partition.txt

   Official CelebA train/validation/test partition metadata used with
   CelebA-HQ-to-CelebA-mapping.txt to identify the CelebAMask-HQ test subset.

The partition file is stored with the validation package so that the downloaded
CelebAMask-HQ directory can remain unchanged.

6. face_regions_overlap_regression_test.py

   Runs a deterministic multi-face check using overlapping face boxes in both
   processing orders. It verifies independent region-mask pixel counts and
   their association with the corresponding FaceAnalysis face instances.

7. face_regions_synthetic_demo.py

   Executes the genuine FaceAnalysis/FaceRegions/SegFace segmentation pathway
   on synthetic mannequin images independently of CelebAMask-HQ. It exports
   class-ID maps, colour visualizations, image overlays, a contact sheet,
   numerical observations, and machine-readable and readable summaries.


Path Handling
-------------
The project location is determined automatically from the location of the
validation scripts, and the dataset is expected at:

datasets/CelebAMask-HQ

Therefore, the validation can be reproduced on another machine without editing
machine-specific dataset paths as long as the documented project structure is
preserved.


Evaluated Component
-------------------
PhysioTrack FaceRegions

Segmentation backend:
SegFace

Inference device used for the reported validation:
CPU


Scientific Evaluation Protocol
------------------------------
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
confusion matrix.

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
dataset. Per-image IoU values are not averaged for the reported benchmark.


Safe Rerun and Output Ownership
-------------------------------
The final validation scripts use the following transactional rerun pattern:

preflight
-> temporary/staging generation
-> validation of newly generated outputs
-> replacement of script-owned final outputs

Previously valid evidence is not removed before newly generated replacement
artifacts pass the relevant validation checks.

Each script owns only its own outputs.

celebamaskhq_segmentation_eval.py owns:
- results/celebamaskhq_class_metrics.csv
- results/celebamaskhq_confusion_matrix.csv
- results/celebamaskhq_segmentation_summary.txt

celebamaskhq_segmentation_plot.py owns:
- results/celebamaskhq_thesis_table.csv
- results/celebamaskhq_thesis_table.md
- results/celebamaskhq_summary_table.csv
- results/celebamaskhq_summary_table.md
- results/figures/celebamaskhq_per_class_metrics.png
- results/figures/celebamaskhq_normalized_confusion_matrix.png

celebamaskhq_segmentation_qualitative.py owns:
- results/qualitative/
- results/figures/celebamaskhq_qualitative_examples.png

face_regions_component_test.py owns:
- results/component_execution/face_regions_component_results.csv
- results/component_execution/face_regions_component_summary.json

face_regions_synthetic_demo.py owns:
- results/synthetic_demo/ (the entire published demonstration directory)

If synthetic inference fails, results/synthetic_demo_last_attempt.json and
results/synthetic_demo_last_attempt.txt record the rejected attempt separately.
A successful new run replaces only the previous synthetic_demo directory after
verification; unsuccessful runs preserve its previously accepted contents.

No script writes generated results into datasets/CelebAMask-HQ.


Running the Scientific Evaluation
---------------------------------
Activate the PhysioTrack thesis environment and move to the validation
directory.

Example:

conda activate PhysioTrack-v2
cd /d "<repository-root>\validation\face_regions"

Run the complete scientific evaluation:

python celebamaskhq_segmentation_eval.py

The complete CPU evaluation can take several hours.

After the evaluator finishes successfully, generate the final tables and
quantitative figures:

python celebamaskhq_segmentation_plot.py

Then generate the qualitative benchmark evidence:

python celebamaskhq_segmentation_qualitative.py


Generated Scientific Results
----------------------------
The evaluator writes the following files under:

validation/face_regions/results/

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

results/figures/celebamaskhq_normalized_confusion_matrix.png
    Row-normalized 19 x 19 confusion-matrix visualization derived from
    celebamaskhq_confusion_matrix.csv. Each ground-truth class forms one row,
    and the pixel counts in each row are normalized to percentages.


Qualitative Benchmark Evidence
------------------------------
The qualitative analysis does not replace or modify the full quantitative
evaluation. It uses the same PhysioTrack FaceRegions component, SegFace
backend, 512 x 512 input size, 19-class label definition, CPU inference
configuration, and full aligned-image initialization used by the quantitative
evaluator.

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

The final selected cases are:

strong_candidate
- HQ image: 7159
- Foreground mIoU: 91.20%

representative_candidate
- HQ image: 6839
- Foreground mIoU: 80.59%

challenging_candidate
- HQ image: 14854
- Foreground mIoU: 52.63%

eye_glasses
- HQ image: 3720
- Foreground mIoU: 66.61%

hat
- HQ image: 15208
- Foreground mIoU: 67.27%

earring
- HQ image: 17175
- Foreground mIoU: 78.96%

necklace
- HQ image: 24531
- Foreground mIoU: 82.73%

high_semantic_diversity
- HQ image: 15008
- Foreground mIoU: 84.49%
- Ground-truth foreground classes: 17

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
for the reported benchmark metrics.

No face-detection rectangle is drawn because the documented protocol initializes
FaceRegions with a bounding box covering the entire aligned 512 x 512 image.

The benchmark qualitative outputs are retained locally as restricted
research evidence. They are not republished in the thesis or GitHub. Only
numerical diagnostics and permitted aggregate plots are used in publicly
reported evaluation results. The qualitative outputs are written under:

results/qualitative/

Expected qualitative outputs include:

results/qualitative/annotated_images/
    Eight individual qualitative evidence figures.

results/qualitative/celebamaskhq_qualitative_selection.csv
    Selection rationale, source image, image-level diagnostics, semantic
    content information, and generated output path for each selected case.

results/figures/celebamaskhq_qualitative_examples.png
    Combined 2 x 4 summary figure containing the eight selected prediction
    overlays.


Final Quantitative Results
--------------------------
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


Reported Scientific Rerun Runtime
---------------------------------
Elapsed time:
11,206.81 seconds

Throughput:
0.2520 images/second

Execution time is hardware- and environment-dependent and is not expected to
match across machines. The segmentation metrics are the reproducibility target.


Independent Scientific Consistency Verification
-----------------------------------------------
The final scientific evaluation outputs were independently checked after rerun.

Confirmed properties:

- Exactly 2,824 official test images were evaluated.
- All 2,824 images completed successfully.
- No inference failure was recorded.
- The accumulated confusion matrix is 19 x 19.
- No confusion-matrix count is negative.
- The accumulated confusion matrix contains exactly:

  2,824 x 512 x 512 = 740,294,656 pixels

- The sum of the confusion-matrix diagonal is 707,609,546 pixels.
- Pixel Accuracy recomputed directly from the confusion matrix is:

  0.95584851283865

- All-class mIoU recomputed directly from the confusion matrix is:

  0.8153278055221213

- Foreground mIoU recomputed directly from the confusion matrix is:

  0.8087183105433643

- All-class Mean Dice recomputed directly from the confusion matrix is:

  0.8935666031059025

- Foreground Mean Dice recomputed directly from the confusion matrix is:

  0.8895406623087769

- Ground-truth pixel counts in celebamaskhq_class_metrics.csv match the
  confusion-matrix row sums exactly.
- Predicted pixel counts in celebamaskhq_class_metrics.csv match the
  confusion-matrix column sums exactly.
- Per-class IoU and Dice values reproduce from the confusion matrix to
  floating-point precision.
- Summary CSV and Markdown values match the detailed evaluator results.
- Per-class thesis CSV and Markdown values match the detailed class metrics
  after the intended rounding.
- The per-class performance figure is consistent with the table values.
- The normalized confusion-matrix figure is consistent with the raw matrix.
- The weaker ear_r and neck_l values are genuine benchmark results, not
  arithmetic or file-integrity defects.


Interpretation
--------------
The FaceRegions component shows strong segmentation performance on the
CelebAMask-HQ test subset. Background, skin, hair, eye_g, and nose obtain among
the highest IoU values, while most major facial regions achieve IoU values near
or above 0.80.

The weakest classes are ear_r and neck_l. Their lower results are retained
explicitly and should be discussed as genuine limitations rather than hidden or
corrected.


Isolated PhysioTrack Face Regions Execution
-------------------------------------------
The isolated component test is run with:

python face_regions_component_test.py

The test uses:

- Pipeline: PhysioTrack FaceAnalysis
- Target component: FaceRegions
- Backend: SegFace
- Device: CPU
- Tracking: disabled
- Unrelated face-analysis components: disabled
- Dataset: same official 2,824-image CelebAMask-HQ test subset
- Input resolution: 512 x 512
- Controlled input box: full aligned image
- Ground-truth segmentation masks: not used
- Accuracy metrics: not computed

Official partition metadata and the HQ-to-CelebA mapping are used only to
select the same 2,824-image subset as the scientific benchmark.

The test captures the real FaceRegions output used by FaceAnalysis and verifies
that the native region-mask pixel counts match the region pixel counts exposed
through the integrated FaceAnalysis result.

The deterministic multi-face check can be run separately with:

python face_regions_overlap_regression_test.py

It exercises the Segmentation.Face, FaceRegions, and FaceAnalysis software paths
with controlled semantic labels rather than loaded model weights. Two overlapping
face boxes are evaluated in both processing orders, with independent class-mask
pixel counts and per-face association verified. This check is not a segmentation
accuracy benchmark.


Isolated Component Outputs
--------------------------
The test writes:

results/component_execution/face_regions_component_results.csv
    Structured numerical output for every observed region on every processed
    image.

results/component_execution/face_regions_component_summary.json
    Execution configuration, coverage counts, region statistics, runtime, and
    final PASS/FAIL status.


Final Isolated Component Results
--------------------------------
Expected images:
2,824

Processed images:
2,824

Images with regions:
2,824

Images without regions:
0

Execution-failed images:
0

Observed foreground region classes:
18

Region rows:
35,111

Result rows:
35,111

Total native foreground pixels:
527,728,682

Runtime:
239.41 minutes

Overall status:
PASS


Independent Isolated-Output Verification
----------------------------------------
The final isolated component outputs were independently audited.

Confirmed properties:

- Exactly 2,824 unique images are represented.
- Every image has status OK.
- No NO_REGIONS record exists.
- No EXECUTION_FAILED record exists.
- No duplicate result row exists.
- No duplicate (image_id, region_name) pair exists.
- Every processed image is 512 x 512.
- Every controlled FaceAnalysis input box is exactly:

  (0, 0, 512, 512)

- Every recorded FaceAnalysis-to-FaceRegions association IoU is 1.0.
- pipeline_regions_available is true for every region row.
- pixel_count_match is true for every region row.
- Native and FaceAnalysis pipeline pixel counts match exactly.
- No native region pixel count is negative.
- All 18 expected foreground semantic classes are observed.
- skin_fraction reproduces from the captured pixel counts and region area to
  floating-point precision.
- The staged output validation passed before final output replacement.


Cross-Check Between Scientific and Isolated Outputs
---------------------------------------------------
The isolated component produced:

527,728,682 total native foreground pixels.

The sum of predicted pixels across the 18 foreground classes in the accepted
scientific benchmark is also:

527,728,682 pixels.

This is an exact numerical match.

The scientific benchmark additionally contains:

212,565,974 predicted background pixels.

Therefore:

527,728,682 foreground pixels
+
212,565,974 background pixels
=
740,294,656 total predicted pixels

which equals exactly:

2,824 x 512 x 512.

This provides strong consistency evidence that the isolated execution exercises
the same FaceRegions segmentation behavior used by the scientific benchmark.

The interpretation remains separate:

- Scientific IoU, Dice, Pixel Accuracy, and confusion matrix = accuracy evidence.
- Isolated component CSV and PASS status = software execution and integration
  evidence.


Synthetic Qualitative Demonstration
-----------------------------------
Purpose and separation
~~~~~~~~~~~~~~~~~~~~~~
The independent synthetic demonstration illustrates the actual predictions
of PhysioTrack FaceRegions/SegFace on computer-generated mannequin images.
These examples do not contain CelebAMask-HQ images or annotations and do not
use ground-truth segmentation masks. They are not part of the official
2,824-image benchmark and do not calculate segmentation IoU, Dice, pixel
accuracy, or class-wise ground-truth accuracy. The quantitative benchmark
remains the scientific measurement of segmentation performance.

Synthetic source media
~~~~~~~~~~~~~~~~~~~~~~
The demonstration processes ten artificial, rendered, mannequin-style PNG
images prepared using OpenAI ChatGPT and supplied independently of the
benchmark. The image set covers frontal and nominally rotated views, changes in mouth
and eye appearance, eyeglasses, a cap, hair occlusion, hand occlusion, and
asymmetric illumination. A single synthetic subject is depicted in each image. Source-image pose and
lighting descriptions are nominal conditions, not calibrated ground truth.

The accepted source files are 01.png through 10.png in:

validation/synthetic_media/face_regions_media/

Each original synthetic input is 1254 x 1254 pixels. These are the dimensions
of the accepted supplied files, not a mandatory size for subsequent inputs.
The script recursively discovers supported input images; the input list is not
hard-coded. File content is identified by SHA-256 checksums in the summary.

Execution protocol
~~~~~~~~~~~~~~~~~~
Run from validation/face_regions with the project environment active:

python face_regions_synthetic_demo.py

The demonstration uses the real project FaceAnalysis -> FaceRegions -> SegFace
pipeline on CPU. Only the Face Regions analysis module is enabled. Images are
resized to 512 x 512 pixels using OpenCV INTER_LINEAR interpolation. A
controlled full-image face box (0, 0, 512, 512) initializes segmentation,
matching the aligned-image initialization used for the scientific benchmark.
The demonstration does not perform automatic face detection, even though the
full FaceAnalysis pipeline path is exercised. The input face box is provided
by a controlled detector returning precisely that rectangle.

The class-ID map is obtained from the original SegFace output without altering
class predictions. FaceRegions native masks and the structured FaceAnalysis
region pixel counts are checked against that map. The 19 labels comprise
background and 18 foreground semantic classes in the order documented under
Ground-Truth Construction; the same labels identify model predictions here.

The display-only colour mapping creates a 19-class segmentation image and a
translucent overlay on the resized input. It does not refine, reassign, or
correct the model predictions. The contact sheet contains all ten overlays.
The raw class-ID PNGs (values 0 through 18) can appear nearly black in ordinary
image viewers; they are machine-readable label maps rather than colour figures.

Validated synthetic results
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The completed demonstration recorded ten discovered and successfully processed
images, zero processing failures, 190 observations (one row for each of the 19
classes in each of the ten images), and status COMPLETE. Each prediction covers
512 x 512 = 262,144 pixels, including background. The CSV and summary report
exact pixel counts for every predicted semantic class.

Image     Scenario                         FG classes   FG pixels
--------- -------------------------------- ---------- ------------
01.png    Frontal face                             12       91,355
02.png    Oblique view, first direction            11      114,095
03.png    Oblique view, opposite direction          9      107,816
04.png    Open mouth                               13       94,333
05.png    Eyes closed                              10       93,722
06.png    Eyeglasses                                9       98,175
07.png    Cap                                      11       95,466
08.png    Hair obscuring one eye                   10      108,240
09.png    Hand partially obscuring face            12      127,483
10.png    Asymmetric side lighting                 11       91,372

The ten predictions together contain 1,022,057 foreground pixels and
1,599,383 background pixels, totaling 2,621,440 pixels. Sixteen distinct
foreground classes are represented by at least one predicted pixel across the
sample; neither ear_r (earring) nor neck_l (necklace) appears. This is a
property of this ten-image sample, not a claim about the 18-class capability
of the model or its official benchmark results.

Eyeglasses (06.png) are assigned to the eye_g category; the cap (07.png) to
hat; and the obscuring hair in 08.png to hair. In 04.png, the open mouth has a
predicted mouth region. In 05.png, no eye pixels are assigned to the eye
classes with the lids closed. Occlusion in 09.png and changes in lighting in
10.png provide additional qualitative behaviour examples. Small irregularities
can be observed at clothing boundaries, lips and occluded areas. These are
unaltered model predictions and cannot be converted into numerical accuracy
claims without independently annotated ground truth.

Every recorded association_iou value is 1.0 because the controlled full-image
region is associated with the same full-image FaceAnalysis instance. This
*association IoU* must not be interpreted as semantic-segmentation IoU or
pixel-level predictive accuracy.

Generated outputs and verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The demonstration owns only:

validation/face_regions/results/synthetic_demo/

The accepted run generated:

- annotated_images/ (ten three-panel images: synthetic input, semantic colour
  prediction, and prediction overlay with a class legend).
- label_maps/ (ten uint8 raw 512 x 512 class-ID PNG maps).
- segmentation_maps/ (ten colour-coded 512 x 512 prediction maps).
- overlays/ (ten 512 x 512 translucent prediction overlays).
- face_regions_synthetic_observations.csv (190 rows of per-image, per-class
  pixel counts, class IDs, source SHA-256, and native/pipeline count checks).
- synthetic_demo_examples.png (one contact sheet of ten overlays).
- synthetic_demo_summary.json (configuration, source hashes, output hashes,
  per-image class counts, input dimensions, software versions, and status).
- synthetic_demo_summary.txt (a complete human-readable projection of the
  same summary data).

The script validates the original input media, checks the predicted class-ID
range and every class's pixel accounting, and checks consistency between the
native FaceRegions masks and FaceAnalysis output. It writes into a temporary
staging directory, independently verifies the saved PNGs, hashes, CSV rows,
summary, and contact sheet, and then publishes the complete output directory.
A successful rerun replaces the entire previous synthetic_demo directory.
A failed or incomplete run does not publish partial output and preserves the
previously accepted directory. Rejected-run reports, when applicable, are
written outside the published directory as synthetic_demo_last_attempt.json
and synthetic_demo_last_attempt.txt under results/.

The accepted synthetic run was recorded on Windows with Python 3.11.17,
OpenCV 5.0.0 and NumPy 2.4.6 on CPU. The summaries contain implementation
module and script SHA-256 hashes and recorded dependency versions. The
pretrained SegFace weights are not hashed by this demonstration; reproducing
bitwise-identical predictions on a different system therefore cannot be
promised from the file provenance information alone.

Media provenance and distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The mannequin images were created as synthetic illustrative media using OpenAI
ChatGPT; they are not frames from CelebAMask-HQ or any restricted benchmark.
Their source files remain under validation/synthetic_media/face_regions_media/.

Synthetic input provenance and terms of use:
- Creation source: OpenAI ChatGPT (AI-generated illustrative PNG images).
- Input set: 01.png through 10.png, stored in the directory above.
- OpenAI Terms of Use: https://openai.com/policies/terms-of-use/
- These images are separate from CelebAMask-HQ benchmark media and do not
  contain official semantic ground-truth annotations.
- OpenAI-generated imagery is not attributed to the Blender Human Base Meshes
  asset and is not described as licensed under CC0.
The project-level VALIDATION_MEDIA_LICENSES.md is the designated register
for media source attribution and permitted use. The independent synthetic examples, not the restricted benchmark
qualitative frames, are the media intended for public GitHub and thesis
illustrations. The synthetic outputs are non-benchmark visual evidence and
should be labelled accordingly wherever displayed.

Evaluation Scope
----------------
The scientific evaluation measures semantic face-region segmentation under
aligned-image initialization.

It does not measure an end-to-end face-detection-plus-segmentation pipeline,
because the full aligned image is supplied directly as the segmentation region.

The isolated component execution also uses the controlled full aligned-image box
so that Face Detection does not contaminate the Face Regions execution check.

The reported scientific results should therefore be described as PhysioTrack
FaceRegions segmentation performance under the documented CelebAMask-HQ
aligned-image evaluation protocol.


Limitations
-----------
Important reporting limitations are:

- The benchmark uses aligned CelebAMask-HQ imagery.
- The full image is used as the segmentation region.
- Face detection performance is not evaluated here.
- The scientific benchmark uses dataset-level confusion-matrix aggregation.
- Per-image qualitative metrics are diagnostic only.
- Rare and small semantic regions can be substantially more difficult than
  dominant classes.
- ear_r and neck_l are the weakest benchmark classes.
- The isolated execution does not calculate segmentation accuracy.
- A PASS result from the isolated execution must not be presented as an
  additional accuracy score.
- CelebAMask-HQ performance is not evidence of universal real-world
  segmentation performance.


Reproducibility
---------------
To reproduce the complete validation on another machine:

1. Obtain and extract the CelebAMask-HQ dataset.
2. Place the extracted dataset at datasets/CelebAMask-HQ.
3. Keep list_eval_partition.txt in the face_regions validation directory.
4. Install the PhysioTrack project and its required dependencies.
5. Run celebamaskhq_segmentation_eval.py.
6. Run celebamaskhq_segmentation_plot.py.
7. Run celebamaskhq_segmentation_qualitative.py.
8. Run face_regions_component_test.py.
9. Compare the generated quantitative files with the reported scientific
   metrics.
10. Inspect the generated quantitative and qualitative figures.
11. Verify the isolated component outputs and confirm zero execution failures.
12. Run face_regions_overlap_regression_test.py to verify deterministic
    multi-face region masks and associations.
13. Independently run face_regions_synthetic_demo.py on the synthetic media
    folder and verify the generated class-ID maps, colour outputs, CSV, contact
    sheet, and summary files. This run does not require CelebAMask-HQ images.
14. Retain benchmark-derived qualitative images locally; publish synthetic
    illustrations separately, according to VALIDATION_MEDIA_LICENSES.md.


Final Files to Preserve
-----------------------
Validation scripts:
- validation/face_regions/celebamaskhq_segmentation_eval.py
- validation/face_regions/celebamaskhq_segmentation_plot.py
- validation/face_regions/celebamaskhq_segmentation_qualitative.py
- validation/face_regions/face_regions_component_test.py
- validation/face_regions/face_regions_overlap_regression_test.py
- validation/face_regions/face_regions_synthetic_demo.py

Static validation metadata:
- validation/face_regions/list_eval_partition.txt

README:
- validation/face_regions/README_CELEBAMASKHQ_FACE_REGIONS.txt

Scientific benchmark outputs:
- validation/face_regions/results/celebamaskhq_class_metrics.csv
- validation/face_regions/results/celebamaskhq_confusion_matrix.csv
- validation/face_regions/results/celebamaskhq_segmentation_summary.txt
- validation/face_regions/results/celebamaskhq_thesis_table.csv
- validation/face_regions/results/celebamaskhq_thesis_table.md
- validation/face_regions/results/celebamaskhq_summary_table.csv
- validation/face_regions/results/celebamaskhq_summary_table.md
- validation/face_regions/results/figures/celebamaskhq_per_class_metrics.png
- validation/face_regions/results/figures/celebamaskhq_normalized_confusion_matrix.png

Benchmark qualitative evidence (local-only; do not include original imagery
or annotated benchmark-image figures in public GitHub or thesis):
- validation/face_regions/results/qualitative/annotated_images/
- validation/face_regions/results/qualitative/celebamaskhq_qualitative_selection.csv
- validation/face_regions/results/figures/celebamaskhq_qualitative_examples.png

Isolated component evidence:
- validation/face_regions/results/component_execution/face_regions_component_results.csv
- validation/face_regions/results/component_execution/face_regions_component_summary.json

Independent synthetic media and demonstration:
- validation/synthetic_media/face_regions_media/ (ten original synthetic PNGs)
- validation/face_regions/results/synthetic_demo/annotated_images/
- validation/face_regions/results/synthetic_demo/label_maps/
- validation/face_regions/results/synthetic_demo/segmentation_maps/
- validation/face_regions/results/synthetic_demo/overlays/
- validation/face_regions/results/synthetic_demo/face_regions_synthetic_observations.csv
- validation/face_regions/results/synthetic_demo/synthetic_demo_examples.png
- validation/face_regions/results/synthetic_demo/synthetic_demo_summary.json
- validation/face_regions/results/synthetic_demo/synthetic_demo_summary.txt

Media provenance:
- VALIDATION_MEDIA_LICENSES.md


Final Validation Status
-----------------------
Scientific benchmark:
ACCEPTED

Quantitative tables and figures:
ACCEPTED

Qualitative benchmark evidence (local-only):
ACCEPTED

Safe-rerun behavior:
VERIFIED

Isolated PhysioTrack Face Regions component execution:
PASS / ACCEPTED

Deterministic multi-face region verification:
PASS

Independent synthetic qualitative demonstration:
COMPLETE / ACCEPTED (10 images; zero failures; 190 class observations)


Downloading and Preparing Benchmark Media
-----------------------------------------
CelebAMask-HQ source images, segmentation masks, and derived
annotated benchmark images are not distributed as public GitHub
media.

Obtain the dataset according to its original usage conditions:
https://github.com/switchablenorms/CelebAMask-HQ

Place the extracted files in the workspace's
datasets/CelebAMask-HQ/ directory, alongside physiotrack-v2:

- CelebA-HQ-img/
- CelebAMask-HQ-mask-anno/
- CelebA-HQ-to-CelebA-mapping.txt

The validation also requires list_eval_partition.txt in the
validation package to select the documented 2,824-image test subset.

After preparing the dataset, run the validation scripts as described
above. Benchmark-derived qualitative images must remain local.

Numerical evaluation results, independent plots, and synthetic
demonstration media are separate from the restricted benchmark images.

See VALIDATION_MEDIA_LICENSES.md for distribution conditions.
