PhysioTrack Integration Validation
==================================

Purpose
-------
This README documents the reproducible integration validation of the complete
PhysioTrack face-analysis pipeline.

The component-level benchmark validations evaluate individual modules under
task-specific protocols. Integration validation has a different purpose: it
verifies that the configured components can operate together inside the
PhysioTrack FaceAnalysis pipeline without breaking feature configuration,
per-face association, temporal processing, tracking, static-image processing,
or native result export.

The integration validation therefore addresses system-level questions that
cannot be established from isolated benchmark results alone:

1. Whether optional facial-analysis components can be enabled and disabled
   without unintended cross-module behavior.
2. Whether the existing landmark-based gaze descriptor and the learned
   GazeEstimator can operate independently and simultaneously.
3. Whether the complete face-analysis pipeline produces internally consistent
   per-frame and temporal outputs.
4. Whether gaze-estimation results remain associated with the correct tracked
   face in multi-person video.
5. Whether the framework's native frame and window exports preserve the
   expected analysis structure and numerical measurements, including FaceQuality
   confidence, brightness, sharpness, face-area ratio, and Emotion outputs where
   applicable.
6. Whether Emotion is propagated as a real per-face numerical output, including
   an eight-class score vector, predicted label, and confidence, rather than only
   as an availability flag.
7. Whether the complete configured pipeline can process an entire test video
   without dropped frames, missing face records, broken tracking, or missing
   enabled modules.
8. Whether static images can be processed without fabricating outputs that
   require a temporal sequence.
9. Whether batch-oriented integration scripts preserve media-level failure
   accounting and continue with remaining media after a per-media processing
   failure, provided all discovered media passed initial preflight.

This validation is a system-integration and regression test. It is not an
additional accuracy benchmark for the individual modules. Accuracy of the
individual components is evaluated separately using their corresponding
benchmark datasets and validation protocols.

Validation Location
-------------------
The integration package is located in:

physiotrack-v2/validation/integration/

The final directory structure is:

validation/integration/
|-- face_pipeline_runtime_smoke_test.py
|-- face_pipeline_e2e_test.py
|-- face_pipeline_native_export_test.py
|-- face_pipeline_multi_person_e2e_test.py
|-- whole_project_video_analysis.py
|-- whole_project_final_e2e_test.py
|-- face_pipeline_image_e2e_test.py
|-- face_pipeline_runtime_benchmark.py
|-- README_INTEGRATION.txt
|-- test_data/
|   |-- single_person/
|   |   `-- ravdess_single_person.mp4
|   |-- multi_person/
|   |   `-- multi_person2.mp4
|   |-- whole_project/
|   |   `-- ravdess_whole_project.mp4
|   `-- images/
|       `-- <integration image fixtures>
`-- results/
    |-- runtime_smoke/
    |-- face_pipeline_e2e/
    |-- native_export/
    |-- multi_person_e2e/
    |-- whole_project_video_analysis/
    |-- whole_project_e2e/
    |-- image_e2e/
    `-- runtime_performance/

Test Data
---------
Integration fixtures are stored under:

physiotrack-v2/validation/integration/test_data/

The current video fixtures are:

single_person/ravdess_single_person.mp4
    Single-person video used for runtime configuration checks, the
    single-person end-to-end comparison, and native export validation.
    Source: Pexels (Pexels License).

multi_person/multi_person2.mp4
    Multi-person video used to verify simultaneous face tracking, per-person
    feature generation, and gaze-estimation association.
    Source: Pexels (Pexels License).

whole_project/ravdess_whole_project.mp4
    Full-pipeline video used for whole-project analysis and the final complete
    end-to-end integration test.
    Source: Pexels (Pexels License).

The images/ directory contains static-image fixtures covering frontal faces,
eye states, mouth configurations, gaze directions, facial expressions,
multi-face scenes, and pose variation.

Image-fixture provenance is documented as follows:

- eyes_1.jpg, eyes_2.jpg, mouth_1.jpg, mouth_2.jpg, multi_1.jpg, multi_2.jpg,
  pose_1.jpg, frontal_1.jpg, frontal_2.jpg, and the seven emotion_*.jpg
  expression fixtures: Pexels (Pexels License).
- gaze_center.png, gaze_up.png, gaze_down.png, gaze_left.png, and
  gaze_right.png: AI-generated controlled images of a synthetic subject.
  These five images provide the required gaze-direction variations.

Dataset Sources and Citations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pexels
    License: Pexels License.
    License reference: https://www.pexels.com/license/
    The following integration fixtures are sourced from Pexels; each entry
    records the media creator and its original source URL.

    Video fixtures:
    multi_person2.mp4 - Tima Miroshnichenko
        https://www.pexels.com/video/women-wearing-suit-while-sitting-together-7203425/
    ravdess_single_person.mp4 - KoolShooters
        https://www.pexels.com/video/man-blinking-at-the-camera-6620642/
    ravdess_whole_project.mp4 - Tima Miroshnichenko
        https://www.pexels.com/video/woman-recording-a-video-4962731/

    Static-image fixtures:
    emotion_angry.jpg - Anna Tarazevich
        https://www.pexels.com/photo/an-angry-woman-with-a-black-top-8066653/
    emotion_disgust.jpg - Pavel Danilyuk
        https://www.pexels.com/photo/a-woman-in-red-long-sleeve-shirt-8638020/
    emotion_fear.jpg - Engin Akyurt
        https://www.pexels.com/photo/close-up-photo-of-shocked-woman-6605132/
    emotion_happy.jpg - Tima Miroshnichenko
        https://www.pexels.com/photo/smiling-elderly-woman-in-white-t-shirt-8727573/
    emotion_neutral.jpg - Liam Anderson
        https://www.pexels.com/photo/shallow-focus-photography-of-woman-1458332/
    emotion_sad.jpg - Alena Darmel
        https://www.pexels.com/photo/close-up-photo-of-a-woman-crying-6643007/
    emotion_surprise.jpg - Anna Tarazevich
        https://www.pexels.com/photo/close-up-photo-of-woman-s-facial-expression-8066636/
    eyes_1.jpg - BYB BYB
        https://www.pexels.com/photo/portrait-of-a-young-woman-smiling-with-her-eyes-shut-19999458/
    eyes_2.jpg - Andrea Piacquadio
        https://www.pexels.com/photo/close-up-photo-of-shocked-woman-with-brown-curly-hair-3764398/
    mouth_1.jpg - Andrea Piacquadio
        https://www.pexels.com/photo/close-up-photo-of-shocked-woman-with-brown-curly-hair-3764392/
    mouth_2.jpg - Pavel Danilyuk
        https://www.pexels.com/photo/a-woman-in-red-long-sleeve-shirt-8637956/
    frontal_1.jpg - Christoph Sixt
        https://www.pexels.com/photo/professional-portrait-of-a-smiling-man-in-business-attire-29856511/
    frontal_2.jpg - Andrea Piacquadio
        https://www.pexels.com/photo/man-in-white-polo-shirt-smiling-3785110/
    pose_1.jpg - Laura Beauty
        https://www.pexels.com/photo/ensaio-masculino-28293335/
    multi_1.jpg - Multi Motion Film Works
        https://www.pexels.com/photo/smiling-group-in-white-shirts-17938967/
    multi_2.jpg - SHVETS production
        https://www.pexels.com/photo/close-up-shot-of-two-men-9775652/

AI-generated gaze-direction fixtures
    gaze_center.png, gaze_up.png, gaze_down.png, gaze_left.png, and
    gaze_right.png were generated for controlled integration testing.
    They do not originate from an external photographic dataset and are
    not used as benchmark-accuracy evidence.
    Source: OpenAI / ChatGPT image generation.
    Source media directory: validation/integration/test_data/images/
    Applicable platform terms: https://openai.com/policies/terms-of-use/
    The five generated images are separate from the Pexels photographic
    fixtures and are not derived from a Blender CC0 model.
    Project-level media provenance and distribution documentation:
    VALIDATION_MEDIA_LICENSES.md.

The Pexels media and AI-generated gaze-direction images are integration
fixtures only. Fixture filenames are identifiers and do not establish
provenance or dataset membership.

These media files are integration fixtures rather than benchmark datasets.
They are not used to estimate model accuracy and do not replace the external
benchmark datasets used by the individual validation components.

Any media distributed with the repository must have redistribution terms that
permit inclusion in the repository. If a fixture cannot legally be
redistributed, it must not be included in a public release.

Supported Media Discovery
-------------------------
The integration scripts discover supported media from their corresponding
repository-relative test-data directories rather than depending on one
hard-coded media filename.

Supported video extensions are:

.avi
.m4v
.mkv
.mov
.mp4
.webm

Supported image extensions are:

.bmp
.jpeg
.jpg
.png
.tif
.tiff
.webp

This design allows integration fixtures to be organized, renamed, or extended
without changing the script source code, provided the files remain in the
expected test-data directory and use a supported extension.

Path Handling
-------------
The integration scripts resolve their location from __file__ and locate test
media relative to:

validation/integration/test_data/

All generated integration results are organized under:

validation/integration/results/

Each script owns a dedicated final result subdirectory. New integration
outputs are generated in a temporary staging directory under results/ and are
validated before promotion. Previously accepted outputs are not removed before
the replacement run has completed successfully and its staged files have passed
validation. A script must not delete another integration test's results, modify
the test media, modify benchmark datasets, or modify project source code.

Integration Scope
-----------------
The integration suite exercises the configured PhysioTrack FaceAnalysis
pipeline and its connected outputs, including:

- Face detection
- Face tracking
- Facial landmarks
- Face quality
- Head pose
- Eye openness
- Blink analysis
- Landmark-based gaze descriptor
- Learned gaze estimation
- Mouth measurements
- Mouth motion
- Emotion Recognition label, confidence, and eight-class probability scores
- Face-region segmentation
- Temporal summaries
- Per-frame native export
- Per-window native export
- Static-image analysis
- Multi-person association
- Batch media processing and failure accounting

The suite verifies presence, configuration behavior, association, accounting,
numerical export, failure handling, and export consistency. It does not
reinterpret integration availability as a new quantitative accuracy score for
these modules.

Why Multiple Integration Tests Are Required
-------------------------------------------
A single end-to-end run is insufficient to validate all integration contracts.
The suite separates distinct system-level properties so that a failure can be
localized and interpreted correctly.

1. face_pipeline_runtime_smoke_test.py

   Purpose:
   Verifies runtime feature configuration and coexistence of the two gaze
   mechanisms.

   The test runs six configurations:

   - both_disabled
   - old_gaze_only
   - gaze_estimation_only
   - both_enabled
   - mouth_without_motion
   - mouth_and_motion_disabled

   It also verifies the dependency contract:

   - mouth_motion_requires_mouth

   This establishes that the existing landmark-based gaze descriptor and the
   learned GazeEstimator are independent optional features. Enabling one must
   not implicitly enable the other, and both must be able to operate
   simultaneously.

   The mouth-related cases verify that MouthOpenness can operate without
   MouthMotion, that both can be disabled together, and that MouthMotion cannot
   be enabled when MouthOpenness is disabled.

   The smoke test uses five frames per configuration. It is deliberately short
   because the objective is configuration-contract verification rather than
   full-video performance measurement.

2. face_pipeline_e2e_test.py

   Purpose:
   Verifies single-person end-to-end integration and checks whether enabling
   the learned GazeEstimator changes the availability of unrelated pipeline
   components.

   Each discovered video is processed in two conditions:

   - gaze estimation disabled
   - gaze estimation enabled

   The test checks complete frame processing, detected-face accounting, module
   availability, numerical frame export, and the expected absence/presence of
   gaze_estimation.

   This verifies that enabling the learned
   gaze-estimation module does not disable or remove the other configured
   analysis outputs.

3. face_pipeline_native_export_test.py

   Purpose:
   Verifies the framework's native per-frame and temporal/window export
   contracts when the complete face-analysis configuration is active.

   The test checks:

   - Complete video processing
   - Positive face detection count
   - Frame-record and window-record counts
   - Presence of the existing gaze key
   - Presence of the learned gaze_estimation key
   - Observation of both gaze mechanisms
   - Finite Eye Openness values in available eye records
   - Finite Mouth Openness values in available mouth records
   - Finite FaceQuality confidence, brightness, sharpness, and face-area-ratio
     values in available quality records
   - Preservation of FaceQuality numerical values in native frame export
   - Finite MouthMotion movement and velocity values when available
   - Numerical consistency of MouthMotion movement and velocity with their
     frame-to-frame definitions and video timing
   - Finite eye and mouth temporal/window summaries
   - Use of the validated blink configuration
   - Expected temporal/window structure
   - Agreement between detected-face counts and exported record counts
   - Preservation of numerical frame- and window-level analysis values

   This test is necessary because successful in-memory inference alone does not
   demonstrate that downstream users can retrieve the same information through
   PhysioTrack's exported analysis records.

4. face_pipeline_multi_person_e2e_test.py

   Purpose:
   Verifies multi-person tracking and per-face feature association.

   The test checks:

   - Presence of at least two tracked identities
   - Presence of frames containing multiple faces
   - No duplicate track ID assigned to two faces within the same frame
   - Gaze-estimation output observed for at least two tracked persons
   - Consistency between face instances and frame/window exports
   - Availability of the configured analysis components
   - Per-person availability and numerical export of FaceQuality descriptors
   - Per-person MouthMotion initialization and subsequent temporal updates
   - Numerical MouthMotion movement/velocity consistency within each tracked
     identity using the video's measured FPS
   - Per-person Blink Rate consistency against elapsed video frames,
     including time gaps when a tracked identity is temporarily absent;
     the per-video JSON summary records the count checked and mismatch count
   - Explicit recording of gaze-estimation failures
   - Persistent per-video PASS/FAIL summaries, including media that fail before
     frame-level records can be produced

   This test is particularly important for gaze estimation because a valid gaze
   vector is not sufficient if it is assigned to the wrong person. The
   multi-person test therefore evaluates association behavior in addition to
   feature availability.

5. whole_project_video_analysis.py

   Purpose:
   Generates a detailed complete-pipeline analysis and export over each
   discovered full-project test video.

   This script is an analysis/export integrity run rather than an independent
   benchmark. It checks full frame processing, face-record accounting, and
   agreement between frame and window export counts before preserving the
   detailed outputs.

   The resulting records provide auditable evidence that the complete
   configured pipeline produces coherent frame-level and temporal data over the
   entire video.

6. whole_project_final_e2e_test.py

   Purpose:
   Performs the final full-pipeline end-to-end integration check.

   The test verifies:

   - Complete frame processing
   - Presence of faces
   - Tracking observation
   - Record-count consistency
   - Observation of every configured module
   - Use of the validated blink configuration
   - Finite Eye Openness values for all available eye samples
   - Finite Mouth Openness values for all available mouth samples
   - Numerical FaceQuality outputs for all available quality samples
   - Finite MouthMotion movement and velocity values
   - Numerical MouthMotion consistency with frame-to-frame mouth-openness
     change and the video's measured FPS
   - Numerical preservation of the integrated measurements and descriptors
   - Overall per-video and aggregate integration PASS/FAIL status

   This is the highest-level video system test in the integration package. It
   complements, rather than replaces, the more focused runtime, export, and
   multi-person tests.

7. face_pipeline_image_e2e_test.py

   Purpose:
   Verifies complete static-image integration across all discovered supported
   image fixtures.

   The test checks:

   - Image loading and face detection
   - Multi-face image handling
   - Static-module availability
   - Finite Eye Openness and Mouth Openness values
   - Numerical FaceQuality export, including confidence, brightness, sharpness,
     and face-area ratio
   - Numerical export of static facial measurements and descriptors
   - Explicit NOT_APPLICABLE handling for tracking, blink, mouth motion, and
     temporal aggregation
   - Absence of fabricated temporal outputs, including no synthetic person ID,
     blink event, blink duration/rate, mouth movement/velocity, or temporal
     summary
   - Per-image PASS/FAIL accounting while continuing after individual image
     failures

   This test establishes that static-image processing uses the same integrated
   facial-analysis components where scientifically applicable while preserving
   the distinction between static and temporal analysis.

Safe Rerun and Transactional Output Replacement
-----------------------------------------------
All eight integration scripts protect previously accepted results through
validated staged outputs and rollback-aware final-directory promotion. Their
staging timing differs: the seven functional scripts create their staging
directories before pipeline execution, whereas the runtime-performance script
measures the video runs first and then stages its result files.

1. Discover the supported media and complete input preflight before processing.
   An unreadable media item fails preflight rather than becoming a per-media
   processing-failure record.
2. Verify that the integration results root is usable.
3. Use a script-owned temporary staging directory: before execution for the
   seven functional scripts and after measurement for the runtime benchmark.
4. Attempt the discovered, preflight-approved media set; a per-media processing
   failure is recorded without preventing subsequent items from being attempted.
5. Write all script-owned outputs to staging rather than to the accepted final
   result directory.
6. Validate that every expected staged JSON/CSV artifact exists, is non-empty,
   and can be parsed or read with a valid CSV header.
7. Promote the staged directory only after the complete integration test has
   passed.
8. Replace the previous script-owned final directory transactionally, with a
   temporary rollback copy retained until promotion succeeds.
9. Restore the previous accepted directory if final promotion fails.
10. Remove temporary staging/rollback directories after completion.

A failed rerun therefore cannot destroy the previously accepted integration
evidence. Failed staged runs are not promoted as final evidence.

Test Design Principles
----------------------
The integration tests follow the following principles:

1. No new model training or fine-tuning is performed.

2. Integration tests do not reuse component benchmark datasets to create new
   accuracy claims.

3. Module availability is distinguished from module accuracy. A module can be
   correctly integrated even though its benchmark accuracy is evaluated
   separately.

4. No arbitrary minimum accuracy threshold is introduced into the integration
   tests.

5. Tests named as tests contain explicit failure conditions rather than relying
   only on console output.

6. Complete-video tests verify frame-count and record-count consistency.

7. Multi-person behavior is checked through tracked identities and per-face
   association rather than assuming that a valid output is automatically
   attached to the correct person.

8. Static-image validation does not synthesize tracking, blink, mouth-motion,
   or temporal outputs when a temporal sequence is unavailable.

9. Generated outputs are isolated by test so that rerunning one integration
   test does not delete evidence from another test.

10. Batch-oriented scripts attempt all discovered supported media after
    preflight succeeds for the complete input set. Failure during an individual
    media processing run is recorded and does not prevent subsequent media
    from being attempted. Aggregate status may therefore be FAIL even when
    some media complete successfully.

11. Media-level summaries preserve processing failures even when no frame-level
    records can be generated. Input-opening or decoding errors caught during
    preflight fail the run before a new staged media-level summary is generated.

12. Integration exports preserve real numerical measurements, descriptors, and
    temporal summaries where available. Boolean availability fields are
    auxiliary status indicators rather than substitutes for numerical output.

13. MouthMotion is validated as a temporal numerical output. For video,
    movement and velocity are checked from sequential mouth-openness values
    using the measured video timing and maintained separately for each tracked
    person. The initial temporal sample is expected to represent initialization
    rather than fabricated motion.

14. Emotion output is treated here as an integrated and operational pipeline
    output. Integration availability is not a scientific benchmark validation
    of emotion-recognition accuracy.

15. FaceQuality is treated as a numerical integrated component. Its confidence,
    brightness, sharpness, and face-area-ratio outputs are preserved in the
    integration exports. Scientific validation of these descriptors is handled
    separately by the FaceQuality controlled correctness and sensitivity
    validation rather than by interpreting integration availability as accuracy.

Running the Integration Validation
----------------------------------
Activate the PhysioTrack thesis environment:

conda activate PhysioTrack-v2

Change to the integration validation directory:

cd /d <project-path>\physiotrack-v2\validation\integration

Run the integration scripts in the following order:

1. Runtime configuration smoke test

python face_pipeline_runtime_smoke_test.py

2. Single-person end-to-end integration test

python face_pipeline_e2e_test.py

3. Native export integration test

python face_pipeline_native_export_test.py

4. Multi-person end-to-end integration test

python face_pipeline_multi_person_e2e_test.py

5. Complete video analysis/export run

python whole_project_video_analysis.py

6. Final complete video end-to-end integration test

python whole_project_final_e2e_test.py

7. Static-image end-to-end integration test

python face_pipeline_image_e2e_test.py

Each script performs input preflight first, creates a script-specific staging
directory before expensive processing begins, writes the complete replacement
run to staging, validates the staged files, and only then replaces the final
result directory that it owns. If execution or validation fails, the previously
accepted final outputs are preserved.

The eighth test, the complementary full runtime-performance benchmark,
is run with:

python face_pipeline_runtime_benchmark.py

Its outputs are stored under results/runtime_performance/ and are reported
separately from the seven functional integration PASS/FAIL tests because runtime
performance is execution evidence rather than predictive-accuracy evidence.

Reported Integration Results
----------------------------
Runtime configuration smoke test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
ravdess_single_person.mp4

Each configuration processed 5 frames and 5 detected faces.

Cases:

both_disabled
    Landmark-based gaze: disabled
    Learned gaze estimation: disabled
    Status: PASS

old_gaze_only
    Landmark-based gaze: enabled
    Learned gaze estimation: disabled
    Status: PASS

gaze_estimation_only
    Landmark-based gaze: disabled
    Learned gaze estimation: enabled
    Status: PASS

both_enabled
    Landmark-based gaze: enabled
    Learned gaze estimation: enabled
    Status: PASS

mouth_without_motion
    MouthOpenness: enabled
    MouthMotion: disabled
    Status: PASS

mouth_and_motion_disabled
    MouthOpenness: disabled
    MouthMotion: disabled
    Status: PASS

Dependency contract:
mouth_motion_requires_mouth
    Enabling MouthMotion while MouthOpenness is disabled is rejected as
    required.
    Status: PASS

Overall runtime smoke status:
PASS

This confirms independent configuration and simultaneous operation of the two
gaze mechanisms, correct MouthOpenness/MouthMotion feature configuration, and
the required MouthMotion dependency on MouthOpenness.

Single-person end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
ravdess_single_person.mp4

Video FPS:
25.0

Video frames:
429

Gaze-estimation-disabled run:
Processed frames: 429
Total face samples: 429
gaze_estimation: ABSENT

Gaze-estimation-enabled run:
Processed frames: 429
Total face samples: 429
gaze_estimation: PASS, 429/429

Combined frame-result table:
858 rows x 156 columns

Module-summary table:
14 rows x 8 columns

The remaining configured modules passed in both runs. FaceQuality and Emotion
were available for all 858 exported face records across the two runs, with
their numerical outputs preserved in the frame table.

MouthMotion produced 429 finite samples in each condition, including one
initialization sample and 428 nonzero temporal updates. The maximum observed
mouth movement was 0.007374363792506053 and the maximum observed mouth velocity
was 0.18435909481265134.

Overall status:
PASS

The single-person end-to-end test completed successfully in both
configurations, with the expected absence or presence of learned gaze
estimation and no loss of the remaining configured module outputs.

Native export integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
ravdess_single_person.mp4

Video FPS:
25.0

Video frames:
429

Processed frames:
429

Detected faces:
429

Frame records:
429

Frame export shape:
429 rows x 124 columns

Window records:
429

Window export shape:
429 rows x 52 columns

Existing landmark-based gaze available:
429

Learned gaze estimation available:
429

FaceQuality available:
429

Blink available:
429

Blink events:
2

Mouth motion available:
429

Temporal output available:
429

Frame records contain the landmark-based gaze key:
True

Frame records contain the learned gaze_estimation key:
True

Frame Eye Openness values are finite:
True

Frame Mouth Openness values are finite:
True

Frame MouthMotion movement/velocity numerical checks:
PASS

Frame Emotion output:
Available for all applicable records with numerical class scores, predicted
label, and confidence preserved consistently.

Window Eye Openness summaries are finite:
True

Window Mouth Openness summaries are finite:
True

Validated blink configuration:
threshold = 0.22
min_closed_frames = 3

Window records have the expected structure:
True

Export record counts match detected faces:
True

Overall status:
PASS

Multi-person end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
multi_person2.mp4

Video FPS:
25.0

Reported video frames:
344

Processed frames:
344

Total face instances:
688

Tracked person IDs:
1, 2

Person 1 frame count:
344

Person 2 frame count:
344

Frames containing multiple faces:
344

Frame records:
688

Window records:
688

Frame export shape:
688 rows x 126 columns

Window export shape:
688 rows x 52 columns

Gaze-estimation unavailable records:
0 total
Person 1: 0
Person 2: 0

Learned gaze-estimation availability:
Person 1: 344/344
Person 2: 344/344

FaceQuality availability:
Person 1: 344/344
Person 2: 344/344

Emotion availability:
Person 1: 344/344
Person 2: 344/344

Landmark-dependent outputs were available on 344/344 person-frames for both
persons. This includes landmarks, eyes, blink, landmark-based gaze, mouth, and
MouthMotion.

MouthMotion initialization:
Person 1: movement=0, velocity=0 on the first sample
Person 2: movement=0, velocity=0 on the first sample

MouthMotion numerical samples:
Person 1: 344
Person 2: 344

MouthMotion nonzero samples:
Person 1: 343
Person 2: 343

MouthMotion movement/velocity consistency:
PASS at 25.0 FPS with per-person temporal state preserved independently

The test retained two stable tracked identities across all 344 video frames.
Per-person availability is reported directly rather than being interpreted as
predictive accuracy. The complete multi-person integration test passed its
association, accounting, numerical-output, and temporal-consistency criteria.

Overall status:
PASS

Whole-project video analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
ravdess_whole_project.mp4

Resolution:
2560 x 1440

FPS:
25.0

Reported video frames:
568

Processed frames:
568

Detected face records:
568

Frame records:
568

Frame export shape:
568 rows x 126 columns

Window records:
568

Window export shape:
568 rows x 52 columns

Tracked identity:
1

MouthMotion available and finite:
568/568

Non-initial MouthMotion samples:
567/568

MouthMotion movement/velocity consistency:
PASS

Frame-count consistency:
PASS

Export-count consistency:
PASS

The complete configured pipeline produced the expected frame-level and
temporal records across the entire video, including real numerical facial
measurements, learned gaze-estimation outputs, Emotion outputs, FaceQuality
descriptors, and temporal summaries.

Overall analysis status:
PASS

Final whole-project end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
ravdess_whole_project.mp4

Resolution:
2560 x 1440

FPS:
25.0

Reported video frames:
568

Processed frames:
568

Frames with faces:
568

Frames without faces:
0

Frames with one face:
568

Frames with multiple faces:
0

Total face samples:
568

Unique track IDs:
1

The following modules produced successful output for 568/568 face samples:

- detection
- tracking
- landmarks
- quality
- head_pose
- eyes
- blink
- gaze
- gaze_estimation
- mouth
- mouth_motion
- emotion
- regions
- temporal

Module-summary table:
14 rows x 8 columns

Final frame export:
568 rows x 157 columns

Frame count matches video:
True

Tracking observed:
True

Record count matches:
True

All configured modules observed:
True

Validated blink configuration:
threshold = 0.22
min_closed_frames = 3
Configuration match: True

Finite Eye Openness samples:
568/568 available eye samples

Finite Mouth Openness samples:
568/568 available mouth samples

FaceQuality samples:
568/568 face samples with numerical confidence, brightness, sharpness, and
face-area-ratio outputs

Finite MouthMotion samples:
568/568 face samples

Non-initial MouthMotion samples:
567/568

MouthMotion movement/velocity consistency:
PASS using the measured 25.0 FPS

Emotion output:
Available for all 568 face samples with numerical class scores, predicted label,
and confidence preserved consistently.

Overall status:
PASS

Static-image end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Input images:
21

Passed images:
21

Failed images:
0

Detected faces:
25

Exported face records:
25

Frame export shape:
25 rows x 72 columns

Static modules evaluated:

- detection
- landmarks
- quality
- head_pose
- eyes
- gaze
- gaze_estimation
- mouth
- emotion
- regions

The following modules are explicitly NOT_APPLICABLE for independent static
images because they require temporal sequence input:

- tracking
- blink
- mouth_motion
- temporal

All 25 detected face records contained the expected applicable static outputs.
FaceQuality was available for 25/25 detected faces and exported numerical
confidence, brightness, sharpness, and face-area-ratio values for every face.
Emotion was also available for 25/25 detected faces with an eight-class score
vector, predicted label, and confidence for every face.

Temporal-only outputs were absent or explicitly neutral as required. No
temporal quantity was fabricated from an independent static image.

Overall status:
PASS

Integration Acceptance Summary
--------------------------
The seven functional integration tests and the runtime benchmark passed on the
documented fixtures.
The accepted run confirms that the currently configured components are
operational together across runtime-configuration, single-person, native-export,
multi-person, full-video, final whole-project, and static-image execution paths.

FaceQuality produced numerical outputs in all applicable records: 858/858
records in the two-condition single-person test, 429/429 records in native export,
688/688 multi-person face instances, 568/568 whole-project video face records,
568/568 final whole-project face samples, and 25/25 static-image face records.

Emotion produced numerical per-face outputs in all applicable integration paths,
including predicted labels, confidence values, and eight-class score vectors.

The multi-person test reports per-component availability at the person-frame
level. Learned gaze estimation and landmark-dependent outputs were available
for both tracked persons on all 344 person-frames per person. Integration
availability is not interpreted as a predictive accuracy measure.

The integration evidence supports software operation, numerical export,
association, accounting, and internal consistency. It does not replace the
separate scientific benchmark or controlled validation evidence for individual
components.

Independent Final Numerical Audit
---------------------------------
The accepted final integration outputs were reviewed at the raw CSV/JSON level
rather than accepted solely from console PASS messages.

The audit confirmed:

- Complete frame accounting for all evaluated videos.
- No missing frame indices in the accepted full-video runs.
- No duplicate frame/person rows in the audited frame exports.
- Finite required numerical outputs with no unexpected Inf values.
- Eye mean openness equal to the mean of the two eye-openness values within
  floating-point precision.
- Mouth openness equal to mouth height divided by mouth width within
  floating-point precision.
- Mouth velocity equal to mouth movement multiplied by measured FPS within
  floating-point precision.
- Learned gaze vectors with unit norm within floating-point precision where
  learned gaze was available.
- Emotion class-score sums, predicted labels, and confidence values internally
  consistent within floating-point precision.
- Temporal window and per-person state behavior consistent with measured video
  timing.
- JSON/CSV export counts consistent for native-export and whole-project analysis.
- Stable tracked identities in the accepted single-person and multi-person
  fixtures, with no duplicate track assignment within a frame.
- Distinct per-person numerical outputs in the multi-person fixture.

These checks confirm that the accepted integration evidence contains real
numerical component outputs rather than availability flags alone.

Result Organization
-------------------
All integration outputs are stored under:

physiotrack-v2/validation/integration/results/

runtime_smoke/
    runtime_smoke_summary.json

face_pipeline_e2e/
    face_pipeline_e2e_frames.csv
    face_pipeline_e2e_results.json
    face_pipeline_e2e_summary.csv

native_export/
    face_pipeline_frames.csv
    face_pipeline_frames.json
    face_pipeline_windows.csv
    face_pipeline_windows.json
    face_pipeline_native_export_summary.json

multi_person_e2e/
    multi_person_full_frames.csv
    multi_person_full_frames.json
    multi_person_full_frame_face_counts.csv
    multi_person_full_gaze_estimation_failures.csv
    multi_person_full_summary.csv
    multi_person_full_video_summary.csv
    multi_person_full_video_summary.json
    multi_person_full_windows.csv
    multi_person_full_windows.json

whole_project_video_analysis/
    whole_project_analysis_summary.json
    whole_project_frames.csv
    whole_project_frames.json
    whole_project_windows.csv
    whole_project_windows.json

whole_project_e2e/
    whole_project_e2e_frames.csv
    whole_project_e2e_modules.csv
    whole_project_e2e_results.json
    whole_project_e2e_summary.json

image_e2e/
    image_e2e_frames.csv
    image_e2e_modules.csv
    image_e2e_results.json
    image_e2e_summary.json

runtime_performance/
    face_pipeline_runtime_frame_times.csv
    face_pipeline_runtime_results.csv
    face_pipeline_runtime_summary.json

Interpretation
--------------
The integration suite demonstrates that the configured PhysioTrack
face-analysis components can operate together on the selected integration
fixtures.

The results establish the following system-level properties:

- The landmark-based gaze descriptor and learned GazeEstimator remain
  independently configurable.
- Both gaze mechanisms can operate simultaneously.
- Enabling learned gaze estimation does not remove the other configured
  single-person analysis outputs.
- Per-frame and temporal exports preserve the expected analysis structure and
  numerical facial measurements.
- FaceQuality operates with the other configured modules and preserves numerical
  confidence, brightness, sharpness, and face-area-ratio values in video,
  multi-person, native-export, whole-project, and static-image integration runs.
- Emotion operates with the complete configured pipeline and preserves a real
  per-face eight-class score vector, predicted label, and confidence in video,
  multi-person, native-export, whole-project, and static-image integration runs.
- Emotion score vectors, predicted labels, and confidence values are internally
  consistent, and multi-person execution preserves distinct per-person Emotion
  outputs.
- MouthMotion exports preserve real numerical movement and velocity values,
  use measured video timing, and maintain independent temporal state per
  tracked identity.
- Learned gaze estimation is associated with both tracked identities in the
  multi-person fixture. Availability was 344/344 for both Persons 1 and 2,
  with no unavailable gaze-estimation records.
- The complete configured pipeline processes the full whole-project fixture
  with consistent frame, face, tracking, module, and export accounting.
- Static-image analysis preserves applicable numerical facial outputs while
  correctly omitting temporal-only outputs rather than synthesizing tracking,
  blink, mouth-motion, or temporal values.
- The validated blink configuration (threshold 0.22, minimum 3 closed frames)
  is used by the complete video integration pipeline.
- Following successful preflight of the complete input set, batch processing
  preserves per-media PASS/FAIL accounting and continues with subsequent media
  after an individual processing failure.

These conclusions concern software integration and execution behavior. They do
not imply that every module has 100% predictive accuracy. Predictive accuracy
must be interpreted from the separate component-level benchmark validations.

Integration results remain system-level evidence even when the corresponding
component has a separate scientific validation. Predictive performance must
always be reported from the dedicated component-level benchmark or controlled
validation package rather than inferred from integration availability.

Related Final System-Level Evidence
-----------------------------------
The runtime-performance benchmark is the eighth integration script and owns
results/runtime_performance/. Controlled robustness is a separate validation
package maintained outside the integration suite and its result ownership.

Runtime / Processing Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The dedicated final runtime benchmark processed the three documented
integration videos with the complete final pipeline configuration:

- Single-person fixture: 429 frames, 3840 x 2160, 25.0 FPS
- Multi-person fixture: 344 frames, 2160 x 3840, 25.0 FPS
- Whole-project fixture: 568 frames, 2560 x 1440, 25.0 FPS
- Total frames: 1341
- Successful frames: 1341
- Failed frames: 0
- Video execution failures: 0
- Total detected faces: 1685
- Processing time: 2198.1849644002796 s
- Wall time: 2208.8969043000034 s
- Effective processing rate: approximately 0.610049 FPS
- Wall-clock processing rate: approximately 0.607090 FPS
- Mean processing time: approximately 1639.21 ms/frame
- Median processing time: approximately 1342.39 ms/frame
- 90th percentile processing time: approximately 2498.77 ms/frame
- 95th percentile processing time: approximately 3432.27 ms/frame
- Input-media read-only verification: PASS

Runtime performance is hardware- and workload-dependent and is reported as
execution performance rather than predictive accuracy. Per-video timing must be
interpreted with the corresponding resolution, source FPS, and number of faces.
The benchmark ran with CUDA unavailable. The measured effective rate of
approximately 0.610 FPS is below the 25.0 FPS source-video rate and therefore
does not demonstrate real-time execution on the documented system.

Controlled Robustness
~~~~~~~~~~~~~~~~~~~~~
The separate controlled robustness package evaluates the static face-analysis
pipeline on one frontal reference image under seven deterministic conditions:

- baseline
- dim lighting
- overexposed lighting
- Gaussian blur
- motion blur
- reduced face scale
- partial occlusion

This controlled robustness study is maintained as a separate validation package.
Its own README and result artifacts are the authoritative source for robustness
outcomes. Robustness results are not used to determine the PASS/FAIL status of
the seven-script integration suite documented here.

This controlled robustness study is a single-image perturbation study. It is
not a population-level robustness benchmark and is interpreted together with
the broader real-media integration tests documented in this README.

Evaluation Scope and Limitations
--------------------------------
This integration package is not an external benchmark and does not provide
ground-truth accuracy metrics for the complete pipeline.

The integration media are controlled regression fixtures. Their purpose is to
exercise software interfaces, configuration behavior, association, failure
handling, and the combined processing path.

A 100% module-availability result on a fixture means that the module generated
a valid output for each evaluated face sample in that fixture. It must not be
reported as 100% task accuracy.

The single-person whole-project fixture cannot validate multi-person
association. Multi-person association is therefore evaluated separately with
multi_person2.mp4.

Similarly, the short runtime smoke test does not replace a full-video test. It
exists only to verify feature configuration and coexistence.

Static-image integration cannot validate temporal event detection or tracking.
Those modules are therefore reported as NOT_APPLICABLE rather than being
assigned synthetic static outputs.

The component benchmark validations remain the authoritative evidence for
scientific performance of individual modules.

Failure Handling
----------------
Before execution, each integration script validates that its discovered input
media can be opened and read. An unreadable video, invalid video FPS, or
unreadable image detected during preflight stops the run before per-media
processing begins. Consequently, no new staged media-level failure summary is
generated for an input that fails preflight; the previously accepted final
results remain unchanged.

After the complete discovered input set passes preflight, batch-oriented
execution attempts each media item. If one item fails during processing, its
failure is recorded with a reason and subsequent media items are still
attempted. Where a processing failure produces no frame-level records, the
failure remains in the staged media-level summary and aggregate status becomes
FAIL even if other media complete successfully. A failed overall run is not
promoted to the accepted final result directory. Previously accepted outputs
are preserved, temporary staged outputs are removed, and the console reports
the failure reason for diagnosis.

Intentional diagnostic failure fixtures are not part of the final integration
package and must not be retained in release-quality result directories.

Reproducibility
---------------
To reproduce the integration validation on another machine:

1. Install the PhysioTrack project and required dependencies.
2. Activate the PhysioTrack thesis environment.
3. Ensure the documented integration fixtures are available under
   validation/integration/test_data/ in the appropriate subdirectories.
4. Change to validation/integration/.
5. Run the seven functional integration scripts in the documented order.
6. Run face_pipeline_runtime_benchmark.py as the eighth test when reproducing
   the documented runtime-performance evidence.
7. Confirm that each clean fixture run completes without an exception.
8. Verify the generated result counts and PASS conditions against the values in
   this README.
9. Verify that numerical measurements are present in the frame/window exports
   where the corresponding module is applicable, including FaceQuality
   confidence, brightness, sharpness, and face-area ratio.
10. Verify Emotion label, confidence, and eight-class score-vector consistency in
    the applicable frame-level exports.
11. Preserve the final result directories as integration evidence.

Runtime can vary across machines and is not a scientific reproducibility
target. The principal reproducibility targets are processed-frame counts, face
and export accounting, feature-configuration behavior, tracked-identity
behavior, static-versus-temporal behavior, module observation, numerical export
structure, and PASS/FAIL invariants.

Regression Use
--------------
The integration suite can also be used for regression checking after changes to
the face-analysis pipeline.

Component-level benchmark validations are separate from integration regression
testing. The integration suite can be rerun to verify that later source-code
changes preserve the established system-level contracts for configuration,
tracking, temporal processing, static-image handling, module availability,
association, failure accounting, and export consistency.

Repository Preservation
-----------------------
The repository should preserve:

- The seven integration scripts
- The complementary face_pipeline_runtime_benchmark.py performance script
- README_INTEGRATION.txt
- Redistributable integration test fixtures
- Final result summaries
- Numerical frame/window exports required to demonstrate the documented runs
- Media-level summary files required to preserve batch PASS/FAIL accounting
- Final audited numerical exports demonstrating Emotion integration
- The separate runtime-performance and controlled-robustness packages referenced
  by this README

Obsolete result-directory layouts, temporary diagnostic outputs, caches,
intentional failure fixtures, and superseded intermediate files should not be
retained in the final integration package.

The integration scripts must remain isolated from benchmark datasets and must
not modify files outside validation/integration during normal validation runs.

Final Integration Status
------------------------
Runtime configuration smoke test:
ACCEPTED

Single-person end-to-end integration:
ACCEPTED

Native export integration:
ACCEPTED

Multi-person end-to-end integration:
ACCEPTED

Whole-project video analysis:
ACCEPTED

Final whole-project end-to-end integration:
ACCEPTED

Static-image end-to-end integration:
ACCEPTED

Runtime performance benchmark:
ACCEPTED

Independent raw numerical audit:
PASS

Emotion integration:
ACCEPTED


Downloading and Preparing Test Media
------------------------------------
The Pexels images and videos used in integration validation are not
distributed with the public GitHub repository. Users must download
these media separately to reproduce the documented results.

The original Pexels download links, creator attributions, and license
information are provided in the "Dataset Sources and Citations" section.

Download the Pexels media and preserve the documented filenames and
directory structure under:

- validation/integration/test_data/single_person/
- validation/integration/test_data/multi_person/
- validation/integration/test_data/whole_project/
- validation/integration/test_data/images/

The five ChatGPT-generated gaze-direction PNG images are included in
the repository under validation/integration/test_data/images/ and do
not require separate downloading.

For the documented validation, use exactly 3 videos and 21 images.
Verify the video resolutions, frame rates, and frame counts against
the values reported above, as different downloads may affect results.

After preparing the media, follow the instructions in
"Running the Integration Validation".

Media provenance and reuse conditions are documented in
VALIDATION_MEDIA_LICENSES.md.