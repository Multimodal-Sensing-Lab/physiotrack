PhysioTrack Integration Validation
==================================

Purpose
-------
This README documents the reproducible integration validation of the complete
PhysioTrack face-analysis pipeline.

The component-level benchmark validations evaluate individual modules under
task-specific protocols. Integration validation has a different purpose: it
verifies that the validated components can operate together inside the
PhysioTrack FaceAnalysis pipeline without breaking feature configuration,
per-face association, temporal processing, tracking, or native result export.

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
   expected analysis structure.
6. Whether the complete configured pipeline can process an entire test video
   without dropped frames, missing face records, broken tracking, or missing
   enabled modules.

This validation is a system-integration and regression test. It is not an
additional accuracy benchmark for the individual modules. Accuracy of the
individual components is evaluated separately using their corresponding
benchmark datasets and validation protocols.

Validation Location
-------------------
The integration package is located in:

physiotrack/validation/integration/

The final directory structure is:

validation/integration/
|-- face_pipeline_e2e_test.py
|-- face_pipeline_multi_person_e2e_test.py
|-- face_pipeline_native_export_test.py
|-- face_pipeline_runtime_smoke_test.py
|-- whole_project_final_e2e_test.py
|-- whole_project_video_analysis.py
|-- test_data/
|   |-- face_blink_pose.mp4
|   |-- multi_person2.mp4
|   `-- istockphoto-1370809321-640_adpp_is.mp4
`-- results/
    |-- face_pipeline_e2e/
    |-- multi_person_e2e/
    |-- native_export/
    |-- runtime_smoke/
    |-- whole_project_e2e/
    `-- whole_project_video_analysis/

Test Data
---------
Integration testing uses fixed video fixtures stored under:

physiotrack/validation/integration/test_data/

The current fixtures are:

face_blink_pose.mp4
    Single-person video used for the runtime configuration checks, the
    single-person end-to-end comparison, and native export validation.

multi_person2.mp4
    Multi-person video used to verify simultaneous face tracking, per-person
    feature generation, and gaze-estimation association.

istockphoto-1370809321-640_adpp_is.mp4
    Full-pipeline video used for whole-project analysis and the final complete
    end-to-end integration test.

These videos are integration fixtures rather than benchmark datasets. They are
not used to estimate model accuracy and do not replace the external benchmark
datasets used by the individual validation components.

Any test video distributed with the repository must have redistribution terms
that permit inclusion in the repository. If a fixture cannot legally be
redistributed, it must be replaced by an equivalent redistributable fixture
before public release.

Path Handling
-------------
The integration scripts resolve their location from __file__ and locate test
videos relative to:

validation/integration/test_data/

The scripts therefore do not require a user-specific test-video input path.

All generated integration results are organized under:

validation/integration/results/

Each script owns a dedicated result subdirectory and may clean only that
subdirectory before regenerating its outputs. A script must not delete another
integration test's results, modify the test videos, or modify project source
code.

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
- Emotion output
- Face-region segmentation
- Temporal summaries
- Per-frame native export
- Per-window native export

The suite verifies presence, configuration behavior, association, accounting,
and export consistency. It does not reinterpret integration availability as a
new quantitative accuracy score for these modules.

Why Multiple Integration Tests Are Required
-------------------------------------------
A single end-to-end video run is insufficient to validate all integration
contracts. The suite separates distinct system-level properties so that a
failure can be localized and interpreted correctly.

1. face_pipeline_runtime_smoke_test.py

   Purpose:
   Verifies runtime feature configuration and coexistence of the two gaze
   mechanisms.

   The test runs four configurations:

   - both_disabled
   - old_gaze_only
   - gaze_estimation_only
   - both_enabled

   This specifically establishes that the existing landmark-based gaze
   descriptor and the learned GazeEstimator are independent optional features.
   Enabling one must not implicitly enable the other, and both must be able to
   operate simultaneously.

   The smoke test uses five frames per configuration. It is deliberately short
   because the objective is configuration-contract verification rather than
   full-video performance measurement.

2. face_pipeline_e2e_test.py

   Purpose:
   Verifies single-person end-to-end integration and checks whether enabling
   the learned GazeEstimator changes the availability of unrelated pipeline
   components.

   The same video is processed in two conditions:

   - gaze estimation disabled
   - gaze estimation enabled

   The test checks complete frame processing, detected-face accounting, module
   availability, and the expected absence/presence of gaze_estimation.

   This provides a direct regression check showing that addition of the learned
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
   - Expected temporal/window structure
   - Agreement between detected-face counts and exported record counts

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
   - Explicit recording of gaze-estimation failures

   This test is particularly important for gaze estimation because a valid gaze
   vector is not sufficient if it is assigned to the wrong person. The
   multi-person test therefore evaluates association behavior in addition to
   feature availability.

5. whole_project_video_analysis.py

   Purpose:
   Generates a detailed complete-pipeline analysis and export over a full test
   video.

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
   - Overall integration PASS/FAIL status

   This is the highest-level system test in the integration package. It
   complements, rather than replaces, the more focused runtime, export, and
   multi-person tests.

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

8. Generated outputs are isolated by test so that rerunning one integration
   test does not delete evidence from another test.

Running the Integration Validation
----------------------------------
Activate the PhysioTrack thesis environment:

conda activate PhysioTrack-Thesis

Change to the integration validation directory:

cd /d <project-path>\physiotrack\validation\integration

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

6. Final complete end-to-end integration test

python whole_project_final_e2e_test.py

Each script cleans only the result directory that it owns before creating new
outputs.

Reported Integration Results
----------------------------
Runtime configuration smoke test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
face_blink_pose.mp4

Cases:
both_disabled
    Processed frames: 5
    Faces: 5
    Landmark-based gaze available: 0
    Learned gaze estimation available: 0
    Status: PASS

old_gaze_only
    Processed frames: 5
    Faces: 5
    Landmark-based gaze available: 5
    Learned gaze estimation available: 0
    Status: PASS

gaze_estimation_only
    Processed frames: 5
    Faces: 5
    Landmark-based gaze available: 0
    Learned gaze estimation available: 5
    Status: PASS

both_enabled
    Processed frames: 5
    Faces: 5
    Landmark-based gaze available: 5
    Learned gaze estimation available: 5
    Status: PASS

Overall runtime smoke status:
PASS

This confirms independent configuration and simultaneous operation of the two
gaze mechanisms.

Single-person end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
face_blink_pose.mp4

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

The remaining configured modules passed in both runs.

Overall status:
PASS

The rerun reproduced the previous integration baseline for the corresponding
single-person summary, with no observed regression after the integration test
reorganization.

Native export integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
face_blink_pose.mp4

Video frames:
429

Processed frames:
429

Detected faces:
429

Frame records:
429

Window records:
429

Existing landmark-based gaze available:
429

Learned gaze estimation available:
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

Frame records:
688

Window records:
688

Gaze-estimation failures:
0

Both tracked persons were present throughout the evaluated video. The configured
modules were available for 344/344 person-frames for each identity in this
fixture. Learned gaze estimation was therefore available for 688/688 evaluated
face instances.

No frame contained fewer than two faces in the accepted run, and no
gaze-estimation failure was recorded.

Overall status:
PASS

Whole-project video analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
istockphoto-1370809321-640_adpp_is.mp4

Reported video frames:
1127

Processed frames:
1127

Detected face records:
1127

Frame records:
1127

Window records:
1127

Tracked identity:
1

Frame-count consistency:
PASS

Export-count consistency:
PASS

The complete configured pipeline produced the expected frame-level and temporal
records across the entire video.

Overall analysis status:
PASS

Final whole-project end-to-end integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test video:
istockphoto-1370809321-640_adpp_is.mp4

Resolution:
768 x 432

FPS:
59.94005994005994

Reported video frames:
1127

Processed frames:
1127

Frames with faces:
1127

Frames without faces:
0

Frames with one face:
1127

Frames with multiple faces:
0

Total face samples:
1127

Unique track IDs:
1

The following modules produced successful output for 1127/1127 face samples:

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

Frame count matches video:
True

Tracking observed:
True

Record count matches:
True

All configured modules observed:
True

Overall status:
PASS

Result Organization
-------------------
All final integration outputs are stored under:

physiotrack/validation/integration/results/

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

Interpretation
--------------
The integration suite demonstrates that the configured PhysioTrack face-analysis
components can operate together on the selected integration fixtures.

The accepted results establish the following system-level properties:

- The original landmark-based gaze descriptor and learned GazeEstimator remain
  independently configurable.
- Both gaze mechanisms can operate simultaneously.
- Enabling learned gaze estimation does not remove the other configured
  single-person analysis outputs.
- Per-frame and temporal exports preserve the expected analysis structure.
- Learned gaze estimation is available for both tracked identities in the
  multi-person fixture without recorded association failure.
- The complete configured pipeline processes the full whole-project fixture
  with consistent frame, face, tracking, module, and export accounting.

These conclusions concern software integration and execution behavior. They do
not imply that every module has 100% predictive accuracy. Predictive accuracy
must be interpreted from the separate component-level benchmark validations.

Evaluation Scope and Limitations
--------------------------------
This integration package is not an external benchmark and does not provide
ground-truth accuracy metrics for the complete pipeline.

The integration videos are controlled regression fixtures. Their purpose is to
exercise the software interfaces and combined processing path.

A 100% module-availability result on a fixture means that the module generated a
valid output for each evaluated face sample in that fixture. It must not be
reported as 100% task accuracy.

The single-person whole-project fixture cannot validate multi-person
association. Multi-person association is therefore evaluated separately with
multi_person2.mp4.

Similarly, the short runtime smoke test does not replace a full-video test. It
exists only to verify feature configuration and coexistence.

The component benchmark validations remain the authoritative evidence for
scientific performance of individual modules.

Reproducibility
---------------
To reproduce the integration validation on another machine:

1. Install the PhysioTrack project and required dependencies.
2. Activate the PhysioTrack thesis environment.
3. Ensure the documented integration test videos are available under
   validation/integration/test_data/.
4. Change to validation/integration/.
5. Run the six scripts in the documented order.
6. Confirm that each test completes without an exception.
7. Verify the generated result counts and PASS conditions against the values in
   this README.
8. Preserve the result directories as integration evidence.

Runtime can vary across machines and is not a scientific reproducibility
target. The principal reproducibility targets are the processed-frame counts,
face and export accounting, feature-configuration behavior, tracked-identity
behavior, module observation, and PASS/FAIL invariants.

Regression Use
--------------
The integration suite is also intended for final regression checking after the
remaining project validation work is complete.

The component-level benchmark validations do not need to be repeated merely
because integration tests are rerun. Instead, the finalized integration suite
can be rerun once as a final regression and integration pass to verify that
subsequent project changes have not broken the established system-level
contracts.

Repository Preservation
-----------------------
The final repository should preserve:

- The six integration scripts
- The integration README
- Redistributable integration test fixtures
- The final result summaries and audit-relevant exports required to demonstrate
  the accepted runs

Obsolete result-directory layouts, temporary diagnostic outputs, caches, and
superseded intermediate files should not be retained in the final integration
package.

The integration scripts must remain isolated from the benchmark datasets and
must not modify files outside validation/integration during normal validation
runs.
