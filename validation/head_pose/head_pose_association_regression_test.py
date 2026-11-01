"""Deterministic head-pose association regression using the real software paths.

No AFLW images, trained model weights, or output files are required. Rotation
matrices are controlled to isolate input-index retention and face association.
Run from the repository root with:
    python validation/head_pose/head_pose_association_regression_test.py
"""

from collections import deque

import numpy as np
import torch

from physiotrack.face.analysis import FaceAnalysis
from physiotrack.face.config import FaceAnalysisConfig
from physiotrack.face.face_orientation import FaceOrientation
from physiotrack.modules._6DRepNet360.inference import HeadPoseEstimator
from physiotrack.results import Instance, Result


class ControlledDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame):
        return Result(
            orig_img=frame,
            task="face",
            instances=[
                Instance(
                    id=100 + index,  # Track identity differs from source box index.
                    box=np.asarray(box, dtype=float),
                    confidence=0.9,
                    cls=0,
                    cls_name="face",
                )
                for index, box in enumerate(self.boxes)
            ],
        )


def make_orientation():
    # __new__ bypasses weight loading; the real crop/index/Result paths run.
    backend = HeadPoseEstimator.__new__(HeadPoseEstimator)
    backend.render_pose = False
    backend.inference_times = deque(maxlen=100)
    backend._inference_batch = lambda crops: [
        torch.eye(3, dtype=torch.float32) for _ in crops
    ]

    orientation = FaceOrientation.__new__(FaceOrientation)
    orientation.estimator = backend
    return orientation


def run_pipeline(boxes, orientation):
    config = FaceAnalysisConfig(
        tracking=False,
        head_pose=True,
        landmarks=False,
        quality=False,
        eyes=False,
        blink=False,
        gaze=False,
        gaze_estimation=False,
        mouth=False,
        mouth_motion=False,
        emotion=False,
        regions=False,
        temporal=False,
    )
    pipeline = FaceAnalysis(
        detector=ControlledDetector(boxes),
        orientation=orientation,
        config=config,
        device="cpu",
    )
    return pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))


def main():
    invalid = [10, 10, 10, 10]
    valid_a = [15, 10, 40, 45]
    valid_b = [55, 10, 90, 50]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    native_orientation = make_orientation()
    native_backend = native_orientation.predict(frame, [invalid, valid_b])
    assert len(native_backend) == 1
    assert native_backend[0].id == 1
    np.testing.assert_array_equal(native_backend[0].box, valid_b)

    first_skipped = run_pipeline([invalid, valid_b], make_orientation())
    assert [face.id for face in first_skipped] == [100, 101]
    assert first_skipped[0].orientation is None
    assert first_skipped[1].orientation is not None
    assert all(np.isfinite(value) for value in first_skipped[1].orientation.values())

    second_skipped = run_pipeline([valid_a, invalid], make_orientation())
    assert second_skipped[0].orientation is not None
    assert second_skipped[1].orientation is None

    # A completely skipped batch produces no pose for any face.
    all_skipped = run_pipeline([invalid, invalid], make_orientation())
    assert len(all_skipped) == 2
    assert all(face.orientation is None for face in all_skipped)

    # Valid crops separated by a skipped box keep their original indices.
    separated = run_pipeline([valid_a, invalid, valid_b], make_orientation())
    assert [face.orientation is not None for face in separated] == [True, False, True]

    # Controlled single-face pipeline and direct predictor share the same pose.
    single_direct = make_orientation().predict(frame, [valid_a])[0].orientation
    single_pipeline = run_pipeline([valid_a], make_orientation())[0].orientation
    assert single_pipeline == single_direct

    class ReorderedOrientation:
        def __init__(self):
            self.orientation = make_orientation()

        def predict(self, input_frame, boxes):
            output = self.orientation.predict(input_frame, boxes)
            for face in output:
                face.orientation = {
                    "yaw": float(face.id + 10),
                    "pitch": 0.0,
                    "roll": 0.0,
                }
            return Result(orig_img=input_frame, task="face",
                          instances=list(reversed(list(output))))

    reordered = run_pipeline([valid_a, valid_b], ReorderedOrientation())
    assert [face.orientation["yaw"] for face in reordered] == [10.0, 11.0]

    class UnindexedOrientation:
        def predict(self, input_frame, boxes):
            return Result(
                orig_img=input_frame, task="face",
                instances=[
                    Instance(id=None, box=np.asarray(boxes[index]),
                             orientation={"yaw": float(index),
                                          "pitch": 0.0, "roll": 0.0})
                    for index in (1, 0)
                ],
            )

    unindexed = run_pipeline([valid_a, valid_b], UnindexedOrientation())
    assert [face.orientation["yaw"] for face in unindexed] == [0.0, 1.0]

    print("B04 original-box-index head-pose association regression: PASS")


if __name__ == "__main__":
    main()
