"""Controlled face-landmark association regression, without model inference.

The real FaceLandmarks crop/remapping logic and FaceAnalysis face iteration are
exercised with synthetic MediaPipe outputs. This is association software evidence,
not a 300-W localization-accuracy benchmark or a MediaPipe detection-rate study.
"""

from types import SimpleNamespace

import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.landmarks import FaceLandmarks
from physiotrack.results import Instance, Result


class ControlledDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame):
        return Result(
            orig_img=frame,
            task="face",
            instances=[
                Instance(id=None, box=np.asarray(box, dtype=float),
                         confidence=1.0, cls=0, cls_name="face")
                for box in self.boxes
            ],
        )


def candidate(crop_box, center):
    x1, y1, x2, y2 = crop_box
    return [
        SimpleNamespace(
            x=(center[0] + (i % 7 - 3) * 0.2 - x1) / (x2 - x1),
            y=(center[1] + (i % 5 - 2) * 0.2 - y1) / (y2 - y1),
            z=0.0,
        )
        for i in range(478)
    ]


def run():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    target = [40, 10, 60, 90]
    neighbor = [10, 20, 35, 50]
    crop_box = FaceLandmarks._square_box(frame, target)
    assert crop_box == (10, 10, 90, 90)
    assert target[0] >= neighbor[2]

    landmarker = FaceLandmarks.__new__(FaceLandmarks)
    landmarker.predict = lambda crop: [candidate(crop_box, (22.5, 35.0))]
    assert landmarker.predict_face(frame, target) is None

    for reversed_order in (False, True):
        groups = [candidate(crop_box, (22.5, 35.0)),
                  candidate(crop_box, (50.0, 50.0))]
        if reversed_order:
            groups.reverse()
        landmarker.predict = lambda crop, groups=groups: groups
        result = landmarker.predict_face(frame, target, other_boxes=[neighbor])
        assert result is not None and len(result) == 478
        assert abs(float(np.median([p.x for p in result])) - 0.5) < 1e-3

    # Real FaceAnalysis uses the other face boxes in its native landmarker path.
    # A controlled backend face at a fixed crop-relative location belongs to
    # the neighboring face, not to the larger target's expanded square.
    landmarker.predict = lambda crop: [
        [SimpleNamespace(x=(22.5 + (i % 5 - 2) * 0.1 - 10) / 80,
                         y=(35 + (i % 7 - 3) * 0.1 - 10) / 80,
                         z=0.0) for i in range(478)]
    ]
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=True, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=False,
        temporal=False,
    )
    for order in ([target, neighbor], [neighbor, target]):
        pipeline = FaceAnalysis(
            config=config, detector=ControlledDetector(order),
            landmarks=landmarker, device="cpu",
        )
        result = pipeline.predict(frame)
        observed = {
            tuple(int(value) for value in face.box):
            face.face_features["landmarks"] for face in result
        }
        assert observed[tuple(target)]["available"] is False
        assert observed[tuple(target)]["count"] == 0
        assert observed[tuple(neighbor)]["available"] is True
        assert observed[tuple(neighbor)]["count"] == 478

    print("B06 face-landmark target association and pipeline regression: PASS")


if __name__ == "__main__":
    run()
