"""Deterministic face-region overlap regression for the PhysioTrack pipeline.

Exercise real Segmentation.Face, FaceRegions and FaceAnalysis code without
loading model weights or changing scientific benchmark outputs. The controlled
parser returns a known semantic label for each face crop, allowing the pixel
counts and association to be checked independently of model accuracy.
"""

import numpy as np

from physiotrack.face import FaceAnalysis, FaceAnalysisConfig
from physiotrack.face.regions import FaceRegions
from physiotrack.results import Instance, Result
from physiotrack.segment import Segmentation


class ControlledSegFace:
    def infer(self, crop):
        label = int(crop[0, 0, 0])
        mask = np.full(crop.shape[:2], label, dtype=np.int32)
        mask[0, -1] = 0
        return mask


class ControlledDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame):
        return Result(orig_img=frame, task="face", instances=[
            Instance(box=np.array(box, dtype=float), confidence=0.9,
                     cls=0, cls_name="face")
            for box in self.boxes
        ])


def main():
    segmenter = Segmentation.Face.__new__(Segmentation.Face)
    segmenter.segmentor = ControlledSegFace()
    face_regions = FaceRegions(segmenter=segmenter)
    config = FaceAnalysisConfig(
        tracking=False, head_pose=False, landmarks=False, quality=False,
        eyes=False, blink=False, gaze=False, gaze_estimation=False,
        mouth=False, mouth_motion=False, emotion=False, regions=True,
        temporal=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    a = [10, 10, 70, 70]
    b = [40, 40, 90, 90]
    frame[10, 10, 0] = 2  # skin
    frame[40, 40, 0] = 10  # nose; within both bounding boxes
    expected = {
        tuple(a): ("skin", 60 * 60 - 1),
        tuple(b): ("nose", 50 * 50 - 1),
    }

    for order in ([a, b], [b, a]):
        native = face_regions.predict(frame, boxes=order)
        assert len(native["faces"]) == 2
        assert len(native["result"].face_seg_maps) == 2
        for face in native["faces"]:
            label, count = expected[tuple(face["box"])]
            assert set(face["regions"]) == {label}
            assert int(face["regions"][label].sum()) == count

        pipeline = FaceAnalysis(detector=ControlledDetector(order),
                                regions=face_regions, config=config, device="cpu")
        result = pipeline.predict(frame)
        assert len(result) == 2
        for instance in result:
            label, count = expected[tuple(int(x) for x in instance.box)]
            region = instance.face_features["regions"]
            assert region["available"] is True
            assert region["pixel_counts"] == {label: count}
            assert region["association_iou"] == 1.0

    print("B03 per-face overlapping segmentation and pipeline association: PASS")


if __name__ == "__main__":
    main()
