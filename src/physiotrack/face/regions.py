import numpy as np

from ..segment import Segmentation


class FaceRegions:
    """Extract semantic regions separately for each detected face."""

    def __init__(self, segmenter=None, device="cpu", verbose=False):
        if segmenter is None:
            segmenter = Segmentation.Face(
                device=device,
                verbose=verbose,
            )

        self.segmenter = segmenter

    @staticmethod
    def _class_ids(result):
        if result.names is None:
            return {}

        return {
            name.lower(): int(class_id)
            for class_id, name in result.names.items()
            if int(class_id) != 0
        }

    def predict(self, frame, boxes=None):
        """Extract semantic regions separately for each face."""
        result = self.segmenter.predict(
            frame,
            boxes=boxes,
        )

        if result.seg_map is None:
            return {
                "result": result,
                "faces": [],
            }

        class_ids = self._class_ids(result)

        height, width = result.seg_map.shape

        faces = []

        for instance in result:

            if instance.box is None:
                continue

            x1, y1, x2, y2 = map(int, instance.box)

            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            face_seg_map = result.seg_map[
                y1:y2,
                x1:x2
            ]

            face_regions = {}

            for name, class_id in class_ids.items():

                mask = face_seg_map == class_id

                if np.any(mask):
                    face_regions[name] = mask

            faces.append(
                {
                    "box": np.array(
                        [x1, y1, x2, y2],
                        dtype=int,
                    ),
                    "regions": face_regions,
                }
            )

        return {
            "result": result,
            "faces": faces,
        }