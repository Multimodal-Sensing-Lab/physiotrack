import numpy as np

from physiotrack.face.emotion import FaceEmotion
from physiotrack.face.face_orientation import FaceOrientation
from physiotrack.face.regions import FaceRegions
from physiotrack.results import Instance, Result


class DummyEmotionRecognizer:
    idx_to_emotion_class = {
        0: "Anger",
        1: "Contempt",
        2: "Disgust",
        3: "Fear",
        4: "Happiness",
        5: "Neutral",
        6: "Sadness",
        7: "Surprise",
    }

    def predict_emotions(self, face_image, logits=False):
        return (
            ["Happiness"],
            np.array(
                [[
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.80,
                    0.05,
                    0.03,
                    0.02,
                ]],
                dtype=float,
            ),
        )


def test_emotion_predict_formats_output():
    emotion = FaceEmotion.__new__(FaceEmotion)
    emotion.recognizer = DummyEmotionRecognizer()

    face = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    result = emotion.predict(face)

    assert result["emotion"] == "Happiness"

    assert np.isclose(
        result["confidence"],
        0.80,
    )

    assert set(
        result["scores"].keys()
    ) == {
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    }


class DummySegmenter:
    def predict(self, frame, boxes=None):
        seg_map = np.zeros(
            (100, 100),
            dtype=np.int32,
        )

        seg_map[20:60, 20:60] = 1
        seg_map[30:40, 30:40] = 2

        instances = [
            Instance(
                box=np.array(
                    [20, 20, 60, 60],
                    dtype=float,
                )
            )
        ]

        result = Result(
            orig_img=frame,
            instances=instances,
            task="segmentation",
        )

        result.seg_map = seg_map

        result.names = {
            0: "background",
            1: "skin",
            2: "l_eye",
        }

        return result


def test_regions_extracts_per_face_masks():
    regions = FaceRegions(
        segmenter=DummySegmenter()
    )

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    output = regions.predict(
        frame,
        boxes=[
            [20, 20, 60, 60]
        ],
    )

    assert len(output["faces"]) == 1

    face = output["faces"][0]

    assert "skin" in face["regions"]
    assert "l_eye" in face["regions"]

    assert face["regions"]["skin"].shape == (
        40,
        40,
    )

    assert face["regions"]["l_eye"].shape == (
        40,
        40,
    )


def test_regions_returns_empty_faces_without_segmentation_map():
    class NoMapSegmenter:
        def predict(self, frame, boxes=None):
            result = Result(
                orig_img=frame,
                instances=[],
                task="segmentation",
            )

            result.seg_map = None
            result.names = None

            return result

    regions = FaceRegions(
        segmenter=NoMapSegmenter()
    )

    frame = np.zeros(
        (50, 50, 3),
        dtype=np.uint8,
    )

    output = regions.predict(frame)

    assert output["faces"] == []


class DummyOrientationModel:
    def predict(self, source, boxes=None):
        instances = []

        for box in boxes:
            instances.append(
                Instance(
                    box=np.asarray(
                        box,
                        dtype=float,
                    ),
                    orientation={
                        "pitch": 1.0,
                        "yaw": 2.0,
                        "roll": 3.0,
                    },
                )
            )

        return Result(
            orig_img=source,
            instances=instances,
            task="face",
        )


def test_orientation_result_preserves_pose_values():
    orientation = FaceOrientation.__new__(
        FaceOrientation
    )

    orientation.model = DummyOrientationModel()

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = orientation.model.predict(
        frame,
        boxes=[
            [10, 10, 40, 40],
            [50, 50, 90, 90],
        ],
    )

    assert len(result) == 2

    for instance in result:
        assert instance.orientation == {
            "pitch": 1.0,
            "yaw": 2.0,
            "roll": 3.0,
        }