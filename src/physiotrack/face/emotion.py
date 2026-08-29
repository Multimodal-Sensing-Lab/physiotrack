import numpy as np

from emotiefflib.facial_analysis import EmotiEffLibRecognizer


class FaceEmotion:
    """Facial expression recognition using EmotiEffLib."""

    def __init__(
        self,
        model_name="enet_b0_8_best_afew",
        engine="onnx",
    ):
        self.recognizer = EmotiEffLibRecognizer(
            engine=engine,
            model_name=model_name,
        )

    def predict(self, face_image):
        """Predict facial expression from a cropped face image."""
        predicted, scores = self.recognizer.predict_emotions(
            face_image,
            logits=False,
        )

        scores = np.asarray(scores[0], dtype=float)

        class_names = [
            self.recognizer.idx_to_emotion_class[i]
            for i in range(
                len(self.recognizer.idx_to_emotion_class)
            )
        ]

        score_dict = {
            class_name: float(score)
            for class_name, score in zip(
                class_names,
                scores,
            )
        }

        label = predicted[0]

        return {
            "emotion": label,
            "confidence": float(score_dict[label]),
            "scores": score_dict,
        }