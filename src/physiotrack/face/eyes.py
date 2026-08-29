import math


class EyeOpenness:
    """Estimate eye openness from facial landmarks."""

    LEFT_EYE = (362, 385, 387, 263, 373, 380)
    RIGHT_EYE = (33, 160, 158, 133, 153, 144)

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def _eye_ratio(self, landmarks, indices):
        p1, p2, p3, p4, p5, p6 = [
            landmarks[i] for i in indices
        ]

        vertical_1 = self._distance(p2, p6)
        vertical_2 = self._distance(p3, p5)
        horizontal = self._distance(p1, p4)

        if horizontal == 0:
            return None

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    def predict(self, landmarks):
        """Calculate openness for the left and right eyes."""
        left = self._eye_ratio(
            landmarks,
            self.LEFT_EYE,
        )

        right = self._eye_ratio(
            landmarks,
            self.RIGHT_EYE,
        )

        if left is None or right is None:
            mean = None
        else:
            mean = (left + right) / 2.0

        return {
            "left_openness": left,
            "right_openness": right,
            "mean_openness": mean,
        }