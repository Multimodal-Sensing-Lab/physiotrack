import math


class MouthOpenness:
    """Estimate mouth openness from facial landmarks."""

    LEFT_CORNER = 61
    RIGHT_CORNER = 291

    UPPER_LIP = 13
    LOWER_LIP = 14

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    def predict(self, landmarks):
        """Calculate normalized mouth openness."""
        left = landmarks[self.LEFT_CORNER]
        right = landmarks[self.RIGHT_CORNER]

        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]

        mouth_width = self._distance(left, right)
        mouth_height = self._distance(upper, lower)

        if mouth_width == 0:
            return {
                "mouth_openness": None,
                "mouth_width": None,
                "mouth_height": None,
            }

        openness = mouth_height / mouth_width

        return {
            "mouth_openness": openness,
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
        }