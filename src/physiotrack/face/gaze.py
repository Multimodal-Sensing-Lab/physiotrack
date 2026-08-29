import math


class GazeDescriptor:
    """Estimate normalized iris position in an eye-centered coordinate system."""

    RIGHT_IRIS = 468
    LEFT_IRIS = 473

    RIGHT_CORNERS = (33, 133)
    LEFT_CORNERS = (362, 263)

    @staticmethod
    def _eye_coordinates(landmarks, iris_index, corner_indices):
        iris = landmarks[iris_index]
        p1 = landmarks[corner_indices[0]]
        p2 = landmarks[corner_indices[1]]

        vx = p2.x - p1.x
        vy = p2.y - p1.y

        width = math.sqrt(vx * vx + vy * vy)

        if width == 0:
            return None, None

        ux = vx / width
        uy = vy / width

        # Perpendicular direction in image coordinates.
        nx = -uy
        ny = ux

        dx = iris.x - p1.x
        dy = iris.y - p1.y

        horizontal = (dx * ux + dy * uy) / width
        vertical = (dx * nx + dy * ny) / width

        return horizontal, vertical

    def predict(self, landmarks):
        """Return normalized iris coordinates for both eyes."""

        right_x, right_y = self._eye_coordinates(
            landmarks,
            self.RIGHT_IRIS,
            self.RIGHT_CORNERS,
        )

        left_x, left_y = self._eye_coordinates(
            landmarks,
            self.LEFT_IRIS,
            self.LEFT_CORNERS,
        )

        mean_x = (
            None
            if right_x is None or left_x is None
            else (right_x + left_x) / 2.0
        )

        mean_y = (
            None
            if right_y is None or left_y is None
            else (right_y + left_y) / 2.0
        )

        return {
            "right_iris_x": right_x,
            "right_iris_y": right_y,
            "left_iris_x": left_x,
            "left_iris_y": left_y,
            "mean_iris_x": mean_x,
            "mean_iris_y": mean_y,
        }