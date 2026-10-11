import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarks:
    """Facial landmark extraction using MediaPipe."""

    def __init__(self, model_path, num_faces=1):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def predict(self, frame):
        """Extract facial landmarks from one image."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self.landmarker.detect(mp_image)

        return result.face_landmarks

    @staticmethod
    def _square_box(frame, box):
        """Create a square face box while keeping it inside the frame."""
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = map(float, box)

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            return None

        side = max(box_width, box_height)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        side = min(side, width, height)

        sx1 = int(round(cx - side / 2.0))
        sy1 = int(round(cy - side / 2.0))
        sx2 = sx1 + int(round(side))
        sy2 = sy1 + int(round(side))

        if sx1 < 0:
            sx2 -= sx1
            sx1 = 0

        if sy1 < 0:
            sy2 -= sy1
            sy1 = 0

        if sx2 > width:
            shift = sx2 - width
            sx1 -= shift
            sx2 = width

        if sy2 > height:
            shift = sy2 - height
            sy1 -= shift
            sy2 = height

        sx1 = max(0, sx1)
        sy1 = max(0, sy1)

        if sx2 <= sx1 or sy2 <= sy1:
            return None

        return sx1, sy1, sx2, sy2

    def predict_face(self, frame, box):
        """Extract landmarks for one tracked face."""
        square_box = self._square_box(frame, box)

        if square_box is None:
            return None

        x1, y1, x2, y2 = square_box

        crop = frame[y1:y2, x1:x2]

        face_landmarks = self.predict(crop)

        if not face_landmarks:
            return None

        landmarks = face_landmarks[0]

        frame_height, frame_width = frame.shape[:2]
        crop_width = x2 - x1
        crop_height = y2 - y1

        for landmark in landmarks:
            landmark.x = (
                x1 + landmark.x * crop_width
            ) / frame_width

            landmark.y = (
                y1 + landmark.y * crop_height
            ) / frame_height

        return landmarks

    def close(self):
        """Release the MediaPipe landmarker."""
        self.landmarker.close()