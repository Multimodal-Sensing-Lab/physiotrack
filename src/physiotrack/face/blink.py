class BlinkDetector:
    """Detect blinks from temporal eye-openness values."""

    def __init__(
        self,
        threshold,
        fps,
        min_closed_frames=2,
    ):
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")

        if fps <= 0:
            raise ValueError("fps must be greater than zero")

        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1")

        self.threshold = float(threshold)
        self.fps = float(fps)
        self.min_closed_frames = int(min_closed_frames)

        self._states = {}

    def _get_state(self, person_id):
        if person_id not in self._states:
            self._states[person_id] = {
                "closed_frames": 0,
                "blink_count": 0,
                "frame_count": 0,
            }

        return self._states[person_id]

    def update(self, openness, person_id=0):
        """Update blink state using one frame."""
        state = self._get_state(person_id)
        state["frame_count"] += 1

        blink = False
        blink_duration = None

        if openness is None:
            # Do not bridge a closed-eye sequence across missing landmarks.
            state["closed_frames"] = 0

            return {
                "eye_state": "unknown",
                "blink": False,
                "blink_count": state["blink_count"],
                "blink_duration": None,
                "blink_rate": self._blink_rate(state),
            }

        if openness < self.threshold:
            state["closed_frames"] += 1
            eye_state = "closed"

        else:
            eye_state = "open"

            if state["closed_frames"] >= self.min_closed_frames:
                blink = True
                state["blink_count"] += 1
                blink_duration = (
                    state["closed_frames"] / self.fps
                )

            state["closed_frames"] = 0

        return {
            "eye_state": eye_state,
            "blink": blink,
            "blink_count": state["blink_count"],
            "blink_duration": blink_duration,
            "blink_rate": self._blink_rate(state),
        }

    def _blink_rate(self, state):
        elapsed_minutes = (
            state["frame_count"] / self.fps
        ) / 60.0

        if elapsed_minutes == 0:
            return 0.0

        return (
            state["blink_count"]
            / elapsed_minutes
        )

    def reset(self, person_id=None):
        """Reset blink history."""
        if person_id is None:
            self._states.clear()
        else:
            self._states.pop(person_id, None)