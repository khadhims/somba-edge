import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ViolationEpisodeTracker:
    start_frames: int = 15
    start_seconds: float = 0.5
    end_seconds: float = 5.0

    in_episode: bool = False
    consecutive_frames: int = 0
    streak_started_at: float | None = None
    last_detected_at: float | None = None
    episode_started_at: float | None = None
    best_name: str | None = None
    best_bbox: list[float] | None = None
    best_confidence: float = 0.0
    best_frame: object | None = field(default=None, repr=False)

    def observe(
        self,
        detected: bool,
        violation_name: str | None,
        bbox: list[float] | None,
        confidence: float,
        frame,
    ) -> dict | None:
        now = time.time()

        if detected and violation_name and bbox is not None:
            self.consecutive_frames += 1
            if self.streak_started_at is None:
                self.streak_started_at = now
            self.last_detected_at = now

            if confidence >= self.best_confidence:
                self.best_confidence = confidence
                self.best_name = violation_name
                self.best_bbox = bbox
                self.best_frame = frame

            if not self.in_episode:
                streak_elapsed = now - self.streak_started_at
                if (
                    self.consecutive_frames >= self.start_frames
                    or streak_elapsed >= self.start_seconds
                ):
                    self.in_episode = True
                    self.episode_started_at = now
            return None

        self.consecutive_frames = 0
        self.streak_started_at = None

        if (
            self.in_episode
            and self.last_detected_at is not None
            and now - self.last_detected_at >= self.end_seconds
        ):
            return self._finalize_episode()

        return None

    def check_end(self) -> dict | None:
        if not self.in_episode or self.last_detected_at is None:
            return None
        if time.time() - self.last_detected_at >= self.end_seconds:
            return self._finalize_episode()
        return None

    def flush(self) -> dict | None:
        if not self.in_episode:
            return None
        return self._finalize_episode()

    def _finalize_episode(self) -> dict | None:
        if not self.best_name or not self.best_bbox or self.best_frame is None:
            self._reset()
            return None

        payload = {
            "violation_name": self.best_name,
            "bbox": self.best_bbox,
            "frame": self.best_frame,
            "detected_at": datetime.fromtimestamp(
                self.episode_started_at or time.time(),
                tz=timezone.utc,
            ).isoformat(),
        }
        self._reset()
        return payload

    def _reset(self):
        self.in_episode = False
        self.consecutive_frames = 0
        self.streak_started_at = None
        self.last_detected_at = None
        self.episode_started_at = None
        self.best_name = None
        self.best_bbox = None
        self.best_confidence = 0.0
        self.best_frame = None
