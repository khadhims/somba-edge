import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class ViolationEpisodeTracker:
    """Collapses a burst of per-frame violation detections into a single
    "episode" alert, and is tolerant of a flaky detector.

    Robustness:
      - An episode starts only after detection is *sustained* (>= start_frames
        consecutive samples OR >= start_seconds), filtering 1-2 frame false
        positives.
      - Brief detection dropouts (<= miss_grace_frames) do NOT reset the streak,
        so a single missed frame can't fragment one real violation into many
        alerts.
      - After an episode ends, a cooldown suppresses immediate re-triggering on
        the same camera.

    Each finalized episode reports event_start/event_end, duration and the
    number of detections, picking the highest-confidence frame as the snapshot.
    """

    start_frames: int = 15
    start_seconds: float = 0.5
    end_seconds: float = 5.0
    miss_grace_frames: int = 5
    cooldown_seconds: float = 10.0

    in_episode: bool = False
    consecutive_frames: int = 0
    miss_frames: int = 0
    streak_started_at: float | None = None
    last_detected_at: float | None = None
    episode_started_at: float | None = None
    detection_count: int = 0
    cooldown_until: float = 0.0
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
            self.miss_frames = 0
            self.consecutive_frames += 1
            if self.streak_started_at is None:
                self.streak_started_at = now
            self.last_detected_at = now
            if self.in_episode:
                self.detection_count += 1

            if confidence >= self.best_confidence:
                self.best_confidence = confidence
                self.best_name = violation_name
                self.best_bbox = bbox
                self.best_frame = frame.copy() if hasattr(frame, "copy") else frame

            if not self.in_episode and now >= self.cooldown_until:
                streak_elapsed = now - (self.streak_started_at or now)
                if (
                    self.consecutive_frames >= self.start_frames
                    or streak_elapsed >= self.start_seconds
                ):
                    self.in_episode = True
                    self.episode_started_at = self.streak_started_at or now
                    self.detection_count = self.consecutive_frames
            return None

        # Non-detection frame.
        self.miss_frames += 1

        if not self.in_episode:
            # Tolerate brief gaps before abandoning the pre-episode streak.
            if self.miss_frames > self.miss_grace_frames:
                self.consecutive_frames = 0
                self.streak_started_at = None
            return None

        # In an episode: only end after a sustained absence.
        if (
            self.last_detected_at is not None
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

        start = self.episode_started_at or self.last_detected_at or time.time()
        end = self.last_detected_at or start
        duration_seconds = round(max(0.0, end - start), 1)

        payload = {
            "violation_name": self.best_name,
            "bbox": self.best_bbox,
            "frame": self.best_frame,
            "detected_at": _iso(start),
            "event_start": _iso(start),
            "event_end": _iso(end),
            "duration_seconds": duration_seconds,
            "total_detections": max(1, self.detection_count),
        }
        self.cooldown_until = time.time() + self.cooldown_seconds
        self._reset()
        return payload

    def _reset(self):
        # Note: cooldown_until is intentionally preserved across resets.
        self.in_episode = False
        self.consecutive_frames = 0
        self.miss_frames = 0
        self.streak_started_at = None
        self.last_detected_at = None
        self.episode_started_at = None
        self.detection_count = 0
        self.best_name = None
        self.best_bbox = None
        self.best_confidence = 0.0
        self.best_frame = None
