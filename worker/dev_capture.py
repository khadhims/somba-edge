import os
import subprocess

import numpy as np

DEV_CAPTURE_WIDTH = int(os.getenv("DEV_CAPTURE_WIDTH", "1280"))
DEV_CAPTURE_HEIGHT = int(os.getenv("DEV_CAPTURE_HEIGHT", "720"))


class DevFfmpegCapture:
    """DEV ONLY: decode go2rtc RTSP through FFmpeg (handles HEVC/other codecs)."""

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.width = DEV_CAPTURE_WIDTH
        self.height = DEV_CAPTURE_HEIGHT
        self.proc: subprocess.Popen | None = None
        self._frame_bytes = self.width * self.height * 3

    def open(self):
        self.release()
        scale_pad = (
            f"scale={self.width}:{self.height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.stream_url,
            "-an",
            "-vf",
            scale_pad,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.proc or not self.proc.stdout:
            return False, None
        if self.proc.poll() is not None:
            return False, None
        raw = self.proc.stdout.read(self._frame_bytes)
        if not raw or len(raw) != self._frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        return True, frame

    def release(self):
        if not self.proc:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def isOpened(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
