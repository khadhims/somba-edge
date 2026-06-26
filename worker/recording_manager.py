import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime

from filename_utils import camera_file_slug, sanitize_filename_part
from media_utils import (
    RECORDING_CRF,
    RECORDING_ENCODE_PRESET,
    is_valid_media,
    remux_with_faststart,
    stop_ffmpeg_gracefully,
    transcode_recording,
)
from s3_upload import upload_file as s3_upload_file
from stream_resolver import is_dev_mode, resolve_recording_stream

POST_BUFFER_SEC = int(os.getenv("RECORDING_POST_BUFFER_SEC", "90"))
PERSON_START_FRAMES = int(os.getenv("PERSON_START_FRAMES", "25"))
MIN_RECORDING_DURATION_SEC = int(os.getenv("MIN_RECORDING_DURATION_SEC", "180"))
MAX_SEGMENT_SEC = int(os.getenv("RECORDING_MAX_SEGMENT_SEC", "300"))


class CameraRecordingSession:
    def __init__(
        self,
        camera_uuid: str,
        camera_slug: str,
        activity: str,
        stream_url: str,
        recordings_dir: str,
        post_buffer_sec: int,
        max_segment_sec: int,
        person_start_frames: int,
        min_recording_duration_sec: int,
        on_finalize,
    ):
        self.camera_uuid = camera_uuid
        self.camera_slug = camera_slug
        self.activity = activity
        self.stream_url = stream_url
        self.recordings_dir = recordings_dir
        self.post_buffer_sec = post_buffer_sec
        self.max_segment_sec = max_segment_sec
        self.person_start_frames = person_start_frames
        self.min_recording_duration_sec = min_recording_duration_sec
        self.on_finalize = on_finalize
        self.live_h264_encode = False
        self.lock = threading.Lock()
        self.is_recording = False
        self.last_activity_at = 0.0
        self.consecutive_person_frames = 0
        self.session_start: datetime | None = None
        self.ffmpeg_proc: subprocess.Popen | None = None
        self.raw_path: str | None = None
        self.final_path: str | None = None
        self._temp_dir: str | None = None

        os.makedirs(self.recordings_dir, exist_ok=True)

    def signal_activity(self, active: bool):
        now = time.time()
        with self.lock:
            if active:
                self.consecutive_person_frames += 1
                self.last_activity_at = now
                if (
                    not self.is_recording
                    and self.consecutive_person_frames >= self.person_start_frames
                ):
                    self._start_recording()
            else:
                self.consecutive_person_frames = 0

    def tick(self):
        with self.lock:
            if not self.is_recording:
                return
            now = time.time()
            if now - self.last_activity_at >= self.post_buffer_sec:
                self._stop_recording()
                return
            if self.session_start and now - self.session_start.timestamp() >= self.max_segment_sec:
                self._stop_recording()

    def _start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        activity_slug = sanitize_filename_part(self.activity) or "activity"
        camera_dir = os.path.join(self.recordings_dir, self.camera_slug)
        os.makedirs(camera_dir, exist_ok=True)
        base_name = f"{self.camera_slug}_{activity_slug}_{timestamp}"
        self._temp_dir = tempfile.mkdtemp(prefix="somba-rec-")
        self.raw_path = os.path.join(self._temp_dir, "capture.mp4")
        self.final_path = os.path.join(camera_dir, f"{base_name}.mp4")

        input_url, self.live_h264_encode = resolve_recording_stream(self.stream_url)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-loglevel",
            "warning",
        ]

        if input_url.startswith("rtsp://"):
            ffmpeg_cmd.extend(["-rtsp_transport", "tcp"])

        ffmpeg_cmd.extend(["-i", input_url, "-map", "0:v:0", "-an"])

        if self.live_h264_encode:
            ffmpeg_cmd.extend([
                "-c:v",
                "libx264",
                "-preset",
                RECORDING_ENCODE_PRESET,
                "-crf",
                str(RECORDING_CRF),
                "-pix_fmt",
                "yuv420p",
            ])
        else:
            ffmpeg_cmd.extend(["-c:v", "copy"])

        ffmpeg_cmd.extend(["-f", "mp4", self.raw_path])

        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.is_recording = True
        self.session_start = datetime.now()
        print(
            f"[{self.camera_slug}/{self.activity}] Recording started → {self.final_path}"
        )

    def _finalize_recording_file(
        self,
        raw_path: str,
        final_path: str,
        *,
        already_h264: bool = False,
    ) -> str | None:
        if not is_valid_media(raw_path):
            print(
                f"[{self.camera_slug}/{self.activity}] Recording Failed: "
                f"raw file invalid or empty"
            )
            return None

        if already_h264 and remux_with_faststart(raw_path, final_path):
            return final_path

        if transcode_recording(raw_path, final_path):
            return final_path

        print(
            f"[{self.camera_slug}/{self.activity}] Transcode failed; "
            f"attempting remux fallback"
        )
        remux_cmd = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-loglevel",
            "warning",
            "-i",
            raw_path,
            "-an",
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            final_path,
        ]
        result = subprocess.run(
            remux_cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        if result.returncode == 0 and is_valid_media(final_path):
            return final_path

        print(
            f"[{self.camera_slug}/{self.activity}] Recording Failed: "
            f"could not produce playable MP4"
        )
        return None

    def _stop_recording(self):
        if not self.is_recording:
            return

        proc = self.ffmpeg_proc
        raw_path = self.raw_path
        final_path = self.final_path
        session_start = self.session_start
        live_h264_encode = self.live_h264_encode
        temp_dir = self._temp_dir
        self.is_recording = False
        self.ffmpeg_proc = None
        self.raw_path = None
        self.final_path = None
        self.session_start = None
        self.live_h264_encode = False
        self._temp_dir = None

        try:
            if proc:
                exit_code = stop_ffmpeg_gracefully(proc)
                if exit_code != 0:
                    stderr = ""
                    if proc.stderr:
                        try:
                            stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
                        except Exception:
                            pass
                    if stderr:
                        print(
                            f"[{self.camera_slug}/{self.activity}] "
                            f"FFmpeg exited with code {exit_code}: {stderr[-500:]}"
                        )

            time.sleep(0.5)

            playable_path = None
            if raw_path and final_path and session_start:
                playable_path = self._finalize_recording_file(
                    raw_path,
                    final_path,
                    already_h264=live_h264_encode,
                )

            if not playable_path or not session_start:
                if final_path and os.path.exists(final_path):
                    try:
                        os.remove(final_path)
                    except OSError:
                        pass
                return

            event_end = datetime.now()
            duration_seconds = (event_end - session_start).total_seconds()
            duration_minutes = round(duration_seconds / 60, 1)
            if duration_seconds < self.min_recording_duration_sec:
                print(
                    f"[{self.camera_slug}/{self.activity}] Recording skipped: "
                    f"duration {duration_minutes} min < "
                    f"{self.min_recording_duration_sec / 60:.1f} min minimum"
                )
                try:
                    os.remove(playable_path)
                except OSError:
                    pass
                return

            self.on_finalize(
                self.camera_uuid,
                self.camera_slug,
                self.activity,
                session_start.isoformat(),
                event_end.isoformat(),
                duration_minutes,
                playable_path,
            )
            print(f"[{self.camera_slug}/{self.activity}] Recording Finished")
        finally:
            _cleanup_temp_dir(temp_dir)


def _cleanup_temp_dir(temp_dir: str | None):
    if not temp_dir:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)


def _cleanup_orphan_raw_files(recordings_dir: str):
    if not recordings_dir or not os.path.isdir(recordings_dir):
        return
    removed = 0
    for root, _, files in os.walk(recordings_dir):
        for name in files:
            if not name.endswith("_raw.mp4"):
                continue
            try:
                os.remove(os.path.join(root, name))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"Removed {removed} orphan raw recording file(s) from {recordings_dir}")


class RecordingManager:
    def __init__(
        self,
        go2rtc_url: str,
        recordings_dir: str,
        edge_store,
        s3_client,
        s3_bucket: str,
        s3_endpoint: str,
        s3_access_key: str | None = None,
    ):
        self.go2rtc_url = go2rtc_url
        self.recordings_dir = recordings_dir
        self.edge_store = edge_store
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint.rstrip("/")
        self.s3_access_key = s3_access_key
        self.sessions: dict[str, CameraRecordingSession] = {}
        self.lock = threading.Lock()
        _cleanup_orphan_raw_files(self.recordings_dir)

    def _upload_recording(self, local_path: str, camera_slug: str) -> str | None:
        basename = os.path.basename(local_path)
        if "_raw" in basename:
            print(f"[{camera_slug}] Recording upload skipped: raw file not allowed")
            return None
        if not is_valid_media(local_path):
            print(f"[{camera_slug}] Recording upload skipped: invalid media file")
            return None
        object_name = f"recordings/{camera_slug}/{basename}"
        try:
            return s3_upload_file(
                self.s3_client,
                local_path,
                self.s3_bucket,
                object_name,
                self.s3_endpoint,
                access_key=self.s3_access_key,
            )
        except Exception as exc:
            print(f"Recording upload error: {exc}")
            return None

    def _on_finalize(
        self,
        camera_uuid: str,
        camera_slug: str,
        activity: str,
        event_start: str,
        event_end: str,
        duration_minutes: float,
        local_path: str,
    ):
        recording_url = self._upload_recording(local_path, camera_slug)
        if not recording_url:
            print(f"[{camera_slug}] Failed to upload recording, keeping local file")
            return

        event_id = self.edge_store.save_recording_event(
            camera_uuid,
            activity,
            event_start,
            event_end,
            duration_minutes,
            recording_url,
        )
        print(f"[{camera_slug}/{activity}] Recording event #{event_id} queued for sync")

        if not is_dev_mode() and os.path.exists(local_path):
            os.remove(local_path)

    def get_session(
        self,
        camera_uuid: str,
        camera_name: str,
        activity: str,
        stream_url: str,
    ) -> CameraRecordingSession:
        camera_slug = camera_file_slug(camera_name)
        with self.lock:
            if camera_uuid not in self.sessions:
                self.sessions[camera_uuid] = CameraRecordingSession(
                    camera_uuid=camera_uuid,
                    camera_slug=camera_slug,
                    activity=activity,
                    stream_url=stream_url,
                    recordings_dir=self.recordings_dir,
                    post_buffer_sec=POST_BUFFER_SEC,
                    max_segment_sec=MAX_SEGMENT_SEC,
                    person_start_frames=PERSON_START_FRAMES,
                    min_recording_duration_sec=MIN_RECORDING_DURATION_SEC,
                    on_finalize=self._on_finalize,
                )
            else:
                session = self.sessions[camera_uuid]
                session.camera_slug = camera_slug
                session.activity = activity
                session.stream_url = stream_url
            return self.sessions[camera_uuid]

    def signal_activity(
        self,
        camera_uuid: str,
        camera_name: str,
        activity: str,
        active: bool,
        stream_url: str,
    ):
        session = self.get_session(camera_uuid, camera_name, activity, stream_url)
        session.signal_activity(active)

    def start_monitor(self):
        def loop():
            while True:
                with self.lock:
                    sessions = list(self.sessions.values())
                for session in sessions:
                    session.tick()
                time.sleep(1)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    def remove_camera(self, camera_uuid: str):
        with self.lock:
            session = self.sessions.pop(camera_uuid, None)
        if session and session.is_recording:
            session._stop_recording()
