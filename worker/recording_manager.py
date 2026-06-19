import os
import signal
import subprocess
import threading
import time
from datetime import datetime


class CameraRecordingSession:
    def __init__(
        self,
        camera_uuid: str,
        activity: str,
        stream_url: str,
        recordings_dir: str,
        post_buffer_sec: int,
        max_segment_sec: int,
        on_finalize,
    ):
        self.camera_uuid = camera_uuid
        self.activity = activity
        self.stream_url = stream_url
        self.recordings_dir = recordings_dir
        self.post_buffer_sec = post_buffer_sec
        self.max_segment_sec = max_segment_sec
        self.on_finalize = on_finalize
        self.lock = threading.Lock()
        self.is_recording = False
        self.last_activity_at = 0.0
        self.session_start: datetime | None = None
        self.ffmpeg_proc: subprocess.Popen | None = None
        self.output_path: str | None = None

        os.makedirs(self.recordings_dir, exist_ok=True)

    def signal_activity(self, active: bool):
        now = time.time()
        with self.lock:
            if active:
                self.last_activity_at = now
                if not self.is_recording:
                    self._start_recording()
            elif self.is_recording and now - self.last_activity_at >= self.post_buffer_sec:
                self._stop_recording()

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
        camera_dir = os.path.join(self.recordings_dir, self.camera_uuid)
        os.makedirs(camera_dir, exist_ok=True)
        self.output_path = os.path.join(
            camera_dir,
            f"{self.camera_uuid}_{self.activity}_{timestamp}.mp4",
        )

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
        ]

        if self.stream_url.startswith("rtsp://"):
            ffmpeg_cmd.extend(["-rtsp_transport", "tcp"])

        ffmpeg_cmd.extend([
            "-i",
            self.stream_url,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            self.output_path,
        ])

        self.ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
        )
        self.is_recording = True
        self.session_start = datetime.now()
        print(
            f"[{self.camera_uuid}/{self.activity}] Recording started → {self.output_path}"
        )

    def _stop_recording(self):
        if not self.is_recording:
            return

        proc = self.ffmpeg_proc
        output_path = self.output_path
        session_start = self.session_start
        self.is_recording = False
        self.ffmpeg_proc = None
        self.output_path = None
        self.session_start = None

        if proc:
            try:
                if proc.stdin:
                    try:
                        proc.stdin.write(b"q")
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                proc.wait(timeout=10)
            except Exception:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

        time.sleep(1)

        success = False
        if output_path and session_start and os.path.exists(output_path):
            if os.path.getsize(output_path) > 0:
                event_end = datetime.now()
                duration_minutes = (event_end - session_start).total_seconds() / 60
                self.on_finalize(
                    self.camera_uuid,
                    self.activity,
                    session_start.isoformat(),
                    event_end.isoformat(),
                    duration_minutes,
                    output_path,
                )
                success = True
            else:
                print(
                    f"[{self.camera_uuid}/{self.activity}] Recording Failed: File is empty (0 bytes)"
                )
        else:
            print(
                f"[{self.camera_uuid}/{self.activity}] Recording Failed: File not found at {output_path}"
            )

        if success:
            print(f"[{self.camera_uuid}/{self.activity}] Recording Finished")
        elif output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


class RecordingManager:
    def __init__(
        self,
        go2rtc_url: str,
        recordings_dir: str,
        edge_store,
        s3_client,
        s3_bucket: str,
        s3_endpoint: str,
    ):
        self.go2rtc_url = go2rtc_url
        self.recordings_dir = recordings_dir
        self.edge_store = edge_store
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint.rstrip("/")
        self.sessions: dict[str, CameraRecordingSession] = {}
        self.lock = threading.Lock()

    def _upload_recording(self, local_path: str, camera_uuid: str) -> str | None:
        object_name = f"recordings/{camera_uuid}/{os.path.basename(local_path)}"
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, object_name)
            return f"{self.s3_endpoint}/{self.s3_bucket}/{object_name}"
        except Exception as exc:
            print(f"Recording upload error: {exc}")
            return None

    def _on_finalize(
        self,
        camera_uuid: str,
        activity: str,
        event_start: str,
        event_end: str,
        duration_minutes: float,
        local_path: str,
    ):
        recording_url = self._upload_recording(local_path, camera_uuid)
        if not recording_url:
            print(f"[{camera_uuid}] Failed to upload recording, keeping local file")
            return

        event_id = self.edge_store.save_recording_event(
            camera_uuid,
            activity,
            event_start,
            event_end,
            duration_minutes,
            recording_url,
        )
        print(f"[{camera_uuid}/{activity}] Recording event #{event_id} queued for sync")

        if os.path.exists(local_path):
            os.remove(local_path)

    def get_session(self, camera_uuid: str, activity: str, stream_url: str) -> CameraRecordingSession:
        with self.lock:
            if camera_uuid not in self.sessions:
                self.sessions[camera_uuid] = CameraRecordingSession(
                    camera_uuid=camera_uuid,
                    activity=activity,
                    stream_url=stream_url,
                    recordings_dir=self.recordings_dir,
                    post_buffer_sec=10,
                    max_segment_sec=300,
                    on_finalize=self._on_finalize,
                )
            return self.sessions[camera_uuid]

    def signal_activity(self, camera_uuid: str, activity: str, active: bool, stream_url: str):
        session = self.get_session(camera_uuid, activity, stream_url)
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
