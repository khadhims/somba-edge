import contextlib
import json
import os
import sys
import threading
import time

# RTSP over TCP reduces HEVC decode errors from UDP packet loss (common on WSL/LAN).
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

import cv2
from ultralytics import YOLO

from db import EdgeStore
from recording_manager import RecordingManager


class InferenceManager:
    def __init__(
        self,
        go2rtc_url: str,
        edge_store: EdgeStore,
        recording_manager: RecordingManager,
        models_dir: str = "/app/models",
    ):
        self.go2rtc_url = go2rtc_url.rstrip("/")
        self.edge_store = edge_store
        self.recording_manager = recording_manager
        self.models_dir = models_dir
        self.models: dict[str, YOLO] = {}
        self.model_lock = threading.Lock()
        self.active_threads: dict[str, threading.Thread] = {}
        self.stop_flags: dict[str, threading.Event] = {}
        self.camera_configs: dict[str, str] = {}

    def _camera_config_key(self, camera: dict) -> str:
        return json.dumps(
            {
                "rtsp_url": camera.get("rtsp_url"),
                "activities": camera.get("activities"),
            },
            sort_keys=True,
            default=str,
        )

    def _resolve_inference_stream(self, camera: dict) -> str | None:
        rtsp_url = (camera.get("rtsp_url") or "").strip()
        if rtsp_url:
            return rtsp_url

        camera_uuid = camera.get("camera_uuid", "")
        if camera_uuid:
            stream_name = f"{camera_uuid}_sub"
            return f"{self.go2rtc_url}/api/stream.mp4?src={stream_name}"

        return None

    @contextlib.contextmanager
    def _suppress_ffmpeg_stderr(self):
        """FFmpeg prints HEVC warmup warnings to stderr; VLC hides the same noise."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            previous = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = previous

    def _open_capture(self, stream_url: str) -> cv2.VideoCapture:
        with self._suppress_ffmpeg_stderr():
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Discard frames until decoder catches a keyframe after connect.
            for _ in range(90):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
        return cap

    def _resolve_model_path(self, model_name: str) -> str:
        stem = (model_name or "yolov8n").strip()
        if stem.lower().endswith(".pt"):
            stem = stem[:-3]

        if os.path.isabs(stem) or "/" in stem or "\\" in stem:
            return stem if stem.endswith(".pt") else f"{stem}.pt"

        local_path = os.path.join(self.models_dir, f"{stem}.pt")
        if os.path.isfile(local_path):
            return local_path

        print(
            f"Model file not found at {local_path}; "
            f"falling back to Ultralytics name '{stem}.pt'"
        )
        return f"{stem}.pt"

    def _get_model(self, model_name: str) -> YOLO:
        model_path = self._resolve_model_path(model_name)
        with self.model_lock:
            if model_path not in self.models:
                print(f"Loading model: {model_path}")
                self.models[model_path] = YOLO(model_path)
            return self.models[model_path]

    def _matches_target_classes(
        self,
        model: YOLO,
        result,
        target_classes: list[str],
        min_confidence: float,
    ) -> bool:
        if not target_classes:
            return False

        normalized_targets = {name.strip().lower() for name in target_classes}
        names = model.names or {}

        for box in result.boxes:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            if confidence < min_confidence:
                continue

            cls_id = int(box.cls[0])
            class_name = str(names.get(cls_id, cls_id)).lower()
            if class_name in normalized_targets:
                return True

        return False

    def _process_camera(self, camera: dict, stop_event: threading.Event):
        camera_uuid = camera["camera_uuid"]
        label = camera.get("name", camera_uuid)
        activities = camera.get("activities") or []
        stream_url = self._resolve_inference_stream(camera)

        if not activities:
            print(f"[{label}] No activities assigned, skipping inference")
            return

        if not stream_url:
            print(f"[{label}] No rtsp_url configured, skipping inference")
            return

        print(f"[{label}] Inference connecting to {stream_url}")
        cap = self._open_capture(stream_url)
        consecutive_failures = 0

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                    print(
                        f"[{label}] Failed to read frame "
                        f"({consecutive_failures}x). Reconnecting..."
                    )
                cap.release()
                time.sleep(min(2 + consecutive_failures, 10))
                cap = self._open_capture(stream_url)
                continue

            consecutive_failures = 0

            for activity in activities:
                model_name = activity.get("ai_model") or "yolov8n"
                target_classes = activity.get("target_classes") or []
                min_confidence = float(activity.get("min_confidence", 0.5))

                model = self._get_model(model_name)
                with self.model_lock:
                    results = model(frame, conf=min_confidence, verbose=False)

                detected = False
                for result in results:
                    if self._matches_target_classes(
                        model, result, target_classes, min_confidence
                    ):
                        detected = True
                        break

                self.recording_manager.signal_activity(
                    camera_uuid, activity, detected, stream_url
                )

            time.sleep(0.01)

        cap.release()
        self.recording_manager.remove_camera(camera_uuid)

    def sync_workers(self, cameras: list[dict]):
        desired_ids = {camera["camera_uuid"] for camera in cameras}

        for camera_uuid in list(self.active_threads.keys()):
            if camera_uuid not in desired_ids:
                print(f"Stopping inference worker for {camera_uuid}")
                self.stop_flags[camera_uuid].set()
                self.active_threads.pop(camera_uuid, None)
                self.stop_flags.pop(camera_uuid, None)
                self.camera_configs.pop(camera_uuid, None)
                self.recording_manager.remove_camera(camera_uuid)

        for camera in cameras:
            camera_uuid = camera["camera_uuid"]
            config_key = self._camera_config_key(camera)

            if camera_uuid in self.active_threads:
                if self.camera_configs.get(camera_uuid) == config_key:
                    continue
                print(f"Restarting inference worker for {camera.get('name', camera_uuid)}")
                self.stop_flags[camera_uuid].set()
                self.active_threads.pop(camera_uuid, None)
                self.stop_flags.pop(camera_uuid, None)
                self.recording_manager.remove_camera(camera_uuid)

            self.camera_configs[camera_uuid] = config_key
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._process_camera,
                args=(camera, stop_event),
                daemon=True,
            )
            self.stop_flags[camera_uuid] = stop_event
            self.active_threads[camera_uuid] = thread
            thread.start()
            print(f"Started inference worker for {camera.get('name', camera_uuid)}")
