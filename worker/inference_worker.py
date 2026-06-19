import contextlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import cv2
from ultralytics import YOLO

from db import EdgeStore
from recording_manager import RecordingManager

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

PERSON_MODEL = os.getenv("PERSON_MODEL", "yolov5s.pt")
VIOLATION_MODEL = os.getenv("VIOLATION_MODEL", "best.pt")
PERSON_CLASS = os.getenv("PERSON_CLASS", "person")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
VIOLATION_FRAME_INTERVAL = int(os.getenv("VIOLATION_FRAME_INTERVAL", "5"))


class InferenceManager:
    def __init__(
        self,
        go2rtc_url: str,
        edge_store: EdgeStore,
        recording_manager: RecordingManager,
        models_dir: str = "/app/models",
        image_dir: str = "/app/data/images",
        s3_client=None,
        s3_bucket: str | None = None,
        s3_endpoint: str | None = None,
    ):
        self.go2rtc_url = go2rtc_url.rstrip("/")
        self.edge_store = edge_store
        self.recording_manager = recording_manager
        self.models_dir = models_dir
        self.image_dir = image_dir
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_endpoint = (s3_endpoint or "").rstrip("/")
        self.models: dict[str, YOLO] = {}
        self.model_lock = threading.Lock()
        self.active_threads: dict[str, threading.Thread] = {}
        self.stop_flags: dict[str, threading.Event] = {}
        self.camera_configs: dict[str, str] = {}
        self.violation_cooldown: dict[str, float] = {}

    def _camera_config_key(self, camera: dict) -> str:
        return json.dumps(
            {
                "rtsp_url": camera.get("rtsp_url"),
                "stream_url": camera.get("stream_url"),
                "activity": camera.get("activity"),
                "alert": camera.get("alert"),
            },
            sort_keys=True,
            default=str,
        )

    def _resolve_inference_stream(self, camera: dict) -> str | None:
        rtsp_url = (camera.get("rtsp_url") or "").strip()
        if rtsp_url:
            return rtsp_url

        stream_url = (camera.get("stream_url") or "").strip()
        if stream_url:
            return stream_url

        camera_uuid = camera.get("camera_uuid", "")
        if camera_uuid:
            stream_name = f"{camera_uuid}_sub"
            return f"{self.go2rtc_url}/api/stream.mp4?src={stream_name}"

        return None

    @contextlib.contextmanager
    def _suppress_ffmpeg_stderr(self):
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
            for _ in range(90):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
        return cap

    def _resolve_model_path(self, model_name: str) -> str:
        stem = (model_name or "yolov5s").strip()
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

    def _detect_class(
        self,
        model: YOLO,
        result,
        target_class: str,
        min_confidence: float,
    ) -> tuple[bool, list[float] | None]:
        normalized_target = target_class.strip().lower()
        names = model.names or {}

        for box in result.boxes:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            if confidence < min_confidence:
                continue

            cls_id = int(box.cls[0])
            class_name = str(names.get(cls_id, cls_id)).lower()
            if class_name == normalized_target:
                xyxy = box.xyxy[0].tolist()
                return True, [float(v) for v in xyxy]

        return False, None

    def _detect_violation(
        self,
        model: YOLO,
        result,
        min_confidence: float,
    ) -> tuple[bool, str | None, list[float] | None]:
        names = model.names or {}
        best_conf = 0.0
        best_name = None
        best_bbox = None

        for box in result.boxes:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            if confidence < min_confidence:
                continue

            cls_id = int(box.cls[0])
            class_name = str(names.get(cls_id, cls_id))
            if confidence > best_conf:
                best_conf = confidence
                best_name = class_name
                best_bbox = [float(v) for v in box.xyxy[0].tolist()]

        if best_name:
            return True, best_name, best_bbox
        return False, None, None

    def _upload_snapshot(self, local_path: str, camera_uuid: str) -> str | None:
        if not self.s3_client or not self.s3_bucket:
            return local_path
        object_name = f"alerts/{camera_uuid}/{os.path.basename(local_path)}"
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, object_name)
            return f"{self.s3_endpoint}/{self.s3_bucket}/{object_name}"
        except Exception as exc:
            print(f"Snapshot upload error: {exc}")
            return local_path

    def _save_violation_alert(
        self,
        camera_uuid: str,
        violation_name: str,
        bbox: list[float],
        frame,
    ):
        now = time.time()
        last = self.violation_cooldown.get(camera_uuid, 0)
        if now - last < 10:
            return
        self.violation_cooldown[camera_uuid] = now

        os.makedirs(self.image_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        local_path = os.path.join(
            self.image_dir,
            f"{camera_uuid}_{violation_name}_{timestamp}.jpg",
        )
        cv2.imwrite(local_path, frame)
        image_url = self._upload_snapshot(local_path, camera_uuid) or local_path

        self.edge_store.save_alert(
            camera_id=camera_uuid,
            violation_name=violation_name,
            severity="high",
            bbox=bbox,
            image_url=image_url,
            detected_at=datetime.now(timezone.utc).isoformat(),
        )
        print(f"[{camera_uuid}] Violation queued: {violation_name}")

    def _process_camera(self, camera: dict, stop_event: threading.Event):
        camera_uuid = camera["camera_uuid"]
        label = camera.get("name", camera_uuid)
        activity = (camera.get("activity") or "").strip()
        alert_enabled = bool(camera.get("alert"))
        stream_url = self._resolve_inference_stream(camera)

        if not activity:
            print(f"[{label}] No activity configured, skipping inference")
            return

        if not stream_url:
            print(f"[{label}] No rtsp_url configured, skipping inference")
            return

        person_model = self._get_model(PERSON_MODEL)
        violation_model = self._get_model(VIOLATION_MODEL) if alert_enabled else None

        print(
            f"[{label}] Inference connecting to {stream_url} "
            f"(activity={activity}, alert={alert_enabled})"
        )
        cap = self._open_capture(stream_url)
        consecutive_failures = 0
        frame_index = 0

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
            frame_index += 1

            with self.model_lock:
                person_results = person_model(frame, conf=MIN_CONFIDENCE, verbose=False)

            person_detected = False
            for result in person_results:
                detected, _ = self._detect_class(
                    person_model, result, PERSON_CLASS, MIN_CONFIDENCE
                )
                if detected:
                    person_detected = True
                    break

            self.recording_manager.signal_activity(
                camera_uuid, activity, person_detected, stream_url
            )

            if alert_enabled and violation_model and frame_index % VIOLATION_FRAME_INTERVAL == 0:
                with self.model_lock:
                    violation_results = violation_model(
                        frame, conf=MIN_CONFIDENCE, verbose=False
                    )
                for result in violation_results:
                    detected, violation_name, bbox = self._detect_violation(
                        violation_model, result, MIN_CONFIDENCE
                    )
                    if detected and violation_name and bbox:
                        self._save_violation_alert(
                            camera_uuid, violation_name, bbox, frame
                        )
                        break

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
