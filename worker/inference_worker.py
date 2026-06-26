import contextlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import cv2
import torch
from ultralytics import YOLO

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# Limit per-inference CPU threads so that many camera worker threads parallelise
# across cores instead of each fighting for all of them (e.g. 12 cameras on a
# 12-core CPU → ~1 core per stream). Tune via INFERENCE_THREADS.
INFERENCE_THREADS = int(os.getenv("INFERENCE_THREADS", "1"))
if INFERENCE_THREADS > 0:
    torch.set_num_threads(INFERENCE_THREADS)
    cv2.setNumThreads(INFERENCE_THREADS)

from db import EdgeStore
from dev_capture import DevFfmpegCapture
from recording_manager import RecordingManager
from filename_utils import camera_file_slug, sanitize_filename_part
from image_utils import annotate_frame
from s3_upload import upload_file as s3_upload_file
from stream_resolver import resolve_inference_stream, should_transcode_go2rtc_rtsp_to_h264
from violation_episode import ViolationEpisodeTracker

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
)

PERSON_MODEL = os.getenv("PERSON_MODEL", "yolov5s.pt")
VIOLATION_MODEL = os.getenv("VIOLATION_MODEL", "best.pt")
PERSON_CLASS = os.getenv("PERSON_CLASS", "person")
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.5"))
VIOLATION_MIN_CONFIDENCE = float(os.getenv("VIOLATION_MIN_CONFIDENCE", "0.5"))
VIOLATION_FRAME_INTERVAL = int(os.getenv("VIOLATION_FRAME_INTERVAL", "1"))
VIOLATION_EPISODE_START_FRAMES = int(os.getenv("VIOLATION_EPISODE_START_FRAMES", "15"))
VIOLATION_EPISODE_START_SEC = float(os.getenv("VIOLATION_EPISODE_START_SEC", "0.5"))
VIOLATION_EPISODE_END_SEC = float(os.getenv("VIOLATION_EPISODE_END_SEC", "5"))
VIOLATION_EPISODE_MISS_GRACE_FRAMES = int(
    os.getenv("VIOLATION_EPISODE_MISS_GRACE_FRAMES", "5")
)
VIOLATION_EPISODE_COOLDOWN_SEC = float(os.getenv("VIOLATION_EPISODE_COOLDOWN_SEC", "10"))
VIOLATION_CLASSES = tuple(
    int(value.strip())
    for value in os.getenv("VIOLATION_CLASSES", "0,1,2").split(",")
    if value.strip() != ""
)
WORKER_STAGGER_SEC = float(os.getenv("WORKER_STAGGER_SEC", "2"))


def _log(message: str):
    print(message, flush=True)


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
        s3_access_key: str | None = None,
    ):
        self.go2rtc_url = go2rtc_url.rstrip("/")
        self.edge_store = edge_store
        self.recording_manager = recording_manager
        self.models_dir = os.path.abspath(models_dir)
        self.image_dir = image_dir
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_endpoint = (s3_endpoint or "").rstrip("/")
        self.s3_access_key = s3_access_key
        self.models: dict[str, YOLO] = {}
        self._load_lock = threading.RLock()
        self._models_ready = threading.Event()
        self.active_threads: dict[str, threading.Thread] = {}
        self.stop_flags: dict[str, threading.Event] = {}
        self.camera_configs: dict[str, str] = {}

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
        return resolve_inference_stream(camera, self.go2rtc_url)

    def _open_stream_capture(self, stream_url: str):
        if should_transcode_go2rtc_rtsp_to_h264(stream_url):
            capture = DevFfmpegCapture(stream_url)
            capture.open()
            return capture
        return self._open_capture(stream_url)

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

        local_path = os.path.abspath(os.path.join(self.models_dir, f"{stem}.pt"))
        if os.path.isfile(local_path):
            return local_path

        # Prefer Ultralytics-compatible yolov5su when yolov5s was requested.
        if stem == "yolov5s":
            alt_path = os.path.abspath(os.path.join(self.models_dir, "yolov5su.pt"))
            if os.path.isfile(alt_path):
                _log(f"Using {alt_path} instead of missing {local_path}")
                return alt_path

        available = sorted(
            name for name in os.listdir(self.models_dir) if name.endswith(".pt")
        ) if os.path.isdir(self.models_dir) else []
        raise FileNotFoundError(
            f"Model file not found at {local_path}. "
            f"Available in {self.models_dir}: {', '.join(available) or '(none)'}"
        )

    def _load_model(self, model_name: str) -> YOLO:
        model_path = self._resolve_model_path(model_name)
        if model_path in self.models:
            return self.models[model_path]

        with self._load_lock:
            if model_path in self.models:
                return self.models[model_path]
            _log(f"Loading model: {model_path}")
            started = time.time()
            self.models[model_path] = YOLO(model_path)
            _log(f"Model ready: {model_path} ({time.time() - started:.1f}s)")
            return self.models[model_path]

    def _get_model(self, model_name: str) -> YOLO:
        if not self._models_ready.is_set():
            self.warmup_models()
        model_path = self._resolve_model_path(model_name)
        if model_path not in self.models:
            return self._load_model(model_name)
        return self.models[model_path]

    def warmup_models(self):
        if self._models_ready.is_set():
            return
        with self._load_lock:
            if self._models_ready.is_set():
                return
            _log(f"Warming up models from {self.models_dir}")
            self._load_model(PERSON_MODEL)
            try:
                self._load_model(VIOLATION_MODEL)
            except FileNotFoundError as exc:
                _log(f"Skipping violation model warmup: {exc}")
            self._models_ready.set()
            _log("Model warmup complete")

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
    ) -> tuple[bool, str | None, list[float] | None, float]:
        names = model.names or {}
        best_conf = 0.0
        best_name = None
        best_bbox = None

        for box in result.boxes:
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            if confidence < min_confidence:
                continue

            cls_id = int(box.cls[0])
            if cls_id not in VIOLATION_CLASSES:
                continue

            class_name = str(names.get(cls_id, cls_id))
            if class_name.lower().startswith("yes"):
                continue

            if confidence > best_conf:
                best_conf = confidence
                best_name = class_name
                best_bbox = [float(v) for v in box.xyxy[0].tolist()]

        if best_name:
            return True, best_name, best_bbox, best_conf
        return False, None, None, 0.0

    def _upload_snapshot(self, local_path: str, camera_slug: str) -> str | None:
        if not self.s3_client or not self.s3_bucket:
            return local_path
        object_name = f"alerts/{camera_slug}/{os.path.basename(local_path)}"
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
            print(f"Snapshot upload error: {exc}")
            return None

    def _save_violation_alert(
        self,
        camera_uuid: str,
        camera_name: str,
        violation_name: str,
        bbox: list[float],
        frame,
        detected_at: str,
        event_start: str,
        event_end: str,
        total_detections: int,
    ):
        camera_slug = camera_file_slug(camera_name)
        violation_slug = sanitize_filename_part(violation_name) or "violation"
        camera_dir = os.path.join(self.image_dir, camera_slug)
        os.makedirs(camera_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        annotated = annotate_frame(frame, bbox, violation_name)
        local_path = os.path.join(
            camera_dir,
            f"{camera_slug}_{violation_slug}_{timestamp}.jpg",
        )
        if not cv2.imwrite(local_path, annotated):
            print(f"[{camera_slug}] Failed to write snapshot: {local_path}")
            return

        image_url = self._upload_snapshot(local_path, camera_slug)
        if not image_url:
            print(f"[{camera_slug}] Snapshot upload failed, alert skipped")
            try:
                os.remove(local_path)
            except OSError:
                pass
            return

        self.edge_store.save_alert(
            camera_id=camera_uuid,
            violation_name=violation_name,
            severity="high",
            bbox=bbox,
            image_url=image_url,
            detected_at=detected_at,
            event_start=event_start,
            event_end=event_end,
            total_detections=total_detections,
        )
        print(
            f"[{camera_slug}] Violation episode queued: {violation_name} "
            f"({total_detections} detections)"
        )

    def _emit_violation_episode(self, camera_uuid: str, camera_name: str, episode: dict):
        self._save_violation_alert(
            camera_uuid,
            camera_name,
            episode["violation_name"],
            episode["bbox"],
            episode["frame"],
            episode["detected_at"],
            episode["event_start"],
            episode["event_end"],
            episode["total_detections"],
        )

    def _process_camera(self, camera: dict, stop_event: threading.Event):
        camera_uuid = camera["camera_uuid"]
        camera_name = (camera.get("name") or "").strip()
        label = camera_name or camera_uuid
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
        violation_model = None
        if alert_enabled:
            try:
                violation_model = self._get_model(VIOLATION_MODEL)
            except FileNotFoundError as exc:
                _log(f"[{label}] {exc}")

        _log(
            f"[{label}] Inference connecting to {stream_url} "
            f"(activity={activity}, alert={alert_enabled})"
        )
        cap = self._open_stream_capture(stream_url)
        if should_transcode_go2rtc_rtsp_to_h264(stream_url):
            _log(f"[{label}] Dev FFmpeg capture started")
        consecutive_failures = 0
        frame_index = 0
        violation_episode = ViolationEpisodeTracker(
            start_frames=VIOLATION_EPISODE_START_FRAMES,
            start_seconds=VIOLATION_EPISODE_START_SEC,
            end_seconds=VIOLATION_EPISODE_END_SEC,
            miss_grace_frames=VIOLATION_EPISODE_MISS_GRACE_FRAMES,
            cooldown_seconds=VIOLATION_EPISODE_COOLDOWN_SEC,
        )

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
                cap = self._open_stream_capture(stream_url)
                continue

            consecutive_failures = 0
            frame_index += 1
            if frame_index == 1:
                _log(f"[{label}] First frame received")

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
                camera_uuid, camera_name, activity, person_detected, stream_url
            )

            run_violation = (
                alert_enabled
                and violation_model
                and frame_index % VIOLATION_FRAME_INTERVAL == 0
            )

            episode_payload = None
            if run_violation and person_detected:
                violation_results = violation_model(
                    frame,
                    conf=VIOLATION_MIN_CONFIDENCE,
                    classes=list(VIOLATION_CLASSES),
                    verbose=False,
                )

                detected = False
                violation_name = None
                bbox = None
                confidence = 0.0
                for result in violation_results:
                    detected, violation_name, bbox, confidence = self._detect_violation(
                        violation_model, result, VIOLATION_MIN_CONFIDENCE
                    )
                    if detected:
                        break

                episode_payload = violation_episode.observe(
                    detected, violation_name, bbox, confidence, frame
                )
            elif run_violation:
                # No person in frame → no possible violation. Skip the violation
                # model entirely and record a miss so the episode end-timer still
                # advances.
                episode_payload = violation_episode.observe(
                    False, None, None, 0.0, frame
                )
            elif alert_enabled:
                episode_payload = violation_episode.check_end()

            if episode_payload:
                self._emit_violation_episode(camera_uuid, camera_name, episode_payload)

            time.sleep(0.01)

        episode_payload = violation_episode.flush()
        if episode_payload:
            self._emit_violation_episode(camera_uuid, camera_name, episode_payload)

        cap.release()
        self.recording_manager.remove_camera(camera_uuid)

    def sync_workers(self, cameras: list[dict]):
        self.warmup_models()

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
            _log(f"Started inference worker for {camera.get('name', camera_uuid)}")
            if WORKER_STAGGER_SEC > 0:
                time.sleep(WORKER_STAGGER_SEC)
