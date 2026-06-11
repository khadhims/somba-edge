import threading
import time

import socketio

from db import EdgeStore
from orchestrator import CameraOrchestrator
from inference_worker import InferenceManager


class EdgeWorkerClient:
    def __init__(
        self,
        server_ws_url: str,
        edge_api_key: str,
        edge_store: EdgeStore,
        orchestrator: CameraOrchestrator,
        inference_manager: InferenceManager,
        sync_interval: int = 5,
        batch_size: int = 50,
    ):
        self.server_ws_url = server_ws_url
        self.edge_api_key = edge_api_key
        self.edge_store = edge_store
        self.orchestrator = orchestrator
        self.inference_manager = inference_manager
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=2,
        )
        self._register_handlers()

    def _register_handlers(self):
        @self.sio.event(namespace="/edge")
        def connect():
            print("Connected to server WebSocket")

        @self.sio.event(namespace="/edge")
        def disconnect():
            print("Disconnected from server WebSocket")

        @self.sio.on("config_updated", namespace="/edge")
        def on_config_updated(_payload):
            print("Received config_updated from server")
            self.bootstrap()

    def bootstrap(self):
        cameras = self.orchestrator.sync_cameras()
        self.inference_manager.sync_workers(cameras)

    def connect(self):
        self.sio.connect(
            self.server_ws_url,
            namespaces=["/edge"],
            headers={
                "Authorization": f"Bearer {self.edge_api_key}",
                "X-Edge-API-Key": self.edge_api_key,
            },
            auth={"token": self.edge_api_key},
            transports=["websocket", "polling"],
            wait_timeout=20,
        )
        self.bootstrap()

    def _sync_pending_recording_events(self):
        pending = self.edge_store.fetch_pending_recording_events(self.batch_size)
        for item in pending:
            payload = {k: v for k, v in item.items() if v is not None}
            try:
                response = self.sio.call(
                    "recording_event",
                    payload,
                    namespace="/edge",
                    timeout=30,
                )
            except Exception as exc:
                print(f"Recording sync failed for #{item['edge_id']}: {exc}")
                continue

            if response and response.get("status") == "success":
                self.edge_store.mark_recording_events_sent([item["edge_id"]])
                print(f"Recording event #{item['edge_id']} acknowledged by server")
            else:
                print(
                    f"Recording event #{item['edge_id']} rejected: "
                    f"{response.get('reason') if response else 'no response'}"
                )

    def _sync_pending_alerts(self):
        pending = self.edge_store.fetch_pending_alerts(self.batch_size)
        for item in pending:
            payload = {k: v for k, v in item.items() if v is not None}
            try:
                response = self.sio.call(
                    "detection_event",
                    payload,
                    namespace="/edge",
                    timeout=20,
                )
            except Exception as exc:
                print(f"Alert sync failed for #{item['edge_id']}: {exc}")
                continue

            if response and response.get("status") == "success":
                self.edge_store.mark_alerts_sent([item["edge_id"]])
                print(f"Alert #{item['edge_id']} acknowledged by server")
            else:
                print(
                    f"Alert #{item['edge_id']} rejected: "
                    f"{response.get('reason') if response else 'no response'}"
                )

    def start_sync_loop(self):
        def loop():
            while True:
                if self.sio.connected:
                    try:
                        self._sync_pending_recording_events()
                        self._sync_pending_alerts()
                    except Exception as exc:
                        print(f"Sync loop error: {exc}")
                time.sleep(self.sync_interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    def wait(self):
        while True:
            time.sleep(60)
