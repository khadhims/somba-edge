import requests

from db import EdgeStore


class CameraOrchestrator:
    def __init__(
        self,
        server_api_url: str,
        edge_api_key: str,
        edge_store: EdgeStore,
    ):
        self.server_api_url = server_api_url.rstrip("/")
        self.edge_api_key = edge_api_key
        self.edge_store = edge_store

    def _headers(self) -> dict:
        return {
            "X-Edge-API-Key": self.edge_api_key,
            "Content-Type": "application/json",
        }

    def fetch_cameras(self) -> list[dict]:
        response = requests.get(
            f"{self.server_api_url}/edge/cameras",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    def sync_cameras(self) -> list[dict]:
        cameras = self.fetch_cameras()
        if not cameras:
            print("No cameras assigned to this edge device")
            self.edge_store.delete_stale_master_cameras([])
            return []

        self.edge_store.upsert_master_cameras(cameras)
        active_uuids = [camera["camera_uuid"] for camera in cameras]
        self.edge_store.delete_stale_master_cameras(active_uuids)
        master = self.edge_store.list_master_cameras()
        print(f"Synced {len(master)} camera(s) to local master_cameras")
        return master
