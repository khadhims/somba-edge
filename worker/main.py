import os

import boto3
from dotenv import load_dotenv

from db import EdgeStore
from inference_worker import InferenceManager
from orchestrator import CameraOrchestrator
from recording_manager import RecordingManager
from ws_client import EdgeWorkerClient

load_dotenv()

SERVER_API_URL = os.getenv("SERVER_API_URL", "http://localhost:3000")
SERVER_WS_URL = os.getenv("SERVER_WS_URL", "http://localhost:3000")
EDGE_API_KEY = os.getenv("EDGE_API_KEY", "")
GO2RTC_URL = os.getenv("GO2RTC_URL", "http://go2rtc:1984")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
DB_PATH = os.getenv("DB_PATH", "/app/data/db/edge.db")
IMAGE_DIR = os.getenv("IMAGE_DIR", "/app/data/images")
RECORDINGS_PATH = os.getenv("RECORDINGS_PATH", "/app/data/recordings")
SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")


def main():
    if not EDGE_API_KEY:
        raise RuntimeError("EDGE_API_KEY is required")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(RECORDINGS_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
    )

    edge_store = EdgeStore(DB_PATH)
    orchestrator = CameraOrchestrator(
        server_api_url=SERVER_API_URL,
        edge_api_key=EDGE_API_KEY,
        edge_store=edge_store,
    )
    recording_manager = RecordingManager(
        go2rtc_url=GO2RTC_URL,
        recordings_dir=RECORDINGS_PATH,
        edge_store=edge_store,
        s3_client=s3_client,
        s3_bucket=S3_BUCKET,
        s3_endpoint=S3_ENDPOINT,
    )
    recording_manager.start_monitor()

    inference_manager = InferenceManager(
        go2rtc_url=GO2RTC_URL,
        edge_store=edge_store,
        recording_manager=recording_manager,
        models_dir=MODELS_DIR,
        image_dir=IMAGE_DIR,
        s3_client=s3_client,
        s3_bucket=S3_BUCKET,
        s3_endpoint=S3_ENDPOINT,
    )
    worker = EdgeWorkerClient(
        server_ws_url=SERVER_WS_URL,
        edge_api_key=EDGE_API_KEY,
        edge_store=edge_store,
        orchestrator=orchestrator,
        inference_manager=inference_manager,
        sync_interval=SYNC_INTERVAL,
        batch_size=BATCH_SIZE,
    )

    worker.connect()
    worker.start_sync_loop()
    worker.wait()


if __name__ == "__main__":
    main()
