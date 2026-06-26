import os
import re
from urllib.parse import urlparse, urlunparse

EDGE_DEV_MODE = os.getenv("EDGE_DEV_MODE", "").strip().lower() in ("1", "true", "yes")
GO2RTC_RTSP_PORT = int(os.getenv("GO2RTC_RTSP_PORT", "8558"))
GO2RTC_RTSP_HOST = os.getenv("GO2RTC_RTSP_HOST", "").strip()

# go2rtc RTSP, e.g. rtsp://100.123.150.54:8558/cam-12?video=all&audio=all
_GO2RTC_RTSP_PATH_RE = re.compile(r"^/[^/]+")


def is_dev_mode() -> bool:
    return EDGE_DEV_MODE


def is_go2rtc_rtsp_url(url: str) -> bool:
    if not url or not url.lower().startswith("rtsp://"):
        return False
    parsed = urlparse(url)
    port = parsed.port if parsed.port is not None else 554
    path = parsed.path or ""
    return port == GO2RTC_RTSP_PORT and bool(_GO2RTC_RTSP_PATH_RE.match(path))


def extract_go2rtc_stream_name(rtsp_url: str) -> str | None:
    name = (urlparse(rtsp_url).path or "").strip("/")
    return name or None


def normalize_go2rtc_rtsp_url(rtsp_url: str) -> str:
    parsed = urlparse(rtsp_url)
    host = GO2RTC_RTSP_HOST or parsed.hostname or "localhost"
    port = parsed.port if parsed.port is not None else GO2RTC_RTSP_PORT
    return urlunparse((
        parsed.scheme,
        f"{host}:{port}",
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def should_transcode_go2rtc_rtsp_to_h264(stream_url: str) -> bool:
    return is_dev_mode() and is_go2rtc_rtsp_url(stream_url)


def resolve_recording_stream(stream_url: str) -> tuple[str, bool]:
    if should_transcode_go2rtc_rtsp_to_h264(stream_url):
        normalized = normalize_go2rtc_rtsp_url(stream_url)
        print(f"[dev] go2rtc RTSP → H.264 live encode: {normalized}")
        return normalized, True
    return stream_url, False


def resolve_inference_stream(camera: dict, go2rtc_http_url: str) -> str | None:
    rtsp_url = (camera.get("rtsp_url") or "").strip()
    if rtsp_url:
        if should_transcode_go2rtc_rtsp_to_h264(rtsp_url):
            normalized = normalize_go2rtc_rtsp_url(rtsp_url)
            print(f"[dev] Inference via FFmpeg decode (go2rtc RTSP): {normalized}")
            return normalized
        return rtsp_url

    stream_url = (camera.get("stream_url") or "").strip()
    if stream_url:
        return stream_url

    camera_uuid = camera.get("camera_uuid", "")
    if camera_uuid:
        stream_name = f"{camera_uuid}_sub"
        return f"{go2rtc_http_url.rstrip('/')}/api/stream.mp4?src={stream_name}"

    return None
