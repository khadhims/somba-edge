import mimetypes
import os

_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


def resolve_api_bucket_name(bucket: str) -> str:
    """Bucket name for boto3 API calls (plain name, e.g. somba)."""
    bucket = (bucket or "").strip()
    if ":" in bucket:
        return bucket.split(":", 1)[1]
    return bucket


def resolve_public_bucket_segment(bucket: str, access_key: str | None = None) -> str:
    """Bucket segment in public Contabo URLs — plain path-style bucket name."""
    return resolve_api_bucket_name(bucket)


def resolve_public_base_url(
    endpoint: str,
    bucket: str,
    *,
    access_key: str | None = None,
) -> str:
    endpoint = (endpoint or "").rstrip("/")
    segment = resolve_public_bucket_segment(bucket, access_key)
    return f"{endpoint}/{segment}"


def build_public_url(public_base_url: str, object_key: str) -> str:
    return f"{public_base_url.rstrip('/')}/{object_key.lstrip('/')}"


def resolve_content_type(local_path: str) -> str:
    ext = os.path.splitext(local_path)[1].lower()
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    guessed, _ = mimetypes.guess_type(local_path)
    return guessed or "application/octet-stream"


def upload_file(
    s3_client,
    local_path: str,
    bucket: str,
    object_key: str,
    endpoint: str,
    *,
    access_key: str | None = None,
) -> str:
    api_bucket = resolve_api_bucket_name(bucket)
    public_base = resolve_public_base_url(
        endpoint,
        bucket,
        access_key=access_key,
    )
    content_type = resolve_content_type(local_path)
    s3_client.upload_file(
        local_path,
        api_bucket,
        object_key,
        ExtraArgs={"ContentType": content_type, "ACL": "public-read"},
    )
    return build_public_url(public_base, object_key)
