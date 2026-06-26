import re

_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\s]+')


def sanitize_filename_part(value: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", (value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned


def camera_file_slug(name: str | None) -> str:
    slug = sanitize_filename_part(name or "")
    return slug or "camera"
