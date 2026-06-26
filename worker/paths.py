import os
from pathlib import Path

from stream_resolver import is_dev_mode

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def resolve_storage_path(env_value: str | None, default_name: str) -> str:
    if is_dev_mode():
        return str(_DATA_DIR / default_name)
    if env_value:
        return os.path.abspath(env_value)
    return f"/app/data/{default_name}"
