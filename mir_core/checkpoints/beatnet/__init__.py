"""BeatNet baseline checkpoint resource."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


BASE_CHECKPOINT_FILENAME = "model_1_weights.pt"
BASE_CHECKPOINT_SHA256 = "619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84"


def base_checkpoint_path() -> Path:
    """Return the packaged BeatNet model-1 baseline checkpoint."""
    resource = files(__package__).joinpath(BASE_CHECKPOINT_FILENAME)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged BeatNet checkpoint is missing: {resource}")
    return Path(str(resource))


__all__ = [
    "BASE_CHECKPOINT_FILENAME",
    "BASE_CHECKPOINT_SHA256",
    "base_checkpoint_path",
]
