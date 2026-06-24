"""BeatNet baseline checkpoint resources."""

from __future__ import annotations

from pathlib import Path

from .._resources import checkpoint_names, checkpoint_path, immutable_mapping


BASELINE_CHECKPOINT_FILENAMES = immutable_mapping(
    {
        "baseline": "model_1_weights.pt",
        "baseline_alt0": "baseline_alt0.pt",
        "baseline_alt1": "baseline_alt1.pt",
    }
)
BASELINE_CHECKPOINT_SHA256 = immutable_mapping(
    {
        "baseline": "619091bc317ca3e83b45591d46f6de3d5a41588bcb39fe9fe7be30cffa6aca84",
        "baseline_alt0": "5878a18c079fa0b0139879b14ed2b5b7595faef8c3d16210aed141fd00fa2d58",
        "baseline_alt1": "0c52a074ea38e8cb4a760ecfa3747c9cf91a1e3cd19f238eed80b0de763989ca",
    }
)

BASE_CHECKPOINT_FILENAME = BASELINE_CHECKPOINT_FILENAMES["baseline"]
BASE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256["baseline"]


def baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BeatNet baseline checkpoint selector names."""
    return checkpoint_names(BASELINE_CHECKPOINT_FILENAMES)


def baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a packaged BeatNet baseline checkpoint path."""
    return checkpoint_path(__package__, BASELINE_CHECKPOINT_FILENAMES, name)


def base_checkpoint_path() -> Path:
    """Return the packaged BeatNet model-1 baseline checkpoint."""
    return baseline_checkpoint_path()


__all__ = [
    "BASE_CHECKPOINT_FILENAME",
    "BASE_CHECKPOINT_SHA256",
    "BASELINE_CHECKPOINT_FILENAMES",
    "BASELINE_CHECKPOINT_SHA256",
    "base_checkpoint_path",
    "baseline_checkpoint_names",
    "baseline_checkpoint_path",
]
