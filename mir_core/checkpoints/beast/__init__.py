"""BEAST baseline checkpoint resources."""

from __future__ import annotations

from pathlib import Path

from .._resources import checkpoint_names, checkpoint_path, immutable_mapping


BASELINE_CHECKPOINT_FILENAMES = immutable_mapping(
    {
        "baseline": "baseline.pt",
    }
)
BASELINE_CHECKPOINT_SHA256 = immutable_mapping(
    {
        "baseline": "e77f272f1c1f68521958d3947b1d7b72616fda2281755417b96dd6d3f7ddddc4",
    }
)

BASE_CHECKPOINT_FILENAME = BASELINE_CHECKPOINT_FILENAMES["baseline"]
BASE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256["baseline"]


def baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BEAST baseline checkpoint selector names."""
    return checkpoint_names(BASELINE_CHECKPOINT_FILENAMES)


def baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a packaged BEAST baseline checkpoint path."""
    return checkpoint_path(__package__, BASELINE_CHECKPOINT_FILENAMES, name)


def base_checkpoint_path() -> Path:
    """Return the default packaged BEAST baseline checkpoint."""
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
