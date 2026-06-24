"""BeatNet+ baseline checkpoint resources."""

from __future__ import annotations

from pathlib import Path

from .._resources import checkpoint_names, checkpoint_path, immutable_mapping


BASELINE_CHECKPOINT_FILENAMES = immutable_mapping(
    {
        "baseline": "baseline.pt",
        "baseline_alt0": "baseline_alt0.pt",
        "baseline_alt1": "baseline_alt1.pt",
    }
)
BASELINE_CHECKPOINT_SHA256 = immutable_mapping(
    {
        "baseline": "ed52f90e27ff9b5ef3c63f59c6d4b37366f60a21a48ea1d46d7c3e18d6f1977e",
        "baseline_alt0": "5bbae630b4112f3c1193654e3dbc946b850caa06a2a82e10e8de9e48bb673519",
        "baseline_alt1": "15fe9c1ec8f2fca75dd3cecac72a3fbf57c9004e8726a786876ad5e269f4895f",
    }
)

BASE_CHECKPOINT_FILENAME = BASELINE_CHECKPOINT_FILENAMES["baseline"]
BASE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256["baseline"]


def baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BeatNet+ baseline checkpoint selector names."""
    return checkpoint_names(BASELINE_CHECKPOINT_FILENAMES)


def baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a packaged BeatNet+ baseline checkpoint path."""
    return checkpoint_path(__package__, BASELINE_CHECKPOINT_FILENAMES, name)


def base_checkpoint_path() -> Path:
    """Return the default packaged BeatNet+ baseline checkpoint."""
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
