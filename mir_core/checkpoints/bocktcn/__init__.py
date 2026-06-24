"""Bock TCN baseline checkpoint resources."""

from __future__ import annotations

from pathlib import Path

from .._resources import checkpoint_names, checkpoint_path, immutable_mapping


BASELINE_CHECKPOINT_FILENAMES = immutable_mapping(
    {
        "baseline": "baseline.pkl",
        "baseline_alt0": "baseline_alt0.pkl",
        "baseline_alt1": "baseline_alt1.pkl",
        "baseline_alt2": "baseline_alt2.pkl",
        "baseline_alt3": "baseline_alt3.pkl",
        "baseline_alt4": "baseline_alt4.pkl",
        "baseline_alt5": "baseline_alt5.pkl",
        "baseline_alt6": "baseline_alt6.pkl",
    }
)
BASELINE_CHECKPOINT_SHA256 = immutable_mapping(
    {
        "baseline": "f406175bf34b017ec2a6930f17d777631eebed312cec0eada9803f4867e99251",
        "baseline_alt0": "c128fd63ce7796c40e312ec9b0d5e6b8aa541cd416b2dab79b86d3af85c88f39",
        "baseline_alt1": "86ce8328ba519a4d75e53a155bc359292b692a01ea0b2ad57dc972ef9e5f8f83",
        "baseline_alt2": "053fb7430e32bbfeef87905e68c345d6d2a0906190022bbdcbbd0ad6c78f5dc2",
        "baseline_alt3": "29fe3fcb244ad8b5f672b9986520fe9b942d35a98db2e2dd6daca3a23e802d24",
        "baseline_alt4": "5c81d29ad45b1f812a18835d404a9bfe5ee26f5888227055fc9ee34a43ca5bb0",
        "baseline_alt5": "20aa2c1306a11657cf90f79ceab6f44cda35bfb8600c6ad003fccd487aae4686",
        "baseline_alt6": "f5cd319aa0a4888b99f91af23608f6ee880c2b470885da0efbf58a6354833fb0",
    }
)

BASE_CHECKPOINT_FILENAME = BASELINE_CHECKPOINT_FILENAMES["baseline"]
BASE_CHECKPOINT_SHA256 = BASELINE_CHECKPOINT_SHA256["baseline"]


def baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available Bock TCN baseline checkpoint selector names."""
    return checkpoint_names(BASELINE_CHECKPOINT_FILENAMES)


def baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a packaged Bock TCN baseline checkpoint path."""
    return checkpoint_path(__package__, BASELINE_CHECKPOINT_FILENAMES, name)


def base_checkpoint_path() -> Path:
    """Return the default packaged Bock TCN baseline checkpoint."""
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
