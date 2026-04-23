# mir_core/beats/experiments/__init__.py
"""Beat tracking experiment registry.

All beat experiment hashes are prefixed with 'btk-' (20 chars total).

Usage:
    from mir_core.beats.experiments import experiment_hash, get_by_hash, get_by_key, PRESETS
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mir_core.utils.hashing import stable_hash
from .presets import (
    Preset,
    PRESETS,
    PRESETS_BY_KEY,
    load_presets,
    get_by_hash,
    get_by_key,
)

TASK_PREFIX = "btk"


def experiment_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the canonical beat tracking experiment hash.

    Format: 'btk-{16-hex-chars}' — always 20 characters.
    The 'btk' prefix identifies this as a beat tracking experiment.
    Pass the unexpanded config (env vars not substituted) for
    machine-independent hashes.
    """
    return f"{TASK_PREFIX}-{stable_hash(config, length=length)}"


__all__ = [
    "TASK_PREFIX",
    "experiment_hash",
    "Preset",
    "PRESETS",
    "PRESETS_BY_KEY",
    "load_presets",
    "get_by_hash",
    "get_by_key",
]
