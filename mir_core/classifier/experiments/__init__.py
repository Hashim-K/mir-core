# mir_core/classifier/experiments/__init__.py
"""Classifier experiment registry.

All classifier experiment hashes are prefixed with 'clf-' (20 chars total).

Usage:
    from mir_core.classifier.experiments import experiment_hash
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

TASK_PREFIX = "clf"


def experiment_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the canonical classifier experiment hash.

    Format: 'clf-{length}-hex-chars'. With the default length=16, this is
    always 20 characters ('clf-' + 16 hex). The 'clf' prefix identifies this
    as a classifier experiment.
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
