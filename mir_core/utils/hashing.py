# mir_core/utils/hashing.py
"""Stable hashing utilities for experiment identity.

Functions:
    canonical_json  — deterministic JSON serialisation
    stable_digest   — SHA-256 hex digest (full or truncated) of any JSON-serialisable value
    stable_hash     — first N hex chars of SHA-256 (default 16)
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialise a value to deterministic JSON (sorted keys, no whitespace).

    Non-JSON-native types (e.g. numpy scalars, datetime) are coerced via str().
    Callers must ensure config values are standard Python types for reliable hashes.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(value: Any, length: int | None = None) -> str:
    """Return a SHA-256 hex digest for a JSON-serialisable value."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the raw experiment hash for a config mapping (no task prefix).

    The returned string is `length` lowercase hex characters. Use
    mir_core.beats.experiments.experiment_hash() or
    mir_core.classifier.experiments.experiment_hash() to get a task-prefixed hash.
    """
    return stable_digest(dict(config), length=length)
