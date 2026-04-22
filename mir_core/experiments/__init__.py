"""Shared experiment identity and hashing utilities.

Keep hash generation here so training, model registry, and future UI code all
derive experiment identities from the same canonical representation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize a value to stable JSON for hashing and comparisons."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_digest(value: Any, length: int | None = None) -> str:
    """Return a SHA-256 digest for a JSON-serializable value."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def stable_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Return the canonical experiment hash for a config mapping."""
    return stable_digest(dict(config), length=length)


def experiment_hash(config: Mapping[str, Any], length: int = 16) -> str:
    """Alias for code that wants a domain-specific name."""
    return stable_hash(config, length=length)


# Re-export preset registry so callers can do:
#   from mir_core.experiments import PRESETS, get_by_hash
from .presets import PRESETS, PRESETS_BY_KEY, Preset, get_by_hash, get_by_key
