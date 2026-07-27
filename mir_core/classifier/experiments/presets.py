"""Validated classifier experiment preset registry.

Each preset is a JSON file named ``clf-{stable-config-hash}.json``. The
registry intentionally remains empty when no presets directory/files exist, so
checking in the first classifier preset requires no Python code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mir_core.utils.hashing import stable_hash

PRESETS_DIR: Path = Path(__file__).parent / "presets"
TASK_PREFIX = "clf"


@dataclass(frozen=True)
class Preset:
    key: str
    hash: str
    citation: str
    config: dict[str, Any]
    notes: list[str]
    category: str = "training"


def _load_preset(path: Path) -> Preset:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed preset file {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Malformed preset file {path.name}: root must be an object")

    try:
        key = data["key"]
        config = data["config"]
        citation = data["citation"]
    except KeyError as exc:
        raise ValueError(
            f"Malformed preset file {path.name}: missing {exc.args[0]!r}"
        ) from exc

    if not isinstance(key, str) or not key.strip():
        raise ValueError(
            f"Malformed preset file {path.name}: key must be a non-empty string"
        )
    if not isinstance(config, dict):
        raise ValueError(f"Malformed preset file {path.name}: config must be an object")
    if not isinstance(citation, str):
        raise ValueError(
            f"Malformed preset file {path.name}: citation must be a string"
        )

    notes = data.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError(
            f"Malformed preset file {path.name}: notes must be a list of strings"
        )
    category = data.get("category", "training")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(
            f"Malformed preset file {path.name}: category must be a non-empty string"
        )

    expected_hash = f"{TASK_PREFIX}-{stable_hash(config)}"
    if path.stem != expected_hash:
        raise ValueError(
            f"Malformed preset file {path.name}: filename stem must be "
            f"{expected_hash} for its config"
        )
    if data.get("hash") != path.stem:
        raise ValueError(
            f"Malformed preset file {path.name}: hash field must match filename stem"
        )
    return Preset(
        key=key,
        hash=path.stem,
        citation=citation,
        config=config,
        notes=list(notes),
        category=category,
    )


def load_presets() -> dict[str, Preset]:
    """Load all checked-in ``clf-*.json`` presets keyed by experiment hash."""

    registry: dict[str, Preset] = {}
    keys: set[str] = set()
    if not PRESETS_DIR.is_dir():
        return registry
    for path in sorted(PRESETS_DIR.glob(f"{TASK_PREFIX}-*.json")):
        preset = _load_preset(path)
        if preset.key in keys:
            raise ValueError(
                f"Malformed preset file {path.name}: duplicate preset key {preset.key!r}"
            )
        keys.add(preset.key)
        registry[preset.hash] = preset
    return registry


PRESETS: dict[str, Preset] = load_presets()
PRESETS_BY_KEY: dict[str, Preset] = {preset.key: preset for preset in PRESETS.values()}


def get_by_hash(hash_key: str) -> Preset | None:
    """Return a preset by canonical experiment hash, or ``None``."""

    return PRESETS.get(hash_key)


def get_by_key(key: str) -> Preset | None:
    """Return a preset by human-readable key, or ``None``."""

    return PRESETS_BY_KEY.get(key)
