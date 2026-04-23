# mir_core/classifier/experiments/presets.py
"""Classifier experiment preset registry (stub — no presets yet).

Follows identical structure to mir_core.beats.experiments.presets.
Populate presets/ with clf-{hash}.json files when classifier experiments are defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRESETS_DIR: Path = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class Preset:
    key: str
    hash: str
    citation: str
    config: dict[str, Any]
    notes: list[str]


def load_presets() -> dict[str, Preset]:
    """Returns empty dict until classifier presets are added."""
    return {}


PRESETS: dict[str, Preset] = {}
PRESETS_BY_KEY: dict[str, Preset] = {}


def get_by_hash(hash_key: str) -> Preset | None:
    return None


def get_by_key(key: str) -> Preset | None:
    return None
