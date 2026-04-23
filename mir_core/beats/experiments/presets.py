# mir_core/beats/experiments/presets.py
"""Beat tracking experiment preset registry.

Each preset is stored as a JSON file in presets/ named by its full hash
(e.g. btk-a14aef639058a4c7.json). The filename stem IS the experiment hash.

Adding a new preset:
  1. Build the config dict with raw (unexpanded) env var strings.
  2. Compute hash (replace my_preset.json with your file):
     python -c "
     from mir_core.beats.experiments import experiment_hash
     import json
     print(experiment_hash(json.load(open('my_preset.json'))['config']))
     "
  3. Save as presets/{hash}.json with key, hash, citation, notes, config fields.
  4. Commit. No Python changes required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mir_core.utils.hashing import stable_hash

PRESETS_DIR: Path = Path(__file__).parent / "presets"
TASK_PREFIX = "btk"


@dataclass(frozen=True)
class Preset:
    key: str          # human identifier, e.g. "rapini2024_salsaset_beatnet"
    hash: str         # experiment_hash(config) — equals the JSON filename stem
    citation: str     # full citation string
    config: dict[str, Any]  # complete config (unexpanded env vars)
    notes: list[str]  # discrepancy notes and caveats


def load_presets() -> dict[str, Preset]:
    """Scan presets/ and load all .json files into a hash-keyed registry."""
    registry: dict[str, Preset] = {}
    if not PRESETS_DIR.is_dir():
        return registry
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            expected_hash = f"{TASK_PREFIX}-{stable_hash(data['config'])}"
            if path.stem != expected_hash:
                raise ValueError(
                    f"Malformed preset file {path.name}: filename stem must be "
                    f"{expected_hash} for its config"
                )
            if data.get("hash") != path.stem:
                raise ValueError(
                    f"Malformed preset file {path.name}: hash field must match filename stem"
                )
            preset = Preset(
                key=data["key"],
                hash=path.stem,  # filename stem IS the hash (btk-…)
                citation=data["citation"],
                config=data["config"],
                notes=data.get("notes", []),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Malformed preset file {path.name}: {exc}") from exc
        registry[preset.hash] = preset
    return registry


# Module-level registry — loaded once at import time.
PRESETS: dict[str, Preset] = load_presets()

# Convenience lookup by human key (e.g. "rapini2024_salsaset_beatnet").
PRESETS_BY_KEY: dict[str, Preset] = {p.key: p for p in PRESETS.values()}


def get_by_hash(hash_key: str) -> Preset | None:
    """Return preset for a given experiment hash, or None."""
    return PRESETS.get(hash_key)


def get_by_key(key: str) -> Preset | None:
    """Return preset for a given human key, or None."""
    return PRESETS_BY_KEY.get(key)
