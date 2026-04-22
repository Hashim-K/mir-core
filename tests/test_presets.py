# tests/test_presets.py
"""Tests for the canonical experiment preset registry."""
import json
from pathlib import Path
import pytest
from mir_core.experiments.presets import Preset, PRESETS, load_presets, PRESETS_DIR


def test_preset_dataclass_fields():
    p = Preset(
        key="test_key",
        hash="abc123",
        citation="Author 2024",
        config={"experiment": {"name": "test"}},
        notes=["note one"],
    )
    assert p.key == "test_key"
    assert p.hash == "abc123"
    assert isinstance(p.config, dict)
    assert isinstance(p.notes, list)


def test_load_presets_returns_dict():
    result = load_presets()
    assert isinstance(result, dict)


def test_presets_dir_exists():
    assert PRESETS_DIR.is_dir(), f"presets/ directory not found at {PRESETS_DIR}"


def test_preset_file_schema():
    """Every .json file in presets/ must have key, citation, config, notes."""
    for f in PRESETS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        assert "key" in data, f"{f.name} missing 'key'"
        assert "citation" in data, f"{f.name} missing 'citation'"
        assert "config" in data, f"{f.name} missing 'config'"
        assert "notes" in data, f"{f.name} missing 'notes'"
        # filename stem must equal hash
        assert data.get("hash", f.stem) == f.stem, (
            f"{f.name}: 'hash' field must match filename stem"
        )


def test_presets_registry_from_exports():
    """PRESETS exported from mir_core.experiments must equal load_presets()."""
    from mir_core.experiments import PRESETS as top_level
    from mir_core.experiments.presets import PRESETS as direct
    assert top_level is direct
