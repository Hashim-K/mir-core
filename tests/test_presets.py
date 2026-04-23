# tests/test_presets.py
"""Tests for the beat tracking canonical experiment preset registry."""
import json
import pytest
from mir_core.beats.experiments import experiment_hash
from mir_core.beats.experiments.presets import Preset, load_presets, PRESETS_DIR


def test_preset_dataclass_fields():
    p = Preset(
        key="test_key",
        hash="btk-abc123def456abcd",
        citation="Author 2024",
        config={"experiment": {"name": "test"}},
        notes=["note one"],
    )
    assert p.key == "test_key"
    assert p.hash == "btk-abc123def456abcd"
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
        # filename stem must be btk-{16 hex chars}
        assert f.stem.startswith("btk-"), (
            f"{f.name}: filename must start with 'btk-', got '{f.stem}'"
        )
        assert len(f.stem) == 20, (
            f"{f.name}: hash must be 20 chars, got {len(f.stem)}"
        )
        assert all(c in "0123456789abcdef" for c in f.stem[4:]), (
            f"{f.name}: hash suffix must be hex, got '{f.stem[4:]}'"
        )
        assert data.get("hash") == f.stem, (
            f"{f.name}: 'hash' field '{data.get('hash')}' must match filename stem '{f.stem}'"
        )
        assert experiment_hash(data["config"]) == f.stem, (
            f"{f.name}: filename/hash must equal experiment_hash(config)"
        )


def test_load_presets_rejects_filename_that_does_not_match_config_hash(tmp_path, monkeypatch):
    from mir_core.beats.experiments import presets as p_module

    bad = tmp_path / "btk-0000000000000000.json"
    bad.write_text(json.dumps({
        "key": "bad",
        "hash": "btk-0000000000000000",
        "citation": "Author 2024",
        "notes": [],
        "config": {"experiment": {"name": "not-the-zero-hash"}},
    }))
    monkeypatch.setattr(p_module, "PRESETS_DIR", tmp_path)

    with pytest.raises(ValueError, match="filename stem must be"):
        p_module.load_presets()


def test_presets_registry_from_exports():
    """PRESETS exported from mir_core.beats.experiments must equal load_presets()."""
    from mir_core.beats.experiments import PRESETS as top_level
    from mir_core.beats.experiments.presets import PRESETS as direct
    assert top_level is direct
