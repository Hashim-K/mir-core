# tests/test_beats_experiments.py
"""Tests for mir_core.beats.experiments."""
import pytest
from mir_core.beats.experiments import (
    Preset,
    PRESETS,
    PRESETS_BY_KEY,
    experiment_hash,
    get_by_hash,
    get_by_key,
)


def test_experiment_hash_has_btk_prefix():
    config = {"model": {"name": "bock_tcn"}, "seed": 42}
    h = experiment_hash(config)
    assert h.startswith("btk-"), f"Expected 'btk-' prefix, got: {h}"


def test_experiment_hash_total_length():
    config = {"model": {"name": "bock_tcn"}}
    h = experiment_hash(config)
    assert len(h) == 20, f"Expected length 20, got {len(h)}: {h}"


def test_experiment_hash_hex_suffix():
    config = {"model": {"name": "bock_tcn"}}
    h = experiment_hash(config)
    suffix = h[4:]  # strip "btk-"
    assert all(c in "0123456789abcdef" for c in suffix), f"Non-hex suffix: {suffix}"


def test_experiment_hash_is_deterministic():
    config = {"model": {"name": "bock_tcn"}, "seed": 42}
    assert experiment_hash(config) == experiment_hash(config)


def test_experiment_hash_is_order_independent():
    config_a = {"a": 1, "b": 2}
    config_b = {"b": 2, "a": 1}
    assert experiment_hash(config_a) == experiment_hash(config_b)


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
    from mir_core.beats.experiments.presets import load_presets
    result = load_presets()
    assert isinstance(result, dict)


def test_presets_registry_is_dict():
    assert isinstance(PRESETS, dict)


def test_get_by_hash_unknown_returns_none():
    assert get_by_hash("btk-0000000000000000") is None


def test_get_by_key_unknown_returns_none():
    assert get_by_key("nonexistent_key") is None


def test_load_presets_raises_on_malformed_json(tmp_path, monkeypatch):
    from mir_core.beats.experiments import presets as p_module
    bad = tmp_path / "btk-badhash0000000.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(p_module, "PRESETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Malformed preset file"):
        p_module.load_presets()


def test_load_presets_raises_on_missing_required_key(tmp_path, monkeypatch):
    import json
    from mir_core.beats.experiments import presets as p_module
    bad = tmp_path / "btk-badhash0000000.json"
    bad.write_text(json.dumps({"citation": "Author 2024"}))  # missing "key" and "config"
    monkeypatch.setattr(p_module, "PRESETS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Malformed preset file"):
        p_module.load_presets()
