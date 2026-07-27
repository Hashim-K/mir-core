from __future__ import annotations

import json

import pytest

from mir_core.classifier.experiments import presets
from mir_core.utils.hashing import stable_hash


def _write_preset(directory, *, key: str, config: dict) -> str:
    experiment_hash = f"clf-{stable_hash(config)}"
    payload = {
        "key": key,
        "hash": experiment_hash,
        "citation": "Test et al. (2026)",
        "notes": ["fixture"],
        "category": "routing",
        "config": config,
    }
    (directory / f"{experiment_hash}.json").write_text(json.dumps(payload))
    return experiment_hash


def test_empty_or_missing_classifier_preset_directory_is_compatible(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path / "missing")
    assert presets.load_presets() == {}

    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path / "empty")
    assert presets.load_presets() == {}


def test_classifier_preset_loader_validates_and_loads_hash_keyed_json(
    tmp_path,
    monkeypatch,
) -> None:
    config = {"model": {"arch": "embedding_stats_mlp"}, "seed": 42}
    experiment_hash = _write_preset(tmp_path, key="router_baseline", config=config)
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path)

    loaded = presets.load_presets()

    assert list(loaded) == [experiment_hash]
    assert loaded[experiment_hash].key == "router_baseline"
    assert loaded[experiment_hash].category == "routing"
    assert loaded[experiment_hash].config == config


def test_classifier_preset_loader_rejects_filename_hash_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        "key": "bad",
        "hash": "clf-0000000000000000",
        "citation": "",
        "config": {"seed": 42},
    }
    path = tmp_path / "clf-0000000000000000.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path)

    with pytest.raises(ValueError, match="filename stem must be"):
        presets.load_presets()


def test_classifier_preset_loader_rejects_duplicate_human_keys(
    tmp_path,
    monkeypatch,
) -> None:
    _write_preset(tmp_path, key="duplicate", config={"seed": 1})
    _write_preset(tmp_path, key="duplicate", config={"seed": 2})
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path)

    with pytest.raises(ValueError, match="duplicate preset key"):
        presets.load_presets()


def test_classifier_preset_loader_ignores_non_classifier_json(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "notes.json").write_text("{not valid json")
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path)
    assert presets.load_presets() == {}
