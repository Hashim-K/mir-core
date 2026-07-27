from __future__ import annotations

import torch

import mir_core.hub as hub
from mir_core.models import GenreClassifier


def test_trainer_checkpoint_is_discovered_and_loaded(tmp_path, monkeypatch) -> None:
    labels = ["candombe", "brid", "salsa", "other"]
    model_config = {
        "arch": "embedding_stats_mlp",
        "embedding_dim": 8,
        "hidden_dim": 6,
        "dropout": 0.0,
    }
    original = GenreClassifier(
        num_classes=len(labels),
        genre_labels=labels,
        **model_config,
    )
    checkpoint_path = tmp_path / "clf-example" / "fold2" / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "model_state_dict": original.state_dict(),
            "arch": model_config["arch"],
            "labels": labels,
            "experiment_hash": "clf-1234567890abcdef",
            "model_config": model_config,
            "split_contract": {"fold_index": 2},
            "epoch": 7,
            "best_metric": "val_macro_f1",
            "best_value": 0.7,
            "validation_metrics": {"acc": 0.75, "macro_f1": 0.7},
            "validation_metrics_calibrated": {
                "acc": 0.8,
                "macro_f1": 0.78,
                "confusion": [[1, 0], [0, 1]],
            },
            "calibration": {
                "enabled": True,
                "status": "calibrated",
                "temperature": 2.0,
                "confidence_threshold": 0.76,
                "threshold_objective": "balanced_router",
                "threshold_objective_value": 0.84,
                "selection_split": "validation",
            },
            "router_config": {
                "fallback_label": "other",
                "confidence_threshold": 0.76,
                "temperature": 1.5,
                "calibration": {"enabled": True},
            },
        },
        checkpoint_path,
    )

    registry = hub.ModelRegistry()
    monkeypatch.setattr(hub, "_registry", registry)
    hub._discover_genre_classifiers(tmp_path)

    name = "genre_classifier-embedding_stats_mlp-f2"
    spec = registry.get(name)
    assert spec is not None
    assert spec.checkpoint_path == str(checkpoint_path)
    assert spec.fold == 2
    assert spec.model_kwargs == {
        "arch": "embedding_stats_mlp",
        "embedding_dim": 8,
        "hidden_dim": 6,
        "dropout": 0.0,
        "genre_labels": labels,
        "num_classes": 4,
        "calibration_temperature": 2.0,
    }
    assert spec.metrics == {"acc": 0.8, "macro_f1": 0.78}
    assert spec.metadata["validation_metrics_source"] == (
        "validation_metrics_calibrated"
    )
    assert spec.metadata["selection"] == {
        "epoch": 7,
        "best_metric": "val_macro_f1",
        "best_value": 0.7,
        "validation_metrics": {"acc": 0.75, "macro_f1": 0.7},
    }
    assert spec.metadata["calibration"]["threshold_objective"] == ("balanced_router")

    loaded = hub.load_model(name)

    assert isinstance(loaded, GenreClassifier)
    assert loaded.arch_name == "embedding_stats_mlp"
    assert loaded.genre_labels == labels
    assert loaded.calibration_temperature == 2.0
    assert loaded.calibration_metadata == spec.metadata["calibration"]
    assert loaded.router_config == spec.metadata["router_config"]
    calibration_metadata = loaded.calibration_metadata
    calibration_metadata["temperature"] = 99.0
    assert loaded.calibration_temperature == 2.0
    assert loaded.calibration_metadata["temperature"] == 2.0
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)


def test_classifier_checkpoint_uses_router_temperature_when_calibration_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    labels = ["target", "other"]
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=2,
        genre_labels=labels,
        embedding_dim=4,
        hidden_dim=3,
    )
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "embedding_stats_mlp",
            "labels": labels,
            "model_config": {
                "arch": "embedding_stats_mlp",
                "embedding_dim": 4,
                "hidden_dim": 3,
            },
            "router_config": {
                "fallback_label": "other",
                "confidence_threshold": 0.71,
                "temperature": 1.75,
            },
        },
        checkpoint_path,
    )

    registry = hub.ModelRegistry()
    monkeypatch.setattr(hub, "_registry", registry)
    hub._discover_genre_classifiers(tmp_path)
    loaded = hub.load_model("genre_classifier-embedding_stats_mlp")

    assert loaded.calibration_temperature == 1.75
    assert loaded.calibration_metadata == {
        "temperature": 1.75,
        "confidence_threshold": 0.71,
    }
    assert loaded.router_config["fallback_label"] == "other"


def test_same_architecture_and_fold_experiments_are_discovered_distinctly(
    tmp_path,
    monkeypatch,
) -> None:
    labels = ["candombe", "brid", "salsa", "other"]
    model_config = {
        "arch": "embedding_stats_mlp",
        "embedding_dim": 8,
        "hidden_dim": 6,
        "dropout": 0.0,
    }

    def write_checkpoint(
        directory: str,
        *,
        feature_type: str,
        experiment_hash: str,
        fill_value: float,
    ) -> GenreClassifier:
        model = GenreClassifier(
            num_classes=len(labels),
            genre_labels=labels,
            **model_config,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(fill_value)
        path = tmp_path / directory / "fold0" / "checkpoints" / "best.pt"
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arch": model_config["arch"],
                "labels": labels,
                "experiment_hash": experiment_hash,
                "feature_config": {"type": feature_type},
                "model_config": model_config,
                "split_contract": {"fold_index": 0},
            },
            path,
        )
        return model

    yamnet = write_checkpoint(
        "yamnet-run",
        feature_type="yamnet_embedding",
        experiment_hash="clf-aaaaaaaaaaaaaaaa",
        fill_value=0.1,
    )
    efficientat = write_checkpoint(
        "efficientat-run",
        feature_type="efficientat_embedding",
        experiment_hash="clf-bbbbbbbbbbbbbbbb",
        fill_value=0.2,
    )

    registry = hub.ModelRegistry()
    monkeypatch.setattr(hub, "_registry", registry)
    hub._discover_genre_classifiers(tmp_path)

    yamnet_name = (
        "genre_classifier-embedding_stats_mlp-yamnet_embedding-"
        "clf-aaaaaaaaaaaaaaaa-f0"
    )
    efficientat_name = (
        "genre_classifier-embedding_stats_mlp-efficientat_embedding-"
        "clf-bbbbbbbbbbbbbbbb-f0"
    )
    assert set(registry.list_names()) == {yamnet_name, efficientat_name}
    assert registry.get("genre_classifier-embedding_stats_mlp-f0") is None

    loaded_yamnet = hub.load_model(yamnet_name)
    loaded_efficientat = hub.load_model(efficientat_name)
    for key, expected in yamnet.state_dict().items():
        torch.testing.assert_close(loaded_yamnet.state_dict()[key], expected)
    for key, expected in efficientat.state_dict().items():
        torch.testing.assert_close(loaded_efficientat.state_dict()[key], expected)
    assert registry.get(yamnet_name).metadata["feature_type"] == "yamnet_embedding"
    assert registry.get(efficientat_name).metadata["feature_type"] == (
        "efficientat_embedding"
    )
