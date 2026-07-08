from __future__ import annotations

import torch

from mir_core.models import BeatNetLogSpectCNN, EmbeddingStatsMLP, GenreClassifier


def test_beatnet_log_spect_classifier_output_shape() -> None:
    model = BeatNetLogSpectCNN(num_classes=4, feature_dim=272)

    logits = model(torch.zeros(2, 1, 272, 150))

    assert logits.shape == (2, 4)


def test_genre_classifier_factory_supports_beatnet_log_spect_cnn() -> None:
    model = GenreClassifier(
        arch="beatnet_log_spect_cnn",
        num_classes=4,
        genre_labels=["candombe", "brid", "salsa", "other"],
        feature_dim=272,
    )

    logits = model(torch.zeros(1, 1, 272, 150))

    assert logits.shape == (1, 4)
    assert model.genre_labels == ["candombe", "brid", "salsa", "other"]


def test_embedding_stats_mlp_output_shape() -> None:
    model = EmbeddingStatsMLP(num_classes=4, embedding_dim=1024, hidden_dim=64)

    logits = model(torch.zeros(3, 1, 1024, 6))

    assert logits.shape == (3, 4)


def test_genre_classifier_factory_supports_embedding_stats_mlp() -> None:
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=4,
        embedding_dim=1024,
        hidden_dim=64,
    )

    logits = model(torch.zeros(1, 1, 1024, 6))

    assert logits.shape == (1, 4)
