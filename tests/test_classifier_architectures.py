from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mir_core.models import (
    BeatNetLogSpectCNN,
    EmbeddingStatsMLP,
    FramewiseEmbeddingMLP,
    GenreClassifier,
)


class _FixedLogits(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [[4.0, 2.0, 0.0, -2.0]],
            dtype=x.dtype,
            device=x.device,
        ).expand(len(x), -1)


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


def test_embedding_stats_mlp_supports_hear_downstream_head_contract() -> None:
    torch.manual_seed(17)
    model = EmbeddingStatsMLP(
        num_classes=4,
        embedding_dim=600,
        hidden_dim=1024,
        hidden_layers=1,
        batch_norm=True,
        dropout=0.1,
        initialization="xavier_uniform",
    )

    layers = list(model.classifier)
    assert [type(layer) for layer in layers] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.Dropout,
        nn.ReLU,
        nn.Linear,
    ]
    assert torch.count_nonzero(layers[0].bias) == 0
    assert model(torch.zeros(3, 1, 600, 23)).shape == (3, 4)


def test_embedding_stats_mlp_rejects_unknown_initialization() -> None:
    with pytest.raises(ValueError, match="initialization must be"):
        EmbeddingStatsMLP(initialization="mystery")


def test_genre_classifier_factory_supports_embedding_stats_mlp() -> None:
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=4,
        embedding_dim=1024,
        hidden_dim=64,
    )

    logits = model(torch.zeros(1, 1, 1024, 6))

    assert logits.shape == (1, 4)


def test_framewise_embedding_mlp_mean_aggregates_shared_frame_logits() -> None:
    model = FramewiseEmbeddingMLP(
        num_classes=2,
        embedding_dim=3,
        hidden_dim=3,
        dropout=0.0,
    )
    model.classifier = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.classifier.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    values = torch.tensor(
        [[[[1.0, 3.0], [4.0, 2.0], [9.0, 9.0]]]],
        dtype=torch.float32,
    )

    logits = model(values)

    torch.testing.assert_close(logits, torch.tensor([[2.0, 3.0]]))


def test_genre_classifier_factory_supports_framewise_embedding_mlp() -> None:
    model = GenreClassifier(
        arch="framewise_embedding_mlp",
        num_classes=4,
        embedding_dim=1024,
        hidden_dim=512,
    )

    assert model(torch.zeros(2, 1, 1024, 7)).shape == (2, 4)


def test_framewise_embedding_mlp_supports_explicit_xavier_initialization() -> None:
    torch.manual_seed(23)
    model = FramewiseEmbeddingMLP(
        num_classes=4,
        embedding_dim=8,
        hidden_dim=5,
        initialization="xavier_uniform",
    )

    linear_layers = [
        layer for layer in model.classifier if isinstance(layer, nn.Linear)
    ]
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linear_layers)
    assert model(torch.zeros(2, 1, 8, 3)).shape == (2, 4)


def test_genre_classifier_predict_applies_calibration_without_scaling_forward() -> None:
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=4,
        embedding_dim=8,
        calibration_temperature=2.0,
    )
    model.model = _FixedLogits()
    model_input = torch.zeros(1, 1, 8, 2)

    logits = model(model_input)
    prediction = model.predict(model_input)

    torch.testing.assert_close(logits, torch.tensor([[4.0, 2.0, 0.0, -2.0]]))
    expected = torch.softmax(logits / 2.0, dim=-1)[0]
    assert prediction["calibration_temperature"] == 2.0
    assert prediction["confidence"] == pytest.approx(float(expected[0]))
    assert prediction["probabilities"] == pytest.approx(
        {
            label: float(probability)
            for label, probability in zip(model.genre_labels, expected)
        }
    )


@pytest.mark.parametrize(
    "temperature",
    [True, 0.0, -1.0, float("nan"), float("inf"), "invalid"],
)
def test_genre_classifier_rejects_invalid_calibration_temperature(
    temperature,
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        GenreClassifier(
            arch="embedding_stats_mlp",
            num_classes=4,
            embedding_dim=8,
            calibration_temperature=temperature,
        )


def test_calibration_temperature_assignment_is_validated() -> None:
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=4,
        embedding_dim=8,
    )
    with pytest.raises(ValueError, match="positive finite"):
        model.calibration_temperature = 0.0
    assert model.calibration_temperature == 1.0
