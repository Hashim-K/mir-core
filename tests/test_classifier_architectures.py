from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mir_core.models import BeatNetLogSpectCNN, EmbeddingStatsMLP, GenreClassifier


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


def test_genre_classifier_factory_supports_embedding_stats_mlp() -> None:
    model = GenreClassifier(
        arch="embedding_stats_mlp",
        num_classes=4,
        embedding_dim=1024,
        hidden_dim=64,
    )

    logits = model(torch.zeros(1, 1, 1024, 6))

    assert logits.shape == (1, 4)


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
