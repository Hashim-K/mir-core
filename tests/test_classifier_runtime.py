from __future__ import annotations

import gc
from dataclasses import FrozenInstanceError
import weakref

import numpy as np
import pytest
import torch
import torch.nn as nn

from mir_core.classifier import StreamingClassifierRuntime
from mir_core.models import GenreClassifier

LABELS = ["candombe", "brid", "salsa", "other"]


class _FeatureLogits(nn.Module):
    """Treat the first feature frame as deterministic classifier logits."""

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        if model_input.ndim == 4:
            return model_input[:, 0, :, 0]
        if model_input.ndim == 3:
            return model_input[:, 0, :]
        raise ValueError(f"Unexpected fixture input shape: {model_input.shape}.")


def _classifier(*, temperature: float = 1.0, arch: str = "embedding_stats_mlp"):
    model_kwargs = (
        {"input_dim": 4}
        if arch == "beatnet_conv"
        else {"embedding_dim": 4, "hidden_dim": 4, "dropout": 0.0}
    )
    classifier = GenreClassifier(
        arch=arch,
        num_classes=4,
        genre_labels=LABELS,
        calibration_temperature=temperature,
        **model_kwargs,
    )
    classifier.model = _FeatureLogits()
    return classifier


def _logit_window(probabilities: list[float]) -> np.ndarray:
    return np.log(np.asarray(probabilities, dtype=np.float32))[np.newaxis, :]


def test_runtime_returns_timestamped_calibrated_route_result() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(temperature=2.0),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )
    window = np.asarray([[4.0, 2.0, 0.0, -2.0]], dtype=np.float32)

    result = runtime.process_window(window, timestamp_seconds=1.25)

    expected = torch.softmax(torch.tensor([4.0, 2.0, 0.0, -2.0]) / 2.0, dim=0)
    assert result.sequence_index == 0
    assert result.timestamp_seconds == pytest.approx(1.25)
    assert result.logits == pytest.approx([4.0, 2.0, 0.0, -2.0])
    assert result.probabilities == pytest.approx(expected.tolist())
    assert result.predicted_label == "candombe"
    assert result.smoothed_label == "candombe"
    assert result.policy_weights == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert result.dominant_route_label == "candombe"
    assert result.routed_label == "candombe"
    assert result.temperature == pytest.approx(2.0)
    assert result.as_dict()["route"]["policy_weights"] == pytest.approx(
        {
            "candombe": 1.0,
            "brid": 0.0,
            "salsa": 0.0,
            "other": 0.0,
        }
    )
    with pytest.raises(FrozenInstanceError):
        result.routed_label = "brid"  # type: ignore[misc]


def test_runtime_ema_can_enter_low_confidence_fallback() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=0.5,
        confidence_threshold=0.6,
        hysteresis_margin=0.0,
    )

    accepted = runtime.process_window(
        _logit_window([0.99, 0.005, 0.003, 0.002]),
        timestamp_seconds=1.0,
    )
    rejected = runtime.process_window(
        _logit_window([0.25, 0.25, 0.25, 0.25]),
        timestamp_seconds=1.5,
    )

    assert accepted.routed_label == "candombe"
    assert accepted.confidence_rejected is False
    assert rejected.smoothed_confidence < 0.6
    assert rejected.confidence_rejected is True
    assert rejected.routed_label == "other"
    assert rejected.policy_weights == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert rejected.fallback_reason == "low_confidence"


def test_runtime_hysteresis_holds_then_switches_dominant_route() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.3,
        hysteresis_margin=0.1,
    )

    first = runtime.process_window(
        _logit_window([0.60, 0.30, 0.05, 0.05]),
        timestamp_seconds=1.0,
    )
    held = runtime.process_window(
        _logit_window([0.43, 0.50, 0.04, 0.03]),
        timestamp_seconds=1.5,
    )
    switched = runtime.process_window(
        _logit_window([0.35, 0.58, 0.04, 0.03]),
        timestamp_seconds=2.0,
    )

    assert first.routed_label == "candombe"
    assert held.dominant_route_label == "brid"
    assert held.routed_label == "candombe"
    assert held.hysteresis_held is True
    assert held.switched is False
    assert switched.routed_label == "brid"
    assert switched.hysteresis_held is False
    assert switched.switched is True


def test_runtime_hysteresis_requires_margin_to_leave_fallback() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.5,
        hysteresis_margin=0.1,
    )

    fallback = runtime.process_window(
        _logit_window([0.40, 0.30, 0.20, 0.10]),
        timestamp_seconds=1.0,
    )
    held = runtime.process_window(
        _logit_window([0.55, 0.20, 0.15, 0.10]),
        timestamp_seconds=1.5,
    )
    released = runtime.process_window(
        _logit_window([0.65, 0.15, 0.10, 0.10]),
        timestamp_seconds=2.0,
    )

    assert fallback.fallback_reason == "low_confidence"
    assert held.dominant_route_label == "candombe"
    assert held.routed_label == "other"
    assert held.hysteresis_held is True
    assert held.fallback_reason == "hysteresis_hold"
    assert released.routed_label == "candombe"
    assert released.switched is True


def test_runtime_distinguishes_native_fallback_from_rejection() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.5,
    )

    result = runtime.process_window(
        _logit_window([0.05, 0.05, 0.10, 0.80]),
        timestamp_seconds=1.0,
    )

    assert result.confidence_rejected is False
    assert result.native_fallback_prediction is True
    assert result.routed_label == "other"
    assert result.fallback_reason == "native_prediction"


@pytest.mark.parametrize(
    ("strategy", "probabilities", "expected_weights", "expected_mode"),
    [
        ("hard", [0.6, 0.3, 0.05, 0.05], [1.0, 0.0, 0.0, 0.0], "hard"),
        ("soft", [0.6, 0.3, 0.05, 0.05], [0.6, 0.3, 0.05, 0.05], "soft_blend"),
        (
            "hybrid",
            [0.6, 0.3, 0.05, 0.05],
            [2 / 3, 1 / 3, 0.0, 0.0],
            "top2_blend",
        ),
        ("hybrid", [0.9, 0.05, 0.03, 0.02], [1.0, 0.0, 0.0, 0.0], "hard"),
    ],
)
def test_runtime_exposes_exact_public_policy_plan(
    strategy: str,
    probabilities: list[float],
    expected_weights: list[float],
    expected_mode: str,
) -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy=strategy,
        ema_alpha=1.0,
        confidence_threshold=0.0,
        blend_threshold=0.8,
    )

    result = runtime.process_window(
        _logit_window(probabilities),
        timestamp_seconds=1.0,
    )

    assert result.policy_weights == pytest.approx(expected_weights)
    assert result.route_mode == expected_mode
    assert sum(result.policy_weights) == pytest.approx(1.0)


def test_runtime_reset_clears_all_temporal_state_and_allows_new_timeline() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )
    runtime.process_window(
        _logit_window([0.8, 0.1, 0.05, 0.05]),
        timestamp_seconds=10.0,
    )

    runtime.reset()

    assert runtime.state.windows_processed == 0
    assert runtime.state.last_timestamp_seconds is None
    assert runtime.state.routed_label is None
    assert runtime.state.smoothed_probabilities == pytest.approx([0.25] * 4)
    restarted = runtime.process_window(
        _logit_window([0.1, 0.8, 0.05, 0.05]),
        timestamp_seconds=0.5,
    )
    assert restarted.sequence_index == 0
    assert restarted.previous_routed_label is None
    assert restarted.routed_label == "brid"


def test_runtime_rejects_non_monotonic_time_without_mutating_state() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )
    runtime.process_window(
        _logit_window([0.8, 0.1, 0.05, 0.05]),
        timestamp_seconds=1.0,
    )
    before = runtime.state

    with pytest.raises(ValueError, match="increase strictly"):
        runtime.process_window(
            _logit_window([0.1, 0.8, 0.05, 0.05]),
            timestamp_seconds=1.0,
        )

    assert runtime.state == before


def test_runtime_supports_beatnet_conv_time_feature_windows() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(arch="beatnet_conv"),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )

    result = runtime.process_window(
        _logit_window([0.1, 0.7, 0.1, 0.1]),
        timestamp_seconds=1.0,
    )

    assert result.predicted_label == "brid"
    assert result.routed_label == "brid"


@pytest.mark.parametrize(
    ("layout", "window"),
    [
        (
            "time_features",
            np.asarray([[0.1, 0.7, 0.1, 0.1]], dtype=np.float32),
        ),
        (
            "features_time",
            np.asarray([[0.1], [0.7], [0.1], [0.1]], dtype=np.float32),
        ),
        (
            "model_input",
            np.asarray([[[[0.1], [0.7], [0.1], [0.1]]]], dtype=np.float32),
        ),
    ],
)
def test_runtime_supports_streaming_feature_layouts(
    layout: str,
    window: np.ndarray,
) -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )

    result = runtime.process_window(
        window,
        timestamp_seconds=1.0,
        feature_layout=layout,
    )

    assert result.predicted_label == "brid"


def test_runtime_retains_compact_state_but_not_feature_window() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )
    window = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    window_reference = weakref.ref(window)

    runtime.process_window(window, timestamp_seconds=1.0)
    del window
    gc.collect()

    assert window_reference() is None
    state = runtime.state
    assert state.windows_processed == 1
    assert state.last_timestamp_seconds == pytest.approx(1.0)
    assert state.routed_label == "candombe"
    assert isinstance(state.smoothed_probabilities, tuple)
    assert not any(
        isinstance(value, (np.ndarray, torch.Tensor))
        for value in (
            state.windows_processed,
            state.last_timestamp_seconds,
            state.routed_label,
            state.smoothed_probabilities,
        )
    )
    assert not hasattr(runtime, "feature_cache")
    assert not hasattr(runtime, "last_result")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_layout": "unknown"}, "feature_layout"),
        ({"temperature": 0.0}, "temperature"),
        ({"strategy": "unknown"}, "strategy"),
        ({"hysteresis_margin": -0.1}, "hysteresis_margin"),
        (
            {"confidence_threshold": 0.9, "hysteresis_margin": 0.2},
            "must not exceed 1",
        ),
    ],
)
def test_runtime_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StreamingClassifierRuntime(_classifier(), **kwargs)
