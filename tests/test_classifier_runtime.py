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
    assert held.fallback_reason == "confirmation_hold"
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


def test_runtime_separates_uncertainty_from_confident_other_execution() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.6,
        switch_margin=0.0,
        min_consecutive_windows=2,
        execution_labels=(
            "candombe",
            "brid",
            "salsa",
            "latin_general",
            "stock",
        ),
        classifier_to_route={
            "candombe": "candombe",
            "brid": "brid",
            "salsa": "salsa",
            "other": "stock",
        },
        uncertainty_fallback_label="latin_general",
        native_fallback_route_label="stock",
        low_confidence_fallback="hold_then_uncertainty",
        low_confidence_hold_windows=1,
    )

    uncertain = runtime.process_window(
        _logit_window([0.25, 0.25, 0.25, 0.25]),
        timestamp_seconds=1.0,
    )
    stock_pending = runtime.process_window(
        _logit_window([0.05, 0.05, 0.05, 0.85]),
        timestamp_seconds=1.5,
    )
    stock = runtime.process_window(
        _logit_window([0.05, 0.05, 0.05, 0.85]),
        timestamp_seconds=2.0,
    )
    held = runtime.process_window(
        _logit_window([0.25, 0.25, 0.25, 0.25]),
        timestamp_seconds=2.5,
    )
    fallback = runtime.process_window(
        _logit_window([0.25, 0.25, 0.25, 0.25]),
        timestamp_seconds=3.0,
    )

    assert uncertain.routed_label == "latin_general"
    assert uncertain.execution_labels == (
        "candombe",
        "brid",
        "salsa",
        "latin_general",
        "stock",
    )
    assert stock_pending.routed_label == "latin_general"
    assert stock.routed_label == "stock"
    assert stock.native_fallback_prediction is True
    assert stock.execution_weights_by_label["stock"] == pytest.approx(1.0)
    assert held.routed_label == "stock"
    assert held.fallback_reason == "low_confidence_hold"
    assert fallback.routed_label == "latin_general"
    assert fallback.fallback_reason == "low_confidence"


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


def test_hard_execution_weights_wait_for_consecutive_confirmation() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.4,
        switch_margin=0.1,
        min_consecutive_windows=2,
    )
    window = _logit_window([0.70, 0.15, 0.10, 0.05])

    pending = runtime.process_window(window, timestamp_seconds=1.0)
    confirmed = runtime.process_window(window, timestamp_seconds=1.5)

    assert pending.policy_weights == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert pending.execution_weights == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert pending.routed_label == "other"
    assert pending.pending_route_label == "candombe"
    assert pending.pending_route_count == 1
    assert pending.hysteresis_held is True
    assert confirmed.execution_weights == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert confirmed.routed_label == "candombe"
    assert confirmed.pending_route_label is None
    assert confirmed.pending_route_count == 0
    assert confirmed.switched is True


@pytest.mark.parametrize(
    ("strategy", "probabilities"),
    [
        ("soft", [0.60, 0.25, 0.10, 0.05]),
        ("hybrid", [0.60, 0.25, 0.10, 0.05]),
    ],
)
def test_blended_execution_weights_remain_raw_while_label_is_pending(
    strategy: str,
    probabilities: list[float],
) -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy=strategy,
        ema_alpha=1.0,
        confidence_threshold=0.3,
        blend_threshold=0.8,
        switch_margin=0.1,
        min_consecutive_windows=3,
    )

    result = runtime.process_window(
        _logit_window(probabilities),
        timestamp_seconds=1.0,
    )

    assert result.route_mode in {"soft_blend", "top2_blend"}
    assert result.routed_label == "other"
    assert result.execution_weights == pytest.approx(result.policy_weights)
    assert result.pending_route_label == "candombe"
    assert result.pending_route_count == 1


def test_hybrid_hard_execution_uses_confirmed_route_while_pending() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hybrid",
        ema_alpha=1.0,
        confidence_threshold=0.3,
        blend_threshold=0.8,
        switch_margin=0.1,
        min_consecutive_windows=2,
    )

    result = runtime.process_window(
        _logit_window([0.90, 0.05, 0.03, 0.02]),
        timestamp_seconds=1.0,
    )

    assert result.route_mode == "hard"
    assert result.policy_weights == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert result.execution_weights == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert result.routed_label == "other"


def test_runtime_resolves_nested_hysteresis_then_top_level_then_overrides() -> None:
    classifier = _classifier()
    classifier.set_routing_metadata(
        router_config={
            "strategy": "hard",
            "ema_alpha": 1.0,
            "confidence_threshold": 0.4,
            "fallback_label": "other",
            "switch_margin": 0.02,
            "min_consecutive_windows": 1,
            "hysteresis": {
                "switch_margin": 0.12,
                "min_consecutive_windows": 3,
                "low_confidence_fallback": "immediate",
            },
        }
    )

    nested = StreamingClassifierRuntime(classifier)
    explicit = StreamingClassifierRuntime(
        classifier,
        switch_margin=0.08,
        min_consecutive_windows=2,
    )

    assert nested.switch_margin == pytest.approx(0.12)
    assert nested.hysteresis_margin == pytest.approx(0.12)
    assert nested.min_consecutive_windows == 3
    assert explicit.switch_margin == pytest.approx(0.08)
    assert explicit.min_consecutive_windows == 2

    top_level_classifier = _classifier()
    top_level_classifier.set_routing_metadata(
        router_config={
            "switch_margin": 0.07,
            "min_consecutive_windows": 4,
        }
    )
    top_level = StreamingClassifierRuntime(top_level_classifier)
    assert top_level.switch_margin == pytest.approx(0.07)
    assert top_level.min_consecutive_windows == 4


def test_native_fallback_is_immediate_and_clears_pending_confirmation() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.3,
        switch_margin=0.1,
        min_consecutive_windows=3,
    )
    runtime.process_window(
        _logit_window([0.70, 0.15, 0.10, 0.05]),
        timestamp_seconds=1.0,
    )

    fallback = runtime.process_window(
        _logit_window([0.05, 0.05, 0.10, 0.80]),
        timestamp_seconds=1.5,
    )

    assert fallback.native_fallback_prediction is True
    assert fallback.routed_label == "other"
    assert fallback.pending_route_label is None
    assert fallback.pending_route_count == 0
    assert runtime.state.pending_route_label is None


def test_runtime_reset_clears_all_temporal_state_and_allows_new_timeline() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
        min_consecutive_windows=2,
    )
    runtime.process_window(
        _logit_window([0.8, 0.1, 0.05, 0.05]),
        timestamp_seconds=10.0,
    )

    runtime.reset()

    assert runtime.state.windows_processed == 0
    assert runtime.state.last_timestamp_seconds is None
    assert runtime.state.routed_label == "other"
    assert runtime.state.pending_route_label is None
    assert runtime.state.pending_route_count == 0
    assert runtime.state.smoothed_probabilities == pytest.approx([0.25] * 4)
    pending = runtime.process_window(
        _logit_window([0.1, 0.8, 0.05, 0.05]),
        timestamp_seconds=0.5,
    )
    restarted = runtime.process_window(
        _logit_window([0.1, 0.8, 0.05, 0.05]),
        timestamp_seconds=1.0,
    )
    assert pending.sequence_index == 0
    assert pending.previous_routed_label == "other"
    assert pending.routed_label == "other"
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
    assert state.pending_route_label is None
    assert state.pending_route_count == 0
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


def test_runtime_reports_non_negative_ordered_stage_timings() -> None:
    runtime = StreamingClassifierRuntime(
        _classifier(),
        strategy="hard",
        ema_alpha=1.0,
        confidence_threshold=0.0,
    )

    result = runtime.process_window(
        _logit_window([0.7, 0.1, 0.1, 0.1]),
        timestamp_seconds=1.0,
    )

    stages = [
        result.timings.input_preparation_ms,
        result.timings.device_transfer_ms,
        result.timings.classifier_inference_ms,
        result.timings.routing_ms,
    ]
    assert all(value >= 0.0 for value in stages)
    assert result.timings.total_ms >= sum(stages)
    assert result.as_dict()["timings"]["setup_scope"] == "excluded"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_layout": "unknown"}, "feature_layout"),
        ({"temperature": 0.0}, "temperature"),
        ({"strategy": "unknown"}, "strategy"),
        ({"hysteresis_margin": -0.1}, "hysteresis_margin"),
        ({"switch_margin": -0.1}, "switch_margin"),
        ({"min_consecutive_windows": 0}, "min_consecutive_windows"),
        ({"min_consecutive_windows": 1.5}, "min_consecutive_windows"),
        (
            {"switch_margin": 0.1, "hysteresis_margin": 0.2},
            "must match",
        ),
        (
            {"confidence_threshold": 0.9, "hysteresis_margin": 0.2},
            "must not exceed 1",
        ),
    ],
)
def test_runtime_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        StreamingClassifierRuntime(_classifier(), **kwargs)


@pytest.mark.parametrize(
    ("hysteresis", "match"),
    [
        ("invalid", "must be a mapping"),
        ({"low_confidence_fallback": "delayed"}, "must be 'immediate'"),
    ],
)
def test_runtime_rejects_invalid_checkpoint_hysteresis(
    hysteresis,
    match: str,
) -> None:
    classifier = _classifier()
    classifier.set_routing_metadata(router_config={"hysteresis": hysteresis})

    with pytest.raises(ValueError, match=match):
        StreamingClassifierRuntime(classifier)
