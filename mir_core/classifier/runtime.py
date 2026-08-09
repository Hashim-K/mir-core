"""Stateful streaming runtime for calibrated genre classification.

The runtime consumes one timestamped feature window at a time. It intentionally
retains only compact routing state (EMA probabilities, the hysteresis-selected
label, a sequence counter, and the last timestamp); feature tensors and prior
results are never cached.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch

from mir_core.models.classifier.genre_classifier import (
    GenreClassifier,
    GenreRouter,
)

_FEATURE_LAYOUTS = {
    "time_features",
    "features_time",
    "model_input",
}
_ROUTER_STRATEGIES = {"hard", "soft", "hybrid"}


@dataclass(frozen=True, slots=True)
class StreamingClassifierState:
    """A defensive snapshot of the runtime's compact temporal state."""

    windows_processed: int
    last_timestamp_seconds: float | None
    routed_label: str
    pending_route_label: str | None
    pending_route_count: int
    low_confidence_count: int
    route_age_windows: int
    smoothed_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StreamingClassifierTimings:
    """Per-window runtime stages, excluding model/checkpoint setup."""

    input_preparation_ms: float
    device_transfer_ms: float
    classifier_inference_ms: float
    routing_ms: float
    total_ms: float

    def as_dict(self) -> dict[str, float | str]:
        """Return a JSON-compatible timing block."""

        return {
            "input_preparation_ms": self.input_preparation_ms,
            "device_transfer_ms": self.device_transfer_ms,
            "classifier_inference_ms": self.classifier_inference_ms,
            "routing_ms": self.routing_ms,
            "total_ms": self.total_ms,
            "unit": "milliseconds",
            "setup_scope": "excluded",
        }


@dataclass(frozen=True, slots=True)
class StreamingClassifierResult:
    """Classifier and routing output for one timestamped feature window.

    ``policy_weights`` are the raw weights produced by :class:`GenreRouter`
    using identity route activations. ``execution_weights`` are the scheduler
    plan: hard routing and hybrid's hard phase use the confirmed
    ``routed_label`` while soft and hybrid top-2 retain the raw blended weights.
    """

    sequence_index: int
    timestamp_seconds: float
    labels: tuple[str, ...]
    execution_labels: tuple[str, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    smoothed_probabilities: tuple[float, ...]
    predicted_label: str
    smoothed_label: str
    confidence: float
    smoothed_confidence: float
    policy_weights: tuple[float, ...]
    execution_weights: tuple[float, ...]
    dominant_route_label: str
    routed_label: str
    previous_routed_label: str | None
    pending_route_label: str | None
    pending_route_count: int
    min_consecutive_windows: int
    low_confidence_count: int
    low_confidence_hold_windows: int
    route_age_windows: int
    min_dwell_windows: int
    switch_margin: float
    switched: bool
    hysteresis_held: bool
    confidence_rejected: bool
    native_fallback_prediction: bool
    fallback_reason: str | None
    route_mode: str
    strategy: str
    temperature: float
    timings: StreamingClassifierTimings

    @property
    def probabilities_by_label(self) -> dict[str, float]:
        """Return calibrated probabilities keyed by the frozen label order."""

        return dict(zip(self.labels, self.probabilities, strict=True))

    @property
    def smoothed_probabilities_by_label(self) -> dict[str, float]:
        """Return EMA probabilities keyed by the frozen label order."""

        return dict(
            zip(
                self.labels,
                self.smoothed_probabilities,
                strict=True,
            )
        )

    @property
    def policy_weights_by_label(self) -> dict[str, float]:
        """Return raw GenreRouter policy weights keyed by route label."""

        return dict(zip(self.execution_labels, self.policy_weights, strict=True))

    @property
    def execution_weights_by_label(self) -> dict[str, float]:
        """Return final scheduler/execution weights keyed by route label."""

        return dict(zip(self.execution_labels, self.execution_weights, strict=True))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this window result."""

        return {
            "sequence_index": self.sequence_index,
            "timestamp_seconds": self.timestamp_seconds,
            "labels": list(self.labels),
            "execution_labels": list(self.execution_labels),
            "logits": list(self.logits),
            "probabilities": self.probabilities_by_label,
            "smoothed_probabilities": self.smoothed_probabilities_by_label,
            "predicted_label": self.predicted_label,
            "smoothed_label": self.smoothed_label,
            "confidence": self.confidence,
            "smoothed_confidence": self.smoothed_confidence,
            "route": {
                "policy_weights": self.policy_weights_by_label,
                "execution_weights": self.execution_weights_by_label,
                "dominant_label": self.dominant_route_label,
                "routed_label": self.routed_label,
                "previous_routed_label": self.previous_routed_label,
                "pending_label": self.pending_route_label,
                "pending_count": self.pending_route_count,
                "min_consecutive_windows": self.min_consecutive_windows,
                "low_confidence_count": self.low_confidence_count,
                "low_confidence_hold_windows": self.low_confidence_hold_windows,
                "route_age_windows": self.route_age_windows,
                "min_dwell_windows": self.min_dwell_windows,
                "switch_margin": self.switch_margin,
                "switched": self.switched,
                "hysteresis_held": self.hysteresis_held,
                "confidence_rejected": self.confidence_rejected,
                "native_fallback_prediction": (self.native_fallback_prediction),
                "fallback_reason": self.fallback_reason,
                "mode": self.route_mode,
                "strategy": self.strategy,
            },
            "temperature": self.temperature,
            "timings": self.timings.as_dict(),
        }


class StreamingClassifierRuntime:
    """Run a :class:`GenreClassifier` over timestamped streaming windows.

    Args:
        classifier: Loaded classifier. Its label order and calibrated
            temperature are reused.
        feature_layout: Default input layout. ``time_features`` expects
            ``(time, features)``, ``features_time`` expects
            ``(features, time)``, and ``model_input`` accepts one already
            model-shaped item with or without its batch dimension.
        temperature: Optional calibrated softmax temperature override.
        strategy: GenreRouter strategy override.
        ema_alpha: GenreRouter EMA update factor override.
        confidence_threshold: Confidence below which the fallback is selected.
        blend_threshold: Hybrid hard-route threshold. If omitted it defaults to
            at least the confidence threshold.
        fallback_label: Baseline/global route label.
        switch_margin: Required EMA probability advantage before a route can
            enter confirmation.
        min_consecutive_windows: Number of consecutive eligible windows needed
            to confirm a non-fallback entry or switch.
        hysteresis_margin: Compatibility alias for ``switch_margin``.

    The runtime does not extract audio features and does not invoke beat models.
    Callers own the streaming frontend and pass each transient feature window
    directly.
    """

    def __init__(
        self,
        classifier: GenreClassifier,
        *,
        feature_layout: str = "time_features",
        temperature: float | None = None,
        strategy: str | None = None,
        ema_alpha: float | None = None,
        confidence_threshold: float | None = None,
        blend_threshold: float | None = None,
        fallback_label: str | None = None,
        switch_margin: float | None = None,
        min_consecutive_windows: int | None = None,
        hysteresis_margin: float | None = None,
        execution_labels: tuple[str, ...] | list[str] | None = None,
        classifier_to_route: Mapping[str, str] | None = None,
        uncertainty_fallback_label: str | None = None,
        native_fallback_route_label: str | None = None,
        low_confidence_fallback: str | None = None,
        low_confidence_hold_windows: int | None = None,
        min_dwell_windows: int | None = None,
    ) -> None:
        if not isinstance(classifier, GenreClassifier):
            raise TypeError("classifier must be a GenreClassifier.")
        self._validate_feature_layout(feature_layout)
        router_config = classifier.router_config
        raw_hysteresis_config = router_config.get("hysteresis", {})
        if raw_hysteresis_config is None:
            raw_hysteresis_config = {}
        if not isinstance(raw_hysteresis_config, Mapping):
            raise ValueError("router_config.hysteresis must be a mapping.")
        hysteresis_config = dict(raw_hysteresis_config)
        resolved_low_confidence_fallback = str(
            low_confidence_fallback
            if low_confidence_fallback is not None
            else hysteresis_config.get(
                "low_confidence_fallback",
                router_config.get("low_confidence_fallback", "immediate"),
            )
        )
        if resolved_low_confidence_fallback not in {
            "immediate",
            "hold_then_uncertainty",
        }:
            raise ValueError(
                "low_confidence_fallback must be 'immediate' or "
                "'hold_then_uncertainty'."
            )

        resolved_temperature = (
            classifier.calibration_temperature
            if temperature is None
            else self._positive_finite(temperature, name="temperature")
        )
        resolved_strategy = str(
            strategy
            if strategy is not None
            else router_config.get("strategy", "hybrid")
        )
        if resolved_strategy not in _ROUTER_STRATEGIES:
            raise ValueError("strategy must be 'hard', 'soft', or 'hybrid'.")
        resolved_ema_alpha = (
            router_config.get("ema_alpha", 0.3) if ema_alpha is None else ema_alpha
        )
        resolved_confidence = (
            router_config.get("confidence_threshold", 0.7)
            if confidence_threshold is None
            else confidence_threshold
        )
        resolved_confidence_number = self._unit_interval(
            resolved_confidence,
            name="confidence_threshold",
        )
        configured_blend = (
            router_config.get("blend_threshold")
            if blend_threshold is None
            else blend_threshold
        )
        resolved_blend = (
            max(0.8, resolved_confidence_number)
            if configured_blend is None
            else configured_blend
        )
        resolved_fallback = str(
            fallback_label
            if fallback_label is not None
            else router_config.get(
                "fallback_label",
                (
                    "other"
                    if "other" in classifier.genre_labels
                    else classifier.genre_labels[-1]
                ),
            )
        )
        if switch_margin is not None and hysteresis_margin is not None:
            canonical_margin = self._unit_interval(
                switch_margin,
                name="switch_margin",
            )
            alias_margin = self._unit_interval(
                hysteresis_margin,
                name="hysteresis_margin",
            )
            if canonical_margin != alias_margin:
                raise ValueError(
                    "switch_margin and hysteresis_margin must match when both "
                    "are provided."
                )
            explicit_margin: Any | None = canonical_margin
        elif switch_margin is not None:
            explicit_margin = switch_margin
        else:
            explicit_margin = hysteresis_margin
        configured_margin = hysteresis_config.get("switch_margin")
        if configured_margin is None:
            configured_margin = router_config.get("switch_margin")
        if configured_margin is None:
            configured_margin = router_config.get("hysteresis_margin", 0.0)
        resolved_switch_margin = (
            configured_margin if explicit_margin is None else explicit_margin
        )
        margin_name = (
            "hysteresis_margin"
            if switch_margin is None and hysteresis_margin is not None
            else "switch_margin"
        )
        resolved_switch_margin_number = self._unit_interval(
            resolved_switch_margin,
            name=margin_name,
        )
        configured_minimum = hysteresis_config.get("min_consecutive_windows")
        if configured_minimum is None:
            configured_minimum = router_config.get(
                "min_consecutive_windows",
                1,
            )
        resolved_minimum = (
            configured_minimum
            if min_consecutive_windows is None
            else min_consecutive_windows
        )
        resolved_minimum_number = self._positive_integer(
            resolved_minimum,
            name="min_consecutive_windows",
        )
        configured_hold = hysteresis_config.get("low_confidence_hold_windows", 0)
        resolved_hold = (
            configured_hold
            if low_confidence_hold_windows is None
            else low_confidence_hold_windows
        )
        resolved_hold_number = self._non_negative_integer(
            resolved_hold,
            name="low_confidence_hold_windows",
        )
        if resolved_low_confidence_fallback == "immediate" and resolved_hold_number:
            raise ValueError(
                "low_confidence_hold_windows must be zero when "
                "low_confidence_fallback is 'immediate'."
            )
        configured_dwell = hysteresis_config.get(
            "min_dwell_windows",
            router_config.get("min_dwell_windows", 0),
        )
        resolved_dwell = (
            configured_dwell if min_dwell_windows is None else min_dwell_windows
        )
        resolved_dwell_number = self._non_negative_integer(
            resolved_dwell,
            name="min_dwell_windows",
        )
        if resolved_confidence_number + resolved_switch_margin_number > 1.0:
            raise ValueError("confidence_threshold + switch_margin must not exceed 1.")

        raw_execution_config = router_config.get("execution_routes", {})
        if raw_execution_config is None:
            raw_execution_config = {}
        if not isinstance(raw_execution_config, Mapping):
            raise ValueError("router_config.execution_routes must be a mapping.")
        execution_config = dict(raw_execution_config)
        raw_route_map = (
            classifier_to_route
            if classifier_to_route is not None
            else execution_config.get("classifier_to_route")
        )
        if raw_route_map is None:
            route_map = {label: label for label in classifier.genre_labels}
        elif not isinstance(raw_route_map, Mapping):
            raise ValueError("classifier_to_route must be a mapping.")
        else:
            route_map = {str(key): str(value) for key, value in raw_route_map.items()}
        if set(route_map) != set(classifier.genre_labels):
            raise ValueError(
                "classifier_to_route must contain every classifier label exactly once."
            )
        if any(not label for label in route_map.values()):
            raise ValueError("classifier_to_route values must be non-empty labels.")
        resolved_uncertainty = str(
            uncertainty_fallback_label
            if uncertainty_fallback_label is not None
            else execution_config.get("uncertainty_fallback_label", resolved_fallback)
        )
        resolved_native_fallback_route = str(
            native_fallback_route_label
            if native_fallback_route_label is not None
            else execution_config.get(
                "native_fallback_route_label",
                route_map[resolved_fallback],
            )
        )
        if route_map[resolved_fallback] != resolved_native_fallback_route:
            raise ValueError(
                "native_fallback_route_label must match the fallback classifier "
                "label's classifier_to_route mapping."
            )
        raw_execution_labels = (
            execution_labels
            if execution_labels is not None
            else execution_config.get("labels")
        )
        if raw_execution_labels is None:
            inferred = list(dict.fromkeys(route_map[label] for label in classifier.genre_labels))
            if resolved_uncertainty not in inferred:
                inferred.append(resolved_uncertainty)
            resolved_execution_labels = tuple(inferred)
        else:
            if isinstance(raw_execution_labels, (str, bytes)):
                raise ValueError("execution_labels must be a sequence of labels.")
            resolved_execution_labels = tuple(str(label) for label in raw_execution_labels)
        if (
            not resolved_execution_labels
            or any(not label for label in resolved_execution_labels)
            or len(set(resolved_execution_labels)) != len(resolved_execution_labels)
        ):
            raise ValueError("execution_labels must be non-empty and unique.")
        required_execution_labels = set(route_map.values()) | {resolved_uncertainty}
        if not required_execution_labels.issubset(resolved_execution_labels):
            raise ValueError(
                "execution_labels must include every mapped and uncertainty route."
            )

        self.classifier = classifier
        self.classifier.eval()
        self.feature_layout = feature_layout
        self.temperature = float(resolved_temperature)
        self.switch_margin = resolved_switch_margin_number
        self.hysteresis_margin = resolved_switch_margin_number
        self.min_consecutive_windows = resolved_minimum_number
        self.low_confidence_hold_windows = resolved_hold_number
        self.min_dwell_windows = resolved_dwell_number
        self.low_confidence_fallback = resolved_low_confidence_fallback
        self.execution_labels = resolved_execution_labels
        self.classifier_to_route = route_map
        self.uncertainty_fallback_label = resolved_uncertainty
        self.native_fallback_route_label = resolved_native_fallback_route
        self.router = GenreRouter(
            genre_labels=list(classifier.genre_labels),
            strategy=resolved_strategy,
            ema_alpha=resolved_ema_alpha,
            confidence_threshold=resolved_confidence_number,
            blend_threshold=resolved_blend,
            fallback_label=resolved_fallback,
        )
        identity = np.eye(len(classifier.genre_labels), dtype=np.float64)
        self._route_basis = {
            label: identity[index]
            for index, label in enumerate(classifier.genre_labels)
        }
        self._label_to_index = {
            label: index for index, label in enumerate(classifier.genre_labels)
        }
        self._execution_label_to_index = {
            label: index for index, label in enumerate(self.execution_labels)
        }
        self._route_to_classifier_indices: dict[str, tuple[int, ...]] = {
            route: tuple(
                self._label_to_index[label]
                for label in classifier.genre_labels
                if self.classifier_to_route[label] == route
            )
            for route in self.execution_labels
        }
        self._windows_processed = 0
        self._last_timestamp_seconds: float | None = None
        self._routed_label = self.uncertainty_fallback_label
        self._pending_route_label: str | None = None
        self._pending_route_count = 0
        self._low_confidence_count = 0
        self._route_age_windows = 0

    @staticmethod
    def _positive_finite(value: Any, *, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite scalar.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite scalar.") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a positive finite scalar.")
        return number

    @staticmethod
    def _unit_interval(value: Any, *, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and between 0 and 1.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and between 0 and 1.") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1.")
        return number

    @staticmethod
    def _positive_integer(value: Any, *, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer.")
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer.") from exc
        if number <= 0 or number != value:
            raise ValueError(f"{name} must be a positive integer.")
        return number

    @staticmethod
    def _non_negative_integer(value: Any, *, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a non-negative integer.")
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a non-negative integer.") from exc
        if number < 0 or number != value:
            raise ValueError(f"{name} must be a non-negative integer.")
        return number

    @staticmethod
    def _validate_feature_layout(layout: str) -> None:
        if layout not in _FEATURE_LAYOUTS:
            raise ValueError(
                "feature_layout must be 'time_features', 'features_time', "
                "or 'model_input'."
            )

    @property
    def state(self) -> StreamingClassifierState:
        """Return a defensive snapshot without retaining any feature data."""

        return StreamingClassifierState(
            windows_processed=self._windows_processed,
            last_timestamp_seconds=self._last_timestamp_seconds,
            routed_label=self._routed_label,
            pending_route_label=self._pending_route_label,
            pending_route_count=self._pending_route_count,
            low_confidence_count=self._low_confidence_count,
            route_age_windows=self._route_age_windows,
            smoothed_probabilities=tuple(
                float(value) for value in self.router.smoothed_probs
            ),
        )

    def reset(self) -> None:
        """Clear EMA, hysteresis, timestamp, and sequence state."""

        self.router.reset()
        self._windows_processed = 0
        self._last_timestamp_seconds = None
        self._routed_label = self.uncertainty_fallback_label
        self._pending_route_label = None
        self._pending_route_count = 0
        self._low_confidence_count = 0
        self._route_age_windows = 0

    def _validated_timestamp(self, timestamp_seconds: Any) -> float:
        if isinstance(timestamp_seconds, bool):
            raise ValueError("timestamp_seconds must be a finite non-negative scalar.")
        try:
            timestamp = float(timestamp_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "timestamp_seconds must be a finite non-negative scalar."
            ) from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_seconds must be a finite non-negative scalar.")
        if (
            self._last_timestamp_seconds is not None
            and timestamp <= self._last_timestamp_seconds
        ):
            raise ValueError(
                "timestamp_seconds must increase strictly between windows; "
                "call reset() before starting a new stream."
            )
        return timestamp

    def _prepare_window(
        self,
        feature_window: np.ndarray | torch.Tensor,
        *,
        layout: str,
    ) -> torch.Tensor:
        self._validate_feature_layout(layout)
        try:
            tensor = torch.as_tensor(feature_window)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "feature_window must be a finite numeric array or tensor."
            ) from exc
        if tensor.numel() == 0 or tensor.is_complex():
            raise ValueError(
                "feature_window must be a non-empty finite real-valued tensor."
            )
        tensor = tensor.to(dtype=torch.float32)
        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                "feature_window must be a non-empty finite real-valued tensor."
            )

        beatnet_conv = self.classifier.arch_name == "beatnet_conv"
        if layout == "time_features":
            if tensor.ndim != 2:
                raise ValueError(
                    "time_features input must have shape (time, features)."
                )
            prepared = (
                tensor.unsqueeze(0)
                if beatnet_conv
                else tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            )
        elif layout == "features_time":
            if tensor.ndim != 2:
                raise ValueError(
                    "features_time input must have shape (features, time)."
                )
            prepared = (
                tensor.transpose(0, 1).unsqueeze(0)
                if beatnet_conv
                else tensor.unsqueeze(0).unsqueeze(0)
            )
        elif beatnet_conv:
            if tensor.ndim == 2:
                prepared = tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                prepared = tensor
            else:
                raise ValueError(
                    "beatnet_conv model_input must have shape "
                    "(time, features) or (1, time, features)."
                )
        else:
            if tensor.ndim == 2:
                prepared = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 3:
                prepared = tensor.unsqueeze(0)
            elif tensor.ndim == 4:
                prepared = tensor
            else:
                raise ValueError(
                    "model_input must have shape (features, time), "
                    "(channels, features, time), or "
                    "(1, channels, features, time)."
                )
        if prepared.shape[0] != 1:
            raise ValueError(
                "A streaming feature window must contain exactly one batch item."
            )
        return prepared.contiguous()

    def _model_device(self) -> torch.device:
        parameter = next(self.classifier.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.classifier.buffers(), None)
        return buffer.device if buffer is not None else torch.device("cpu")

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0

    def _clear_pending_route(self) -> None:
        self._pending_route_label = None
        self._pending_route_count = 0

    def _confirm_route(
        self,
        candidate_label: str,
        smoothed_probabilities: np.ndarray,
        *,
        confidence_rejected: bool,
    ) -> tuple[str, bool]:
        current = self._routed_label
        if confidence_rejected:
            self._low_confidence_count += 1
            self._clear_pending_route()
            if self._low_confidence_count <= self.low_confidence_hold_windows:
                return current, True
            return self.uncertainty_fallback_label, False
        self._low_confidence_count = 0
        if candidate_label == current:
            self._clear_pending_route()
            return current, False

        candidate_indices = self._route_to_classifier_indices[candidate_label]
        candidate_probability = float(
            sum(smoothed_probabilities[index] for index in candidate_indices)
        )
        current_indices = self._route_to_classifier_indices[current]
        reference_probability = (
            self.router.confidence_threshold
            if current == self.uncertainty_fallback_label
            else float(
                sum(smoothed_probabilities[index] for index in current_indices)
            )
        )
        if candidate_probability - reference_probability < self.switch_margin:
            self._clear_pending_route()
            return current, True

        if self._pending_route_label == candidate_label:
            self._pending_route_count += 1
        else:
            self._pending_route_label = candidate_label
            self._pending_route_count = 1
        if (
            self._pending_route_count < self.min_consecutive_windows
            or self._route_age_windows < self.min_dwell_windows
        ):
            return current, True

        self._clear_pending_route()
        return candidate_label, False

    def _route_mode(
        self,
        *,
        confidence_rejected: bool,
        smoothed_confidence: float,
    ) -> str:
        if confidence_rejected:
            return "confidence_fallback"
        if self.router.strategy == "hard":
            return "hard"
        if self.router.strategy == "soft":
            return "soft_blend"
        if smoothed_confidence >= self.router.blend_threshold:
            return "hard"
        return "top2_blend"

    def _execution_weights(
        self,
        policy_weights: np.ndarray,
        *,
        routed_label: str,
        route_mode: str,
    ) -> np.ndarray:
        if self.router.strategy == "hard" or (
            self.router.strategy == "hybrid" and route_mode == "hard"
        ):
            weights = np.zeros(len(self.execution_labels), dtype=np.float64)
            weights[self._execution_label_to_index[routed_label]] = 1.0
            return weights
        return np.array(policy_weights, dtype=np.float64, copy=True)

    def _map_policy_weights(
        self,
        classifier_weights: np.ndarray,
        *,
        confidence_rejected: bool,
    ) -> np.ndarray:
        weights = np.zeros(len(self.execution_labels), dtype=np.float64)
        if confidence_rejected:
            weights[self._execution_label_to_index[self.uncertainty_fallback_label]] = 1.0
            return weights
        for classifier_label, value in zip(
            self.classifier.genre_labels,
            classifier_weights,
            strict=True,
        ):
            route_label = self.classifier_to_route[classifier_label]
            weights[self._execution_label_to_index[route_label]] += float(value)
        return weights

    def process_window(
        self,
        feature_window: np.ndarray | torch.Tensor,
        *,
        timestamp_seconds: float,
        feature_layout: str | None = None,
    ) -> StreamingClassifierResult:
        """Classify and route one transient timestamped feature window."""

        total_started = time.perf_counter_ns()
        timestamp = self._validated_timestamp(timestamp_seconds)
        if isinstance(feature_window, torch.Tensor):
            self._synchronize(feature_window.device)
        preparation_started = time.perf_counter_ns()
        prepared = self._prepare_window(
            feature_window,
            layout=(self.feature_layout if feature_layout is None else feature_layout),
        )
        self._synchronize(prepared.device)
        input_preparation_ms = self._elapsed_ms(preparation_started)

        model_device = self._model_device()
        self._synchronize(model_device)
        transfer_started = time.perf_counter_ns()
        model_input = prepared.to(
            device=model_device,
            dtype=torch.float32,
        )
        self._synchronize(model_device)
        device_transfer_ms = self._elapsed_ms(transfer_started)

        self.classifier.eval()
        self._synchronize(model_device)
        inference_started = time.perf_counter_ns()
        with torch.inference_mode():
            logits_tensor = self.classifier(model_input)
        self._synchronize(model_device)
        classifier_inference_ms = self._elapsed_ms(inference_started)

        routing_started = time.perf_counter_ns()
        if logits_tensor.ndim != 2 or logits_tensor.shape != (
            1,
            len(self.classifier.genre_labels),
        ):
            raise RuntimeError(
                "Streaming classifier output must have shape " "(1, number_of_labels)."
            )
        logits_tensor = logits_tensor.detach().float()
        if not torch.isfinite(logits_tensor).all().item():
            raise RuntimeError("Streaming classifier produced non-finite logits.")
        probabilities_tensor = torch.softmax(
            logits_tensor / self.temperature,
            dim=-1,
        )
        logits = logits_tensor[0].cpu().numpy().astype(np.float64)
        probabilities = probabilities_tensor[0].cpu().numpy().astype(np.float64)

        smoothed = self.router.update_probs(probabilities)
        classifier_policy_weights = np.asarray(
            self.router.route(self._route_basis),
            dtype=np.float64,
        )
        predicted_index = int(np.argmax(probabilities))
        smoothed_index = int(np.argmax(smoothed))
        predicted_label = self.classifier.genre_labels[predicted_index]
        smoothed_label = self.classifier.genre_labels[smoothed_index]
        dominant_route_label = self.classifier_to_route[smoothed_label]
        confidence = float(probabilities[predicted_index])
        smoothed_confidence = float(smoothed[smoothed_index])
        confidence_rejected = smoothed_confidence < self.router.confidence_threshold
        previous_routed_label = self._routed_label
        native_fallback = smoothed_label == self.router.fallback_label
        policy_weights = self._map_policy_weights(
            classifier_policy_weights,
            confidence_rejected=confidence_rejected,
        )
        routed_label, hysteresis_held = self._confirm_route(
            dominant_route_label,
            smoothed,
            confidence_rejected=confidence_rejected,
        )
        switched = routed_label != previous_routed_label
        route_mode = self._route_mode(
            confidence_rejected=confidence_rejected,
            smoothed_confidence=smoothed_confidence,
        )
        execution_weights = self._execution_weights(
            policy_weights,
            routed_label=routed_label,
            route_mode=route_mode,
        )
        if confidence_rejected and hysteresis_held:
            fallback_reason = "low_confidence_hold"
        elif confidence_rejected:
            fallback_reason = "low_confidence"
        elif native_fallback and routed_label == self.native_fallback_route_label:
            fallback_reason = "native_prediction"
        elif hysteresis_held:
            fallback_reason = "confirmation_hold"
        else:
            fallback_reason = None
        self._routed_label = routed_label
        self._route_age_windows = 1 if switched else self._route_age_windows + 1
        self._last_timestamp_seconds = timestamp
        self._windows_processed += 1
        routing_ms = self._elapsed_ms(routing_started)
        timings = StreamingClassifierTimings(
            input_preparation_ms=input_preparation_ms,
            device_transfer_ms=device_transfer_ms,
            classifier_inference_ms=classifier_inference_ms,
            routing_ms=routing_ms,
            total_ms=self._elapsed_ms(total_started),
        )

        result = StreamingClassifierResult(
            sequence_index=self._windows_processed - 1,
            timestamp_seconds=timestamp,
            labels=tuple(self.classifier.genre_labels),
            execution_labels=self.execution_labels,
            logits=tuple(float(value) for value in logits),
            probabilities=tuple(float(value) for value in probabilities),
            smoothed_probabilities=tuple(float(value) for value in smoothed),
            predicted_label=predicted_label,
            smoothed_label=smoothed_label,
            confidence=confidence,
            smoothed_confidence=smoothed_confidence,
            policy_weights=tuple(float(value) for value in policy_weights),
            execution_weights=tuple(float(value) for value in execution_weights),
            dominant_route_label=dominant_route_label,
            routed_label=routed_label,
            previous_routed_label=previous_routed_label,
            pending_route_label=self._pending_route_label,
            pending_route_count=self._pending_route_count,
            min_consecutive_windows=self.min_consecutive_windows,
            low_confidence_count=self._low_confidence_count,
            low_confidence_hold_windows=self.low_confidence_hold_windows,
            route_age_windows=self._route_age_windows,
            min_dwell_windows=self.min_dwell_windows,
            switch_margin=self.switch_margin,
            switched=switched,
            hysteresis_held=hysteresis_held,
            confidence_rejected=confidence_rejected,
            native_fallback_prediction=native_fallback,
            fallback_reason=fallback_reason,
            route_mode=route_mode,
            strategy=self.router.strategy,
            temperature=self.temperature,
            timings=timings,
        )
        return result


__all__ = [
    "StreamingClassifierResult",
    "StreamingClassifierRuntime",
    "StreamingClassifierState",
    "StreamingClassifierTimings",
]
