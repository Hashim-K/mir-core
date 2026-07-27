"""Stateful streaming runtime for calibrated genre classification.

The runtime consumes one timestamped feature window at a time. It intentionally
retains only compact routing state (EMA probabilities, the hysteresis-selected
label, a sequence counter, and the last timestamp); feature tensors and prior
results are never cached.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    routed_label: str | None
    smoothed_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StreamingClassifierResult:
    """Classifier and routing output for one timestamped feature window.

    ``policy_weights`` are the exact weights produced by :class:`GenreRouter`
    using identity route activations. ``dominant_route_label`` is their argmax.
    ``routed_label`` is the discrete control label after hysteresis; for soft or
    hybrid routing it can intentionally differ from the continuously varying
    policy-weight argmax while a switch is being held.
    """

    sequence_index: int
    timestamp_seconds: float
    labels: tuple[str, ...]
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    smoothed_probabilities: tuple[float, ...]
    predicted_label: str
    smoothed_label: str
    confidence: float
    smoothed_confidence: float
    policy_weights: tuple[float, ...]
    dominant_route_label: str
    routed_label: str
    previous_routed_label: str | None
    switched: bool
    hysteresis_held: bool
    confidence_rejected: bool
    native_fallback_prediction: bool
    fallback_reason: str | None
    route_mode: str
    strategy: str
    temperature: float

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
        """Return exact GenreRouter policy weights keyed by route label."""

        return dict(zip(self.labels, self.policy_weights, strict=True))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this window result."""

        return {
            "sequence_index": self.sequence_index,
            "timestamp_seconds": self.timestamp_seconds,
            "labels": list(self.labels),
            "logits": list(self.logits),
            "probabilities": self.probabilities_by_label,
            "smoothed_probabilities": self.smoothed_probabilities_by_label,
            "predicted_label": self.predicted_label,
            "smoothed_label": self.smoothed_label,
            "confidence": self.confidence,
            "smoothed_confidence": self.smoothed_confidence,
            "route": {
                "policy_weights": self.policy_weights_by_label,
                "dominant_label": self.dominant_route_label,
                "routed_label": self.routed_label,
                "previous_routed_label": self.previous_routed_label,
                "switched": self.switched,
                "hysteresis_held": self.hysteresis_held,
                "confidence_rejected": self.confidence_rejected,
                "native_fallback_prediction": (self.native_fallback_prediction),
                "fallback_reason": self.fallback_reason,
                "mode": self.route_mode,
                "strategy": self.strategy,
            },
            "temperature": self.temperature,
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
        hysteresis_margin: Required probability advantage before changing a
            non-fallback route. Leaving fallback additionally requires
            ``confidence_threshold + hysteresis_margin``. A low-confidence
            fallback is immediate and is never blocked by hysteresis.

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
        hysteresis_margin: float | None = None,
    ) -> None:
        if not isinstance(classifier, GenreClassifier):
            raise TypeError("classifier must be a GenreClassifier.")
        self._validate_feature_layout(feature_layout)
        router_config = classifier.router_config

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
        resolved_hysteresis = (
            router_config.get("hysteresis_margin", 0.0)
            if hysteresis_margin is None
            else hysteresis_margin
        )
        resolved_hysteresis_number = self._unit_interval(
            resolved_hysteresis,
            name="hysteresis_margin",
        )
        if resolved_confidence_number + resolved_hysteresis_number > 1.0:
            raise ValueError(
                "confidence_threshold + hysteresis_margin must not exceed 1."
            )

        self.classifier = classifier
        self.classifier.eval()
        self.feature_layout = feature_layout
        self.temperature = float(resolved_temperature)
        self.hysteresis_margin = resolved_hysteresis_number
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
        self._windows_processed = 0
        self._last_timestamp_seconds: float | None = None
        self._routed_label: str | None = None

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
            smoothed_probabilities=tuple(
                float(value) for value in self.router.smoothed_probs
            ),
        )

    def reset(self) -> None:
        """Clear EMA, hysteresis, timestamp, and sequence state."""

        self.router.reset()
        self._windows_processed = 0
        self._last_timestamp_seconds = None
        self._routed_label = None

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

    def _apply_hysteresis(
        self,
        candidate_label: str,
        smoothed_probabilities: np.ndarray,
        *,
        confidence_rejected: bool,
    ) -> tuple[str, bool]:
        previous = self._routed_label
        if previous is None or candidate_label == previous:
            return candidate_label, False
        if confidence_rejected:
            return self.router.fallback_label, False

        candidate_probability = float(
            smoothed_probabilities[self._label_to_index[candidate_label]]
        )
        if previous == self.router.fallback_label:
            exit_threshold = self.router.confidence_threshold + self.hysteresis_margin
            if candidate_probability < exit_threshold:
                return previous, True
            return candidate_label, False

        previous_probability = float(
            smoothed_probabilities[self._label_to_index[previous]]
        )
        if candidate_probability < previous_probability + self.hysteresis_margin:
            return previous, True
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

    def process_window(
        self,
        feature_window: np.ndarray | torch.Tensor,
        *,
        timestamp_seconds: float,
        feature_layout: str | None = None,
    ) -> StreamingClassifierResult:
        """Classify and route one transient timestamped feature window."""

        timestamp = self._validated_timestamp(timestamp_seconds)
        prepared = self._prepare_window(
            feature_window,
            layout=(self.feature_layout if feature_layout is None else feature_layout),
        )
        self.classifier.eval()
        with torch.inference_mode():
            logits_tensor = self.classifier(
                prepared.to(
                    device=self._model_device(),
                    dtype=torch.float32,
                )
            )
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
        policy_weights = np.asarray(
            self.router.route(self._route_basis),
            dtype=np.float64,
        )
        predicted_index = int(np.argmax(probabilities))
        smoothed_index = int(np.argmax(smoothed))
        dominant_route_index = int(np.argmax(policy_weights))
        predicted_label = self.classifier.genre_labels[predicted_index]
        smoothed_label = self.classifier.genre_labels[smoothed_index]
        dominant_route_label = self.classifier.genre_labels[dominant_route_index]
        confidence = float(probabilities[predicted_index])
        smoothed_confidence = float(smoothed[smoothed_index])
        confidence_rejected = smoothed_confidence < self.router.confidence_threshold
        previous_routed_label = self._routed_label
        routed_label, hysteresis_held = self._apply_hysteresis(
            dominant_route_label,
            smoothed,
            confidence_rejected=confidence_rejected,
        )
        switched = (
            previous_routed_label is not None and routed_label != previous_routed_label
        )
        native_fallback = smoothed_label == self.router.fallback_label
        if routed_label != self.router.fallback_label:
            fallback_reason = None
        elif confidence_rejected:
            fallback_reason = "low_confidence"
        elif native_fallback:
            fallback_reason = "native_prediction"
        elif hysteresis_held:
            fallback_reason = "hysteresis_hold"
        else:
            fallback_reason = "route_policy"

        result = StreamingClassifierResult(
            sequence_index=self._windows_processed,
            timestamp_seconds=timestamp,
            labels=tuple(self.classifier.genre_labels),
            logits=tuple(float(value) for value in logits),
            probabilities=tuple(float(value) for value in probabilities),
            smoothed_probabilities=tuple(float(value) for value in smoothed),
            predicted_label=predicted_label,
            smoothed_label=smoothed_label,
            confidence=confidence,
            smoothed_confidence=smoothed_confidence,
            policy_weights=tuple(float(value) for value in policy_weights),
            dominant_route_label=dominant_route_label,
            routed_label=routed_label,
            previous_routed_label=previous_routed_label,
            switched=switched,
            hysteresis_held=hysteresis_held,
            confidence_rejected=confidence_rejected,
            native_fallback_prediction=native_fallback,
            fallback_reason=fallback_reason,
            route_mode=self._route_mode(
                confidence_rejected=confidence_rejected,
                smoothed_confidence=smoothed_confidence,
            ),
            strategy=self.router.strategy,
            temperature=self.temperature,
        )
        self._routed_label = routed_label
        self._last_timestamp_seconds = timestamp
        self._windows_processed += 1
        return result


__all__ = [
    "StreamingClassifierResult",
    "StreamingClassifierRuntime",
    "StreamingClassifierState",
]
