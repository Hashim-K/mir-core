"""Deterministic confidence calibration and fallback-threshold selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .metrics import accuracy, macro_f1


def _positive_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive finite scalar.") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be a positive finite scalar.")
    return value


def _score_matrix(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric score matrix.") from exc
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must have shape (N, C) with N > 0 and C >= 2.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _target_indices(targets: Any, *, num_samples: int, num_classes: int) -> np.ndarray:
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    try:
        values = np.asarray(targets)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets must be array-like class indices.") from exc
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim == 2 and values.shape[1] == num_classes:
        try:
            one_hot = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("targets one-hot rows must be numeric.") from exc
        if (
            not np.isfinite(one_hot).all()
            or (one_hot < 0).any()
            or not np.allclose(one_hot.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError("targets one-hot rows must be non-negative and sum to 1.")
        values = np.argmax(one_hot, axis=1)
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("targets must be class indices or one-hot rows.")
    if values.shape[0] != num_samples:
        raise ValueError(
            "scores and targets must contain the same number of samples "
            f"({num_samples} != {values.shape[0]})."
        )
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets class indices must be numeric.") from exc
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, np.rint(numeric)):
        raise ValueError("targets class indices must be finite integers.")
    indices = np.rint(numeric).astype(np.int64)
    if (indices < 0).any() or (indices >= num_classes).any():
        raise ValueError(f"targets class indices must be in [0, {num_classes - 1}].")
    return indices


def _probability_matrix(
    values: Any,
    *,
    from_logits: bool,
    temperature: float,
) -> np.ndarray:
    scores = _score_matrix(values, name="logits" if from_logits else "probabilities")
    if from_logits:
        scaled = scores / _positive_temperature(temperature)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
    if temperature != 1.0:
        raise ValueError("temperature can only be applied when from_logits=True.")
    if (scores < 0).any():
        raise ValueError("probabilities must be non-negative.")
    row_sums = scores.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("probability rows must sum to 1.")
    return scores / row_sums


def apply_temperature(logits: Any, temperature: float) -> Any:
    """Divide logits by a positive scalar temperature.

    A torch tensor produces a torch tensor on the same device; other array-like
    input produces a NumPy array. A temperature above one softens confidence.
    """

    value = _positive_temperature(temperature)
    if isinstance(logits, torch.Tensor):
        if logits.ndim not in (1, 2) or logits.shape[-1] < 2:
            raise ValueError("logits must have shape (C,) or (N, C) with C >= 2.")
        if not torch.isfinite(logits).all().item():
            raise ValueError("logits must contain only finite values.")
        return logits / value
    scores = _score_matrix(logits, name="logits")
    scaled = scores / value
    return scaled[0] if np.asarray(logits).ndim == 1 else scaled


def softmax_probabilities(logits: Any, *, temperature: float = 1.0) -> np.ndarray:
    """Return stable softmax probabilities after optional temperature scaling."""

    return _probability_matrix(logits, from_logits=True, temperature=temperature)


def negative_log_likelihood(
    values: Any,
    targets: Any,
    *,
    from_logits: bool = True,
    temperature: float = 1.0,
) -> float:
    """Return mean multiclass negative log-likelihood.

    ``values`` are logits by default. Set ``from_logits=False`` for calibrated
    probabilities; in that mode ``temperature`` must remain one.
    """

    probabilities = _probability_matrix(
        values,
        from_logits=from_logits,
        temperature=temperature,
    )
    target_indices = _target_indices(
        targets,
        num_samples=probabilities.shape[0],
        num_classes=probabilities.shape[1],
    )
    selected = probabilities[np.arange(probabilities.shape[0]), target_indices]
    return float(-np.log(np.clip(selected, np.finfo(np.float64).tiny, 1.0)).mean())


def expected_calibration_error(
    values: Any,
    targets: Any,
    *,
    n_bins: int = 15,
    from_logits: bool = True,
    temperature: float = 1.0,
) -> float:
    """Return top-label expected calibration error using equal-width bins."""

    if isinstance(n_bins, bool) or int(n_bins) != n_bins or int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer.")
    probabilities = _probability_matrix(
        values,
        from_logits=from_logits,
        temperature=temperature,
    )
    target_indices = _target_indices(
        targets,
        num_samples=probabilities.shape[0],
        num_classes=probabilities.shape[1],
    )
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == target_indices
    bins = np.minimum(
        (confidence * int(n_bins)).astype(np.int64),
        int(n_bins) - 1,
    )
    error = 0.0
    for bin_index in range(int(n_bins)):
        mask = bins == bin_index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(error)


def fit_temperature(
    logits: Any,
    targets: Any,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    max_iter: int = 100,
) -> float:
    """Fit one deterministic scalar temperature by minimizing validation NLL.

    Optimization uses a bounded golden-section search in log-temperature space,
    avoiding optimizer state and random initialization. If scaling does not
    improve NLL, the identity temperature ``1.0`` is returned.
    """

    scores = _score_matrix(logits, name="logits")
    target_indices = _target_indices(
        targets,
        num_samples=scores.shape[0],
        num_classes=scores.shape[1],
    )
    lower = _positive_temperature(min_temperature)
    upper = _positive_temperature(max_temperature)
    if lower >= upper:
        raise ValueError("min_temperature must be smaller than max_temperature.")
    if not lower <= 1.0 <= upper:
        raise ValueError("temperature bounds must include the identity value 1.0.")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer.")

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(
            scores,
            target_indices,
            temperature=float(np.exp(log_temperature)),
        )

    lo = float(np.log(lower))
    hi = float(np.log(upper))
    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    left = hi - inverse_phi * (hi - lo)
    right = lo + inverse_phi * (hi - lo)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(int(max_iter)):
        if left_value <= right_value:
            hi = right
            right = left
            right_value = left_value
            left = hi - inverse_phi * (hi - lo)
            left_value = objective(left)
        else:
            lo = left
            left = right
            left_value = right_value
            right = lo + inverse_phi * (hi - lo)
            right_value = objective(right)

    candidates = [1.0, lower, upper, float(np.exp((lo + hi) / 2.0))]
    scored = [
        (negative_log_likelihood(scores, target_indices, temperature=value), value)
        for value in candidates
    ]
    identity_nll = scored[0][0]
    best_nll, best_temperature = min(
        scored,
        key=lambda item: (item[0], abs(np.log(item[1]))),
    )
    tolerance = 1e-12 * max(1.0, abs(identity_nll))
    return float(best_temperature if best_nll < identity_nll - tolerance else 1.0)


@dataclass(frozen=True)
class RoutingThresholdSelection:
    """Validation result for a confidence-based fallback threshold."""

    threshold: float
    objective: str
    score: float
    coverage: float
    fallback_rate: float
    accuracy: float
    macro_f1: float
    target_accept_recall: float
    out_of_set_rejection: float


def _threshold_candidates(
    confidence: np.ndarray,
    candidates: Sequence[float] | None,
) -> np.ndarray:
    if candidates is None:
        above_observed = np.nextafter(np.unique(confidence), np.inf)
        values = np.concatenate(
            (
                np.asarray([0.0]),
                above_observed[above_observed <= 1.0],
                np.asarray([1.0]),
            )
        )
        return np.unique(values)
    try:
        values = np.asarray(tuple(candidates), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidates must contain numeric thresholds.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("candidates must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("candidate thresholds must be finite and between 0 and 1.")
    return np.unique(values)


def select_routing_threshold(
    values: Any,
    targets: Any,
    *,
    fallback_class: int | None = None,
    objective: str = "macro_f1",
    candidates: Sequence[float] | None = None,
    min_coverage: float = 0.0,
    from_logits: bool = True,
    temperature: float = 1.0,
) -> RoutingThresholdSelection:
    """Select a validation confidence threshold for fallback routing.

    Samples with maximum probability below the threshold are routed to
    ``fallback_class`` (the final class by default). Supported objectives are
    ``"accuracy"``, ``"macro_f1"``, and ``"balanced_router"``. The latter is
    the mean of target-class acceptance recall and out-of-set rejection.

    Ties prefer greater classifier coverage and then the lower threshold, which
    avoids fallback when it offers no measured validation benefit.
    """

    probabilities = _probability_matrix(
        values,
        from_logits=from_logits,
        temperature=temperature,
    )
    target_indices = _target_indices(
        targets,
        num_samples=probabilities.shape[0],
        num_classes=probabilities.shape[1],
    )
    num_classes = probabilities.shape[1]
    fallback = num_classes - 1 if fallback_class is None else fallback_class
    if isinstance(fallback, bool) or int(fallback) != fallback:
        raise ValueError("fallback_class must be an integer class index.")
    fallback = int(fallback)
    if fallback < 0 or fallback >= num_classes:
        raise ValueError(f"fallback_class must be in [0, {num_classes - 1}].")
    if objective not in {"accuracy", "macro_f1", "balanced_router"}:
        raise ValueError(
            "objective must be 'accuracy', 'macro_f1', or 'balanced_router'."
        )
    try:
        required_coverage = float(min_coverage)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_coverage must be between 0 and 1.") from exc
    if (
        not np.isfinite(required_coverage)
        or required_coverage < 0
        or required_coverage > 1
    ):
        raise ValueError("min_coverage must be between 0 and 1.")

    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    target_mask = target_indices != fallback
    fallback_mask = ~target_mask
    results: list[RoutingThresholdSelection] = []
    for threshold in _threshold_candidates(confidence, candidates):
        accepted = confidence >= threshold
        coverage = float(accepted.mean())
        if coverage + 1e-15 < required_coverage:
            continue
        routed = np.where(accepted, predictions, fallback)
        route_accuracy = accuracy(routed, target_indices)
        route_macro_f1 = macro_f1(
            routed,
            target_indices,
            num_classes=num_classes,
        )
        target_accept = (
            float((routed[target_mask] != fallback).mean())
            if target_mask.any()
            else 1.0
        )
        out_of_set_rejection = (
            float((routed[fallback_mask] == fallback).mean())
            if fallback_mask.any()
            else 1.0
        )
        scores = {
            "accuracy": route_accuracy,
            "macro_f1": route_macro_f1,
            "balanced_router": (target_accept + out_of_set_rejection) / 2.0,
        }
        results.append(
            RoutingThresholdSelection(
                threshold=float(threshold),
                objective=objective,
                score=float(scores[objective]),
                coverage=coverage,
                fallback_rate=float((routed == fallback).mean()),
                accuracy=route_accuracy,
                macro_f1=route_macro_f1,
                target_accept_recall=target_accept,
                out_of_set_rejection=out_of_set_rejection,
            )
        )
    if not results:
        raise ValueError("No candidate threshold satisfies min_coverage.")
    return max(
        results,
        key=lambda result: (
            result.score,
            result.coverage,
            -result.threshold,
        ),
    )


select_fallback_threshold = select_routing_threshold
