"""Dependency-light classification metrics with strict input validation.

The functions accept class-index predictions or a two-dimensional score matrix.
Targets may be class indices or a two-dimensional one-hot matrix. Class indices
are intentionally required to be non-negative integers so label ordering stays
unambiguous across training, evaluation, and deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def _as_numpy(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    try:
        return np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like.") from exc


def _class_indices(values: Any, *, name: str, scores_allowed: bool) -> np.ndarray:
    array = _as_numpy(values, name=name)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2:
        if array.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one class.")
        if array.shape[1] == 1:
            array = array[:, 0]
        elif scores_allowed:
            try:
                finite = np.isfinite(array)
            except TypeError as exc:
                raise ValueError(f"{name} scores must be numeric.") from exc
            if not finite.all():
                raise ValueError(f"{name} scores must be finite.")
            return np.asarray(np.argmax(array, axis=1), dtype=np.int64)
        else:
            try:
                numeric = np.asarray(array, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must contain class indices or one-hot rows."
                ) from exc
            if not np.isfinite(numeric).all():
                raise ValueError(f"{name} must be finite.")
            if (numeric < 0).any() or not np.allclose(
                numeric.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"{name} one-hot rows must be non-negative and sum to 1."
                )
            is_binary = np.logical_or(
                np.isclose(numeric, 0.0, rtol=0.0, atol=1e-8),
                np.isclose(numeric, 1.0, rtol=0.0, atol=1e-8),
            )
            if not is_binary.all():
                raise ValueError(f"{name} one-hot rows may only contain 0 and 1.")
            return np.asarray(np.argmax(numeric, axis=1), dtype=np.int64)
    elif array.ndim != 1:
        expected = (
            "class indices or a score matrix" if scores_allowed else "class indices"
        )
        raise ValueError(f"{name} must be one-dimensional {expected}.")

    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} class indices must be numeric.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} class indices must be finite.")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} class indices must be integers.")
    indices = rounded.astype(np.int64)
    if (indices < 0).any():
        raise ValueError(f"{name} class indices must be non-negative.")
    return indices


def _prediction_target_indices(
    predictions: Any,
    targets: Any,
    *,
    allow_empty: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = _class_indices(predictions, name="predictions", scores_allowed=True)
    expected = _class_indices(targets, name="targets", scores_allowed=False)
    if predicted.shape[0] != expected.shape[0]:
        raise ValueError(
            "predictions and targets must contain the same number of samples "
            f"({predicted.shape[0]} != {expected.shape[0]})."
        )
    if not allow_empty and predicted.size == 0:
        raise ValueError("predictions and targets must not be empty.")
    return predicted, expected


def _label_indices(
    predicted: np.ndarray,
    targets: np.ndarray,
    *,
    labels: Sequence[int] | None,
    num_classes: int | None,
) -> np.ndarray:
    if labels is not None and num_classes is not None:
        raise ValueError("Pass either labels or num_classes, not both.")
    if num_classes is not None:
        if isinstance(num_classes, bool) or int(num_classes) != num_classes:
            raise ValueError("num_classes must be a positive integer.")
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be a positive integer.")
        result = np.arange(int(num_classes), dtype=np.int64)
    elif labels is not None:
        result = _class_indices(labels, name="labels", scores_allowed=False)
        if result.size == 0:
            raise ValueError("labels must not be empty.")
        if np.unique(result).size != result.size:
            raise ValueError("labels must not contain duplicates.")
    else:
        result = np.unique(np.concatenate((targets, predicted))).astype(np.int64)

    allowed = set(result.tolist())
    observed = set(targets.tolist()) | set(predicted.tolist())
    unknown = sorted(observed - allowed)
    if unknown:
        raise ValueError(f"Observed class indices are absent from labels: {unknown}.")
    return result


def accuracy(predictions: Any, targets: Any) -> float:
    """Return the fraction of correctly classified samples.

    ``predictions`` may be class indices with shape ``(N,)`` or logits/
    probabilities with shape ``(N, C)``. ``targets`` may be class indices or
    one-hot rows. Empty inputs raise ``ValueError`` because accuracy would be
    undefined.
    """

    predicted, expected = _prediction_target_indices(predictions, targets)
    return float(np.mean(predicted == expected))


def confusion_matrix(
    predictions: Any,
    targets: Any,
    *,
    labels: Sequence[int] | None = None,
    num_classes: int | None = None,
) -> np.ndarray:
    """Return an integer confusion matrix with target rows and prediction columns.

    By default the matrix contains the sorted union of observed classes. Pass
    ``labels`` to control row/column order or ``num_classes`` to include every
    index in ``range(num_classes)``.
    """

    predicted, expected = _prediction_target_indices(predictions, targets)
    ordered_labels = _label_indices(
        predicted,
        expected,
        labels=labels,
        num_classes=num_classes,
    )
    positions = {int(label): index for index, label in enumerate(ordered_labels)}
    matrix = np.zeros((ordered_labels.size, ordered_labels.size), dtype=np.int64)
    for target, prediction in zip(expected, predicted):
        matrix[positions[int(target)], positions[int(prediction)]] += 1
    return matrix


def macro_f1(
    predictions: Any,
    targets: Any,
    *,
    labels: Sequence[int] | None = None,
    num_classes: int | None = None,
    zero_division: float = 0.0,
) -> float:
    """Return the unweighted mean of per-class F1 scores.

    Classes selected through ``labels`` or ``num_classes`` but absent from both
    predictions and targets receive ``zero_division`` (default ``0.0``).
    """

    try:
        zero_value = float(zero_division)
    except (TypeError, ValueError) as exc:
        raise ValueError("zero_division must be finite and between 0 and 1.") from exc
    if not np.isfinite(zero_value) or not 0.0 <= zero_value <= 1.0:
        raise ValueError("zero_division must be finite and between 0 and 1.")
    matrix = confusion_matrix(
        predictions,
        targets,
        labels=labels,
        num_classes=num_classes,
    )
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.full(denominator.shape, zero_value, dtype=np.float64)
    np.divide(
        2.0 * true_positive,
        denominator,
        out=scores,
        where=denominator != 0,
    )
    return float(scores.mean())
