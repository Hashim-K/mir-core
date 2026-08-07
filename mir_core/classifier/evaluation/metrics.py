"""Dependency-light classification metrics with strict input validation.

The functions accept class-index predictions or a two-dimensional score matrix.
Targets may be class indices or a two-dimensional one-hot matrix. Class indices
are intentionally required to be non-negative integers so label ordering stays
unambiguous across training, evaluation, and deployment.
"""

from __future__ import annotations

import math
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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f_beta(
    precision: float,
    recall: float,
    *,
    beta: float,
) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    return (
        float((1.0 + beta_squared) * precision * recall / denominator)
        if denominator
        else 0.0
    )


def _binary_ranking_summary(
    scores: np.ndarray,
    positives: np.ndarray,
) -> dict[str, float | None]:
    """Return deterministic threshold-grouped one-vs-rest ranking metrics."""

    positive_count = int(positives.sum())
    negative_count = int(len(positives) - positive_count)
    if positive_count == 0:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "average_precision": None,
        }

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_positives = positives[order].astype(np.int64)
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    cumulative_true_positive = np.cumsum(ordered_positives)[group_ends]
    cumulative_count = group_ends + 1
    cumulative_false_positive = cumulative_count - cumulative_true_positive
    recall = cumulative_true_positive.astype(np.float64) / positive_count
    precision = cumulative_true_positive.astype(np.float64) / cumulative_count

    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))
    pr_recall = np.r_[0.0, recall]
    pr_precision = np.r_[1.0, precision]
    pr_auc = float(
        np.sum(np.diff(pr_recall) * (pr_precision[:-1] + pr_precision[1:]) / 2.0)
    )

    roc_auc: float | None = None
    if negative_count:
        false_positive_rate = (
            cumulative_false_positive.astype(np.float64) / negative_count
        )
        true_positive_rate = recall
        roc_fpr = np.r_[0.0, false_positive_rate]
        roc_tpr = np.r_[0.0, true_positive_rate]
        roc_auc = float(np.sum(np.diff(roc_fpr) * (roc_tpr[:-1] + roc_tpr[1:]) / 2.0))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }


def _mean_defined(
    values: Sequence[float | None],
    *,
    weights: Sequence[float] | None = None,
) -> float | None:
    defined = np.asarray([value is not None for value in values], dtype=bool)
    if not defined.any():
        return None
    numeric = np.asarray(
        [0.0 if value is None else float(value) for value in values],
        dtype=np.float64,
    )
    if weights is None:
        return float(numeric[defined].mean())
    numeric_weights = np.asarray(weights, dtype=np.float64)
    selected_weights = numeric_weights[defined]
    if float(selected_weights.sum()) == 0.0:
        return float(numeric[defined].mean())
    return float(np.average(numeric[defined], weights=selected_weights))


def classification_diagnostics(
    probabilities: Any,
    targets: Any,
    *,
    class_names: Sequence[str] | None = None,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Return a JSON-safe, detailed single-label multiclass report.

    The report contains count and row-normalized confusion matrices,
    per-class one-vs-rest metrics, macro/micro/weighted aggregates, ranking
    metrics, multiclass Brier score, log loss, and top-label calibration bins.
    Undefined ROC/PR metrics are represented by ``None`` when a class has no
    positive or negative examples.
    """

    matrix = _as_numpy(probabilities, name="probabilities")
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a numeric matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, C) with N > 0 and C >= 2.")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("probabilities must be finite and non-negative.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("probability rows must sum to 1.")
    matrix = matrix / row_sums[:, np.newaxis]

    num_samples, num_classes = matrix.shape
    expected = _class_indices(targets, name="targets", scores_allowed=False)
    if len(expected) != num_samples:
        raise ValueError(
            "probabilities and targets must contain the same number of samples "
            f"({num_samples} != {len(expected)})."
        )
    if expected.size and int(expected.max()) >= num_classes:
        raise ValueError(f"targets class indices must be in [0, {num_classes - 1}].")
    if isinstance(calibration_bins, bool) or int(calibration_bins) != calibration_bins:
        raise ValueError("calibration_bins must be a positive integer.")
    calibration_bins = int(calibration_bins)
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be a positive integer.")

    if class_names is None:
        names = tuple(str(index) for index in range(num_classes))
    else:
        names = tuple(str(name) for name in class_names)
        if len(names) != num_classes:
            raise ValueError(
                "class_names must contain one name for each probability column."
            )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("class_names must be non-empty and unique.")

    predicted = np.argmax(matrix, axis=1).astype(np.int64)
    counts = confusion_matrix(
        predicted,
        expected,
        num_classes=num_classes,
    )
    support = counts.sum(axis=1).astype(np.int64)
    predicted_support = counts.sum(axis=0).astype(np.int64)
    true_positive = np.diag(counts).astype(np.int64)
    false_positive = predicted_support - true_positive
    false_negative = support - true_positive
    true_negative = num_samples - true_positive - false_positive - false_negative

    per_class: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(names):
        precision = _safe_ratio(
            float(true_positive[class_index]),
            float(true_positive[class_index] + false_positive[class_index]),
        )
        recall = _safe_ratio(
            float(true_positive[class_index]),
            float(true_positive[class_index] + false_negative[class_index]),
        )
        specificity_denominator = (
            true_negative[class_index] + false_positive[class_index]
        )
        ranking = _binary_ranking_summary(
            matrix[:, class_index],
            expected == class_index,
        )
        per_class.append(
            {
                "class_index": class_index,
                "label": class_name,
                "support": int(support[class_index]),
                "predicted_support": int(predicted_support[class_index]),
                "true_positive": int(true_positive[class_index]),
                "false_positive": int(false_positive[class_index]),
                "false_negative": int(false_negative[class_index]),
                "true_negative": int(true_negative[class_index]),
                "precision": precision,
                "recall": recall,
                "specificity": (
                    _safe_ratio(
                        float(true_negative[class_index]),
                        float(specificity_denominator),
                    )
                    if specificity_denominator
                    else None
                ),
                "f1": _f_beta(precision, recall, beta=1.0),
                "f2": _f_beta(precision, recall, beta=2.0),
                **ranking,
            }
        )

    aggregate_metric_names = ("precision", "recall", "f1", "f2")
    macro = {
        metric: float(np.mean([row[metric] for row in per_class]))
        for metric in aggregate_metric_names
    }
    weighted = {
        metric: float(
            np.average(
                [row[metric] for row in per_class],
                weights=support,
            )
        )
        for metric in aggregate_metric_names
    }
    total_true_positive = int(true_positive.sum())
    total_false_positive = int(false_positive.sum())
    total_false_negative = int(false_negative.sum())
    micro_precision = _safe_ratio(
        total_true_positive,
        total_true_positive + total_false_positive,
    )
    micro_recall = _safe_ratio(
        total_true_positive,
        total_true_positive + total_false_negative,
    )
    micro = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": _f_beta(micro_precision, micro_recall, beta=1.0),
        "f2": _f_beta(micro_precision, micro_recall, beta=2.0),
    }

    ranking_names = ("roc_auc", "pr_auc", "average_precision")
    for metric in ranking_names:
        values = [row[metric] for row in per_class]
        macro[metric] = _mean_defined(values)
        weighted[metric] = _mean_defined(values, weights=support)
    one_hot = np.eye(num_classes, dtype=np.float64)[expected]
    micro.update(
        _binary_ranking_summary(
            matrix.reshape(-1),
            one_hot.reshape(-1).astype(bool),
        )
    )

    normalized = np.zeros_like(counts, dtype=np.float64)
    np.divide(
        counts,
        support[:, np.newaxis],
        out=normalized,
        where=support[:, np.newaxis] != 0,
    )
    accuracy_value = float((predicted == expected).mean())
    expected_agreement = float(
        np.dot(support.astype(np.float64), predicted_support.astype(np.float64))
        / (num_samples * num_samples)
    )
    kappa_denominator = 1.0 - expected_agreement
    cohen_kappa = (
        float((accuracy_value - expected_agreement) / kappa_denominator)
        if kappa_denominator
        else None
    )
    mcc_numerator = total_true_positive * num_samples - float(
        np.dot(support, predicted_support)
    )
    mcc_denominator = math.sqrt(
        float(num_samples * num_samples - np.dot(predicted_support, predicted_support))
        * float(num_samples * num_samples - np.dot(support, support))
    )
    matthews_correlation = (
        float(mcc_numerator / mcc_denominator) if mcc_denominator else None
    )

    confidence = matrix.max(axis=1)
    correct = predicted == expected
    bin_indices = np.minimum(
        (confidence * calibration_bins).astype(np.int64),
        calibration_bins - 1,
    )
    reliability: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    for bin_index in range(calibration_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        bin_accuracy = float(correct[mask].mean()) if count else None
        gap = abs(float(bin_accuracy) - float(mean_confidence)) if count else None
        if gap is not None:
            expected_calibration_error += (count / num_samples) * gap
            maximum_calibration_error = max(maximum_calibration_error, gap)
        reliability.append(
            {
                "bin_index": bin_index,
                "lower": bin_index / calibration_bins,
                "upper": (bin_index + 1) / calibration_bins,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
                "gap": gap,
            }
        )

    selected_probability = matrix[np.arange(num_samples), expected]
    return {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "labels": list(names),
        "accuracy": accuracy_value,
        "balanced_accuracy": macro["recall"],
        "cohen_kappa": cohen_kappa,
        "matthews_correlation_coefficient": matthews_correlation,
        "log_loss": float(
            -np.log(
                np.clip(
                    selected_probability,
                    np.finfo(np.float64).tiny,
                    1.0,
                )
            ).mean()
        ),
        "brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
        "mean_confidence": float(confidence.mean()),
        "averages": {
            "macro": macro,
            "micro": micro,
            "weighted": weighted,
        },
        "per_class": per_class,
        "confusion": counts.tolist(),
        "confusion_normalized": normalized.tolist(),
        "calibration_bins": reliability,
    }
