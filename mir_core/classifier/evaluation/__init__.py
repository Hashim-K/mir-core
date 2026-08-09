"""Metrics and confidence calibration for genre-classifier routing."""

from .calibration import (
    RoutingThresholdSelection,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    select_fallback_threshold,
    select_routing_threshold,
    softmax_probabilities,
)
from .metrics import (
    accuracy,
    binary_ranking_diagnostics,
    classification_diagnostics,
    confidence_rejection_diagnostics,
    confusion_matrix,
    macro_f1,
)

__all__ = [
    "RoutingThresholdSelection",
    "accuracy",
    "apply_temperature",
    "binary_ranking_diagnostics",
    "classification_diagnostics",
    "confidence_rejection_diagnostics",
    "confusion_matrix",
    "expected_calibration_error",
    "fit_temperature",
    "macro_f1",
    "negative_log_likelihood",
    "select_fallback_threshold",
    "select_routing_threshold",
    "softmax_probabilities",
]
