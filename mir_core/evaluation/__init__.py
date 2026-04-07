"""
Evaluation utilities for beat tracking.

Provides metrics computation, visualization, and comprehensive evaluation.

Key components:
    compute_beat_metrics     — all mir_eval beat metrics for a single track (70ms default).
    compute_downbeat_metrics — downbeat metrics for a single track.
    evaluate_beats           — aggregate metrics across multiple tracks.
    BeatEvaluator            — stateful evaluator with per-fold CV, per-track tables,
                               worst/best track analysis, and text report generation.
    plot_predictions         — spectrogram overlay with annotation + prediction vlines.
    plot_activations         — beat activation curve with annotation markers.
"""

from .metrics import (
    compute_beat_metrics,
    compute_downbeat_metrics,
    compute_ibi_stats,
    compute_interdownbeat_stats,
    ibi_distribution_text,
    evaluate_beats,
    evaluate_downbeats,
    evaluate_tempo,
    compute_per_track_metrics,
)

from .evaluator import BeatEvaluator

from .visualization import (
    plot_predictions,
    plot_activations,
    print_evaluation_summary,
)

__all__ = [
    # Metrics
    "compute_beat_metrics",
    "compute_downbeat_metrics",
    "compute_ibi_stats",
    "compute_interdownbeat_stats",
    "ibi_distribution_text",
    "evaluate_beats",
    "evaluate_downbeats",
    "evaluate_tempo",
    "compute_per_track_metrics",
    # Evaluator
    "BeatEvaluator",
    # Visualization
    "plot_predictions",
    "plot_activations",
    "print_evaluation_summary",
]
