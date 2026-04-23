"""
Core evaluation metrics for beat and downbeat tracking.

Wraps mir_eval library and adds additional metrics following:
- Rapini & Jordanous (2024): CMLt, AMLt metrics
- Maia et al. (2023): F-measure at 70ms tolerance
- Davies et al. (2009): Continuity-based metrics

Functions:
    compute_beat_metrics     — all mir_eval beat metrics for a single track (70ms default).
    compute_downbeat_metrics — downbeat metrics for a single track.
    evaluate_beats           — aggregate metrics across multiple tracks.
    evaluate_downbeats       — aggregate downbeat metrics.
    evaluate_tempo           — tempo evaluation with Acc1/Acc2.
    compute_per_track_metrics — per-track metrics from test results.
    compute_ibi_stats        — inter-beat interval statistics.
    compute_interdownbeat_stats — inter-downbeat interval statistics.
    ibi_distribution_text    — human-readable IBI summary.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import mir_eval


# =============================================================================
# Inter-Beat Interval (IBI) Statistics
# =============================================================================

def compute_ibi_stats(
    beats: np.ndarray,
    label: str = "ibi",
) -> Dict[str, float]:
    """
    Compute inter-beat interval (IBI) statistics.

    Measures the distribution of time between consecutive beats:
    mean, median, std, coefficient of variation, 95th and 99th percentiles,
    min/max, range, and tempo implied by the mean interval.

    Args:
        beats: Beat times in seconds (will be sorted internally)
        label: Prefix for returned keys (e.g. "ibi" -> "ibi_mean", ...)

    Returns:
        Dict with keys ``{label}_mean``, ``{label}_median``,
        ``{label}_std``, ``{label}_cv``, ``{label}_p95``, ``{label}_p99``,
        ``{label}_min``, ``{label}_max``, ``{label}_range``,
        ``{label}_bpm_mean``, and ``{label}_n``. Interval values are in
        seconds; BPM is beats per minute inferred from the mean interval.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    empty = {
        f"{label}_mean": 0.0,
        f"{label}_median": 0.0,
        f"{label}_std": 0.0,
        f"{label}_cv": 0.0,
        f"{label}_p95": 0.0,
        f"{label}_p99": 0.0,
        f"{label}_min": 0.0,
        f"{label}_max": 0.0,
        f"{label}_range": 0.0,
        f"{label}_bpm_mean": 0.0,
        f"{label}_n": 0,
    }
    if len(beats) < 2:
        return empty
    intervals = np.diff(beats)
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    return {
        f"{label}_mean":  mean,
        f"{label}_median": float(np.median(intervals)),
        f"{label}_std":   std,
        f"{label}_cv":    float(std / mean) if mean > 0 else 0.0,
        f"{label}_p95":   float(np.percentile(intervals, 95)),
        f"{label}_p99":   float(np.percentile(intervals, 99)),
        f"{label}_min":   float(np.min(intervals)),
        f"{label}_max":   float(np.max(intervals)),
        f"{label}_range": float(np.max(intervals) - np.min(intervals)),
        f"{label}_bpm_mean": float(60.0 / mean) if mean > 0 else 0.0,
        f"{label}_n":     int(len(intervals)),
    }


def compute_count_tempo_diagnostics(
    beats_pred: np.ndarray,
    beats_ann: np.ndarray,
) -> Dict[str, float]:
    """Compute count, IBI-difference, and tempo-ratio diagnostics."""
    pred_stats = compute_ibi_stats(beats_pred, "ibi")
    ann_stats = compute_ibi_stats(beats_ann, "ibi_ann")
    num_pred = int(len(beats_pred))
    num_ann = int(len(beats_ann))

    pred_tempo = pred_stats["ibi_bpm_mean"]
    ann_tempo = ann_stats["ibi_ann_bpm_mean"]
    count_error = num_pred - num_ann
    ibi_mean_error = pred_stats["ibi_mean"] - ann_stats["ibi_ann_mean"]
    ibi_std_error = pred_stats["ibi_std"] - ann_stats["ibi_ann_std"]
    tempo_error = pred_tempo - ann_tempo
    tempo_ratio = pred_tempo / ann_tempo if ann_tempo > 0 else 0.0

    return {
        "beat_count_ratio": float(num_pred / num_ann) if num_ann > 0 else 0.0,
        "beat_count_error": float(count_error),
        "beat_count_abs_error": float(abs(count_error)),
        "beat_count_abs_error_pct": float(abs(count_error) / num_ann) if num_ann > 0 else 0.0,
        "ibi_mean_error": float(ibi_mean_error),
        "ibi_mean_abs_error": float(abs(ibi_mean_error)),
        "ibi_mean_abs_error_pct": (
            float(abs(ibi_mean_error) / ann_stats["ibi_ann_mean"])
            if ann_stats["ibi_ann_mean"] > 0
            else 0.0
        ),
        "ibi_std_error": float(ibi_std_error),
        "ibi_std_abs_error": float(abs(ibi_std_error)),
        "tempo_pred_bpm": float(pred_tempo),
        "tempo_ann_bpm": float(ann_tempo),
        "tempo_error_bpm": float(tempo_error),
        "tempo_abs_error_bpm": float(abs(tempo_error)),
        "tempo_abs_error_pct": float(abs(tempo_error) / ann_tempo) if ann_tempo > 0 else 0.0,
        "tempo_ratio": float(tempo_ratio),
        "tempo_doubling_suspected": float(abs(tempo_ratio - 2.0) <= 0.15),
        "tempo_halving_suspected": float(abs(tempo_ratio - 0.5) <= 0.075),
    }


def compute_interdownbeat_stats(
    downbeats: np.ndarray,
) -> Dict[str, float]:
    """
    Convenience wrapper: IBI stats for downbeats (bar durations).

    Keys are prefixed with ``idbi_`` (inter-downbeat interval).
    """
    return compute_ibi_stats(downbeats, label="idbi")


def ibi_distribution_text(
    beats: np.ndarray,
    label: str = "IBI",
    unit_ms: bool = True,
) -> str:
    """
    Return a one-line human-readable summary of inter-beat interval stats.

    Args:
        beats: Beat times in seconds
        label: Display name
        unit_ms: If True, show values in milliseconds; else seconds

    Returns:
        Formatted string, e.g.
        "IBI  mean=480ms  std=12ms  p95=501ms  p99=523ms  range=61ms  n=128"
    """
    stats = compute_ibi_stats(beats)
    scale = 1000.0 if unit_ms else 1.0
    u = "ms" if unit_ms else "s"
    return (
        f"{label}  "
        f"mean={stats['ibi_mean']*scale:.1f}{u}  "
        f"std={stats['ibi_std']*scale:.1f}{u}  "
        f"p95={stats['ibi_p95']*scale:.1f}{u}  "
        f"p99={stats['ibi_p99']*scale:.1f}{u}  "
        f"range={stats['ibi_range']*scale:.1f}{u}  "
        f"n={stats['ibi_n']}"
    )


# =============================================================================
# Core Evaluation Functions (wrapping mir_eval)
# =============================================================================

def compute_beat_metrics(
    beats_pred: np.ndarray,
    beats_ann: np.ndarray,
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """
    Compute all beat tracking metrics for a single track.

    Metrics included (from mir_eval):
    - F-measure: Precision/recall harmonic mean at tolerance threshold
    - Cemgil: Gaussian-weighted accuracy
    - P-score: Phase/tempo accuracy
    - CMLc, CMLt, AMLc, AMLt: Continuity-based metrics
    - Goto: Goto accuracy
    - Information gain: Mutual information based metric

    Args:
        beats_pred: Predicted beat times in seconds
        beats_ann: Annotated beat times in seconds
        tolerance: Tolerance window in seconds (default 70ms per literature)

    Returns:
        Dictionary with all computed metrics
    """
    beats_pred = np.sort(np.asarray(beats_pred, dtype=float))
    beats_ann = np.sort(np.asarray(beats_ann, dtype=float))

    metrics: Dict[str, float] = {}

    # Handle empty predictions/annotations while keeping a uniform schema.
    if len(beats_pred) == 0 or len(beats_ann) == 0:
        metrics.update(
            {
                "fmeasure": 0.0,
                "cemgil": 0.0,
                "pscore": 0.0,
                "goto": 0.0,
                "cmlc": 0.0,
                "cmlt": 0.0,
                "amlc": 0.0,
                "amlt": 0.0,
                "information_gain": 0.0,
                "num_pred": len(beats_pred),
                "num_ann": len(beats_ann),
            }
        )
        metrics.update(compute_ibi_stats(beats_pred, "ibi"))
        metrics.update(compute_ibi_stats(beats_ann,  "ibi_ann"))
        metrics.update(compute_count_tempo_diagnostics(beats_pred, beats_ann))
        return metrics

    # F-measure
    metrics["fmeasure"] = mir_eval.beat.f_measure(
        beats_ann, beats_pred, f_measure_threshold=tolerance
    )

    # Cemgil score (returns tuple of (score, score_with_max_time), we want first)
    cemgil_result = mir_eval.beat.cemgil(
        beats_ann, beats_pred, cemgil_sigma=tolerance
    )
    metrics["cemgil"] = cemgil_result[0] if isinstance(cemgil_result, tuple) else cemgil_result

    # P-score
    metrics["pscore"] = mir_eval.beat.p_score(
        beats_ann, beats_pred, p_score_threshold=tolerance
    )

    # Goto accuracy
    metrics["goto"] = mir_eval.beat.goto(
        beats_ann, beats_pred, goto_threshold=tolerance, goto_mu=0.2, goto_sigma=0.2
    )

    # Continuity-based metrics (CMLc, CMLt, AMLc, AMLt)
    # These are key metrics from Rapini & Jordanous (2024)
    cmlc, cmlt, amlc, amlt = mir_eval.beat.continuity(
        beats_ann, beats_pred, continuity_phase_threshold=0.175
    )
    metrics["cmlc"] = cmlc
    metrics["cmlt"] = cmlt
    metrics["amlc"] = amlc
    metrics["amlt"] = amlt

    # Information gain
    metrics["information_gain"] = mir_eval.beat.information_gain(
        beats_ann, beats_pred
    )

    # Count statistics
    metrics["num_pred"] = len(beats_pred)
    metrics["num_ann"] = len(beats_ann)

    # Inter-beat interval stats for predictions and annotations
    metrics.update(compute_ibi_stats(beats_pred, "ibi"))
    metrics.update(compute_ibi_stats(beats_ann,  "ibi_ann"))
    metrics.update(compute_count_tempo_diagnostics(beats_pred, beats_ann))

    return metrics


def compute_downbeat_metrics(
    downbeats_pred: np.ndarray,
    downbeats_ann: np.ndarray,
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """
    Compute downbeat tracking metrics for a single track.

    Uses the same metrics as beat tracking but applied to downbeats.

    Args:
        downbeats_pred: Predicted downbeat times in seconds
        downbeats_ann: Annotated downbeat times in seconds
        tolerance: Tolerance window in seconds

    Returns:
        Dictionary with metrics (prefixed with 'db_')
    """
    base_metrics = compute_beat_metrics(downbeats_pred, downbeats_ann, tolerance)
    # Add interdownbeat interval stats (idbi prefix, then db_ outer prefix)
    base_metrics.update(compute_ibi_stats(downbeats_pred, "idbi"))
    base_metrics.update(compute_ibi_stats(downbeats_ann,  "idbi_ann"))
    return {f"db_{k}": v for k, v in base_metrics.items()}


def evaluate_beats(
    predictions: Dict[str, np.ndarray],
    annotations: Dict[str, np.ndarray],
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """
    Evaluate beat predictions against annotations for multiple tracks.

    Args:
        predictions: Dict mapping track_id to predicted beat times
        annotations: Dict mapping track_id to annotated beat times
        tolerance: Tolerance window in seconds

    Returns:
        Dictionary with mean/std of evaluation metrics
    """
    all_metrics = defaultdict(list)

    for track_id in predictions:
        if track_id not in annotations:
            continue

        pred = predictions[track_id]
        ann = annotations[track_id]

        track_metrics = compute_beat_metrics(pred, ann, tolerance)
        for key, value in track_metrics.items():
            if key not in ["num_pred", "num_ann"]:
                all_metrics[key].append(value)

    # Compute mean and std for each metric
    result = {"num_tracks": len(all_metrics.get("fmeasure", []))}
    for key, values in all_metrics.items():
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))

    return result


def evaluate_downbeats(
    predictions: Dict[str, np.ndarray],
    annotations: Dict[str, np.ndarray],
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """
    Evaluate downbeat predictions against annotations.

    Args:
        predictions: Dict mapping track_id to predicted downbeat times
        annotations: Dict mapping track_id to annotated downbeat times
        tolerance: Tolerance window in seconds

    Returns:
        Dictionary with evaluation metrics
    """
    # Same metrics as beats
    return evaluate_beats(predictions, annotations, tolerance)


def evaluate_tempo(
    predictions: Dict[str, float],
    annotations: Dict[str, float],
    tolerance: float = 0.04,  # 4% tolerance
) -> Dict[str, float]:
    """
    Evaluate tempo predictions.

    Args:
        predictions: Dict mapping track_id to predicted tempo (BPM)
        annotations: Dict mapping track_id to annotated tempo
        tolerance: Relative tolerance (0.04 = 4%)

    Returns:
        Dictionary with Acc1, Acc2 metrics
    """
    acc1_scores = []
    acc2_scores = []

    for track_id in predictions:
        if track_id not in annotations:
            continue

        pred = predictions[track_id]
        ann = annotations[track_id]

        # Acc1: correct if within tolerance
        if abs(pred - ann) / ann < tolerance:
            acc1_scores.append(1.0)
        else:
            acc1_scores.append(0.0)

        # Acc2: also accept double/half tempo
        if (abs(pred - ann) / ann < tolerance or
            abs(pred - 2*ann) / (2*ann) < tolerance or
            abs(pred - ann/2) / (ann/2) < tolerance):
            acc2_scores.append(1.0)
        else:
            acc2_scores.append(0.0)

    return {
        "acc1": np.mean(acc1_scores),
        "acc2": np.mean(acc2_scores),
        "num_tracks": len(acc1_scores),
    }


def compute_per_track_metrics(
    test_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each track from test results.

    Args:
        test_results: List of dicts with track_id, beats_target, beats_pred

    Returns:
        Dict mapping track_id to metrics dict
    """
    metrics = {}

    for result in test_results:
        track_id = result["track_id"]
        ann = result["beats_target"]
        pred = result["beats_pred"]

        if len(pred) == 0 or len(ann) == 0:
            metrics[track_id] = {
                "fmeasure": 0.0,
                "cemgil": 0.0,
                "num_beats_ann": len(ann),
                "num_beats_pred": len(pred),
            }
            continue

        cemgil_result = mir_eval.beat.cemgil(ann, pred)
        cemgil_score = cemgil_result[0] if isinstance(cemgil_result, tuple) else cemgil_result
        metrics[track_id] = {
            "fmeasure": mir_eval.beat.f_measure(ann, pred),
            "cemgil": cemgil_score,
            "num_beats_ann": len(ann),
            "num_beats_pred": len(pred),
        }

    return metrics
