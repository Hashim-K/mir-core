"""
Core evaluation metrics for beat and downbeat tracking.

Wraps mir_eval library and adds additional metrics following:
- Rapini & Jordanous (2024): CMLt, AMLt metrics
- Maia et al. (2023): F-measure at 70ms tolerance
- Davies et al. (2009): Continuity-based metrics

Functions:
    compute_beat_metrics     — all mir_eval beat metrics for a single track (70ms default).
    compute_downbeat_metrics — downbeat metrics for a single track.
    compute_event_timing_errors — signed errors for one-to-one matched events.
    compute_event_timing_error_stats — millisecond timing-error diagnostics.
    compute_realtime_event_times — event times after software availability delay.
    compute_realtime_event_metrics — precision/recall/RT-F1 at one tolerance.
    compute_realtime_f1_curve — RT-F1 across the thesis tolerance curve.
    compute_joint_realtime_f1_curve — pointwise joint RT-F1 and its nAUC.
    evaluate_beats           — aggregate metrics across multiple tracks.
    evaluate_downbeats       — aggregate downbeat metrics.
    evaluate_tempo           — tempo evaluation with Acc1/Acc2.
    compute_per_track_metrics — per-track metrics from test results.
    compute_ibi_stats        — inter-beat interval statistics.
    compute_interdownbeat_stats — inter-downbeat interval statistics.
    ibi_distribution_text    — human-readable IBI summary.
"""

from typing import Any, Dict, List, Sequence, Tuple
from collections import defaultdict

import numpy as np
import mir_eval


DEFAULT_REALTIME_TOLERANCES_SECONDS = (0.03, 0.05, 0.07, 0.10, 0.15)


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
# Matched-Event Timing Error
# =============================================================================


def compute_realtime_event_times(
    events_pred: np.ndarray,
    output_ready_times: np.ndarray,
) -> np.ndarray:
    """Return the physical event times available before actuation.

    A causal system can schedule an event that it predicts before the intended
    musical timestamp, but it cannot emit an event before the software has made
    the decision available.  The effective pre-actuator event time is therefore
    ``max(predicted musical time, software output-ready time)`` for each event.

    This deliberately excludes actuator and transport delay.  Those are
    measured separately by the end-to-end experiment.
    """

    predicted = np.asarray(events_pred, dtype=float).reshape(-1)
    ready = np.asarray(output_ready_times, dtype=float).reshape(-1)
    if predicted.shape != ready.shape:
        raise ValueError(
            "events_pred and output_ready_times must contain one aligned value "
            "per emitted event"
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("events_pred must contain only finite timestamps")
    if not np.all(np.isfinite(ready)):
        raise ValueError("output_ready_times must contain only finite timestamps")
    return np.maximum(predicted, ready)


def compute_realtime_event_metrics(
    events_pred: np.ndarray,
    output_ready_times: np.ndarray,
    events_ann: np.ndarray,
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """Compute causal precision, recall, and RT-F1 before actuation.

    First, conventional maximum-cardinality one-to-one matching establishes
    which annotated musical event each predicted timestamp represents.  A
    matched prediction is an RT true positive only when its effective output
    time also falls within ``tolerance`` of that *same* annotation.  Preserving
    event identity prevents a prediction delayed by one or more beat periods
    from being incorrectly reassigned to a later beat.
    """

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number of seconds")
    predicted = np.asarray(events_pred, dtype=float).reshape(-1)
    effective = compute_realtime_event_times(events_pred, output_ready_times)
    prediction_order = np.argsort(predicted, kind="stable")
    predicted = predicted[prediction_order]
    effective = effective[prediction_order]
    annotated = np.sort(np.asarray(events_ann, dtype=float).reshape(-1))
    if not np.all(np.isfinite(annotated)):
        raise ValueError("events_ann must contain only finite timestamps")

    conventional_matches = (
        mir_eval.util.match_events(annotated, predicted, tolerance)
        if len(annotated) and len(effective)
        else []
    )
    matched = int(
        sum(
            abs(effective[pred_index] - annotated[ann_index]) <= tolerance
            for ann_index, pred_index in conventional_matches
        )
    )
    precision = float(matched / len(effective)) if len(effective) else 0.0
    recall = float(matched / len(annotated)) if len(annotated) else 0.0
    denominator = precision + recall
    f1 = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    return {
        "rt_precision": precision,
        "rt_recall": recall,
        "rt_f1": float(f1),
        "rt_matched": float(matched),
        "conventional_matched": float(len(conventional_matches)),
        "rt_num_pred": float(len(effective)),
        "rt_num_ann": float(len(annotated)),
        "rt_tolerance_seconds": tolerance,
    }


def compute_realtime_f1_curve(
    events_pred: np.ndarray,
    output_ready_times: np.ndarray,
    events_ann: np.ndarray,
    tolerances: Sequence[float] = DEFAULT_REALTIME_TOLERANCES_SECONDS,
) -> Dict[str, Any]:
    """Compute the thesis RT-F1 tolerance curve and normalized AUC (nAUC).

    The returned nAUC is normalized by the tolerance span, so it remains on the
    same 0–1 scale as F1.  The default curve evaluates 30, 50, 70, 100, and
    150 ms.  A one-point curve has that point's F1 as its nAUC.
    """

    tolerance_values = np.asarray(tuple(tolerances), dtype=float).reshape(-1)
    if not len(tolerance_values):
        raise ValueError("tolerances must contain at least one value")
    if not np.all(np.isfinite(tolerance_values)) or np.any(
        tolerance_values <= 0.0
    ):
        raise ValueError("tolerances must contain positive finite seconds")
    if len(tolerance_values) > 1 and np.any(np.diff(tolerance_values) <= 0.0):
        raise ValueError("tolerances must be strictly increasing")

    rows = [
        compute_realtime_event_metrics(
            events_pred,
            output_ready_times,
            events_ann,
            tolerance=float(tolerance),
        )
        for tolerance in tolerance_values
    ]
    precision = np.asarray([row["rt_precision"] for row in rows], dtype=float)
    recall = np.asarray([row["rt_recall"] for row in rows], dtype=float)
    f1 = np.asarray([row["rt_f1"] for row in rows], dtype=float)
    matched = np.asarray([row["rt_matched"] for row in rows], dtype=float)
    nauc = compute_normalized_curve_area(f1, tolerance_values)
    return {
        "tolerances_seconds": tolerance_values,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "nauc": nauc,
        # Temporary compatibility for callers that predate the explicit nAUC
        # naming. New experiment outputs use ``nauc`` exclusively.
        "auc": nauc,
        "effective_event_times_seconds": compute_realtime_event_times(
            events_pred,
            output_ready_times,
        ),
    }


def compute_normalized_curve_area(
    values: Sequence[float],
    coordinates: Sequence[float],
) -> float:
    """Return trapezoidal curve area normalized by the coordinate span.

    Normalizing by the span keeps a curve whose values lie in ``[0, 1]`` on
    that same scale. A one-point curve is defined to have the value of its
    only point.
    """

    value_array = np.asarray(tuple(values), dtype=float).reshape(-1)
    coordinate_array = np.asarray(tuple(coordinates), dtype=float).reshape(-1)
    if not len(value_array):
        raise ValueError("values and coordinates must contain at least one point")
    if len(value_array) != len(coordinate_array):
        raise ValueError("values and coordinates must have the same length")
    if not np.all(np.isfinite(value_array)):
        raise ValueError("values must contain only finite numbers")
    if not np.all(np.isfinite(coordinate_array)):
        raise ValueError("coordinates must contain only finite numbers")
    if len(coordinate_array) == 1:
        return float(value_array[0])
    if np.any(np.diff(coordinate_array) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    return float(
        np.trapz(value_array, coordinate_array)
        / (coordinate_array[-1] - coordinate_array[0])
    )


def compute_joint_realtime_f1_curve(
    beat_f1: Sequence[float],
    downbeat_f1: Sequence[float],
    tolerances: Sequence[float] = DEFAULT_REALTIME_TOLERANCES_SECONDS,
) -> Dict[str, Any]:
    """Join beat/downbeat RT-F1 pointwise, then integrate the joint curve.

    The harmonic mean is applied independently at every tolerance. The joint
    nAUC is then the normalized area under that joint curve; it is deliberately
    not the harmonic mean of the separately integrated beat and downbeat
    nAUCs.
    """

    beat = np.asarray(tuple(beat_f1), dtype=float).reshape(-1)
    downbeat = np.asarray(tuple(downbeat_f1), dtype=float).reshape(-1)
    tolerance_values = np.asarray(tuple(tolerances), dtype=float).reshape(-1)
    if len(beat) != len(downbeat) or len(beat) != len(tolerance_values):
        raise ValueError(
            "beat_f1, downbeat_f1, and tolerances must have the same length"
        )
    if not np.all(np.isfinite(beat)) or not np.all(np.isfinite(downbeat)):
        raise ValueError("RT-F1 curves must contain only finite numbers")
    if np.any(beat < 0.0) or np.any(downbeat < 0.0):
        raise ValueError("RT-F1 curves must be non-negative")
    denominator = beat + downbeat
    joint = np.divide(
        2.0 * beat * downbeat,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    return {
        "tolerances_seconds": tolerance_values,
        "f1": joint,
        "nauc": compute_normalized_curve_area(joint, tolerance_values),
    }


def compute_event_timing_errors(
    events_pred: np.ndarray,
    events_ann: np.ndarray,
    tolerance: float = 0.07,
) -> np.ndarray:
    """Return signed errors for the one-to-one event matches used by F-measure.

    The matching is maximum-cardinality and one-to-one within ``tolerance``,
    using :func:`mir_eval.util.match_events`.  Returned values are in seconds
    and follow ``prediction - annotation``: negative values are early and
    positive values are late.  Unmatched predictions and annotations are not
    represented here; precision, recall, or F-measure must be inspected beside
    this conditional timing diagnostic.
    """

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be a positive finite number of seconds")
    predicted = np.sort(np.asarray(events_pred, dtype=float).reshape(-1))
    annotated = np.sort(np.asarray(events_ann, dtype=float).reshape(-1))
    predicted = predicted[np.isfinite(predicted)]
    annotated = annotated[np.isfinite(annotated)]
    if not len(predicted) or not len(annotated):
        return np.empty(0, dtype=float)
    matches = mir_eval.util.match_events(annotated, predicted, tolerance)
    return np.asarray(
        [
            predicted[pred_index] - annotated[ann_index]
            for ann_index, pred_index in matches
        ],
        dtype=float,
    )


def compute_event_timing_error_stats(
    events_pred: np.ndarray,
    events_ann: np.ndarray,
    tolerance: float = 0.07,
) -> Dict[str, float]:
    """Summarize matched-event timing errors in milliseconds.

    ``timing_error_mae_ms`` is the main answer to "how many milliseconds is
    the event off by?"  ``timing_error_mean_ms`` retains direction and is the
    signed timing bias.  All error statistics are conditional on matched
    events inside the tolerance window.
    """

    predicted = np.asarray(events_pred, dtype=float).reshape(-1)
    annotated = np.asarray(events_ann, dtype=float).reshape(-1)
    predicted_count = int(np.count_nonzero(np.isfinite(predicted)))
    annotated_count = int(np.count_nonzero(np.isfinite(annotated)))
    errors_ms = 1000.0 * compute_event_timing_errors(
        predicted,
        annotated,
        tolerance=tolerance,
    )
    matched = int(len(errors_ms))
    empty = {
        "timing_error_mean_ms": 0.0,
        "timing_error_std_ms": 0.0,
        "timing_error_mae_ms": 0.0,
        "timing_error_std_absolute_ms": 0.0,
        "timing_error_median_absolute_ms": 0.0,
        "timing_error_p95_absolute_ms": 0.0,
        "timing_error_p99_absolute_ms": 0.0,
        "timing_matched": float(matched),
        "timing_matched_reference_fraction": (
            float(matched / annotated_count) if annotated_count else 0.0
        ),
        "timing_matched_prediction_fraction": (
            float(matched / predicted_count) if predicted_count else 0.0
        ),
    }
    if not matched:
        return empty
    absolute_errors_ms = np.abs(errors_ms)
    return {
        **empty,
        "timing_error_mean_ms": float(np.mean(errors_ms)),
        "timing_error_std_ms": float(np.std(errors_ms)),
        "timing_error_mae_ms": float(np.mean(absolute_errors_ms)),
        "timing_error_std_absolute_ms": float(np.std(absolute_errors_ms)),
        "timing_error_median_absolute_ms": float(np.median(absolute_errors_ms)),
        "timing_error_p95_absolute_ms": float(
            np.percentile(absolute_errors_ms, 95)
        ),
        "timing_error_p99_absolute_ms": float(
            np.percentile(absolute_errors_ms, 99)
        ),
    }


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
    metrics.update(
        compute_event_timing_error_stats(
            beats_pred,
            beats_ann,
            tolerance=tolerance,
        )
    )

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
