"""Tests for beat/downbeat evaluation metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mir_core.evaluation.metrics import (
    compute_beat_metrics,
    compute_event_timing_error_stats,
    compute_event_timing_errors,
    compute_ibi_stats,
)


def test_compute_ibi_stats_includes_tempo_and_variability() -> None:
    stats = compute_ibi_stats(np.array([0.0, 0.5, 1.0, 1.6]), label="ibi")

    assert stats["ibi_n"] == 3
    assert stats["ibi_median"] == 0.5
    assert stats["ibi_bpm_mean"] > 0
    assert "ibi_cv" in stats


def test_compute_beat_metrics_includes_diagnostics() -> None:
    ann = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    pred = np.array([0.0, 0.5, 1.0, 1.5])

    metrics = compute_beat_metrics(pred, ann)

    assert metrics["num_pred"] == 4
    assert metrics["num_ann"] == 5
    assert metrics["beat_count_ratio"] == 0.8
    assert metrics["beat_count_abs_error"] == 1.0
    assert metrics["tempo_pred_bpm"] == 120.0
    assert metrics["tempo_ann_bpm"] == 120.0
    assert "ibi_median" in metrics
    assert "ibi_ann_median" in metrics


def test_event_timing_errors_are_signed_one_to_one_matches() -> None:
    annotated = np.array([0.1, 0.6, 1.1, 1.6])
    predicted = np.array([0.090, 0.620, 1.300, 1.590])

    errors = compute_event_timing_errors(predicted, annotated, tolerance=0.07)
    stats = compute_event_timing_error_stats(predicted, annotated, tolerance=0.07)

    assert len(errors) == 3
    assert errors * 1000.0 == pytest.approx([-10.0, 20.0, -10.0])
    assert stats["timing_error_mean_ms"] == pytest.approx(0.0)
    assert stats["timing_error_mae_ms"] == pytest.approx(40.0 / 3.0)
    assert stats["timing_error_std_absolute_ms"] == pytest.approx(
        np.std([10.0, 20.0, 10.0])
    )
    assert stats["timing_error_p95_absolute_ms"] == pytest.approx(19.0)
    assert stats["timing_error_p99_absolute_ms"] == pytest.approx(19.8)
    assert stats["timing_matched_reference_fraction"] == pytest.approx(0.75)
    assert stats["timing_matched_prediction_fraction"] == pytest.approx(0.75)


def test_compute_beat_metrics_empty_case_uses_full_schema() -> None:
    populated = compute_beat_metrics(
        np.arange(0.0, 12.0, 0.5),
        np.arange(0.0, 12.0, 0.5),
    )
    empty = compute_beat_metrics(np.array([]), np.arange(0.0, 12.0, 0.5))

    assert set(empty) == set(populated)
    assert empty["fmeasure"] == 0.0
    assert empty["ibi_n"] == 0
    assert empty["ibi_ann_n"] == 23
