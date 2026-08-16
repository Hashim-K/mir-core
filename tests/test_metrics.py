"""Tests for beat/downbeat evaluation metric helpers."""

from __future__ import annotations

import inspect

import mir_eval
import numpy as np
import pytest

import mir_core.evaluation.metrics as metrics_module
from mir_core.evaluation.metrics import (
    DEFAULT_CEMGIL_SIGMA_SECONDS,
    DEFAULT_CONTINUITY_PERIOD_THRESHOLD,
    DEFAULT_CONTINUITY_PHASE_THRESHOLD,
    DEFAULT_F1_TOLERANCE_SECONDS,
    DEFAULT_GOTO_MU,
    DEFAULT_GOTO_SIGMA,
    DEFAULT_GOTO_THRESHOLD,
    DEFAULT_INFORMATION_GAIN_BINS,
    DEFAULT_P_SCORE_THRESHOLD,
    compute_beat_metrics,
    compute_event_timing_error_stats,
    compute_event_timing_errors,
    compute_ibi_stats,
    compute_realtime_event_metrics,
    compute_realtime_event_times,
    compute_realtime_f1_curve,
)


def test_conventional_metric_defaults_match_mir_eval() -> None:
    assert DEFAULT_F1_TOLERANCE_SECONDS == inspect.signature(
        mir_eval.beat.f_measure
    ).parameters["f_measure_threshold"].default
    assert DEFAULT_CEMGIL_SIGMA_SECONDS == inspect.signature(
        mir_eval.beat.cemgil
    ).parameters["cemgil_sigma"].default
    assert DEFAULT_P_SCORE_THRESHOLD == inspect.signature(
        mir_eval.beat.p_score
    ).parameters["p_score_threshold"].default

    goto_defaults = inspect.signature(mir_eval.beat.goto).parameters
    assert DEFAULT_GOTO_THRESHOLD == goto_defaults["goto_threshold"].default
    assert DEFAULT_GOTO_MU == goto_defaults["goto_mu"].default
    assert DEFAULT_GOTO_SIGMA == goto_defaults["goto_sigma"].default

    continuity_defaults = inspect.signature(mir_eval.beat.continuity).parameters
    assert (
        DEFAULT_CONTINUITY_PHASE_THRESHOLD
        == continuity_defaults["continuity_phase_threshold"].default
    )
    assert (
        DEFAULT_CONTINUITY_PERIOD_THRESHOLD
        == continuity_defaults["continuity_period_threshold"].default
    )
    assert DEFAULT_INFORMATION_GAIN_BINS == inspect.signature(
        mir_eval.beat.information_gain
    ).parameters["bins"].default


def test_f1_tolerance_does_not_change_other_metric_parameters(monkeypatch) -> None:
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    def capture(name: str, result: object):
        def wrapped(*args: object, **kwargs: object) -> object:
            calls[name] = (args, kwargs)
            return result

        return wrapped

    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "f_measure",
        capture("f_measure", 0.1),
    )
    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "cemgil",
        capture("cemgil", (0.2, 0.2)),
    )
    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "p_score",
        capture("p_score", 0.3),
    )
    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "goto",
        capture("goto", 0.4),
    )
    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "continuity",
        capture("continuity", (0.5, 0.6, 0.7, 0.8)),
    )
    monkeypatch.setattr(
        metrics_module.mir_eval.beat,
        "information_gain",
        capture("information_gain", 0.9),
    )

    annotated = np.arange(0.0, 12.0, 0.5)
    predicted = annotated + 0.05
    metrics = compute_beat_metrics(
        predicted,
        annotated,
        f1_tolerance=0.123,
    )

    assert calls["f_measure"][1] == {"f_measure_threshold": 0.123}
    assert calls["cemgil"][1] == {"cemgil_sigma": 0.04}
    assert calls["p_score"][1] == {"p_score_threshold": 0.2}
    assert calls["goto"][1] == {
        "goto_threshold": 0.35,
        "goto_mu": 0.2,
        "goto_sigma": 0.2,
    }
    assert calls["continuity"][1] == {
        "continuity_phase_threshold": 0.175,
        "continuity_period_threshold": 0.175,
    }
    assert calls["information_gain"][1] == {"bins": 41}
    assert metrics["timing_matched"] == len(annotated)


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


def test_realtime_event_times_wait_for_late_decisions_only() -> None:
    predicted = np.asarray([1.0, 2.0, 3.0])
    ready = np.asarray([0.9, 2.08, 2.5])

    effective = compute_realtime_event_times(predicted, ready)

    assert effective == pytest.approx([1.0, 2.08, 3.0])


def test_realtime_f1_penalizes_timestamp_correct_but_late_events() -> None:
    annotated = np.arange(0.5, 12.5, 0.5)
    predicted = annotated.copy()
    ready = predicted.copy()
    ready[8:] += 0.25

    conventional = compute_beat_metrics(
        predicted,
        annotated,
        f1_tolerance=0.07,
    )
    realtime = compute_realtime_event_metrics(
        predicted,
        ready,
        annotated,
        tolerance=0.07,
    )

    assert conventional["fmeasure"] == pytest.approx(1.0)
    assert realtime["rt_matched"] == 8
    assert realtime["rt_precision"] == pytest.approx(1.0 / 3.0)
    assert realtime["rt_recall"] == pytest.approx(1.0 / 3.0)
    assert realtime["rt_f1"] == pytest.approx(1.0 / 3.0)


def test_realtime_f1_curve_uses_agreed_tolerances_and_nauc() -> None:
    annotated = np.asarray([1.0, 2.0])
    predicted = annotated.copy()
    ready = np.asarray([1.04, 2.12])

    curve = compute_realtime_f1_curve(predicted, ready, annotated)

    assert curve["tolerances_seconds"] == pytest.approx(
        [0.03, 0.05, 0.07, 0.10, 0.15]
    )
    assert curve["f1"] == pytest.approx([0.0, 0.5, 0.5, 0.5, 1.0])
    expected_nauc = np.trapz(
        [0.0, 0.5, 0.5, 0.5, 1.0],
        [0.03, 0.05, 0.07, 0.10, 0.15],
    ) / 0.12
    assert curve["nauc"] == pytest.approx(expected_nauc)


def test_realtime_metrics_reject_misaligned_event_availability() -> None:
    with pytest.raises(ValueError, match="one aligned value"):
        compute_realtime_event_metrics(
            np.asarray([1.0, 2.0]),
            np.asarray([1.0]),
            np.asarray([1.0, 2.0]),
        )


def test_realtime_f1_does_not_rematch_long_latency_to_later_periodic_beats() -> None:
    annotated = np.arange(0.5, 20.5, 0.5)
    predicted = annotated.copy()
    ready = predicted + 2.0

    realtime = compute_realtime_event_metrics(
        predicted,
        ready,
        annotated,
        tolerance=0.07,
    )

    assert realtime["conventional_matched"] == len(annotated)
    assert realtime["rt_matched"] == 0
    assert realtime["rt_f1"] == pytest.approx(0.0)
