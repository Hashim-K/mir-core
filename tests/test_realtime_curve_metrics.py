"""Regression tests for normalized real-time F1 curves."""

from __future__ import annotations

import pytest

from mir_core.evaluation.metrics import compute_joint_realtime_f1_curve


def test_joint_rt_f1_nauc_joins_each_tolerance_before_integration() -> None:
    result = compute_joint_realtime_f1_curve(
        beat_f1=[1.0, 0.0],
        downbeat_f1=[0.0, 1.0],
        tolerances=[0.03, 0.15],
    )

    assert result["f1"] == pytest.approx([0.0, 0.0])
    assert result["nauc"] == pytest.approx(0.0)
    # Integrating first would give 0.5 for each event type and would therefore
    # produce a misleading harmonic mean of 0.5.
    assert result["nauc"] != pytest.approx(0.5)
