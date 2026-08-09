from __future__ import annotations

import numpy as np
import pytest
import torch

from mir_core.classifier.evaluation import (
    accuracy,
    apply_temperature,
    binary_ranking_diagnostics,
    classification_diagnostics,
    confidence_rejection_diagnostics,
    confusion_matrix,
    expected_calibration_error,
    fit_temperature,
    macro_f1,
    negative_log_likelihood,
    select_routing_threshold,
    softmax_probabilities,
)


def test_metrics_accept_logits_and_use_target_rows() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
        ]
    )
    targets = np.asarray([0, 1, 2, 2])

    assert accuracy(logits, targets) == pytest.approx(0.5)
    np.testing.assert_array_equal(
        confusion_matrix(logits, targets, num_classes=3),
        np.asarray(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
            ]
        ),
    )
    assert macro_f1(logits, targets, num_classes=3) == pytest.approx(0.5)


def test_metrics_accept_one_hot_targets_and_explicit_absent_class() -> None:
    predictions = [0, 1, 1]
    targets = np.eye(3)[[0, 1, 2]]

    assert accuracy(predictions, targets) == pytest.approx(2 / 3)
    assert confusion_matrix(predictions, targets, num_classes=4).shape == (4, 4)
    assert macro_f1(predictions, targets, num_classes=4) == pytest.approx(
        (1.0 + 2.0 / 3.0 + 0.0 + 0.0) / 4.0
    )


def test_classification_diagnostics_are_detailed_and_json_safe() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.20, 0.70, 0.10],
            [0.10, 0.60, 0.30],
            [0.10, 0.20, 0.70],
        ]
    )
    report = classification_diagnostics(
        probabilities,
        [0, 1, 2, 2],
        class_names=["candombe", "salsa", "other"],
        calibration_bins=5,
    )

    assert report["accuracy"] == pytest.approx(0.75)
    assert report["balanced_accuracy"] == pytest.approx((1.0 + 1.0 + 0.5) / 3.0)
    assert report["labels"] == ["candombe", "salsa", "other"]
    assert report["averages"]["micro"]["f1"] == pytest.approx(0.75)
    assert report["averages"]["macro"]["roc_auc"] is not None
    assert report["averages"]["macro"]["average_precision"] is not None
    assert report["per_class"][2]["support"] == 2
    assert report["per_class"][2]["recall"] == pytest.approx(0.5)
    assert len(report["calibration_bins"]) == 5
    assert report["confusion"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    assert report["confusion_normalized"][2] == pytest.approx([0.0, 0.5, 0.5])


def test_classification_diagnostics_marks_undefined_ranking_metrics() -> None:
    report = classification_diagnostics(
        [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
        [0, 0],
        class_names=["present", "absent-a", "absent-b"],
    )

    assert report["per_class"][0]["roc_auc"] is None
    assert report["per_class"][0]["average_precision"] == pytest.approx(1.0)
    assert report["per_class"][1]["roc_auc"] is None
    assert report["per_class"][1]["average_precision"] is None
    assert report["averages"]["macro"]["roc_auc"] is None


def test_binary_ranking_diagnostics_report_prevalence_and_ties() -> None:
    report = binary_ranking_diagnostics(
        [0.9, 0.8, 0.8, 0.1],
        [1, 0, 1, 0],
    )

    assert report["num_samples"] == 4
    assert report["positive_count"] == 2
    assert report["negative_count"] == 2
    assert report["positive_prevalence"] == pytest.approx(0.5)
    assert report["roc_auc"] == pytest.approx(0.875)
    assert report["average_precision"] == pytest.approx(5 / 6)


def test_confidence_rejection_diagnostics_distinguish_known_non_target() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.15, 0.10, 0.75],
            [0.40, 0.35, 0.25],
        ]
    )
    report = confidence_rejection_diagnostics(
        probabilities,
        [0, 1, 2, 2],
        non_target_index=2,
        coverage_levels=(0.5, 1.0),
    )

    error = report["maximum_softmax_error_detection"]
    assert error["positive_count"] == 1
    assert error["roc_auc"] == pytest.approx(1.0)
    selective = report["selective_prediction"]
    assert selective["aurc"] == pytest.approx(0.0625)
    assert selective["operating_points"][0]["risk"] == pytest.approx(0.0)
    assert selective["operating_points"][1]["risk"] == pytest.approx(0.25)
    non_target = report["non_target_detection"]
    assert non_target["valid_unseen_ood_claim"] is False
    assert non_target["explicit_non_target_probability"]["roc_auc"] == pytest.approx(
        1.0
    )
    assert non_target["target_bank_rejection"]["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("scores", "targets", "match"),
    [
        ([], [], "must not be empty"),
        ([0.1], [0, 1], "same number"),
        ([float("nan")], [1], "scores must be finite"),
        ([0.1], [2], "only contain 0 and 1"),
    ],
)
def test_binary_ranking_diagnostics_reject_invalid_inputs(
    scores, targets, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        binary_ranking_diagnostics(scores, targets)


def test_confidence_rejection_diagnostics_validate_non_target_index() -> None:
    with pytest.raises(ValueError, match="non_target_index"):
        confidence_rejection_diagnostics(
            [[0.8, 0.2]],
            [0],
            non_target_index=2,
        )


@pytest.mark.parametrize(
    ("predictions", "targets", "match"),
    [
        ([], [], "must not be empty"),
        ([0, 1], [0], "same number"),
        ([0.5], [0], "must be integers"),
        ([[float("nan"), 0.0]], [0], "must be finite"),
        ([2], [0], "absent from labels"),
    ],
)
def test_metrics_reject_invalid_inputs(predictions, targets, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        if match == "absent from labels":
            confusion_matrix(predictions, targets, labels=[0, 1])
        else:
            accuracy(predictions, targets)


def test_apply_temperature_preserves_torch_device_and_numpy_shape() -> None:
    tensor = torch.tensor([[2.0, 0.0]])
    scaled_tensor = apply_temperature(tensor, 2.0)
    assert isinstance(scaled_tensor, torch.Tensor)
    assert scaled_tensor.device == tensor.device
    torch.testing.assert_close(scaled_tensor, torch.tensor([[1.0, 0.0]]))

    scaled_array = apply_temperature(np.asarray([2.0, 0.0]), 2.0)
    np.testing.assert_allclose(scaled_array, [1.0, 0.0])


def test_temperature_fit_is_deterministic_and_improves_overconfident_nll() -> None:
    logits = np.asarray(
        [
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
        ]
    )
    targets = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])

    temperature = fit_temperature(logits, targets)

    assert temperature == pytest.approx(fit_temperature(logits, targets))
    assert temperature > 1.0
    assert negative_log_likelihood(
        logits,
        targets,
        temperature=temperature,
    ) < negative_log_likelihood(logits, targets)


def test_flat_logits_keep_identity_temperature() -> None:
    logits = np.zeros((4, 4))
    assert fit_temperature(logits, [0, 1, 2, 3]) == 1.0


def test_ece_and_nll_accept_probabilities() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.6, 0.4]])
    targets = np.asarray([0, 1])

    assert expected_calibration_error(
        probabilities,
        targets,
        n_bins=1,
        from_logits=False,
    ) == pytest.approx(0.2)
    assert negative_log_likelihood(
        probabilities,
        targets,
        from_logits=False,
    ) == pytest.approx(-np.log([0.8, 0.4]).mean())


def test_softmax_probabilities_are_normalized_after_temperature() -> None:
    probabilities = softmax_probabilities([[3.0, 1.0, -1.0]], temperature=2.0)
    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.max() < softmax_probabilities([[3.0, 1.0, -1.0]]).max()


def test_threshold_selection_improves_fallback_routing_deterministically() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.03, 0.02],
            [0.40, 0.35, 0.15, 0.10],
            [0.45, 0.20, 0.15, 0.20],
            [0.10, 0.10, 0.20, 0.60],
        ]
    )
    targets = np.asarray([0, 1, 3, 3])

    result = select_routing_threshold(
        probabilities,
        targets,
        fallback_class=3,
        objective="accuracy",
        from_logits=False,
    )

    assert 0.45 < result.threshold <= 0.6
    assert result.score == pytest.approx(0.75)
    assert result.accuracy == pytest.approx(0.75)
    assert result.coverage == pytest.approx(0.5)
    assert result.out_of_set_rejection == pytest.approx(1.0)


def test_threshold_selection_honors_minimum_coverage() -> None:
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.4, 0.35, 0.25]])
    result = select_routing_threshold(
        probabilities,
        [0, 2],
        from_logits=False,
        objective="accuracy",
        min_coverage=1.0,
    )
    assert result.threshold == 0.0
    assert result.coverage == 1.0


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan")])
def test_temperature_must_be_positive(temperature: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        apply_temperature([[1.0, 0.0]], temperature)
