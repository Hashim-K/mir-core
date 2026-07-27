from __future__ import annotations

import numpy as np
import pytest
import torch

from mir_core.models import GenreClassifier, GenreRouter


def _activations() -> dict[str, np.ndarray]:
    return {
        "candombe": np.asarray([1.0, 2.0]),
        "brid": np.asarray([3.0, 4.0]),
        "salsa": np.asarray([5.0, 6.0]),
        "other": np.asarray([7.0, 8.0]),
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"genre_labels": []}, "must not be empty"),
        ({"genre_labels": ["same", "same"]}, "must be unique"),
        ({"strategy": "unknown"}, "strategy must be"),
        ({"ema_alpha": 1.1}, "ema_alpha"),
        (
            {"confidence_threshold": 0.8, "blend_threshold": 0.7},
            "blend_threshold",
        ),
        ({"fallback_label": "global"}, "fallback_label"),
    ],
)
def test_router_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GenreRouter(**kwargs)


def test_router_validates_probabilities_before_mutating_state() -> None:
    router = GenreRouter(ema_alpha=1.0)
    initial = router.smoothed_probs

    with pytest.raises(ValueError, match="sum to 1"):
        router.update_probs(np.asarray([0.8, 0.3, 0.0, 0.0]))
    np.testing.assert_array_equal(router.smoothed_probs, initial)

    updated = router.update_probs(torch.tensor([0.8, 0.1, 0.05, 0.05]))
    np.testing.assert_allclose(updated, [0.8, 0.1, 0.05, 0.05], atol=1e-7)


def test_smoothed_probabilities_are_defensive_copies() -> None:
    router = GenreRouter()
    probabilities = router.smoothed_probs
    probabilities[0] = 1.0
    np.testing.assert_allclose(router.smoothed_probs, np.full(4, 0.25))


def test_router_uses_configured_fallback_below_confidence_threshold() -> None:
    router = GenreRouter(
        fallback_label="other",
        confidence_threshold=0.7,
        ema_alpha=1.0,
    )
    router.update_probs(np.asarray([0.4, 0.3, 0.2, 0.1]))
    output = router.route(_activations())
    np.testing.assert_array_equal(output, _activations()["other"])

    output[0] = -1
    assert _activations()["other"][0] == 7.0


def test_soft_router_returns_weighted_floating_point_activations() -> None:
    router = GenreRouter(
        strategy="soft",
        confidence_threshold=0.0,
        ema_alpha=1.0,
    )
    probabilities = np.asarray([0.5, 0.25, 0.125, 0.125])
    router.update_probs(probabilities)
    integer_activations = {
        label: values.astype(np.int64) for label, values in _activations().items()
    }

    routed = router.route(integer_activations)

    expected = sum(
        probability * integer_activations[label]
        for probability, label in zip(probabilities, router.genre_labels)
    )
    assert np.issubdtype(routed.dtype, np.floating)
    np.testing.assert_allclose(routed, expected)


def test_hybrid_tie_breaking_follows_label_order() -> None:
    router = GenreRouter(
        strategy="hybrid",
        ema_alpha=1.0,
        confidence_threshold=0.0,
        blend_threshold=0.8,
    )
    router.update_probs(np.asarray([0.4, 0.4, 0.1, 0.1]))
    np.testing.assert_allclose(
        router.route(_activations()),
        (_activations()["candombe"] + _activations()["brid"]) / 2,
    )


def test_router_requires_complete_shape_compatible_activations() -> None:
    router = GenreRouter()
    missing = _activations()
    missing.pop("brid")
    with pytest.raises(ValueError, match="missing genre labels"):
        router.route(missing)

    mismatched = _activations()
    mismatched["brid"] = np.zeros(3)
    with pytest.raises(ValueError, match="same shape"):
        router.route(mismatched)


def test_router_reset_restores_uniform_state() -> None:
    router = GenreRouter(ema_alpha=1.0)
    router.update_probs(np.asarray([0.7, 0.1, 0.1, 0.1]))
    router.reset()
    np.testing.assert_allclose(router.smoothed_probs, np.full(4, 0.25))


def test_genre_classifier_requires_label_contract_matching_num_classes() -> None:
    with pytest.raises(ValueError, match="exactly num_classes"):
        GenreClassifier(
            arch="embedding_stats_mlp",
            num_classes=4,
            genre_labels=["a", "b"],
            embedding_dim=8,
        )
