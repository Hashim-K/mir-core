"""Model-to-preprocessor contract helpers."""

from __future__ import annotations

from typing import Callable

from .madmom_features import PreProcessor, BeatNetPreProcessor, BeatNetPlusPreProcessor
from .mel_features import BeastPreProcessor
from .harmonic_features import SpecTNTPreProcessor


PREPROCESSOR_BY_MODEL: dict[str, Callable[[], object]] = {
    "bock_tcn": PreProcessor,
    "bocktcn": PreProcessor,
    "beatnet": BeatNetPreProcessor,
    "beatnet_crnn": BeatNetPreProcessor,
    "multihead_beatnet": BeatNetPreProcessor,
    "beatnet_plus": BeatNetPlusPreProcessor,
    "beatnet+": BeatNetPlusPreProcessor,
    "beatnet-plus": BeatNetPlusPreProcessor,
    "beast": BeastPreProcessor,
    "spectnt": SpecTNTPreProcessor,
}


def get_preprocessor_for_model(model_name: str, **kwargs):
    """
    Instantiate the canonical preprocessor for a supported model.

    Preprocessing is part of the architecture contract for the beat-tracking
    models. This helper keeps model/front-end pairing explicit at call sites.
    """
    key = model_name.lower().replace("-", "_")
    if key not in PREPROCESSOR_BY_MODEL:
        supported = ", ".join(sorted(PREPROCESSOR_BY_MODEL))
        raise ValueError(
            f"No canonical preprocessor registered for {model_name!r}. "
            f"Supported model names: {supported}."
        )
    return PREPROCESSOR_BY_MODEL[key](**kwargs)
