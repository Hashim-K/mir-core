"""Model-specific preprocessing contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from mir_core.preprocessing import (
    BEAST_FEATURE_DIM,
    BEAST_HOP_LENGTH,
    BEAST_N_FFT,
    BEAST_N_MELS,
    BEAST_SAMPLE_RATE,
    BEATNET_PLUS_FEATURE_DIM,
    BEATNET_PLUS_HOP_LENGTH,
    BEATNET_PLUS_N_BANDS,
    BEATNET_PLUS_SAMPLE_RATE,
    BEATNET_PLUS_WIN_LENGTH,
    SPECTNT_N_FREQUENCIES,
    SPECTNT_N_FFT,
    SPECTNT_N_HARMONIC,
    SPECTNT_HOP_LENGTH,
    SPECTNT_SAMPLE_RATE,
    BeastPreProcessor,
    SpecTNTPreProcessor,
    get_preprocessor_for_model,
)


def test_beatnet_plus_preprocessor_defaults_are_reference_contract() -> None:
    pytest.importorskip("madmom")
    from mir_core.preprocessing import BeatNetPlusPreProcessor

    processor = BeatNetPlusPreProcessor()

    assert processor.sample_rate == BEATNET_PLUS_SAMPLE_RATE
    assert processor.win_length == BEATNET_PLUS_WIN_LENGTH
    assert processor.hop_size == BEATNET_PLUS_HOP_LENGTH
    assert processor.n_bands == BEATNET_PLUS_N_BANDS
    assert processor.feature_dim == BEATNET_PLUS_FEATURE_DIM


def test_beast_preprocessor_defaults_are_reference_contract() -> None:
    processor = BeastPreProcessor()

    assert processor.sample_rate == BEAST_SAMPLE_RATE
    assert processor.n_fft == BEAST_N_FFT
    assert processor.hop_length == BEAST_HOP_LENGTH
    assert processor.n_mels == BEAST_N_MELS
    assert processor.feature_dim == BEAST_FEATURE_DIM


def test_beast_preprocessor_returns_db_mel_frames() -> None:
    pytest.importorskip("librosa")
    processor = BeastPreProcessor()

    features = processor(np.zeros(BEAST_SAMPLE_RATE, dtype=np.float32), BEAST_SAMPLE_RATE)

    assert features.ndim == 2
    assert features.shape[1] == BEAST_FEATURE_DIM
    assert np.all(features <= 0)


def test_spectnt_preprocessor_defaults_are_reference_contract() -> None:
    pytest.importorskip("torchaudio")
    processor = SpecTNTPreProcessor()

    assert processor.sample_rate == SPECTNT_SAMPLE_RATE
    assert processor.n_fft == SPECTNT_N_FFT
    assert processor.hop_length == SPECTNT_HOP_LENGTH
    assert processor.n_harmonic == SPECTNT_N_HARMONIC
    assert processor.level == SPECTNT_N_FREQUENCIES


def test_spectnt_preprocessor_returns_harmonic_tensor_shape() -> None:
    pytest.importorskip("torchaudio")
    processor = SpecTNTPreProcessor()

    features = processor(np.zeros(SPECTNT_SAMPLE_RATE, dtype=np.float32), SPECTNT_SAMPLE_RATE)

    assert features.ndim == 3
    assert features.shape[0] == SPECTNT_N_HARMONIC
    assert features.shape[1] == SPECTNT_N_FREQUENCIES


def test_preprocessor_registry_uses_model_specific_frontends() -> None:
    assert get_preprocessor_for_model("beast").__class__ is BeastPreProcessor

    pytest.importorskip("torchaudio")
    assert get_preprocessor_for_model("spectnt").__class__ is SpecTNTPreProcessor
