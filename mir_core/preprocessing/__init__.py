"""
Audio preprocessing utilities for beat tracking.

Provides spectrogram computation and feature extraction.

Classes:
    PreProcessor         — madmom SequentialProcessor producing 81-dim log-mel
                           spectrograms at 100 FPS (for BockTCN).
    BeatNetPreProcessor  — Official BeatNet LOG_SPECT pipeline producing 272-dim
                           features (136 filterbank + 136 spectral diff) at 50 FPS.
    SimpleMelPreProcessor — Lightweight librosa mel-spectrogram (for quick experiments).

Functions:
    infer_tempo   — estimate BPM from beat times via histogram peak picking.
    pad_features  — repeat-pad first/last frames (used by BeatDataset).
"""

from .utils import (
    FPS,
    NUM_BANDS,
    FFT_SIZE,
    MASK_VALUE,
    BEATNET_SAMPLE_RATE,
    BEATNET_HOP_LENGTH,
    BEATNET_WIN_LENGTH,
    BEATNET_N_BANDS,
    infer_tempo,
    pad_features,
)

from .madmom_features import PreProcessor, BeatNetPreProcessor
from .mel_features import SimpleMelPreProcessor

__all__ = [
    # Constants
    "FPS",
    "NUM_BANDS",
    "FFT_SIZE",
    "MASK_VALUE",
    "BEATNET_SAMPLE_RATE",
    "BEATNET_HOP_LENGTH",
    "BEATNET_WIN_LENGTH",
    "BEATNET_N_BANDS",
    # Classes
    "PreProcessor",
    "BeatNetPreProcessor",
    "SimpleMelPreProcessor",
    # Functions
    "infer_tempo",
    "pad_features",
]
