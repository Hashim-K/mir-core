"""
Lightweight mel-spectrogram preprocessor using librosa.

Classes:
    SimpleMelPreProcessor — For quick experiments where the full madmom
                            pipeline is not needed.
"""

import numpy as np

from .utils import (
    BEAST_SAMPLE_RATE,
    BEAST_N_FFT,
    BEAST_HOP_LENGTH,
    BEAST_N_MELS,
    BEAST_FMIN,
    BEAST_FMAX,
    BEAST_FEATURE_DIM,
)


class SimpleMelPreProcessor:
    """
    Simple mel-spectrogram preprocessor using librosa.

    For use cases where the full BeatNet pipeline is not needed.

    Args:
        sample_rate: Target sample rate
        n_fft: FFT window size
        hop_length: Hop length in samples
        n_mels: Number of mel bands
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 220,
        n_mels: int = 81,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fps = sample_rate / hop_length

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio to log-mel spectrogram.

        Args:
            audio: Audio signal
            sr: Sample rate of input audio

        Returns:
            Log-mel spectrogram of shape (time, n_mels)
        """
        import librosa

        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Apply log compression
        log_mel = np.log(mel_spec + 1e-6)

        # Transpose to (time, freq)
        return log_mel.T


class BeastPreProcessor:
    """
    BEAST mel-spectrogram preprocessor.

    This follows the released BEAST data path: mono 44.1 kHz audio,
    4096-point FFT, 1024-sample hop, 128 mel bands from 30 Hz to 11 kHz, then
    `librosa.power_to_db(..., ref=np.max)` before model input.
    """

    def __init__(
        self,
        sample_rate: int = BEAST_SAMPLE_RATE,
        n_fft: int = BEAST_N_FFT,
        hop_length: int = BEAST_HOP_LENGTH,
        n_mels: int = BEAST_N_MELS,
        fmin: float = BEAST_FMIN,
        fmax: float = BEAST_FMAX,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.fps = sample_rate / hop_length
        self.feature_dim = BEAST_FEATURE_DIM

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio to BEAST dB-scaled mel features of shape (time, 128).
        """
        import librosa

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel_spec, ref=np.max).T
