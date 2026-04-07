"""
Lightweight mel-spectrogram preprocessor using librosa.

Classes:
    SimpleMelPreProcessor — For quick experiments where the full madmom
                            pipeline is not needed.
"""

import numpy as np


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
