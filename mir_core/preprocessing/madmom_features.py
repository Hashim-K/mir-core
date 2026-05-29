"""
Madmom-based audio preprocessing for beat tracking.

Classes:
    PreProcessor         — madmom SequentialProcessor producing 81-dim log-mel
                           spectrograms at 100 FPS (for BockTCN).
    BeatNetPreProcessor  — Official BeatNet LOG_SPECT pipeline producing 272-dim
                           features (136 filterbank + 136 spectral diff) at 50 FPS.
"""

import numpy as np

import madmom
from madmom.processors import SequentialProcessor, ParallelProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)

from .utils import (
    FPS,
    NUM_BANDS,
    FFT_SIZE,
    BEATNET_SAMPLE_RATE,
    BEATNET_HOP_LENGTH,
    BEATNET_WIN_LENGTH,
    BEATNET_N_BANDS,
    BEATNET_FEATURE_DIM,
    BEATNET_PLUS_SAMPLE_RATE,
    BEATNET_PLUS_HOP_LENGTH,
    BEATNET_PLUS_WIN_LENGTH,
    BEATNET_PLUS_N_BANDS,
    BEATNET_PLUS_FEATURE_DIM,
)


class PreProcessor(SequentialProcessor):
    """
    Audio preprocessor that converts raw audio to log-magnitude spectrograms.

    Pipeline:
    1. Resample to 44100 Hz, mono
    2. Frame the signal with given FPS
    3. Compute STFT
    4. Apply mel filterbank
    5. Apply log compression

    Args:
        frame_size: FFT window size
        num_bands: Number of mel bands per octave
        log: Log function for compression
        add: Small value to add before log (for numerical stability)
        fps: Frames per second
    """

    def __init__(
        self,
        frame_size: int = FFT_SIZE,
        num_bands: int = NUM_BANDS,
        log=np.log,
        add: float = 1e-6,
        fps: int = FPS
    ):
        # Resample to fixed sample rate for consistent features
        sig = SignalProcessor(num_channels=1, sample_rate=44100)
        # Split audio signal into overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # Compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # Apply mel filterbank
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # Apply log compression
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)

        # Initialize sequential processor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))

        # Store fps as attribute (needed for quantization)
        self.fps = fps


class BeatNetPreProcessor:
    """
    Official BeatNet feature extraction (LOG_SPECT).

    This matches the exact feature extraction from the official BeatNet:
    https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/log_spect.py

    Extracts log-magnitude filterbank features with spectral difference,
    resulting in 272-dimensional features (136 filterbank + 136 diff).

    Args:
        sample_rate: Target sample rate (22050 Hz for BeatNet)
        win_length: Window length in samples (1408 = 64ms)
        hop_size: Hop size in samples (441 = 20ms -> 50 FPS)
        n_bands: Number of filterbank bands per octave (24 -> 136 total)
        mode: Processing mode ('online', 'offline', 'realtime', 'stream')
    """

    def __init__(
        self,
        sample_rate: int = BEATNET_SAMPLE_RATE,
        win_length: int = BEATNET_WIN_LENGTH,
        hop_size: int = BEATNET_HOP_LENGTH,
        n_bands: int = BEATNET_N_BANDS,
        mode: str = 'online',
    ):
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_size = hop_size
        self.n_bands = n_bands
        self.fps = sample_rate / hop_size  # Should be ~50 FPS
        self.feature_dim = BEATNET_FEATURE_DIM if win_length == BEATNET_WIN_LENGTH else None

        # Build the madmom processing pipeline (matching official LOG_SPECT)
        sig = SignalProcessor(num_channels=1, sample_rate=sample_rate)

        multi = ParallelProcessor([])
        frame_sizes = [win_length]
        num_bands_list = [n_bands]

        for frame_size, num_bands in zip(frame_sizes, num_bands_list):
            if mode == 'online' or mode == 'offline':
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size)
            else:  # realtime/stream modes
                frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, num_frames=4)

            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True
            )
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack
            )

            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))

        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def __call__(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Process audio to BeatNet features.

        Args:
            audio: Audio signal (numpy array or madmom Signal)
            sr: Sample rate (optional, used for resampling if provided)

        Returns:
            Features of shape (time, 272) - 136 filterbank + 136 diff
        """
        import librosa

        # Handle madmom Signal objects
        if hasattr(audio, 'sample_rate'):
            return self.pipe(audio)  # Already (time, 272)

        # Resample if necessary
        if sr is not None and sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # Create madmom signal
        signal = madmom.audio.Signal(audio, self.sample_rate, num_channels=1)

        # Process - already (time, features)
        feats = self.pipe(signal)
        return feats

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Alternative interface matching official LOG_SPECT.process_audio()."""
        if hasattr(audio, 'sample_rate'):
            return self.pipe(audio)
        signal = madmom.audio.Signal(audio, self.sample_rate, num_channels=1)
        return self.pipe(signal)


class BeatNetPlusPreProcessor(BeatNetPreProcessor):
    """
    Official BeatNet+ LOG_SPECT feature extraction.

    This matches the BeatNet+ reference defaults:
    https://github.com/mjhydri/BeatNet-Plus/blob/main/src/BeatNetPlus/log_spect.py

    The pipeline is the same LOG_SPECT family as BeatNet, but BeatNet+ uses an
    80 ms analysis window. This increases the filtered spectrogram width and
    produces 288-dimensional features (144 filterbank + 144 diff).
    """

    def __init__(
        self,
        sample_rate: int = BEATNET_PLUS_SAMPLE_RATE,
        win_length: int = BEATNET_PLUS_WIN_LENGTH,
        hop_size: int = BEATNET_PLUS_HOP_LENGTH,
        n_bands: int = BEATNET_PLUS_N_BANDS,
        mode: str = 'online',
    ):
        super().__init__(
            sample_rate=sample_rate,
            win_length=win_length,
            hop_size=hop_size,
            n_bands=n_bands,
            mode=mode,
        )
        self.feature_dim = BEATNET_PLUS_FEATURE_DIM
