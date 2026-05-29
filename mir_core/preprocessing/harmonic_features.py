"""
Harmonic-STFT preprocessing for SpecTNT beat tracking.

This mirrors the feature extractor used by the released SpecTNT beat config.
"""

import numpy as np

from .utils import (
    SPECTNT_SAMPLE_RATE,
    SPECTNT_N_FFT,
    SPECTNT_HOP_LENGTH,
    SPECTNT_N_HARMONIC,
    SPECTNT_SEMITONE_SCALE,
    SPECTNT_N_FREQUENCIES,
)


def _midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _initialize_filterbank(sample_rate: int, n_harmonic: int, semitone_scale: int):
    import librosa

    low_midi = librosa.core.note_to_midi("C1")
    high_note = librosa.core.hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = librosa.core.note_to_midi(high_note)
    level = int((high_midi - low_midi) * semitone_scale)
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = _midi_to_hz(midi[:-1])

    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))
    return harmonic_hz.astype("float32"), level


class SpecTNTPreProcessor:
    """
    SpecTNT harmonic-STFT preprocessor for beat/downbeat tracking.

    The default parameters match `configs/beats.yaml` in the local SpecTNT
    reference code: 16 kHz audio, 512-point STFT, 256-sample hop, six harmonic
    channels, and two bins per semitone. The output shape is
    `(6, 128, time)` for a single audio signal, ready to be batched before
    passing to `mir_core.models.SpecTNT`.
    """

    def __init__(
        self,
        sample_rate: int = SPECTNT_SAMPLE_RATE,
        n_fft: int = SPECTNT_N_FFT,
        hop_length: int = SPECTNT_HOP_LENGTH,
        n_harmonic: int = SPECTNT_N_HARMONIC,
        semitone_scale: int = SPECTNT_SEMITONE_SCALE,
        bw_q: float = 1.0,
        checkpoint: str | None = None,
        device: str = "cpu",
    ):
        import torch
        import torchaudio

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_harmonic = n_harmonic
        self.semitone_scale = semitone_scale
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7
        self.fps = sample_rate / hop_length
        self.feature_dim = SPECTNT_N_FREQUENCIES
        self.device = torch.device(device)

        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            window_fn=torch.hann_window,
            power=2,
            normalized=False,
        ).to(self.device)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

        harmonic_hz, self.level = _initialize_filterbank(
            sample_rate,
            n_harmonic,
            semitone_scale,
        )
        self.f0 = torch.tensor(harmonic_hz, device=self.device)
        self.bw_q = torch.tensor(np.array([bw_q]).astype("float32"), device=self.device)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=self.device)
            hstft_state = {
                k.replace("hstft.", ""): v
                for k, v in state_dict.items()
                if "hstft." in k
            }
            if "bw_Q" in hstft_state:
                self.bw_q = hstft_state["bw_Q"].to(self.device)

    def _harmonic_filterbank(self, n_bins: int):
        import torch

        fft_bins = torch.linspace(0, self.sample_rate // 2, n_bins, device=self.device)
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_q
        bw = bw.unsqueeze(0)
        f0 = self.f0.unsqueeze(0)
        fft_bins = fft_bins.unsqueeze(1)
        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        return torch.maximum(
            torch.zeros(1, device=self.device),
            torch.minimum(down_slope, up_slope),
        )

    def process_tensor(self, audio):
        """
        Process a torch tensor to harmonic features.

        Input shape can be `(time,)` or `(batch, time)`. Output shape is
        `(6, 128, time)` for unbatched input or `(batch, 6, 128, time)` for
        batched input.
        """
        import torch

        unbatched = audio.ndim == 1
        if unbatched:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        spectrogram = self.spec(audio)
        harmonic_fb = self._harmonic_filterbank(spectrogram.size(1))
        harmonic_spec = torch.matmul(
            spectrogram.transpose(1, 2),
            harmonic_fb,
        ).transpose(1, 2)
        harmonic_spec = harmonic_spec.view(
            audio.shape[0],
            self.n_harmonic,
            self.level,
            spectrogram.size(-1),
        )
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec.squeeze(0) if unbatched else harmonic_spec

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Process audio to SpecTNT harmonic features.
        """
        import librosa
        import torch

        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        tensor = torch.from_numpy(audio.astype("float32"))
        return self.process_tensor(tensor).detach().cpu().numpy()
