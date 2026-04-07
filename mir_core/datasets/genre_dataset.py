"""
Dataset for genre classification from fixed-duration audio segments.
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class GenreDataset(Dataset):
    """Dataset for genre classification from fixed-duration audio segments.

    Extracts multiple random segments per track and computes log-mel
    spectrograms (or MFCCs) on the fly. Each item is a dict with
    ``mel``, ``label`` (int), and ``genre`` (str).

    Args:
        audio_paths: List of paths to audio files.
        genre_labels: List of genre label strings (same length as audio_paths).
        genre_to_idx: Mapping from genre label string to integer class index.
        sample_rate: Target sample rate for loading audio.
        segment_duration: Duration of each segment in seconds.
        n_mels: Number of mel bands.
        hop_length: STFT hop length.
        segments_per_track: How many random segments to extract per track.
        augment: Whether to apply basic data augmentation (gain jitter).
        feature_type: "mel" for log-mel spectrogram, "mfcc" for MFCCs+deltas.
        n_mfcc: Number of MFCCs (only used when feature_type="mfcc").
    """

    def __init__(
        self,
        audio_paths: List[str],
        genre_labels: List[str],
        genre_to_idx: Dict[str, int],
        sample_rate: int = 22050,
        segment_duration: float = 3.0,
        n_mels: int = 128,
        hop_length: int = 512,
        segments_per_track: int = 3,
        augment: bool = False,
        feature_type: str = "mel",
        n_mfcc: int = 20,
    ):
        self.audio_paths = audio_paths
        self.genre_labels = genre_labels
        self.genre_to_idx = genre_to_idx
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segments_per_track = segments_per_track
        self.augment = augment
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc

        self.target_samples = int(sample_rate * segment_duration)

    def __len__(self) -> int:
        return len(self.audio_paths) * self.segments_per_track

    def __getitem__(self, idx: int) -> Dict[str, object]:
        import librosa

        track_idx = idx // self.segments_per_track
        seg_idx = idx % self.segments_per_track

        path = self.audio_paths[track_idx]
        genre = self.genre_labels[track_idx]
        label = self.genre_to_idx[genre]

        # Load full track
        audio, sr = librosa.load(path, sr=self.sample_rate, mono=True)

        # Extract a segment (deterministic per seg_idx for reproducibility)
        audio_seg = self._extract_segment(audio, seg_idx)

        # Optional augmentation
        if self.augment:
            gain = np.random.uniform(0.8, 1.2)
            audio_seg = audio_seg * gain

        # Compute features
        if self.feature_type == "mfcc":
            features = self._compute_mfcc(audio_seg)
        else:
            features = self._compute_mel(audio_seg)

        return {
            "mel": torch.from_numpy(features).float(),
            "label": label,
            "genre": genre,
        }

    def _extract_segment(self, audio: np.ndarray, seg_idx: int) -> np.ndarray:
        """Extract a fixed-length segment from audio."""
        if len(audio) <= self.target_samples:
            return np.pad(audio, (0, self.target_samples - len(audio)))

        max_start = len(audio) - self.target_samples
        # Spread segments evenly across the track
        if self.segments_per_track > 1:
            start = int(seg_idx * max_start / (self.segments_per_track - 1))
        else:
            start = max_start // 2
        start = min(start, max_start)
        return audio[start : start + self.target_samples]

    def _compute_mel(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram. Returns shape (1, n_mels, time)."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=self.n_mels, hop_length=self.hop_length,
        )
        log_mel = np.log(mel + 1e-6)
        return log_mel[np.newaxis, :, :]  # (1, n_mels, time)

    def _compute_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Compute MFCCs + deltas. Returns shape (1, n_mfcc*3, time)."""
        import librosa

        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            n_mfcc=self.n_mfcc, hop_length=self.hop_length,
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)
        return features[np.newaxis, :, :]  # (1, n_mfcc*3, time)
