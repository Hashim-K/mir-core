"""
PyTorch Dataset for BockTCN beat tracking (81-dim madmom mel-spectrogram, 100 FPS).
"""

from typing import Dict, List, Tuple, Any

import numpy as np
import madmom
from scipy.ndimage import maximum_filter1d
from torch.utils.data import Dataset

from mir_core.preprocessing import PreProcessor, FPS


class BeatDataset(Dataset):
    """
    PyTorch Dataset for beat tracking.

    Handles audio loading, preprocessing to spectrograms,
    and beat annotation quantization.

    Args:
        tracks: Dictionary of mirdata track objects
        track_keys: List of track IDs to include
        fps: Frames per second for spectrogram
        widen: Whether to widen beat targets for training
        pad_frames: Number of frames to pad at start/end
        include_downbeats: Whether to include downbeat targets
        include_tempo: Whether to include tempo targets
    """

    def __init__(
        self,
        tracks: Dict[str, Any],
        track_keys: List[str],
        fps: int = FPS,
        widen: bool = False,
        pad_frames: int = 2,
        include_downbeats: bool = False,
        include_tempo: bool = False,
    ):
        self.fps = fps
        self.keys = track_keys
        self.tracks = {k: tracks[k] for k in track_keys}
        self.pre_processor = PreProcessor(fps=fps)
        self.pad_frames = pad_frames
        self.widen = widen
        self.include_downbeats = include_downbeats
        self.include_tempo = include_tempo

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        data = {}
        tid = self.keys[idx]
        track = self.tracks[tid]

        # Load audio
        audio, sr = track.audio
        if audio.ndim == 2:
            # Handle stereo by averaging channels
            if audio.shape[0] == 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2:
                audio = audio.mean(axis=1)

        # Preprocess to spectrogram
        signal = madmom.audio.Signal(audio, sr, num_channels=1)
        x = self.pre_processor(signal)

        # Pad features
        x_padded = self._pad_features(x)

        # Handle missing beat annotations — skip forward but guard against
        # infinite recursion when *all* tracks lack beats.
        if track.beats is None:
            next_idx = (idx + 1) % len(self)
            if next_idx == idx:
                raise RuntimeError(f"Track {tid} has no beat annotations and it is the only track.")
            print(f"Warning: Track {tid} has no beat information. Skipping.")
            return self.__getitem__(next_idx)

        # Quantize beats
        beats = track.beats.times
        beats_quantized = madmom.utils.quantize_events(
            beats, fps=self.fps, length=len(x)
        ).astype("float32")

        if self.widen:
            beats_quantized = self._widen_targets(beats_quantized)

        # Build output dictionary
        data["x"] = np.expand_dims(x_padded, axis=0)  # Add channel dim
        data["audio"] = audio
        data["sr"] = sr
        data["beats"] = beats_quantized
        data["beats_ann"] = track.beats.times
        data["track_id"] = tid

        # Optional: downbeats
        if self.include_downbeats:
            data["downbeats"], data["downbeats_ann"] = self._get_downbeats(track, len(x))

        # Optional: tempo
        if self.include_tempo:
            data["tempo"] = self._get_tempo(track.beats.times)

        return data

    def _pad_features(self, x: np.ndarray) -> np.ndarray:
        """Pad features at start and end."""
        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)
        return np.concatenate((pad_start, x, pad_stop))

    def _widen_targets(
        self,
        targets: np.ndarray,
        size: int = 3,
        value: float = 0.5
    ) -> np.ndarray:
        """Widen targets to give neighboring frames intermediate values."""
        if not np.allclose(targets, -1):  # Skip masked values
            np.maximum(
                targets,
                maximum_filter1d(targets, size=size) * value,
                out=targets
            )
        return targets

    def _get_downbeats(
        self,
        track: Any,
        length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract downbeat annotations."""
        try:
            positions = track.beats.positions.astype(int)
            downbeat_mask = positions == 1
            downbeat_times = track.beats.times[downbeat_mask]
            downbeats_quantized = madmom.utils.quantize_events(
                downbeat_times, fps=self.fps, length=length
            ).astype("float32")

            if self.widen:
                downbeats_quantized = self._widen_targets(downbeats_quantized)

            return downbeats_quantized, downbeat_times
        except AttributeError:
            # No downbeat info available, return masked values
            return np.ones(length, dtype="float32") * -1, np.array([])

    def _get_tempo(
        self,
        beat_times: np.ndarray,
        num_bins: int = 300
    ) -> np.ndarray:
        """Infer tempo from beat times."""
        from mir_core.preprocessing import infer_tempo

        try:
            tempo = infer_tempo(beat_times * self.fps / 100, fps=self.fps)
            # One-hot encode tempo
            tempo_idx = int(np.round(tempo))
            if 0 <= tempo_idx < num_bins:
                tempo_target = np.zeros(num_bins, dtype="float32")
                tempo_target[tempo_idx] = 1.0
                # Widen tempo targets
                if self.widen:
                    tempo_target = self._widen_targets(tempo_target)
                    tempo_target = self._widen_targets(tempo_target)
                return tempo_target
        except (IndexError, ValueError):
            pass

        # Return masked values if tempo extraction fails
        return np.ones(num_bins, dtype="float32") * -1
