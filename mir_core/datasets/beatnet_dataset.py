"""
PyTorch Dataset for BeatNet training (272-dim LOG_SPECT, 50 FPS, 3-class targets).
"""

from typing import Dict, List, Any

import numpy as np
import madmom
from torch.utils.data import Dataset


class BeatNetDataset(Dataset):
    """
    PyTorch Dataset for BeatNet training with official 272-dim features.

    Uses the correct BeatNet feature extraction (LOG_SPECT):
    - 272-dim = 136 filterbank + 136 spectral difference
    - 50 FPS (20ms hop)
    - 3-class targets: non-beat (0), beat (1), downbeat (2)

    Args:
        tracks: Dictionary of mirdata track objects
        track_keys: List of track IDs to include
        fps: Frames per second (default 50 for BeatNet)
        widen_targets: Whether to widen beat targets
    """

    def __init__(
        self,
        tracks: Dict[str, Any],
        track_keys: List[str],
        fps: int = 50,
        widen_targets: bool = False,
    ):
        from mir_core.preprocessing import BeatNetPreProcessor

        self.fps = fps
        self.keys = track_keys
        self.tracks = {k: tracks[k] for k in track_keys}
        self.pre_processor = BeatNetPreProcessor()
        self.widen = widen_targets

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
            if audio.shape[0] == 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2:
                audio = audio.mean(axis=1)

        # Extract 272-dim features
        x = self.pre_processor(audio, sr)

        # Handle missing beat annotations — skip forward but guard against
        # infinite recursion when *all* tracks lack beats.
        if track.beats is None:
            next_idx = (idx + 1) % len(self)
            if next_idx == idx:
                raise RuntimeError(f"Track {tid} has no beat annotations and it is the only track.")
            print(f"Warning: Track {tid} has no beat info. Skipping.")
            return self.__getitem__(next_idx)

        # Get beat and downbeat times
        beat_times = track.beats.times

        # Create 3-class targets (non-beat=0, beat=1, downbeat=2)
        num_frames = len(x)
        targets = np.zeros(num_frames, dtype=np.int64)

        # Mark beat frames
        beat_frames = madmom.utils.quantize_events(
            beat_times, fps=self.fps, length=num_frames
        )

        # Mark downbeat frames (where position == 1)
        try:
            positions = track.beats.positions.astype(int)
            downbeat_mask = positions == 1
            downbeat_times = beat_times[downbeat_mask]
            downbeat_frames = madmom.utils.quantize_events(
                downbeat_times, fps=self.fps, length=num_frames
            )
        except AttributeError:
            downbeat_frames = np.zeros(num_frames)

        # Assign class labels
        for i in range(num_frames):
            if downbeat_frames[i] > 0.5:
                targets[i] = 2  # Downbeat
            elif beat_frames[i] > 0.5:
                targets[i] = 1  # Beat (non-downbeat)
            else:
                targets[i] = 0  # Non-beat

        # Build output dictionary
        data["x"] = x.astype(np.float32)  # (time, 272)
        data["targets"] = targets  # (time,) int64 for CE loss
        data["beats"] = beat_frames.astype(np.float32)  # For BCE if needed
        data["audio"] = audio
        data["sr"] = sr
        data["beats_ann"] = beat_times
        data["track_id"] = tid

        return data
