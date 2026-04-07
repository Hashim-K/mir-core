"""
PyTorch Dataset for beat/downbeat tracking.

Only loads audio and annotations. Feature extraction is NOT the dataset's
job — that belongs in the training pipeline (transforms, collate, or
explicit preprocessing step).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class BeatTrackingDataset(Dataset):
    """Dataset that loads audio waveforms and beat/downbeat annotations.

    Returns a dict with raw audio and annotation arrays. The training
    pipeline is responsible for feature extraction (madmom, BeatNet
    LOG_SPECT, mel, etc.) and target quantization.

    Args:
        tracks: Dictionary of mirdata track objects (must expose ``.audio``
            and ``.beats``).
        track_keys: Track IDs to include.
    """

    def __init__(
        self,
        tracks: Dict[str, Any],
        track_keys: List[str],
    ):
        self.keys = track_keys
        self.tracks = {k: tracks[k] for k in track_keys}

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tid = self.keys[idx]
        track = self.tracks[tid]

        # --- Load audio ---
        audio, sr = track.audio
        if audio.ndim == 2:
            if audio.shape[0] == 2:
                audio = audio.mean(axis=0)
            elif audio.shape[1] == 2:
                audio = audio.mean(axis=1)

        # --- Handle missing beat annotations ---
        if track.beats is None:
            next_idx = (idx + 1) % len(self)
            if next_idx == idx:
                raise RuntimeError(
                    f"Track {tid} has no beat annotations and it is the only track."
                )
            print(f"Warning: Track {tid} has no beat info. Skipping.")
            return self.__getitem__(next_idx)

        # --- Beat times ---
        beat_times = track.beats.times

        # --- Downbeat times (where metrical position == 1) ---
        downbeat_times: Optional[np.ndarray] = None
        try:
            positions = track.beats.positions.astype(int)
            downbeat_times = beat_times[positions == 1]
        except AttributeError:
            pass

        return {
            "audio": audio,
            "sr": sr,
            "beat_times": beat_times,
            "downbeat_times": downbeat_times,
            "track_id": tid,
        }
