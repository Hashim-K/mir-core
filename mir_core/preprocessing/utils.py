"""
Preprocessing utility constants and functions for beat tracking.

Constants:
    FPS, NUM_BANDS, FFT_SIZE, MASK_VALUE — BockTCN defaults.
    BEATNET_SAMPLE_RATE, BEATNET_HOP_LENGTH, BEATNET_WIN_LENGTH, BEATNET_N_BANDS — BeatNet defaults.

Functions:
    infer_tempo  — estimate BPM from beat times via histogram peak picking.
    pad_features — repeat-pad first/last frames (used by BeatDataset).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

import madmom

# Default constants
FPS = 100  # Frames per second
NUM_BANDS = 12  # Number of frequency bands
FFT_SIZE = 2048  # FFT window size
MASK_VALUE = -1  # Value for masked targets

# BeatNet-specific constants (from official implementation)
BEATNET_SAMPLE_RATE = 22050
BEATNET_HOP_LENGTH = 441  # 20ms hop -> 50 FPS
BEATNET_WIN_LENGTH = 1408  # 64ms window
BEATNET_N_BANDS = 24  # Results in 136-dim filterbank (with diff -> 272)


def infer_tempo(
    beats: np.ndarray,
    hist_smooth: int = 15,
    fps: int = FPS,
    no_tempo: float = MASK_VALUE
) -> float:
    """
    Infer global tempo from beat times.

    Args:
        beats: Beat times in frames (at given fps)
        hist_smooth: Smoothing window for histogram
        fps: Frames per second
        no_tempo: Value to return if no tempo can be inferred

    Returns:
        Estimated tempo in BPM
    """
    # Compute inter-beat intervals
    ibis = np.diff(beats) * fps
    bins = np.bincount(np.round(ibis).astype(int))

    # No beats = no tempo
    if not bins.any():
        return no_tempo

    intervals = np.arange(len(bins))

    # Smooth histogram
    if hist_smooth > 0:
        bins = madmom.audio.signal.smooth(bins, hist_smooth)

    # Interpolate for finer resolution
    interpolation_fn = interp1d(intervals, bins, 'quadratic')
    intervals = np.arange(intervals[0], intervals[-1], 0.001)
    tempi = 60.0 * fps / intervals
    bins = interpolation_fn(intervals)

    # Find peaks
    peaks = argrelmax(bins, mode='wrap')[0]

    if len(peaks) == 0:
        return no_tempo

    # Return strongest tempo
    sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
    return tempi[sorted_peaks][0]


def pad_features(x: np.ndarray, pad_frames: int = 2) -> np.ndarray:
    """
    Pad features by repeating first and last frames.

    Args:
        x: Feature array of shape (time, features)
        pad_frames: Number of frames to pad at each end

    Returns:
        Padded feature array
    """
    pad_start = np.repeat(x[:1], pad_frames, axis=0)
    pad_stop = np.repeat(x[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, x, pad_stop))
