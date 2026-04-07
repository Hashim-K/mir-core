"""
Visualization utilities for beat tracking evaluation.

Functions:
    plot_predictions         — spectrogram overlay with annotation + prediction vlines.
    plot_activations         — beat activation curve with annotation markers.
    print_evaluation_summary — formatted text summary of metrics.
"""

from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_predictions(
    audio: np.ndarray,
    sr: int,
    annotations: np.ndarray,
    predictions: np.ndarray,
    title: str = "Beat Predictions",
    max_duration: float = 20.0,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Plot spectrogram with beat annotations and predictions overlaid.

    Args:
        audio: Audio signal
        sr: Sample rate
        annotations: Annotated beat times
        predictions: Predicted beat times
        title: Plot title
        max_duration: Maximum duration to display (seconds)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    hop_length = 512

    # Truncate to max duration
    max_samples = int(max_duration * sr)
    audio_plot = audio[:max_samples]
    ann_plot = annotations[annotations <= max_duration]
    pred_plot = predictions[predictions <= max_duration]

    # Compute spectrogram
    spec = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_plot, hop_length=hop_length)),
        ref=np.max
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    img = librosa.display.specshow(
        spec,
        y_axis='log',
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        ax=ax
    )

    # Plot annotations (top half)
    ax.vlines(
        ann_plot,
        hop_length * 2,
        sr / 2,
        linestyles='dotted',
        colors='white',
        label='Annotations'
    )
    ax.text(
        0.5, hop_length * 1.65,
        'Annotations (above)',
        color='white',
        fontsize=10
    )

    # Plot predictions (bottom half)
    ax.vlines(
        pred_plot,
        0,
        hop_length,
        linestyles='solid',
        colors='cyan',
        label='Predictions'
    )
    ax.text(
        0.5, hop_length * 1.1,
        'Predictions (below)',
        color='cyan',
        fontsize=10
    )

    ax.set_title(title)
    plt.colorbar(img, ax=ax, format="%+2.f dB")

    return fig


def plot_activations(
    activations: np.ndarray,
    annotations: np.ndarray,
    fps: int = 100,
    title: str = "Beat Activations",
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Plot beat activations with annotations.

    Args:
        activations: Beat activation function (per-frame probabilities)
        annotations: Beat times in seconds
        fps: Frames per second
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    time = np.arange(len(activations)) / fps
    ax.plot(time, activations, label='Activations', color='blue', alpha=0.7)

    # Plot annotations as vertical lines
    for ann in annotations:
        if ann < time[-1]:
            ax.axvline(ann, color='red', linestyle='--', alpha=0.5)

    ax.axhline(0.5, color='green', linestyle=':', alpha=0.5, label='Threshold')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activation')
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)

    return fig


def print_evaluation_summary(
    metrics: Dict[str, float],
    name: str = "Evaluation"
) -> None:
    """
    Print formatted evaluation summary.

    Args:
        metrics: Dictionary of metric values
        name: Name for the evaluation
    """
    print(f"\n{'='*50}")
    print(f" {name} Results")
    print(f"{'='*50}")

    if "fmeasure_mean" in metrics:
        print(f" F-measure: {metrics['fmeasure_mean']:.4f} +/- {metrics['fmeasure_std']:.4f}")

    if "cemgil_mean" in metrics:
        print(f" Cemgil:    {metrics['cemgil_mean']:.4f} +/- {metrics['cemgil_std']:.4f}")

    if "pscore_mean" in metrics:
        print(f" P-score:   {metrics['pscore_mean']:.4f} +/- {metrics['pscore_std']:.4f}")

    if "acc1" in metrics:
        print(f" Acc1:      {metrics['acc1']:.4f}")
        print(f" Acc2:      {metrics['acc2']:.4f}")

    if "num_tracks" in metrics:
        print(f" Tracks:    {metrics['num_tracks']}")

    print(f"{'='*50}\n")
