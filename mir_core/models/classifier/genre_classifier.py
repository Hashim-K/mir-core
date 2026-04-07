"""
Genre classification: GenreClassifier factory and GenreRouter for activation routing.

Provides a unified interface for all classifier architectures, plus a
GenreRouter that combines activations from genre-adapted beat trackers.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import CLASSIFIER_ARCHITECTURES


GENRE_LABELS = ["candombe", "samba", "salsa", "other"]


# ---------------------------------------------------------------------------
# GenreClassifier -- unified factory + inference interface
# ---------------------------------------------------------------------------

class GenreClassifier(nn.Module):
    """Factory wrapper providing a unified interface for all classifier architectures.

    Args:
        arch: Architecture name (mel_cnn, mfcc_cnn, mel_cnn_attention, beatnet_conv).
        num_classes: Number of genre classes.
        genre_labels: List of genre label strings. Defaults to GENRE_LABELS.
        **kwargs: Forwarded to the underlying architecture constructor.
    """

    def __init__(
        self,
        arch: str = "mel_cnn",
        num_classes: int = 4,
        genre_labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        if arch not in CLASSIFIER_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Choose from: {list(CLASSIFIER_ARCHITECTURES.keys())}"
            )
        self.arch_name = arch
        self.genre_labels = genre_labels or GENRE_LABELS[:num_classes]
        self.model = CLASSIFIER_ARCHITECTURES[arch](num_classes=num_classes, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits. Shape: (batch, num_classes)."""
        return self.model(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, object]:
        """Run inference and return genre prediction with probabilities.

        Returns:
            dict with keys:
                genre: str -- predicted genre label
                probabilities: dict[str, float] -- per-genre softmax probabilities
                confidence: float -- max probability
        """
        self.eval()
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)  # (batch, num_classes)
        # Take first item in batch
        probs_np = probs[0].cpu().numpy()
        top_idx = int(probs_np.argmax())
        return {
            "genre": self.genre_labels[top_idx],
            "probabilities": {
                label: float(p) for label, p in zip(self.genre_labels, probs_np)
            },
            "confidence": float(probs_np[top_idx]),
        }

    @staticmethod
    def preprocess_audio(
        audio: np.ndarray,
        sr: int = 22050,
        duration: float = 3.0,
        n_mels: int = 128,
        hop_length: int = 512,
    ) -> torch.Tensor:
        """Convert raw audio numpy array to a log-mel tensor for MelCNN variants.

        Args:
            audio: 1-D audio waveform.
            sr: Sample rate of *audio*.
            duration: Target duration in seconds (truncate or pad).
            n_mels: Number of mel bands.
            hop_length: STFT hop length.

        Returns:
            Tensor of shape (1, 1, n_mels, time) ready for forward().
        """
        import librosa

        target_len = int(sr * duration)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length,
        )
        log_mel = np.log(mel + 1e-6)
        tensor = torch.from_numpy(log_mel).float().unsqueeze(0).unsqueeze(0)
        return tensor  # (1, 1, n_mels, time)

    @staticmethod
    def preprocess_mfcc(
        audio: np.ndarray,
        sr: int = 22050,
        duration: float = 3.0,
        n_mfcc: int = 20,
        hop_length: int = 512,
    ) -> torch.Tensor:
        """Convert raw audio to MFCC + delta + delta-delta tensor for MFCCCNN.

        Returns:
            Tensor of shape (1, 1, n_mfcc*3, time).
        """
        import librosa

        target_len = int(sr * duration)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, time)
        tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
        return tensor


# ---------------------------------------------------------------------------
# GenreRouter -- combines activations from multiple genre-adapted models
# ---------------------------------------------------------------------------

class GenreRouter:
    """Combines beat-tracker activations from genre-adapted models.

    Maintains EMA-smoothed genre probabilities and applies one of three
    routing strategies to produce a single combined activation output.

    Args:
        genre_labels: Genre label strings (must match activation dict keys).
        strategy: Routing strategy -- "hard", "soft", or "hybrid".
        ema_alpha: EMA smoothing factor (0 = no smoothing, 1 = no memory).
        confidence_threshold: Below this, fall back to "other" / baseline.
        blend_threshold: For hybrid mode, hard-route above this confidence.
    """

    def __init__(
        self,
        genre_labels: Optional[List[str]] = None,
        strategy: str = "hybrid",
        ema_alpha: float = 0.3,
        confidence_threshold: float = 0.7,
        blend_threshold: float = 0.8,
    ):
        self.genre_labels = genre_labels or GENRE_LABELS
        self.strategy = strategy
        self.ema_alpha = ema_alpha
        self.confidence_threshold = confidence_threshold
        self.blend_threshold = blend_threshold

        # Initialise smoothed probabilities to uniform
        n = len(self.genre_labels)
        self._smoothed_probs = np.ones(n) / n

    def update_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Update smoothed genre probabilities with new classifier output.

        Args:
            raw_probs: Softmax probabilities from classifier, shape (num_classes,).

        Returns:
            EMA-smoothed probabilities.
        """
        self._smoothed_probs = (
            self.ema_alpha * raw_probs
            + (1.0 - self.ema_alpha) * self._smoothed_probs
        )
        return self._smoothed_probs.copy()

    def route(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-genre activations according to the current strategy.

        Args:
            activations: Mapping from genre label to activation array.
                         All arrays must have the same shape.

        Returns:
            Combined activation array (same shape as individual activations).
        """
        probs = self._smoothed_probs
        top_idx = int(np.argmax(probs))
        top_genre = self.genre_labels[top_idx]
        top_conf = probs[top_idx]

        # Below confidence threshold -> use baseline ("other")
        if top_conf < self.confidence_threshold:
            return activations.get("other", activations[self.genre_labels[-1]])

        if self.strategy == "hard":
            return activations[top_genre]

        elif self.strategy == "soft":
            combined = np.zeros_like(next(iter(activations.values())))
            for i, genre in enumerate(self.genre_labels):
                if genre in activations:
                    combined = combined + probs[i] * activations[genre]
            return combined

        elif self.strategy == "hybrid":
            if top_conf >= self.blend_threshold:
                return activations[top_genre]
            # Blend top-2
            sorted_idx = np.argsort(probs)[::-1]
            g1 = self.genre_labels[sorted_idx[0]]
            g2 = self.genre_labels[sorted_idx[1]]
            p1, p2 = probs[sorted_idx[0]], probs[sorted_idx[1]]
            total = p1 + p2
            return (p1 / total) * activations[g1] + (p2 / total) * activations[g2]

        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

    def get_status(self) -> str:
        """Human-readable status string for UI display."""
        probs = self._smoothed_probs
        top_idx = int(np.argmax(probs))
        parts = [
            f"{g.capitalize()} ({p:.0%})"
            for g, p in zip(self.genre_labels, probs)
        ]
        prefix = f"Auto [{self.strategy}]"
        return f"{prefix}: {' | '.join(parts)}"

    def reset(self):
        """Reset smoothed probabilities to uniform."""
        n = len(self.genre_labels)
        self._smoothed_probs = np.ones(n) / n
