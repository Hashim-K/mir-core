"""
Genre classification: GenreClassifier factory and GenreRouter for activation routing.

Provides a unified interface for all classifier architectures, plus a
GenreRouter that combines activations from genre-adapted beat trackers.
"""

from collections.abc import Mapping
from copy import deepcopy
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import CLASSIFIER_ARCHITECTURES

GENRE_LABELS = ["candombe", "brid", "salsa", "other"]


# ---------------------------------------------------------------------------
# GenreClassifier -- unified factory + inference interface
# ---------------------------------------------------------------------------


class GenreClassifier(nn.Module):
    """Factory wrapper providing a unified interface for all classifier architectures.

    Args:
        arch: Architecture name (mel_cnn, mfcc_cnn, mel_cnn_attention,
            beatnet_log_spect_cnn, embedding_stats_mlp, beatnet_conv).
        num_classes: Number of genre classes.
        genre_labels: List of genre label strings. Defaults to GENRE_LABELS.
        calibration_temperature: Positive temperature applied by :meth:`predict`.
            :meth:`forward` always returns unscaled logits.
        **kwargs: Forwarded to the underlying architecture constructor.
    """

    def __init__(
        self,
        arch: str = "mel_cnn",
        num_classes: int = 4,
        genre_labels: Optional[List[str]] = None,
        calibration_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if arch not in CLASSIFIER_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Choose from: {list(CLASSIFIER_ARCHITECTURES.keys())}"
            )
        if isinstance(num_classes, bool) or int(num_classes) != num_classes:
            raise ValueError("num_classes must be a positive integer.")
        num_classes = int(num_classes)
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        labels = (
            GENRE_LABELS[:num_classes] if genre_labels is None else list(genre_labels)
        )
        if len(labels) != num_classes:
            raise ValueError(
                "genre_labels must contain exactly num_classes entries "
                f"({len(labels)} != {num_classes})."
            )
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("genre_labels must contain non-empty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("genre_labels must be unique.")
        self.arch_name = arch
        self.genre_labels = labels
        self._calibration_temperature = 1.0
        self.calibration_temperature = calibration_temperature
        self._calibration_metadata: dict[str, object] = {}
        self._router_config: dict[str, object] = {}
        self.model = CLASSIFIER_ARCHITECTURES[arch](num_classes=num_classes, **kwargs)

    @property
    def calibration_temperature(self) -> float:
        """Positive temperature used by :meth:`predict`."""

        return self._calibration_temperature

    @calibration_temperature.setter
    def calibration_temperature(self, value: float) -> None:
        if isinstance(value, bool):
            raise ValueError(
                "calibration_temperature must be a positive finite scalar."
            )
        try:
            temperature = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "calibration_temperature must be a positive finite scalar."
            ) from exc
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "calibration_temperature must be a positive finite scalar."
            )
        self._calibration_temperature = temperature

    @property
    def calibration_metadata(self) -> dict[str, object]:
        """Return a defensive copy of checkpoint calibration provenance."""

        return deepcopy(self._calibration_metadata)

    @property
    def router_config(self) -> dict[str, object]:
        """Return a defensive copy of the checkpoint router configuration."""

        return deepcopy(self._router_config)

    def set_routing_metadata(
        self,
        *,
        calibration: Mapping[str, object] | None = None,
        router_config: Mapping[str, object] | None = None,
    ) -> None:
        """Attach immutable-by-copy deployment metadata loaded with a checkpoint."""

        if calibration is not None and not isinstance(calibration, Mapping):
            raise ValueError("calibration metadata must be a mapping.")
        if router_config is not None and not isinstance(router_config, Mapping):
            raise ValueError("router_config metadata must be a mapping.")
        self._calibration_metadata = deepcopy(dict(calibration or {}))
        self._router_config = deepcopy(dict(router_config or {}))

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
                calibration_temperature: float -- temperature used for probabilities
        """
        self.eval()
        logits = self.forward(x)
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError(
                "Classifier output must have shape (batch, num_classes) "
                "with a non-empty batch."
            )
        if logits.shape[1] != len(self.genre_labels):
            raise ValueError(
                "Classifier output class count does not match genre_labels "
                f"({logits.shape[1]} != {len(self.genre_labels)})."
            )
        probs = F.softmax(
            logits / self.calibration_temperature,
            dim=-1,
        )  # (batch, num_classes)
        # Take first item in batch
        probs_np = probs[0].cpu().numpy()
        top_idx = int(probs_np.argmax())
        return {
            "genre": self.genre_labels[top_idx],
            "probabilities": {
                label: float(p) for label, p in zip(self.genre_labels, probs_np)
            },
            "confidence": float(probs_np[top_idx]),
            "calibration_temperature": self.calibration_temperature,
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
            y=audio,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
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

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, time)
        tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
        return tensor

    @staticmethod
    def preprocess_beatnet_log_spect(
        audio: np.ndarray,
        sr: int = 22050,
        duration: float = 3.0,
    ) -> torch.Tensor:
        """Convert raw audio to BeatNet LOG_SPECT tensor for router models.

        Returns:
            Tensor of shape ``(1, 1, 272, time)`` for
            ``beatnet_log_spect_cnn``.
        """
        from mir_core.preprocessing import BeatNetPreProcessor

        target_len = int(sr * duration)
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))

        preprocessor = BeatNetPreProcessor(sample_rate=sr)
        features = preprocessor(audio, sr=sr)
        features = np.asarray(features, dtype=np.float32).T
        return torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)


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
        ema_alpha: EMA update factor (0 = retain state, 1 = no memory).
        confidence_threshold: Below this, fall back to "other" / baseline.
        blend_threshold: For hybrid mode, hard-route above this confidence.
        fallback_label: Baseline activation used below the confidence threshold.
    """

    def __init__(
        self,
        genre_labels: Optional[List[str]] = None,
        strategy: str = "hybrid",
        ema_alpha: float = 0.3,
        confidence_threshold: float = 0.7,
        blend_threshold: float = 0.8,
        fallback_label: Optional[str] = None,
    ):
        labels = list(GENRE_LABELS if genre_labels is None else genre_labels)
        if not labels:
            raise ValueError("genre_labels must not be empty.")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("genre_labels must contain non-empty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("genre_labels must be unique.")
        if strategy not in {"hard", "soft", "hybrid"}:
            raise ValueError("strategy must be 'hard', 'soft', or 'hybrid'.")
        if strategy == "hybrid" and len(labels) < 2:
            raise ValueError("hybrid routing requires at least two genre labels.")

        alpha = self._unit_interval(ema_alpha, name="ema_alpha")
        confidence = self._unit_interval(
            confidence_threshold,
            name="confidence_threshold",
        )
        blend = self._unit_interval(blend_threshold, name="blend_threshold")
        if blend < confidence:
            raise ValueError(
                "blend_threshold must be greater than or equal to "
                "confidence_threshold."
            )
        fallback = (
            ("other" if "other" in labels else labels[-1])
            if fallback_label is None
            else fallback_label
        )
        if fallback not in labels:
            raise ValueError("fallback_label must be present in genre_labels.")

        self.genre_labels = labels
        self.strategy = strategy
        self.ema_alpha = alpha
        self.confidence_threshold = confidence
        self.blend_threshold = blend
        self.fallback_label = fallback

        # Initialise smoothed probabilities to uniform
        n = len(self.genre_labels)
        self._smoothed_probs = np.full(n, 1.0 / n, dtype=np.float64)

    @staticmethod
    def _unit_interval(value: float, *, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and between 0 and 1.") from exc
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1.")
        return number

    @property
    def smoothed_probs(self) -> np.ndarray:
        """Return a defensive copy of the current routing probabilities."""

        return self._smoothed_probs.copy()

    def update_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Update smoothed genre probabilities with new classifier output.

        Args:
            raw_probs: Softmax probabilities from classifier, shape (num_classes,).

        Returns:
            EMA-smoothed probabilities.
        """
        if isinstance(raw_probs, torch.Tensor):
            raw_probs = raw_probs.detach().cpu().numpy()
        try:
            probabilities = np.asarray(raw_probs, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw_probs must be a numeric probability vector.") from exc
        expected_shape = (len(self.genre_labels),)
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"raw_probs must have shape {expected_shape}, got {probabilities.shape}."
            )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("raw_probs must contain finite, non-negative values.")
        total = float(probabilities.sum())
        if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
            raise ValueError("raw_probs must sum to 1.")
        probabilities = probabilities / total
        self._smoothed_probs = (
            self.ema_alpha * probabilities
            + (1.0 - self.ema_alpha) * self._smoothed_probs
        )
        return self._smoothed_probs.copy()

    def _validated_activations(
        self,
        activations: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not isinstance(activations, Mapping):
            raise ValueError("activations must be a mapping keyed by genre label.")
        missing = [label for label in self.genre_labels if label not in activations]
        if missing:
            raise ValueError(f"activations are missing genre labels: {missing}.")

        arrays: dict[str, np.ndarray] = {}
        expected_shape: tuple[int, ...] | None = None
        for label in self.genre_labels:
            try:
                array = np.asarray(activations[label])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Activation for {label!r} must be array-like."
                ) from exc
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"Activation for {label!r} must be numeric.")
            if not np.isfinite(array).all():
                raise ValueError(f"Activation for {label!r} must be finite.")
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError(
                    "All activation arrays must have the same shape "
                    f"({array.shape} != {expected_shape} for {label!r})."
                )
            arrays[label] = array
        return arrays

    def route(self, activations: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-genre activations according to the current strategy.

        Args:
            activations: Mapping from genre label to activation array.
                         All arrays must have the same shape.

        Returns:
            Combined activation array (same shape as individual activations).
        """
        arrays = self._validated_activations(activations)
        probs = self._smoothed_probs
        top_idx = int(np.argmax(probs))
        top_genre = self.genre_labels[top_idx]
        top_conf = probs[top_idx]

        # Below confidence threshold -> use baseline ("other")
        if top_conf < self.confidence_threshold:
            return np.array(arrays[self.fallback_label], copy=True)

        if self.strategy == "hard":
            return np.array(arrays[top_genre], copy=True)

        elif self.strategy == "soft":
            dtype = np.result_type(
                np.float64,
                *(array.dtype for array in arrays.values()),
            )
            combined = np.zeros_like(next(iter(arrays.values())), dtype=dtype)
            for i, genre in enumerate(self.genre_labels):
                combined = combined + probs[i] * arrays[genre]
            return combined

        elif self.strategy == "hybrid":
            if top_conf >= self.blend_threshold:
                return np.array(arrays[top_genre], copy=True)
            # Blend top-2
            sorted_idx = np.argsort(-probs, kind="stable")
            g1 = self.genre_labels[sorted_idx[0]]
            g2 = self.genre_labels[sorted_idx[1]]
            p1, p2 = probs[sorted_idx[0]], probs[sorted_idx[1]]
            total = p1 + p2
            return (p1 / total) * arrays[g1] + (p2 / total) * arrays[g2]

    def get_status(self) -> str:
        """Human-readable status string for UI display."""
        probs = self._smoothed_probs
        top_idx = int(np.argmax(probs))
        parts = [
            f"{g.capitalize()} ({p:.0%})" for g, p in zip(self.genre_labels, probs)
        ]
        prefix = f"Auto [{self.strategy}]"
        return f"{prefix}: {' | '.join(parts)}"

    def reset(self):
        """Reset smoothed probabilities to uniform."""
        n = len(self.genre_labels)
        self._smoothed_probs = np.full(n, 1.0 / n, dtype=np.float64)
