"""
Classifier architecture implementations for genre classification.

Architectures (all share the same interface):
    MelCNN              -- 3-layer CNN on 128-bin log-mel (~50K params). Primary.
    MFCCCNN             -- CNN on 60-dim MFCCs + deltas (~25K params).
    MelCNNAttention     -- MelCNN + SE channel attention (~55K params).
    BeatNetLogSpectCNN    -- CNN on BeatNet LOG_SPECT features.
    EmbeddingStatsMLP   -- MLP over mean/std-pooled pretrained embeddings.
    FramewiseEmbeddingMLP -- shared MLP per embedding frame, then mean logits.
    BeatNetConvClassifier -- Tiny head on BeatNet conv1 features (~5K params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _initialize_linear_layers(module: nn.Module, initialization: str) -> None:
    """Apply an explicit, checkpointed initialization to every linear layer."""

    normalized = str(initialization).strip().lower()
    initializers = {
        "pytorch_default": None,
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
    }
    if normalized not in initializers:
        raise ValueError(
            "initialization must be pytorch_default, xavier_uniform, or "
            "xavier_normal."
        )
    initializer = initializers[normalized]
    if initializer is None:
        return
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            initializer(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class MelCNN(nn.Module):
    """3-layer CNN on 128-bin log-mel spectrograms. ~50K params.

    Input shape: (batch, 1, n_mels, time) -- e.g. (B, 1, 128, ~130) for 3s.
    Output: logits of shape (batch, num_classes).
    """

    def __init__(self, num_classes: int = 4, n_mels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class MFCCCNN(nn.Module):
    """CNN on MFCCs + deltas (60-dim input). ~25K params.

    Input shape: (batch, 1, n_features, time) -- e.g. (B, 1, 60, ~130).
    Output: logits of shape (batch, num_classes).
    """

    def __init__(self, num_classes: int = 4, n_features: int = 60, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        scale = x.mean(dim=(2, 3))  # (B, C)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class MelCNNAttention(nn.Module):
    """MelCNN + Squeeze-and-Excitation channel attention. ~55K params.

    Adds an SE block after the last conv layer to help the model focus
    on frequency channels most relevant for genre discrimination.

    Input shape: (batch, 1, n_mels, time).
    Output: logits of shape (batch, num_classes).
    """

    def __init__(self, num_classes: int = 4, n_mels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.attention = _SEBlock(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class BeatNetLogSpectCNN(nn.Module):
    """Small CNN on BeatNet LOG_SPECT features.

    This classifier is intended for the realtime router path where the beat
    tracker already computes BeatNet's 272-dimensional LOG_SPECT frontend. The
    input shape is ``(batch, 1, feature_dim, time)``, where ``feature_dim`` is
    272 for BeatNet and 288 for BeatNet+.
    """

    def __init__(
        self,
        num_classes: int = 4,
        feature_dim: int = 272,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class EmbeddingStatsMLP(nn.Module):
    """Classifier head for frame-level pretrained audio embeddings.

    The input shape is ``(batch, 1, embedding_dim, time)``. The head computes
    per-track mean and standard deviation over time and classifies the resulting
    ``2 * embedding_dim`` summary. This is suitable for frozen YAMNet,
    EfficientAT, or similar embedding extractors.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        hidden_layers: int = 1,
        batch_norm: bool = False,
        initialization: str = "pytorch_default",
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        if isinstance(hidden_layers, bool) or int(hidden_layers) != hidden_layers:
            raise ValueError("hidden_layers must be a positive integer.")
        hidden_layers = int(hidden_layers)
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be a positive integer.")
        layers: list[nn.Module] = []
        input_dim = self.embedding_dim * 2
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if batch_norm:
                # Match the HEAR downstream head: normalization precedes
                # dropout and ReLU for every hidden layer.
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.extend((nn.Dropout(dropout), nn.ReLU(inplace=True)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        _initialize_linear_layers(self.classifier, initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, 1, F, T), got {tuple(x.shape)}.")
        x = x.squeeze(1).transpose(1, 2)  # (B, T, F)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return self.classifier(torch.cat([mean, std], dim=-1))


class FramewiseEmbeddingMLP(nn.Module):
    """Frozen-embedding transfer head with causal mean logit aggregation.

    TensorFlow's official YAMNet transfer-learning example applies the same
    shallow classifier to every 1024-dimensional YAMNet frame and averages its
    outputs for a clip decision.  Applying that operation to a right-aligned
    prefix keeps the identical head usable for every validated live context
    horizon.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        initialization: str = "pytorch_default",
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        layers: list[nn.Module] = [
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        _initialize_linear_layers(self.classifier, initialization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, 1, F, T), got {tuple(x.shape)}.")
        frames = x.squeeze(1).transpose(1, 2)  # (B, T, F)
        if frames.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {frames.shape[-1]}."
            )
        return self.classifier(frames).mean(dim=1)


class BeatNetConvClassifier(nn.Module):
    """Tiny genre classifier on BeatNet conv1 output features. ~5K params.

    Reuses the shared BeatNet conv frontend output as input, so the
    convolutional computation is free when BeatNet is already running.
    Couples the classifier to the BeatNet architecture.

    Input shape: (batch, time, conv_features) -- output of BeatNet conv1+pool+linear0.
    Output: logits of shape (batch, num_classes).
    """

    def __init__(self, num_classes: int = 4, input_dim: int = 150, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -- average over time
        x = x.mean(dim=1)  # (batch, features)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

CLASSIFIER_ARCHITECTURES = {
    "mel_cnn": MelCNN,
    "mfcc_cnn": MFCCCNN,
    "mel_cnn_attention": MelCNNAttention,
    "beatnet_log_spect_cnn": BeatNetLogSpectCNN,
    "embedding_stats_mlp": EmbeddingStatsMLP,
    "framewise_embedding_mlp": FramewiseEmbeddingMLP,
    "beatnet_conv": BeatNetConvClassifier,
}
