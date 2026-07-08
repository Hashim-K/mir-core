"""
Classifier architecture implementations for genre classification.

Architectures (all share the same interface):
    MelCNN              -- 3-layer CNN on 128-bin log-mel (~50K params). Primary.
    MFCCCNN             -- CNN on 60-dim MFCCs + deltas (~25K params).
    MelCNNAttention     -- MelCNN + SE channel attention (~55K params).
    BeatNetLogSpectCNN    -- CNN on BeatNet LOG_SPECT features.
    EmbeddingStatsMLP   -- MLP over mean/std-pooled pretrained embeddings.
    BeatNetConvClassifier -- Tiny head on BeatNet conv1 features (~5K params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, 1, F, T), got {tuple(x.shape)}.")
        x = x.squeeze(1).transpose(1, 2)  # (B, T, F)
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        return self.classifier(torch.cat([mean, std], dim=-1))


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
    "beatnet_conv": BeatNetConvClassifier,
}
