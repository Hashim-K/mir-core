"""
BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer.

From Chang & Su (ICASSP 2024). Uses a streaming Transformer with contextual
block processing for online beat and downbeat tracking with low latency (<50ms).

Architecture:
1. 3-layer CNN frontend (32->64->256 channels) with max pooling
2. Contextual Block Streaming Transformer encoder (9 layers)
3. Beat/downbeat output head (2 outputs: beat, downbeat)
4. Optional tempo head (300 tempo bins)

Reference:
Chang & Su (2024). "BEAST: Online Joint Beat and Downbeat Tracking
Based on Streaming Transformer." ICASSP 2024.
https://arxiv.org/abs/2312.17156
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextualBlockTransformerEncoder


class BEAST(nn.Module):
    """
    BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer.

    From Chang & Su (ICASSP 2024). Uses a streaming Transformer with contextual
    block processing for online beat and downbeat tracking with low latency (<50ms).

    Key Features:
    - Relative positional encoding for streaming scenarios
    - Contextual block processing for online inference
    - Joint beat and downbeat tracking
    - F1=80.04% beat, 46.78% downbeat on benchmark (online)

    Args:
        n_mels: Number of mel frequency bands (default 128)
        d_model: Transformer model dimension (default 256)
        n_head: Number of attention heads (default 8)
        d_ff: Feed-forward hidden dimension (default 1024)
        n_layers: Number of Transformer layers (default 9)
        dropout: Dropout rate (default 0.1)
        left_size: Left context frames (default 256)
        center_size: Center block frames (default 16)
        right_size: Look-ahead frames (default 16)
        include_tempo: Include tempo prediction head (default False)
        n_tempo_bins: Number of tempo bins for tempo head (default 300)
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_head: int = 8,
        d_ff: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.1,
        left_size: int = 256,
        center_size: int = 16,
        right_size: int = 16,
        include_tempo: bool = False,
        n_tempo_bins: int = 300,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.d_model = d_model
        self.include_tempo = include_tempo

        # CNN Frontend (3 conv blocks with max pooling)
        # Input: (batch, 1, time, n_mels)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 12), stride=1, padding=(0, 0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(64, d_model, kernel_size=(2, 6), stride=1, padding=(1, 0))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(dropout)

        # Streaming Transformer Encoder
        self.encoder = ContextualBlockTransformerEncoder(
            input_size=d_model,
            output_size=d_model,
            attention_heads=n_head,
            linear_units=d_ff,
            num_blocks=n_layers,
            dropout_rate=dropout,
            left_size=left_size,
            center_size=center_size,
            right_size=right_size,
        )

        # Beat/Downbeat output head (2 outputs: beat prob, downbeat prob)
        self.out_linear = nn.Linear(d_model, 2)

        # Optional tempo head
        if include_tempo:
            self.tempo_dropout = nn.Dropout(0.5)
            self.tempo_linear = nn.Linear(d_model, n_tempo_bins)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input mel spectrogram of shape (batch, time, n_mels)
               OR (batch, 1, time, n_mels) for 4D input

        Returns:
            Dictionary with 'beats', 'downbeats', and optionally 'tempo'
        """
        # Handle 3D input (batch, time, freq)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, time, freq)

        # CNN Frontend
        x = self.conv1(x)
        x = x[:, :, :-2, :]  # Trim padding
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = x[:, :, :-1, :]  # Trim padding
        x = self.maxpool3(x)
        x = F.relu(x)
        x = self.dropout3(x)  # (batch, d_model, time, 1)

        # Reshape for Transformer: (batch, time, d_model)
        x = x.squeeze(-1).transpose(1, 2).contiguous()

        # Transformer Encoder
        x = self.encoder(x)

        # Beat/Downbeat head
        beat_out = self.out_linear(F.relu(x))  # (batch, time, 2)

        # Apply sigmoid for probabilities
        beat_probs = torch.sigmoid(beat_out[:, :, 0])
        downbeat_probs = torch.sigmoid(beat_out[:, :, 1])

        outputs = {
            "beats": beat_probs.unsqueeze(-1),
            "downbeats": downbeat_probs.unsqueeze(-1),
        }

        # Optional tempo head
        if self.include_tempo:
            # Global average pooling over time
            tempo_feat = x.mean(dim=1)  # (batch, d_model)
            tempo_feat = self.tempo_dropout(F.relu(tempo_feat))
            tempo_logits = self.tempo_linear(tempo_feat)  # (batch, n_tempo_bins)
            outputs["tempo"] = F.softmax(tempo_logits, dim=-1)

        return outputs

    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field in input frames.

        Returns:
            Number of input frames the model can "see"
        """
        return self.encoder.block_size


class BEASTBatch(nn.Module):
    """
    Batch-optimized BEAST for training without streaming constraints.

    Same architecture as BEAST but processes full sequences without
    block-wise streaming, making it more efficient for batch training.
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        n_head: int = 8,
        d_ff: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.1,
        include_tempo: bool = False,
        n_tempo_bins: int = 300,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.d_model = d_model
        self.include_tempo = include_tempo

        # CNN Frontend
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 12), stride=1, padding=(0, 0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(64, d_model, kernel_size=(2, 6), stride=1, padding=(1, 0))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(dropout)

        # Standard Transformer Encoder (no streaming)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        # Output heads
        self.out_linear = nn.Linear(d_model, 2)

        if include_tempo:
            self.tempo_dropout = nn.Dropout(0.5)
            self.tempo_linear = nn.Linear(d_model, n_tempo_bins)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for batch processing."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN Frontend
        x = self.conv1(x)
        x = x[:, :, :-2, :]
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = x[:, :, :-1, :]
        x = self.maxpool3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Reshape for Transformer
        x = x.squeeze(-1).transpose(1, 2).contiguous()

        # Transformer Encoder
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Output
        beat_out = self.out_linear(F.relu(x))
        beat_probs = torch.sigmoid(beat_out[:, :, 0])
        downbeat_probs = torch.sigmoid(beat_out[:, :, 1])

        outputs = {
            "beats": beat_probs.unsqueeze(-1),
            "downbeats": downbeat_probs.unsqueeze(-1),
        }

        if self.include_tempo:
            tempo_feat = x.mean(dim=1)
            tempo_feat = self.tempo_dropout(F.relu(tempo_feat))
            outputs["tempo"] = F.softmax(self.tempo_linear(tempo_feat), dim=-1)

        return outputs
