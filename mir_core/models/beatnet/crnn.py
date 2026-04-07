"""
BeatNet: CRNN model for online joint beat and downbeat tracking.

Based on Heydari et al. "BeatNet: CRNN and Particle Filtering
for Online Joint Beat Downbeat and Meter Tracking" (ISMIR 2021)

Architecture:
1. Conv1D: 2 filters, kernel_size=10, along frequency dimension
2. MaxPool1D: kernel_size=2
3. Linear projection to hidden_dim (150)
4. 2-layer unidirectional LSTM (150 hidden units)
5. Linear output layer (3 classes: non-beat, beat, downbeat)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeatNetCRNN(nn.Module):
    """
    Official BeatNet CRNN architecture (BDA - Beat Downbeat Activation).

    Based on Heydari et al. "BeatNet: CRNN and Particle Filtering
    for Online Joint Beat Downbeat and Meter Tracking" (ISMIR 2021)

    This is the exact architecture from the official implementation:
    https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/model.py

    Input features:
    - 272-dim = 136 filterbank bins (24 bands x ~5.7) + 136 first-order difference
    - Sample rate: 22050 Hz
    - Hop: 441 samples (20ms) -> 50 FPS
    - Window: 1408 samples (64ms)

    Args:
        input_dim: Input feature dimension (default 272 for official BeatNet)
        hidden_dim: LSTM hidden dimension (default 150)
        num_layers: Number of LSTM layers (default 2)
        device: torch device for hidden state initialization
    """

    def __init__(
        self,
        input_dim: int = 272,
        hidden_dim: int = 150,
        num_layers: int = 2,
        device: str = 'cpu',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device_str = device
        self.conv_out = 150  # Projection dimension after conv
        self.kernel_size = 10

        # Convolutional layer: 1D conv along frequency dimension
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=self.kernel_size,
            padding=0
        )

        # Calculate output size after conv and maxpool
        # After conv: (input_dim - kernel_size + 1) = input_dim - 9
        # After maxpool (kernel=2): (input_dim - 9) // 2
        # Total: 2 * ((input_dim - 9) // 2)
        conv_out_dim = 2 * int((input_dim - self.kernel_size + 1) / 2)

        # Linear projection to LSTM input dimension
        self.linear0 = nn.Linear(conv_out_dim, self.conv_out)

        # LSTM layers (unidirectional for online/causal processing)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Output layer: 3 classes (non-beat, beat, downbeat)
        self.linear = nn.Linear(hidden_dim, 3)
        self.softmax = nn.Softmax(dim=0)  # Per official implementation

        # Register hidden and cell states as buffers so they move with the model
        self.register_buffer('hidden', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('cell', torch.zeros(num_layers, 1, hidden_dim))

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset LSTM hidden and cell states (call between sequences)."""
        if device is None:
            device = next(self.parameters()).device
        self.hidden = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        self.cell = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching official BeatNet implementation.

        This matches the exact forward pass from:
        https://github.com/mjhydri/BeatNet/blob/main/src/BeatNet/model.py

        Args:
            data: Input tensor of shape (batch, time, freq) where freq=272
                  (136 filterbank + 136 spectral difference)

        Returns:
            Output logits of shape (batch, 3, time)
        """
        batch_size = data.shape[0]

        # Ensure hidden states match batch size and device
        if self.hidden.shape[1] != batch_size:
            self.reset_hidden(batch_size, data.device)

        x = data
        # Reshape to (batch * time, freq)
        x = torch.reshape(x, (-1, self.input_dim))
        # Add channel dimension: (batch * time, 1, freq)
        x = x.unsqueeze(0).transpose(0, 1)

        # Conv + ReLU + MaxPool
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)

        # Flatten: (batch * time, conv_features)
        x = x.view(-1, self._num_flat_features(x))

        # Linear projection
        x = self.linear0(x)

        # Reshape back to sequence: (batch, time, conv_out)
        x = torch.reshape(x, (data.shape[0], data.shape[1], self.conv_out))

        # LSTM with hidden state persistence (detach to avoid backprop through hidden state history)
        hidden = self.hidden.detach()
        cell = self.cell.detach()
        x, (self.hidden, self.cell) = self.lstm(x, (hidden, cell))

        # Output projection
        out = self.linear(x)
        out = out.transpose(1, 2)  # (batch, 3, time)

        return out

    def final_pred(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply softmax to get final predictions (per official implementation)."""
        return self.softmax(logits)

    def _num_flat_features(self, x: torch.Tensor) -> int:
        """Calculate flattened feature size."""
        size = x.size()[1:]  # All dimensions except batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_beat_downbeat_activations(
        self, data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get beat and downbeat activation probabilities.

        This is a convenience wrapper that runs forward + softmax
        and returns activations in the same format as BockTCN.

        Args:
            data: Input features of shape (batch, time, 272)

        Returns:
            Dictionary with 'beats', 'downbeats', 'activations' tensors
        """
        # Forward pass
        logits = self.forward(data)  # (batch, 3, time)

        # Apply softmax along class dimension
        probs = F.softmax(logits, dim=1)  # (batch, 3, time)

        # Transpose to (batch, time, 3)
        probs = probs.transpose(1, 2)

        # Class 0: non-beat, Class 1: beat, Class 2: downbeat
        beats = probs[:, :, 1] + probs[:, :, 2]  # Beat = beat + downbeat
        downbeats = probs[:, :, 2]

        return {
            "beats": beats.unsqueeze(-1),
            "downbeats": downbeats.unsqueeze(-1),
            "activations": probs,
        }


class BeatNetBatch(nn.Module):
    """
    Batch-optimized BeatNet for training and offline inference.

    Same architecture as BeatNetCRNN but without stateful hidden states,
    making it more efficient for batch training.

    Args:
        input_dim: Input feature dimension (272 for official features)
        hidden_dim: LSTM hidden dimension (150)
        num_layers: Number of LSTM layers (2)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        input_dim: int = 272,
        hidden_dim: int = 150,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_out = 150
        self.kernel_size = 10

        # Convolutional layer
        self.conv1 = nn.Conv1d(1, 2, kernel_size=self.kernel_size, padding=0)

        # Calculate output size after conv and maxpool
        conv_out_dim = 2 * int((input_dim - self.kernel_size + 1) / 2)

        # Linear projection
        self.linear0 = nn.Linear(conv_out_dim, self.conv_out)

        # LSTM (no hidden state persistence for batch training)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.linear = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for batch processing.

        Args:
            x: Input of shape (batch, time, freq) where freq=272

        Returns:
            Dictionary with 'beats', 'downbeats', 'activations'
        """
        batch_size, time_steps, freq = x.shape

        # Reshape for conv: (batch * time, 1, freq)
        x_flat = x.view(batch_size * time_steps, 1, freq)

        # Conv + ReLU + MaxPool
        x_conv = F.max_pool1d(F.relu(self.conv1(x_flat)), 2)

        # Flatten
        x_flat = x_conv.view(batch_size * time_steps, -1)

        # Linear projection
        x_proj = self.linear0(x_flat)

        # Reshape back to sequence
        x_seq = x_proj.view(batch_size, time_steps, -1)

        # LSTM (no hidden state persistence)
        lstm_out, _ = self.lstm(x_seq)

        # Output
        logits = self.linear(lstm_out)  # (batch, time, 3)
        probs = F.softmax(logits, dim=-1)

        # Extract beat and downbeat probabilities
        beats = probs[:, :, 1] + probs[:, :, 2]
        downbeats = probs[:, :, 2]

        return {
            "beats": beats.unsqueeze(-1),
            "downbeats": downbeats.unsqueeze(-1),
            "activations": probs,
        }


# Backward compatibility alias
BeatNetCRNNBatch = BeatNetBatch


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Remove a prefix from all keys in a state dict."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
