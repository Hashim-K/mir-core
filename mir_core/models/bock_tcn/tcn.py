"""
BockTCN: Temporal Convolutional Network for beat and downbeat tracking.

Architecture matches Davies & Böck 2019, "Temporal convolutional networks
for musical audio beat tracking" (EUSIPCO), Table I:

  Signal conditioning:
    - 44.1 kHz audio, 2048-sample Hann window (46.4 ms), 10 ms hop → 100 fps
    - 81-band log-magnitude filterbank (12 bands/octave, 30–17000 Hz)

  Conv frontend (collapses freq dimension to 1):
    Layer 1: Conv2d 3×3, 16 filters  → MaxPool (1×3)   [81→26 freq bins]
    Layer 2: Conv2d 3×3, 16 filters  → MaxPool (1×3)   [26→8  freq bins]
    Layer 3: Conv2d 1×8, 16 filters  → no pooling      [8→1   freq bin]

  TCN: 1 stack, dilations 2^0…2^10, 16 filters, kernel 5, ELU, dropout 0.1

  Heads: sigmoid beat activation (and optional downbeat activation)

  Training: Adam lr=1e-3, batch=1, binary cross-entropy, ReduceLROnPlateau
             (factor 0.2, stop if no improvement for 50 epochs)

NOTE — time axis: valid padding on the two 3×3 layers removes 2 frames each,
so output time = input time − 4. Account for this in target alignment.

Paper: https://doi.org/10.23919/EUSIPCO.2019.8902843
"""

from typing import Any, List, Dict, Tuple
import torch
import torch.nn as nn

from mir_core.beats.schema import EVENT_ACTIVATION_DEFINITION, FRAME_CLASS_DEFINITION
from mir_core.beats.tensor_converters import (
    event_activations_to_frame_class_activations,
    event_activations_to_frame_classes,
)


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions for TCN.

    Features two parallel dilated convolutions with different dilation rates,
    concatenated and combined with a residual connection.

    Args:
        dilation_rate: Base dilation rate (second conv uses 2x)
        n_filters: Number of output filters
        kernel_size: Convolution kernel size
        padding: Padding mode ('same' or 'valid')
        dropout_rate: Dropout probability
        in_channels: Number of input channels
    """

    def __init__(
        self,
        dilation_rate: int,
        n_filters: int,
        kernel_size: int = 5,
        padding: str = "same",
        dropout_rate: float = 0.1,
        in_channels: int = 16,
    ):
        super().__init__()

        # Residual projection
        self.res = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            padding=padding,
        )

        # First dilated convolution
        self.conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )

        # Second dilated convolution (2x dilation)
        self.conv_2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            dilation=dilation_rate * 2,
            padding=padding,
        )

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

        # 1x1 conv to match dimensions
        self.conv_3 = nn.Conv1d(
            in_channels=2 * n_filters,
            out_channels=n_filters,
            kernel_size=1,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (output + residual, skip connection)
        """
        res_x = self.res(x)

        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(x)

        concat = torch.cat([conv_1, conv_2], dim=1)
        out = self.elu(concat)
        out = self.dropout(out)
        out = self.conv_3(out)

        return res_x + out, out


class TCN(nn.Module):
    """
    Temporal Convolutional Network with exponentially increasing dilations.

    Args:
        n_filters: Number of filters per layer
        kernel_size: Convolution kernel size
        dilations: List of dilation rates
        padding: Padding mode
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        n_filters: int = 16,
        kernel_size: int = 5,
        dilations: List[int] = None,
        padding: str = "same",
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        self.tcn_layers = nn.ModuleDict()
        for idx, d in enumerate(dilations):
            self.tcn_layers[f"tcn_{idx}"] = ResBlock(
                d,
                n_filters,
                kernel_size,
                padding,
                dropout_rate,
                in_channels=n_filters,
            )

        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (activated output, sum of skip connections)
        """
        skip_connections = []

        for layer_name in self.tcn_layers:
            x, skip_out = self.tcn_layers[layer_name](x)
            skip_connections.append(skip_out)

        x = self.activation(x)
        skip = torch.stack(skip_connections, dim=-1).sum(dim=-1)

        return x, skip


class BockTCN(nn.Module):
    """
    Multi-task TCN for beat, downbeat, and tempo tracking.

    Based on Davies & Böck "Temporal convolutional networks for musical
    audio beat tracking" (EUSIPCO 2019), Table I.

    Architecture:
    1. Conv frontend: 2 Conv2D blocks with MaxPool + 1 Conv2D block WITHOUT pooling
       (conv3 collapses the frequency dimension to 1 — no pooling on layer 3)
    2. TCN with 11 dilated residual blocks
    3. Task-specific heads for beats, downbeats, tempo

    Args:
        n_filters: Number of filters per layer
        n_dilations: Number of TCN layers
        kernel_size: TCN kernel size
        dropout_rate: Dropout probability
        include_downbeats: Include downbeat head
        include_tempo: Include tempo head
        verbose: Print shape info during forward pass
    """

    output_definition = EVENT_ACTIVATION_DEFINITION
    data_definition = EVENT_ACTIVATION_DEFINITION
    frame_class_definition = FRAME_CLASS_DEFINITION
    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        n_filters: int = 16,
        n_dilations: int = 11,
        kernel_size: int = 5,
        dropout_rate: float = 0.1,
        include_downbeats: bool = False,
        include_tempo: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self.verbose = verbose
        self.include_downbeats = include_downbeats
        self.include_tempo = include_tempo

        # Convolutional frontend
        self.conv_1 = nn.Conv2d(1, n_filters, (3, 3), padding="valid")
        self.elu_1 = nn.ELU()
        self.mp_1 = nn.MaxPool2d((1, 3))
        self.dropout_1 = nn.Dropout(dropout_rate)

        # Conv 2: 16 filters, 3×3, MaxPool 1×3 (freq only)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, (3, 3), padding="valid")
        self.elu_2 = nn.ELU()
        self.mp_2 = nn.MaxPool2d((1, 3))
        self.dropout_2 = nn.Dropout(dropout_rate)

        # Conv 3: 16 filters, 1×8, NO pooling — collapses freq to 1
        # Input freq after conv2+pool: 81→79→26→24→8; after conv3: 8→1
        self.conv_3 = nn.Conv2d(n_filters, n_filters, (1, 8), padding="valid")
        self.elu_3 = nn.ELU()
        self.dropout_3 = nn.Dropout(dropout_rate)
        # NOTE: no mp_3 — paper specifies no pooling on third layer

        # TCN
        dilations = [2 ** i for i in range(n_dilations)]
        self.tcn = TCN(n_filters, kernel_size, dilations, "same", dropout_rate)

        # Beat head
        self.beats_dropout = nn.Dropout(dropout_rate)
        self.beats_dense = nn.Linear(n_filters, 1)
        self.beats_act = nn.Sigmoid()

        # Optional downbeat head
        if include_downbeats:
            self.downbeats_dropout = nn.Dropout(dropout_rate)
            self.downbeats_dense = nn.Linear(n_filters, 1)
            self.downbeats_act = nn.Sigmoid()

        # Optional tempo head
        if include_tempo:
            self.tempo_dropout = nn.Dropout(dropout_rate)
            self.tempo_pool = nn.AdaptiveAvgPool1d(1)
            self.tempo_dense = nn.Linear(n_filters, 300)
            self.tempo_act = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, time, freq)

        Returns:
            Dictionary with 'beats' (and optionally 'downbeats', 'tempo')
        """
        if self.verbose:
            print("input shape:", x.shape)

        # Conv frontend
        x = self.dropout_1(self.mp_1(self.elu_1(self.conv_1(x))))
        if self.verbose:
            print("block1 out:", x.shape)

        x = self.dropout_2(self.mp_2(self.elu_2(self.conv_2(x))))
        if self.verbose:
            print("block2 out:", x.shape)

        x = self.dropout_3(self.elu_3(self.conv_3(x)))
        if self.verbose:
            print("block3 out:", x.shape)

        # Reshape for TCN: (batch, channels, time)
        x = torch.squeeze(x, -1)
        if self.verbose:
            print("reshape:", x.shape)

        # TCN
        x, skip = self.tcn(x)
        if self.verbose:
            print("tcn out:", x.shape, "skip:", skip.shape)

        # Reshape for linear layers: (batch, time, channels)
        x = x.transpose(-2, -1)
        if self.verbose:
            print("transposed:", x.shape)

        # Beat head
        beats = self.beats_act(self.beats_dense(self.beats_dropout(x)))
        if self.verbose:
            print("beats:", beats.shape)

        outputs = {"beats": beats}

        # Optional heads
        if self.include_downbeats:
            downbeats = self.downbeats_act(
                self.downbeats_dense(self.downbeats_dropout(x))
            )
            outputs["downbeats"] = downbeats

        downbeats_for_event = outputs.get("downbeats", torch.zeros_like(beats))
        event_activations = torch.cat((beats, downbeats_for_event), dim=-1)
        outputs.update(
            {
                "event_activations": event_activations,
                "frame_class_activations": event_activations_to_frame_class_activations(
                    event_activations
                ),
                "frame_classes": event_activations_to_frame_classes(event_activations),
                "data_definition": self.output_definition,
            }
        )

        if self.include_tempo:
            # Use skip connections for tempo (global context)
            skip_t = skip.transpose(-2, -1)
            tempo = self.tempo_dropout(skip_t)
            tempo = tempo.mean(dim=1)  # Global average pool over time
            tempo = self.tempo_act(self.tempo_dense(tempo))
            outputs["tempo"] = tempo

        return outputs
