"""BeatNet+ CRNN variants.

This module mirrors the BeatNet+ reference implementation structure while
adapting outputs to the shared mir-core training/evaluation interface.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mir_core.beats.schema import (
    EVENT_ACTIVATION_DEFINITION,
    FRAME_CLASS_DEFINITION,
    EventChannel,
)
from mir_core.beats.tensor_converters import frame_class_activations_to_event_activations


class BeatNetPlusBatch(nn.Module):
    """
    BeatNet+ single-branch CRNN for batch training and inference.

    This matches the inference branch from the BeatNet+ reference code:
    288-dim LOG_SPECT features, a Conv1d frontend, and a 4-layer causal LSTM.
    """

    output_definition = FRAME_CLASS_DEFINITION
    data_definition = FRAME_CLASS_DEFINITION
    frame_class_definition = FRAME_CLASS_DEFINITION
    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        input_dim: int = 288,
        hidden_dim: int = 150,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_out = hidden_dim
        self.kernel_size = 10

        self.conv1 = nn.Conv1d(1, 2, kernel_size=self.kernel_size)
        conv_out_dim = 2 * ((input_dim - self.kernel_size + 1) // 2)
        self.linear0 = nn.Linear(conv_out_dim, self.conv_out)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_linear = nn.Linear(hidden_dim, 3)

    def _forward_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_steps, freq = x.shape
        x_flat = x.reshape(batch_size * time_steps, 1, freq)
        x_conv = F.max_pool1d(F.relu(self.conv1(x_flat)), 2)
        x_flat = x_conv.view(batch_size * time_steps, -1)
        x_proj = self.linear0(x_flat)
        x_seq = x_proj.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(x_seq)
        logits = self.output_linear(lstm_out)
        return logits, logits

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        logits, latent = self._forward_impl(x)
        probs = F.softmax(logits, dim=-1)
        event_activations = frame_class_activations_to_event_activations(probs)
        return {
            "logits": logits,
            "latent": latent,
            "beats": event_activations[:, :, int(EventChannel.beat)].unsqueeze(-1),
            "downbeats": event_activations[:, :, int(EventChannel.downbeat)].unsqueeze(-1),
            "activations": probs,
            "frame_class_activations": probs,
            "frame_classes": probs.argmax(dim=-1),
            "event_activations": event_activations,
            "data_definition": self.output_definition,
        }


class BeatNetPlusOnline(nn.Module):
    """Stateful online BeatNet+ branch compatible with Heydari's reference code."""

    output_definition = FRAME_CLASS_DEFINITION
    data_definition = FRAME_CLASS_DEFINITION
    frame_class_definition = FRAME_CLASS_DEFINITION
    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        input_dim: int = 288,
        hidden_dim: int = 150,
        num_layers: int = 4,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_out = hidden_dim
        self.kernel_size = 10
        self.device = torch.device(device)

        self.conv1 = nn.Conv1d(1, 2, kernel_size=self.kernel_size)
        conv_out_dim = 2 * ((input_dim - self.kernel_size + 1) // 2)
        self.linear0 = nn.Linear(conv_out_dim, self.conv_out)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.output_linear = nn.Linear(hidden_dim, 3)
        self.register_buffer(
            "hidden",
            torch.zeros(num_layers, 1, hidden_dim),
            persistent=False,
        )
        self.register_buffer(
            "cell",
            torch.zeros(num_layers, 1, hidden_dim),
            persistent=False,
        )
        self.to(self.device)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, freq = x.shape
        x_flat = x.reshape(batch_size * time_steps, 1, freq)
        x_conv = F.max_pool1d(F.relu(self.conv1(x_flat)), 2)
        x_flat = x_conv.view(batch_size * time_steps, -1)
        x_proj = self.linear0(x_flat)
        return x_proj.view(batch_size, time_steps, self.conv_out)

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = next(self.parameters()).device
        self.hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        self.cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.hidden.shape[1] != batch_size or self.hidden.device != x.device:
            self.reset_hidden(batch_size=batch_size, device=x.device)
        features = self._extract_features(x)
        out, (self.hidden, self.cell) = self.lstm(features, (self.hidden, self.cell))
        return self.output_linear(out).transpose(1, 2)

    def train_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_features(x)
        out = self.lstm(features)[0]
        latent = self.output_linear(out)
        return latent.transpose(1, 2), latent

    def inference_forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.train_forward(x)
        return logits

    @staticmethod
    def softmax_pred(logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=1)


class BeatNetPlusDualBatch(nn.Module):
    """BeatNet+ dual-branch wrapper for source-separated training."""

    output_definition = FRAME_CLASS_DEFINITION
    data_definition = FRAME_CLASS_DEFINITION
    frame_class_definition = FRAME_CLASS_DEFINITION
    event_activation_definition = EVENT_ACTIVATION_DEFINITION

    def __init__(
        self,
        input_dim: int = 288,
        hidden_dim: int = 150,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.main_branch = BeatNetPlusBatch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.aux_branch = BeatNetPlusBatch(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def train_forward(
        self,
        main_x: torch.Tensor,
        aux_x: torch.Tensor,
    ) -> Dict[str, Any]:
        main_logits, main_latent = self.main_branch._forward_impl(main_x)
        aux_logits, aux_latent = self.aux_branch._forward_impl(aux_x)
        probs = F.softmax(main_logits, dim=-1)
        event_activations = frame_class_activations_to_event_activations(probs)
        return {
            "logits": main_logits,
            "aux_logits": aux_logits,
            "main_latent": main_latent,
            "aux_latent": aux_latent,
            "beats": event_activations[:, :, int(EventChannel.beat)].unsqueeze(-1),
            "downbeats": event_activations[:, :, int(EventChannel.downbeat)].unsqueeze(-1),
            "activations": probs,
            "frame_class_activations": probs,
            "frame_classes": probs.argmax(dim=-1),
            "event_activations": event_activations,
            "data_definition": self.output_definition,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        return self.main_branch(x)

    def get_main_state_dict(self) -> dict[str, torch.Tensor]:
        return self.main_branch.state_dict()
