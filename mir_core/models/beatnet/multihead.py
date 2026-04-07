"""
MultiHeadBeatNet: BeatNet with shared conv frontend and multiple genre-specific heads.

Supports training modes A (fine-tune heads with frozen pretrained conv) and
C (shared retrained conv + per-genre heads).
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .crnn import BeatNetBatch, _strip_prefix


class MultiHeadBeatNet(nn.Module):
    """BeatNet with shared conv frontend and multiple genre-specific heads.

    For training modes A (fine-tune heads with frozen pretrained conv) and
    C (shared retrained conv + per-genre heads). Runs conv1 once per frame,
    then fans out to N genre-specific heads, each consisting of
    linear0 + LSTM + output linear.

    Args:
        genre_labels: List of genre label strings (keys for heads).
        input_dim: Input feature dimension (272 for BeatNet LOG_SPECT).
        hidden_dim: LSTM hidden dimension per head.
        num_layers: Number of LSTM layers per head.
        dropout: Dropout for LSTM (applied when num_layers > 1).
    """

    def __init__(
        self,
        genre_labels: list,
        input_dim: int = 272,
        hidden_dim: int = 150,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.genre_labels = list(genre_labels)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = 10
        self.conv_out = 150

        # Shared conv frontend
        self.conv1 = nn.Conv1d(1, 2, kernel_size=self.kernel_size, padding=0)
        conv_out_dim = 2 * int((input_dim - self.kernel_size + 1) / 2)

        # Per-genre heads: each has linear0 + LSTM + output linear
        self.heads = nn.ModuleDict()
        for genre in self.genre_labels:
            self.heads[genre] = nn.ModuleDict({
                "linear0": nn.Linear(conv_out_dim, self.conv_out),
                "lstm": nn.LSTM(
                    input_size=self.conv_out,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=False,
                    dropout=dropout if num_layers > 1 else 0,
                ),
                "linear": nn.Linear(hidden_dim, 3),
            })

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Shared conv frontend. Run once per frame.

        Args:
            x: Input of shape (batch, time, freq) where freq=input_dim.

        Returns:
            Conv features of shape (batch * time, conv_out_dim).
        """
        batch_size, time_steps, freq = x.shape
        x_flat = x.view(batch_size * time_steps, 1, freq)
        x_conv = F.max_pool1d(F.relu(self.conv1(x_flat)), 2)
        return x_conv.view(batch_size * time_steps, -1)

    def forward_head(
        self, conv_features: torch.Tensor, genre: str,
        batch_size: int, time_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Run a single genre head on pre-computed conv features.

        Args:
            conv_features: Output of forward_conv, shape (batch*time, conv_out_dim).
            genre: Genre label key.
            batch_size: Batch dimension for reshaping.
            time_steps: Time dimension for reshaping.

        Returns:
            Dict with 'beats', 'downbeats', 'activations' tensors.
        """
        head = self.heads[genre]
        x_proj = head["linear0"](conv_features)
        x_seq = x_proj.view(batch_size, time_steps, -1)
        lstm_out, _ = head["lstm"](x_seq)
        logits = head["linear"](lstm_out)  # (batch, time, 3)
        probs = F.softmax(logits, dim=-1)
        beats = probs[:, :, 1] + probs[:, :, 2]
        downbeats = probs[:, :, 2]
        return {
            "beats": beats.unsqueeze(-1),
            "downbeats": downbeats.unsqueeze(-1),
            "activations": probs,
        }

    def forward_all(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Run shared conv once, then all genre heads.

        Args:
            x: Input of shape (batch, time, freq).

        Returns:
            Dict mapping genre label -> {beats, downbeats, activations}.
        """
        batch_size, time_steps, _ = x.shape
        conv_features = self.forward_conv(x)
        return {
            genre: self.forward_head(conv_features, genre, batch_size, time_steps)
            for genre in self.genre_labels
        }

    def forward(self, x: torch.Tensor, genre: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for a single genre (or all if genre is None).

        When genre is specified, returns that head's output directly.
        When genre is None, returns forward_all() result.
        """
        if genre is not None:
            batch_size, time_steps, _ = x.shape
            conv_features = self.forward_conv(x)
            return self.forward_head(conv_features, genre, batch_size, time_steps)
        return self.forward_all(x)

    @classmethod
    def from_checkpoints(
        cls,
        genre_checkpoints: Dict[str, str],
        baseline_checkpoint: Optional[str] = None,
        mode: str = "A",
        device: str = "cpu",
    ) -> "MultiHeadBeatNet":
        """Assemble a MultiHeadBeatNet from per-genre BeatNet checkpoints.

        Args:
            genre_checkpoints: Mapping genre_label -> checkpoint path.
                Each checkpoint is a standard BeatNetBatch state_dict.
            baseline_checkpoint: Path to baseline (pretrained) checkpoint.
                Required for mode A (conv from baseline).
            mode: "A" -- conv from baseline, heads from genre ckpts.
                  "C" -- conv from first genre ckpt (shared retrained), heads from genre ckpts.
            device: Device to load tensors on.

        Returns:
            Assembled MultiHeadBeatNet with loaded weights.
        """
        genre_labels = list(genre_checkpoints.keys())
        model = cls(genre_labels=genre_labels)

        # Load conv weights
        if mode == "A":
            if baseline_checkpoint is None:
                raise ValueError("Mode A requires baseline_checkpoint for conv weights.")
            baseline_sd = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
            if "state_dict" in baseline_sd:
                baseline_sd = baseline_sd["state_dict"]
            # Strip module prefix if present (Lightning wrapping)
            baseline_sd = _strip_prefix(baseline_sd, "model.")
            model.conv1.load_state_dict({
                k.replace("conv1.", ""): v
                for k, v in baseline_sd.items() if k.startswith("conv1.")
            })
        elif mode == "C":
            # Use conv from first checkpoint (all should share the same retrained conv)
            first_ckpt = list(genre_checkpoints.values())[0]
            sd = torch.load(first_ckpt, map_location=device, weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _strip_prefix(sd, "model.")
            model.conv1.load_state_dict({
                k.replace("conv1.", ""): v
                for k, v in sd.items() if k.startswith("conv1.")
            })

        # Load per-genre head weights
        for genre, ckpt_path in genre_checkpoints.items():
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _strip_prefix(sd, "model.")
            head = model.heads[genre]
            for layer_name in ("linear0", "lstm", "linear"):
                layer_sd = {
                    k.replace(f"{layer_name}.", ""): v
                    for k, v in sd.items() if k.startswith(f"{layer_name}.")
                }
                head[layer_name].load_state_dict(layer_sd)

        return model
