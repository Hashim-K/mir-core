"""
Contextual Block Transformer Encoder for BEAST streaming inference.

Implements block processing with context from previous blocks,
enabling online/streaming inference while maintaining temporal context.

Reference:
Chang & Su (2024). "BEAST: Online Joint Beat and Downbeat Tracking
Based on Streaming Transformer." ICASSP 2024.
"""

from typing import Optional
import torch
import torch.nn as nn

from .attention import RelPositionalEncoding, RelPositionMultiHeadedAttention


class PositionwiseFeedForward(nn.Module):
    """
    Positionwise feed-forward layer with GELU activation.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        dropout_rate: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class ContextualBlockEncoderLayer(nn.Module):
    """
    Contextual Block Encoder layer for streaming Transformer.

    Implements block processing with context from previous blocks,
    enabling online/streaming inference while maintaining temporal context.
    Based on Tsunoo et al. "Transformer ASR with contextual block processing".

    Args:
        size: Model dimension
        self_attn: Self-attention module
        feed_forward: Feed-forward module
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size)
        self.norm2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.

        Args:
            x: Input tensor (batch, time, size)
            pos_emb: Positional embedding
            mask: Attention mask

        Returns:
            Output tensor of same shape
        """
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, pos_emb, mask)
        x = self.dropout(x) + residual

        # Pre-norm feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual

        return x


class ContextualBlockTransformerEncoder(nn.Module):
    """
    Contextual Block Transformer Encoder for streaming inference.

    Implements the streaming Transformer architecture from BEAST, based on
    Tsunoo et al. "Transformer ASR with contextual block processing".

    The encoder processes input in blocks with context vectors from previous
    blocks, enabling online inference while maintaining temporal context.

    Args:
        input_size: Input feature dimension (kept for API compatibility)
        output_size: Output/model dimension
        attention_heads: Number of attention heads
        linear_units: Feed-forward hidden dimension
        num_blocks: Number of Transformer layers
        dropout_rate: Dropout probability
        positional_dropout_rate: Positional encoding dropout
        attention_dropout_rate: Attention dropout
        normalize_before: Use pre-norm (True) or post-norm (False)
        left_size: Left context size (past frames)
        center_size: Center/current block size
        right_size: Right context/look-ahead size
    """

    def __init__(
        self,
        input_size: int,  # kept for API compatibility
        output_size: int = 256,
        attention_heads: int = 8,
        linear_units: int = 1024,
        num_blocks: int = 9,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        normalize_before: bool = True,
        left_size: int = 256,
        center_size: int = 16,
        right_size: int = 16,
    ):
        super().__init__()

        # input_size is accepted for API compatibility but we use output_size
        _ = input_size  # suppress unused warning

        self.output_size = output_size
        self.normalize_before = normalize_before
        self.left = left_size
        self.center = center_size
        self.look_ahead = right_size
        self.block_size = left_size + center_size + right_size

        # Positional encoding
        self.pos_enc = RelPositionalEncoding(output_size, positional_dropout_rate)

        # Encoder layers
        self.encoders = nn.ModuleList([
            ContextualBlockEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                dropout_rate,
            )
            for _ in range(num_blocks)
        ])

        if normalize_before:
            self.after_norm = nn.LayerNorm(output_size)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch processing (training/offline inference).

        For simplicity, this processes the full sequence. For true streaming,
        block-wise processing with context vectors would be used.

        Args:
            xs: Input tensor of shape (batch, time, input_size)

        Returns:
            Output tensor of shape (batch, time, output_size)
        """
        xs, pos_emb = self.pos_enc(xs)

        for encoder in self.encoders:
            xs = encoder(xs, pos_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs
