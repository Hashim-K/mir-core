"""
Attention modules for BEAST streaming Transformer.

Provides relative positional encoding and multi-headed attention with
relative position bias, based on Transformer-XL (Dai et al.).

Reference:
Chang & Su (2024). "BEAST: Online Joint Beat and Downbeat Tracking
Based on Streaming Transformer." ICASSP 2024.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding for streaming Transformer.

    Based on Dai et al. "Transformer-XL" - provides relative position information
    rather than absolute position, which is crucial for streaming scenarios where
    sequence positions shift with each new block.

    Args:
        d_model: Model dimension
        dropout_rate: Dropout probability
        max_len: Maximum sequence length for precomputing encodings
    """

    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)

        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, time, d_model)

        Returns:
            Tuple of (x with dropout, positional encoding)
        """
        pos_emb = self.pe[:x.size(1), :]
        return self.dropout(x), pos_emb


class RelPositionMultiHeadedAttention(nn.Module):
    """
    Multi-headed attention with relative positional encoding.

    Implements the relative position attention mechanism from Transformer-XL,
    which is critical for streaming Transformers to capture relative timing
    information in music.

    Args:
        n_head: Number of attention heads
        d_model: Model dimension
        dropout_rate: Attention dropout rate
    """

    def __init__(self, n_head: int, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        # Relative position bias terms
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(n_head, self.d_k))

        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = self.d_k ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relative position multi-head attention.

        Args:
            query: Query tensor (batch, time, d_model)
            key: Key tensor (batch, time, d_model)
            value: Value tensor (batch, time, d_model)
            pos_emb: Positional embedding (time, d_model)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch, time, d_model)
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.linear_q(query).view(batch_size, -1, self.n_head, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.n_head, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.n_head, self.d_k)

        # Transpose for attention: (batch, head, time, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Position projection: (time, d_model) -> (time, n_head, d_k) -> (n_head, time, d_k)
        p = self.linear_pos(pos_emb).view(-1, self.n_head, self.d_k)
        p = p.permute(1, 0, 2)  # (n_head, time, d_k)

        # Content attention (with position bias u)
        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        # Position attention (with position bias v)
        # q_with_v: (batch, n_head, time, d_k) @ p^T: (1, n_head, d_k, time) -> (batch, n_head, time, time)
        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_with_v, p.transpose(-2, -1).unsqueeze(0))

        # Combine scores
        scores = (content_score + pos_score) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)

        return self.linear_out(out)
