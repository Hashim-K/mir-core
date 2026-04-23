"""Smoke tests for BEAST."""

from __future__ import annotations

import torch

from mir_core.models import BEAST


def test_beast_is_public_model() -> None:
    assert BEAST.__name__ == "BEAST"


def test_beast_preserves_checkpoint_key_names() -> None:
    model = BEAST(
        dmodel=32,
        nhead=4,
        d_hid=64,
        nlayers=2,
        left_size=16,
        center_size=8,
        right_size=4,
    )
    keys = set(model.state_dict())

    assert "conv1.weight" in keys
    assert "encoder.encoders.0.self_attn.linear_q.weight" in keys
    assert "out_linear.weight" in keys
    assert "out_linear_t.weight" in keys


def test_beast_forward_returns_raw_logits() -> None:
    torch.manual_seed(0)
    model = BEAST(
        dmodel=32,
        nhead=4,
        d_hid=64,
        nlayers=2,
        left_size=16,
        center_size=8,
        right_size=4,
    )
    model.eval()

    with torch.no_grad():
        logits, tempo_logits = model(torch.zeros(2, 40, 128))

    assert logits.shape == (2, 40, 2)
    assert tempo_logits.shape == (2, 300)
    assert not any(isinstance(module, torch.nn.Sigmoid) for module in model.modules())
    assert not torch.allclose(
        tempo_logits.softmax(dim=-1),
        tempo_logits,
    )
