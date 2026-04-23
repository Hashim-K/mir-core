"""Tests for BEAST layer freezing groups."""

from __future__ import annotations

from mir_core.models import BEAST
from mir_core.training.freezing import setup_layer_freezing


def test_beast_conv_frontend_freezing_group() -> None:
    model = BEAST(
        dmodel=32,
        nhead=4,
        d_hid=64,
        nlayers=2,
        left_size=16,
        center_size=8,
        right_size=4,
    )

    setup_layer_freezing(model, "beast", ["conv_frontend"], verbose=False)

    assert not model.conv1.weight.requires_grad
    assert not model.conv2.weight.requires_grad
    assert not model.conv3.weight.requires_grad
    assert model.encoder.encoders[0].self_attn.linear_q.weight.requires_grad
