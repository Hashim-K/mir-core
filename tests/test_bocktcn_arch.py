"""Verify BockTCN architecture matches Davies & Böck 2019 (EUSIPCO), Table I."""
import torch
from mir_core.beats.schema import EVENT_ACTIVATION_DEFINITION
from mir_core.models.bock_tcn.tcn import BockTCN


def test_output_shape_beat_only():
    """beats output must be (batch, time, 1) — time axis must survive conv frontend."""
    model = BockTCN(n_filters=16, n_dilations=2, include_downbeats=False)
    model.eval()
    # (batch=2, channels=1, time=200, freq=81) — 81 bands per paper
    x = torch.zeros(2, 1, 200, 81)
    with torch.no_grad():
        out = model(x)
    assert "beats" in out
    assert out["beats"].shape[0] == 2      # batch preserved
    assert out["beats"].shape[2] == 1      # single activation per frame
    assert out["data_definition"] is EVENT_ACTIVATION_DEFINITION
    assert out["event_activations"].shape == (2, 196, 2)
    assert out["frame_class_activations"].shape == (2, 196, 3)
    assert out["frame_classes"].shape == (2, 196)
    # time axis shrinks by 4 (2 frames lost per 3×3 valid conv × 2 layers)
    assert out["beats"].shape[1] == 196


def test_output_shape_with_downbeats():
    """downbeats output must have identical shape to beats output — same batch, time, and activation dimensions."""
    model = BockTCN(n_filters=16, n_dilations=2, include_downbeats=True)
    model.eval()
    x = torch.zeros(1, 1, 200, 81)
    with torch.no_grad():
        out = model(x)
    assert "beats" in out and "downbeats" in out
    assert out["beats"].shape == out["downbeats"].shape


def test_conv_frontend_freq_reduction():
    """Freq dim must collapse to 1 after the three conv layers (paper path)."""
    model = BockTCN(n_filters=16, n_dilations=1, include_downbeats=False)
    # Inspect intermediate shape by hooking into conv_3 output
    captured = {}
    def hook(module, inp, out):
        captured["shape"] = out.shape
    handle = model.conv_3.register_forward_hook(hook)
    model.eval()
    x = torch.zeros(1, 1, 100, 81)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    # After conv_3 (1×8, valid): freq dim must be 1
    assert captured["shape"][-1] == 1, (
        f"Expected freq=1 after conv_3, got {captured['shape'][-1]}. "
        "Check that conv layers match Davies & Böck 2019 Table I."
    )
