import importlib.util
from pathlib import Path

import pytest
import torch

from mir_core.beats.schema import EVENT_ACTIVATION_DEFINITION, FRAME_CLASS_DEFINITION, EventChannel, FrameClass
from mir_core.models.beatnet.crnn import BeatNetBatch, BeatNetCRNN
from mir_core.models.beatnet.beatnet_plus import (
    BeatNetPlusBatch,
    BeatNetPlusDualBatch,
    BeatNetPlusOnline,
)


BEATNET_PLUS_REFERENCE = (
    Path(__file__).resolve().parents[2]
    / "thesis-docs"
    / "literature"
    / "codebases"
    / "beat-detection"
    / "beatnet-plus"
    / "src"
    / "BeatNetPlus"
    / "model.py"
)


def load_beatnet_plus_reference_module():
    if not BEATNET_PLUS_REFERENCE.exists():
        pytest.skip(f"BeatNet+ reference code not available: {BEATNET_PLUS_REFERENCE}")
    spec = importlib.util.spec_from_file_location(
        "beatnet_plus_reference_model",
        BEATNET_PLUS_REFERENCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beatnet_architecture_matches_official_bda_shapes():
    model = BeatNetCRNN()

    assert sum(p.numel() for p in model.parameters()) == 402_325
    assert tuple(model.conv1.weight.shape) == (2, 1, 10)
    assert tuple(model.linear0.weight.shape) == (150, 262)
    assert tuple(model.lstm.weight_ih_l0.shape) == (600, 150)
    assert tuple(model.lstm.weight_hh_l1.shape) == (600, 150)
    assert tuple(model.linear.weight.shape) == (3, 150)


def test_beatnet_uses_official_output_class_order():
    model = BeatNetBatch()
    for param in model.parameters():
        param.data.zero_()
    model.linear.bias.data = torch.tensor([4.0, 2.0, -4.0])

    output = model(torch.zeros(2, 5, 272))
    expected = torch.softmax(model.linear.bias, dim=0)

    assert model.output_definition is FRAME_CLASS_DEFINITION
    assert model.event_activation_definition is EVENT_ACTIVATION_DEFINITION
    assert output["data_definition"] is FRAME_CLASS_DEFINITION
    assert output["event_activations"].shape == (2, 5, 2)
    assert output["frame_class_activations"].shape == (2, 5, 3)
    assert output["frame_classes"].shape == (2, 5)
    assert torch.allclose(output["beats"].squeeze(-1), expected[int(FrameClass.beat)].expand(2, 5))
    assert torch.allclose(
        output["downbeats"].squeeze(-1),
        expected[int(FrameClass.downbeat)].expand(2, 5),
    )
    assert torch.allclose(
        output["event_activations"][:, :, int(EventChannel.beat)],
        expected[int(FrameClass.beat)].expand(2, 5),
    )
    assert torch.allclose(
        output["activations"][:, :, int(FrameClass.non_beat)],
        expected[int(FrameClass.non_beat)].expand(2, 5),
    )


def test_beatnet_final_pred_softmaxes_over_class_axis_for_batched_logits():
    model = BeatNetCRNN()
    logits = torch.randn(2, 3, 7)

    probs = model.final_pred(logits)

    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 7), atol=1e-6)


def test_beatnet_plus_branch_matches_heydari_reference_train_forward():
    reference = load_beatnet_plus_reference_module()
    torch.manual_seed(1234)
    reference_model = reference.BeatNetPlusBranch(dim_in=288, num_cells=150, num_layers=4, device="cpu")
    model = BeatNetPlusBatch(input_dim=288, hidden_dim=150, num_layers=4)
    model.load_state_dict(reference_model.state_dict(), strict=True)
    x = torch.randn(2, 13, 288)

    reference_logits, reference_latent = reference_model.train_forward(x)
    logits, latent = model._forward_impl(x)

    assert torch.allclose(logits.transpose(1, 2), reference_logits, atol=1e-6)
    assert torch.allclose(latent, reference_latent, atol=1e-6)


def test_beatnet_plus_online_matches_heydari_reference_stateful_forward():
    reference = load_beatnet_plus_reference_module()
    torch.manual_seed(2345)
    reference_model = reference.BeatNetPlusBranch(dim_in=288, num_cells=150, num_layers=4, device="cpu")
    model = BeatNetPlusOnline(input_dim=288, hidden_dim=150, num_layers=4, device="cpu")
    model.load_state_dict(reference_model.state_dict(), strict=True)
    x1 = torch.randn(1, 7, 288)
    x2 = torch.randn(1, 5, 288)

    reference_out1 = reference_model(x1)
    reference_out2 = reference_model(x2)
    out1 = model(x1)
    out2 = model(x2)

    assert torch.allclose(out1, reference_out1, atol=1e-6)
    assert torch.allclose(out2, reference_out2, atol=1e-6)
    assert torch.allclose(model.hidden, reference_model.hidden, atol=1e-6)
    assert torch.allclose(model.cell, reference_model.cell, atol=1e-6)


def test_beatnet_plus_dual_branch_matches_heydari_reference_train_forward():
    reference = load_beatnet_plus_reference_module()
    torch.manual_seed(5678)
    reference_model = reference.BeatNetPlus(dim_in=288, num_cells=150, num_layers=4, device="cpu")
    model = BeatNetPlusDualBatch(input_dim=288, hidden_dim=150, num_layers=4)
    model.load_state_dict(reference_model.state_dict(), strict=True)
    main_x = torch.randn(2, 11, 288)
    aux_x = torch.randn(2, 11, 288)

    ref_main_logits, ref_aux_logits, ref_main_latent, ref_aux_latent = reference_model.train_forward(
        main_x,
        aux_x,
    )
    outputs = model.train_forward(main_x, aux_x)

    assert torch.allclose(outputs["logits"].transpose(1, 2), ref_main_logits, atol=1e-6)
    assert torch.allclose(outputs["aux_logits"].transpose(1, 2), ref_aux_logits, atol=1e-6)
    assert torch.allclose(outputs["main_latent"], ref_main_latent, atol=1e-6)
    assert torch.allclose(outputs["aux_latent"], ref_aux_latent, atol=1e-6)
