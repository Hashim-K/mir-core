import torch

from mir_core.models.beatnet.crnn import BeatNetBatch, BeatNetCRNN


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

    assert torch.allclose(output["beats"].squeeze(-1), expected[0].expand(2, 5))
    assert torch.allclose(output["downbeats"].squeeze(-1), expected[1].expand(2, 5))
    assert torch.allclose(output["activations"][:, :, 2], expected[2].expand(2, 5))


def test_beatnet_final_pred_softmaxes_over_class_axis_for_batched_logits():
    model = BeatNetCRNN()
    logits = torch.randn(2, 3, 7)

    probs = model.final_pred(logits)

    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 7), atol=1e-6)
