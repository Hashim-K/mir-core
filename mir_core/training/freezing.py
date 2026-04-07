"""
Layer freezing utilities for fine-tuning beat tracking models.

Following Rapini & Jordanous (2024): freeze convolutional frontend,
fine-tune TCN layers and output heads.

Functions:
    freeze_layers              — Freeze layers matching specified patterns.
    unfreeze_layers            — Unfreeze layers matching specified patterns.
    get_trainable_params       — Count trainable and total parameters.
    setup_layer_freezing       — Setup layer freezing for a model based on group names.
    get_bocktcn_layer_groups   — Get layer groups for BockTCN model.
    get_beatnet_layer_groups   — Get layer groups for BeatNet CRNN model.
"""

from typing import Dict, List, Optional, Tuple

import torch.nn as nn


def get_bocktcn_layer_groups() -> Dict[str, List[str]]:
    """
    Get layer groups for BockTCN model.

    Following Rapini & Jordanous (2024): freeze convolutional frontend,
    fine-tune TCN layers and output heads.

    Returns:
        Dictionary mapping group names to layer name patterns
    """
    return {
        "conv_frontend": ["conv_1", "conv_2", "conv_3", "elu_1", "elu_2", "elu_3",
                          "mp_1", "mp_2", "mp_3", "dropout_1", "dropout_2", "dropout_3"],
        "tcn": ["tcn"],
        "beat_head": ["beats_dropout", "beats_dense", "beats_act"],
        "downbeat_head": ["downbeats_dropout", "downbeats_dense", "downbeats_act"],
        "tempo_head": ["tempo_dropout", "tempo_pool", "tempo_dense", "tempo_act"],
    }


def get_beatnet_layer_groups() -> Dict[str, List[str]]:
    """
    Get layer groups for BeatNet CRNN model.

    Architecture (matching official BeatNet):
    - conv1: Conv1D(1, 2, kernel_size=10)
    - linear0: Linear projection to 150
    - lstm: 2-layer LSTM (150 hidden)
    - linear: Output layer (3 classes)

    For fine-tuning (Rapini & Jordanous 2024):
    - Freeze: convolutional layers only (conv1)
    - Train: LSTM layers and final layer (linear0, lstm, linear)

    Returns:
        Dictionary mapping group names to layer name patterns
    """
    return {
        "conv": ["conv1"],  # Only the conv layer (frozen during fine-tuning)
        "projection": ["linear0"],  # Linear projection (trainable)
        "lstm": ["lstm"],  # LSTM layers (trainable)
        "output": ["linear"],  # Output layer (trainable)
    }


def freeze_layers(
    model: nn.Module,
    layer_patterns: List[str],
    verbose: bool = True,
) -> List[str]:
    """
    Freeze layers matching specified patterns.

    Args:
        model: Neural network model
        layer_patterns: List of layer name patterns to freeze
        verbose: Whether to print frozen layers

    Returns:
        List of frozen parameter names
    """
    frozen = []
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in layer_patterns):
            param.requires_grad = False
            frozen.append(name)
            if verbose:
                print(f"  Frozen: {name}")

    return frozen


def unfreeze_layers(
    model: nn.Module,
    layer_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Unfreeze layers matching specified patterns (or all if None).

    Args:
        model: Neural network model
        layer_patterns: List of layer name patterns to unfreeze (None = all)
        verbose: Whether to print unfrozen layers

    Returns:
        List of unfrozen parameter names
    """
    unfrozen = []
    for name, param in model.named_parameters():
        if layer_patterns is None or any(pattern in name for pattern in layer_patterns):
            param.requires_grad = True
            unfrozen.append(name)
            if verbose:
                print(f"  Unfrozen: {name}")

    return unfrozen


def get_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters.

    Args:
        model: Neural network model

    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def setup_layer_freezing(
    model: nn.Module,
    model_type: str,
    freeze_groups: List[str],
    verbose: bool = True,
) -> nn.Module:
    """
    Setup layer freezing for a model based on group names.

    Args:
        model: Neural network model
        model_type: Type of model ('bocktcn' or 'beatnet')
        freeze_groups: List of layer group names to freeze
        verbose: Whether to print details

    Returns:
        Model with frozen layers
    """
    if model_type.lower() in ["bocktcn", "bock_tcn", "tcn"]:
        layer_groups = get_bocktcn_layer_groups()
    elif model_type.lower() in ["beatnet", "beatnet_crnn", "crnn"]:
        layer_groups = get_beatnet_layer_groups()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Collect patterns to freeze
    patterns_to_freeze = []
    for group in freeze_groups:
        if group in layer_groups:
            patterns_to_freeze.extend(layer_groups[group])
        else:
            print(f"Warning: Unknown layer group '{group}'")

    if verbose:
        print(f"\nFreezing layer groups: {freeze_groups}")

    frozen = freeze_layers(model, patterns_to_freeze, verbose=verbose)

    trainable, total = get_trainable_params(model)
    if verbose:
        print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model
