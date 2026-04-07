"""
Training utilities for beat tracking models.

Provides PyTorch Lightning modules and layer freezing helpers.

Modules:
    BeatTrackingModule    — BCE-loss module for BockTCN (1-class beat output).
    BeatNetModule         — CE-loss module for BeatNet (3-class: non-beat / beat / downbeat).
    GenreClassifierModule — CE-loss module for genre classification.

Freezing utilities:
    freeze_layers              — Freeze layers matching specified patterns.
    unfreeze_layers            — Unfreeze layers matching specified patterns.
    get_trainable_params       — Count trainable and total parameters.
    setup_layer_freezing       — Setup layer freezing for a model based on group names.
    get_bocktcn_layer_groups   — Get layer groups for BockTCN model.
    get_beatnet_layer_groups   — Get layer groups for BeatNet CRNN model.
"""

from .modules import BeatTrackingModule, BeatNetModule, GenreClassifierModule
from .freezing import (
    freeze_layers,
    unfreeze_layers,
    get_trainable_params,
    setup_layer_freezing,
    get_bocktcn_layer_groups,
    get_beatnet_layer_groups,
)

__all__ = [
    # Modules
    "BeatTrackingModule",
    "BeatNetModule",
    "GenreClassifierModule",
    # Freezing
    "freeze_layers",
    "unfreeze_layers",
    "get_trainable_params",
    "setup_layer_freezing",
    "get_bocktcn_layer_groups",
    "get_beatnet_layer_groups",
]
