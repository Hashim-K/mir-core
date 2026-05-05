"""
Training utilities for beat tracking models.

Provides PyTorch Lightning modules and layer freezing helpers.

Modules:
    BeatTrackingModule    — BCE-loss module for BockTCN (1-class beat output).
    BeatNetModule         — CE-loss module for BeatNet (3-class: beat / downbeat / non-beat).
    GenreClassifierModule — CE-loss module for genre classification.

Freezing utilities:
    freeze_layers              — Freeze layers matching specified patterns.
    unfreeze_layers            — Unfreeze layers matching specified patterns.
    get_trainable_params       — Count trainable and total parameters.
    setup_layer_freezing       — Setup layer freezing for a model based on group names.
    get_bocktcn_layer_groups   — Get layer groups for BockTCN model.
    get_beatnet_layer_groups   — Get layer groups for BeatNet CRNN model.
"""

from .freezing import (
    freeze_layers,
    unfreeze_layers,
    get_trainable_params,
    setup_layer_freezing,
    get_bocktcn_layer_groups,
    get_beast_layer_groups,
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
    "get_beast_layer_groups",
    "get_beatnet_layer_groups",
]


def __getattr__(name: str):
    """Load Lightning training modules only when requested."""
    if name in {"BeatTrackingModule", "BeatNetModule", "GenreClassifierModule"}:
        from . import modules

        value = getattr(modules, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
