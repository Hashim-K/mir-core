"""
Data loading and dataset handling utilities for beat tracking.

Datasets:
    BeatDataset    — PyTorch Dataset for BockTCN (81-dim madmom mel-spectrogram, 100 FPS).
    BeatNetDataset — PyTorch Dataset for BeatNet (272-dim LOG_SPECT, 50 FPS, 3-class targets).
    GenreDataset   — PyTorch Dataset for genre classification from audio segments.

Loaders (via mirdata or custom loaders):
    load_dataset_tracks       — GTZAN, BRID, Candombe, and other mirdata datasets.
    load_salsaset_tracks      — SalsaSet (custom loader from GitHub).
    load_salsa_dataset_tracks — Salsa Dataset (custom loader from GitHub).
    load_genre_dataset        — Load audio paths and genre labels from multiple datasets.

Splitting:
    get_dataset_splits               — Simple train/val/test split.
    get_kfold_splits                 — K-fold CV (default 5-fold per Rapini & Jordanous, 2024).
    get_stratified_kfold_splits      — Stratified K-fold by label (e.g. tempo range).
    get_incremental_training_splits  — 1..N file incremental splits (Maia et al., 2023).
    CrossValidationRunner            — Orchestrates fold creation and result aggregation.
"""

from .beat_dataset import BeatDataset
from .beatnet_dataset import BeatNetDataset
from .genre_dataset import GenreDataset
from .loaders import (
    load_dataset_tracks,
    load_salsaset_tracks,
    load_salsa_dataset_tracks,
    load_genre_dataset,
    SALSASET_REPO_URL,
    SALSA_DATASET_REPO_URL,
)
from .splits import (
    get_dataset_splits,
    get_kfold_splits,
    get_stratified_kfold_splits,
    get_incremental_training_splits,
    create_dataloaders,
    CrossValidationRunner,
)

__all__ = [
    # Datasets
    "BeatDataset",
    "BeatNetDataset",
    "GenreDataset",
    # Loaders
    "load_dataset_tracks",
    "load_salsaset_tracks",
    "load_salsa_dataset_tracks",
    "load_genre_dataset",
    "SALSASET_REPO_URL",
    "SALSA_DATASET_REPO_URL",
    # Splits
    "get_dataset_splits",
    "get_kfold_splits",
    "get_stratified_kfold_splits",
    "get_incremental_training_splits",
    "create_dataloaders",
    "CrossValidationRunner",
]
