"""
Dataset splitting utilities for beat tracking experiments.

Functions:
    get_dataset_splits               — Simple train/val/test split.
    get_kfold_splits                 — K-fold CV (default 5-fold per Rapini & Jordanous, 2024).
    get_stratified_kfold_splits      — Stratified K-fold by label (e.g. tempo range).
    get_incremental_training_splits  — 1..N file incremental splits (Maia et al., 2023).
    create_dataloaders               — Create PyTorch DataLoaders for train/val/test.

Classes:
    CrossValidationRunner            — Orchestrates fold creation and result aggregation.
"""

from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from .beat_tracking_dataset import BeatTrackingDataset


def get_dataset_splits(
    track_keys: List[str],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split track keys into train/val/test sets.

    Args:
        track_keys: List of track IDs
        test_size: Fraction for test set
        val_size: Fraction of remaining for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_keys, val_keys, test_keys)
    """
    from sklearn.model_selection import train_test_split

    train_keys, test_keys = train_test_split(
        track_keys,
        test_size=test_size,
        random_state=random_state
    )
    train_keys, val_keys = train_test_split(
        train_keys,
        test_size=val_size,
        random_state=random_state
    )

    return train_keys, val_keys, test_keys


def create_dataloaders(
    train_dataset: BeatTrackingDataset,
    val_dataset: BeatTrackingDataset,
    test_dataset: Optional[BeatTrackingDataset] = None,
    batch_size: int = 1,
    num_workers: int = 4,
) -> Tuple:
    """
    Create PyTorch DataLoaders for train/val/test datasets.

    Note: batch_size=1 is typical for variable-length sequences.
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


# =============================================================================
# Cross-Validation Utilities
# =============================================================================

def get_kfold_splits(
    track_keys: List[str],
    n_folds: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate K-fold cross-validation splits.

    Following Rapini & Jordanous (2024) methodology using 5-fold CV.

    Args:
        track_keys: List of track IDs
        n_folds: Number of folds (default 5 per literature)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting

    Returns:
        List of (train_keys, test_keys) tuples for each fold
    """
    from sklearn.model_selection import KFold

    track_keys = np.array(track_keys)
    kfold = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    splits = []
    for train_idx, test_idx in kfold.split(track_keys):
        train_keys = track_keys[train_idx].tolist()
        test_keys = track_keys[test_idx].tolist()
        splits.append((train_keys, test_keys))

    return splits


def get_stratified_kfold_splits(
    track_keys: List[str],
    labels: List[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate stratified K-fold splits based on labels (e.g., tempo range, genre).

    Useful when dataset has subgroups that should be evenly distributed.

    Args:
        track_keys: List of track IDs
        labels: List of labels for stratification (same length as track_keys)
        n_folds: Number of folds
        random_state: Random seed

    Returns:
        List of (train_keys, test_keys) tuples
    """
    from sklearn.model_selection import StratifiedKFold

    track_keys = np.array(track_keys)
    labels = np.array(labels)

    skfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    splits = []
    for train_idx, test_idx in skfold.split(track_keys, labels):
        train_keys = track_keys[train_idx].tolist()
        test_keys = track_keys[test_idx].tolist()
        splits.append((train_keys, test_keys))

    return splits


def get_incremental_training_splits(
    track_keys: List[str],
    max_files: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[List[str]]]:
    """
    Generate splits for incremental training experiments.

    Following Maia et al. (2023) methodology: evaluate with 1, 2, ..., max_files
    training samples while keeping test set fixed.

    Args:
        track_keys: List of track IDs
        max_files: Maximum number of training files to use
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Tuple of (test_keys, list of training_keys for each n)
    """
    from sklearn.model_selection import train_test_split

    train_pool, test_keys = train_test_split(
        track_keys,
        test_size=test_size,
        random_state=random_state
    )

    # Shuffle training pool
    rng = np.random.default_rng(random_state)
    train_pool = list(train_pool)
    rng.shuffle(train_pool)

    # Generate incremental splits
    training_splits = []
    for n in range(1, min(max_files + 1, len(train_pool) + 1)):
        training_splits.append(train_pool[:n])

    return test_keys, training_splits


class CrossValidationRunner:
    """
    Helper class for running cross-validation experiments.

    Manages fold splits, training, and result aggregation.

    Args:
        tracks: Dictionary of track objects
        track_keys: List of track IDs
        n_folds: Number of folds
        random_state: Random seed
    """

    def __init__(
        self,
        tracks: Dict[str, Any],
        track_keys: List[str],
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.tracks = tracks
        self.track_keys = track_keys
        self.n_folds = n_folds
        self.random_state = random_state

        # Generate splits
        self.splits = get_kfold_splits(
            track_keys, n_folds=n_folds, random_state=random_state
        )

        # Results storage
        self.fold_results = {}

    def get_fold_datasets(
        self,
        fold: int,
        val_size: float = 0.1,
    ) -> Tuple[BeatTrackingDataset, BeatTrackingDataset, BeatTrackingDataset]:
        """
        Get train/val/test datasets for a specific fold.

        Args:
            fold: Fold index (0 to n_folds-1)
            val_size: Fraction of training data for validation

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_keys, test_keys = self.splits[fold]

        # Split training into train/val
        if val_size > 0:
            from sklearn.model_selection import train_test_split

            train_keys, val_keys = train_test_split(
                train_keys,
                test_size=val_size,
                random_state=self.random_state + fold
            )
        else:
            val_keys = train_keys[:max(1, len(train_keys) // 10)]

        train_dataset = BeatTrackingDataset(self.tracks, train_keys)
        val_dataset = BeatTrackingDataset(self.tracks, val_keys)
        test_dataset = BeatTrackingDataset(self.tracks, test_keys)

        return train_dataset, val_dataset, test_dataset

    def get_fold_dataloaders(
        self,
        fold: int,
        batch_size: int = 1,
        num_workers: int = 4,
        **dataset_kwargs,
    ) -> Tuple:
        """
        Get dataloaders for a specific fold.

        Args:
            fold: Fold index
            batch_size: Batch size
            num_workers: Number of data loading workers
            **dataset_kwargs: Additional arguments for BeatTrackingDataset

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_ds, val_ds, test_ds = self.get_fold_datasets(fold, **dataset_kwargs)
        return create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def __len__(self) -> int:
        return self.n_folds

    def __iter__(self):
        """Iterate over folds."""
        for fold in range(self.n_folds):
            yield fold, self.splits[fold]
