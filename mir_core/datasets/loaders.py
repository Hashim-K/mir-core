"""
Data loading utilities for beat tracking datasets.

Loaders (via mirdata or custom loaders):
    load_dataset_tracks       — GTZAN, BRID, Candombe, and other mirdata datasets.
    load_salsaset_tracks      — SalsaSet (custom loader from GitHub).
    load_salsa_dataset_tracks — Salsa Dataset (custom loader from GitHub).
    load_genre_dataset        — Load audio paths and genre labels from multiple datasets.
"""

import os
import re
import sys
import glob
import importlib.util
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import mirdata


SALSASET_REPO_URL = "https://github.com/Hashim-K/salsaset.git"
SALSA_DATASET_REPO_URL = "https://github.com/Hashim-K/salsa-dataset.git"


def load_dataset_tracks(
    dataset_name: str,
    data_home: Optional[str] = None,
    version: Optional[str] = None,
    download: bool = False,
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """
    Load a dataset and its tracks using mirdata.

    Args:
        dataset_name: Name of the mirdata dataset (e.g., 'gtzan_genre', 'brid', 'candombe')
        data_home: Path to store dataset files
        version: Dataset version (e.g., 'mini' for GTZAN)
        download: Whether to download the dataset

    Returns:
        Tuple of (dataset object, tracks dict, track keys list)
    """
    # Initialize dataset
    init_kwargs = {"data_home": data_home} if data_home else {}
    if version:
        init_kwargs["version"] = version

    dataset = mirdata.initialize(dataset_name, **init_kwargs)

    if download:
        dataset.download()

    tracks = dataset.load_tracks()
    keys = list(tracks.keys())

    # Special filtering for BRID dataset
    if dataset_name == "brid":
        pattern = r'^\[\d{4}\] M\d+-\d+-[A-Z]+$'
        keys = [k for k in keys if bool(re.match(pattern, k))]

    return dataset, tracks, keys


def load_salsaset_tracks(
    data_home: str,
    download: bool = False,
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load SalsaSet dataset using the custom mirdata-compatible loader.

    If the dataset directory does not exist and *download* is True, it will be
    cloned from GitHub automatically.

    Args:
        data_home: Path to the salsaset directory (containing salsaset_audio/, etc.)
        download: If True, clone from GitHub when the dataset is missing.

    Returns:
        Tuple of (dataset object, tracks dict, track keys list)
    """
    loader_path = os.path.join(data_home, "salsaset_loader.py")

    if not os.path.isfile(loader_path):
        if download:
            import subprocess

            parent = os.path.dirname(data_home)
            os.makedirs(parent, exist_ok=True)
            print(f"SalsaSet not found at {data_home} — cloning from GitHub …")
            subprocess.check_call(
                ["git", "clone", SALSASET_REPO_URL, data_home],
            )
        else:
            raise FileNotFoundError(
                f"SalsaSet dataset not found at {data_home}. "
                f"Clone it with: git clone {SALSASET_REPO_URL} {data_home}"
            )

    # Add the salsaset directory to sys.path so we can import the loader
    if data_home not in sys.path:
        sys.path.insert(0, data_home)

    import salsaset_loader  # custom mirdata-compatible loader

    dataset = salsaset_loader.Dataset(data_home=data_home)
    tracks = dataset.load_tracks()
    keys = list(tracks.keys())

    return dataset, tracks, keys


def _load_dataset_from_loader(
    data_home: str,
    repo_url: str,
    dataset_name: str,
    loader_filenames: List[str],
    download: bool = False,
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load a dataset using a repo-local custom loader module."""
    class _BeatProxy:
        def __init__(self, times: np.ndarray):
            self.times = np.asarray(times, dtype=float)

    class _TrackProxy:
        def __init__(self, raw_track: Any):
            self._raw = raw_track
            self.track_id = getattr(raw_track, "track_id", "")
            self.beats = self._adapt_beats(raw_track)
            self.audio_path = getattr(raw_track, "audio_path", None)

        @property
        def audio(self):
            return getattr(self._raw, "audio", None)

        @staticmethod
        def _adapt_beats(raw_track: Any) -> Optional[_BeatProxy]:
            beats = getattr(raw_track, "beats", None)
            if beats is not None and hasattr(beats, "times"):
                return beats

            raw_frames = getattr(raw_track, "beat_frames", None)
            if raw_frames is None:
                return None

            arr = np.asarray(raw_frames, dtype=float)
            if arr.size == 0:
                return None

            # salsa-dataset annotations are millisecond timestamps.
            times = arr / 1000.0 if np.nanmax(arr) > 1000 else arr
            return _BeatProxy(times=times)

    loader_path = None
    for filename in loader_filenames:
        candidate = os.path.join(data_home, filename)
        if os.path.isfile(candidate):
            loader_path = candidate
            break

    if loader_path is None:
        # Fallback: discover likely loader files recursively
        candidates = sorted(
            glob.glob(os.path.join(data_home, "**", "*loader.py"), recursive=True)
        )
        if candidates:
            loader_path = candidates[0]

    if loader_path is None:
        if download:
            import subprocess

            parent = os.path.dirname(data_home)
            os.makedirs(parent, exist_ok=True)
            print(f"{dataset_name} not found at {data_home} — cloning from GitHub …")
            subprocess.check_call(
                ["git", "clone", repo_url, data_home],
            )
            for filename in loader_filenames:
                candidate = os.path.join(data_home, filename)
                if os.path.isfile(candidate):
                    loader_path = candidate
                    break
            if loader_path is None:
                candidates = sorted(
                    glob.glob(os.path.join(data_home, "**", "*loader.py"), recursive=True)
                )
                if candidates:
                    loader_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {data_home}. "
                f"Clone it with: git clone {repo_url} {data_home}"
            )

    if loader_path is None:
        raise FileNotFoundError(
            f"Could not find loader module in {data_home}. "
            f"Expected one of: {', '.join(loader_filenames)}"
        )

    module_name = f"{dataset_name.lower().replace('-', '_')}_loader_runtime"
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import dataset loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    # Some libraries (e.g., dataclasses) inspect sys.modules during class creation.
    # Register before execution to avoid module lookup failures.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    dataset_cls = getattr(module, "Dataset", None) or getattr(module, "SalsaDataset", None)
    if dataset_cls is None:
        raise AttributeError(
            f"Loader at {loader_path} does not expose Dataset or SalsaDataset class."
        )

    try:
        dataset = dataset_cls(data_home=data_home)
    except TypeError:
        dataset = dataset_cls(root_dir=data_home)

    if hasattr(dataset, "load_tracks"):
        tracks = dataset.load_tracks()
    elif hasattr(dataset, "track_ids"):
        tracks = {tid: _TrackProxy(dataset[tid]) for tid in dataset.track_ids}
    else:
        raise AttributeError(
            f"Dataset loader at {loader_path} must expose load_tracks() or track_ids."
        )

    # Keep only usable tracks for training (beat times present + audio file path exists when exposed)
    tracks = {
        k: t for k, t in tracks.items()
        if getattr(getattr(t, "beats", None), "times", None) is not None
        and (
            getattr(t, "audio_path", None) is None
            or os.path.exists(str(t.audio_path))
        )
    }
    keys = list(tracks.keys())
    return dataset, tracks, keys


def load_salsa_dataset_tracks(
    data_home: str,
    download: bool = False,
) -> Tuple[Any, Dict[str, Any], List[str]]:
    """Load Salsa Dataset from https://github.com/Hashim-K/salsa-dataset."""
    return _load_dataset_from_loader(
        data_home=data_home,
        repo_url=SALSA_DATASET_REPO_URL,
        dataset_name="salsa-dataset",
        loader_filenames=[
            "salsa_dataset_loader.py",
            "salsaset_loader.py",
            "loader.py",
        ],
        download=download,
    )


def load_genre_dataset(
    datasets_config: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[str]]:
    """Load audio paths and genre labels from multiple datasets.

    Uses the existing ``load_dataset_tracks``, ``load_salsaset_tracks``, and
    ``load_salsa_dataset_tracks`` loaders. Each entry in *datasets_config*
    maps a genre label to a dict specifying how to load that dataset.

    Args:
        datasets_config: Mapping of genre -> loader config, e.g.::

            {
                "candombe": {"loader": "mirdata", "name": "candombe", "data_home": "/data/candombe"},
                "samba":    {"loader": "mirdata", "name": "brid", "data_home": "/data/brid"},
                "salsa":    {"loader": "salsa_dataset", "data_home": "/data/salsa-dataset"},
                "other":    {"loader": "mirdata", "name": "gtzan_genre", "data_home": "/data/gtzan"},
            }

    Returns:
        Tuple of (audio_paths, genre_labels) -- parallel lists.
    """
    audio_paths: List[str] = []
    genre_labels: List[str] = []

    for genre, cfg in datasets_config.items():
        loader = cfg.get("loader", "mirdata")

        if loader == "mirdata":
            _, tracks, keys = load_dataset_tracks(
                dataset_name=cfg["name"],
                data_home=cfg.get("data_home"),
            )
        elif loader == "salsaset":
            _, tracks, keys = load_salsaset_tracks(
                data_home=cfg["data_home"],
            )
        elif loader == "salsa_dataset":
            _, tracks, keys = load_salsa_dataset_tracks(
                data_home=cfg["data_home"],
            )
        else:
            raise ValueError(f"Unknown loader type: {loader}")

        for key in keys:
            track = tracks[key]
            path = getattr(track, "audio_path", None)
            if path is None:
                # Try to get path from audio property
                audio_data = getattr(track, "audio", None)
                if audio_data is not None and hasattr(audio_data, "__len__"):
                    continue  # No path available, skip
                continue
            if os.path.exists(str(path)):
                audio_paths.append(str(path))
                genre_labels.append(genre)

    return audio_paths, genre_labels
