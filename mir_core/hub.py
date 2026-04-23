"""
Model Hub: Easy loading of pretrained and fine-tuned beat tracking models.

Provides a simple registry-based interface to load models with different
pretrained weights for various genres and training configurations.

Example usage:
    from mir_core import load_model, list_models

    # List all available models
    models = list_models()

    # Load a specific model
    model = load_model("bocktcn-candombe-ft")

    # Load with custom config
    model = load_model("beatnet-brid-ft", device="cuda")
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import torch
import torch.nn as nn


class ModelType(str, Enum):
    """Supported model architectures."""
    BOCKTCN = "bocktcn"
    BEATNET = "beatnet"
    BEAST = "beast"
    GENRE_CLASSIFIER = "genre_classifier"


class TrainingMethod(str, Enum):
    """Training/adaptation method."""
    PRETRAINED = "pretrained"  # Original pretrained weights
    FINE_TUNING = "ft"         # Fine-tuned on target genre
    GENRE_ONLY = "genre"       # Trained on genre from scratch
    INCREMENTAL = "incr"       # Incrementally trained with N files


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    name: str                          # Short name (e.g., "bocktcn-candombe-ft")
    display_name: str                  # Human readable name
    model_type: ModelType              # Architecture type
    training_method: TrainingMethod    # How it was trained
    genre: Optional[str] = None        # Target genre (candombe, brid, salsa, etc.)
    checkpoint_path: Optional[str] = None  # Path to checkpoint
    description: str = ""              # Model description
    metrics: Dict[str, float] = field(default_factory=dict)  # Performance metrics
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # Model init args
    n_files: Optional[int] = None      # For incremental: number of training files
    fold: Optional[int] = None         # For CV: fold number

    def __post_init__(self):
        # Auto-generate name if not provided
        if not self.name:
            parts = [self.model_type.value]
            if self.genre:
                parts.append(self.genre)
            parts.append(self.training_method.value)
            if self.n_files:
                parts.append(f"{self.n_files}files")
            if self.fold is not None:
                parts.append(f"fold{self.fold}")
            self.name = "-".join(parts)


class ModelRegistry:
    """
    Registry for managing pretrained and fine-tuned models.

    Allows registering models with their checkpoints and metadata,
    then loading them easily by name.
    """

    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._base_path: Optional[Path] = None

    def set_base_path(self, path: Union[str, Path]) -> None:
        """Set the base path for checkpoint files."""
        self._base_path = Path(path)

    def register(self, spec: ModelSpec) -> None:
        """Register a model specification."""
        self._models[spec.name] = spec

    def register_from_dict(self, d: Dict[str, Any]) -> None:
        """Register a model from dictionary config."""
        spec = ModelSpec(**d)
        self.register(spec)

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get a model specification by name."""
        return self._models.get(name)

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        genre: Optional[str] = None,
        training_method: Optional[TrainingMethod] = None,
    ) -> List[ModelSpec]:
        """
        List registered models with optional filtering.

        Args:
            model_type: Filter by architecture
            genre: Filter by target genre
            training_method: Filter by training method

        Returns:
            List of matching ModelSpec objects
        """
        results = []
        for spec in self._models.values():
            if model_type and spec.model_type != model_type:
                continue
            if genre and spec.genre != genre:
                continue
            if training_method and spec.training_method != training_method:
                continue
            results.append(spec)
        return results

    def list_names(self) -> List[str]:
        """Get all registered model names."""
        return list(self._models.keys())

    def resolve_path(self, spec: ModelSpec) -> Path:
        """Resolve the full checkpoint path for a model."""
        if not spec.checkpoint_path:
            raise ValueError(f"No checkpoint path for model: {spec.name}")

        path = Path(spec.checkpoint_path)
        if path.is_absolute():
            return path

        if self._base_path:
            return self._base_path / path

        return path


# =============================================================================
# Global Registry Instance
# =============================================================================

_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry


def _get_package_root() -> Path:
    """Get the package root directory."""
    return Path(__file__).parent


def _get_default_checkpoints_path() -> Path:
    """Get the default checkpoints path."""
    # Try experiments/checkpoints first
    pkg_root = _get_package_root()
    exp_checkpoints = pkg_root.parent / "experiments" / "checkpoints"
    if exp_checkpoints.exists():
        return exp_checkpoints
    return pkg_root / "weights"


def _discover_genre_classifiers(base_path: Path) -> None:
    """Discover genre classifier checkpoints.

    Expected structure: {base_path}/{arch}/fold{n}/best-*.ckpt
    """
    for arch_dir in base_path.iterdir():
        if not arch_dir.is_dir():
            continue
        arch = arch_dir.name  # e.g. mel_cnn
        for fold_dir in arch_dir.iterdir():
            if not fold_dir.is_dir():
                continue
            for ckpt in fold_dir.glob("best-*.ckpt"):
                fold_name = fold_dir.name  # e.g. fold0
                fold = None
                if fold_name.startswith("fold"):
                    try:
                        fold = int(fold_name.replace("fold", ""))
                    except ValueError:
                        pass

                name_parts = ["genre_classifier", arch]
                if fold is not None:
                    name_parts.append(f"f{fold}")
                name = "-".join(name_parts)

                if _registry.get(name):
                    continue

                _registry.register(ModelSpec(
                    name=name,
                    display_name=f"Genre Classifier ({arch}) fold{fold}",
                    model_type=ModelType.GENRE_CLASSIFIER,
                    training_method=TrainingMethod.FINE_TUNING,
                    checkpoint_path=str(ckpt),
                    fold=fold,
                    model_kwargs={"arch": arch},
                ))


def _auto_discover_checkpoints() -> None:
    """Auto-discover and register checkpoints from standard locations."""
    checkpoints_path = _get_default_checkpoints_path()
    weights_path = _get_package_root() / "weights"

    # Register baseline/pretrained models
    if weights_path.exists():
        bocktcn_ckpt = weights_path / "bocktcn.ckpt"
        if bocktcn_ckpt.exists():
            _registry.register(ModelSpec(
                name="bocktcn-pretrained",
                display_name="BockTCN (Pretrained)",
                model_type=ModelType.BOCKTCN,
                training_method=TrainingMethod.PRETRAINED,
                checkpoint_path=str(bocktcn_ckpt),
                description="Original BockTCN trained on GTZAN",
            ))

        tcn_lamir = weights_path / "tcn_lamir_pretrained.ckpt"
        if tcn_lamir.exists():
            _registry.register(ModelSpec(
                name="bocktcn-lamir-pretrained",
                display_name="BockTCN LAMIR (Pretrained)",
                model_type=ModelType.BOCKTCN,
                training_method=TrainingMethod.PRETRAINED,
                checkpoint_path=str(tcn_lamir),
                description="BockTCN pretrained on Latin American music",
            ))

    # Auto-discover fine-tuned models from checkpoints
    if checkpoints_path.exists():
        _registry.set_base_path(checkpoints_path)

        # Pattern: {model}_{genre}_{method}_fold{n}_best.ckpt
        for ckpt in checkpoints_path.glob("*_best*.ckpt"):
            name = ckpt.stem.replace("_best", "").replace("-v1", "").replace("-v2", "")
            parts = name.split("_")

            if len(parts) < 2:
                continue

            model_str = parts[0].lower()
            genre = parts[1].lower() if len(parts) > 1 else None

            # Determine model type
            if "bocktcn" in model_str:
                model_type = ModelType.BOCKTCN
            elif "beatnet" in model_str:
                model_type = ModelType.BEATNET
            elif "beast" in model_str:
                model_type = ModelType.BEAST
            else:
                continue

            # Determine training method
            method = TrainingMethod.FINE_TUNING
            n_files = None
            fold = None

            for part in parts:
                if "fine_tuning" in part or "finetune" in part:
                    method = TrainingMethod.FINE_TUNING
                elif "genre_only" in part:
                    method = TrainingMethod.GENRE_ONLY
                elif "incremental" in part:
                    method = TrainingMethod.INCREMENTAL

                if part.startswith("fold"):
                    try:
                        fold = int(part.replace("fold", ""))
                    except ValueError:
                        pass

                if "files" in part:
                    try:
                        n_files = int(part.replace("files", ""))
                    except ValueError:
                        pass

            # Create short name
            short_name_parts = [model_type.value]
            if genre:
                short_name_parts.append(genre)
            short_name_parts.append(method.value)
            if n_files:
                short_name_parts.append(f"{n_files}f")
            if fold is not None:
                short_name_parts.append(f"f{fold}")
            short_name = "-".join(short_name_parts)

            # Skip if already registered
            if _registry.get(short_name):
                continue

            _registry.register(ModelSpec(
                name=short_name,
                display_name=f"{model_type.value.upper()} {genre or ''} {method.value}",
                model_type=model_type,
                training_method=method,
                genre=genre,
                checkpoint_path=str(ckpt),
                n_files=n_files,
                fold=fold,
            ))

    # Discover genre classifier checkpoints
    genre_clf_path = checkpoints_path / "genre_classifier" if checkpoints_path.exists() else None
    if genre_clf_path and genre_clf_path.exists():
        _discover_genre_classifiers(genre_clf_path)

    # Also look in results/genre_classifier/checkpoints
    results_clf = _get_package_root().parent / "results" / "genre_classifier" / "checkpoints"
    if results_clf.exists():
        _discover_genre_classifiers(results_clf)

    # Also discover from the new model-centric outputs structure
    outputs_path = _get_package_root().parent / "experiments" / "outputs"
    if outputs_path.exists():
        _discover_from_outputs_structure(outputs_path)


def _discover_from_outputs_structure(outputs_path: Path) -> None:
    """
    Discover models from the model-centric outputs structure.

    Structure:
        outputs/{model}/{genre}/{method}/{fold_or_files}/checkpoint.ckpt
    """
    METHOD_MAP = {
        "fine-tuning": TrainingMethod.FINE_TUNING,
        "genre-only": TrainingMethod.GENRE_ONLY,
        "incremental": TrainingMethod.INCREMENTAL,
    }

    for model_dir in outputs_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name.lower()

        # Map to ModelType
        if "bocktcn" in model_name:
            model_type = ModelType.BOCKTCN
        elif "beatnet" in model_name:
            model_type = ModelType.BEATNET
        elif "beast" in model_name:
            model_type = ModelType.BEAST
        else:
            continue

        for genre_dir in model_dir.iterdir():
            if not genre_dir.is_dir():
                continue

            genre = genre_dir.name.lower()

            for method_dir in genre_dir.iterdir():
                if not method_dir.is_dir():
                    continue

                method_name = method_dir.name.lower()
                method = METHOD_MAP.get(method_name, TrainingMethod.FINE_TUNING)

                for exp_dir in method_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue

                    ckpt = exp_dir / "checkpoint.ckpt"
                    if not ckpt.exists():
                        continue

                    exp_name = exp_dir.name
                    fold = None
                    n_files = None

                    if exp_name.startswith("fold"):
                        try:
                            fold = int(exp_name.replace("fold", ""))
                        except ValueError:
                            pass
                    elif "files" in exp_name:
                        try:
                            n_files = int(exp_name.replace("files", ""))
                        except ValueError:
                            pass

                    # Create name
                    name_parts = [model_type.value, genre, method.value]
                    if n_files:
                        name_parts.append(f"{n_files}f")
                    if fold is not None:
                        name_parts.append(f"f{fold}")
                    name = "-".join(name_parts)

                    # Skip if already registered
                    if _registry.get(name):
                        continue

                    # Load metadata if available
                    meta_file = exp_dir / "metadata.json"
                    metrics = {}
                    if meta_file.exists():
                        try:
                            import json
                            with open(meta_file) as f:
                                meta = json.load(f)
                            metrics = meta.get("metrics", {})
                        except Exception:
                            pass

                    _registry.register(ModelSpec(
                        name=name,
                        display_name=f"{model_type.value.upper()} {genre} {method.value}",
                        model_type=model_type,
                        training_method=method,
                        genre=genre,
                        checkpoint_path=str(ckpt),
                        n_files=n_files,
                        fold=fold,
                        metrics=metrics,
                    ))


# =============================================================================
# Public API
# =============================================================================

def load_model(
    name: str,
    device: str = "cpu",
    strict: bool = False,
    **model_kwargs,
) -> nn.Module:
    """
    Load a model by name from the registry.

    Args:
        name: Model name (e.g., "bocktcn-candombe-ft", "beatnet-brid-ft-f0")
        device: Device to load model to
        strict: Strict state dict loading
        **model_kwargs: Override model construction arguments

    Returns:
        Loaded model

    Example:
        >>> model = load_model("bocktcn-candombe-ft")
        >>> model = load_model("beatnet-pretrained", device="cuda")
    """
    # Import model classes here to avoid circular imports
    from mir_core.models import BockTCN, BeatNetBatch, BEAST
    from mir_core.models import GenreClassifier

    MODEL_CLASSES = {
        ModelType.BOCKTCN: BockTCN,
        ModelType.BEATNET: BeatNetBatch,
        ModelType.BEAST: BEAST,
        ModelType.GENRE_CLASSIFIER: GenreClassifier,
    }

    spec = _registry.get(name)
    if not spec:
        available = _registry.list_names()
        raise ValueError(
            f"Model '{name}' not found. Available models: {available[:10]}..."
            if len(available) > 10 else f"Model '{name}' not found. Available models: {available}"
        )

    # Get model class
    model_class = MODEL_CLASSES.get(spec.model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    # Merge kwargs
    kwargs = {**spec.model_kwargs, **model_kwargs}

    # Instantiate model
    model = model_class(**kwargs)

    # Load checkpoint if available
    if spec.checkpoint_path:
        ckpt_path = _registry.resolve_path(spec)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                # Remove 'model.' prefix if present
                state_dict = {
                    k.replace("model.", ""): v
                    for k, v in state_dict.items()
                }
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=strict)
        else:
            print(f"Warning: Checkpoint not found: {ckpt_path}")

    model = model.to(device)
    model.eval()

    return model


def list_models(
    model_type: Optional[str] = None,
    genre: Optional[str] = None,
    training_method: Optional[str] = None,
    verbose: bool = True,
) -> List[ModelSpec]:
    """
    List available models in the registry.

    Args:
        model_type: Filter by architecture ("bocktcn", "beatnet", "beast")
        genre: Filter by genre ("candombe", "brid", "salsa", etc.)
        training_method: Filter by method ("pretrained", "ft", "genre", "incr")
        verbose: Print the model list

    Returns:
        List of ModelSpec objects

    Example:
        >>> list_models(genre="candombe")
        >>> list_models(model_type="beatnet", training_method="ft")
    """
    # Convert strings to enums if needed
    mt = ModelType(model_type) if model_type else None
    tm = TrainingMethod(training_method) if training_method else None

    models = _registry.list_models(model_type=mt, genre=genre, training_method=tm)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Available Models ({len(models)} found)")
        print(f"{'='*60}")

        # Group by model type
        by_type: Dict[str, List[ModelSpec]] = {}
        for m in models:
            key = m.model_type.value
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(m)

        for mtype, specs in by_type.items():
            print(f"\n{mtype.upper()}")
            print("-" * 40)
            for s in specs:
                metrics_str = ""
                if s.metrics:
                    f1 = s.metrics.get("beat_f1", s.metrics.get("f_measure"))
                    if f1:
                        metrics_str = f" (F1={f1:.2%})"
                print(f"  {s.name:<35} {s.description or ''}{metrics_str}")

        print(f"\n{'='*60}\n")

    return models


def register_model(
    name: str,
    checkpoint_path: str,
    model_type: str = "bocktcn",
    training_method: str = "ft",
    genre: Optional[str] = None,
    description: str = "",
    metrics: Optional[Dict[str, float]] = None,
    **kwargs,
) -> None:
    """
    Register a custom model in the registry.

    Args:
        name: Unique model name
        checkpoint_path: Path to checkpoint file
        model_type: Architecture type ("bocktcn", "beatnet", "beast")
        training_method: Training method ("pretrained", "ft", "genre", "incr")
        genre: Target genre
        description: Human readable description
        metrics: Performance metrics dict
        **kwargs: Additional model kwargs

    Example:
        >>> register_model(
        ...     name="bocktcn-salsa-custom",
        ...     checkpoint_path="/path/to/checkpoint.ckpt",
        ...     genre="salsa",
        ...     description="Custom salsa fine-tuned model",
        ...     metrics={"beat_f1": 0.85}
        ... )
    """
    spec = ModelSpec(
        name=name,
        display_name=name.replace("-", " ").title(),
        model_type=ModelType(model_type),
        training_method=TrainingMethod(training_method),
        genre=genre,
        checkpoint_path=checkpoint_path,
        description=description,
        metrics=metrics or {},
        model_kwargs=kwargs,
    )
    _registry.register(spec)


def get_model_info(name: str) -> Optional[ModelSpec]:
    """Get detailed information about a registered model."""
    return _registry.get(name)


def save_registry(path: Union[str, Path]) -> None:
    """Save the current registry to a JSON file."""
    path = Path(path)
    data = {}
    for name, spec in _registry._models.items():
        data[name] = {
            "name": spec.name,
            "display_name": spec.display_name,
            "model_type": spec.model_type.value,
            "training_method": spec.training_method.value,
            "genre": spec.genre,
            "checkpoint_path": spec.checkpoint_path,
            "description": spec.description,
            "metrics": spec.metrics,
            "model_kwargs": spec.model_kwargs,
            "n_files": spec.n_files,
            "fold": spec.fold,
        }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_registry(path: Union[str, Path]) -> None:
    """Load registry from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    for name, spec_dict in data.items():
        spec_dict["model_type"] = ModelType(spec_dict["model_type"])
        spec_dict["training_method"] = TrainingMethod(spec_dict["training_method"])
        spec = ModelSpec(**spec_dict)
        _registry.register(spec)


# Auto-discover on import
_auto_discover_checkpoints()
