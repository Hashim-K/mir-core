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

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
import re
import torch
import torch.nn as nn

from mir_core.utils.hashing import stable_digest


class ModelType(str, Enum):
    """Supported model architectures."""

    BOCKTCN = "bocktcn"
    BEATNET = "beatnet"
    BEAST = "beast"
    GENRE_CLASSIFIER = "genre_classifier"


class TrainingMethod(str, Enum):
    """Training/adaptation method."""

    PRETRAINED = "pretrained"  # Original pretrained weights
    FINE_TUNING = "ft"  # Fine-tuned on target genre
    GENRE_ONLY = "genre"  # Trained on genre from scratch
    INCREMENTAL = "incr"  # Incrementally trained with N files


@dataclass
class ModelSpec:
    """Specification for a registered model."""

    name: str  # Short name (e.g., "bocktcn-candombe-ft")
    display_name: str  # Human readable name
    model_type: ModelType  # Architecture type
    training_method: TrainingMethod  # How it was trained
    genre: Optional[str] = None  # Target genre (candombe, brid, salsa, etc.)
    checkpoint_path: Optional[str] = None  # Path to checkpoint
    description: str = ""  # Model description
    metrics: Dict[str, float] = field(default_factory=dict)  # Performance metrics
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # Model init args
    n_files: Optional[int] = None  # For incremental: number of training files
    fold: Optional[int] = None  # For CV: fold number
    metadata: Dict[str, Any] = field(default_factory=dict)  # Checkpoint provenance

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
        """Get a model by canonical name or an unambiguous legacy alias."""

        direct = self._models.get(name)
        if direct is not None:
            return direct
        alias_matches = [
            spec
            for spec in self._models.values()
            if spec.metadata.get("legacy_alias") == name
        ]
        return alias_matches[0] if len(alias_matches) == 1 else None

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


def _safe_load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """Load tensor/basic metadata without allowing discovery failures to escape."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ):
        return {}
    return checkpoint if isinstance(checkpoint, dict) else {}


def _classifier_kwargs_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the classifier trainer's checkpoint metadata into model kwargs."""

    model_config = checkpoint.get("model_config", {})
    kwargs = dict(model_config) if isinstance(model_config, dict) else {}
    kwargs.pop("name", None)
    kwargs.pop("num_classes", None)
    kwargs.pop("genre_labels", None)
    config_arch = kwargs.pop("arch", None)
    arch = checkpoint.get("arch") or config_arch
    if isinstance(arch, str) and arch:
        kwargs["arch"] = arch
    labels = checkpoint.get("labels")
    if (
        isinstance(labels, (list, tuple))
        and labels
        and all(isinstance(label, str) and label for label in labels)
    ):
        kwargs["genre_labels"] = list(labels)
        kwargs["num_classes"] = len(labels)
    calibration_temperature = _classifier_calibration_temperature(checkpoint)
    if calibration_temperature is not None:
        kwargs["calibration_temperature"] = calibration_temperature
    return kwargs


def _mapping_copy(value: Any) -> Dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _metadata_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def _classifier_calibration_temperature(
    checkpoint: Mapping[str, Any],
) -> Any | None:
    calibration = checkpoint.get("calibration")
    if isinstance(calibration, Mapping) and calibration.get("temperature") is not None:
        temperature = _metadata_scalar(calibration["temperature"])
        if temperature is not None:
            return temperature
    router_config = checkpoint.get("router_config")
    if (
        isinstance(router_config, Mapping)
        and router_config.get("temperature") is not None
    ):
        return _metadata_scalar(router_config["temperature"])
    return None


def _classifier_routing_metadata(
    checkpoint: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return rich calibration provenance and router configuration."""

    calibration = _mapping_copy(checkpoint.get("calibration"))
    router_config = _mapping_copy(checkpoint.get("router_config"))
    if (
        "temperature" not in calibration
        and router_config.get("temperature") is not None
    ):
        calibration["temperature"] = deepcopy(router_config["temperature"])
    if (
        "confidence_threshold" not in calibration
        and router_config.get("confidence_threshold") is not None
    ):
        calibration["confidence_threshold"] = deepcopy(
            router_config["confidence_threshold"]
        )
    return calibration, router_config


def _fold_from_classifier_checkpoint(
    checkpoint: Dict[str, Any],
    relative_path: Path,
) -> Optional[int]:
    split_contract = checkpoint.get("split_contract")
    if isinstance(split_contract, dict):
        fold = split_contract.get("fold_index")
        if isinstance(fold, int) and not isinstance(fold, bool) and fold >= 0:
            return fold
    for part in relative_path.parts:
        match = re.fullmatch(r"fold[-_]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _model_name_component(value: Any) -> str:
    component = re.sub(r"[^a-z0-9_]+", "-", str(value).strip().lower())
    return component.strip("-_")


def _classifier_discovery_names(
    *,
    arch: str,
    checkpoint: Mapping[str, Any],
    fold: Optional[int],
    relative_path: Path,
) -> tuple[str, str]:
    """Return collision-resistant canonical name and the old short alias."""

    legacy_parts = ["genre_classifier", arch]
    if fold is not None:
        legacy_parts.append(f"f{fold}")
    legacy_name = "-".join(legacy_parts)

    canonical_parts = ["genre_classifier", _model_name_component(arch)]
    feature_config = checkpoint.get("feature_config")
    feature_type = (
        feature_config.get("type") if isinstance(feature_config, Mapping) else None
    )
    if feature_type:
        feature_component = _model_name_component(feature_type)
        if feature_component:
            canonical_parts.append(feature_component)
    experiment_hash = checkpoint.get("experiment_hash")
    if experiment_hash:
        experiment_component = _model_name_component(experiment_hash)
        if experiment_component:
            canonical_parts.append(experiment_component)
        else:
            canonical_parts.append(
                f"ckpt-{stable_digest(relative_path.as_posix(), length=8)}"
            )
    else:
        canonical_parts.append(
            f"ckpt-{stable_digest(relative_path.as_posix(), length=8)}"
        )
    if fold is not None:
        canonical_parts.append(f"f{fold}")
    return "-".join(canonical_parts), legacy_name


def _unique_classifier_name(name: str, checkpoint_path: Path) -> str:
    existing = _registry._models.get(name)
    if existing is None:
        return name
    if (
        existing.checkpoint_path is not None
        and Path(existing.checkpoint_path).resolve() == checkpoint_path.resolve()
    ):
        return name
    path_suffix = stable_digest(str(checkpoint_path.resolve()), length=8)
    candidate = f"{name}-p{path_suffix}"
    counter = 2
    while candidate in _registry._models:
        candidate = f"{name}-p{path_suffix}-{counter}"
        counter += 1
    return candidate


def _discover_genre_classifiers(base_path: Path) -> None:
    """Discover legacy and current classifier-training checkpoints.

    Supported structures include ``{arch}/fold{n}/best-*.ckpt`` and the
    classifier trainer's ``.../fold{n}/checkpoints/best.pt``. Trainer metadata
    is used to reconstruct architecture kwargs and genre labels.
    """

    if not base_path.is_dir():
        return
    checkpoints = set(base_path.rglob("best-*.ckpt"))
    checkpoints.update(
        path for path in base_path.rglob("best.pt") if path.parent.name == "checkpoints"
    )
    if base_path.name == "checkpoints" and (base_path / "best.pt").is_file():
        checkpoints.add(base_path / "best.pt")
    for ckpt in sorted(checkpoints):
        relative_path = ckpt.relative_to(base_path)
        checkpoint = _safe_load_checkpoint_metadata(ckpt)
        model_kwargs = _classifier_kwargs_from_checkpoint(checkpoint)
        arch = model_kwargs.get("arch")
        if not isinstance(arch, str) or not arch:
            # Legacy layout: the first directory below base_path names the arch.
            arch = relative_path.parts[0] if len(relative_path.parts) > 1 else None
            if not arch:
                continue
            model_kwargs["arch"] = arch
        fold = _fold_from_classifier_checkpoint(checkpoint, relative_path)

        name, legacy_alias = _classifier_discovery_names(
            arch=arch,
            checkpoint=checkpoint,
            fold=fold,
            relative_path=relative_path,
        )
        name = _unique_classifier_name(name, ckpt)
        existing = _registry._models.get(name)
        if existing is not None and existing.checkpoint_path == str(ckpt):
            continue

        calibrated_metrics = checkpoint.get("validation_metrics_calibrated")
        if isinstance(calibrated_metrics, Mapping):
            validation_metrics = calibrated_metrics
            validation_metrics_source = "validation_metrics_calibrated"
        else:
            validation_metrics = checkpoint.get("validation_metrics", {})
            validation_metrics_source = "validation_metrics"
        metrics = (
            {
                str(key): float(value)
                for key, value in validation_metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if isinstance(validation_metrics, Mapping)
            else {}
        )
        calibration_metadata, router_config = _classifier_routing_metadata(checkpoint)
        selection_metadata = {
            key: deepcopy(checkpoint[key])
            for key in ("epoch", "best_metric", "best_value")
            if checkpoint.get(key) is not None
        }
        if isinstance(checkpoint.get("validation_metrics"), Mapping):
            selection_metadata["validation_metrics"] = deepcopy(
                dict(checkpoint["validation_metrics"])
            )
        fold_suffix = f" fold{fold}" if fold is not None else ""
        _registry.register(
            ModelSpec(
                name=name,
                display_name=f"Genre Classifier ({arch}){fold_suffix}",
                model_type=ModelType.GENRE_CLASSIFIER,
                training_method=TrainingMethod.FINE_TUNING,
                checkpoint_path=str(ckpt),
                description=(
                    f"Classifier experiment {checkpoint['experiment_hash']}"
                    if checkpoint.get("experiment_hash")
                    else ""
                ),
                metrics=metrics,
                metadata={
                    "calibration": calibration_metadata,
                    "experiment_hash": checkpoint.get("experiment_hash"),
                    "feature_type": (
                        checkpoint.get("feature_config", {}).get("type")
                        if isinstance(checkpoint.get("feature_config"), Mapping)
                        else None
                    ),
                    "legacy_alias": legacy_alias,
                    "router_config": router_config,
                    "selection": selection_metadata,
                    "validation_metrics_source": validation_metrics_source,
                },
                fold=fold,
                model_kwargs=model_kwargs,
            )
        )


def _auto_discover_checkpoints() -> None:
    """Auto-discover and register checkpoints from standard locations."""
    checkpoints_path = _get_default_checkpoints_path()
    weights_path = _get_package_root() / "weights"

    # Register baseline/pretrained models
    if weights_path.exists():
        bocktcn_ckpt = weights_path / "bocktcn.ckpt"
        if bocktcn_ckpt.exists():
            _registry.register(
                ModelSpec(
                    name="bocktcn-pretrained",
                    display_name="BockTCN (Pretrained)",
                    model_type=ModelType.BOCKTCN,
                    training_method=TrainingMethod.PRETRAINED,
                    checkpoint_path=str(bocktcn_ckpt),
                    description="Original BockTCN trained on GTZAN",
                )
            )

        tcn_lamir = weights_path / "tcn_lamir_pretrained.ckpt"
        if tcn_lamir.exists():
            _registry.register(
                ModelSpec(
                    name="bocktcn-lamir-pretrained",
                    display_name="BockTCN LAMIR (Pretrained)",
                    model_type=ModelType.BOCKTCN,
                    training_method=TrainingMethod.PRETRAINED,
                    checkpoint_path=str(tcn_lamir),
                    description="BockTCN pretrained on Latin American music",
                )
            )

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

            _registry.register(
                ModelSpec(
                    name=short_name,
                    display_name=f"{model_type.value.upper()} {genre or ''} {method.value}",
                    model_type=model_type,
                    training_method=method,
                    genre=genre,
                    checkpoint_path=str(ckpt),
                    n_files=n_files,
                    fold=fold,
                )
            )

    # Discover genre classifier checkpoints
    genre_clf_path = (
        checkpoints_path / "genre_classifier" if checkpoints_path.exists() else None
    )
    if genre_clf_path and genre_clf_path.exists():
        _discover_genre_classifiers(genre_clf_path)

    # Also look in results/genre_classifier/checkpoints
    results_clf = (
        _get_package_root().parent / "results" / "genre_classifier" / "checkpoints"
    )
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

                    _registry.register(
                        ModelSpec(
                            name=name,
                            display_name=f"{model_type.value.upper()} {genre} {method.value}",
                            model_type=model_type,
                            training_method=method,
                            genre=genre,
                            checkpoint_path=str(ckpt),
                            n_files=n_files,
                            fold=fold,
                            metrics=metrics,
                        )
                    )


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
            if len(available) > 10
            else f"Model '{name}' not found. Available models: {available}"
        )

    # Get model class
    model_class = MODEL_CLASSES.get(spec.model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    checkpoint = None
    if spec.checkpoint_path:
        ckpt_path = _registry.resolve_path(spec)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        else:
            print(f"Warning: Checkpoint not found: {ckpt_path}")

    # Current classifier checkpoints carry the exact constructor contract.
    checkpoint_kwargs: Dict[str, Any] = {}
    if spec.model_type == ModelType.GENRE_CLASSIFIER and isinstance(checkpoint, dict):
        checkpoint_kwargs = _classifier_kwargs_from_checkpoint(checkpoint)

    # Explicit call kwargs override registry metadata, which overrides checkpoint
    # metadata inferred during loading.
    kwargs = {**checkpoint_kwargs, **spec.model_kwargs, **model_kwargs}
    model = model_class(**kwargs)
    if spec.model_type == ModelType.GENRE_CLASSIFIER and isinstance(checkpoint, dict):
        calibration_metadata, router_config = _classifier_routing_metadata(checkpoint)
        model.set_routing_metadata(
            calibration=calibration_metadata,
            router_config=router_config,
        )

    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format for model: {name}")
        if "model_state_dict" in checkpoint:
            # Native classifier trainer format; keys already target the wrapper.
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Lightning modules commonly wrap the exported network as ``model``.
            state_dict = {
                key.removeprefix("model."): value for key, value in state_dict.items()
            }
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint state for model {name!r} is not a mapping.")
        model.load_state_dict(state_dict, strict=strict)

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
            "metadata": spec.metadata,
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
