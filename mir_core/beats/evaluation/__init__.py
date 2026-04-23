# mir_core/beats/evaluation/__init__.py
"""Beat tracking evaluation metrics.

Re-exports from mir_core.evaluation.metrics. Import from here for
task-scoped imports:

    from mir_core.beats.evaluation import compute_beat_metrics
"""
import importlib.util
import pathlib

_metrics_path = (
    pathlib.Path(__file__).parent.parent.parent  # mir_core/
    / "evaluation"
    / "metrics.py"
)
_spec = importlib.util.spec_from_file_location(
    "mir_core._evaluation_metrics_direct", _metrics_path
)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"Cannot locate mir_core.evaluation.metrics at {_metrics_path}. "
        "Has the package layout changed?"
    )
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_beat_metrics = _mod.compute_beat_metrics
compute_downbeat_metrics = _mod.compute_downbeat_metrics
evaluate_beats = _mod.evaluate_beats
evaluate_downbeats = _mod.evaluate_downbeats
compute_per_track_metrics = _mod.compute_per_track_metrics
compute_ibi_stats = _mod.compute_ibi_stats

__all__ = [
    "compute_beat_metrics",
    "compute_downbeat_metrics",
    "evaluate_beats",
    "evaluate_downbeats",
    "compute_per_track_metrics",
    "compute_ibi_stats",
]
