"""
Comprehensive beat tracking evaluator with per-fold CV support.

Classes:
    BeatEvaluator — stateful evaluator with per-fold CV, per-track tables,
                    worst/best track analysis, and text report generation.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

from .metrics import compute_beat_metrics, compute_downbeat_metrics, compute_ibi_stats


class BeatEvaluator:
    """
    Comprehensive beat tracking evaluation class.

    Wraps mir_eval functionality and adds:
    - Per-track and aggregate metrics
    - Cross-validation fold support
    - Downbeat evaluation
    - Report generation
    - Comparison between models/conditions

    Following evaluation methodology from:
    - Rapini & Jordanous (2024): F1, CMLt, AMLt at 70ms tolerance
    - Maia et al. (2023): F-measure comparison for Latin American music

    Attributes:
        fps: Frames per second for activation processing
        tolerance: Evaluation tolerance in seconds (default 70ms)
        results: Dictionary storing per-track results
    """

    def __init__(
        self,
        fps: int = 100,
        tolerance: float = 0.07,
        name: str = "BeatEvaluator",
    ):
        self.fps = fps
        self.tolerance = tolerance
        self.name = name

        # Results storage
        self.results: Dict[str, Dict[str, Any]] = {}
        self.fold_results: Dict[int, Dict[str, Dict[str, Any]]] = {}

    def reset(self):
        """Clear all stored results."""
        self.results = {}
        self.fold_results = {}

    def add_result(
        self,
        track_id: str,
        beats_pred: np.ndarray,
        beats_ann: np.ndarray,
        activations: Optional[np.ndarray] = None,
        downbeats_pred: Optional[np.ndarray] = None,
        downbeats_ann: Optional[np.ndarray] = None,
        fold: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a single track's results.

        Args:
            track_id: Unique identifier for the track
            beats_pred: Predicted beat times in seconds
            beats_ann: Annotated beat times in seconds
            activations: Optional beat activation function
            downbeats_pred: Optional predicted downbeat times
            downbeats_ann: Optional annotated downbeat times
            fold: Optional fold number for cross-validation
            metadata: Optional additional metadata
        """
        result = {
            "beats_pred": np.asarray(beats_pred),
            "beats_ann": np.asarray(beats_ann),
            "activations": activations,
            "downbeats_pred": downbeats_pred,
            "downbeats_ann": downbeats_ann,
            "metadata": metadata or {},
        }

        # Compute metrics immediately
        result["metrics"] = compute_beat_metrics(
            result["beats_pred"],
            result["beats_ann"],
            self.tolerance
        )

        # Add downbeat metrics if available
        if downbeats_pred is not None and downbeats_ann is not None:
            result["metrics"].update(compute_downbeat_metrics(
                np.asarray(downbeats_pred),
                np.asarray(downbeats_ann),
                self.tolerance
            ))

        # Store in appropriate location
        if fold is not None:
            if fold not in self.fold_results:
                self.fold_results[fold] = {}
            self.fold_results[fold][track_id] = result
        else:
            self.results[track_id] = result

    def get_metrics_dataframe(self) -> "pd.DataFrame":
        """
        Get per-track metrics as a pandas DataFrame.

        Returns:
            DataFrame with track_id as index and metrics as columns
        """
        import pandas as pd

        rows = []
        for track_id, result in self.results.items():
            row = {"track_id": track_id}
            row.update(result["metrics"])
            rows.append(row)

        return pd.DataFrame(rows).set_index("track_id")

    def compute_aggregate_metrics(
        self,
        metric_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics across all tracks.

        Args:
            metric_keys: Specific metrics to aggregate (default: all)

        Returns:
            Dictionary with mean and std for each metric
        """
        if not self.results:
            return {}

        all_metrics = defaultdict(list)
        for result in self.results.values():
            for key, value in result["metrics"].items():
                if key not in ["num_pred", "num_ann"]:
                    if metric_keys is None or key in metric_keys:
                        all_metrics[key].append(value)

        aggregate = {"num_tracks": len(self.results)}
        for key, values in all_metrics.items():
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_std"] = float(np.std(values))

        return aggregate

    def compute_cv_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute cross-validation metrics across all folds.

        Returns:
            Dictionary with per-fold and overall metrics
        """
        if not self.fold_results:
            return {}

        cv_metrics = {}
        all_folds_metrics = defaultdict(list)

        for fold, fold_data in self.fold_results.items():
            fold_metrics = defaultdict(list)

            for result in fold_data.values():
                for key, value in result["metrics"].items():
                    if key not in ["num_pred", "num_ann"]:
                        fold_metrics[key].append(value)
                        all_folds_metrics[key].append(value)

            cv_metrics[f"fold_{fold}"] = {
                key: float(np.mean(values))
                for key, values in fold_metrics.items()
            }
            cv_metrics[f"fold_{fold}"].update({
                f"{key}_std": float(np.std(values))
                for key, values in fold_metrics.items()
            })
            cv_metrics[f"fold_{fold}"]["num_tracks"] = len(fold_data)

        # Overall CV metrics
        cv_metrics["overall"] = {
            f"{key}_mean": float(np.mean(values))
            for key, values in all_folds_metrics.items()
        }
        cv_metrics["overall"].update({
            f"{key}_std": float(np.std(values))
            for key, values in all_folds_metrics.items()
        })
        cv_metrics["overall"]["num_tracks"] = sum(
            len(fold_data) for fold_data in self.fold_results.values()
        )
        cv_metrics["overall"]["num_folds"] = len(self.fold_results)

        return cv_metrics

    def get_worst_tracks(
        self,
        n: int = 5,
        metric: str = "fmeasure"
    ) -> List[Tuple[str, float]]:
        """Get n worst performing tracks by specified metric."""
        tracks_metrics = [
            (tid, result["metrics"].get(metric, 0))
            for tid, result in self.results.items()
        ]
        sorted_tracks = sorted(tracks_metrics, key=lambda x: x[1])
        return sorted_tracks[:n]

    def get_best_tracks(
        self,
        n: int = 5,
        metric: str = "fmeasure"
    ) -> List[Tuple[str, float]]:
        """Get n best performing tracks by specified metric."""
        tracks_metrics = [
            (tid, result["metrics"].get(metric, 0))
            for tid, result in self.results.items()
        ]
        sorted_tracks = sorted(tracks_metrics, key=lambda x: x[1], reverse=True)
        return sorted_tracks[:n]

    def print_summary(
        self,
        show_cv: bool = False,
        show_per_track: bool = True,
        metrics: Optional[List[str]] = None,
    ):
        """
        Print formatted evaluation summary.

        Args:
            show_cv: Whether to show cross-validation breakdown
            show_per_track: Whether to show per-track results within each fold
            metrics: Specific metrics to display (default: key metrics)
        """
        if metrics is None:
            metrics = ["fmeasure", "cmlt", "amlt", "cemgil", "pscore"]

        print(f"\n{'='*60}")
        print(f" {self.name} Results")
        print(f"{'='*60}")

        if self.fold_results and show_cv:
            cv_metrics = self.compute_cv_metrics()

            print(f"\n Cross-Validation Results ({cv_metrics['overall']['num_folds']} folds)")
            print(f" {'-'*56}")

            for fold_key in sorted([k for k in cv_metrics if k.startswith("fold_")]):
                fold_data = cv_metrics[fold_key]
                fold_num = int(fold_key.split("_")[1])
                f1 = fold_data.get("fmeasure", 0)
                f1_std = fold_data.get("fmeasure_std", 0)
                cmlt = fold_data.get("cmlt", 0)
                amlt = fold_data.get("amlt", 0)
                print(f"   Fold {fold_num}: F1={f1:.4f} +/- {f1_std:.4f}, CMLt={cmlt:.4f}, AMLt={amlt:.4f} (n={fold_data['num_tracks']})")

                if show_per_track and fold_num in self.fold_results:
                    per_track = [
                        (tid, res["metrics"].get("fmeasure", 0))
                        for tid, res in self.fold_results[fold_num].items()
                    ]
                    per_track.sort(key=lambda x: x[1])
                    for tid, score in per_track:
                        marker = " x" if score < 0.5 else ""
                        print(f"       track {tid:>4s}: F1={score:.4f}{marker}")

            print(f" {'-'*56}")
            overall = cv_metrics["overall"]
            print(f" Overall ({overall['num_tracks']} tracks):")
            for m in metrics:
                mean_key = f"{m}_mean"
                std_key = f"{m}_std"
                if mean_key in overall:
                    print(f"   {m.upper():12s}: {overall[mean_key]:.4f} +/- {overall[std_key]:.4f}")

            # Per-track summary across all folds
            # each track appears in exactly 1 test fold in CV
            all_track_info = {}  # tid -> (fold, f1)
            fold_means = {}
            for fold_num_key, fold_data in self.fold_results.items():
                fold_scores = [res["metrics"].get("fmeasure", 0) for res in fold_data.values()]
                fold_means[fold_num_key] = float(np.mean(fold_scores)) if fold_scores else 0.0
                for tid, res in fold_data.items():
                    all_track_info[tid] = (fold_num_key, res["metrics"].get("fmeasure", 0))

            if all_track_info:
                scores_arr = np.array([v[1] for v in all_track_info.values()])
                p1  = float(np.percentile(scores_arr, 1))
                p99 = float(np.percentile(scores_arr, 99))
                n_fail = int((scores_arr < 0.5).sum())

                print(f"\n Per-Track F1 Summary (all {len(scores_arr)} tracks):")
                print(f"   mean={scores_arr.mean():.4f}  std={scores_arr.std():.4f}  "
                      f"min={scores_arr.min():.4f}  max={scores_arr.max():.4f}  "
                      f"p1={p1:.4f}  p99={p99:.4f}  fails(<0.5): {n_fail}/{len(scores_arr)}")

                # Table sorted by track ID
                print(f"\n {'Track':>6}  {'Fold':>4}  {'F1':>6}  {'vs fold avg':>10}")
                print(f" {'-'*34}")
                try:
                    sorted_tids = sorted(all_track_info.keys(), key=lambda x: int(x))
                except (ValueError, TypeError):
                    sorted_tids = sorted(all_track_info.keys())
                for tid in sorted_tids:
                    fold_k, score = all_track_info[tid]
                    delta = score - fold_means[fold_k]
                    marker = " x" if score < 0.5 else ""
                    delta_str = f"{delta:+.3f}"
                    print(f"   {tid:>4s}    {fold_k:>3}   {score:.4f}   {delta_str:>8}{marker}")

                # Worst tracks block
                worst = sorted(all_track_info.items(), key=lambda x: x[1][1])[:10]
                print(f"\n Worst 10 tracks:")
                print(f" {'Track':>6}  {'Fold':>4}  {'F1':>6}  {'vs fold avg':>10}")
                print(f" {'-'*34}")
                for tid, (fold_k, score) in worst:
                    delta = score - fold_means[fold_k]
                    delta_str = f"{delta:+.3f}"
                    print(f"   {tid:>4s}    {fold_k:>3}   {score:.4f}   {delta_str:>8}  x outlier" if delta < -0.2 else
                          f"   {tid:>4s}    {fold_k:>3}   {score:.4f}   {delta_str:>8}  (hard fold)")
        else:
            aggregate = self.compute_aggregate_metrics()
            print(f" Total tracks: {aggregate.get('num_tracks', 0)}")
            print(f" {'-'*56}")

            for m in metrics:
                mean_key = f"{m}_mean"
                std_key = f"{m}_std"
                if mean_key in aggregate:
                    print(f"   {m.upper():12s}: {aggregate[mean_key]:.4f} +/- {aggregate[std_key]:.4f}")

            if show_per_track and self.results:
                print(f"\n Per-Track Results (sorted by F1):")
                print(f" {'-'*56}")
                per_track = [
                    (tid, res["metrics"].get("fmeasure", 0))
                    for tid, res in self.results.items()
                ]
                per_track.sort(key=lambda x: x[1])
                for tid, score in per_track:
                    marker = " x" if score < 0.5 else ""
                    print(f"   track {tid:>4s}: F1={score:.4f}{marker}")

        self.print_ibi_summary()
        print(f"{'='*60}\n")

    def generate_report(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a detailed text report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        lines = []
        lines.append(f"Beat Tracking Evaluation Report: {self.name}")
        lines.append("=" * 60)
        lines.append(f"Tolerance: {self.tolerance * 1000:.0f}ms")
        lines.append("")

        aggregate = self.compute_aggregate_metrics()
        lines.append(f"Aggregate Results ({aggregate.get('num_tracks', 0)} tracks)")
        lines.append("-" * 40)

        for key in ["fmeasure", "cmlt", "amlt", "cemgil", "pscore", "goto", "information_gain"]:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in aggregate:
                lines.append(f"  {key:20s}: {aggregate[mean_key]:.4f} +/- {aggregate[std_key]:.4f}")

        lines.append("")
        lines.append("Inter-Beat Interval Distribution (pooled)")
        lines.append("-" * 40)
        pred_stats = self.compute_pooled_ibi_stats(use_predictions=True)
        ann_stats  = self.compute_pooled_ibi_stats(use_predictions=False)
        for label, s in [("Predicted", pred_stats), ("Annotated", ann_stats)]:
            lines.append(
                f"  {label:10s}  "
                f"mean={s['ibi_mean']*1000:.1f}ms  "
                f"std={s['ibi_std']*1000:.1f}ms  "
                f"p95={s['ibi_p95']*1000:.1f}ms  "
                f"p99={s['ibi_p99']*1000:.1f}ms  "
                f"range={s['ibi_range']*1000:.1f}ms  "
                f"n={s['ibi_n']}"
            )

        lines.append("")
        lines.append("Best Tracks (F-measure)")
        lines.append("-" * 40)
        for track_id, score in self.get_best_tracks(5):
            lines.append(f"  {track_id}: {score:.4f}")

        lines.append("")
        lines.append("Worst Tracks (F-measure)")
        lines.append("-" * 40)
        for track_id, score in self.get_worst_tracks(5):
            lines.append(f"  {track_id}: {score:.4f}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)

        return report

    @property
    def predictions(self) -> Dict[str, np.ndarray]:
        """Get predictions dict for backward compatibility."""
        return {tid: r["beats_pred"] for tid, r in self.results.items()}

    @property
    def annotations(self) -> Dict[str, np.ndarray]:
        """Get annotations dict for backward compatibility."""
        return {tid: r["beats_ann"] for tid, r in self.results.items()}

    @property
    def activations(self) -> Dict[str, np.ndarray]:
        """Get activations dict for backward compatibility."""
        return {tid: r["activations"] for tid, r in self.results.items() if r["activations"] is not None}

    def compute_metrics(self) -> Dict[str, float]:
        """Backward compatible method."""
        return self.compute_aggregate_metrics()

    def get_per_track_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for each track."""
        return {tid: result["metrics"] for tid, result in self.results.items()}

    def compute_pooled_ibi_stats(
        self,
        use_predictions: bool = True,
        include_folds: bool = True,
    ) -> Dict[str, float]:
        """
        Compute IBI statistics pooled across *all* tracks (beats concatenated).

        Pooling gives a more representative distribution than averaging per-track
        stats, since short tracks contribute proportionally fewer intervals.

        Args:
            use_predictions: If True use predicted beats, else use annotations.
            include_folds: Also pull beats from fold_results.

        Returns:
            Dict with the same keys as :func:`compute_ibi_stats` (label="ibi").
        """
        all_beats: List[np.ndarray] = []

        key = "beats_pred" if use_predictions else "beats_ann"

        for result in self.results.values():
            b = result.get(key)
            if b is not None and len(b) > 0:
                all_beats.append(np.asarray(b))

        if include_folds:
            for fold_data in self.fold_results.values():
                for result in fold_data.values():
                    b = result.get(key)
                    if b is not None and len(b) > 0:
                        all_beats.append(np.asarray(b))

        if not all_beats:
            return compute_ibi_stats(np.array([]))

        pooled = np.sort(np.concatenate(all_beats))
        return compute_ibi_stats(pooled)

    def print_ibi_summary(self, unit_ms: bool = True) -> None:
        """Print inter-beat interval distribution summary for this evaluator."""
        u = "ms" if unit_ms else "s"
        scale = 1000.0 if unit_ms else 1.0

        pred_stats = self.compute_pooled_ibi_stats(use_predictions=True)
        ann_stats  = self.compute_pooled_ibi_stats(use_predictions=False)

        print(f"\n  Inter-Beat Interval Distribution (pooled across all tracks)")
        print(f"  {'':10s}  {'mean':>8}  {'std':>8}  {'p95':>8}  {'p99':>8}  {'range':>8}  {'n':>6}")
        print(f"  {'-'*64}")

        def _row(name, s):
            print(
                f"  {name:10s}  "
                f"{s['ibi_mean']*scale:>7.1f}{u}  "
                f"{s['ibi_std']*scale:>7.1f}{u}  "
                f"{s['ibi_p95']*scale:>7.1f}{u}  "
                f"{s['ibi_p99']*scale:>7.1f}{u}  "
                f"{s['ibi_range']*scale:>7.1f}{u}  "
                f"{s['ibi_n']:>6d}"
            )

        _row("Predicted", pred_stats)
        _row("Annotated", ann_stats)
