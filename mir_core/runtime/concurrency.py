"""Concurrent execution for independent, stateful real-time lanes.

Each label owns a persistent worker lane.  CUDA lanes use distinct streams so
separately weighted models are not accidentally ordered on PyTorch's default
stream.  CPU lanes use persistent single-worker threads, preserving submission
order for stateful models and postprocessors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
import threading
import time
from types import MappingProxyType
from typing import Generic, TypeVar

import torch


T = TypeVar("T")


@dataclass(frozen=True)
class ConcurrentJobBankResult(Generic[T]):
    """Outputs and observed readiness for one bank dispatch."""

    outputs: Mapping[str, T]
    lane_seconds: Mapping[str, float]
    completion_seconds: Mapping[str, float]
    wall_seconds: float
    execution_mode: str


class ConcurrentJobBank:
    """Run one independent callable per label on persistent execution lanes.

    A bank dispatch is deliberately synchronous at its public boundary: all
    outputs are valid when :meth:`run` returns.  The per-label completion times
    retain when each lane became ready, while ``wall_seconds`` records the
    all-lane join.  Callables submitted to a CUDA bank must enqueue their Torch
    work on the current stream and return without changing streams themselves.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        device: str | torch.device,
        thread_name_prefix: str = "mir-lane",
        thread_initializer: Callable[[], None] | None = None,
        cuda_priority: int = 0,
    ) -> None:
        normalized = tuple(str(label) for label in labels)
        if not normalized or any(not label for label in normalized):
            raise ValueError("ConcurrentJobBank requires non-empty labels.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("ConcurrentJobBank labels must be unique.")

        self.labels = normalized
        self.device = torch.device(device)
        if isinstance(cuda_priority, bool) or not isinstance(cuda_priority, int):
            raise TypeError("cuda_priority must be an integer.")
        self.cuda_priority = int(cuda_priority)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but CUDA is unavailable.")
        self._lock = threading.Lock()
        self._closed = False
        self._executor: ThreadPoolExecutor | None = None
        self._streams: dict[str, torch.cuda.Stream] = {}
        if self.device.type == "cuda":
            self._streams = {
                label: torch.cuda.Stream(
                    device=self.device,
                    priority=self.cuda_priority,
                )
                for label in self.labels
            }
        elif len(self.labels) > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=len(self.labels),
                thread_name_prefix=thread_name_prefix,
                initializer=thread_initializer,
            )

    @property
    def execution_mode(self) -> str:
        if self.device.type == "cuda":
            return "cuda_streams"
        if self._executor is not None:
            return "cpu_threads"
        return "inline"

    def _validate_jobs(self, jobs: Mapping[str, Callable[[], T]]) -> None:
        if tuple(jobs) != self.labels:
            raise ValueError(
                "Job labels/order must exactly match the bank: "
                f"expected={self.labels!r}, got={tuple(jobs)!r}."
            )
        if any(not callable(job) for job in jobs.values()):
            raise TypeError("Every bank job must be callable.")

    def run(
        self,
        jobs: Mapping[str, Callable[[], T]],
    ) -> ConcurrentJobBankResult[T]:
        """Dispatch all jobs and return after every lane is complete."""

        self._validate_jobs(jobs)
        with self._lock:
            if self._closed:
                raise RuntimeError("ConcurrentJobBank is closed.")
            if self.device.type == "cuda":
                return self._run_cuda(jobs)
            return self._run_cpu(jobs)

    def _run_cpu(
        self,
        jobs: Mapping[str, Callable[[], T]],
    ) -> ConcurrentJobBankResult[T]:
        dispatch_started = time.perf_counter()

        def timed(job: Callable[[], T]) -> tuple[T, float, float]:
            lane_started = time.perf_counter()
            output = job()
            lane_ended = time.perf_counter()
            return output, lane_ended - lane_started, lane_ended - dispatch_started

        if self._executor is None:
            output, lane_seconds, completion_seconds = timed(jobs[self.labels[0]])
            outputs = {self.labels[0]: output}
            lanes = {self.labels[0]: lane_seconds}
            completions = {self.labels[0]: completion_seconds}
        else:
            futures: dict[str, Future[tuple[T, float, float]]] = {
                label: self._executor.submit(timed, jobs[label])
                for label in self.labels
            }
            wait(tuple(futures.values()))
            outputs = {}
            lanes = {}
            completions = {}
            for label in self.labels:
                output, lane_seconds, completion_seconds = futures[label].result()
                outputs[label] = output
                lanes[label] = lane_seconds
                completions[label] = completion_seconds

        wall_seconds = time.perf_counter() - dispatch_started
        return ConcurrentJobBankResult(
            outputs=MappingProxyType(outputs),
            lane_seconds=MappingProxyType(lanes),
            completion_seconds=MappingProxyType(completions),
            wall_seconds=wall_seconds,
            execution_mode=self.execution_mode,
        )

    def _run_cuda(
        self,
        jobs: Mapping[str, Callable[[], T]],
    ) -> ConcurrentJobBankResult[T]:
        dispatch_started = time.perf_counter()
        caller_stream = torch.cuda.current_stream(self.device)
        origin = torch.cuda.Event(enable_timing=True)
        origin.record(caller_stream)
        starts = {
            label: torch.cuda.Event(enable_timing=True) for label in self.labels
        }
        ends = {
            label: torch.cuda.Event(enable_timing=True) for label in self.labels
        }
        outputs: dict[str, T] = {}
        try:
            for label in self.labels:
                stream = self._streams[label]
                stream.wait_event(origin)
                with torch.cuda.stream(stream):
                    starts[label].record(stream)
                    outputs[label] = jobs[label]()
                    ends[label].record(stream)
            for label in self.labels:
                ends[label].synchronize()
        except BaseException:
            # Finish any work already enqueued before allowing stateful lanes to
            # be reused after the caller handles the exception.
            torch.cuda.synchronize(self.device)
            raise

        wall_seconds = time.perf_counter() - dispatch_started
        lanes = {
            label: max(0.0, starts[label].elapsed_time(ends[label]) / 1000.0)
            for label in self.labels
        }
        completions = {
            label: max(0.0, origin.elapsed_time(ends[label]) / 1000.0)
            for label in self.labels
        }
        if not math.isfinite(wall_seconds):  # defensive parity with timings
            raise RuntimeError("Concurrent CUDA dispatch produced invalid timing.")
        return ConcurrentJobBankResult(
            outputs=MappingProxyType(outputs),
            lane_seconds=MappingProxyType(lanes),
            completion_seconds=MappingProxyType(completions),
            wall_seconds=wall_seconds,
            execution_mode=self.execution_mode,
        )

    def close(self) -> None:
        """Drain and release persistent CPU workers."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "ConcurrentJobBank":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["ConcurrentJobBank", "ConcurrentJobBankResult"]
