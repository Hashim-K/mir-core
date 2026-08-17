from __future__ import annotations

import threading

import pytest
import torch

from mir_core.runtime import ConcurrentJobBank


def test_cpu_bank_runs_lanes_concurrently_and_preserves_label_order() -> None:
    labels = ("candombe", "salsa", "stock")
    rendezvous = threading.Barrier(len(labels), timeout=2.0)
    thread_ids: dict[str, int] = {}
    initialized: set[int] = set()

    def initialize() -> None:
        initialized.add(threading.get_ident())

    def job(label: str) -> str:
        thread_ids[label] = threading.get_ident()
        rendezvous.wait()
        return label.upper()

    with ConcurrentJobBank(
        labels,
        device="cpu",
        thread_initializer=initialize,
    ) as bank:
        result = bank.run({label: lambda label=label: job(label) for label in labels})

    assert tuple(result.outputs) == labels
    assert result.outputs == {
        "candombe": "CANDOMBE",
        "salsa": "SALSA",
        "stock": "STOCK",
    }
    assert len(set(thread_ids.values())) == len(labels)
    assert initialized == set(thread_ids.values())
    assert result.execution_mode == "cpu_threads"
    assert all(value >= 0.0 for value in result.lane_seconds.values())
    assert all(value >= 0.0 for value in result.completion_seconds.values())


def test_bank_rejects_reordered_or_incomplete_jobs() -> None:
    with ConcurrentJobBank(("first", "second"), device="cpu") as bank:
        with pytest.raises(ValueError, match="labels/order"):
            bank.run({"second": lambda: 2, "first": lambda: 1})
        with pytest.raises(ValueError, match="labels/order"):
            bank.run({"first": lambda: 1})


def test_bank_rejects_boolean_cuda_priority() -> None:
    with pytest.raises(TypeError, match="cuda_priority"):
        ConcurrentJobBank(("only",), device="cpu", cuda_priority=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bank_uses_distinct_high_priority_streams_per_lane() -> None:
    labels = ("one", "two", "three")
    with ConcurrentJobBank(labels, device="cuda", cuda_priority=-1) as bank:
        result = bank.run(
            {
                label: lambda label=label: (
                    torch.cuda.current_stream().cuda_stream,
                    torch.tensor(float(len(label)), device="cuda") * 2.0,
                )
                for label in labels
            }
        )

    stream_ids = [result.outputs[label][0] for label in labels]
    assert len(set(stream_ids)) == len(labels)
    assert {stream.priority for stream in bank._streams.values()} == {-1}
    assert result.execution_mode == "cuda_streams"
    assert [result.outputs[label][1].item() for label in labels] == [6.0, 6.0, 10.0]
    assert all(value >= 0.0 for value in result.lane_seconds.values())
    assert all(value >= 0.0 for value in result.completion_seconds.values())
