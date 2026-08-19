"""Timing primitives. Pure functions + dataclasses, no I/O.

Designed for both CPU and GPU work via the ``cuda_sync`` flag: when True we
call ``torch.cuda.synchronize()`` around each measurement so async kernel
launches don't get misattributed to the next iteration.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResult:
    """One scenario's wall-clock summary over multiple runs.

    All durations are in seconds. All fields are Python primitives so the
    dataclass serializes cleanly to JSON via ``dataclasses.asdict``.
    """

    name: str
    n_warmup: int
    n_runs: int
    median_s: float
    min_s: float
    max_s: float
    mean_s: float
    stdev_s: float

    def __str__(self) -> str:
        return (
            f"{self.name:<50s}  "
            f"median {self.median_s * 1000:8.2f} ms  "
            f"min {self.min_s * 1000:8.2f} ms  "
            f"stdev {self.stdev_s * 1000:7.2f} ms  "
            f"(n={self.n_runs})"
        )


def time_callable(
    name: str,
    fn: Callable[[], object],
    *,
    n_warmup: int = 1,
    n_runs: int = 5,
    cuda_sync: bool = False,
) -> BenchmarkResult:
    """Time ``fn`` with warmups; return summary statistics.

    Args:
        name: human-readable scenario name (shown in output, saved in JSON).
        fn: zero-argument callable to time. Return value is discarded.
        n_warmup: warmup iterations (not measured). Defaults to 1; use more
            if the first call materializes lazy weights or hits a cold cache.
        n_runs: measured iterations. Higher = better stats, more time.
        cuda_sync: when True, call ``torch.cuda.synchronize()`` before taking
            timestamps. Required for accurate GPU timing; no-op on CPU.

    Returns:
        ``BenchmarkResult`` with median, min, max, mean, stdev across runs.
    """
    sync = _make_synchronizer(cuda_sync)

    print(f"\n--- {name}  (warmup={n_warmup}, runs={n_runs})", flush=True)

    for i in range(n_warmup):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        d = time.perf_counter() - start
        print(f"    warmup {i + 1}/{n_warmup}: {d:7.2f}s", flush=True)

    durations: list[float] = []
    for i in range(n_runs):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        d = time.perf_counter() - start
        durations.append(d)
        print(f"    run     {i + 1}/{n_runs}: {d:7.2f}s", flush=True)

    result = BenchmarkResult(
        name=name,
        n_warmup=n_warmup,
        n_runs=n_runs,
        median_s=statistics.median(durations),
        min_s=min(durations),
        max_s=max(durations),
        mean_s=statistics.mean(durations),
        stdev_s=statistics.stdev(durations) if n_runs > 1 else 0.0,
    )
    print(
        f"    -> median {result.median_s:7.2f}s  "
        f"min {result.min_s:7.2f}s  stdev {result.stdev_s:6.2f}s",
        flush=True,
    )
    return result


def _make_synchronizer(cuda_sync: bool) -> Callable[[], None]:
    """Pick a sync function once; avoid the ``if`` inside the hot loop."""
    if cuda_sync and torch.cuda.is_available():
        return torch.cuda.synchronize
    return _noop


def _noop() -> None:
    pass
