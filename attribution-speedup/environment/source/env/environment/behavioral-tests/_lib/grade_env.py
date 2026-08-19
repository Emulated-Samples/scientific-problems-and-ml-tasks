"""Grade-time environment stash — imported by the root conftest FIRST.

The grading engine hands the hidden suite two sensitive values through the
environment: the metrics-sidecar path (LAT_BENCH_METRICS_FILE — continuous
scores and metric multipliers are read from it, so writing it is writing
reward) and the grader-owned pristine-baseline location
(LAT_BENCH_PRISTINE_SRC). Several correctness tests import the agent's
``lat`` into this very process, and any subprocess a test spawns inherits
``os.environ`` — so as long as the values live in the environment, agent
code can discover them (e.g. register an ``atexit`` forger at import time
that out-writes every honestly recorded metric after the last test
finishes).

This module runs before any test module or agent code is imported (the
root conftest imports it immediately after setting sys.path) and POPS both
variables out of ``os.environ`` into module state. Tests reach the values
through the accessors below; agent code imported later finds nothing in
the environment, and every test-spawned subprocess is born clean.

LAT_BENCH_BASE_FAMILY stays in the environment on purpose: it only gates
collection of base-shaped tests and reveals nothing exploitable.
"""

import json
import os

_METRICS_FILE = os.environ.pop("LAT_BENCH_METRICS_FILE", None)
_PRISTINE_SRC = os.environ.pop("LAT_BENCH_PRISTINE_SRC", None)
_GOLD_SRC = os.environ.pop("LAT_BENCH_GOLD_SRC", None)


def metrics_file() -> str | None:
    return _METRICS_FILE


def pristine_src() -> str | None:
    """Grader-owned pristine workspace src, or None on problems without a
    pristine baseline (callers use this as their skip gate)."""
    return _PRISTINE_SRC


def gold_src() -> str | None:
    """Grader-owned GOLD reference workspace src (the optimized reference
    solution), staged for host-relative speedup calibration — or None when the
    grader did not stage gold (then the continuous target stays the constant).
    Popped from the environment for the same reason as the others: agent code
    imported later must not discover it."""
    return _GOLD_SRC


def record_metric(key: str, value: float) -> None:
    """Write one measurement to the metrics sidecar for the grader's
    continuous scoring / metric multipliers. No-op when no sidecar was
    configured (plain local pytest runs)."""
    if not _METRICS_FILE:
        return
    try:
        with open(_METRICS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[key] = value
    with open(_METRICS_FILE, "w") as f:
        json.dump(data, f)
