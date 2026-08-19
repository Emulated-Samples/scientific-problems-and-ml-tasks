"""Test-coverage grading for base-bugs: does the agent's suite CATCH each bug?

Mutation-kill grading of the agent's visible test suite, judged behaviorally
against grader-owned library states (never by reading their test code):

- ``FIXED``: the canonical workspace with all six bugs corrected.
- ``STATE_<bug>``: all bugs corrected except one.

The agent's ``workspace/tests`` runs against every state. Bug ``k`` counts as
COVERED iff some individual test passes on ``FIXED`` and fails on
``STATE_k`` — the per-test differential. That makes the verdict:

- style-agnostic (any behavioral test that detects the bug earns the kill,
  regardless of name or shape);
- immune to incidental coupling (a test pinned to the agent's own internals
  fails on both states → no differential, no credit);
- immune to the planted corruptions (a complicit test PASSES on the bugged
  state and fails on FIXED — the inverse signature — so it can never score);
- fail-closed (an uncollectable suite produces no differentials → 0).

Self-gated: runs only when the staged pristine tree carries the planted bugs
(i.e. the base-bugs problem); skips neutrally everywhere else.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from _base_bugs_states import BUGS, pristine_is_base_bugs, stage_states
from _lib.grade_env import pristine_src, record_metric as _record_metric

_PRISTINE = pristine_src()

pytestmark = pytest.mark.skipif(
    not (_PRISTINE and pristine_is_base_bugs(Path(_PRISTINE))),
    reason="base-bugs test-coverage harness (pristine tree is not the bugged base)",
)


def _agent_tests_dir() -> Path:
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and (Path(entry) / "lat" / "__init__.py").exists():
            return Path(entry).parent / "tests"
    raise AssertionError("agent workspace tests dir not found via PYTHONPATH")


def _run_suite(src: Path, tests_dir: Path, xml_out: Path) -> dict[str, str]:
    """Run the agent's suite against ``src``; return {test_id: status}.

    Status is 'passed' only for a genuinely green testcase; failures, errors,
    and skips all count as not-passed (a skipped test can't witness a kill).
    """
    # os.environ no longer carries LAT_BENCH_* (stashed by _lib.grade_env),
    # so the agent-suite subprocess is born clean; the pop below is
    # belt-and-braces.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("LAT_BENCH_METRICS_FILE", None)
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(tests_dir),
            "-q", "--no-header", "-p", "no:cacheprovider",
            "--continue-on-collection-errors",
            f"--junitxml={xml_out}",
        ],
        capture_output=True,
        text=True,
        timeout=8 * 60,
        env=env,
        cwd=str(tests_dir),
    )
    statuses: dict[str, str] = {}
    if xml_out.exists():
        for case in ET.parse(xml_out).getroot().iter("testcase"):
            name = case.attrib.get("name", "")
            classname = case.attrib.get("classname", "")
            tid = f"{classname}::{name}"
            bad = case.find("failure") is not None or case.find("error") is not None
            skipped = case.find("skipped") is not None
            statuses[tid] = "passed" if not (bad or skipped) else "not-passed"
    # A suite that produced no report at all yields {} -> zero differentials.
    del proc
    return statuses


@pytest.fixture(scope="module")
def kills(tmp_path_factory):
    """{bug: set of agent test ids that pass on FIXED and fail on STATE_bug}."""
    root = tmp_path_factory.mktemp("bug-states")
    states = stage_states(Path(_PRISTINE), root)
    tests_dir = _agent_tests_dir()
    assert tests_dir.exists(), f"agent tests dir missing: {tests_dir}"

    results: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            name: pool.submit(_run_suite, src, tests_dir, root / f"{name}.xml")
            for name, src in states.items()
        }
        for name, fut in futures.items():
            results[name] = fut.result()

    fixed = results["FIXED"]
    out: dict[str, set] = {}
    for bug in BUGS:
        state = results[bug]
        out[bug] = {
            tid
            for tid, status in fixed.items()
            if status == "passed" and state.get(tid) == "not-passed"
        }
    return out


@pytest.mark.parametrize("bug", list(BUGS))
def test_agent_suite_catches(bug, kills):
    """The agent's visible suite must contain at least one behavioral test
    that detects this bug (passes on the fixed library, fails when exactly
    this bug is re-injected)."""
    witnesses = kills[bug]
    _record_metric(f"coverage_{bug}", float(len(witnesses)))
    assert witnesses, (
        f"no test in the agent's suite distinguishes the fixed library from "
        f"the library with {bug} re-injected — the regression is uncovered"
    )
