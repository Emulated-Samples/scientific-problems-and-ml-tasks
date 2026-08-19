from pathlib import Path

from grader.gates import library_scan


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    ROOT / "tasks/from_scratch_pca/instruction.md",
    ROOT / "environment/problems.yaml",
]


def test_visible_prompts_define_outcomes_without_disclosing_fast_architecture():
    forbidden = {
        "sample-by-sample gram": "sufficient-statistic architecture",
        "z z.t": "explicit Gram construction",
        "z z^t": "explicit Gram construction",
        "random byte offset": "spatial-sampling architecture",
        "random byte-offset": "spatial-sampling architecture",
        "block read": "spatial-sampling architecture",
        "far more variants than samples": "dimension-reduction hint",
        "identity probe": "private evaluator mechanism",
        "full-scan reference": "private evaluator anchor",
        "speed factor": "private reward mechanics",
        "per-dataset reward": "private reward mechanics",
        "adequate marker coverage": "private shortcut detector",
        "strong fast solution": "private score calibration",
        "around 1.5": "private score calibration",
        "random/naive": "private score calibration",
        "several demographic regimes": "hidden suite composition",
        "confirm your environment": "prescribed development step",
        "test it on vcfs you create yourself": "prescribed validation strategy",
        "simple working pca": "prescribed implementation strategy",
    }
    for path in PROMPTS:
        visible = path.read_text().lower()
        for phrase, meaning in forbidden.items():
            assert phrase not in visible, f"{path}: leaked {meaning}: {phrase!r}"


def test_visible_prompts_keep_the_exact_statistical_contract():
    for path in PROMPTS:
        visible = path.read_text().lower()
        assert "sqrt(2 p (1-p))" in visible or "sqrt(2 p (1 - p))" in visible
        assert "a/c/g/t" in visible
        assert "haploid" in visible
        assert "partial" in visible
        assert "1 <= k <" in visible
        assert "positive-variance sample directions" in visible
        assert "systematically exclude" in visible


def test_every_prompt_states_the_rules_the_gates_actually_enforce():
    """A gate may only enforce what the prompt says. An unstated rule that zeroes a valid expert
    submission is the worst thing this environment can do -- run_019f6281 lost a whole Opus rollout
    to exactly that, and the repair was to DISCLOSE the rule, not to relax the sandbox.

    So each entry below is a rule some gate hard-enforces, paired with the words that make it
    discoverable from the prompt alone. `library_scan` gives `import ctypes` a hard integrity zero,
    and the runtime refuses every `ctypes.*` audit event; before this test, the live rollout prompt
    (`problems.yaml`) mentioned ctypes ONLY inside the narrow multiprocessing clause, while the
    packaged `instruction.md` said "ctypes-loaded native code" -- two surfaces, two different rules,
    neither matching the one enforced.

    This deliberately does not check for prose. It checks that the capability class is named, which
    is what a solver needs in order to not walk into it.
    """
    # Derived from the gate, not restated: whatever `library_scan` refuses to import, every prompt
    # surface must say is unavailable.
    banned_by_the_static_gate = library_scan._DENY_STDLIB  # noqa: SLF001
    assert "ctypes" in banned_by_the_static_gate, (
        "this test is anchored on library_scan rejecting `import ctypes`; if that changed, retarget "
        "the test rather than deleting it"
    )

    for prompt in PROMPTS:
        text = prompt.read_text().lower()
        assert "ctypes" in text, (
            f"{prompt.name} never mentions ctypes, but library_scan hard-zeroes `import ctypes` "
            f"and the runtime refuses every ctypes.* event"
        )
        # Naming the word is not stating the rule, and this is the trap the first version of this
        # test fell into: it accepted `instruction.md`'s "ctypes-loaded native code" as compliant.
        # That phrase bans USING ctypes to load native code. The gate bans IMPORTING ctypes at all,
        # for any purpose. A solver who read that surface and imported ctypes to call, say, madvise
        # would still take a hard zero for a rule nobody wrote down -- so the narrower surface left
        # the defect fully intact while the test went green.
        #
        # Require the capability CLASS to be named, which is the only phrasing that covers the
        # enforced rule without enumerating alternatives.
        assert "foreign-function" in text, (
            f"{prompt.name} mentions ctypes but does not state the enforced rule: library_scan "
            f"rejects `import ctypes` OUTRIGHT, not just ctypes-loaded native code. Name the "
            f"capability class (foreign-function interfaces)."
        )


def test_both_prompt_surfaces_agree_with_each_other():
    """Two prompt authorities can drift exactly as two gates can, and it happened here.

    `problems.yaml` is the LIVE rollout prompt; `instruction.md` ships with the packaged Harbor
    task. On ctypes they stated two DIFFERENT narrower rules -- "ctypes-backed multiprocessing
    shared objects" versus "ctypes-loaded native code" -- and neither matched what the gates
    enforce. Fixing one surface and not the other is how a two-authority defect survives its own
    fix, so pin that they agree on every capability class, not just that each says something.
    """
    texts = {prompt.name: prompt.read_text().lower() for prompt in PROMPTS}
    for capability in ("foreign-function", "ctypes", "sharedctypes", "fork"):
        stating = {name for name, text in texts.items() if capability in text}
        assert stating in ({*texts}, set()), (
            f"prompt surfaces disagree about `{capability}`: stated by {sorted(stating)}, "
            f"missing from {sorted(set(texts) - stating)}"
        )
