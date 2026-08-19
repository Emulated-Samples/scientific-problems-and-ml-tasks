"""Canonical bug/fix pairs for the base-bugs problem, as grader-side data.

Each entry maps a bug id to (relative path, buggy snippet, fixed snippet)
against the CANONICAL problem workspace (the pristine copy staged by the
environment). ``stage_states`` materializes the seven library states the
test-coverage harness grades against: FIXED (every bug corrected) and one
STATE_<bug> per bug (everything corrected except that bug).

These snippets patch the pristine tree only — never the agent's tree — so
they always apply exactly.
"""

import shutil
from pathlib import Path

BUGS: dict[str, tuple[str, str, str]] = {
    "bug1": (
        "lat/primitives.py",
        """    d = x0.shape[-1]
    s0_sq = x0.pow(2).mean(dim=-1, keepdim=True) + eps
    radial = x0 * (x0 * grad).sum(dim=-1, keepdim=True) / (d * s0_sq)
    return gamma * (grad - radial) / s0_sq.sqrt()""",
        """    d = x0.shape[-1]
    s0_sq = x0.pow(2).mean(dim=-1, keepdim=True) + eps
    scaled = gamma * grad / s0_sq.sqrt()
    radial = x0 * (x0 * scaled).sum(dim=-1, keepdim=True) / (d * s0_sq)
    return scaled - radial""",
    ),
    "bug2": (
        "lat/attribution/_phases.py",
        """        cutoff = rows.abs().topk(max_edges, dim=-1).values[..., -1:]
        rows.masked_fill_(rows.abs() <= cutoff, 0.0)""",
        """        _, topk_idx = torch.topk(rows.abs(), max_edges, dim=-1)
        mask = torch.zeros_like(rows, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        rows.masked_fill_(~mask, 0.0)""",
    ),
    "bug3": (
        "lat/circuit/graph.py",
        'order = np.argsort(-weights, kind="stable")',
        'order = np.argsort(-np.abs(weights), kind="stable")',
    ),
    "bug4": (
        "lat/attribution/_ranker.py",
        "normalized /= normalized.amax(dim=1, keepdim=True).clamp(min=1e-8)",
        "normalized /= normalized.sum(dim=1, keepdim=True).clamp(min=1e-8)",
    ),
    "bug5": (
        "lat/circuit/graph.py",
        """        adjacent = {e.target for e in self.edges_from(node_id)}
        return sorted(adjacent)""",
        """        adjacent = {e.target for e in self.edges_from(node_id)}
        adjacent |= {e.source for e in self.edges_into(node_id)}
        return sorted(adjacent)""",
    ),
    "bug6": (
        "lat/circuit/graph.py",
        "return [n for n in self.nodes_list() if q in n.node_id or q in n.label.lower()]",
        "return [n for n in self.nodes_list() if q in n.node_id.lower() or q in n.label.lower()]",
    ),
}


def pristine_is_base_bugs(pristine_src: Path) -> bool:
    """True when the staged pristine tree carries the planted bugs — the
    self-gate that scopes the coverage harness to the base-bugs problem."""
    try:
        text = (pristine_src / "lat" / "primitives.py").read_text()
    except OSError:
        return False
    return BUGS["bug1"][1] in text


def stage_states(pristine_src: Path, out_dir: Path) -> dict[str, Path]:
    """Materialize FIXED + one STATE_<bug> per bug. Returns name -> src path."""
    states: dict[str, Path] = {}
    for name in ["FIXED", *BUGS]:
        dst = out_dir / name / "src"
        shutil.copytree(pristine_src, dst)
        for bug, (rel, buggy, fixed) in BUGS.items():
            if name == bug:
                continue  # leave exactly this bug in place
            p = dst / rel
            text = p.read_text()
            assert buggy in text, f"{bug}: buggy snippet missing from pristine {rel}"
            p.write_text(text.replace(buggy, fixed))
        states[name] = dst
    return states
