"""Canonical divergence/fix pairs for base-port-parity (grader-side data).

Same shape as base-bugs' _base_bugs_states.py: bug id -> (relative path,
buggy snippet, fixed snippet) against the canonical problem workspace.
stage_states materializes FIXED + one STATE_<bug> per bug.
"""

import shutil
from pathlib import Path

BUGS: dict[str, tuple[str, str, str]] = {
    # D1 — any/all confusion in the causality gate
    "d1": (
        "lat_fast/primitives.py",
        "self.causal = any(layer.attention.is_causal for layer in ir.layers)",
        "self.causal = all(layer.attention.is_causal for layer in ir.layers)",
    ),
    # D2 — fencepost in the error-row record
    "d2": (
        "lat_fast/primitives.py",
        """        # Error rows read upstream signal; the cone's boundary row carries
        # only the batch's own seeds.
        upstream = max(pos_limit - 1, 0)
        error_contribs[:, layer_idx, :upstream] = (
            grad[:, :upstream] * layer.mlp.error_vec[:upstream]
        ).sum(dim=-1)""",
        """        error_contribs[:, layer_idx, :pos_limit] = (
            grad * layer.mlp.error_vec[:pos_limit]
        ).sum(dim=-1)""",
    ),
    # D3 — seed splice hoisted above the records
    "d3": (
        "lat_fast/primitives.py",
        """        grad = grad + _seed(
            grad,
            layer.mlp.pre_norm if pos_limit == n_pos else _norm_prefix(layer.mlp.pre_norm, pos_limit),
            mask=inject_layers == layer_idx,
            positions=inject_positions,
            vectors=inject_vectors,
        )

        groups = plan.groups[layer_idx]""",
        """        groups = plan.groups[layer_idx]""",
    ),
    # d3 part 2 handled via PAIRED replacement below (record block must move
    # back above the splice); see _apply.
    # D6 — radial projection applied before the fused gamma/scale multiply
    "d6": (
        "lat_fast/primitives.py",
        """    if snapshot.radial is not None:
        u = snapshot.radial
        d_out = d_out - u * (d_out * u).sum(dim=-1, keepdim=True)
    d_centered = d_out * (snapshot.gamma / snapshot.scale.unsqueeze(-1))""",
        """    d_centered = d_out * (snapshot.gamma / snapshot.scale.unsqueeze(-1))
    if snapshot.radial is not None:
        u = snapshot.radial
        d_centered = d_centered - u * (d_centered * u).sum(dim=-1, keepdim=True)""",
    ),
}

# D3 is a two-part move: removing the hoisted splice AND re-inserting it
# after the record block.
D3_TAIL = (
    """        # Error rows read upstream signal; the cone's boundary row carries
        # only the batch's own seeds.
        upstream = max(pos_limit - 1, 0)
        error_contribs[:, layer_idx, :upstream] = (
            grad[:, :upstream] * layer.mlp.error_vec[:upstream]
        ).sum(dim=-1)

        attn = layer.attention""",
    """        # Error rows read upstream signal; the cone's boundary row carries
        # only the batch's own seeds.
        upstream = max(pos_limit - 1, 0)
        error_contribs[:, layer_idx, :upstream] = (
            grad[:, :upstream] * layer.mlp.error_vec[:upstream]
        ).sum(dim=-1)

        grad = grad + _seed(
            grad,
            layer.mlp.pre_norm if pos_limit == n_pos else _norm_prefix(layer.mlp.pre_norm, pos_limit),
            mask=inject_layers == layer_idx,
            positions=inject_positions,
            vectors=inject_vectors,
        )

        attn = layer.attention""",
)


def _apply_fix(dst: Path, bug: str) -> None:
    rel, buggy, fixed = BUGS[bug]
    p = dst / rel
    text = p.read_text()
    assert buggy in text, f"{bug}: buggy snippet missing from {rel}"
    text = text.replace(buggy, fixed)
    if bug == "d3":
        old, new = D3_TAIL
        # the error block may already be fixed (d2) — try both variants
        if old in text:
            text = text.replace(old, new)
        else:
            old2 = old.replace(
                """        # Error rows read upstream signal; the cone's boundary row carries
        # only the batch's own seeds.
        upstream = max(pos_limit - 1, 0)
        error_contribs[:, layer_idx, :upstream] = (
            grad[:, :upstream] * layer.mlp.error_vec[:upstream]
        ).sum(dim=-1)""",
                """        error_contribs[:, layer_idx, :pos_limit] = (
            grad * layer.mlp.error_vec[:pos_limit]
        ).sum(dim=-1)""",
            )
            new2 = new.replace(
                """        # Error rows read upstream signal; the cone's boundary row carries
        # only the batch's own seeds.
        upstream = max(pos_limit - 1, 0)
        error_contribs[:, layer_idx, :upstream] = (
            grad[:, :upstream] * layer.mlp.error_vec[:upstream]
        ).sum(dim=-1)""",
                """        error_contribs[:, layer_idx, :pos_limit] = (
            grad * layer.mlp.error_vec[:pos_limit]
        ).sum(dim=-1)""",
            )
            assert old2 in text, "d3 tail: neither error-block variant found"
            text = text.replace(old2, new2)
    p.write_text(text)


def stage_states(bugged_src: Path, out_dir: Path) -> dict[str, Path]:
    """FIXED (all fixed) + STATE_<bug> (all fixed except <bug>)."""
    states: dict[str, Path] = {}
    for name in ["FIXED", *BUGS]:
        dst = out_dir / name / "src"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(bugged_src, dst)
        for bug in BUGS:
            if name == bug:
                continue
            _apply_fix(dst, bug)
        states[name] = dst
    return states
