"""Subprocess runner for the port-parity anchors.

    python _anchor_runner.py --src <src-dir> --engine {ref,fast} \
        --anchor {mixed_causality,error_rows,feature_records,jacobian_mode} \
        --out <tensors.pt>

Builds the anchor's seeded fixture, computes attribution rows through the
requested engine (``lat`` reference or ``lat_fast``), and saves the outputs.
The parent test runs this twice — reference against the GRADER-OWNED
pristine src, fast against the agent's src — so neither side of the
comparison is agent-controlled twice over.

Each anchor fixture isolates one public contract: the fixtures are
engineered so that a divergence in any OTHER part of the engine cannot move
this anchor's outputs (zero error vectors where errors aren't the subject,
logit-only injections where records aren't the subject, and so on).
"""

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--engine", choices=["ref", "fast"], required=True)
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.src)

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)

    from lat.ir import (
        LinearizedAttention,
        LinearizedAttribution,
        LinearizedLayer,
        LinearizedMLP,
        NormSnapshot,
    )

    def mknorm(n_pos, d, g, radial=False):
        return NormSnapshot(
            scale=torch.rand(n_pos, generator=g) + 0.5,
            gamma=torch.randn(d, generator=g) * 0.2 + 1.0,
            mean=None,
            beta=None,
            radial=(
                torch.nn.functional.normalize(torch.randn(n_pos, d, generator=g), dim=-1)
                / (d ** 0.5)
            )
            if radial
            else None,
        )

    def mkattn(n_pos, d, g, causal=True, radial=False, heads=2, dh=4):
        pat = torch.tril(torch.ones(n_pos, n_pos)) if causal else torch.ones(n_pos, n_pos)
        pat = pat / pat.sum(-1, keepdim=True)
        return LinearizedAttention(
            pattern=pat.unsqueeze(0).repeat(heads, 1, 1) * 0.4,
            W_V=torch.randn(heads, d, dh, generator=g) * 0.25,
            W_O=torch.randn(heads, dh, d, generator=g) * 0.25,
            b_V=None,
            b_O=None,
            pre_norm=mknorm(n_pos, d, g, radial),
            is_causal=causal,
        )

    def mkmlp(n_pos, d, g, positions=None, W_enc=None, W_dec=None, error=False, radial=False):
        if positions is None:
            positions = torch.zeros(0, dtype=torch.long)
        na = positions.shape[0]
        if W_enc is None:
            W_enc = torch.randn(na, d, generator=g) * 0.3
        if W_dec is None:
            W_dec = torch.randn(na, d, generator=g) * 0.3
        return LinearizedMLP(
            active_positions=positions,
            active_feature_idx=torch.arange(na, dtype=torch.long),
            active_activations=torch.ones(na),
            W_enc_active=W_enc,
            W_dec_active=W_dec,
            b_enc_active=torch.zeros(na),
            b_dec=torch.zeros(d),
            d_transcoder=max(int(na), 64),
            pre_norm=mknorm(n_pos, d, g, radial),
            error_vec=(torch.randn(n_pos, d, generator=g) * 0.1 if error else torch.zeros(n_pos, d)),
        )

    def mkir(layers, n_pos, d, g, radial=False):
        return LinearizedAttribution(
            input_tokens=torch.arange(n_pos, dtype=torch.long) % 7,
            resid_0=torch.randn(n_pos, d, generator=g) * 0.3,
            embed_vecs=torch.randn(n_pos, d, generator=g) * 0.3,
            layers=tuple(layers),
            final_norm=mknorm(n_pos, d, g, radial),
            W_unembed=torch.randn(d, 16, generator=g),
            b_unembed=None,
            logits=torch.randn(n_pos, 16, generator=g),
        )

    n_pos, d = 8, 12
    if args.anchor == "mixed_causality":
        g = torch.Generator().manual_seed(11)
        layers = [
            LinearizedLayer(layer_idx=0, attention=mkattn(n_pos, d, g, causal=True), mlp=mkmlp(n_pos, d, g)),
            LinearizedLayer(layer_idx=1, attention=mkattn(n_pos, d, g, causal=False), mlp=mkmlp(n_pos, d, g)),
        ]
        ir = mkir(layers, n_pos, d, g)
        L, P = torch.tensor([2]), torch.tensor([2])
        V = torch.randn(1, d, generator=g)
    elif args.anchor == "error_rows":
        g = torch.Generator().manual_seed(22)
        layers = [
            LinearizedLayer(layer_idx=i, attention=mkattn(n_pos, d, g), mlp=mkmlp(n_pos, d, g, error=True))
            for i in range(2)
        ]
        ir = mkir(layers, n_pos, d, g)
        L, P = torch.tensor([2]), torch.tensor([2])
        V = torch.randn(1, d, generator=g)
    elif args.anchor == "feature_records":
        g = torch.Generator().manual_seed(33)
        w = torch.randn(1, d, generator=g) * 0.5
        pos = torch.tensor([3], dtype=torch.long)
        layers = [
            LinearizedLayer(layer_idx=0, attention=mkattn(n_pos, d, g), mlp=mkmlp(n_pos, d, g)),
            LinearizedLayer(
                layer_idx=1,
                attention=mkattn(n_pos, d, g),
                mlp=mkmlp(n_pos, d, g, positions=pos, W_enc=w, W_dec=w.clone()),
            ),
        ]
        ir = mkir(layers, n_pos, d, g)
        L, P = torch.tensor([1]), torch.tensor([3])
        V = w.clone()
    elif args.anchor == "jacobian_mode":
        g = torch.Generator().manual_seed(66)
        layers = [
            LinearizedLayer(
                layer_idx=i, attention=mkattn(n_pos, d, g, radial=True), mlp=mkmlp(n_pos, d, g, radial=True)
            )
            for i in range(2)
        ]
        ir = mkir(layers, n_pos, d, g, radial=True)
        L, P = torch.tensor([2]), torch.tensor([5])
        V = torch.randn(1, d, generator=g)
    else:
        raise SystemExit(f"unknown anchor {args.anchor}")

    if args.engine == "ref":
        from lat.primitives import backward_inject_batch
    else:
        from lat_fast.primitives import backward_inject_batch

    rows = backward_inject_batch(ir, inject_layers=L, inject_positions=P, inject_vectors=V)
    torch.save(
        {"features": rows.features, "errors": rows.errors, "tokens": rows.tokens},
        args.out,
    )


if __name__ == "__main__":
    main()
