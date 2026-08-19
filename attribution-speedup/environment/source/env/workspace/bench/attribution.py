"""Time the attribution sweep on synthetic IRs (CPU-friendly, no model).

``lat.attribute_from_linearized`` accepts a hand-built
``LinearizedAttribution``, so the backward sweep can be profiled without
downloading a model: this script builds random synthetic IRs of configurable
shape and times attribution over them.

Run::

    python -m bench.attribution
    python -m bench.attribution --n-layers 12 --n-pos 64 --feats-per-pos 32
    python -m bench.attribution --cap 256          # capped run
    python -m bench.attribution --profile          # torch.profiler table

The IR is random, so the resulting graph is meaningless — the point is that
the *work* is real: the same primitives run as on a real model, with the
shape knobs controlling how much of each kind there is.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

import lat
from lat.ir import (
    LinearizedAttention,
    LinearizedAttribution,
    LinearizedLayer,
    LinearizedMLP,
    NormSnapshot,
)

from ._harness import time_callable


class _StubTokenizer:
    """attribute_from_linearized only ever calls ``decode``."""

    def decode(self, ids):
        if isinstance(ids, int):
            return f"<t{ids}>"
        return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())


def _norm(n_pos: int, d_model: int) -> NormSnapshot:
    return NormSnapshot(scale=torch.ones(n_pos), gamma=torch.ones(d_model), mean=None, beta=None)


def build_synthetic_ir(
    *,
    n_layers: int,
    n_pos: int,
    d_model: int,
    n_heads: int,
    d_head: int,
    feats_per_pos: int,
    d_vocab: int = 64,
    seed: int = 0,
) -> LinearizedAttribution:
    """A random causal IR of the given shape.

    Every layer carries a lower-triangular attention pattern and
    ``feats_per_pos`` active features at each position.
    """
    g = torch.Generator().manual_seed(seed)
    pattern = torch.tril(torch.ones(n_pos, n_pos))
    pattern = pattern / pattern.sum(-1, keepdim=True)

    n_active = feats_per_pos * n_pos
    positions = torch.arange(n_pos, dtype=torch.long).repeat_interleave(feats_per_pos)

    layers = []
    for i in range(n_layers):
        w = torch.randn(n_active, d_model, generator=g) * 0.3
        attention = LinearizedAttention(
            pattern=pattern.unsqueeze(0).repeat(n_heads, 1, 1) * 0.1,
            W_V=torch.randn(n_heads, d_model, d_head, generator=g) * 0.1,
            W_O=torch.randn(n_heads, d_head, d_model, generator=g) * 0.1,
            b_V=None,
            b_O=None,
            pre_norm=_norm(n_pos, d_model),
            is_causal=True,
        )
        mlp = LinearizedMLP(
            active_positions=positions,
            active_feature_idx=torch.arange(n_active, dtype=torch.long),
            active_activations=torch.ones(n_active),
            W_enc_active=w,
            W_dec_active=w * 0.9,
            b_enc_active=torch.zeros(n_active),
            b_dec=torch.zeros(d_model),
            d_transcoder=max(n_active, 64),
            pre_norm=_norm(n_pos, d_model),
            error_vec=torch.zeros(n_pos, d_model),
        )
        layers.append(LinearizedLayer(layer_idx=i, attention=attention, mlp=mlp))

    return LinearizedAttribution(
        input_tokens=torch.zeros(n_pos, dtype=torch.long),
        resid_0=torch.randn(n_pos, d_model, generator=g) * 0.1,
        embed_vecs=torch.randn(n_pos, d_model, generator=g) * 0.1,
        layers=tuple(layers),
        final_norm=_norm(n_pos, d_model),
        W_unembed=torch.randn(d_model, d_vocab, generator=g),
        b_unembed=None,
        logits=torch.randn(n_pos, d_vocab, generator=g),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.attribution", description=__doc__)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-pos", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--feats-per-pos", type=int, default=16)
    parser.add_argument("--cap", type=int, default=None, help="max_feature_nodes (default: no cap)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="print a torch.profiler table")
    parser.add_argument("--out", default=None, help="write the results as JSON to this path")
    args = parser.parse_args()

    ir = build_synthetic_ir(
        n_layers=args.n_layers,
        n_pos=args.n_pos,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_head=args.d_head,
        feats_per_pos=args.feats_per_pos,
        seed=args.seed,
    )
    tokenizer = _StubTokenizer()
    print(
        f"[bench.attribution] IR: {ir.n_layers} layers x {ir.n_pos} pos, "
        f"{ir.total_active_features} active features, cap={args.cap}"
    )

    def run():
        return lat.attribute_from_linearized(
            ir,
            tokenizer,
            show_progress=False,
            max_feature_nodes=args.cap,
            batch_size=args.batch_size,
        )

    if args.profile:
        run()  # warmup outside the profiled region
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        return

    result = time_callable(
        f"attribute_from_linearized (cap={args.cap})",
        run,
        n_warmup=args.warmup,
        n_runs=args.runs,
    )

    if args.out:
        payload = {
            "timestamp": time.time(),
            "shape": {
                "n_layers": args.n_layers,
                "n_pos": args.n_pos,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "d_head": args.d_head,
                "feats_per_pos": args.feats_per_pos,
                "cap": args.cap,
                "batch_size": args.batch_size,
            },
            "result": dataclasses.asdict(result),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[bench.attribution] wrote {args.out}")


if __name__ == "__main__":
    main()
