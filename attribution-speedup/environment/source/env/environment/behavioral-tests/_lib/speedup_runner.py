"""Subprocess runner for the base-speedup A/B timing harness.

Invoked once per side by test_base_attribution_speedup.py:

    python _speedup_runner.py --src <lat-src-dir> --out <artifacts.pt>

Prepends ``--src`` to sys.path, builds the fixture IRs, times
``attribute_from_linearized`` on each (single-threaded CPU time, one warmup +
best-of-3 measured reps — a cold first call amortizes allocator/lazy-import
costs into the warmup, and best-of is robust to one-sided scheduler noise),
and saves per-fixture timings plus the produced graphs (selected-feature
triples + adjacency) so the parent can verify the two sides computed the
same thing before trusting the ratio.

Fixtures are MIXED workloads (features spread over every position and layer)
so any legitimate speedup of the sweep moves the number — mechanism-specific
classification belongs to the asymmetry discriminators, not this file.
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--mode", choices=["sweep", "ranker"], default="sweep")
    args = parser.parse_args()

    sys.path.insert(0, args.src)

    import torch

    torch.set_num_threads(1)

    import lat
    from lat.ir import (
        LinearizedAttention,
        LinearizedAttribution,
        LinearizedLayer,
        LinearizedMLP,
        NormSnapshot,
    )

    def norm(n_pos, d_model, g):
        return NormSnapshot(
            scale=torch.rand(n_pos, generator=g) + 0.5,
            gamma=torch.randn(d_model, generator=g) * 0.2 + 1.0,
            mean=None,
            beta=None,
        )

    def build_ir(*, n_layers, n_pos, d_model, n_heads, d_head, feats_per_pos, seed):
        """Mixed synthetic causal IR: active features at every (layer, position)."""
        g = torch.Generator().manual_seed(seed)
        pattern = torch.tril(torch.ones(n_pos, n_pos))
        pattern = pattern / pattern.sum(-1, keepdim=True)
        n_active = feats_per_pos * n_pos
        positions = torch.arange(n_pos, dtype=torch.long).repeat_interleave(feats_per_pos)
        layers = []
        for i in range(n_layers):
            attention = LinearizedAttention(
                pattern=pattern.unsqueeze(0).repeat(n_heads, 1, 1) * 0.3,
                W_V=torch.randn(n_heads, d_model, d_head, generator=g) * 0.2,
                W_O=torch.randn(n_heads, d_head, d_model, generator=g) * 0.2,
                b_V=None,
                b_O=None,
                pre_norm=norm(n_pos, d_model, g),
                is_causal=True,
            )
            mlp = LinearizedMLP(
                active_positions=positions,
                active_feature_idx=torch.arange(n_active, dtype=torch.long),
                active_activations=torch.rand(n_active, generator=g) + 0.2,
                W_enc_active=torch.randn(n_active, d_model, generator=g) * 0.3,
                W_dec_active=torch.randn(n_active, d_model, generator=g) * 0.3,
                b_enc_active=torch.zeros(n_active),
                b_dec=torch.zeros(d_model),
                d_transcoder=n_active,
                pre_norm=norm(n_pos, d_model, g),
                error_vec=torch.randn(n_pos, d_model, generator=g) * 0.05,
            )
            layers.append(LinearizedLayer(layer_idx=i, attention=attention, mlp=mlp))
        return LinearizedAttribution(
            input_tokens=torch.arange(n_pos, dtype=torch.long) % 11,
            resid_0=torch.randn(n_pos, d_model, generator=g) * 0.3,
            embed_vecs=torch.randn(n_pos, d_model, generator=g) * 0.3,
            layers=tuple(layers),
            final_norm=norm(n_pos, d_model, g),
            W_unembed=torch.randn(d_model, 32, generator=g),
            b_unembed=None,
            logits=torch.randn(n_pos, 32, generator=g),
        )

    class _StubTokenizer:
        def decode(self, ids):
            if isinstance(ids, int):
                return f"<t{ids}>"
            return " ".join(f"<t{int(i)}>" for i in torch.as_tensor(ids).flatten())

    def build_ranker_ir(*, n_layers, n_pos, d_model, feats_per_pos, seed):
        """Ranking-dominated fixture: feature k at every layer reads/writes
        residual channel ``k % d_model`` plus faint random cross-talk, so the
        influence graph is densely connected and the ranker's work dominates
        (d_model is tiny, so the per-batch backward work is negligible)."""
        g = torch.Generator().manual_seed(seed)
        n_active = feats_per_pos * n_pos
        positions = torch.arange(n_pos, dtype=torch.long).repeat_interleave(feats_per_pos)
        chan = torch.arange(n_active) % d_model
        layers = []
        for i in range(n_layers):
            w = torch.zeros(n_active, d_model)
            w[torch.arange(n_active), chan] = 1.0
            w += torch.randn(n_active, d_model, generator=g) * 0.02
            attention = LinearizedAttention(
                pattern=torch.zeros(2, n_pos, n_pos),
                W_V=torch.zeros(2, d_model, 4),
                W_O=torch.zeros(2, 4, d_model),
                b_V=None,
                b_O=None,
                pre_norm=norm(n_pos, d_model, g),
            )
            mlp = LinearizedMLP(
                active_positions=positions,
                active_feature_idx=torch.arange(n_active, dtype=torch.long),
                active_activations=torch.ones(n_active),
                W_enc_active=w,
                W_dec_active=w * 0.9,
                b_enc_active=torch.zeros(n_active),
                b_dec=torch.zeros(d_model),
                d_transcoder=n_active,
                pre_norm=norm(n_pos, d_model, g),
                error_vec=torch.zeros(n_pos, d_model),
            )
            layers.append(LinearizedLayer(layer_idx=i, attention=attention, mlp=mlp))
        return LinearizedAttribution(
            input_tokens=torch.zeros(n_pos, dtype=torch.long),
            resid_0=torch.randn(n_pos, d_model, generator=g) * 0.1,
            embed_vecs=torch.randn(n_pos, d_model, generator=g) * 0.1,
            layers=tuple(layers),
            final_norm=norm(n_pos, d_model, g),
            W_unembed=torch.randn(d_model, 32, generator=g),
            b_unembed=None,
            logits=torch.randn(n_pos, 32, generator=g),
        )

    # sweep: two mixed no-cap regimes ("long" position-heavy, "deep"
    # layer-heavy) covering the backward sweep's scaling axes.
    # ranker: two CAPPED ranking-dominated fixtures — "shallow" stays well
    # inside any fixed-point method's convergence range (used for the
    # selection-parity gate), "deep" has influence chains past it (timing
    # only: methods that are exact past that range may legitimately select
    # differently there).
    if args.mode == "sweep":
        fixtures = {
            "long": dict(n_layers=6, n_pos=96, d_model=40, n_heads=4, d_head=8,
                         feats_per_pos=6, seed=101),
            "deep": dict(n_layers=32, n_pos=16, d_model=40, n_heads=4, d_head=8,
                         feats_per_pos=8, seed=202),
        }
        build = build_ir
        run_kwargs = {
            "long": dict(show_progress=False, batch_size=64),
            "deep": dict(show_progress=False, batch_size=64),
        }
    else:
        fixtures = {
            "shallow": dict(n_layers=8, n_pos=4, d_model=8, feats_per_pos=128,
                            seed=303),
            "deep": dict(n_layers=150, n_pos=4, d_model=8, feats_per_pos=128,
                         seed=404),
        }
        build = build_ranker_ir
        run_kwargs = {
            "shallow": dict(show_progress=False, max_feature_nodes=128, batch_size=4),
            "deep": dict(show_progress=False, max_feature_nodes=512, batch_size=4),
        }

    tokenizer = _StubTokenizer()
    results = {}
    for name, shape in fixtures.items():
        ir = build(**shape)
        kwargs = run_kwargs[name]

        graph = lat.attribute_from_linearized(ir, tokenizer, **kwargs)  # warmup

        best = float("inf")
        for _ in range(args.reps):
            t0 = time.process_time()
            lat.attribute_from_linearized(ir, tokenizer, **kwargs)
            best = min(best, time.process_time() - t0)

        triples = [
            tuple(graph.active_features[int(i)].tolist())
            for i in graph.selected_features.tolist()
        ]
        results[name] = {
            "best_s": best,
            "adjacency": graph.adjacency_matrix.to(torch.float32),
            "selected_triples": triples,
            "logit_nodes": [tuple(n) for n in graph.logit_nodes],
        }

    torch.save(results, args.out)
    for name, r in results.items():
        print(f"{name}: best {r['best_s']:.3f}s CPU", flush=True)


if __name__ == "__main__":
    main()
