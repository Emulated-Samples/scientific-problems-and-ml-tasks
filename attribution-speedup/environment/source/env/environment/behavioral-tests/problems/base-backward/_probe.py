"""Isolated agent-code probe for base-backward.

The only process that imports a workspace ``lat`` at grade time. Spawned
with a scrubbed env (no LAT_BENCH_* vars). Modes:

  agent-backward : call the src's backward_norm / backward_attention on the
                   seeded fixtures under torch.no_grad() AND under a runtime
                   sensor that replaces the actual autograd function objects
                   (torch.autograd.grad/backward, Tensor.backward,
                   torch.func / torch.autograd.functional VJP-family) and the
                   forward-linear ops (apply_*_linear / apply_norm /
                   apply_attention) with trip-wires. A hand-derived adjoint is
                   pure tensor arithmetic and touches none of these; autograd
                   (however aliased, getattr'd, imported from a sibling module,
                   or wrapped in enable_grad) resolves to a patched object and
                   raises, as does the numerical-Jacobian shortcut that sweeps
                   basis vectors through the forward. Per-case flag records the
                   raise, so that anchor scores 0 with partial credit intact.
  forward-vjp    : ground truth — VJP of the src's forward-linear ops
                   (apply_norm_linear / apply_attention_linear) at the same
                   fixtures (autograd allowed here); save the adjoints.
  oracle         : run the src's attribute_linear on gelu-2l UNDER no_grad;
                   save the graph for parity vs circuit_tracer.

    python _probe.py --src <src> --mode <mode> --out <f>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.src)
    import torch

    torch.set_num_threads(1)
    import _backward_fixtures as fx

    if args.mode == "agent-backward":
        # Runtime sensor: swap the real autograd + forward-linear function
        # OBJECTS for trip-wires BEFORE importing agent code, so aliasing,
        # getattr, sibling-module imports and enable_grad all resolve to a
        # patched object and raise. Patch the compute entry points only
        # (enable_grad alone computes nothing and patching it risks internal
        # torch); if autograd is re-enabled and then invoked, the invocation
        # trips. Ground truth runs in the separate forward-vjp process, so
        # this patch never touches it.
        class _Tripped(RuntimeError):
            pass

        def _trip(what):
            def f(*a, **k):
                raise _Tripped(f"autograd/forward API used in backward: {what}")
            return f

        torch.autograd.grad = _trip("torch.autograd.grad")
        torch.autograd.backward = _trip("torch.autograd.backward")
        try:
            torch.Tensor.backward = _trip("Tensor.backward")
        except (AttributeError, TypeError):
            pass
        import torch.autograd.functional as _af
        for _nm in ("vjp", "jvp", "vhp", "hvp", "jacobian", "hessian"):
            if hasattr(_af, _nm):
                setattr(_af, _nm, _trip(f"torch.autograd.functional.{_nm}"))
        try:
            import torch.func as _tf
            for _nm in ("vjp", "jvp", "vhp", "hvp", "jacrev", "jacfwd",
                        "hessian", "grad", "grad_and_value", "linearize"):
                if hasattr(_tf, _nm):
                    setattr(_tf, _nm, _trip(f"torch.func.{_nm}"))
        except ImportError:
            pass

        from lat.primitives import backward_attention, backward_norm
        import lat.primitives as _lp

        # forward-linear sensor: a hand-derived adjoint never calls the forward
        # operator; the numerical-Jacobian shortcut does. Patch after import.
        for _nm in ("apply_norm_linear", "apply_attention_linear",
                    "apply_norm", "apply_attention"):
            if hasattr(_lp, _nm):
                setattr(_lp, _nm, _trip(f"forward op {_nm}() called inside backward"))

        out = {}
        for name, kw in fx.NORM_CASES.items():
            snap = fx.norm_snapshot(**kw)
            dov = fx.d_out(n_pos=snap.scale.shape[0], d=snap.gamma.shape[0], seed=fx.DOUT_SEED[name])
            try:
                with torch.no_grad():
                    res = backward_norm(snap, dov)
                out[name] = {"ok": True, "val": res.detach().clone()}
            except Exception as e:  # autograd path raises under no_grad
                out[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"}
        for name, kw in fx.ATTN_CASES.items():
            attn = fx.attention(**kw)
            n_pos = attn.pattern.shape[-1]
            d = attn.W_V.shape[1]
            dov = fx.d_out(n_pos=n_pos, d=d, seed=fx.DOUT_SEED[name])
            try:
                with torch.no_grad():
                    res = backward_attention(attn, dov)
                out[name] = {"ok": True, "val": res.detach().clone()}
            except Exception as e:
                out[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"}
        torch.save(out, args.out)
        return

    if args.mode == "forward-vjp":
        from lat.primitives import apply_attention_linear, apply_norm_linear

        out = {}
        for name, kw in fx.NORM_CASES.items():
            snap = fx.norm_snapshot(**kw)
            n_pos, d = snap.scale.shape[0], snap.gamma.shape[0]
            x0 = torch.zeros(n_pos, d, dtype=torch.float64)
            dov = fx.d_out(n_pos=n_pos, d=d, seed=fx.DOUT_SEED[name])
            _, vjp = torch.func.vjp(lambda x: apply_norm_linear(snap, x), x0)
            out[name] = vjp(dov)[0].detach().clone()
        for name, kw in fx.ATTN_CASES.items():
            attn = fx.attention(**kw)
            n_pos, d = attn.pattern.shape[-1], attn.W_V.shape[1]
            x0 = torch.zeros(n_pos, d, dtype=torch.float64)
            dov = fx.d_out(n_pos=n_pos, d=d, seed=fx.DOUT_SEED[name])
            _, vjp = torch.func.vjp(lambda x: apply_attention_linear(attn, x), x0)
            out[name] = vjp(dov)[0].detach().clone()
        torch.save(out, args.out)
        return

    if args.mode == "oracle":
        import torch.nn.functional as F
        from circuit_tracer.replacement_model.replacement_model_transformerlens import (
            TransformerLensReplacementModel,
        )
        from circuit_tracer.transcoder.single_layer_transcoder import (
            SingleLayerTranscoder,
            TranscoderSet,
        )

        import lat
        from lat.linearize import AttributionModel

        torch.manual_seed(0)
        tc = {}
        for layer in range(2):
            t = SingleLayerTranscoder(
                d_model=512, d_transcoder=64, activation_function=F.relu,
                layer_idx=layer, device=torch.device("cpu"), dtype=torch.float32,
            )
            with torch.no_grad():
                t.W_enc.copy_(torch.randn(64, 512) * (512 ** -0.5))
                t.W_dec.copy_(torch.randn(64, 512) * (64 ** -0.5))
            tc[layer] = t
        rm = TransformerLensReplacementModel.from_pretrained_and_transcoders(
            "gelu-2l",
            TranscoderSet(tc, feature_input_hook="mlp.hook_in",
                          feature_output_hook="mlp.hook_out", scan_name="t"),
            device="cpu",
        )
        model = AttributionModel.from_circuit_tracer(rm)
        ok = True
        err = ""
        try:
            with torch.no_grad():
                g = lat.attribute_linear(model, "The capital of France is", show_progress=False)
            payload = {
                "ok": True,
                "adjacency": g.adjacency_matrix.detach().to(torch.float64),
                "active_features": g.active_features.detach().clone(),
                "selected": g.selected_features.detach().clone(),
            }
        except Exception as e:
            payload = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:120]}"}
        torch.save(payload, args.out)
        return

    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    main()
