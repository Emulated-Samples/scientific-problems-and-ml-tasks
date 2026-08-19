"""Timing tools for lat.

- ``python -m bench.attribution`` — time the attribution sweep on synthetic
  IRs (CPU-friendly, no model download). ``--profile`` prints a
  torch.profiler breakdown.
- ``_harness`` — the shared timing primitives.
"""
