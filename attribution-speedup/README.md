# Attribution backward-sweep speedup (lat)

Make attribution-graph construction in our circuit-tracing library substantially faster, with the attribution outputs matching the original within tight numerical tolerance on every input and the public interface unchanged.

Models capture the general speedups and miss the temporal causal structure: twenty-one of twenty-five runs leave the largest exact optimization on the table.

## The task

This task challenges the model to make attribution-graph construction substantially faster. The attribution outputs must match the original within tight numerical tolerance on every input, and the full test suite must stay green.

The task came out of our own research implementing Anthropic's open-source circuit tracer, where we found the backward sweep dominating end-to-end attribution time at scale. The prompt is the complaint as an engineer would send it — "I need help speeding up attribution in our circuit-tracing library," profiling on long prompts and deep models shows the backward sweep dominating, make it substantially faster — and it closes with a deadline: "You have 24 hours."

The structure that makes the sweep cheap is causal. A feature can only be influenced by computation below its own layer, and a causal decoder can only read from earlier token positions, so most of the dense backward pass computes values that are structurally zero. Exploiting both bounds is exact and leaves the graph unchanged.

The engineer who built this environment wrote the library the task is set in for his interpretability research. Agents often ship speedups that quietly change the graph, and we needed the author here to make the verifier robust: outputs are compared against the original on every fixture, so speed bought by changing the answer earns nothing and every real, exact win earns proportional credit.

## Verifier design

We time the agent's code against the original and compare the outputs numerically, so speed bought by changing the answer earns nothing.

| What we check | How |
| --- | --- |
| It actually got faster | Timed A/B against the original on workloads built to expose the backward sweep, with the baseline re-timed on the same machine |
| The graph did not change | Attribution outputs must match the original within tight numerical tolerance on every fixture; an optimization that moves the results is zeroed however fast it is |
| The speedup is structural | Backward cost must shrink with the token-position causal cone and with layer depth, measured as separate axes |
| Partial work counts | Each axis is scored continuously, so a real but incomplete optimization lands between zero and full credit |

## Trace walkthrough

Two runs in the set effectively solve this task, at 1.0 and 0.995, and both did it by attacking the axis the rest left untouched: the temporal cone. Below them, the set separates by how much of the causal structure each run exploited.

### A strong run

1. **Sweep only what can be nonzero.** The perfect run captured a deterministic baseline before touching anything, then rebuilt the backward sweep around both causal bounds: rows sorted by injection layer so each layer's work runs only on rows that can still carry gradient, and rows cut into position-sorted chunks so each chunk sweeps a truncated context. Work that is guaranteed to be zero is never launched.
2. **Kill the giant intermediate.** The old recording step materialized a per-layer temporary that reached thirteen gigabytes on one test shape — the original gets killed outright there. It regrouped that as one matrix multiply per position group, wrote injection seeds on just the seeded slice instead of allocating full zero buffers, and removed a staging copy by writing sweep results straight into the edge matrix.
3. **Prove the graph did not move.** It wrote a forty-one-case equivalence suite against a naive full-width reference, mutation-tested that suite until every seeded bug was caught, and diffed complete graphs over thirty configurations: selected features identical, adjacency within about one part in ten million. The diff surfaced a real regression — a per-head split rounded bfloat16 once per head — which it fixed by accumulating in fp32. Eighty-six minutes and 123 steps ended four to fifteen times faster depending on shape, and a perfect score on all three axes.

### A failed run

1. **Build the impressive thing.** The lowest-scoring run spent its sixty-nine minutes writing a multithreaded C kernel with SIMD intrinsics for the sweep, plus vectorized fallbacks, and closed reporting roughly fifteen times faster with all 184 tests green.
2. **Never measure on the shapes that matter.** On the graded fixtures the combined ratio came back at 0.96 — the long-prompt fixture ran slower than the code it replaced — and the speedup axis scored zero. The kernel made the arithmetic faster but still swept the full width and depth, so both causality axes measured a ratio of one.
3. **Take the causal idea halfway.** The middle of the set restructures the sweep by layer and earns most of that axis, then attributes an early-position feature at the same cost as a full-prompt sweep: temporal ratio 1.0 against a required 1.5. Thirteen runs cluster between 0.34 and 0.69 on exactly this profile.
4. **Stop at a strong partial.** One run cleared the overall speedup bar outright and took full layer credit, landing exactly 0.75 — and its late-versus-early timing ratio was 0.97 where the reference solution measures around twenty-four.

The verifier's message on the axis most runs failed reads like a review comment: an early-position feature's causal cone is a few tokens, and attributing it should not cost a full-prompt sweep. Twenty-one of twenty-five runs never acted on that observation.

## Failure modes

These are the failure modes we saw across the evaluated runs, and most rollouts exhibit several at once.

| Failure mode | What goes wrong |
| --- | --- |
| The temporal cone unexploited | Twenty-one of twenty-five runs compute the full token width on every backward pass, and the positional axis of the score goes unearned. |
| Layer causality taken partway | Sweeps restructured by layer but batched so upper-layer work stays alive land around half credit on that axis instead of full. |
| Speed that exists only on the author's benchmark | One run reported fifteen times faster with every test green, then measured 0.96x on the graded fixtures — slower than the original on long prompts. |
| A weak micro-optimization | Overhead trimming with the sweep's structure untouched lands 0.12 to 0.16, a small fraction of the available score. |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/scientific-problems-and-ml-tasks/attribution-speedup --agent oracle -k 1 -o jobs/
```

The reference solution replays against a 0.90 floor. Grading is wall-clock-relative with the baseline re-timed on the host, and the task grades CPU-only on 4 CPUs and 16 GB.
