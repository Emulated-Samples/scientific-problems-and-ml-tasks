# lat-bench

Hyperfocal RL environment built from the `lat` gold state: a linearized
attribution tracer for transformers with transcoders (Python, pytest).

## Layout

- `hyperfocal.yaml`: orchestrator config (agent, model, paths).
- `workspace/`: the codebase the solver agent modifies. Gold state on
  `main`; each problem is a perturbation on its `problem/{id}` branch.
- `environment/`: grader entry point (TypeScript), `problems.yaml`
  prompts, and the hidden `behavioral-tests/` pytest suite. Never
  visible to the solver.

## Problems

| id | branch | type | technique |
|----|--------|------|-----------|
| `dag-ranker` | `problem/dag-ranker` | feature | structural removal |

## Grading

`runTests` runs the hidden suite from `environment/behavioral-tests/`
with `PYTHONPATH=workspace/src`: the agent's `lat` package is imported,
but the tests themselves are never agent-editable. The suite is the lat
gold test suite with the private-API DAG ranker tests replaced by
public-API behavioral tests, so solutions are graded on observable
behavior (power/dag selection equivalence, circuit-tracer oracle parity
at cap, API contract) plus the full regression surface of the library.
