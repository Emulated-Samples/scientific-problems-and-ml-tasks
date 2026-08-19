# Scientific Problems and ML Tasks

Three agentic ML engineering tasks in harbor format, with sample rollouts.

## Layout

```
attribution-speedup/        task: speed up a neural attribution pipeline
mooncake-oplog/             task: implement an oplog batching layer in a C++ KV-cache store
population-genetics-pca/    task: scale a population-genetics PCA pipeline
trajectory/                 sample rollouts, grouped by task and model
```

Each task directory contains:

```
task.toml           task metadata and resource requirements
instruction.md      the prompt given to the agent
environment/        Dockerfile and sources for the task environment
tests/              verifier that scores a solution
solution/           reference solution
README.md           task description
THIRD_PARTY_NOTICES.md
```

## Usage

Requires [harbor](https://pypi.org/project/harbor) 0.18.0 and Docker.

Run an agent against a task:

```
harbor run -p <task-dir> -a claude-code -m <model>
```

Verify the reference solution with the built-in oracle agent:

```
harbor run -p <task-dir> -a oracle
```

The verifier writes the score to `verifier/reward.json` in the trial directory.

## Trajectories

`trajectory/<task>/<model>/rollout_<harness>_<n>/` holds complete rollouts from
Claude Opus 5, GPT-5.6 Sol, and Gemini 3.7 Flash: the agent session under `agent/`,
the task and outcome in `config.json` and `result.json`, and the verifier record
under `verifier/`.

## Licensing

See `LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
