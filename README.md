# EvolveBench Harbor

Benchmark harness for running and evaluating AI agents (codex, claude-code, cocoa, …) on various kinds of tasks. Each task ships with a Dockerfile and an LLM-as-judge that scores the agent's output.

## Setup

```bash
cp env.template .env   # fill in API keys (UNIAPI_KEY, or OPENAI_API_KEY / ANTHROPIC_API_KEY, etc.)
pip install modal harbor && modal setup
```

Judge credentials priority: `JUDGE_API_KEY` → `OPENAI_API_KEY` → `UNIAPI_KEY`.
Agent credentials: provider-specific env var → `UNIAPI_KEY`.

## Run via Modal (fire-and-forget)

Deploys an orchestrator on Modal so your laptop can disconnect.

```bash
# Deploy once
modal deploy harbor_modal_runner.py

# Trigger a run
python trigger_harbor.py --agent codex --tasks-dir tasks/real_118_refactored
python trigger_harbor.py --agent claude-code --first-n 5
python trigger_harbor.py --agent codex --task-names task-01-...,task-02-...

# Collect results when done
modal run harbor_modal_runner.py --collect <run-tag>
```

Results land in `results/<run-tag>/<task-name>/` with canonical Harbor layout (`agent/`, `verifier/reward.json`, `artifacts/`, `result.json`).

## Run via Harbor (local, laptop awake)

Uses Harbor's CLI directly with Docker or Modal sandboxes.

```bash
harbor run -e docker -a codex tasks/real_118_refactored/task-01-...
harbor run -e modal  -a claude-code tasks/real_118_refactored
```

## Visualize results

```bash
python visualizer/server.py --data-dir results/<run-tag> --port 8085
# open http://localhost:8085
```

## Layout

- `harbor_modal_runner.py` / `trigger_harbor.py` — Modal orchestrator + trigger
- `harness/` — code injected into task envs (`run_judge.py`, `test.sh`, `evaluator.py`, agent adapters)
- `tasks/` — task suites (`real_118_refactored/`, `updated-deprivacy-100/`, …); each task has `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/test_task.py`
- `agentic_judge/` — browser-verifying judge (optional)
- `visualizer/` — interactive trace viewer
