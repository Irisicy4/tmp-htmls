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

## Running Agent Judging Separately

`harness/rejudge_n_times.py` allows LLM judges to be run on already-collected agent
traces which can be useful for reproducibility, reliability, and robustness checks. Inputs are held fixed per task; only the
judge call varies. This implementation is stand-alone from the main pipeline. We show a default configuration below:

```bash
python harness/rejudge_n_times.py \
  --source cocoa-deprivacy-100 \
  --results-root results \
  --tasks-root tasks/updated-deprivacy-100 \
  --out-dir results/judge_variance \
  --n-calls 5 \
  --judge-models gpt-4o,claude-sonnet-4-20250514 \
  --temperature 1.0 \
  --workers 8
```

This reads agent traces from `results/<source>/trials/<task>/` with priority:
`agent/result.json` → `agent_result.json` → `result.json`. It reads the rubric
from `tasks/<tasks-root>/<task>/tests/test_task.py`. Requires `OPENAI_API_KEY`
and `LLM_BASE_URL` (or `OPENAI_BASE_URL`) in `.env`.

**Outputs** (`results/judge_variance/`):

| File | One row per | Key fields |
|---|---|---|
| `<source>_runs.jsonl` | (task, judge_model, call_index) | `raw_response`, `dimension_scores`, `overall_score`, `passed`, `pass_threshold`, `dimension_weights`, `input_hash`, errors, timing |
| `<source>_inputs.jsonl` | task | `system_prompt`, `user_content`, `input_hash`, `rubric_flavor`, dimension metadata |
| `<source>_summary.json` | run | judges, n_calls, temperature, counts, paths |

The `input_hash` is a sha256 of `(system_prompt || user_content)` to make sure judgements are matched.

The script is **resumable** in the sense that `(task, judge_model, call_index)` triples already
present in `runs.jsonl` are skipped.

**Rubric flavors.** Two formats are auto-detected in `test_task.py`:
- **A** (94 tasks in `updated-deprivacy-100`): `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` + `DIMENSION_WEIGHTS`; judge returns JSON inside `<Answer>...</Answer>`.
- **B** (6 tasks): single `RUBRIC` string + `DIMENSIONS` list; weights parsed out of rubric text via regex; judge returns bare JSON.

Score parsing tries `<Answer>` tags first, then `json` fences, then raw, then a `{...}` substring. `overall_score` is recomputed from `dimension_scores × dimension_weights` when all dimensions parse; otherwise the judge-reported `overall_score` is used.

## Judge Reliability Analysis

`analysis/rejudge_validity_analysis.py` consumes the `*_runs.jsonl` produced by
`harness/rejudge_n_times.py` and computes pairwise TVD-MI on the binary
verdict, pairwise TVD-MI on `overall_score` at the finest histogram available,
Cohen's κ, and Cronbach's α. It writes three figures and a `report.md`.

```bash
# After running rejudge_n_times.py:
python analysis/rejudge_validity_analysis.py \
    --source cocoa-deprivacy-100 \
    --rejudge-dir results/judge_variance \
    --out-dir analysis/output

# Smoke test (no API, synthesizes a small runs.jsonl):
python analysis/rejudge_validity_analysis.py --smoke
```

**Interpretation.** TVD-MI(X;Y) = ½·Σ|P(X,Y) − P(X)P(Y)| equals the total
variation distance between the joint and the product of marginals. By the
variational characterisation of TV, ½·TVD-MI is the advantage over chance of
an optimal classifier distinguishing same-task judgement pairs from
independently-paired judgements. So welfare = 0.4 implies an optimal
evaluator can distinguish same-trace from independently-paired judgements
with accuracy 0.7.

**Outputs** (`--out-dir`):

| File | Contents |
|---|---|
| `mi_matrix.npy` | k×k pairwise TVD-MI on binary pass/fail |
| `mi_matrix_fine.npy` | k×k pairwise TVD-MI on `overall_score` at finest histogram |
| `figure_heatmap.png` | Heatmap of pairwise binary TVD-MI with per-model block dividers |
| `figure_reliability.png` | Per-annotator mean off-diagonal welfare (bar chart) |
| `figure_passdist.png` | Histogram of pass votes per task |
| `report.md` | Numbers and figure links; per-model and cross-model blocks |

## Layout

- `harbor_modal_runner.py` / `trigger_harbor.py` — Modal orchestrator + trigger
- `harness/` — code injected into task envs (`run_judge.py`, `test.sh`, `evaluator.py`, agent adapters) plus the post-hoc `rejudge_n_times.py` analysis tool
- `tasks/` — task suites (`real_118_refactored/`, `updated-deprivacy-100/`, …); each task has `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/test_task.py`
- `agentic_judge/` — browser-verifying judge (optional)
- `visualizer/` — interactive trace viewer
