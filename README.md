# EvolveBench Harbor

Benchmark harness for running and evaluating AI agents on [EvolveBench](https://github.com/cocoabench/evolvebench) tasks.
Supports multiple agents (codex, claude-code, cocoa, …) with **fire-and-forget Modal execution** — dispatch an experiment and disconnect your laptop.

---

## Approaches

There are two runners, each serving a different purpose:

### 1. `harbor_modal_runner.py` — multi-agent, fire-and-forget

Uses [Harbor](https://harborframework.com/docs) to install and run any agent (codex, claude-code, gemini-cli, …) inside Modal Sandboxes.
The orchestrator is a deployed Modal Function — your laptop can disconnect after triggering.

```bash
# Deploy once
modal deploy harbor_modal_runner.py

# Trigger a run — then disconnect
python trigger_harbor.py --agent codex
python trigger_harbor.py --agent codex --first-n 5
python trigger_harbor.py --agent codex --task-names task-01-...,task-02-...

# Collect results when done
modal run harbor_modal_runner.py --collect <run-tag>
```

Supported agents: `codex`, `claude-code`, `gemini-cli`, `aider`, and any other agent Harbor supports.
API keys are read from `.env` and injected automatically per agent.

### 2. `standalone_modal_runner.py` — cocoa-agent + skill experiments

Runs **cocoa-agent** directly on Modal (no Harbor, no sandboxes). Designed for **2-phase skill experiments**: Phase 1 collects skills, Phase 2 injects them.

```bash
# Deploy once
modal deploy standalone_modal_runner.py

# Trigger a skill experiment — then disconnect
python trigger_experiment.py
python trigger_experiment.py --first-n 5

# Collect results when done
modal run standalone_modal_runner.py --collect <run-tag>
```

### 3. `harbor_runner.sh` — local/Harbor native (requires laptop awake)

Runs Harbor directly with `-e modal` or `-e docker`. Simpler setup but your laptop must stay connected for the duration.

```bash
./harbor_runner.sh tasks/batch-1
ENV=docker ./harbor_runner.sh tasks/batch-1/task-01-im-looking-for-backpack-under
```

---

## Prerequisites

- Python 3.12+
- [Harbor](https://harborframework.com/docs): `uv tool install harbor` or `pip install harbor`
- [Modal](https://modal.com) account + CLI: `pip install modal && modal setup`
- API key — OpenAI directly, or any provider via [UniAPI](https://uniapi.io) proxy

## Setup

```bash
cp env.template .env
# Edit .env — set OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL
```

### `.env` reference

```bash
OPENAI_API_KEY=sk-your-key               # Required — OpenAI key or UniAPI key
LLM_BASE_URL=https://api.uniapi.io/v1   # Optional — proxy for non-OpenAI models
LLM_MODEL=gpt-4.1-mini                  # Optional — model used by both agent and evaluator
```

With UniAPI as `LLM_BASE_URL`, you can use Claude, Gemini, or any model via the OpenAI-compatible API.

---

## Repository Structure

```
evolve_bench_harbor/
├── harbor_modal_runner.py     # Multi-agent fire-and-forget runner (Harbor + Modal deploy)
├── trigger_harbor.py          # Trigger script for harbor_modal_runner
├── standalone_modal_runner.py # Cocoa-agent runner with 2-phase skill experiments
├── trigger_experiment.py      # Trigger script for standalone_modal_runner
├── run_skill_experiment.sh    # End-to-end wrapper: Phase 1 → extract skills → Phase 2 → compare
├── harbor_runner.sh           # Legacy: Harbor CLI runner (laptop must stay awake)
├── generate_report.py         # Generate HTML/Excel report from experiment results
├── scripts/
│   ├── convert_tasks.py       # Convert cocoa-agent format → Harbor format
│   ├── extract_skills.py      # Post-process Phase 1 results to extract skills
│   └── sync-harness.sh        # Sync harness/ + configs/ into task environments
├── env.template               # Copy to .env for API keys
├── agents/
│   └── cocoa_harbor_agent.py  # Harbor wrapper for CocoaAgent
├── harness/                   # Files injected into cocoa-agent task containers
│   ├── run_task.py            #   Agent entry point + skill middleware
│   ├── skill_store.py         #   Skill persistence + LLM search
│   ├── skill_extractor.py     #   Skill extraction
│   └── adapters/
│       └── cocoa_adapter.py   #   Cocoa-agent-specific adapter
├── configs/
│   ├── harbor-config.json     # Default config (no skills)
│   ├── skill-phase1.json      # Phase 1: store skills
│   └── skill-phase2.json      # Phase 2: inject skills
├── visualizer/                # Interactive trace viewer (web app)
│   ├── server.py              #   Python HTTP server
│   └── index.html             #   Single-page app
└── tasks/
    ├── batch-1/               # 100 EvolveBench tasks
    ├── cocoa-synthetic-50/    # 50 synthetic tasks (cocoa-agent format)
    └── updated-deprivacy-100/ # 100 updated deprivacy tasks
        └── <task-name>/
            ├── instruction.md      # Task prompt
            ├── task.toml           # Harbor task config (timeout, resources)
            ├── environment/
            │   ├── Dockerfile      # Task container (cocoa-agent + harness)
            │   └── docker-compose.yaml
            └── tests/
                ├── test.sh         # Verifier entry point
                └── test_task.py    # LLM-as-judge scorer
```

---

## Skill System (2-Phase, cocoa-agent)

The skill system is middleware in `harness/run_task.py` that works with cocoa-agent:

- **Phase 1** (`store_skills=true`): after each successful task, extracts a reusable skill and saves it as a `.md` file.
- **Phase 2** (`use_skills=true`): before each task, retrieves relevant skills and injects them into the instruction.

For a full end-to-end run:

```bash
# Runs Phase 1, extracts skills, then Phase 2, then prints comparison
./run_skill_experiment.sh tasks/cocoa-synthetic-50
```

Or manually via `standalone_modal_runner.py` for fire-and-forget.

---

## Adding Tasks

Tasks live under `tasks/`. To add a new task:

1. Create `tasks/<task-name>/` with `instruction.md`, `task.toml`, and `tests/`
2. Copy `environment/` from any existing task
3. Run `./scripts/sync-harness.sh` to populate harness files into the environment

To convert tasks from cocoa-agent format:

```bash
python scripts/convert_tasks.py /path/to/cocoa-agent-tasks/ tasks/<output-dir>/
```

---

## Results & Visualization

Results are stored in the Modal Volume `evolve-bench-results` and downloaded to `results/`.

```bash
# Download a run
modal run harbor_modal_runner.py --collect <run-tag>
modal run standalone_modal_runner.py --collect <run-tag>
```

**Trace visualizer** — interactive web app for replaying a task's execution step by step (thought/action/screenshot), with Phase 1 vs Phase 2 comparison:

```bash
python visualizer/server.py \
  --data-dir results/modal/<run-tag>/phase1 \
  --compare-dir results/modal/<run-tag>/phase2 \
  --port 8085
# Open http://localhost:8085
```

**Report generator** — static HTML dashboard + Excel spreadsheet:

```bash
python generate_report.py results/modal/<run-tag>/
```

---

## License

Apache 2.0
