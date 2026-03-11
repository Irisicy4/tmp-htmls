# EvolveBench Harbor

Run **any agent** with [Harbor](https://harborframework.com/docs) on **Docker** (local) or **Modal** (cloud, parallel). Ships with **CocoaAgent** — a generic browser agent that works with any task ([cocoabench/cocoa-agent](https://github.com/cocoabench/cocoa-agent)).

## Prerequisites

- [Harbor](https://harborframework.com/docs) (`uv tool install harbor` or `pip install harbor`)
- Docker (for local runs)
- [Modal](https://modal.com) account + CLI (for cloud runs): `pip install modal && modal setup`
- API key — OpenAI directly, or any provider via [UniAPI](https://uniapi.io) proxy

## Setup

```bash
# 1. Configure API key and optional settings
cp env.template .env && $EDITOR .env

# 2. For Modal: store your API key as a Modal secret
modal secret create openai-secret OPENAI_API_KEY=sk-your-key
```

### `.env` reference

```bash
OPENAI_API_KEY=sk-your-key          # Required — OpenAI key or UniAPI key
# LLM_BASE_URL=https://api.uniapi.io/v1  # Optional — proxy for non-OpenAI models
# LLM_MODEL=claude-sonnet-4-20250514     # Optional — model override (default: gpt-4.1-mini)
```

With UniAPI as `LLM_BASE_URL`, you can use any model (Claude, Gemini, etc.) — UniAPI translates the OpenAI API format to the target provider.

## Quick Start

```bash
# Run all 43 tasks on Modal (default):
./harbor_runner.sh

# Run a single task on Modal:
./harbor_runner.sh tasks/task-01-im-looking-for-backpack-under

# Run locally with Docker:
ENV=docker ./harbor_runner.sh tasks/task-01-im-looking-for-backpack-under
```

## Run Options

```bash
./harbor_runner.sh [task_path] [output_dir]
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `modal` | `modal` (cloud, parallel) or `docker` (local) |
| `OPENAI_API_KEY` | — | API key (or UniAPI key for non-OpenAI models) |
| `LLM_BASE_URL` | — | Proxy endpoint (e.g. UniAPI) |
| `LLM_MODEL` | `gpt-4.1-mini` | Model override |
| `COCOA_MAX_ITERATIONS` | `50` | Max agent iterations per task |
| `MODAL_SECRET` | `openai-secret` | Modal secret name |
| `AGENT` | `agents.cocoa_agent:CocoaHarborAgent` | Agent import path |
| `MODEL` | `openai/gpt-4.1-mini` | Harbor model string |
| `COCOA_CONFIG` | `/cocoa-agent/configs/skill-phase1.json` | Config path inside container |

```bash
# Override model via .env or inline:
LLM_MODEL=claude-sonnet-4-20250514 ./harbor_runner.sh

# Use a different config:
COCOA_CONFIG=/cocoa-agent/configs/harbor-config.json ./harbor_runner.sh

# Limit iterations for quick testing:
COCOA_MAX_ITERATIONS=10 ./harbor_runner.sh tasks -l 3
```

## Structure

```
evolve_bench_harbor/
├── harbor_runner.sh              # Main entry — runs on Modal (default) or Docker
├── standalone_modal_runner.py    # Alternative: direct Modal runner (bypasses Harbor)
├── env.template                  # Copy to .env for API keys
├── agents/
│   └── cocoa_agent/              # CocoaAgent — default agent
│       ├── cocoa_harbor_agent.py #   Harbor wrapper
│       ├── run_task.py           #   Multi-agent entry point (inside container)
│       ├── overlay.py            #   skip_docker monkey-patch
│       └── agents_overlay/       #   Agent imports overlay
├── configs/
│   ├── harbor-config.json        # Default config (gpt-4.1-mini, 50 iterations)
│   ├── skill-phase1.json         # Skill experiment config
│   └── uniapi.json               # UniAPI config
└── tasks/
    └── <task-name>/
        ├── instruction.md        # Task prompt
        ├── task.toml             # Harbor task config (timeout, resources, verifier env)
        ├── environment/
        │   ├── Dockerfile        # Self-contained: clones cocoa-agent + glue files
        │   ├── docker-compose.yaml
        │   ├── run_task.py       # Copied glue files (for Modal build context)
        │   ├── overlay.py
        │   ├── agents__init__.py
        │   ├── harbor-config.json
        │   └── skill-phase1.json
        └── tests/
            ├── test.sh           # Verifier entry
            └── test_task.py      # LLM-as-judge scorer
```

## Adding a New Task

1. Create `tasks/<task-name>/` with `instruction.md`, `task.toml`, `tests/`
2. Copy `environment/` from any existing task — the Dockerfile and glue files are identical across tasks
3. Run: `./harbor_runner.sh tasks/<task-name>`

## Multi-Agent Support

`run_task.py` supports multiple agent types via the `agent_type` field in the config JSON:

- `cocoa` (default) — browser-based agent with sandbox
- `skill` — CocoaAgent with skill store/retrieval
- `claude_code`, `codex`, `gemini_cli`, `manus` — CLI/API agents (no sandbox needed)
- `openai_deep_research`, `gemini_deep_research` — deep research agents

Browser-based agents (`cocoa`, `skill`) start the sandbox; pure-API agents skip it.

## Standalone Modal Runner

`standalone_modal_runner.py` is an alternative that bypasses Harbor and talks to Modal directly. It builds a single shared image (faster for large batches) and manages its own dispatch/aggregation.

```bash
modal run standalone_modal_runner.py
modal run standalone_modal_runner.py --first-n 3
modal run standalone_modal_runner.py --task-names task-01-...,task-02-...
```

For most use cases, prefer `./harbor_runner.sh` (uses Harbor's native Modal support).

## Visualization

**Trace visualizer** — interactive web app for replaying a single task's execution step by step (think/action/screenshot):

```bash
python visualizer/server.py --data-dir results/modal/ --port 8085
# Open http://localhost:8085, select a task from the dropdown
```

**Report generator** — static HTML dashboard + Excel spreadsheet summarizing an experiment:

```bash
# Single experiment
python generate_report.py results/modal/

# Compare multiple experiments side by side
python generate_report.py results/exp1/ results/exp2/ results/exp3/
```

## License

Apache 2.0
