# EvolveBench Harbor

Run **any agent** with [Harbor](https://harborframework.com/docs). Start with **CocoaAgent** — a generic agent that works with any task ([official cocoabench/cocoa-agent](https://github.com/cocoabench/cocoa-agent)).

## Prerequisites

- Docker
- [Harbor](https://harborframework.com/docs) (`uv tool install harbor` or `pip install harbor`)
- OpenAI API key (used by the agent and the LLM-as-judge verifier)
- For Modal: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from [modal.com/settings/tokens](https://modal.com/settings/tokens)

```bash
cp env.template .env && $EDITOR .env   # optional; run.sh loads .env automatically
```

## Quick Start

```bash
# 1. API key (or add to .env)
export OPENAI_API_KEY=sk-your-key

# 2. Run task (Docker)
./run.sh

# Modal (cloud)
ENV=modal ./run.sh
```

No pre-built image — the task's `environment/Dockerfile` builds from scratch.

## Structure

Agents are pluggable. Each agent lives in `agents/` and implements Harbor's [installed agents](https://harborframework.com/docs/agents#installed-agents) interface. **CocoaAgent** is generic — it takes the task instruction from Harbor and runs it inside the container.

```
evolvebench/
├── run.sh                    # Entry: harbor run with CocoaAgent
├── env.template              # Copy to .env for API keys
├── agents/
│   └── cocoa_agent/          # CocoaAgent — default
├── configs/
│   └── skill-phase1.json
├── scripts/
│   └── prepare-modal-context.sh   # Prep for Modal (run.sh calls it when ENV=modal)
└── tasks/
    └── <task-name>/
        ├── instruction.md
        ├── task.toml
        ├── environment/
        │   ├── Dockerfile          # Base image; COPYs agents/ and configs/ (Docker: repo root context; Modal: prepare-modal-context)
        │   └── docker-compose.yaml
        ├── solution/
        │   └── solve.sh            # Oracle
        └── tests/
            ├── test.sh             # Verifier: LLM-as-judge
            └── test_task.py
```

## Adding a New Task

1. Create `tasks/<task-name>/` with `instruction.md`, `task.toml`, `solution/`, `tests/`
2. Copy `environment/` from task-01 — update `docker-compose.yaml` paths (context, dockerfile) if the task name differs
3. Run: `./run.sh tasks/<task-name>`

## Config (evolve_bench parity)

Uses `configs/skill-phase1.json` by default — same controller (gpt-4.1-mini) and sandbox (max_iterations: 50) as evolve_bench's `run_skill_phase1.sh`. Produces comparable results.

```bash
# Use default skill-phase1 config (same as evolve_bench)
./run.sh

# Use harbor-config instead
COCOA_CONFIG=/cocoa-agent/configs/harbor-config.json ./run.sh
```

## Run Options

```bash
./run.sh [task_path] [output_dir]

# ENV: docker (default) | modal | e2b | daytona | runloop | gke
ENV=modal ./run.sh

# Override agent, model, or config
AGENT=agents.cocoa_agent:CocoaHarborAgent ./run.sh
MODEL=openai/gpt-4.1-mini ./run.sh
COCOA_CONFIG=/cocoa-agent/configs/harbor-config.json ./run.sh
```

**Modal:** `run.sh` calls `prepare-modal-context.sh` when `ENV=modal` — it copies `agents/` and `configs/` into the task's `environment/` since Modal uses that as the build context. First run takes ~10–20 min (Playwright + Chromium). Use `--debug` if the build fails.

## Task

**task-01-im-looking-for-backpack-under**: Find 3–5 backpacks under $75 with features similar to https://www.amazon.com/dp/B09YRC9Y3G and summarize key features and prices.

### Sample output (rubric scores)

```
=== Customized Rubric Score: 3.8/5 ===
  source_navigation: 3/5
  budget_compliance: 5/5
  recommendation_count: 5/5
  feature_comparison: 3/5
  actionability: 3/5

Customized reasoning: The agent claims to have accessed the original Amazon URL, but the provided steps show limited interaction, resulting in a vague understanding of the product's specific features, thus a score of 3 for source_navigation. All alternative products are priced under $75, earning a score of 5 for budget_compliance. The agent presented exactly 3 distinct product alternatives, scoring a 5 for recommendation_count. However, the feature comparison against the original product is basic, only generally mentioning shared features like compartments and water resistance without detail, resulting in a score of 3 for feature_comparison. Finally, the response is structured to some extent but lacks precise product identifiers or direct links, only justifying a score of 3 for actionability.

=== Generic Rubric Score: 2.6/5 ===
  task_completion: 3/5
  information_quality: 2/5
  response_quality: 2/5
  completeness: 3/5

Generic reasoning: The AI agent provided three backpack options under $75 with features that partially match the original. However, the response lacks a thorough comparison to the original backpack, and it did not fully research or verify all features of the original item. The information is somewhat vague and lacks proper citations or links to the product pages for verification. The structure is adequate but lacks detail and depth, making it less useful for making an informed decision.

Evaluation: passed=True  score=3.80
```

## License

Apache 2.0
