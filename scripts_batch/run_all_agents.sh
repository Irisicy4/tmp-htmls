#!/usr/bin/env bash
#
# Run deprivacy-100 across ALL agent combinations.
#
# Prerequisites:
#   1. modal deploy scripts_batch/deprivacy_modal_runner.py
#   2. .env has ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
#
# Usage:
#   bash scripts_batch/run_all_agents.sh              # all 100 tasks, all agents
#   bash scripts_batch/run_all_agents.sh --first-n 3  # smoke test with 3 tasks
#
# Each agent runs as a separate Modal experiment (parallel on Modal's side).
# All triggers fire within seconds, then your laptop is free.

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS="${*}"

echo "============================================"
echo "  Deprivacy-100: All-Agent Batch Run"
echo "  Extra args: ${EXTRA_ARGS:-none}"
echo "============================================"
echo ""

# --- Agents to run ---
# Comment out any agent you don't want or don't have keys for.

AGENTS=(
    "claude-code"
    "codex"
    "gemini-cli"
    "aider"
)

for agent in "${AGENTS[@]}"; do
    echo ">>> Triggering: ${agent}"
    python scripts_batch/trigger_deprivacy.py --agent "${agent}" ${EXTRA_ARGS}
    echo ""
done

echo "============================================"
echo "  All agents dispatched!"
echo "  Your laptop can disconnect now."
echo ""
echo "  To check status:"
echo "    modal run scripts_batch/deprivacy_modal_runner.py --list-runs"
echo ""
echo "  To collect results (per agent):"
echo "    modal run scripts_batch/deprivacy_modal_runner.py --collect deprivacy/<agent>/<timestamp>"
echo "============================================"
