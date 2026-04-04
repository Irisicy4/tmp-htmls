#!/usr/bin/env bash
#
# Mini test: 1 task x 3 agents
# Verifies the full pipeline works before running the real experiment.
#
# Usage:
#   bash scripts_batch/test_mini.sh
#
# Takes ~5-20 min total (3 parallel agent runs on Modal).
# After all 3 finish, auto-collects results and runs analysis.

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="task-01-im-looking-for-backpack-under"
AGENTS=("claude-code" "codex" "gemini-cli")
TAG_PREFIX="test-mini"

echo "============================================"
echo "  Mini Test: 1 task x 3 agents"
echo "  Task: ${TASK}"
echo "  Agents: ${AGENTS[*]}"
echo "============================================"
echo ""

# --- Deploy (idempotent — safe to re-run) ---
echo ">>> Deploying Modal app..."
modal deploy scripts_batch/deprivacy_modal_runner.py
echo ""

# --- Trigger all 3 agents ---
TAGS=()
for agent in "${AGENTS[@]}"; do
    TAG="${TAG_PREFIX}/${agent}"
    TAGS+=("${TAG}")
    echo ">>> Triggering: ${agent} (tag: ${TAG})"
    python scripts_batch/trigger_deprivacy.py \
        --agent "${agent}" \
        --task-names "${TASK}" \
        --run-tag "${TAG}"
    echo ""
done

echo "============================================"
echo "  All 3 agents dispatched!"
echo ""
echo "  Wait for them to finish (~5-20 min), then run:"
echo ""
echo "    bash scripts_batch/test_mini_collect.sh"
echo ""
echo "  Or collect manually:"
for tag in "${TAGS[@]}"; do
    echo "    modal run scripts_batch/deprivacy_modal_runner.py --collect ${tag}"
done
echo ""
echo "  Then analyze:"
echo "    python scripts_batch/analyze_cross_agent.py results/deprivacy/"
echo "============================================"
