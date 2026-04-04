#!/usr/bin/env bash
#
# Collect mini test results and run analysis.
# Run this after test_mini.sh agents have finished.
#
# Usage:
#   bash scripts_batch/test_mini_collect.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TAG_PREFIX="test-mini"
AGENTS=("claude-code" "codex" "gemini-cli")

echo ">>> Collecting results..."
for agent in "${AGENTS[@]}"; do
    echo "  - ${agent}"
    modal run scripts_batch/deprivacy_modal_runner.py --collect "${TAG_PREFIX}/${agent}" 2>&1 || echo "    (not ready yet or failed)"
done

echo ""
echo ">>> Running cross-agent analysis..."
python scripts_batch/analyze_cross_agent.py results/deprivacy/

echo ""
echo ">>> Per-task results:"
for agent in "${AGENTS[@]}"; do
    f="results/deprivacy/${TAG_PREFIX}/${agent}/task-01-im-looking-for-backpack-under.json"
    if [ -f "$f" ]; then
        score=$(python3 -c "import json; d=json.load(open('$f')); print(f\"{d['agent']:15s} score={d['score']:.2f}  passed={d['passed']}\")")
        echo "  ${score}"
    else
        echo "  ${agent}: result not found"
    fi
done
