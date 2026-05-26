#!/bin/bash
# Harbor canonical verifier shim. All tasks share this exact script.
# Reads the agent's parsed result, runs the LLM judge N times, writes reward.json
# (plus judge_runs.json + feedback.txt) into /logs/verifier/.
#
# JUDGE_RUNS controls variance studies — default 1, set higher to sample.
# JUDGE_MODEL and JUDGE_TEMPERATURE optionally override judge defaults.
set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Both sandbox images install judge deps into /opt/python3.12. Prefer that
# interpreter; fall back to the PATH python3.
if [ -z "${PYTHON:-}" ]; then
    if [ -x "/opt/python3.12/bin/python3" ]; then
        PYTHON="/opt/python3.12/bin/python3"
    else
        PYTHON="python3"
    fi
fi

mkdir -p /logs/verifier
ARGS=(
    --tests-dir "$TESTS_DIR"
    --agent-result /logs/agent/agent_result.json
    --output-dir /logs/verifier
    --n-runs "${JUDGE_RUNS:-1}"
)
[ -n "${JUDGE_MODEL:-}" ] && ARGS+=(--model "$JUDGE_MODEL")
[ -n "${JUDGE_TEMPERATURE:-}" ] && ARGS+=(--temperature "$JUDGE_TEMPERATURE")

"$PYTHON" /harness/run_judge.py "${ARGS[@]}"
