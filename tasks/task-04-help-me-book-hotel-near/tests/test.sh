#!/bin/bash
# Harbor verifier: reads /logs/agent/result.json, runs LLM-as-judge, writes reward.

set -e

RESULT_FILE="/logs/agent/result.json"
REWARD_FILE="/logs/verifier/reward.txt"
REWARD_JSON="/logs/verifier/reward.json"

mkdir -p /logs/verifier

if [ ! -f "$RESULT_FILE" ]; then
    echo "No result.json — agent may have crashed."
    echo 0 > "$REWARD_FILE"
    exit 0
fi

# Use skill-phase1.json (same config as agent) for api_key/base_url when not in env
CONFIG_FILE="/cocoa-agent/configs/skill-phase1.json"
if [ -z "$OPENAI_API_KEY" ] && [ -f "$CONFIG_FILE" ]; then
    export OPENAI_API_KEY=$(/opt/python3.12/bin/python -c "
import json
cfg = json.load(open('$CONFIG_FILE'))
print(cfg.get('controller',{}).get('args',{}).get('api_key',''))
" 2>/dev/null || echo "")
fi

if [ -z "$OPENAI_BASE_URL" ] && [ -f "$CONFIG_FILE" ]; then
    export OPENAI_BASE_URL=$(/opt/python3.12/bin/python -c "
import json
cfg = json.load(open('$CONFIG_FILE'))
print(cfg.get('controller',{}).get('args',{}).get('base_url',''))
" 2>/dev/null || echo "")
fi

/opt/python3.12/bin/python - <<'PYEOF'
import json
import sys
import time
import importlib.util

with open("/logs/agent/result.json") as f:
    result = json.load(f)

spec = importlib.util.spec_from_file_location("test_task", "/tests/test_task.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

start = time.perf_counter()
eval_result = mod.test(result)
execution_time = time.perf_counter() - start

passed = bool(eval_result.get("passed", False))
details = eval_result.get("details", {}) or {}
score = float(details.get("overall_score", 0))

with open("/logs/verifier/reward.txt", "w") as f:
    f.write("1" if passed else "0")

# Full eval structure (evolve_bench parity)
eval_output = {
    "passed": passed,
    "feedback": eval_result.get("feedback", ""),
    "details": details,
    "execution_time": execution_time,
    "reward": 1 if passed else 0,
    "overall_score": score,
    "generic_score": float(details.get("generic_score", 0)),
}
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(eval_output, f, indent=2)

feedback = eval_result.get("feedback", "")
if feedback:
    print(feedback)
print(f"\nEvaluation: passed={passed}  score={score:.2f}")
sys.exit(0 if passed else 1)
PYEOF
