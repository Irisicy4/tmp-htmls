#!/bin/bash
# Harbor verifier: reads /logs/agent/result.json (or synthesises one from
# /logs/agent/{claude-code.txt,codex.txt} when running plain `harbor run`),
# runs LLM-as-judge, writes reward.

set -e

RESULT_FILE="/logs/agent/result.json"
REWARD_FILE="/logs/verifier/reward.txt"
REWARD_JSON="/logs/verifier/reward.json"

mkdir -p /logs/verifier

# Fallback: synthesise result.json from raw agent stream output.
# Plain `harbor run` writes claude-code.txt / codex.txt instead of result.json.
if [ ! -f "$RESULT_FILE" ]; then
    /opt/python3.12/bin/python - <<'PYEOF'
import json
from pathlib import Path

CC = Path("/logs/agent/claude-code.txt")
CX = Path("/logs/agent/codex.txt")
OUT = Path("/logs/agent/result.json")

def parse_claude_code(text):
    conversation, tool_lines, last_assistant = [], [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        msg = ev.get("message", {}) or {}
        role = msg.get("role", ev.get("type"))
        content = msg.get("content", "")
        if role == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = block.get("name", "tool")
                    inp = block.get("input") or {}
                    parts = []
                    if isinstance(inp, dict):
                        for k in ("command", "url", "query", "path", "selector", "text", "description"):
                            v = inp.get(k)
                            if v:
                                parts.append(f"{k}={str(v)[:150]}")
                    detail = ", ".join(parts) or str(inp)[:200]
                    tool_lines.append(f"{name}({detail})")
        if role not in ("assistant", "user"):
            continue
        if isinstance(content, list):
            txt = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()
        else:
            txt = str(content).strip()
        if not txt:
            continue
        conversation.append({"role": role, "content": txt})
        if role == "assistant":
            last_assistant = txt
    return {
        "task_result": last_assistant,
        "conversation": conversation,
        "execution_summary": "\n".join(tool_lines[-60:]),
    }

def parse_codex(text):
    conversation, tool_lines, last_assistant = [], [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        etype = ev.get("type", "")
        if etype == "agent.message":
            txt = (ev.get("message") or "").strip()
            if txt:
                conversation.append({"role": "assistant", "content": txt})
                last_assistant = txt
        elif etype in ("tool.call", "function_call", "shell.call"):
            name = ev.get("name") or ev.get("tool") or etype
            args = ev.get("arguments") or ev.get("input") or {}
            tool_lines.append(f"{name}({str(args)[:200]})")
        elif etype == "user.message":
            txt = (ev.get("message") or "").strip()
            if txt:
                conversation.append({"role": "user", "content": txt})
    return {
        "task_result": last_assistant,
        "conversation": conversation,
        "execution_summary": "\n".join(tool_lines[-60:]),
    }

if CC.is_file():
    data = parse_claude_code(CC.read_text())
elif CX.is_file():
    data = parse_codex(CX.read_text())
else:
    data = None

if data is not None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data))
    print(f"[test.sh] synthesised result.json from {CC if CC.is_file() else CX}")
PYEOF
fi

if [ ! -f "$RESULT_FILE" ]; then
    echo "No result.json - agent may have crashed."
    echo 0 > "$REWARD_FILE"
    exit 0
fi

# Use skill-phase1.json (same config as agent) for api_key/base_url when not in env
CONFIG_FILE="/harness/configs/skill-phase1.json"
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
