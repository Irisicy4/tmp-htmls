#!/usr/bin/env python3
"""
Convert tasks from cocoa-agent format to Harbor format.

Source format (cocoa-agent):
  task-XX-name/
    task.yaml        -> instruction + metadata comment
    test.py          -> LLM-as-judge evaluator
    Dockerfile       -> sandbox image (not used directly by Harbor)
    docker-compose.yaml

Target format (Harbor):
  task-XX-name/
    instruction.md   -> extracted from task.yaml
    task.toml        -> Harbor config (from metadata)
    tests/
      test.sh        -> standard verifier wrapper
      test_task.py   -> copied from test.py
    environment/
      Dockerfile     -> Harbor Dockerfile (cocoa-agent + harness)
      docker-compose.yaml
"""

import os
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

TASK_TOML_TEMPLATE = '''version = "1.0"

[metadata]
category = "{category}"
tags = ["evolvebench", "{ability}", "{domain_tag}", "{lang}"]

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 300.0

[verifier.env]
OPENAI_API_KEY = "${{OPENAI_API_KEY}}"
OPENAI_BASE_URL = "${{LLM_BASE_URL}}"

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240
allow_internet = true
# 20 min for Modal: Playwright + Chromium install is slow on cloud build
build_timeout_sec = 1200.0
'''

TEST_SH = r'''#!/bin/bash
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
'''

HARBOR_DOCKERFILE = '''# Self-contained environment for running agents.
# Works with both Harbor Docker (-e docker) and Harbor Modal (-e modal).

FROM ghcr.io/agent-infra/sandbox:latest

# Clone cocoabench/cocoa-agent + install deps
RUN git clone --depth 1 https://github.com/cocoabench/cocoa-agent.git /cocoa-agent-src \\
    && /opt/python3.12/bin/pip install -r /cocoa-agent-src/requirements.txt \\
       agent-sandbox "anthropic>=0.75.0" "colorama>=0.4.6" "google-genai>=0.5.0" \\
       "openai>=2.6.1" "pyyaml>=6.0.3" "websocket-client>=1.9.0" playwright --quiet \\
    && /opt/python3.12/bin/playwright install chromium \\
    && ln -sf /cocoa-agent-src /cocoa-agent

# --- Harness (agent-agnostic) ---
COPY run_task.py skill_store.py skill_extractor.py /harness/
COPY adapters/ /harness/adapters/
COPY configs/ /harness/configs/
COPY skills/ /skills/

WORKDIR /harness
'''

DOCKER_COMPOSE = '''# Build context is this directory (environment/).
# All glue files are co-located — no repo-root context needed.
services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
'''


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_task_yaml(yaml_path: Path) -> dict:
    """Parse task.yaml to extract instruction and metadata."""
    text = yaml_path.read_text()

    # Extract instruction (everything after "instruction: |" until the comment line)
    instruction_match = re.search(
        r'instruction:\s*\|?\s*\n(.*?)(?=\n#\s*Row|\Z)',
        text, re.DOTALL
    )
    if instruction_match:
        raw = instruction_match.group(1)
        # Split into lines first, then strip leading/trailing blank lines
        lines = raw.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        # Remove common leading indentation (typically 2 spaces from YAML block scalar)
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        min_indent = min(indents) if indents else 0
        instruction = '\n'.join(
            line[min_indent:] if len(line) >= min_indent else line
            for line in lines
        )
    else:
        # Try simple single-line instruction
        m = re.search(r'instruction:\s*(.+)', text)
        instruction = m.group(1).strip().strip('"').strip("'") if m else ""

    # Extract metadata from comment line: # Row N | ability: X | domain: Y | lang: Z
    metadata = {"ability": "web_navigation", "domain": "General", "lang": "en"}
    meta_match = re.search(r'#\s*Row\s+\d+\s*\|(.+)', text)
    if meta_match:
        meta_str = meta_match.group(1)
        for field in meta_str.split('|'):
            field = field.strip()
            if ':' in field:
                key, val = field.split(':', 1)
                metadata[key.strip()] = val.strip()

    return {"instruction": instruction, "metadata": metadata}


def convert_task(src_dir: Path, dst_dir: Path):
    """Convert a single task from cocoa-agent format to Harbor format."""
    task_name = src_dir.name

    # Parse task.yaml
    task_yaml = src_dir / "task.yaml"
    if not task_yaml.exists():
        print(f"  [skip] {task_name}: no task.yaml")
        return False

    parsed = parse_task_yaml(task_yaml)
    instruction = parsed["instruction"]
    meta = parsed["metadata"]

    if not instruction:
        print(f"  [skip] {task_name}: empty instruction")
        return False

    # Create output directories
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "tests").mkdir(exist_ok=True)
    (dst_dir / "environment").mkdir(exist_ok=True)

    # Write instruction.md
    (dst_dir / "instruction.md").write_text(instruction + "\n")

    # Write task.toml
    category = meta.get("domain", "General")
    ability = meta.get("ability", "web_navigation")
    lang = meta.get("lang", "en")
    domain_tag = category.lower().replace(" & ", "_").replace(" ", "_")

    toml_content = TASK_TOML_TEMPLATE.format(
        category=category,
        ability=ability,
        domain_tag=domain_tag,
        lang=lang,
    )
    (dst_dir / "task.toml").write_text(toml_content)

    # Copy test.py -> tests/test_task.py
    src_test = src_dir / "test.py"
    if src_test.exists():
        shutil.copy2(src_test, dst_dir / "tests" / "test_task.py")
    else:
        print(f"  [warn] {task_name}: no test.py")

    # Write tests/test.sh
    (dst_dir / "tests" / "test.sh").write_text(TEST_SH)
    os.chmod(dst_dir / "tests" / "test.sh", 0o755)

    # Write environment/Dockerfile
    (dst_dir / "environment" / "Dockerfile").write_text(HARBOR_DOCKERFILE)

    # Write environment/docker-compose.yaml
    (dst_dir / "environment" / "docker-compose.yaml").write_text(DOCKER_COMPOSE)

    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_tasks.py <src_dir> <dst_dir>")
        print("  src_dir: directory containing cocoa-agent format tasks")
        print("  dst_dir: output directory for Harbor format tasks")
        sys.exit(1)

    src_root = Path(sys.argv[1])
    dst_root = Path(sys.argv[2])

    if not src_root.is_dir():
        print(f"Error: {src_root} is not a directory")
        sys.exit(1)

    # Find all task directories
    task_dirs = sorted([
        d for d in src_root.iterdir()
        if d.is_dir() and d.name.startswith("task-")
    ])

    print(f"Found {len(task_dirs)} tasks in {src_root}")
    print(f"Output: {dst_root}")
    print()

    converted = 0
    for task_dir in task_dirs:
        dst_task = dst_root / task_dir.name
        if convert_task(task_dir, dst_task):
            converted += 1
            print(f"  [ok] {task_dir.name}")

    print(f"\nConverted {converted}/{len(task_dirs)} tasks")


if __name__ == "__main__":
    main()
