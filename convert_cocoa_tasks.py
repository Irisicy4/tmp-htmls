#!/usr/bin/env python3
"""Convert cocoa-agent sample tasks to harbor format.

Reads from: /Users/cuiyuanxiuchen/Desktop/cocoa-agent/sample-50-synthetic-tasks
Writes to:  tasks/cocoa-synthetic-50/

Each task gets:
  instruction.md        — plain task text
  task.toml             — metadata + resource config
  tests/test_task.py    — copied from cocoa-agent test.py (already harbor-compatible)
  tests/test.sh         — standard harbor verifier shell script
  environment/Dockerfile
  environment/docker-compose.yaml
"""

import re
import shutil
from pathlib import Path

SRC_DIR = Path("/Users/cuiyuanxiuchen/Desktop/cocoa-agent/sample-50-synthetic-tasks")
DST_DIR = Path("tasks/cocoa-synthetic-50")

# Grab standard files from an existing harbor task
TEMPLATE_TASK = Path("tasks/updated-deprivacy-100/task-01-im-looking-for-backpack-under")
TEMPLATE_TEST_SH = TEMPLATE_TASK / "tests" / "test.sh"
TEMPLATE_DOCKERFILE = TEMPLATE_TASK / "environment" / "Dockerfile"
TEMPLATE_DOCKER_COMPOSE = TEMPLATE_TASK / "environment" / "docker-compose.yaml"

TASK_TOML_TEMPLATE = """\
version = "1.0"

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
"""


def parse_task_yaml(yaml_content: str):
    """Return (instruction, ability, domain, lang) from task.yaml content."""
    # Parse instruction block using PyYAML
    import yaml
    data = yaml.safe_load(yaml_content)
    instruction = (data.get("instruction") or "").strip()

    # Parse metadata from the YAML comment line
    ability = "web_navigation"
    domain = "General"
    lang = "en"

    comment_match = re.search(
        r"#\s*Row\s*\d+\s*\|"
        r"\s*ability:\s*([^|]+)\|"
        r"\s*domain:\s*([^|]+)\|"
        r"\s*lang:\s*(\w+)",
        yaml_content,
    )
    if comment_match:
        ability = comment_match.group(1).strip()
        domain = comment_match.group(2).strip()
        lang = comment_match.group(3).strip()

    return instruction, ability, domain, lang


def domain_to_tag(domain: str) -> str:
    return (
        domain.lower()
        .replace(" & ", "_and_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(d for d in SRC_DIR.iterdir() if d.is_dir())
    converted = 0

    for task_src in task_dirs:
        yaml_file = task_src / "task.yaml"
        test_py_file = task_src / "test.py"

        if not yaml_file.exists() or not test_py_file.exists():
            print(f"  SKIP {task_src.name}: missing task.yaml or test.py")
            continue

        task_name = task_src.name
        task_dst = DST_DIR / task_name

        instruction, ability, domain, lang = parse_task_yaml(yaml_file.read_text())

        # Create dirs
        (task_dst / "tests").mkdir(parents=True, exist_ok=True)
        (task_dst / "environment").mkdir(parents=True, exist_ok=True)

        # instruction.md
        (task_dst / "instruction.md").write_text(instruction + "\n")

        # task.toml
        toml = TASK_TOML_TEMPLATE.format(
            category=domain,
            ability=ability,
            domain_tag=domain_to_tag(domain),
            lang=lang,
        )
        (task_dst / "task.toml").write_text(toml)

        # tests/test_task.py — already in harbor-compatible format
        shutil.copy2(test_py_file, task_dst / "tests" / "test_task.py")

        # tests/test.sh — standard harbor verifier
        shutil.copy2(TEMPLATE_TEST_SH, task_dst / "tests" / "test.sh")

        # environment/
        shutil.copy2(TEMPLATE_DOCKERFILE, task_dst / "environment" / "Dockerfile")
        shutil.copy2(TEMPLATE_DOCKER_COMPOSE, task_dst / "environment" / "docker-compose.yaml")

        converted += 1
        print(f"  OK  {task_name}  [{domain} | {lang}]")

    print(f"\nDone. Converted {converted} tasks → {DST_DIR}")


if __name__ == "__main__":
    main()
