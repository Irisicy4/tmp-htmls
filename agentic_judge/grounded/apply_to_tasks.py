#!/usr/bin/env python3
"""One-time installer: for each of the 10 wired tasks, append the addendum
to instruction.md and replace tests/test_task.py with the grounded shim.

Idempotent: re-running won't append the addendum twice (checks for marker).
"""
from __future__ import annotations

import importlib
import pathlib

# repo root (relative to this script)
ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS_ROOT = ROOT / "tasks" / "real_118"

# (task_dir_name, spec_module)
WIRED = [
    ("task-01-im-looking-for-backpack-under",        "task_01_backpack"),
    ("task-05-go-to-nbacom-and-check",               "task_05_nba"),
    ("task-15-help-me-get-all-2025",                 "task_15_douban"),
    ("task-32-help-me-choose-the-perfect",           "task_32_amazon_jp"),
    ("task-65-find-recommended-dinner-restaurants-in","task_65_tuscany"),
    ("task-78-go-to-notion-weekly-planner-fill-sections","task_78_notion"),
    ("task-101-go-to-sec-edgar-find",                "task_101_sec"),
    ("task-104-go-to-redfin-search",                 "task_104_redfin"),
    ("task-106-go-to-airbnb-and",                    "task_106_airbnb"),
    ("task-117-go-to-the-fdas",                      "task_117_fda"),
]

MARKER = "<!-- agentic_judge.grounded addendum -->"

SHIM_TEMPLATE = '''"""Grounded-judge shim — delegates to
agentic_judge.grounded.framework.grounded_judge_test using the per-task spec
{spec_module}.

TASK_INSTRUCTION is a triple-quoted string copy of instruction.md (body +
agentic_judge.grounded addendum) so that scripts_batch/deprivacy_modal_runner.py
can extract it via AST and feed it to the agent unchanged.
"""
from importlib import import_module


_SPEC_MODULE = "{spec_module}"


TASK_INSTRUCTION = {instruction_literal}


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{{_SPEC_MODULE}}")
    return grounded_judge_test(result, spec)
'''


def apply(task_dir: str, spec_module: str) -> None:
    td = TASKS_ROOT / task_dir
    assert td.is_dir(), f"missing task dir {td}"
    inst = td / "instruction.md"
    tests_dir = td / "tests"
    assert inst.is_file()
    assert tests_dir.is_dir()

    # 1) Append addendum if not already present
    body = inst.read_text(encoding="utf-8")
    if MARKER not in body:
        spec = importlib.import_module(
            f"agentic_judge.grounded.task_specs.{spec_module}")
        addendum = spec.SUMMARY_SCHEMA.to_instruction_addendum()
        inst.write_text(body.rstrip() + "\n" + MARKER + "\n" + addendum,
                          encoding="utf-8")
        print(f"  instruction.md: appended addendum ({len(addendum)} chars)")
    else:
        print(f"  instruction.md: marker present, skipping addendum")

    # 2) Replace tests/test_task.py with a shim that embeds the FULL
    # instruction.md (so the Modal runner's ast.literal_eval extraction
    # gives the agent the addendum).  Use repr() to produce a clean
    # python string literal regardless of quote/escape contents.
    full_instruction = inst.read_text(encoding="utf-8")
    new_tp = tests_dir / "test_task.py"
    new_tp.write_text(
        SHIM_TEMPLATE.format(
            spec_module=spec_module,
            instruction_literal=repr(full_instruction),
        ),
        encoding="utf-8",
    )
    print(f"  tests/test_task.py: replaced with grounded shim ({spec_module}, "
          f"{len(full_instruction)} chars of TASK_INSTRUCTION)")


def main() -> int:
    import sys
    # Add repo root to path so imports work
    sys.path.insert(0, str(ROOT))
    for task_dir, spec in WIRED:
        print(f"=== {task_dir} ({spec}) ===")
        apply(task_dir, spec)
    print(f"\nApplied to {len(WIRED)} tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
