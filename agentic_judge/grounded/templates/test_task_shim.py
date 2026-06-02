"""Drop-in replacement for tasks/<id>/tests/test_task.py.

Imports the per-task spec from agentic_judge.grounded.task_specs.<id> and
delegates to grounded_judge_test.  Keeps the `test(result) -> dict`
signature that evolve_bench_harbor/tests/test.sh expects.
"""
# Each task copies this file as tests/test_task.py and sets _SPEC_MODULE to
# the spec name (e.g. "task_01_backpack").
from importlib import import_module

_SPEC_MODULE = "task_XX_replace_me"


def test(result: dict) -> dict:
    from agentic_judge.grounded.framework import grounded_judge_test
    spec = import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")
    return grounded_judge_test(result, spec)
