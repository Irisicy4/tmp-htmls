#!/usr/bin/env python3
"""Rebuild tasks/real_118/<task>/tests/test_grounded.py from the per-task
spec module currently in agentic_judge/grounded/task_specs/.

After this script runs:
  - Each task's tests/test_grounded.py contains the per-task spec (schema,
    hard constraints, faithfulness, rubric) AND a generic `grade(result)`
    entry point that calls the framework runner.
  - agentic_judge/grounded/task_specs/ becomes obsolete; the discovery
    function in agentic_judge.grounded.run_grade reads tests/test_grounded.py
    files instead.

Idempotent.  test_task.py is never touched.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "tasks" / "real_118"
SPECS_PKG = "agentic_judge.grounded.task_specs"

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


HEADER = '''"""Grounded (deterministic) judging spec for {task_id}.

Lives alongside the existing LLM-as-judge `test_task.py` — same `tests/`
folder, two independent grading layers:

  * test_task.py     — LLM-as-judge (Harbor's authoritative verdict).
                       Source of truth for the soft rubric.
  * test_grounded.py — typed hard-constraint predicates + URL faithfulness
                       verification (this file).  Adds machine-checkable
                       gating; does NOT redefine the soft rubric.

The soft-rubric data (DIMENSIONS / DIMENSION_WEIGHTS / TASK_RUBRIC) is
**inherited from the sibling test_task.py** at import time — if you tune
the LLM judge's anchors, this file picks up the change automatically.

Only the constraint-specific data — SUMMARY_SCHEMA, HARD_CONSTRAINTS,
FAITHFULNESS_CHECKS — is defined per task in this file.

Call via:

  spec = importlib.util.spec_from_file_location(
      "test_grounded", "tasks/real_118/{task_id}/tests/test_grounded.py"
  )
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  m.grade(result)        # hard + faithfulness, no LLM
  m.grade_with_llm(result)  # full mix incl. effectiveness LLM call
"""
import importlib.util as _ilu
import pathlib as _pth
import sys as _sys


def _load_sibling_test_task():
    """Load tests/test_task.py from this directory; tests/ is not a package
    so we use importlib.util by file path."""
    here = _pth.Path(__file__).resolve().parent
    target = here / "test_task.py"
    spec = _ilu.spec_from_file_location(f"{{here.name}}_test_task", target)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm = _load_sibling_test_task()

# --- Soft-rubric data: inherited from the LLM-as-judge sibling --------
DIMENSION_WEIGHTS = dict(_llm.DIMENSION_WEIGHTS)
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = _llm.USER_PROMPT_TEMPLATE   # includes the per-axis 1-5 anchors
'''

FOOTER = '''

# --- generic entry points (don't customise per task) -----------------

def grade(result: dict) -> dict:
    """Hard constraints + faithfulness only; no LLM call."""
    from agentic_judge.grounded.framework.extractor import extract_summary_json
    from agentic_judge.grounded.framework.faithfulness import (
        verify_url_claims, aggregate as _agg_faith,
    )
    agent_text = result.get("task_result") or ""
    if not isinstance(agent_text, str) or not agent_text.strip():
        for m in reversed(result.get("conversation") or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content") or ""
                if isinstance(c, str) and len(c) > 20:
                    agent_text = c
                    break
    summary, src = extract_summary_json(agent_text)
    if summary is None:
        return {"summary_parsed": False, "summary_source": src,
                "hard_report": [{"name": "summary_json_parseable",
                                  "passed": False,
                                  "detail": f"could not extract JSON (src={src})"}],
                "faithfulness_report": {}}
    hard = [c.check(summary) for c in HARD_CONSTRAINTS]
    hard_report = [r.to_dict() for r in hard]
    faith = {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}
    try:
        findings = verify_url_claims(FAITHFULNESS_CHECKS(summary))
        faith = _agg_faith(findings)
    except Exception as e:
        faith["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return {"summary_parsed": True, "summary_source": src,
            "summary_json": summary, "hard_report": hard_report,
            "hard_pass_rate": sum(1 for r in hard if r.passed) / max(len(hard), 1),
            "faithfulness_report": faith}


def grade_with_llm(result: dict) -> dict:
    """Full mix: hard + faithfulness + LLM effectiveness judge."""
    from agentic_judge.grounded.framework import grounded_judge_test
    return grounded_judge_test(result, _sys.modules[__name__])


def main() -> int:
    import json, sys
    if len(sys.argv) != 2:
        print("usage: test_grounded.py <result.json>", file=sys.stderr)
        return 2
    print(json.dumps(grade(json.load(open(sys.argv[1]))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from importlib import import_module

    for task_dir_name, spec_module in WIRED:
        spec = import_module(f"{SPECS_PKG}.{spec_module}")
        out = TASKS_ROOT / task_dir_name / "tests" / "test_grounded.py"

        spec_src_path = pathlib.Path(inspect.getsourcefile(spec))
        spec_src = spec_src_path.read_text()
        # Convert relative imports inside the spec to absolute imports
        spec_src = spec_src.replace(
            "from ..framework.constraints import",
            "from agentic_judge.grounded.framework.constraints import")
        spec_src = spec_src.replace(
            "from ..framework.summary_schema import",
            "from agentic_judge.grounded.framework.summary_schema import")

        # Strip the leading docstring (we replace it with our header)
        import re as _re
        spec_src = _re.sub(r'^"""[\s\S]*?"""\s*\n', "", spec_src, count=1)

        # The per-task spec previously declared DIMENSIONS / DIMENSION_WEIGHTS /
        # TASK_RUBRIC / TASK_INSTRUCTION inline.  We now inherit those from
        # sibling test_task.py — so strip those four blocks from the spec body
        # to avoid double-definition (the header has the inherits-from logic).
        for var in ("TASK_INSTRUCTION", "DIMENSIONS", "DIMENSION_WEIGHTS",
                    "TASK_RUBRIC"):
            # Single-name assignment: NAME = <python literal>, possibly
            # spread over multiple lines, terminating at the next top-level
            # assignment or a blank line followed by a top-level identifier.
            spec_src = _re.sub(
                rf'^{var}\s*=\s*(?:\(.*?\)|\[.*?\]|\{{.*?\}}|"""[\s\S]*?"""|[^\n]+)\s*\n',
                "", spec_src, count=1, flags=_re.M | _re.S)

        # Tidy up any double-blank lines introduced by the strip
        spec_src = _re.sub(r"\n{3,}", "\n\n", spec_src).strip()

        header = HEADER.format(task_id=task_dir_name,
                                slug=task_dir_name.replace("-", "_"))
        out.write_text(header + "\n" + spec_src + "\n" + FOOTER,
                        encoding="utf-8")
        print(f"  wrote {out.relative_to(REPO_ROOT)}  ({len(out.read_text())} chars)")

    print(f"\nGenerated {len(WIRED)} test_grounded.py files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
