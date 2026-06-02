"""Grounded (deterministic) judging spec for task-78-go-to-notion-weekly-planner-fill-sections.

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
      "test_grounded", "tasks/real_118/task-78-go-to-notion-weekly-planner-fill-sections/tests/test_grounded.py"
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
    spec = _ilu.spec_from_file_location(f"{here.name}_test_task", target)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm = _load_sibling_test_task()

# --- Soft-rubric data: inherited from the LLM-as-judge sibling --------
DIMENSION_WEIGHTS = dict(_llm.DIMENSION_WEIGHTS)
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = _llm.USER_PROMPT_TEMPLATE   # includes the per-axis 1-5 anchors

from agentic_judge.grounded.framework.constraints import (
    JSONSchemaConforms, ContainsAllSubstrings, ListLengthInRange,
    FieldPresent, Custom,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("template_url",          "string",       "Should be the Notion weekly planner template URL."),
        RequiredField("sections_filled",       "list[object]", "Each section you filled in."),
        RequiredField("sections_filled[].name","string",       "Section label as shown on the planner."),
        RequiredField("sections_filled[].content_preview", "string", "First ~120 chars of the content you filled in."),
        RequiredField("sections_filled[].placeholder_or_real", "string", "'placeholder' (fake but realistic) or 'real'."),
        RequiredField("completion_evidence",   "string",       "URL to your filled copy OR path to a screenshot."),
    ],
    optional=[
        OptionalField("sections_total",        "integer",      "Sections found on the template, filled or not."),
        OptionalField("sections_skipped",      "list[string]", "Names of sections you couldn't access/fill."),
        OptionalField("editing_method",        "string",       "How you edited (duplicate template + edit, fork, etc.)."),
        OptionalField("notion_account_used",   "boolean",      "Did you sign into a Notion account?"),
    ],
    examples={
        "template_url": "https://www.notion.so/templates/weekly-planner",
        "sections_filled": [
            {"name": "Goals", "content_preview": "1. Ship Q3 OKR ...",
             "placeholder_or_real": "placeholder"},
            {"name": "Monday tasks", "content_preview": "Standup @ 9, ...",
             "placeholder_or_real": "placeholder"},
        ],
        "completion_evidence": "/output/notion_filled.png",
    },
)

def _has_canonical_sections(summary, ctx):
    rs = [(s or {}).get("name", "").lower() for s in (summary.get("sections_filled") or [])]
    expected_any = {"goals", "tasks", "notes"}
    hit = expected_any & {n for n in rs} or set()
    # Loose: also accept partial matches
    for n in rs:
        for e in ("goal", "task", "note", "monday", "tuesday", "wednesday",
                   "thursday", "friday", "weekly", "priority"):
            if e in n:
                hit.add(e)
    if len(hit) >= 3:
        return True, f"covered {sorted(hit)}"
    return False, f"only covered {sorted(hit)}; need ≥3 of goals/tasks/notes/days"

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("template_url", ["notion.so"], name="template_url_on_notion"),
    ListLengthInRange("sections_filled", 3, 50, name="at_least_3_sections_filled"),
    Custom("canonical_sections_covered", _has_canonical_sections),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    if (t := summary.get("template_url")):
        out.append({"url": t, "claim": "Notion"})
    return out


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
