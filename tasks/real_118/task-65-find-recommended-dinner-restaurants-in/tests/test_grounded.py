"""Grounded (deterministic) judging spec for task-65-find-recommended-dinner-restaurants-in.

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
      "test_grounded", "tasks/real_118/task-65-find-recommended-dinner-restaurants-in/tests/test_grounded.py"
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
        RequiredField("location",                       "string",       "Should contain 'Tuscany'."),
        RequiredField("meal_type",                      "string",       "Should be 'dinner'."),
        RequiredField("date_range",                     "list[string]", "['2025-12-26','2025-12-27'] or similar (must include both days)."),
        RequiredField("restaurants",                    "list[object]", "Recommended dinner restaurants."),
        RequiredField("restaurants[].name",             "string",       ""),
        RequiredField("restaurants[].city",             "string",       "Town within Tuscany."),
        RequiredField("restaurants[].source_url",       "string",       "TripAdvisor/Google Maps/TheFork URL."),
        RequiredField("restaurants[].open_dec_26",      "boolean",      "Open dinner Dec 26?"),
        RequiredField("restaurants[].open_dec_27",      "boolean",      "Open dinner Dec 27?"),
        RequiredField("restaurants[].verification_method","string",     "How you verified the hours."),
    ],
    optional=[
        OptionalField("restaurants[].cuisine",          "string",       ""),
        OptionalField("restaurants[].price_band",       "string",       "$/$$/$$$/$$$$."),
        OptionalField("restaurants[].rating",           "number 0-5",   ""),
        OptionalField("restaurants[].reservation_url",  "string",       ""),
        OptionalField("sources_used",                   "list[string]", "All discovery platforms consulted."),
    ],
    examples={
        "location": "Tuscany, Italy", "meal_type": "dinner",
        "date_range": ["2025-12-26", "2025-12-27"],
        "restaurants": [
            {"name": "Trattoria ...", "city": "Florence",
             "source_url": "https://www.tripadvisor.com/...",
             "open_dec_26": True, "open_dec_27": True,
             "verification_method": "Site hours page says 'open daily incl. holidays'"}
        ],
    },
)

def _both_dates_present(summary, ctx):
    dr = summary.get("date_range") or []
    s = " ".join(str(x) for x in dr).lower()
    have_26 = "26" in s and ("dec" in s or "12-26" in s or "12/26" in s)
    have_27 = "27" in s and ("dec" in s or "12-27" in s or "12/27" in s)
    if have_26 and have_27:
        return True, "date_range covers both Dec 26 AND Dec 27"
    return False, f"date_range={dr!r} missing one of Dec 26 or Dec 27"

def _restaurants_open_both(summary, ctx):
    rs = summary.get("restaurants") or []
    bad = [i for i, r in enumerate(rs)
           if not (r.get("open_dec_26") and r.get("open_dec_27"))]
    if bad:
        return False, f"items {bad} not confirmed open on BOTH days"
    return True, f"all {len(rs)} restaurants confirmed open on both days"

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("location", ["tuscany"], name="location_in_tuscany"),
    ContainsAllSubstrings("meal_type", ["dinner"], name="meal_type_dinner"),
    Custom("date_range_covers_dec_26_and_27", _both_dates_present),
    Custom("all_restaurants_open_both_days", _restaurants_open_both),
    ListLengthInRange("restaurants", 1, 20, name="at_least_one_restaurant"),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for r in (summary.get("restaurants") or [])[:5]:
        if (u := r.get("source_url")):
            out.append({"url": u, "claim": (r.get("name") or "")[:25]})
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
