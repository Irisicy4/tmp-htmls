"""Grounded (deterministic) judging spec for task-106-go-to-airbnb-and.

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
      "test_grounded", "tasks/real_118/task-106-go-to-airbnb-and/tests/test_grounded.py"
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
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch, FieldPresent, Custom,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("airbnb_listings",                "list[object]", "Exactly 3 Airbnb listings."),
        RequiredField("airbnb_listings[].title",        "string",       ""),
        RequiredField("airbnb_listings[].url",          "string",       "Canonical airbnb.com URL."),
        RequiredField("airbnb_listings[].nightly_rate_usd","number",    "USD/night."),
        RequiredField("airbnb_listings[].available_nights_30d","integer","Next-30-day available nights."),
        RequiredField("airbnb_listings[].review_count", "integer",      "≥10."),
        RequiredField("airbnb_listings[].review_score", "number",       "Out of 5 (Airbnb) or 10 (VRBO)."),
        RequiredField("vrbo_listings",                  "list[object]", "Exactly 2 VRBO listings."),
        RequiredField("vrbo_listings[].title",          "string",       ""),
        RequiredField("vrbo_listings[].url",            "string",       "Canonical vrbo.com URL."),
        RequiredField("vrbo_listings[].nightly_rate_usd","number",      ""),
        RequiredField("airdna.avg_daily_rate_usd",      "number",       ""),
        RequiredField("airdna.occupancy_rate",          "number 0-1",   "Decimal fraction."),
        RequiredField("airdna.source_url",              "string",       ""),
        RequiredField("revenue_table",                  "list[object]", "ROI rows for the 5 listings."),
        RequiredField("revenue_table[].platform",       "string",       "'airbnb'|'vrbo'."),
        RequiredField("revenue_table[].monthly_revenue_usd","number",   ""),
        RequiredField("revenue_table[].roi_comment",    "string",       ""),
    ],
    optional=[
        OptionalField("airbnb_listings[].guests_capacity","integer",    ""),
        OptionalField("airbnb_listings[].entire_home",  "boolean",      ""),
        OptionalField("vrbo_listings[].review_count",   "integer",      ""),
        OptionalField("vrbo_listings[].review_score",   "number",       ""),
        OptionalField("methodology_notes",              "string",       ""),
    ],
    examples={
        "airbnb_listings": [],
        "vrbo_listings": [],
        "airdna": {"avg_daily_rate_usd": 162, "occupancy_rate": 0.72,
                     "source_url": "https://www.airdna.co/vacation-rental-data/app/pt/lisbon/lisbon/overview"},
        "revenue_table": [],
    },
)

def _all_reviews_ge_10(summary, ctx):
    bad = [i for i, l in enumerate(summary.get("airbnb_listings") or [])
            if (l.get("review_count") or 0) < 10]
    if bad:
        return False, f"items {bad} have <10 reviews"
    return True, "all Airbnb listings have ≥10 reviews"

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("airbnb_listings", 3, 3, name="exactly_3_airbnb"),
    ListLengthInRange("vrbo_listings", 2, 2, name="exactly_2_vrbo"),
    AllURLsMatch("airbnb_listings", "url", r"airbnb\.", name="airbnb_urls_canonical"),
    AllURLsMatch("vrbo_listings", "url", r"vrbo\.", name="vrbo_urls_canonical"),
    Custom("airbnb_reviews_ge_10", _all_reviews_ge_10),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for l in (summary.get("airbnb_listings") or [])[:3]:
        if (u := l.get("url")):
            out.append({"url": u, "claim": (l.get("title") or "")[:25]})
    for l in (summary.get("vrbo_listings") or [])[:2]:
        if (u := l.get("url")):
            out.append({"url": u, "claim": (l.get("title") or "")[:25]})
    if (a := (summary.get("airdna") or {}).get("source_url")):
        out.append({"url": a, "claim": "Lisbon"})
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
