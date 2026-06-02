"""Grounded (deterministic) judging spec for task-104-go-to-redfin-search.

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
      "test_grounded", "tasks/real_118/task-104-go-to-redfin-search/tests/test_grounded.py"
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
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch,
    AllValuesBetween, ContainsAllSubstrings, FieldPresent, Custom,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("filters_applied.city",         "string",       "Should be 'Austin'."),
        RequiredField("filters_applied.beds_min",     "integer",      ""),
        RequiredField("filters_applied.baths_min",    "integer",      ""),
        RequiredField("filters_applied.price_min",    "integer",      ""),
        RequiredField("filters_applied.price_max",    "integer",      ""),
        RequiredField("filters_applied.property_type","string",       "'single-family'."),
        RequiredField("listings",                     "list[object]", "5 listings, most-days-on-market first."),
        RequiredField("listings[].address",           "string",       ""),
        RequiredField("listings[].list_price",        "integer",      "USD."),
        RequiredField("listings[].days_on_market",    "integer",      ""),
        RequiredField("listings[].sqft",              "integer",      ""),
        RequiredField("listings[].price_per_sqft",    "number",       ""),
        RequiredField("listings[].agent_name",        "string",       ""),
        RequiredField("listings[].redfin_url",        "string",       ""),
        RequiredField("listings[].traviscad_url",     "string",       ""),
        RequiredField("listings[].assessed_value",    "integer",      "From TravisCAD."),
        RequiredField("listings[].la_ratio",          "number",       "list_price / assessed_value."),
        RequiredField("listings[].opportunity_signal","string",       ""),
    ],
    optional=[
        OptionalField("listings[].year_built",        "integer",      ""),
        OptionalField("listings[].tax_history",       "list[object]", "Per-year tax payments."),
        OptionalField("listings[].photos_url",        "string",       ""),
        OptionalField("methodology_notes",            "string",       ""),
    ],
    examples={
        "filters_applied": {"city": "Austin", "beds_min": 3, "baths_min": 2,
                             "price_min": 400000, "price_max": 600000,
                             "property_type": "single-family"},
        "listings": [{
            "address": "1234 Oak St, Austin, TX",
            "list_price": 549000, "days_on_market": 142, "sqft": 2100,
            "price_per_sqft": 261.4, "agent_name": "Jane Doe",
            "redfin_url": "https://www.redfin.com/TX/Austin/1234-Oak-St-78704/home/...",
            "traviscad_url": "https://search.traviscad.org/Property/View/...",
            "assessed_value": 575000, "la_ratio": 0.955,
            "opportunity_signal": "listed below assessed value",
        }],
    },
)

def _la_ratio_consistent(summary, ctx):
    bad = []
    for i, l in enumerate(summary.get("listings") or []):
        lp, av, lar = l.get("list_price"), l.get("assessed_value"), l.get("la_ratio")
        if None in (lp, av, lar) or av == 0:
            continue
        expected = float(lp) / float(av)
        if abs(expected - float(lar)) > 0.05:
            bad.append(f"item[{i}] computed={expected:.3f}, reported={lar:.3f}")
    if bad:
        return False, f"L/A ratio off: {bad[:3]}"
    return True, "all L/A ratios within ±0.05 of list/assessed"

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ContainsAllSubstrings("filters_applied.city", ["austin"], name="city_is_austin"),
    ListLengthInRange("listings", 5, 5, name="exactly_5_listings"),
    AllValuesBetween("listings", "list_price", 400000, 600000, name="all_in_400k_600k"),
    AllURLsMatch("listings", "redfin_url", r"redfin\.com", name="redfin_urls_canonical"),
    AllURLsMatch("listings", "traviscad_url", r"traviscad\.(org|com)|search\.traviscad",
                  name="traviscad_urls_present"),
    Custom("la_ratio_matches_inputs", _la_ratio_consistent),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for i, l in enumerate((summary.get("listings") or [])[:5]):
        ru = l.get("redfin_url")
        if ru:
            out.append({"url": ru, "claim": (l.get("address") or "")[:25]})
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
