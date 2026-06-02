"""Grounded (deterministic) judging spec for task-101-go-to-sec-edgar-find.

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
      "test_grounded", "tasks/real_118/task-101-go-to-sec-edgar-find/tests/test_grounded.py"
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
    JSONSchemaConforms, ListLengthInRange, AllURLsMatch, FieldPresent,
    Custom,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("filings",                          "list[object]", "Three 10-K filings."),
        RequiredField("filings[].company",                "string",       ""),
        RequiredField("filings[].ticker",                 "string",       ""),
        RequiredField("filings[].filing_date",            "string",       "ISO date."),
        RequiredField("filings[].edgar_url",              "string",       "Canonical EDGAR filing URL."),
        RequiredField("filings[].climate_risk_quote",     "string",       "Verbatim disclosure paragraph."),
        RequiredField("related_lawsuits",                 "list[object]", "CourtListener matches OR empty list with explicit 'no_match=true'."),
        RequiredField("courtlistener_searched",           "boolean",      "True if you actually performed the search."),
    ],
    optional=[
        OptionalField("filings[].sector",                 "string",       "S&P 500 sector ('Energy' expected)."),
        OptionalField("filings[].form_type",              "string",       "'10-K'."),
        OptionalField("related_lawsuits[].case_name",     "string",       ""),
        OptionalField("related_lawsuits[].court",         "string",       ""),
        OptionalField("related_lawsuits[].status",        "string",       ""),
        OptionalField("related_lawsuits[].courtlistener_url","string",    ""),
        OptionalField("methodology_notes",                "string",       "How you narrowed S&P 500 energy."),
    ],
    examples={
        "filings": [
            {"company": "ExxonMobil", "ticker": "XOM", "filing_date": "2024-02-28",
             "edgar_url": "https://www.sec.gov/cgi-bin/browse-edgar?...",
             "climate_risk_quote": "..."}
        ],
        "related_lawsuits": [],
        "courtlistener_searched": True,
    },
)

def _sources_include_courtlistener(summary, ctx):
    if summary.get("courtlistener_searched"):
        return True, "courtlistener_searched=true"
    return False, "courtlistener was not searched"

def _quotes_present(summary, ctx):
    fs = summary.get("filings") or []
    bad = [i for i, f in enumerate(fs) if len((f.get("climate_risk_quote") or "").split()) < 8]
    if bad:
        return False, f"items {bad} have <8-word quotes"
    return True, f"all {len(fs)} quotes ≥ 8 words"

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("filings", 3, 3, name="exactly_3_filings"),
    AllURLsMatch("filings", "edgar_url", r"(sec\.gov|secdatabase\.com)",
                  name="all_edgar_urls_on_sec_gov"),
    Custom("courtlistener_searched", _sources_include_courtlistener),
    Custom("quotes_substantive", _quotes_present),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for f in (summary.get("filings") or [])[:3]:
        u = f.get("edgar_url")
        if not u:
            continue
        # Verify quote (first 6 words) appears on the page
        q = (f.get("climate_risk_quote") or "").strip()
        first_words = " ".join(q.split()[:6])
        out.append({"url": u, "claim": first_words or (f.get("ticker") or "")})
    for lw in (summary.get("related_lawsuits") or [])[:2]:
        if (u := (lw or {}).get("courtlistener_url")):
            out.append({"url": u, "claim": (lw.get("case_name") or "")[:30]})
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
