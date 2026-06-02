"""Grounded (deterministic) judging spec for task-01-im-looking-for-backpack-under.

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
      "test_grounded", "tasks/real_118/task-01-im-looking-for-backpack-under/tests/test_grounded.py"
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
    HardConstraint, AllValuesLE, ListLengthInRange, AllURLsMatch,
    FieldPresent, JSONSchemaConforms, Custom,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("reference_product.url",       "string",          "Canonical Amazon URL of the original backpack."),
        RequiredField("reference_product.name",      "string",          "Product name as shown on the page."),
        RequiredField("reference_product.features",  "list[string]",    "Key features pulled from the original listing."),
        RequiredField("options",                     "list[object]",    "3-5 alternative backpacks."),
        RequiredField("options[].name",              "string",          "Product name."),
        RequiredField("options[].url",               "string",          "Canonical product URL (Amazon, manufacturer, etc.)."),
        RequiredField("options[].price",             "number (USD)",    "Selling price in USD."),
        RequiredField("options[].features",          "list[string]",    "3+ feature bullets."),
        RequiredField("options[].comparison_notes",  "string",          "How it compares to the original (pros and cons)."),
    ],
    optional=[
        OptionalField("options[].asin",              "string",          "Amazon ASIN if applicable."),
        OptionalField("options[].rating",            "number 1-5",      "Review rating."),
        OptionalField("options[].review_count",      "integer",         "Number of reviews."),
        OptionalField("options[].verified_in_stock", "boolean",         "True if you saw stock indicator."),
        OptionalField("reference_product.price_estimate","number (USD)","Your estimate of the original's MSRP."),
        OptionalField("recommendation",              "string",          "Your top pick and why."),
    ],
    examples={
        "reference_product": {"url": "https://www.amazon.com/dp/B09YRC9Y3G", "name": "...", "features": ["...", "..."]},
        "options": [
            {"name": "MATEIN ...", "url": "https://www.amazon.com/dp/B07...",
             "price": 64.99, "features": ["40L expandable", "USB port"], "comparison_notes": "..."}
        ],
    },
)

# --- hard constraints --------------------------------------------------------
HARD_CONSTRAINTS: list[HardConstraint] = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    ListLengthInRange("options", 3, 5, name="quantity_3_to_5"),
    AllValuesLE("options", "price", 75.0, name="all_prices_under_75_usd"),
    AllURLsMatch("options", "url", r"^https?://", name="every_option_has_url"),
    FieldPresent("reference_product.url", name="reference_url_present"),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    """Verify the reference URL plus a sample of option URLs and prices."""
    checks: list[dict] = []
    ref = (summary.get("reference_product") or {}).get("url")
    if ref:
        checks.append({"url": ref, "claim": "backpack"})  # weak check; site often blocks fetch
    for i, opt in enumerate(summary.get("options") or []):
        url = (opt or {}).get("url")
        if not url:
            continue
        # Confirm the page mentions the agent's claimed name
        claim = (opt.get("name") or "").split()[0] if opt.get("name") else ""
        checks.append({"url": url, "claim": claim})
        if i >= 2:  # cap at first 3 to limit fetch budget
            break
    return checks


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
