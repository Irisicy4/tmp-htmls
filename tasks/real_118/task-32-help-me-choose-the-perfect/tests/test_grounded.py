"""Grounded (deterministic) judging spec for task-32-help-me-choose-the-perfect.

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
      "test_grounded", "tasks/real_118/task-32-help-me-choose-the-perfect/tests/test_grounded.py"
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
    JSONSchemaConforms, ListLengthInRange, AllValuesBetween,
    AllURLsMatch, ContainsAllSubstrings, EqualsValue, FieldPresent,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("platform",               "string",       "Should be 'amazon.co.jp'."),
        RequiredField("recipient_profile",      "object",       "Snapshot of the recipient used."),
        RequiredField("recipient_profile.age",  "integer",      ""),
        RequiredField("recipient_profile.interests", "list[string]", ""),
        RequiredField("options",                "list[object]", "Exactly 3 chosen items."),
        RequiredField("options[].name",         "string",       "Product name."),
        RequiredField("options[].url",          "string",       "Amazon.co.jp product URL."),
        RequiredField("options[].asin",         "string",       "Amazon ASIN."),
        RequiredField("options[].price_jpy",    "integer",      "Price in JPY (yen, no symbol)."),
        RequiredField("options[].category",     "string",       "Category fit ('gaming'|'computers'|'gadgets')."),
        RequiredField("options[].fit_notes",    "string",       "Why it fits the profile."),
        RequiredField("options[].cart_action",  "string",       "'added'|'attempted'|'not_added' — be honest."),
        RequiredField("cart_url_or_screenshot", "string",       "Cart URL or path to cart screenshot."),
    ],
    optional=[
        OptionalField("options[].rating",       "number 1-5",    "Review rating."),
        OptionalField("options[].review_count", "integer",       ""),
        OptionalField("options[].prime_eligible", "boolean",     ""),
        OptionalField("options[].delivery_estimate", "string",   ""),
        OptionalField("budget_used_jpy",        "integer",       "Sum of the 3 prices."),
    ],
    examples={
        "platform": "amazon.co.jp",
        "recipient_profile": {"age": 28, "interests": ["gaming", "computers", "gadgets"]},
        "options": [
            {"name": "...", "url": "https://www.amazon.co.jp/dp/B0...", "asin": "B0XXXX",
             "price_jpy": 12800, "category": "gaming", "fit_notes": "...", "cart_action": "added"}
        ],
        "cart_url_or_screenshot": "https://www.amazon.co.jp/gp/cart/view.html",
    },
)

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("platform", "amazon.co.jp", case_insensitive=True, name="platform_is_amazon_jp"),
    ListLengthInRange("options", 3, 3, name="exactly_3_options"),
    AllValuesBetween("options", "price_jpy", 10000, 15000, name="all_prices_in_10k_15k_jpy"),
    AllURLsMatch("options", "url", r"amazon\.co\.jp", name="all_urls_amazon_jp"),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    out = []
    for opt in (summary.get("options") or [])[:3]:
        url = (opt or {}).get("url")
        if url:
            # Verify ASIN appears in page text
            asin = (opt.get("asin") or "")
            out.append({"url": url, "claim": asin or (opt.get("name") or "")[:20]})
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
