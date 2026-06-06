"""Grounded (deterministic) judging spec for task-12-please-compose-an-email-to.

Borrowed from gpt5-v3-11tasks's programmatic_checks for the email task:
  recipient_correct, subject_present, body_present.

Soft-rubric data (DIMENSIONS / DIMENSION_WEIGHTS / TASK_RUBRIC) is
inherited from the sibling test_task.py at import time.
"""
import importlib.util as _ilu
import pathlib as _pth
import sys as _sys


def _load_sibling_test_task():
    here = _pth.Path(__file__).resolve().parent
    target = here / "test_task.py"
    spec = _ilu.spec_from_file_location(f"{here.name}_test_task", target)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm = _load_sibling_test_task()
# Some test_task.py variants expose DIMENSION_WEIGHTS as a dict; others
# expose a DIMENSIONS list + weights baked into a RUBRIC string.  Handle
# both shapes — the test_grounded layer only needs SOMETHING to render
# to the LLM judge call, plus the dim name list.
DIMENSION_WEIGHTS = dict(getattr(_llm, "DIMENSION_WEIGHTS",
                                   {d: 1.0 / len(_llm.DIMENSIONS) for d in _llm.DIMENSIONS}))
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = getattr(_llm, "USER_PROMPT_TEMPLATE",
                       getattr(_llm, "RUBRIC", ""))


from agentic_judge.grounded.framework.constraints import (
    HardConstraint, EqualsValue, FieldPresent, JSONSchemaConforms,
    ContainsAllSubstrings, Custom,
)
from agentic_judge.grounded.framework.summary_schema import (
    SummarySchema, RequiredField, OptionalField,
)


SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("recipient",  "string", "Recipient email address (should be dgs@bu.edu)."),
        RequiredField("subject",    "string", "Email subject line."),
        RequiredField("body",       "string", "Full email body text."),
        RequiredField("dates_mentioned", "list[string]", "Dates offered in the body (Nov 29 / Nov 30)."),
        RequiredField("includes_cold_warning", "boolean", "True if you included the cold-weather caution per the brief."),
        RequiredField("sent",       "boolean", "Did you click send? (Should be false — task says stop before sending.)"),
    ],
    optional=[
        OptionalField("salutation", "string", "Opening line."),
        OptionalField("signature",  "string", "Sign-off line."),
        OptionalField("draft_path", "string", "Path where the draft was saved, if anywhere."),
    ],
    examples={
        "recipient": "dgs@bu.edu",
        "subject": "Java one-day class — Nov 29 or 30 availability",
        "body": "Dear Mr. Sullivan, ...",
        "dates_mentioned": ["November 29", "November 30"],
        "includes_cold_warning": True,
        "sent": False,
    },
)


def _has_either_date(summary, ctx):
    text = " ".join([
        str((summary.get("body") or "")),
        " ".join(summary.get("dates_mentioned") or []),
    ]).lower()
    have_29 = "november 29" in text or "nov 29" in text or "11/29" in text or "11-29" in text
    have_30 = "november 30" in text or "nov 30" in text or "11/30" in text or "11-30" in text
    if have_29 and have_30:
        return True, "both Nov 29 AND Nov 30 mentioned"
    return False, f"missing one of Nov 29 / Nov 30 (have_29={have_29}, have_30={have_30})"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("recipient", "dgs@bu.edu", case_insensitive=True, name="recipient_correct"),
    FieldPresent("subject", name="subject_present"),
    FieldPresent("body", name="body_present"),
    EqualsValue("sent", False, name="not_actually_sent"),
    Custom("both_dates_offered", _has_either_date),
]


def FAITHFULNESS_CHECKS(summary):
    return []  # no URLs to verify for this task


def grade(result):
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
                    agent_text = c; break
    summary, src = extract_summary_json(agent_text)
    if summary is None:
        return {"summary_parsed": False, "summary_source": src,
                "hard_report": [{"name": "summary_json_parseable",
                                  "passed": False, "detail": f"could not extract JSON (src={src})"}],
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


def grade_with_llm(result):
    from agentic_judge.grounded.framework import grounded_judge_test
    return grounded_judge_test(result, _sys.modules[__name__])
