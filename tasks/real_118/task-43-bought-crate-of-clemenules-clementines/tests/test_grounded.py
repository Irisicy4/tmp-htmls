"""Grounded (deterministic) judging spec for task-43-bought-crate-of-clemenules-clementines.

Borrowed from gpt5-v3-11tasks's programmatic_checks for the clementines task:
  normally_seedless_value (bool), normally_seedless_correct (must be True),
  has_source_url.

Per agronomy literature (e.g. UF/IFAS, USDA, CGN Clementines monograph), Clemenules
ARE normally seedless when grown isolated from other compatible mandarins; cross-
pollination from a nearby compatible variety is the standard cause of seed development.
So the agent's `normally_seedless` value should be True.

Soft-rubric data (DIMENSIONS / DIMENSION_WEIGHTS / TASK_RUBRIC) is inherited from
the sibling test_task.py at import time.
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
DIMENSION_WEIGHTS = dict(_llm.DIMENSION_WEIGHTS)
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = _llm.USER_PROMPT_TEMPLATE


from agentic_judge.grounded.framework.constraints import (
    HardConstraint, EqualsValue, FieldPresent, JSONSchemaConforms,
    AllURLsMatch, Custom,
)
from agentic_judge.grounded.framework.summary_schema import (
    SummarySchema, RequiredField, OptionalField,
)


SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("normally_seedless", "boolean",
                       "True if Clemenules clementines are normally seedless under standard agronomic conditions."),
        RequiredField("cross_pollination_explanation", "string",
                       "Brief explanation of how seeds end up in normally-seedless varieties (cross-pollination from compatible cultivars)."),
        RequiredField("source_url", "string",
                       "Authoritative source URL (university extension, FAO, USDA, peer-reviewed paper, etc.)."),
        RequiredField("evidence_quote", "string",
                       "Verbatim ≤30-word quote from the source supporting the claim."),
    ],
    optional=[
        OptionalField("alternative_sources", "list[string]",
                       "Additional source URLs you cross-checked."),
        OptionalField("conditions_for_seeds", "list[string]",
                       "Specific conditions that cause seed development (e.g. proximity to Murcott, Page, etc.)."),
    ],
    examples={
        "normally_seedless": True,
        "cross_pollination_explanation": "Clemenules are parthenocarpic; seeds form when bees move pollen from a compatible nearby mandarin.",
        "source_url": "https://www.mdpi.com/2073-4395/11/10/2023",
        "evidence_quote": "Clemenules is normally seedless under isolation; seed numbers rise sharply when planted near compatible varieties.",
    },
)


def _quote_substantive(summary, ctx):
    q = (summary.get("evidence_quote") or "").strip()
    nwords = len(q.split())
    if 4 <= nwords <= 60:
        return True, f"quote is {nwords} words"
    return False, f"quote is {nwords} words (expected 4-60)"


HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("normally_seedless", True, name="normally_seedless_correct"),
    FieldPresent("source_url", name="has_source_url"),
    AllURLsMatch("source_url" if False else "alternative_sources" if False else "",
                  "url", r"", name="placeholder"),  # placeholder; replaced below
]
# Replace the placeholder — we just want the source_url field to match an http URL
from agentic_judge.grounded.framework.constraints import HardConstraint as _HC

class _SourceURLValid(_HC):
    name = "source_url_is_http"
    def check(self, summary, ctx=None):
        from agentic_judge.grounded.framework.constraints import HardConstraintResult
        v = (summary.get("source_url") or "")
        ok = isinstance(v, str) and v.startswith(("http://", "https://"))
        return HardConstraintResult(self.name, ok,
            f"source_url={v!r}" if ok else f"source_url not http(s): {v!r}")

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("normally_seedless", True, name="normally_seedless_correct"),
    FieldPresent("source_url", name="has_source_url"),
    _SourceURLValid(),
    Custom("evidence_quote_substantive", _quote_substantive),
]


def FAITHFULNESS_CHECKS(summary):
    out = []
    url = (summary.get("source_url") or "").strip()
    if url:
        q = (summary.get("evidence_quote") or "").split()
        snippet = " ".join(q[:6]) if q else "Clemenules"
        out.append({"url": url, "claim": snippet})
    return out


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
