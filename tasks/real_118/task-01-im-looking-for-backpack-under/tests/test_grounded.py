"""Grounded (deterministic) checks for task-01-im-looking-for-backpack-under.

Sits alongside the existing LLM-as-judge `test_task.py` — same `tests/`
folder, but a separate file.  This module does NOT replace the LLM judge;
it is an independent grading layer that:

  1. Extracts the agent's structured summary JSON (between `=== JSON RESULT ===`
     delimiters in the agent stdout).
  2. Runs the per-task HARD_CONSTRAINTS predicates against that JSON.
  3. Runs the FAITHFULNESS_CHECKS (URL fetches + claim verification).
  4. Returns a structured report.  The LLM-judge layer (test_task.py)
     remains the official Harbor verdict; this report is for downstream
     analysis and the agentic_judge.grounded.run_grade aggregator.

Call via:

    from tests.test_grounded import grade
    grade(result)  # → {"hard_report": [...], "faithfulness_report": {...}}

Or from the cohort-level driver:

    python -m agentic_judge.grounded.run_grade --tag <run-tag>
"""
from importlib import import_module


_SPEC_MODULE = "task_01_backpack"


def _spec():
    return import_module(f"agentic_judge.grounded.task_specs.{_SPEC_MODULE}")


def grade(result: dict) -> dict:
    """Run hard-constraint + faithfulness checks; return a structured dict.
    Never raises — captures exceptions in the returned dict instead."""
    from agentic_judge.grounded.framework.extractor import extract_summary_json
    from agentic_judge.grounded.framework.faithfulness import (
        verify_url_claims, aggregate as agg_faithfulness,
    )

    spec = _spec()
    agent_text = result.get("task_result") or ""
    if not isinstance(agent_text, str) or not agent_text.strip():
        # Try conversation.assistant fallback
        for m in reversed(result.get("conversation") or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content") or ""
                if isinstance(c, str) and len(c) > 20:
                    agent_text = c
                    break

    summary, src = extract_summary_json(agent_text)
    if summary is None:
        return {
            "summary_parsed": False, "summary_source": src,
            "hard_report": [{"name": "summary_json_parseable",
                              "passed": False,
                              "detail": f"could not extract JSON (src={src})"}],
            "faithfulness_report": {},
        }
    hard_results = [c.check(summary) for c in spec.HARD_CONSTRAINTS]
    hard_report = [r.to_dict() for r in hard_results]
    faith = {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}
    try:
        if hasattr(spec, "FAITHFULNESS_CHECKS"):
            findings = verify_url_claims(spec.FAITHFULNESS_CHECKS(summary))
            faith = agg_faithfulness(findings)
    except Exception as e:
        faith["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return {
        "summary_parsed": True,
        "summary_source": src,
        "summary_json": summary,
        "hard_report": hard_report,
        "hard_pass_rate": sum(1 for r in hard_results if r.passed) / max(len(hard_results), 1),
        "faithfulness_report": faith,
    }


def main() -> int:
    import json, sys
    if len(sys.argv) != 2:
        print("usage: test_grounded.py <result.json>", file=sys.stderr)
        return 2
    result = json.load(open(sys.argv[1]))
    print(json.dumps(grade(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
