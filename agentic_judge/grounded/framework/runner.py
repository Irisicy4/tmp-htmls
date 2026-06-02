"""Top-level orchestrator — `grounded_judge_test(result, spec)`.

Called by each task's test_task.py shim with the Harbor result dict and a
per-task spec module that exports:
  - SUMMARY_SCHEMA       (SummarySchema)
  - HARD_CONSTRAINTS     (list[HardConstraint])
  - FAITHFULNESS_CHECKS  (callable: summary_json -> list[{url, claim}])
  - DIMENSIONS           (list[str])
  - DIMENSION_WEIGHTS    (dict[str, float])
  - TASK_INSTRUCTION     (str)  — the human-readable instruction
  - TASK_RUBRIC          (str)  — anchor language for the LLM judge

The return shape matches what evolve_bench_harbor/tests/test.sh expects:
  {passed: bool, feedback: str, details: dict}.
"""
from __future__ import annotations

import json
from typing import Any

from .extractor import extract_summary_json
from .faithfulness import verify_url_claims, aggregate as agg_faithfulness


PASS_THRESHOLD = 3.0


def _extract_agent_text(result: dict) -> str:
    text = result.get("task_result") or ""
    if isinstance(text, str) and text.strip():
        return text
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20:
                return c
    return ""


def grounded_judge_test(result: dict, spec) -> dict:
    """Run hard + faithfulness + effectiveness layers and aggregate."""
    agent_text = _extract_agent_text(result)
    if not agent_text.strip():
        return {
            "passed": False,
            "feedback": "No response from agent.",
            "details": {"phase": "extract", "error": "empty agent_text"},
        }

    # --- 1. Extract the summary JSON ---
    summary, src = extract_summary_json(agent_text)
    format_ok = summary is not None and src in ("delim", "json-fence")

    # --- 2. Hard constraints (always run, even on missing JSON) ---
    hard_report = []
    if summary is None:
        hard_report.append({"name": "summary_json_parseable", "passed": False,
                             "detail": f"could not extract JSON (src={src})"})
        hard_pass_rate = 0.0
    else:
        results = [c.check(summary) for c in spec.HARD_CONSTRAINTS]
        hard_report = [r.to_dict() for r in results]
        hard_pass_rate = sum(1 for r in results if r.passed) / max(len(results), 1)

    # --- 3. Faithfulness (skip if no URL list available) ---
    faith_report = {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}
    try:
        if summary is not None and hasattr(spec, "FAITHFULNESS_CHECKS"):
            checks = spec.FAITHFULNESS_CHECKS(summary)
            findings = verify_url_claims(checks)
            faith_report = agg_faithfulness(findings)
    except Exception as e:
        faith_report["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # --- 4. Effectiveness (LLM judge over soft dimensions) ---
    eff: dict = {}
    try:
        from .effectiveness import call_llm_judge
        eff = call_llm_judge(
            task_instruction=spec.TASK_INSTRUCTION,
            agent_response=agent_text,
            summary_json=summary,
            hard_constraint_report=hard_report,
            faithfulness_report=faith_report,
            dimensions=spec.DIMENSIONS,
            dimension_weights=spec.DIMENSION_WEIGHTS,
            task_specific_rubric=spec.TASK_RUBRIC,
        )
    except Exception as e:
        eff = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # --- 5. Aggregate.  Final score = LLM overall, with hard-constraint and
    # faithfulness penalty: lose 0.5 for every failed hard constraint up to 2.0;
    # bonus 0.0-0.5 from faithfulness on top.  The mixer is intentionally
    # explicit so a curator can audit.
    llm_overall = float(eff.get("overall_score") or 0.0)
    hard_fails = sum(1 for r in hard_report if not r["passed"])
    hard_penalty = min(2.0, 0.5 * hard_fails)
    faith_bonus = round(0.5 * faith_report["score_5"] / 5.0, 2)  # 0..0.5
    final = max(0.0, min(5.0, llm_overall - hard_penalty + faith_bonus))

    # The gate: any failed hard constraint OR any failed faithfulness check
    # below threshold makes pass borderline only if LLM overall is strong.
    passed = (final >= PASS_THRESHOLD) and (hard_fails == 0 or llm_overall >= 4.0)

    feedback_lines = [
        f"Final score: {final:.2f}/5 (threshold {PASS_THRESHOLD})",
        f"  LLM judge overall: {llm_overall:.2f}",
        f"  Hard-constraint pass-rate: {hard_pass_rate*100:.0f}% "
        f"({len(hard_report) - hard_fails}/{len(hard_report)})  → penalty -{hard_penalty:.2f}",
        f"  Faithfulness: {faith_report['matched']}/{faith_report['total']} claims matched, "
        f"score {faith_report['score_5']:.1f}/5  → bonus +{faith_bonus:.2f}",
        f"  Summary JSON parseable: {format_ok}",
    ]
    failed_hc = [r for r in hard_report if not r["passed"]]
    if failed_hc:
        feedback_lines.append("\nFailed hard constraints:")
        for r in failed_hc[:10]:
            feedback_lines.append(f"  - {r['name']}: {r['detail']}")
    if eff.get("dimension_reasoning"):
        feedback_lines.append("\nDimension reasoning:")
        for d, why in eff["dimension_reasoning"].items():
            feedback_lines.append(f"  {d}: {why}")

    return {
        "passed": bool(passed),
        "feedback": "\n".join(feedback_lines),
        "details": {
            "overall_score": final,
            "llm_overall_score": llm_overall,
            "hard_constraint_pass_rate": hard_pass_rate,
            "hard_constraint_report": hard_report,
            "faithfulness_report": faith_report,
            "summary_json_parsed": format_ok,
            "summary_json_source": src,
            "summary_json": summary,
            "effectiveness_judge": eff,
            "pass_threshold": PASS_THRESHOLD,
            "score_components": {
                "llm_overall": llm_overall,
                "hard_penalty": hard_penalty,
                "faithfulness_bonus": faith_bonus,
            },
            "dimension_scores": {d: eff.get(d) for d in spec.DIMENSIONS},
        },
    }
