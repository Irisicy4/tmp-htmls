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

from .extractor import extract_summary_json, resolve_best_summary
from .faithfulness import (
    verify as verify_faithfulness,
    verify_url_claims_llm,
    aggregate as agg_faithfulness,
)


PASS_THRESHOLD = 3.0
MAX_PENDING_QUESTIONS = 3


def _extract_agent_text(result: dict) -> str:
    text = result.get("task_result") or ""
    if isinstance(text, str) and text.strip():
        return text
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and c.strip():
                return c
    return ""


def _auto_faith_checks(summary, agent_text: str, agent_stdout: str) -> list[dict]:
    """Default URL extractor when no per-task FAITHFULNESS_CHECKS is provided.

    Walk the summary JSON (preferred), the agent's final response, and the
    tail of the stdout trace, extract every http(s) URL, dedup, and pair each
    with a short claim derived from surrounding context. Lets the source-
    checking helper run on tasks that haven't been hand-wired.
    """
    import re as _re

    url_pat = _re.compile(r"https?://[^\s\"'<>()\\\]]+", _re.I)
    out: list[dict] = []
    seen: set[str] = set()

    def _push(url: str, claim: str):
        url = url.rstrip(".,);]\"'")
        if url in seen or len(out) >= 8:
            return
        seen.add(url)
        out.append({"url": url, "claim": (claim or "").strip()[:200]})

    # 1) Pull URLs out of the structured summary first; pair each with its
    # sibling keys so the claim has some context.
    def _walk(obj, ctx_path: tuple[str, ...] = ()):
        if isinstance(obj, dict):
            ctx_items = {k: v for k, v in obj.items()
                          if isinstance(v, (str, int, float, bool))}
            ctx_str = " · ".join(f"{k}={v}" for k, v in list(ctx_items.items())[:6])
            for k, v in obj.items():
                if isinstance(v, str):
                    for u in url_pat.findall(v):
                        _push(u, ctx_str or k)
                elif isinstance(v, (dict, list)):
                    _walk(v, ctx_path + (k,))
        elif isinstance(obj, list):
            for it in obj:
                _walk(it, ctx_path)
        elif isinstance(obj, str):
            for u in url_pat.findall(obj):
                _push(u, "/".join(ctx_path) or "agent summary URL")

    if isinstance(summary, dict):
        _walk(summary)

    # 2) Fall back to scraping URLs from prose if the summary didn't yield any.
    if not out:
        for source in (agent_text or "", (agent_stdout or "")[-12000:]):
            for u in url_pat.findall(source):
                # Take a window of surrounding text as the claim
                idx = source.find(u)
                window = source[max(0, idx - 120): idx + len(u) + 80]
                # Strip URL itself from the window for a cleaner claim
                window = window.replace(u, "<URL>")
                _push(u, window)
                if len(out) >= 8:
                    break
    return out


def _spec_task_instruction(spec) -> str:
    """Look up TASK_INSTRUCTION robustly — first on the grounded spec,
    then on its sibling test_task module (`_llm`), then placeholder."""
    direct = getattr(spec, "TASK_INSTRUCTION", "") or ""
    if direct.strip():
        return direct
    sibling = getattr(spec, "_llm", None)
    if sibling is not None:
        inherited = getattr(sibling, "TASK_INSTRUCTION", "") or ""
        if inherited.strip():
            return inherited
    return "(task instruction not exported — using rubric only)"


def grounded_judge_test(result: dict, spec) -> dict:
    """Run hard + faithfulness + effectiveness layers and aggregate."""
    agent_text = _extract_agent_text(result)
    agent_stdout = (result.get("agent_stdout") or "") if isinstance(result, dict) else ""
    if not agent_text.strip() and not agent_stdout.strip():
        return {
            "passed": False,
            "feedback": "No response from agent.",
            "details": {"phase": "extract", "error": "empty agent_text and stdout"},
        }
    task_instruction = _spec_task_instruction(spec)

    # --- 1. Extract the summary JSON ---
    # Prefer the richest schema-conforming candidate among all embedded JSON
    # objects (turn-1 draft, turn-2 answer, or a `=== FILE: summary.json ===`
    # the agent wrote to /output while inlining only a pointer). Falls back to
    # plain last-block extraction when no schema is available.
    required_paths = None
    schema = getattr(spec, "SUMMARY_SCHEMA", None)
    if schema is not None and hasattr(schema, "required_paths"):
        try:
            required_paths = schema.required_paths()
        except Exception:
            required_paths = None
    if required_paths:
        summary, src = resolve_best_summary(agent_text, required_paths)
    else:
        summary, src = extract_summary_json(agent_text)
    format_ok = summary is not None and src in ("delim", "json-fence", "richest")

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

    # --- 3. Faithfulness ---
    # Preferred: schema-guided forward re-extraction over FAITHFULNESS_URL_KEYS
    # (two-turn: judge independently extracts sibling schema fields from each
    # cited page, then cross-checks per field). Falls back to the legacy
    # FAITHFULNESS_CHECKS callable, then to auto URL scraping.
    faith_report = {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}
    try:
        if (hasattr(spec, "FAITHFULNESS_URL_KEYS") and summary is not None
                and getattr(spec, "SUMMARY_SCHEMA", None) is not None):
            from .faithfulness import verify_url_keys
            findings = verify_url_keys(summary, spec.FAITHFULNESS_URL_KEYS,
                                        spec.SUMMARY_SCHEMA)
            faith_report = agg_faithfulness(findings)
        else:
            checks: list[dict] = []
            if hasattr(spec, "FAITHFULNESS_CHECKS") and summary is not None:
                try:
                    checks = spec.FAITHFULNESS_CHECKS(summary) or []
                except Exception:
                    checks = []
            if not checks:
                checks = _auto_faith_checks(summary, agent_text, agent_stdout)
            if len(checks) > 8:
                checks = checks[:8]
            if checks:
                findings = verify_faithfulness(checks)
                faith_report = agg_faithfulness(findings)
    except Exception as e:
        faith_report["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    # --- 4. Effectiveness (main LLM judge) — first pass ---
    eff: dict = {}
    prior_eff: dict | None = None
    answered_questions: list[dict] = []
    try:
        from .effectiveness import call_llm_judge, revise_with_faith_answers
        eff = call_llm_judge(
            task_instruction=task_instruction,
            agent_response=agent_text,
            agent_stdout=agent_stdout,
            summary_json=summary,
            hard_constraint_report=hard_report,
            faithfulness_report=faith_report,
            dimensions=spec.DIMENSIONS,
            dimension_weights=spec.DIMENSION_WEIGHTS,
            task_specific_rubric=spec.TASK_RUBRIC,
        )
        # --- 4b. Pending-question loop: dispatch any pending_faith_questions
        # to the LLM-agentic Faithfulness sub-agent, then re-call Eff with
        # the answers so it can revise dim scores.
        pending = (eff or {}).get("pending_faith_questions") or []
        if isinstance(pending, list) and pending and "error" not in eff:
            pending = [p for p in pending if isinstance(p, dict)
                        and p.get("url") and p.get("question")]
            pending = pending[:MAX_PENDING_QUESTIONS]
        else:
            pending = []
        if pending:
            try:
                findings_q = verify_url_claims_llm(pending)
                answered_questions = [f.to_dict() for f in findings_q]
                prior_eff = dict(eff)
                eff_revised = revise_with_faith_answers(
                    task_instruction=task_instruction,
                    agent_response=agent_text,
                    agent_stdout=agent_stdout,
                    summary_json=summary,
                    hard_constraint_report=hard_report,
                    faithfulness_report=faith_report,
                    dimensions=spec.DIMENSIONS,
                    dimension_weights=spec.DIMENSION_WEIGHTS,
                    task_specific_rubric=spec.TASK_RUBRIC,
                    prior_judgment=prior_eff,
                    answered_faith_questions=answered_questions,
                )
                if "error" not in eff_revised:
                    eff = eff_revised
            except Exception as e:
                eff["pending_loop_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    except Exception as e:
        eff = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    # --- 5. Aggregate.  Final score = LLM overall, with hard-constraint and
    # faithfulness penalty: lose 0.5 for every failed hard constraint up to 2.0;
    # bonus 0.0-0.5 from faithfulness on top.  Missing the entire JSON block
    # is treated as a moderate fault (-1.0), because without structured output
    # the structural and faithfulness layers cannot run.  The mixer is
    # intentionally explicit so a curator can audit.
    llm_overall = float(eff.get("overall_score") or 0.0)
    hard_fails = sum(1 for r in hard_report if not r["passed"])
    no_json_block = summary is None
    if no_json_block:
        hard_penalty = 1.0
    else:
        hard_penalty = min(2.0, 0.5 * hard_fails)
    # Faithfulness is a bonus-only signal in this version: well-supported
    # claims add up to +0.5, unsupported ones do not subtract. The
    # downside for fabricated URLs comes from the hard-constraint layer
    # (URL-format and required-source predicates) and from the rubric
    # judge reading the faithfulness report. A real penalty for low
    # faith rate is intentionally left out — see PR discussion.
    faith_bonus = round(0.5 * faith_report["score_5"] / 5.0, 2)  # 0..0.5
    final = max(0.0, min(5.0, llm_overall - hard_penalty + faith_bonus))

    # The pass gate: a passing final score plus either a clean hard-
    # constraint sweep or a strong rubric overall to override a borderline
    # hard miss.
    passed = (final >= PASS_THRESHOLD) and (hard_fails == 0 or llm_overall >= 4.0)

    json_note = " (NO JSON BLOCK — -1.0)" if no_json_block else ""
    feedback_lines = [
        f"Final score: {final:.2f}/5 (threshold {PASS_THRESHOLD})",
        f"  LLM judge overall: {llm_overall:.2f}",
        f"  Hard-constraint pass-rate: {hard_pass_rate*100:.0f}% "
        f"({len(hard_report) - hard_fails}/{len(hard_report)})  → penalty -{hard_penalty:.2f}{json_note}",
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
    if answered_questions:
        feedback_lines.append(
            f"\nPending-question loop: Effectiveness asked Faithfulness "
            f"{len(answered_questions)} question(s).")
        for a in answered_questions:
            feedback_lines.append(
                f"  · {a.get('url','')[:60]}\n"
                f"    Q: {a.get('question','')[:160]}\n"
                f"    A: {a.get('answer','')[:160]}  (supported={a.get('claim_matched')})")
        if prior_eff:
            prior_overall = prior_eff.get("overall_score")
            new_overall = eff.get("overall_score")
            if prior_overall != new_overall:
                feedback_lines.append(
                    f"  LLM overall revised: {prior_overall} → {new_overall}")

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
            "effectiveness_judge_prior": prior_eff,
            "pending_faith_questions_answered": answered_questions,
            "pass_threshold": PASS_THRESHOLD,
            "score_components": {
                "llm_overall": llm_overall,
                "hard_penalty": hard_penalty,
                "faithfulness_bonus": faith_bonus,
            },
            "dimension_scores": {d: eff.get(d) for d in spec.DIMENSIONS},
        },
    }
