"""
LLM-as-judge evaluator for EvolveBench task-41.
Category: Marketing & Analytics
"""
import os, json, re

TASK_INSTRUCTION = 'Analyze the pros and cons of AI agent services such as Genspark AI, Kimi, Flowith, Manus AI, etc. Measure and list a score for each service.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a comparative analysis task covering AI agent platforms. The agent must both analyse and score each service — not just list features.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Services mentioned: Genspark AI, Kimi, Flowith, Manus AI — all 4 must be covered; "etc." suggests additional services are welcome\n- Required: pros AND cons for each (not just pros)\n- Required: a numeric score for each service with methodology explained\n- Research: agent should search for current capabilities/reviews rather than relying solely on prior knowledge\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Were all 4 named services covered (Genspark, Kimi, Flowith, Manus)?\n- Are pros AND cons provided for each?\n- Is a score given for each service? What methodology?\n- Did the agent research current information or rely on prior knowledge?\n\n### Step 2: Dimension Scoring\n\n#### A. Service Coverage\nWere all required services covered?\n\n5 — All 4 named services covered; 1+ additional services included as bonus.\n4 — All 4 named services covered; no additional services.\n3 — 3 of 4 services covered; one missing or only briefly mentioned.\n2 — 2 of 4 services covered with depth.\n1 — Fewer than 2 services covered.\n\n#### B. Analysis Depth\nIs the pros/cons analysis substantive and specific?\n\n5 — 2+ specific pros and 2+ specific cons per service with concrete examples (e.g. "Kimi\'s long context window handles 128k tokens" rather than "good at long documents").\n4 — 1–2 pros and 1–2 cons per service; mostly specific but some vague.\n3 — Pros and cons present but generic (e.g. "easy to use", "limited features") without specifics.\n2 — Only pros or only cons for most services.\n1 — No meaningful analysis; just feature lists or descriptions.\n\n#### C. Scoring Methodology\nIs the scoring transparent, consistent, and justified?\n\n5 — Clear scoring methodology (e.g. weighted criteria: capability, ease of use, pricing, reliability); scores explained per service; consistent scale used.\n4 — Scores provided with brief justification; methodology partially explained.\n3 — Scores given without methodology; scores appear arbitrary.\n2 — Relative ranking only (1st, 2nd, 3rd) without numeric scores.\n1 — No scores provided.\n\n#### D. Output Structure\nIs the analysis well-organised and easy to compare across services?\n\n5 — Structured format: per-service sections with pros/cons + score; summary comparison table or final ranking.\n4 — Per-service sections present; no summary comparison table.\n3 — Services discussed in narrative form; hard to compare directly.\n2 — Mixed narrative with service names scattered throughout.\n1 — Unstructured response.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "service_coverage": <1-5>,\n  "analysis_depth": <1-5>,\n  "scoring_methodology": <1-5>,\n  "output_structure": <1-5>,\n  "dimension_reasoning": {{"service_coverage": "<one sentence>", "analysis_depth": "<one sentence>", "scoring_methodology": "<one sentence>", "output_structure": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'service_coverage': 0.2, 'analysis_depth': 0.3, 'scoring_methodology': 0.3, 'output_structure': 0.2}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
    return ""

def _parse(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}