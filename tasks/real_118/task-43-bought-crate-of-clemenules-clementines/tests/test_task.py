"""
LLM-as-judge evaluator for EvolveBench task-43.
Category: Daily Activities
"""
import os, json, re

TASK_INSTRUCTION = "I bought a crate of Clemenules clementines and they all had seeds—shouldn't this variety be seedless? Can someone confirm whether Clemenules are normally seedless?"
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a factual horticultural Q&A task. There is a clear correct answer: Clemenules are normally seedless BUT can develop seeds through cross-pollination with compatible citrus nearby. The agent should confirm seedlessness as the norm and explain the exception.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- This has a verifiable answer: Clemenules (Clementine Nules) are normally seedless/parthenocarpic\n- Exception: seeds develop when Clemenules are grown near compatible citrus (bees cross-pollinate)\n- The agent should: (1) confirm seedlessness as normal, (2) explain why seeds may have occurred, (3) cite credible horticultural source\n- This does not require browser navigation — the agent can answer from knowledge, but should verify or cite a source\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent confirm Clemenules are normally seedless?\n- Did the agent explain the cross-pollination exception?\n- Was a credible source cited (agriculture extension, citrus grower association, horticultural database)?\n- Is the answer clear and directly responsive to the user\'s concern?\n\n### Step 2: Dimension Scoring\n\n#### A. Factual Accuracy\nIs the core factual answer correct?\n\n5 — Agent correctly states: (a) Clemenules are normally seedless, AND (b) seeds occur due to cross-pollination with nearby compatible citrus.\n4 — Agent correctly states seedlessness as normal; cross-pollination explanation is vague or partial.\n3 — Agent states Clemenules are normally seedless without explaining why the user\'s batch had seeds.\n2 — Agent is uncertain or gives a hedged answer without confirming seedlessness.\n1 — Agent gives wrong answer (e.g. states Clemenules do have seeds normally) or refuses to answer.\n\n#### B. Explanation Quality\nIs the explanation of the seeded exception clear and accurate?\n\n5 — Clear explanation: Clemenules are parthenocarpic; seeds develop when bees cross-pollinate with mandarin, tangerine, or other compatible citrus planted nearby; commercial orchards control for this.\n4 — Cross-pollination mentioned as cause; parthenocarpy or commercial control not explained.\n3 — Agent mentions that conditions can cause seeds but explanation is vague.\n2 — Agent says "sometimes they have seeds" without mechanistic explanation.\n1 — No explanation of the seeded exception.\n\n#### C. Source Credibility\nDid the agent cite or reference a credible horticultural source?\n\n5 — Specific source cited: citrus variety database, agricultural extension service (e.g. UC Riverside Citrus Variety Collection), grower association, or peer-reviewed horticulture reference.\n4 — Source type mentioned but not specifically identified.\n3 — Agent indicated it checked a source but no citation given.\n2 — No source; answer from prior knowledge only.\n1 — Fabricated or irrelevant source.\n\n#### D. Practical Guidance\nDid the agent address the user\'s underlying concern (were their clementines normal or defective)?\n\n5 — Agent directly addresses user\'s concern: confirms their experience is a known exception (not fraud or mislabelling); may suggest checking whether the orchard had other citrus nearby.\n4 — User\'s concern addressed; suggestion for next steps not provided.\n3 — Factual answer given but user\'s specific situation not directly addressed.\n2 — Generic information without addressing whether the user\'s purchase was normal.\n1 — No practical guidance.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "factual_accuracy": <1-5>,\n  "explanation_quality": <1-5>,\n  "source_credibility": <1-5>,\n  "practical_guidance": <1-5>,\n  "dimension_reasoning": {{"factual_accuracy": "<one sentence>", "explanation_quality": "<one sentence>", "source_credibility": "<one sentence>", "practical_guidance": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'factual_accuracy': 0.4, 'explanation_quality': 0.3, 'source_credibility': 0.15, 'practical_guidance': 0.15}
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
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except: return None

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