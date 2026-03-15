"""
LLM-as-judge evaluator for EvolveBench task-33.
Category: Data & ML Engineering
"""
import os, json, re

TASK_INSTRUCTION = 'Research the institution: Beijing Lehe Xinyi Chronic Disease Research Center. Include but not limited to: authenticity of the institution, quality of medical staff, patient feedback, and what products and services they offer with corresponding prices.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a due-diligence research task on a specific Chinese medical institution. Authenticity verification is particularly important — the agent should assess whether the institution is legitimate.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Institution: Beijing Lehe Xinyi Chronic Disease Research Center (北京乐和心益慢性病研究中心 or similar)\n- Required aspects: (1) authenticity/legitimacy, (2) medical staff quality, (3) patient feedback/reviews, (4) products/services with prices\n- Sources: Chinese business registry, medical licensing databases, consumer review platforms (e.g. Dianping, Baidu Maps, 39.net)\n- This may be a questionable or low-credibility institution — the agent should flag concerns if found\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent search for this specific institution? What sources?\n- Was the institution\'s registration/license status checked?\n- Were patient reviews or complaints found?\n- Were specific products/services and prices listed?\n- Did the agent flag any red flags about legitimacy?\n\n### Step 2: Dimension Scoring\n\n#### A. Source Quality\nDid the agent use credible, appropriate sources?\n\n5 — Used 2+ of: Chinese business registry (企查查/天眼查), medical licensing database (卫生部), consumer review platform (大众点评), or official institution website.\n4 — Used 1 credible source; others are secondary.\n3 — Used general web search; some results from credible platforms but not specifically targeted.\n2 — Used low-quality or unverifiable sources only.\n1 — No sources; response from prior knowledge.\n\n#### B. Authenticity Assessment\nDid the agent investigate and report on the institution\'s legitimacy?\n\n5 — Agent verified (or attempted to verify) business registration, medical licensing, and noted any red flags (complaints, regulatory issues, unclear credentials).\n4 — Registration or licensing checked; patient complaint history not investigated.\n3 — Agent mentioned legitimacy concern but did not verify with primary sources.\n2 — Authenticity not investigated; agent assumed institution is legitimate.\n1 — No authenticity assessment.\n\n#### C. Information Coverage\nWere all 4 required aspects addressed?\n\n5 — All 4 aspects: authenticity, staff quality, patient feedback, products/prices — each with specific findings.\n4 — 3 of 4 aspects covered with specific findings; one is briefly mentioned.\n3 — 2–3 aspects covered; products/prices or patient feedback missing.\n2 — Only 1–2 aspects covered.\n1 — No substantive coverage of required aspects.\n\n#### D. Output Organisation\nIs the research presented in a clear, structured format?\n\n5 — Sections for each required aspect; findings clearly attributed to sources; any red flags highlighted.\n4 — Structured output; source attribution thin for some sections.\n3 — Content present but mixed together without clear section structure.\n2 — Narrative paragraph with relevant facts buried.\n1 — No structured output.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "source_quality": <1-5>,\n  "authenticity_assessment": <1-5>,\n  "information_coverage": <1-5>,\n  "output_organisation": <1-5>,\n  "dimension_reasoning": {{"source_quality": "<one sentence>", "authenticity_assessment": "<one sentence>", "information_coverage": "<one sentence>", "output_organisation": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'source_quality': 0.2, 'authenticity_assessment': 0.3, 'information_coverage': 0.3, 'output_organisation': 0.2}
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