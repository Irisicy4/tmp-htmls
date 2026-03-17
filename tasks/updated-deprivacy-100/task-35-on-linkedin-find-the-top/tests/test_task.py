"""
LLM-as-judge evaluator for EvolveBench task-35.
Category: Marketing & Analytics
"""
import os, json, re

TASK_INSTRUCTION = 'On LinkedIn, find the top 30 US design agencies and browse the latest 10 posts each has published. Provide a report that summarizes their posted content and engagement on those posts.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a large-scale LinkedIn research task: 30 agencies × 10 posts = 300 posts. Full completion is exceptional. Apply graduated credit based on actual coverage achieved.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Platform: LinkedIn specifically\n- Scope: top 30 US design agencies (agent must identify and justify which 30)\n- Per-agency: latest 10 posts each\n- Report must include: (1) summary of posted content per agency, (2) engagement metrics (likes, comments, shares) per post or per agency\n- Scale: 300 posts total; partial completion should receive graduated credit\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent navigate LinkedIn? Cite trace evidence.\n- How many agencies were identified? Are they recognisable US design firms?\n- For how many agencies were posts actually browsed?\n- Are engagement metrics (likes/comments) reported for any posts?\n- Is a structured report produced?\n\n### Step 2: Dimension Scoring\n\n#### A. LinkedIn Execution\nDid the agent actively navigate LinkedIn as instructed?\n\n5 — Agent navigated LinkedIn, searched for design agencies, and browsed company pages/posts.\n4 — Agent accessed LinkedIn but was blocked by login walls or rate limits; navigated partially.\n3 — Agent described LinkedIn approach but only retrieved limited data (e.g. from LinkedIn search snippets).\n2 — Agent used non-LinkedIn sources (e.g. agency websites, general web) instead.\n1 — No LinkedIn navigation performed.\n\n#### B. Agency Coverage\nHow many of the target 30 US design agencies are covered?\n\n5 — 20+ recognisable US design agencies identified and included in report.\n4 — 10–19 agencies covered with specific post data.\n3 — 5–9 agencies covered.\n2 — 2–4 agencies covered.\n1 — Fewer than 2 agencies or agencies are not specifically US design firms.\n\n#### C. Post Content Summary\nAre the agencies\' post topics and themes summarised?\n\n5 — For 10+ agencies: content themes summarised per agency (e.g. "Agency X posts about brand identity projects and case studies").\n4 — Content summarised for 5–9 agencies with themes identified.\n3 — Content summarised for 2–4 agencies; others described generically.\n2 — Only 1 agency\'s content described; or summaries are too vague to be useful.\n1 — No content summary; just agency names listed.\n\n#### D. Engagement Reporting\nAre engagement metrics reported?\n\n5 — Likes/comments/shares reported for posts from 5+ agencies; average or total engagement noted.\n4 — Engagement reported for 2–4 agencies with specific numbers.\n3 — Engagement mentioned for 1–2 agencies, or described qualitatively ("high engagement") without numbers.\n2 — Engagement not reported but acknowledged as a gap.\n1 — Engagement completely absent from report.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "linkedin_execution": <1-5>,\n  "agency_coverage": <1-5>,\n  "post_content_summary": <1-5>,\n  "engagement_reporting": <1-5>,\n  "dimension_reasoning": {{"linkedin_execution": "<one sentence>", "agency_coverage": "<one sentence>", "post_content_summary": "<one sentence>", "engagement_reporting": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'linkedin_execution': 0.25, 'agency_coverage': 0.3, 'post_content_summary': 0.25, 'engagement_reporting': 0.2}
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