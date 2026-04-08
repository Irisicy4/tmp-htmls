"""
LLM-as-judge evaluator for EvolveBench task-63.

Category: Marketing & Analytics
Task: Please search the internet for the company at https://www.peloton.com and find all their weak points.
"""

import os, json, re
PASS_THRESHOLD = 3.0

def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip(): return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20: return content
    return ""

def _parse(text):
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if not match: return None
    try: return json.loads(match.group(1).strip())
    except json.JSONDecodeError: return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_INSTRUCTION,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= PASS_THRESHOLD
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)])
        else:
            scores = first; scores["_votes_used"] = 1
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0)
    passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"]
    for dim in DIMENSIONS:
        if dim in scores: lines.append(f"  {dim}: {scores[dim]}/5")
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    reasoning = scores.get("dimension_reasoning", {})
    if reasoning:
        lines.append("\nDimension reasoning:")
        for dim, reason in reasoning.items(): lines.append(f"  {dim}: {reason}")
    if scores.get("_votes_used", 1) > 1:
        lines.append(f"\n(Borderline case: {scores['_votes_used']} judge calls used, majority vote applied)")
    return {
        "passed": bool(passed), "feedback": "\n".join(lines),
        "details": {"task_completed": result.get("status") == "success", "overall_score": overall,
                    "dimension_scores": {d: scores.get(d) for d in DIMENSIONS},
                    "evidence_summary": scores.get("evidence_summary"),
                    "dimension_reasoning": scores.get("dimension_reasoning"),
                    "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)},
    }


TASK_INSTRUCTION = """Please search the internet for the company at https://www.peloton.com and find all their weak points."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves analyzing a company's website and online presence to identify their weaknesses for competitive intelligence purposes. The target company is Peloton (peloton.com), a fitness equipment and subscription service company.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Target: https://www.peloton.com — must visit this site
- Scope: web presence, reviews, SEO, product/service gaps, customer complaints
- Depth: multiple angles (website quality, social proof, pricing, complaints)
- Output: actionable list of competitor weaknesses

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent visit peloton.com?
- What aspects of the site were analyzed (design, content, SEO, offerings)?
- Were external sources checked (Google reviews, social media, complaints)?
- Are specific weaknesses identified with evidence?

### Step 2: Dimension Scoring

#### A. Site Analysis (0.3)
Did the agent analyze the company's website?

5 — Agent visited the site and analyzed: design quality, content, product/service offering, UX, and value proposition.
4 — Visited site but analysis is shallow.
3 — Visited site but only described it without identifying weaknesses.
2 — Described what to look for without visiting.
1 — No site visit.

#### B. External Research (0.25)
Did the agent research external sources?

5 — Checked Google reviews, social media, and/or complaint platforms for customer sentiment.
4 — Checked one external source.
3 — Brief external search without substantive findings.
2 — Only searched the company's own site.
1 — No external research.

#### C. Weakness Identification (0.35)
Are specific, actionable weaknesses identified?

5 — 5+ specific weaknesses with evidence (e.g. high price point, subscription churn, hardware reliability issues, intense competition, thin profit margins).
4 — 3-4 specific weaknesses.
3 — 1-2 specific weaknesses.
2 — Vague weaknesses without evidence.
1 — No weaknesses identified.

#### D. Competitive Framing (0.1)
Are weaknesses framed as opportunities?

5 — Each weakness explicitly linked to how a competitor can exploit it.
4 — Some competitive framing.
3 — Weaknesses listed without competitive context.
2 — Analysis from wrong perspective.
1 — No competitive framing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "site_analysis": <1-5>,
  "external_research": <1-5>,
  "weakness_identification": <1-5>,
  "competitive_framing": <1-5>,
  "dimension_reasoning": {{
    "site_analysis": "<one sentence citing specific evidence>",
    "external_research": "<one sentence citing specific evidence>",
    "weakness_identification": "<one sentence citing specific evidence>",
    "competitive_framing": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "site_analysis": 0.3,
    "external_research": 0.25,
    "weakness_identification": 0.35,
    "competitive_framing": 0.1,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())