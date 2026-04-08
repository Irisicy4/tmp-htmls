"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Compare monetization policies of TikTok Creator Fund, YouTube Partner Program, and Instagram Reels Bonuses.
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


TASK_INSTRUCTION = """Find and compare the monetization policies of TikTok Creator Fund, YouTube Partner Program, and Instagram Reels Bonuses. Specifically, highlight differences in eligibility criteria, payout rates, and regional availability as stated on their official pages."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to compare monetization policies across three major social media platforms: TikTok, YouTube, and Instagram. A successful completion requires the agent to provide accurate details about eligibility criteria, payout rates, and regional availability for each platform, as stated on their official pages.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find and compare the monetization policies of TikTok Creator Fund, YouTube Partner Program, and Instagram Reels Bonuses. Specifically, highlight differences in eligibility criteria, payout rates, and regional availability as stated on their official pages.

## Task-Specific Constraints
- Must visit tiktok.com, youtube.com, and instagram.com.
- Must include eligibility criteria, payout rates, and regional availability for all three platforms.
- Output must be organized as a structured table or list.
- Must explicitly highlight differences between the platforms.
- Must source information directly from official pages of each platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tiktok.com, youtube.com, and instagram.com?
- Are eligibility criteria, payout rates, and regional availability present for all three platforms?
- Is the output organized as a structured table or list?
- Are differences between the platforms explicitly highlighted?
- Are the claims accurate and sourced from official pages?

### Step 2: Dimension Scoring

#### A. Accuracy of Monetization Details (0.35)
Measures whether the agent provided correct and complete information about eligibility criteria, payout rates, and regional availability.

5 — All details are correct and complete for all three platforms.
4 — Minor inaccuracies or omissions in one platform's details.
3 — Partial completion; significant omissions or inaccuracies in one or more platforms.
2 — Major inaccuracies or missing details for most platforms.
1 — No correct or relevant details provided.

#### B. Coverage of Platforms (0.30)
Measures whether the agent included information for all three specified platforms.

5 — Includes details for TikTok, YouTube, and Instagram.
4 — Includes details for two platforms; minor omission for the third.
3 — Includes details for one platform; major omissions for others.
2 — Minimal coverage; only mentions platforms without details.
1 — No coverage of the specified platforms.

#### C. Depth of Comparison (0.25)
Measures whether the agent provided detailed comparisons between the platforms.

5 — Provides detailed comparisons, including specific differences in eligibility, payout rates, and regional availability.
4 — Provides comparisons but lacks depth or specificity in one area.
3 — Provides basic comparisons; lacks depth in multiple areas.
2 — Minimal or vague comparisons; lacks meaningful detail.
1 — No comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sourced from credible official pages.

5 — Output is structured as a clear table or list and all claims are sourced from official pages.
4 — Output is structured but lacks clarity or has minor sourcing issues.
3 — Output is partially structured; some claims lack sourcing.
2 — Output is disorganized; most claims lack sourcing.
1 — Output is completely unstructured or unsourced.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "accuracy_of_monetization_details": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "accuracy_of_monetization_details": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "accuracy_of_monetization_details": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())