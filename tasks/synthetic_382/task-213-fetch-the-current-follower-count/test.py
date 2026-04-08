"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Fetch follower counts for three creators on Instagram, TikTok, and YouTube, calculate their engagement rates, and recommend the creator with the highest engagement rate.
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


TASK_INSTRUCTION = """Fetch the current follower count for three popular creators across Instagram, TikTok, and YouTube. Calculate their engagement rates using their last 5 posts/videos (likes+comments/views). Recommend the creator with the highest engagement rate and explain your calculation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch follower counts for three popular creators from Instagram, TikTok, and YouTube, calculate their engagement rates using the last 5 posts or videos (likes + comments divided by views), and recommend the creator with the highest engagement rate. A successful completion must include accurate data from all three platforms, correct calculations, and a clear recommendation with an explanation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch the current follower count for three popular creators across Instagram, TikTok, and YouTube. Calculate their engagement rates using their last 5 posts/videos (likes+comments/views). Recommend the creator with the highest engagement rate and explain your calculation.

## Task-Specific Constraints
- Must visit Instagram, TikTok, and YouTube to fetch data.
- Must retrieve follower counts for three creators on each platform.
- Must calculate engagement rates using the formula: (likes + comments) / views for the last 5 posts/videos.
- Must recommend the creator with the highest engagement rate.
- Must explain the recommendation with clear calculations and reasoning.
- Output must be structured and easy to follow.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram, TikTok, and YouTube? Which platforms were actually visited?
- Did the agent retrieve follower counts for three creators on each platform?
- Did the agent calculate engagement rates using the correct formula?
- Did the agent recommend the creator with the highest engagement rate?
- Is the explanation of the recommendation clear and supported by calculations?

### Step 2: Dimension Scoring

#### A. Data Retrieval Accuracy (0.35)
Measures whether the agent successfully retrieved the required data from all three platforms.

5 — Successfully retrieved follower counts and post data for three creators on all platforms.
4 — Retrieved data for at least two creators on all platforms, with minor gaps.
3 — Retrieved partial data (e.g., one creator missing or incomplete data).
2 — Retrieved very limited or incorrect data.
1 — Did not retrieve any meaningful data.

#### B. Engagement Rate Calculation (0.30)
Measures whether the agent correctly calculated engagement rates for all creators.

5 — Correctly calculated engagement rates for all creators using the specified formula.
4 — Calculated engagement rates with minor errors or omissions.
3 — Calculated engagement rates for some creators but with significant gaps.
2 — Attempted calculations but mostly incorrect or incomplete.
1 — Did not attempt engagement rate calculations.

#### C. Recommendation Quality (0.20)
Measures whether the agent made a clear and justified recommendation based on the engagement rates.

5 — Recommended the correct creator with the highest engagement rate and provided a clear explanation.
4 — Recommended a creator with minor errors or unclear reasoning.
3 — Made a recommendation but with significant errors or lack of clarity.
2 — Attempted a recommendation but mostly incorrect or unsupported.
1 — Did not make a recommendation.

#### D. Output Structure and Clarity (0.15)
Measures whether the agent's response is well-organized and easy to follow.

5 — Output is structured, clear, and easy to understand, with all calculations shown.
4 — Output is mostly clear but with minor formatting or clarity issues.
3 — Output is somewhat clear but lacks structure or is hard to follow.
2 — Output is poorly organized or unclear.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_retrieval_accuracy": <1-5>,
  "engagement_rate_calculation": <1-5>,
  "recommendation_quality": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "data_retrieval_accuracy": "<one sentence citing specific evidence>",
    "engagement_rate_calculation": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_retrieval_accuracy": 0.35,
    "engagement_rate_calculation": 0.30,
    "recommendation_quality": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())