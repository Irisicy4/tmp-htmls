"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Determine the best engagement rate among three verified fitness influencers on Instagram and recommend one for a sponsorship deal.
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


TASK_INSTRUCTION = """Fetch engagement metrics for three verified influencers on Instagram who focus on fitness and wellness. Calculate which influencer has the best engagement rate (likes and comments compared to follower count) and recommend them for a hypothetical sponsorship deal."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch engagement metrics (likes, comments, follower count) for three verified fitness and wellness influencers on Instagram. The agent must calculate the engagement rate for each influencer and recommend the one with the highest rate for a sponsorship deal. A successful completion includes accurate metrics, correct calculations, and a clear recommendation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch engagement metrics for three verified influencers on Instagram who focus on fitness and wellness. Calculate which influencer has the best engagement rate (likes and comments compared to follower count) and recommend them for a hypothetical sponsorship deal.

## Task-Specific Constraints
- Must fetch data for exactly three verified influencers in the fitness and wellness domain.
- Must include likes, comments, and follower count for each influencer.
- Must calculate engagement rate as (likes + comments) / follower count.
- Must recommend the influencer with the highest engagement rate.
- Output must be structured as a table or clear list with metrics and calculations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram and fetch metrics for three verified influencers?
- Are likes, comments, and follower count present for all three influencers?
- Is the engagement rate calculated correctly for each influencer?
- Is the recommendation based on the highest engagement rate?
- Is the output structured as a table or clear list?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the engagement metrics and recommendation are correct and complete.

5 — Metrics for all three influencers are accurate, engagement rates are calculated correctly, and the recommendation is correct.
4 — Metrics are mostly accurate, engagement rates are calculated correctly, but minor errors exist in the recommendation.
3 — Metrics are partially accurate, engagement rates are calculated with minor errors, and the recommendation is usable but flawed.
2 — Metrics are mostly missing or incorrect, engagement rates are calculated incorrectly, and the recommendation is poor.
1 — Metrics are absent or completely wrong, and no recommendation is provided.

#### B. Coverage of Required Items (0.30)
Measures whether all required metrics and influencers are included.

5 — Metrics for all three verified influencers are present and complete.
4 — Metrics for three influencers are present but missing minor details.
3 — Metrics for fewer than three influencers are present or incomplete.
2 — Metrics for only one influencer are present or mostly incomplete.
1 — No metrics are provided.

#### C. Depth and Specificity (0.20)
Measures the detail and clarity of the calculations and recommendation.

5 — Engagement rates are calculated with clear steps, and the recommendation is well-justified.
4 — Engagement rates are calculated correctly but lack detailed explanation.
3 — Engagement rates are calculated with minor errors or unclear justification for the recommendation.
2 — Engagement rates are calculated incorrectly or lack clarity.
1 — No calculations or justification are provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the response.

5 — Output is well-structured as a table or clear list, and metrics are sourced credibly.
4 — Output is mostly well-structured, with minor formatting issues or unclear sourcing.
3 — Output is usable but poorly structured or lacks credible sourcing.
2 — Output is disorganized or mostly unclear.
1 — Output is absent or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_items": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_items": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_items": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())