"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Evaluate whether the agent successfully identified and recommended the best cell phone plan for a user needing 20GB of data per month, unlimited text, and minimal call minutes by using Verizon, T-Mobile, and AT&T websites.
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


TASK_INSTRUCTION = """Calculate and recommend the best cell phone plan for a user who needs 20GB of data per month, unlimited text, and minimal call minutes. Use Verizon, T-Mobile, and AT&T sites to gather data on their current plans and pricing."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to calculate and recommend the best cell phone plan for a user with specific needs (20GB of data, unlimited text, minimal call minutes). The agent must gather data from Verizon, T-Mobile, and AT&T websites and provide a structured comparison to identify the best plan.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate and recommend the best cell phone plan for a user who needs 20GB of data per month, unlimited text, and minimal call minutes. Use Verizon, T-Mobile, and AT&T sites to gather data on their current plans and pricing.

## Task-Specific Constraints
- Must visit Verizon, T-Mobile, and AT&T websites to gather data.
- Must include pricing and plan details for the 20GB data requirement.
- Must compare plans in a structured format (e.g., table or list).
- Must recommend the best plan based on price and user requirements.
- Must provide evidence or reasoning for the recommendation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Verizon, T-Mobile, and AT&T websites? Which ones were actually visited?
- Does the response include pricing and plan details for the 20GB data requirement?
- Is the output organized in a structured format (e.g., table or list)?
- Does the response recommend a single best plan with reasoning?
- Are the pricing and plan details accurate and sourced from the specified websites?

### Step 2: Dimension Scoring

#### A. Recommendation Accuracy (0.35)
Measures whether the agent correctly identifies and recommends the best plan based on the user's requirements.

5 — Recommends the best plan with accurate reasoning and all required details.
4 — Recommends a good plan but with minor inaccuracies or missing details.
3 — Recommends a plan but with significant omissions or errors.
2 — Recommends an incorrect or irrelevant plan.
1 — Fails to recommend a plan.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and included data from them.

5 — Includes data from all three platforms (Verizon, T-Mobile, AT&T).
4 — Includes data from two platforms.
3 — Includes data from one platform.
2 — Mentions platforms but provides no data.
1 — Does not mention or use any platforms.

#### C. Data Specificity (0.20)
Measures the level of detail in the pricing and plan comparison.

5 — Provides detailed pricing, plan features, and comparisons for all platforms.
4 — Provides detailed data for most platforms but misses minor details.
3 — Provides basic data but lacks detail or comparisons.
2 — Provides vague or incomplete data.
1 — Provides no data.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Output is well-structured (e.g., table or list) and easy to follow.
4 — Output is mostly structured but could be clearer.
3 — Output is minimally structured but understandable.
2 — Output is disorganized or hard to follow.
1 — Output is incoherent or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "recommendation_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "data_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "recommendation_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "recommendation_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "data_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())