"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Filter men's running shoes priced under $120 on nike.com and extract the top 5 options based on customer ratings and availability in size 10.
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


TASK_INSTRUCTION = """Go to Nike's official website, filter for men's running shoes priced under $120, and extract the top 5 options based on customer ratings and availability. Verify that all options are available in size 10."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate Nike's official website, filter for men's running shoes priced under $120, and extract the top 5 options based on customer ratings and availability. The agent must verify that all options are available in size 10. A successful completion includes a structured output listing shoe names, prices, ratings, and availability in size 10.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Nike's official website, filter for men's running shoes priced under $120, and extract the top 5 options based on customer ratings and availability. Verify that all options are available in size 10.

## Task-Specific Constraints
- Must navigate to nike.com and apply the correct filters for men's running shoes under $120.
- Must extract customer ratings for each shoe.
- Must verify availability of all shoes in size 10.
- Must provide a structured output listing shoe names, prices, ratings, and availability.
- Must include exactly 5 options in the final output.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to nike.com and apply the correct filters?
- Are customer ratings included for all shoes listed in the response?
- Is availability in size 10 verified for each shoe?
- Is the output structured correctly as a table or list with shoe names, prices, ratings, and availability?
- Are there exactly 5 options listed in the response?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and listed the top 5 men's running shoes under $120 with ratings and size 10 availability.

5 — All 5 shoes are correctly identified with accurate ratings and verified availability in size 10.
4 — 4 shoes are correctly identified with accurate ratings and verified availability in size 10.
3 — At least 3 shoes are correctly identified with partial accuracy in ratings or availability.
2 — Fewer than 3 shoes are correctly identified or significant inaccuracies in ratings/availability.
1 — No correct shoes identified or entirely incorrect response.

#### B. Coverage of Requirements (0.30)
Measures whether the agent fulfilled all task-specific constraints.

5 — All constraints are fully satisfied (filters applied, ratings included, size 10 verified, structured output).
4 — Most constraints are satisfied with minor omissions.
3 — Some constraints are satisfied, but key elements are missing.
2 — Few constraints are satisfied; major omissions.
1 — No constraints are satisfied.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the agent's response.

5 — Includes detailed ratings, prices, and availability for all shoes.
4 — Includes most details but with minor omissions.
3 — Includes partial details; some elements missing.
2 — Includes minimal details; significant omissions.
1 — No details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and based on credible evidence.

5 — Output is fully structured and credible; all data sourced correctly.
4 — Output is mostly structured and credible; minor formatting or sourcing issues.
3 — Output is partially structured; some credibility issues.
2 — Output is poorly structured or lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
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
    "coverage_of_requirements": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())