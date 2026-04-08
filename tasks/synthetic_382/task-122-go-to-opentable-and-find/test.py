"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Find three restaurants in Los Angeles with outdoor seating and a price range of $15–30 per meal, including names, ratings, and reservation times for tomorrow evening between 6 PM and 9 PM.
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


TASK_INSTRUCTION = """Go to OpenTable and find three restaurants in Los Angeles with outdoor seating and a price range of $15–30 per meal. Extract their names, ratings, and available reservation times for tomorrow evening between 6 PM and 9 PM."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use OpenTable to find three restaurants in Los Angeles that meet specific criteria: outdoor seating, a price range of $15–30 per meal, and available reservations for tomorrow evening between 6 PM and 9 PM. A successful completion includes extracting the names, ratings, and reservation times of these restaurants.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to OpenTable and find three restaurants in Los Angeles with outdoor seating and a price range of $15–30 per meal. Extract their names, ratings, and available reservation times for tomorrow evening between 6 PM and 9 PM.

## Task-Specific Constraints
- Must use OpenTable to find the required information.
- Must identify exactly three restaurants that meet all criteria.
- Must include names, ratings, and reservation times for each restaurant.
- Reservation times must fall between 6 PM and 9 PM tomorrow evening.
- Output must be structured as a table or clearly organized list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to OpenTable and use it to gather information?
- Are three restaurants identified, and do they meet all criteria (outdoor seating, price range, reservation times)?
- Are names, ratings, and reservation times present in the response?
- Is the output structured as a table or clearly organized list?
- Are the reservation times within the specified window (6 PM to 9 PM tomorrow evening)?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified three restaurants that meet all specified criteria.

5 — All three restaurants meet all criteria, and names, ratings, and reservation times are correct.
4 — Two restaurants meet all criteria, and the third partially meets them.
3 — At least one restaurant meets all criteria, but others are incomplete or incorrect.
2 — Restaurants identified, but none meet all criteria.
1 — No valid restaurants identified.

#### B. Coverage of Required Criteria (0.30)
Measures whether the agent addressed all specified constraints (outdoor seating, price range, reservation times).

5 — All criteria are addressed for all three restaurants.
4 — Most criteria are addressed, with minor omissions.
3 — Some criteria are addressed, but major omissions exist.
2 — Few criteria are addressed.
1 — No criteria are addressed.

#### C. Depth of Information (0.20)
Measures the level of detail provided for each restaurant, including ratings and reservation times.

5 — Detailed ratings and reservation times are provided for all three restaurants.
4 — Detailed information is provided for two restaurants; minor omissions for the third.
3 — Basic information is provided for at least one restaurant; others lack detail.
2 — Minimal information provided for all restaurants.
1 — No meaningful information provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and whether the information appears credible.

5 — Output is structured as a clear table or list, and all information is credible.
4 — Output is mostly well-organized, with minor formatting issues.
3 — Output is usable but disorganized or unclear.
2 — Output is poorly organized or lacks credibility.
1 — Output is absent or completely unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_criteria": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_criteria": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_criteria": 0.30,
    "depth_of_information": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())