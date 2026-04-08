"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Configure a Trello board for a Mediterranean cruise, including lists and excursion details.
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


TASK_INSTRUCTION = """Configure a trip planning board for a week-long Mediterranean cruise in Trello. Create lists for 'Packing', 'Excursions', and 'Itinerary'. Add three checklist cards under 'Excursions' for options from Royal Caribbean’s excursions page, and include details and approximate costs."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves configuring a Trello board for a Mediterranean cruise. The agent must create three lists: 'Packing', 'Excursions', and 'Itinerary'. Under 'Excursions', the agent must add three checklist cards with options sourced from Royal Caribbean’s excursions page, including details and approximate costs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Configure a trip planning board for a week-long Mediterranean cruise in Trello. Create lists for 'Packing', 'Excursions', and 'Itinerary'. Add three checklist cards under 'Excursions' for options from Royal Caribbean’s excursions page, and include details and approximate costs.

## Task-Specific Constraints
- Must create three lists: 'Packing', 'Excursions', and 'Itinerary'.
- Must add three checklist cards under 'Excursions' sourced from Royal Caribbean’s excursions page.
- Each checklist card must include details and approximate costs.
- Must navigate to trello.com and royalcaribbean.com as part of the task.
- Output must clearly show the Trello board structure and checklist card contents.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to trello.com and royalcaribbean.com? Were both platforms used appropriately?
- Are three lists ('Packing', 'Excursions', 'Itinerary') present in the Trello board?
- Are three checklist cards under 'Excursions' present, with details and approximate costs?
- Is the Trello board structure clearly described in the output?
- Are the excursion details sourced accurately from royalcaribbean.com?

### Step 2: Dimension Scoring

#### A. Trello Board Structure Accuracy (0.35)
Measures whether the Trello board structure matches the task requirements.

5 — Includes all three lists ('Packing', 'Excursions', 'Itinerary') and checklist cards under 'Excursions' are complete.
4 — Includes all lists but checklist cards are incomplete or missing minor details.
3 — Includes most lists but checklist cards are incomplete or missing.
2 — Includes only one list or checklist cards are mostly missing.
1 — No lists or checklist cards created.

#### B. Platform Usage Coverage (0.30)
Measures whether the agent navigated to and used the required platforms.

5 — Successfully navigated and used trello.com and royalcaribbean.com as required.
4 — Used both platforms but with minor omissions or errors.
3 — Used one platform fully but the other partially or incorrectly.
2 — Used only one platform and missed key requirements.
1 — Did not use the required platforms.

#### C. Excursion Details Specificity (0.20)
Measures the accuracy and completeness of excursion details.

5 — Includes details and approximate costs for three excursions, all sourced from royalcaribbean.com.
4 — Includes details and costs for three excursions, but with minor inaccuracies.
3 — Includes details and costs for two excursions, or lacks specificity.
2 — Includes details for one excursion, or details are mostly incorrect.
1 — No excursion details provided.

#### D. Output Organization and Clarity (0.15)
Measures the clarity and organization of the final output.

5 — Output is well-organized, clearly structured, and easy to understand.
4 — Output is organized but contains minor formatting issues.
3 — Output is partially organized but lacks clarity in some areas.
2 — Output is poorly organized and difficult to follow.
1 — Output is unstructured and unclear.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "trello_board_structure_accuracy": <1-5>,
  "platform_usage_coverage": <1-5>,
  "excursion_details_specificity": <1-5>,
  "output_organization_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "trello_board_structure_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "excursion_details_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "trello_board_structure_accuracy": 0.35,
    "platform_usage_coverage": 0.30,
    "excursion_details_specificity": 0.20,
    "output_organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())