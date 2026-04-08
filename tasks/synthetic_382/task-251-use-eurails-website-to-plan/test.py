"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Plan a train journey through Germany using Eurail's website, select the best Eurail pass, and report the cost and journey details.
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


TASK_INSTRUCTION = """Use Eurail's website to plan a train journey through Germany, starting in Berlin on April 1st and ending in Munich on April 5th with one stop in Frankfurt. Select the best Eurail pass for this itinerary and report the full cost and journey details as displayed on the final confirmation page."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to plan a train journey through Germany using Eurail's website, select the best Eurail pass for the itinerary, and report the full cost and journey details. This task falls under the Travel & Planning domain, and successful completion requires accurate journey planning, correct pass selection, and complete reporting of cost and details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Eurail's website to plan a train journey through Germany, starting in Berlin on April 1st and ending in Munich on April 5th with one stop in Frankfurt. Select the best Eurail pass for this itinerary and report the full cost and journey details as displayed on the final confirmation page.

## Task-Specific Constraints
- Must use Eurail's website to plan the journey.
- Must include stops in Berlin, Frankfurt, and Munich.
- Must select the best Eurail pass based on the itinerary.
- Must report the total cost of the pass and journey details.
- Output must include departure and arrival times for each leg of the journey.
- Output must be structured as a clear table or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Eurail's website and use it to plan the journey?
- Are Berlin, Frankfurt, and Munich included as stops in the itinerary?
- Is the selected Eurail pass appropriate for the itinerary?
- Is the total cost of the pass reported accurately?
- Is the output structured as a clear table or list with departure and arrival times?

### Step 2: Dimension Scoring

#### A. Journey Planning Accuracy (0.35)
Measures whether the agent correctly planned the journey through Germany as instructed.

5 — Includes Berlin, Frankfurt, and Munich with accurate departure and arrival times.
4 — Includes all stops but with minor inaccuracies in timing details.
3 — Includes all stops but lacks timing details or has significant inaccuracies.
2 — Missing one stop or major inaccuracies in timing.
1 — Missing multiple stops or completely incorrect.

#### B. Eurail Pass Selection (0.30)
Measures whether the agent selected the best Eurail pass for the itinerary.

5 — Pass selection is optimal and cost is reported accurately.
4 — Pass selection is good but cost reporting has minor inaccuracies.
3 — Pass selection is acceptable but lacks justification or has significant inaccuracies.
2 — Pass selection is poor or inappropriate for the itinerary.
1 — No pass selected or completely incorrect.

#### C. Cost Reporting Specificity (0.20)
Measures whether the agent reported the total cost and journey details accurately.

5 — Total cost and journey details are fully accurate and specific.
4 — Total cost and journey details are mostly accurate with minor omissions.
3 — Total cost is reported but lacks specificity or has significant omissions.
2 — Cost reporting is mostly missing or inaccurate.
1 — No cost reported or completely incorrect.

#### D. Output Structure Quality (0.15)
Measures whether the agent's response is well-organized and easy to understand.

5 — Output is structured as a clear table or list with all required details.
4 — Output is mostly clear but has minor formatting issues.
3 — Output is usable but lacks clarity or organization.
2 — Output is poorly structured or difficult to understand.
1 — Output is completely unstructured or unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated Eurail's website and included Berlin, Frankfurt, and Munich in the itinerary. The selected Eurail pass appears appropriate, and the total cost and journey details are reported with minor inaccuracies. The output is structured as a clear table.",
  "journey_planning_accuracy": 4,
  "eurail_pass_selection": 4,
  "cost_reporting_specificity": 4,
  "output_structure_quality": 5,
  "dimension_reasoning": {
    "journey_planning_accuracy": "All stops are included, but timing details have minor inaccuracies.",
    "eurail_pass_selection": "The pass selection is appropriate, but cost reporting has minor inaccuracies.",
    "cost_reporting_specificity": "Cost and journey details are mostly accurate but lack full specificity.",
    "output_structure_quality": "The output is structured as a clear table with all required details."
  },
  "overall_score": 4.05,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "journey_planning_accuracy": 0.35,
    "eurail_pass_selection": 0.30,
    "cost_reporting_specificity": 0.20,
    "output_structure_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())