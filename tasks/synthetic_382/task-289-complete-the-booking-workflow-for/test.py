"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Book a round-trip train ticket from London to Edinburgh with specific requirements.
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


TASK_INSTRUCTION = """Complete the booking workflow for a round-trip train ticket from London to Edinburgh departing on the 10th of next month and returning on the 13th through Trainline. Select options that include standard class tickets. Report the final screen's output including price and trip details."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves booking a round-trip train ticket from London to Edinburgh using Trainline, departing on the 10th of next month and returning on the 13th. The agent must select standard class tickets and report the final screen's output, including price and trip details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete the booking workflow for a round-trip train ticket from London to Edinburgh departing on the 10th of next month and returning on the 13th through Trainline. Select options that include standard class tickets. Report the final screen's output including price and trip details.

## Task-Specific Constraints
- Must navigate to Trainline and complete the booking workflow.
- Must select standard class tickets for both legs of the journey.
- Must report the price of the round-trip ticket.
- Must include trip details such as departure and arrival times for both legs.
- Output must be structured as a clear table or list.
- Must ensure the dates match the instruction (10th and 13th of next month).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Trainline and complete the booking workflow?
- Are the departure and return dates correct (10th and 13th of next month)?
- Did the agent select standard class tickets for both legs of the journey?
- Is the price of the ticket included in the response?
- Is the output structured as a clear table or list with trip details?

### Step 2: Dimension Scoring

#### A. Booking Accuracy (0.35)
Measures whether the agent successfully completed the booking workflow and provided correct trip details.

5 — Booking completed with correct dates, standard class tickets, and trip details (departure/arrival times).
4 — Booking completed but minor errors in trip details or ticket class.
3 — Booking partially completed; missing some trip details or incorrect ticket class.
2 — Booking attempted but mostly incorrect or incomplete.
1 — Booking not attempted or completely wrong.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent navigated to Trainline and completed the workflow as instructed.

5 — Trainline fully used to complete the booking workflow.
4 — Trainline used but with minor deviations or incomplete workflow.
3 — Trainline partially used; workflow incomplete.
2 — Trainline visited but booking not attempted.
1 — Trainline not visited.

#### C. Detail Specificity (0.20)
Measures the inclusion of specific details such as price, departure/arrival times, and ticket class.

5 — All required details (price, times, ticket class) are included and correct.
4 — Most required details included; minor omissions or inaccuracies.
3 — Some required details included; significant omissions or inaccuracies.
2 — Few required details included; mostly missing or incorrect.
1 — No required details included.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required information.
4 — Output is mostly clear but minor formatting issues.
3 — Output is partially clear; some formatting issues.
2 — Output is unclear or poorly formatted.
1 — Output is completely unstructured or illegible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent navigated to Trainline and attempted the booking workflow. The response includes price and trip details, but the ticket class for both legs is unclear. The output is structured as a table.",
  "booking_accuracy": 4,
  "coverage_of_required_platforms": 5,
  "detail_specificity": 4,
  "output_structure_and_clarity": 4,
  "dimension_reasoning": {
    "booking_accuracy": "Dates and trip details are correct, but ticket class is unclear.",
    "coverage_of_required_platforms": "Trainline was fully used to complete the workflow.",
    "detail_specificity": "Price and trip details are included, but ticket class is ambiguous.",
    "output_structure_and_clarity": "Output is structured as a table with minor formatting issues."
  },
  "overall_score": 4.2,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "booking_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())