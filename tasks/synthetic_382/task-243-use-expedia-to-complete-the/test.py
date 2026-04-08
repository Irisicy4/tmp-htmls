"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Complete a flight booking workflow for a one-way trip from New York City (JFK) to London Heathrow (LHR) on December 15th, selecting economy class with the cheapest available ticket and reporting flight details.
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


TASK_INSTRUCTION = """Use Expedia to complete the flight booking workflow for a one-way trip from New York City (JFK) to London Heathrow (LHR) on December 15th. Select economy class with the cheapest available ticket and report the flight details including airline, departure time, and price."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task involves using Expedia to book a one-way flight from New York City (JFK) to London Heathrow (LHR) on December 15th. The agent must select economy class and identify the cheapest available ticket. A successful completion requires reporting the airline, departure time, and price of the selected flight.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Expedia to complete the flight booking workflow for a one-way trip from New York City (JFK) to London Heathrow (LHR) on December 15th. Select economy class with the cheapest available ticket and report the flight details including airline, departure time, and price.

## Task-Specific Constraints
- Must navigate to expedia.com and perform the flight search workflow.
- Must select economy class for the flight.
- Must identify the cheapest available ticket based on price.
- Must report the airline name, departure time, and ticket price accurately.
- Output must be structured as a clear list or table.
- Must avoid reporting incomplete or speculative data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to expedia.com and perform the flight search workflow?
- Did the agent select economy class and identify the cheapest available ticket?
- Are the airline name, departure time, and ticket price present in the response?
- Is the output structured as a clear list or table?
- Are the reported details accurate and consistent with the task requirements?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and reported the cheapest economy-class ticket details.

5 — All required details (airline, departure time, price) are correct and complete.
4 — Most details are correct, but one minor error or omission exists.
3 — Partial completion: at least one correct detail, but others are missing or incorrect.
2 — Poor: incorrect details or missing most required information.
1 — No attempt or completely incorrect.

#### B. Coverage of Workflow Steps (0.30)
Measures whether the agent followed all required steps in the flight booking workflow.

5 — All required steps (platform navigation, economy class selection, cheapest ticket identification) are completed.
4 — Most steps are completed, but one minor step is missed.
3 — Partial completion: at least one major step is completed, but others are missing.
2 — Poor: minimal steps completed or significant omissions.
1 — No attempt or completely missed.

#### C. Specificity of Reported Details (0.20)
Measures the depth and specificity of the reported flight details.

5 — Includes precise airline name, exact departure time, and ticket price.
4 — Includes most details, but one is slightly vague or incomplete.
3 — Includes at least one specific detail, but others are vague or missing.
2 — Poor: details are mostly vague or incomplete.
1 — No attempt or completely absent.

#### D. Output Structure and Clarity (0.15)
Measures the organization and clarity of the agent's final response.

5 — Output is well-structured as a clear list or table, easy to read and understand.
4 — Output is mostly clear, but minor formatting issues exist.
3 — Output is partially clear, but some formatting or organization issues exist.
2 — Poor: output is disorganized or difficult to interpret.
1 — No attempt or completely incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_workflow_steps": <1-5>,
  "specificity_of_reported_details": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_workflow_steps": "<one sentence citing specific evidence>",
    "specificity_of_reported_details": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_workflow_steps": 0.30,
    "specificity_of_reported_details": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())