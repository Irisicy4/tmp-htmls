"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Navigate the Eurail website and complete a ticket purchase workflow for a 5-day pass allowing travel between Paris, Zurich, and Milan, selecting the youth category, and report the final ticket price in EUR.
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


TASK_INSTRUCTION = """Go to the Eurail site and complete a ticket purchase workflow for a 5-day pass allowing travel between Paris, Zurich, and Milan. Select the youth category, and report the final ticket price in EUR shown on the confirmation screen (no actual purchase required)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating the Eurail website to complete a ticket purchase workflow for a 5-day pass allowing travel between Paris, Zurich, and Milan. The agent must select the youth category and report the final ticket price in EUR shown on the confirmation screen. A successful completion requires accurate navigation, correct ticket selection, and reporting the price in EUR.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the Eurail site and complete a ticket purchase workflow for a 5-day pass allowing travel between Paris, Zurich, and Milan. Select the youth category, and report the final ticket price in EUR shown on the confirmation screen (no actual purchase required).

## Task-Specific Constraints
- Must navigate to eurail.com and complete the ticket selection workflow.
- Must select the 5-day pass option and include travel between Paris, Zurich, and Milan.
- Must choose the youth category for the ticket.
- Must report the final ticket price in EUR as shown on the confirmation screen.
- Output must clearly state the ticket price in EUR.
- No actual purchase is required.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to eurail.com and complete the ticket selection workflow?
- Did the agent select the 5-day pass option and include travel between Paris, Zurich, and Milan?
- Did the agent choose the youth category for the ticket?
- Did the agent report the final ticket price in EUR as shown on the confirmation screen?
- Is the output structured clearly and does it explicitly state the ticket price?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly reported the final ticket price in EUR and completed the workflow.

5 — Reports the correct ticket price in EUR and completes the workflow accurately.
4 — Reports the ticket price but with minor inaccuracies or omissions in the workflow.
3 — Reports the ticket price but with significant inaccuracies or incomplete workflow.
2 — Attempts the workflow but fails to report the ticket price or selects incorrect options.
1 — Does not attempt the workflow or report the ticket price.

#### B. Coverage of Requirements (0.30)
Measures whether the agent fulfilled all task-specific constraints.

5 — Fully satisfies all constraints (platform visited, correct pass, youth category, price reported).
4 — Satisfies most constraints but misses minor details.
3 — Satisfies some constraints but misses key elements.
2 — Satisfies few constraints and misses major elements.
1 — Does not satisfy any constraints.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the agent's response.

5 — Provides detailed and specific information (e.g., exact ticket price, pass type, category).
4 — Provides good detail but lacks minor specifics.
3 — Provides basic detail but lacks depth or specificity.
2 — Provides minimal detail with significant omissions.
1 — Provides no meaningful detail.

#### D. Output Structure and Clarity (0.15)
Measures whether the agent's response is well-organized and easy to understand.

5 — Output is clear, well-structured, and explicitly states the ticket price.
4 — Output is mostly clear but could be better organized.
3 — Output is understandable but lacks clarity or structure.
2 — Output is poorly structured and hard to follow.
1 — Output is incoherent or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
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
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())