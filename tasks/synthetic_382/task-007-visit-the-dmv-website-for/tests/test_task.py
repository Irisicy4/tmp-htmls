"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Schedule a driver's license renewal appointment on the California DMV website and report the first three available dates and times.
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


TASK_INSTRUCTION = """Visit the DMV website for California, find the appointment scheduling tool, and go through the workflow to select an available date for a driver's license renewal appointment in ZIP code 90210. Report the first three available appointment dates and times."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating the California DMV website to schedule a driver's license renewal appointment in ZIP code 90210. The agent must report the first three available appointment dates and times. A successful completion requires accurate navigation of the DMV website and correct reporting of the required information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit the DMV website for California, find the appointment scheduling tool, and go through the workflow to select an available date for a driver's license renewal appointment in ZIP code 90210. Report the first three available appointment dates and times.

## Task-Specific Constraints
- Must navigate to the official California DMV website (dmv.ca.gov).
- Must locate and use the appointment scheduling tool.
- Must select ZIP code 90210 for the appointment search.
- Must report exactly three available appointment dates and times.
- Output must be structured as a list or table with clear formatting.
- Dates and times must match the actual availability on the DMV website.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the California DMV website (dmv.ca.gov)?
- Did the agent locate and use the appointment scheduling tool?
- Did the agent correctly select ZIP code 90210 for the search?
- Are exactly three appointment dates and times reported?
- Is the output structured as a clear list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and reported the first three available appointment dates and times.

5 — All three dates and times are correct and match the DMV website.
4 — Two dates and times are correct; one minor error.
3 — At least one date and time is correct; others missing or incorrect.
2 — Attempted but mostly incorrect or incomplete.
1 — No correct dates or times reported.

#### B. Platform Navigation Coverage (0.30)
Measures whether the agent successfully navigated the California DMV website and used the required tools.

5 — Successfully navigated to the DMV website and used the scheduling tool.
4 — Navigated to the DMV website but minor issues with tool usage.
3 — Partial navigation; tool usage incomplete or incorrect.
2 — Attempted navigation but failed to use the tool.
1 — Did not navigate to the DMV website.

#### C. Output Specificity and Detail (0.20)
Measures whether the output includes specific dates and times in a clear and detailed format.

5 — Dates and times are presented in a clear, structured list or table.
4 — Dates and times are mostly clear but minor formatting issues.
3 — Dates and times are present but poorly formatted or unclear.
2 — Dates and times are incomplete or difficult to interpret.
1 — No specific dates or times provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and credible.

5 — Output is well-organized, credible, and free of errors.
4 — Output is mostly organized and credible with minor issues.
3 — Output is somewhat organized but contains noticeable errors.
2 — Output is poorly organized or lacks credibility.
1 — Output is disorganized and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "platform_navigation_coverage": <1-5>,
  "output_specificity_and_detail": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_navigation_coverage": "<one sentence citing specific evidence>",
    "output_specificity_and_detail": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "platform_navigation_coverage": 0.30,
    "output_specificity_and_detail": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())