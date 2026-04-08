"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Book an appointment for a health check-up at a clinic in Los Angeles using specified websites and report confirmation details.
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


TASK_INSTRUCTION = """Book an appointment for a health check-up at a clinic in Los Angeles. Use the clinic’s website to enter a fictional patient profile (name: John Doe, DOB: January 1, 1990), select a date within the next two weeks, and report the confirmation details shown on the final page."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves booking a health check-up appointment at a clinic in Los Angeles using specified websites. The agent must create a fictional patient profile, select a date within the next two weeks, and provide confirmation details from the final page. A successful completion requires navigating the specified platforms and meeting all task constraints.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Book an appointment for a health check-up at a clinic in Los Angeles. Use the clinic’s website to enter a fictional patient profile (name: John Doe, DOB: January 1, 1990), select a date within the next two weeks, and report the confirmation details shown on the final page.

## Task-Specific Constraints
- Must navigate to at least one of the specified platforms: zocdoc.com, healthgrades.com, mayoclinic.org.
- Must enter the fictional patient profile (name: John Doe, DOB: January 1, 1990) correctly.
- Must select a date within the next two weeks from the current date.
- Must provide confirmation details from the final booking page.
- Response must clearly state the clinic name, appointment date, and time.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least one of the required platforms? Which ones were visited?
- Was the fictional patient profile entered correctly (name and DOB)?
- Was the appointment date selected within the next two weeks?
- Does the response include confirmation details (clinic name, appointment date, and time)?
- Is the output structured clearly and easy to understand?

### Step 2: Dimension Scoring

#### A. Appointment Booking Accuracy (0.35)
Measures whether the agent successfully booked an appointment and provided correct confirmation details.

5 — Includes clinic name, appointment date, and time, all correct and complete.
4 — Includes most required details but minor errors or omissions.
3 — Includes some required details but lacks completeness.
2 — Attempted but mostly incorrect or incomplete.
1 — No booking details provided.

#### B. Platform Navigation Coverage (0.30)
Measures whether the agent navigated to the required platforms and used them appropriately.

5 — Successfully navigated and used at least one of the specified platforms.
4 — Navigated to a platform but usage was incomplete or unclear.
3 — Attempted navigation but did not complete the task properly.
2 — Minimal evidence of navigation.
1 — No navigation attempted.

#### C. Patient Profile Entry Accuracy (0.20)
Measures whether the fictional patient profile was entered correctly.

5 — Name and DOB entered correctly and confirmed in the response.
4 — Name and DOB mostly correct but minor errors.
3 — Partial entry of patient profile details.
2 — Attempted but incorrect or incomplete.
1 — No patient profile entered.

#### D. Output Structure and Clarity (0.15)
Measures whether the agent's response is well-structured and easy to understand.

5 — Response is clear, organized, and easy to follow.
4 — Response is mostly clear but minor formatting issues.
3 — Response is partially clear but lacks organization.
2 — Response is unclear or poorly formatted.
1 — Response is disorganized and incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "appointment_booking_accuracy": <1-5>,
  "platform_navigation_coverage": <1-5>,
  "patient_profile_entry_accuracy": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "appointment_booking_accuracy": "<one sentence citing specific evidence>",
    "platform_navigation_coverage": "<one sentence citing specific evidence>",
    "patient_profile_entry_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "appointment_booking_accuracy": 0.35,
    "platform_navigation_coverage": 0.30,
    "patient_profile_entry_accuracy": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())