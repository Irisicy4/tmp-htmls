"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Book a flu vaccination appointment on Walgreens' website and report the confirmed time and location.
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


TASK_INSTRUCTION = """Book an appointment for a flu vaccination using Walgreens' online scheduling tool. Enter ZIP code 94103, select the first available appointment at a nearby store, and complete the workflow. Report back the confirmed appointment time and location shown at the final screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to book a flu vaccination appointment on Walgreens' website. The agent must enter ZIP code 94103, select the first available appointment at a nearby store, and complete the workflow. A successful completion requires reporting the confirmed appointment time and location as shown on the final screen.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Book an appointment for a flu vaccination using Walgreens' online scheduling tool. Enter ZIP code 94103, select the first available appointment at a nearby store, and complete the workflow. Report back the confirmed appointment time and location shown at the final screen.

## Task-Specific Constraints
- Must navigate to walgreens.com and use the online scheduling tool.
- Must enter ZIP code 94103 and search for nearby stores.
- Must select the first available appointment at any nearby store.
- Must complete the scheduling workflow to reach the confirmation screen.
- Must report the confirmed appointment time and location exactly as shown on the final screen.
- The response must be clear and structured.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to walgreens.com and use the scheduling tool?
- Did the agent enter ZIP code 94103 and search for nearby stores?
- Did the agent select the first available appointment at a nearby store?
- Did the agent complete the workflow and reach the confirmation screen?
- Does the response clearly and accurately report the confirmed appointment time and location?

### Step 2: Dimension Scoring

#### A. Appointment Booking Accuracy (0.35)
Measures whether the agent successfully booked the appointment and reported the correct details.

5 — The agent booked the appointment and reported the exact time and location as shown on the confirmation screen.
4 — The agent booked the appointment but reported slightly incomplete or imprecise details.
3 — The agent partially booked the appointment but did not fully complete the workflow or missed key details.
2 — The agent attempted to book but failed to complete the workflow or reported incorrect details.
1 — The agent did not attempt to book the appointment.

#### B. Workflow Completion (0.30)
Measures whether the agent followed the required steps to complete the workflow.

5 — The agent completed all required steps, including entering ZIP code 94103 and selecting the first available appointment.
4 — The agent completed most steps but missed minor details.
3 — The agent completed some steps but skipped or incorrectly performed key parts.
2 — The agent attempted the workflow but made significant errors or omissions.
1 — The agent did not attempt the workflow.

#### C. Detail Specificity (0.20)
Measures the level of detail and specificity in the agent's response.

5 — The response includes all relevant details (time, location) with high specificity and accuracy.
4 — The response includes most relevant details but is slightly vague or incomplete.
3 — The response includes some relevant details but lacks specificity.
2 — The response is vague and missing key details.
1 — The response is entirely unclear or irrelevant.

#### D. Response Clarity and Structure (0.15)
Measures the clarity and organization of the agent's response.

5 — The response is clear, well-structured, and easy to understand.
4 — The response is mostly clear but could be better organized.
3 — The response is somewhat clear but has noticeable issues with structure or clarity.
2 — The response is unclear or poorly organized.
1 — The response is completely disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "appointment_booking_accuracy": <1-5>,
  "workflow_completion": <1-5>,
  "detail_specificity": <1-5>,
  "response_clarity_and_structure": <1-5>,
  "dimension_reasoning": {{
    "appointment_booking_accuracy": "<one sentence citing specific evidence>",
    "workflow_completion": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "response_clarity_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "appointment_booking_accuracy": 0.35,
    "workflow_completion": 0.30,
    "detail_specificity": 0.20,
    "response_clarity_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())