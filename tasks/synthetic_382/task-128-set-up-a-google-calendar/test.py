"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Set up a Google Calendar weekly schedule for a remote worker with specific focus blocks and reminders.
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


TASK_INSTRUCTION = """Set up a Google Calendar weekly schedule for a remote worker based in Chicago. Include three daily pre-scheduled focus blocks (9 AM–12 PM, 2–3 PM, 4–5 PM), and add reminders for lunch at 12:30 PM and a mid-day stretch break at 1:30 PM."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a Google Calendar weekly schedule for a remote worker based in Chicago. The schedule must include three daily focus blocks (9 AM–12 PM, 2–3 PM, 4–5 PM) and reminders for lunch at 12:30 PM and a mid-day stretch break at 1:30 PM. A successful completion includes all required events and reminders correctly scheduled.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a Google Calendar weekly schedule for a remote worker based in Chicago. Include three daily pre-scheduled focus blocks (9 AM–12 PM, 2–3 PM, 4–5 PM), and add reminders for lunch at 12:30 PM and a mid-day stretch break at 1:30 PM.

## Task-Specific Constraints
- Must use Google Calendar to create the schedule.
- Must include three focus blocks per day at the specified times.
- Must include reminders for lunch at 12:30 PM and a stretch break at 1:30 PM.
- The schedule must cover all weekdays (Monday to Friday).
- The response must clearly list all events and reminders created.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Google Calendar to create the schedule?
- Are all three daily focus blocks (9 AM–12 PM, 2–3 PM, 4–5 PM) present for each weekday?
- Are reminders for lunch at 12:30 PM and a stretch break at 1:30 PM included for each weekday?
- Is the schedule clearly presented in the response?
- Are there any errors or omissions in the schedule or reminders?

### Step 2: Dimension Scoring

#### A. Schedule Completeness (0.35)
Measures whether the schedule includes all required focus blocks and reminders.

5 — All focus blocks and reminders are present for all weekdays.
4 — One or two minor omissions (e.g., a single block or reminder missing).
3 — Several omissions but the schedule is partially usable.
2 — Many omissions, making the schedule largely incomplete.
1 — No meaningful schedule created.

#### B. Adherence to Instructions (0.30)
Measures whether the agent followed the specific times and structure outlined in the task.

5 — All times and structure match the instructions exactly.
4 — Minor deviations (e.g., slight timing errors).
3 — Some deviations but the overall structure is recognizable.
2 — Major deviations from the instructions.
1 — No adherence to the instructions.

#### C. Clarity of Output (0.20)
Measures how clearly the schedule is presented in the response.

5 — The schedule is well-organized and easy to understand.
4 — Mostly clear but with minor formatting issues.
3 — Somewhat clear but requires effort to interpret.
2 — Poorly organized and difficult to follow.
1 — Completely unclear or disorganized.

#### D. Use of Google Calendar (0.15)
Measures whether the agent used Google Calendar as required.

5 — Google Calendar was used and evidence is clearly provided.
4 — Google Calendar was used but evidence is incomplete.
3 — Unclear whether Google Calendar was used but the schedule is plausible.
2 — Google Calendar was not used but an attempt was made.
1 — No attempt to use Google Calendar.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "schedule_completeness": <1-5>,
  "adherence_to_instructions": <1-5>,
  "clarity_of_output": <1-5>,
  "use_of_google_calendar": <1-5>,
  "dimension_reasoning": {{
    "schedule_completeness": "<one sentence citing specific evidence>",
    "adherence_to_instructions": "<one sentence citing specific evidence>",
    "clarity_of_output": "<one sentence citing specific evidence>",
    "use_of_google_calendar": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "schedule_completeness": 0.35,
    "adherence_to_instructions": 0.30,
    "clarity_of_output": 0.20,
    "use_of_google_calendar": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())