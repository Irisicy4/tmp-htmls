"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Set up a Google Calendar with a week-long study schedule using recommendations from multiple platforms.
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


TASK_INSTRUCTION = """Set up a Google Calendar to organize a week-long study schedule for an upcoming exam. Schedule three daily sessions (morning, afternoon, evening) and include reminders. Find session recommendations from Khan Academy, Quizlet, and SparkNotes to fill the study slots."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a week-long study schedule in Google Calendar, with three daily study sessions (morning, afternoon, evening) and reminders for each session. The agent must use recommendations sourced from Khan Academy, Quizlet, and SparkNotes to fill the study slots.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a Google Calendar to organize a week-long study schedule for an upcoming exam. Schedule three daily sessions (morning, afternoon, evening) and include reminders. Find session recommendations from Khan Academy, Quizlet, and SparkNotes to fill the study slots.

## Task-Specific Constraints
- Must visit at least three specified platforms: calendar.google.com, khanacademy.org, quizlet.com, sparknotes.com.
- Must create a structured schedule with three daily study sessions per day for seven days.
- Each session must include recommendations sourced from the platforms.
- Reminders must be set for each session.
- Output must be organized in a clear, structured format (e.g., table or list).
- Must provide evidence of platform navigation and content sourcing.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent create a structured schedule with three daily sessions per day for seven days?
- Are the sessions filled with recommendations sourced from the specified platforms?
- Are reminders included for each session?
- Is the output organized in a clear, structured format?

### Step 2: Dimension Scoring

#### A. Schedule Accuracy (0.35)
Measures whether the schedule is correctly structured with three daily sessions per day for seven days.

5 — All seven days include three sessions, correctly labeled (morning, afternoon, evening).
4 — Six days include three sessions; one day may have errors or omissions.
3 — At least five days include three sessions; others may be incomplete.
2 — Fewer than five days include three sessions; major omissions.
1 — No structured schedule created.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and sourced recommendations from the required platforms.

5 — Recommendations sourced from all four specified platforms.
4 — Recommendations sourced from three platforms.
3 — Recommendations sourced from two platforms.
2 — Recommendations sourced from one platform.
1 — No evidence of platform usage.

#### C. Recommendation Specificity (0.20)
Measures the quality and specificity of the recommendations included in the schedule.

5 — Recommendations are detailed, specific, and clearly tied to the platforms.
4 — Recommendations are somewhat detailed but may lack specificity.
3 — Recommendations are generic or partially complete.
2 — Recommendations are vague or mostly missing.
1 — No recommendations included.

#### D. Output Organization (0.15)
Measures whether the schedule and reminders are presented in a clear, structured format.

5 — Output is highly organized (e.g., table or structured list) with all reminders clearly included.
4 — Output is organized but may have minor formatting issues.
3 — Output is partially organized; reminders may be incomplete.
2 — Output is disorganized or difficult to follow.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "schedule_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "recommendation_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "schedule_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "recommendation_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "schedule_accuracy": 0.35,
    "platform_coverage": 0.30,
    "recommendation_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())