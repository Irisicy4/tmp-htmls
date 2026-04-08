"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Research and identify the cause of a payment error on Booking.com and suggest a fix using information from Booking.com help center, Reddit, and TripAdvisor forums.
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


TASK_INSTRUCTION = """A user is having trouble booking a hotel room on Booking.com because the payment page displays an error after entering credit card details. Research this issue using Booking.com help center, Reddit, and TripAdvisor forums to identify the cause and suggest a fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research the cause of a payment error on Booking.com and suggest a fix. The agent must use Booking.com help center, Reddit, and TripAdvisor forums to gather information. A successful completion involves identifying the root cause of the issue and providing a clear, actionable solution based on credible sources.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user is having trouble booking a hotel room on Booking.com because the payment page displays an error after entering credit card details. Research this issue using Booking.com help center, Reddit, and TripAdvisor forums to identify the cause and suggest a fix.

## Task-Specific Constraints
- Must visit Booking.com help center, Reddit, and TripAdvisor forums.
- Must clearly identify the root cause of the payment error.
- Must provide a specific and actionable fix for the issue.
- Must cite credible sources for the information provided.
- Output must be organized as a structured list or steps.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Booking.com help center, Reddit, and TripAdvisor forums? Which ones were actually visited?
- Does the response clearly identify the root cause of the payment error?
- Is the suggested fix actionable and specific?
- Are all claims backed by credible sources?
- Is the output organized as a structured list or steps?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the payment error.

5 — Identifies the root cause with detailed explanation and supporting evidence.
4 — Identifies the root cause with minor gaps in explanation or evidence.
3 — Identifies the root cause but lacks sufficient detail or evidence.
2 — Incorrect or incomplete identification of the root cause.
1 — Fails to identify the root cause entirely.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms to gather information.

5 — Uses Booking.com help center, Reddit, and TripAdvisor forums with relevant evidence from each.
4 — Uses all three platforms but evidence from one is weak or missing.
3 — Uses at least two platforms with partial evidence.
2 — Uses only one platform or provides minimal evidence.
1 — Fails to use any of the required platforms.

#### C. Suggested Fix Specificity (0.25)
Measures whether the agent provides a clear, actionable fix for the issue.

5 — Suggests a fix that is specific, actionable, and well-supported by evidence.
4 — Suggests a fix that is actionable but lacks minor details or evidence.
3 — Suggests a fix that is partially actionable or vague.
2 — Suggests a fix that is mostly incorrect or impractical.
1 — Fails to suggest a fix entirely.

#### D. Output Organization and Credibility (0.10)
Measures the structure of the response and the credibility of the sources cited.

5 — Response is well-organized and cites credible sources for all claims.
4 — Response is organized but lacks minor source credibility.
3 — Response is usable but disorganized or missing credible sources.
2 — Response is poorly organized or lacks credible sources.
1 — Response is completely disorganized and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "suggested_fix_specificity": <1-5>,
  "output_organization_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "suggested_fix_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "suggested_fix_specificity": 0.25,
    "output_organization_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())