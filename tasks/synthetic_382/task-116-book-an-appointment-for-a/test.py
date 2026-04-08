"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Book a haircut appointment in New York City on three platforms and report details from one.
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


TASK_INSTRUCTION = """Book an appointment for a haircut in New York City on StyleSeat, Booksy, and Fresha. Complete the booking workflow, filtering for stylists with over 4-star reviews, and report the date, time, and stylist selected from one platform."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to book a haircut appointment in New York City using three platforms: StyleSeat, Booksy, and Fresha. The agent must filter for stylists with over 4-star reviews, complete the booking workflow, and report the date, time, and stylist selected from one platform. A successful completion includes evidence of platform navigation, filtering, and structured reporting of the required details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Book an appointment for a haircut in New York City on StyleSeat, Booksy, and Fresha. Complete the booking workflow, filtering for stylists with over 4-star reviews, and report the date, time, and stylist selected from one platform.

## Task-Specific Constraints
- Must navigate to StyleSeat, Booksy, and Fresha.
- Must filter for stylists with over 4-star reviews on each platform.
- Must complete the booking workflow on at least one platform.
- Must report the date, time, and stylist name from the completed booking.
- Output must be structured as a clear list or table.
- Must provide evidence of platform navigation and filtering.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to StyleSeat, Booksy, and Fresha? Which platforms were actually visited?
- Did the agent filter for stylists with over 4-star reviews on each platform?
- Did the agent complete the booking workflow on at least one platform?
- Are the reported date, time, and stylist name present in the response?
- Is the output structured as a clear list or table?

### Step 2: Dimension Scoring

#### A. Booking Completion Accuracy (0.35)
Measures whether the agent successfully completed the booking workflow and reported the required details.

5 — Booking completed with date, time, and stylist name reported accurately.
4 — Booking completed, but one detail (date, time, or stylist name) is missing or unclear.
3 — Booking attempted but incomplete, or details are partially incorrect.
2 — Booking not completed, and details are mostly incorrect.
1 — No booking attempted or reported.

#### B. Platform Coverage (0.30)
Measures whether the agent navigated to all three specified platforms and performed the required filtering.

5 — All three platforms visited, and filtering for over 4-star reviews performed on each.
4 — Two platforms visited with filtering performed correctly.
3 — One platform visited and filtering performed.
2 — Platforms visited but no filtering performed.
1 — No platforms visited.

#### C. Detail Specificity (0.20)
Measures the presence and specificity of details in the agent's response.

5 — Response includes date, time, stylist name, and platform used, with clear formatting.
4 — Response includes most details but lacks clarity or formatting.
3 — Response includes some details but is incomplete or vague.
2 — Response includes minimal details or unclear information.
1 — No meaningful details provided.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and credible.

5 — Output is structured as a clear list or table, with credible evidence of platform navigation.
4 — Output is mostly structured but lacks clarity or evidence.
3 — Output is partially structured but disorganized or missing evidence.
2 — Output is poorly structured and lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "booking_completion_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "booking_completion_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "booking_completion_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())