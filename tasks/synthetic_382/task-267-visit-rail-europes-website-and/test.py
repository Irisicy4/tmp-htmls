"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Book a round-trip train ticket from London to Edinburgh on Rail Europe's website, applying filters for 'Standard Class' and reporting the total price and ticket terms.
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


TASK_INSTRUCTION = """Visit Rail Europe's website and complete the booking workflow for a round-trip train ticket from London to Edinburgh departing July 12 and returning July 14. Use filters for 'Standard Class' and report the total price and ticket terms shown on the final checkout screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to book a round-trip train ticket from London to Edinburgh on Rail Europe's website, applying filters for 'Standard Class'. The agent must report the total price and ticket terms shown on the final checkout screen. This task is in the Travel & Planning domain and involves navigating a booking workflow.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit Rail Europe's website and complete the booking workflow for a round-trip train ticket from London to Edinburgh departing July 12 and returning July 14. Use filters for 'Standard Class' and report the total price and ticket terms shown on the final checkout screen.

## Task-Specific Constraints
- Must navigate Rail Europe's website and use its booking workflow.
- Must apply filters for 'Standard Class' tickets.
- Must provide the total price shown on the final checkout screen.
- Must include ticket terms (e.g., refundability, seat reservation details).
- Output must be structured as a clear summary or table.
- Must accurately reflect the data from the final checkout screen.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Rail Europe's website and use the booking workflow?
- Did the agent apply the 'Standard Class' filter correctly?
- Is the total price of the tickets included in the response?
- Are the ticket terms (e.g., refundability, seat reservation details) present?
- Is the output structured as a clear summary or table?

### Step 2: Dimension Scoring

#### A. Booking Accuracy (0.35)
Measures whether the agent successfully completed the booking workflow and reported the correct total price.

5 — Booking workflow completed successfully; total price reported accurately.
4 — Booking workflow completed; minor inaccuracies in total price.
3 — Booking workflow partially completed; total price unclear or incomplete.
2 — Booking workflow mostly incomplete; total price missing or incorrect.
1 — Booking workflow not attempted or completely incorrect.

#### B. Filter Application (0.30)
Measures whether the agent correctly applied the 'Standard Class' filter.

5 — 'Standard Class' filter applied correctly and verified in the response.
4 — 'Standard Class' filter applied; minor errors in verification.
3 — 'Standard Class' filter partially applied or unclear.
2 — 'Standard Class' filter mostly incorrect or missing.
1 — 'Standard Class' filter not applied or completely incorrect.

#### C. Ticket Terms Inclusion (0.20)
Measures whether the agent included ticket terms (e.g., refundability, seat reservation details).

5 — Ticket terms included comprehensively and accurately.
4 — Ticket terms included; minor omissions or inaccuracies.
3 — Ticket terms partially included or unclear.
2 — Ticket terms mostly missing or incorrect.
1 — Ticket terms not included or completely incorrect.

#### D. Output Structure (0.15)
Measures whether the agent's response is well-organized and easy to understand.

5 — Output is structured as a clear summary or table; easy to interpret.
4 — Output is structured; minor formatting issues.
3 — Output partially structured; somewhat unclear.
2 — Output mostly unstructured or hard to interpret.
1 — Output completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "booking_accuracy": <1-5>,
  "filter_application": <1-5>,
  "ticket_terms_inclusion": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "booking_accuracy": "<one sentence citing specific evidence>",
    "filter_application": "<one sentence citing specific evidence>",
    "ticket_terms_inclusion": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "booking_accuracy": 0.35,
    "filter_application": 0.30,
    "ticket_terms_inclusion": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())