"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Research and compare three flight options from New York City (JFK) to Tokyo (NRT) departing on the 15th of next month and returning on the 22nd, using Skyscanner, Kayak, and Expedia.
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


TASK_INSTRUCTION = """Research and compare three flight options from New York City (JFK) to Tokyo (NRT) departing on the 15th of next month and returning on the 22nd. Include major airlines, economy class tickets, and price, layover duration, and total travel time as criteria for comparison. Use Skyscanner, Kayak, and Expedia for your research."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research and compare three flight options from New York City (JFK) to Tokyo (NRT), departing on the 15th of next month and returning on the 22nd. The agent must use Skyscanner, Kayak, and Expedia to gather data. A successful completion includes providing economy class ticket options from major airlines, along with price, layover duration, and total travel time for comparison.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three flight options from New York City (JFK) to Tokyo (NRT) departing on the 15th of next month and returning on the 22nd. Include major airlines, economy class tickets, and price, layover duration, and total travel time as criteria for comparison. Use Skyscanner, Kayak, and Expedia for your research.

## Task-Specific Constraints
- Must visit Skyscanner, Kayak, and Expedia during the task.
- Must include price, layover duration, and total travel time for all three flight options.
- Must compare flights from major airlines only.
- Output must be organized as a table or structured list.
- Must specify departure and return dates clearly in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Skyscanner, Kayak, and Expedia? Which platforms were actually visited?
- Are price, layover duration, and total travel time included for all three flight options?
- Are the flight options from major airlines only?
- Is the output organized as a table or structured list?
- Are the departure and return dates clearly specified in the response?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the flight comparison data is correct, complete, and matches the task requirements.

5 — All three flight options include accurate price, layover duration, and total travel time.
4 — Two flight options are complete and accurate; one is partially complete.
3 — At least one flight option is complete and accurate; others are partially complete.
2 — Data is mostly incomplete or inaccurate.
1 — No usable flight comparison data is provided.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Agent visited Skyscanner, Kayak, and Expedia and gathered data from all three.
4 — Agent visited at least two platforms and gathered data from them.
3 — Agent visited one platform and gathered data from it.
2 — Agent visited platforms but failed to gather usable data.
1 — Agent did not visit any required platforms.

#### C. Detail Specificity (0.25)
Measures whether the response includes detailed comparisons (e.g., specific airline names, layover times).

5 — Response includes airline names, layover times, and all required details for all options.
4 — Response includes most required details but lacks minor specifics for one option.
3 — Response includes some required details but is missing specifics for multiple options.
2 — Response includes minimal details and lacks key specifics.
1 — Response lacks any meaningful details.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and uses credible sources.

5 — Output is organized as a clear table or structured list and cites credible sources.
4 — Output is mostly well-organized but has minor formatting issues.
3 — Output is usable but poorly organized or lacks source credibility.
2 — Output is disorganized or unclear.
1 — Output is unusable or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
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
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())