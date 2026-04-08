"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Filter flights from Chicago to Tokyo departing March 15th and returning March 25th, extract economy class options with baggage included and lowest prices.
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


TASK_INSTRUCTION = """Go to popular flight booking websites and filter for flights from Chicago to Tokyo departing on March 15th and returning on March 25th. Extract economy class options with airlines offering baggage included and the lowest prices. Use sites like Kayak, Expedia, and Skyscanner."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit popular flight booking websites (Kayak, Expedia, Skyscanner) and filter flights from Chicago to Tokyo departing March 15th and returning March 25th. The agent must extract economy class options with baggage included and identify the lowest prices. A successful completion includes structured output with airline names, prices, and baggage details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to popular flight booking websites and filter for flights from Chicago to Tokyo departing on March 15th and returning on March 25th. Extract economy class options with airlines offering baggage included and the lowest prices. Use sites like Kayak, Expedia, and Skyscanner.

## Task-Specific Constraints
- Must visit at least 3 specified platforms: Kayak, Expedia, Skyscanner.
- Must include airline names, prices, and baggage details for all options extracted.
- Output must be organized as a structured table or list.
- Must identify the lowest-priced economy class options with baggage included.
- Must ensure departure and return dates match the task instruction.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are airline names, prices, and baggage details present in the response?
- Is the output organized as a structured table or list?
- Did the agent correctly filter for the specified departure and return dates?
- Are the lowest-priced economy class options with baggage included accurately identified?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly extracted and identified the lowest-priced economy class options with baggage included.

5 — Extracts all required data (airline names, prices, baggage details) and correctly identifies the lowest-priced options.
4 — Extracts most required data but may miss minor details or slightly misidentify lowest-priced options.
3 — Extracts partial data; lowest-priced options may be incomplete or inaccurate.
2 — Extracts minimal data; lowest-priced options are mostly incorrect.
1 — Fails to extract data or identify lowest-priced options.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and utilized all required platforms (Kayak, Expedia, Skyscanner).

5 — Successfully visits and extracts data from all three platforms.
4 — Visits and extracts data from two platforms; minor omissions.
3 — Visits and extracts data from one platform; significant omissions.
2 — Visits platforms but fails to extract usable data.
1 — Fails to visit required platforms.

#### C. Depth and Specificity (0.25)
Measures the level of detail and specificity in the extracted data (e.g., accurate prices, baggage details).

5 — Provides highly detailed and accurate data for all options.
4 — Provides mostly detailed and accurate data; minor inconsistencies.
3 — Provides partially detailed data; some inaccuracies or missing details.
2 — Provides minimal detail; significant inaccuracies or omissions.
1 — Provides no usable detail.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and the data appears credible.

5 — Output is highly structured (e.g., table or list) and data is credible.
4 — Output is mostly structured; minor formatting issues or credibility concerns.
3 — Output is partially structured; moderate formatting or credibility issues.
2 — Output is poorly structured; significant credibility concerns.
1 — Output is unstructured or completely lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())