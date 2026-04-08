"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Compare flight prices for round-trip economy class, non-stop flights from New York (JFK) to London (LHR) across three platforms.
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


TASK_INSTRUCTION = """Complete a flight booking price comparison workflow. Search for round-trip flights from New York (JFK) to London (LHR) on August 10-17, 2024. Use Skyscanner, Kayak, and Google Flights to select economy class, non-stop flights. Report the lowest fares from each platform."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves comparing flight prices for round-trip economy class, non-stop flights from New York (JFK) to London (LHR) across three platforms: Skyscanner, Kayak, and Google Flights. A successful completion requires the agent to report the lowest fares from each platform in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete a flight booking price comparison workflow. Search for round-trip flights from New York (JFK) to London (LHR) on August 10-17, 2024. Use Skyscanner, Kayak, and Google Flights to select economy class, non-stop flights. Report the lowest fares from each platform.

## Task-Specific Constraints
- Must visit Skyscanner, Kayak, and Google Flights.
- Must include price data for economy class, non-stop flights only.
- Must report the lowest fare from each platform.
- Output must be organized as a table or structured list.
- Must specify the source platform for each price.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Skyscanner, Kayak, and Google Flights? Which platforms were actually visited?
- Are economy class, non-stop flight prices included for each platform?
- Is the output organized as a table or structured list?
- Are the lowest fares correctly reported and sourced from the respective platforms?
- Are any required elements missing or incorrect?

### Step 2: Dimension Scoring

#### A. Price Accuracy (0.35)
Measures whether the reported lowest fares are correct and match the specified criteria.

5 — All reported fares are correct and match the economy class, non-stop flight criteria.
4 — Most fares are correct, but one contains minor inaccuracies.
3 — At least one fare is correct, but others are missing or inaccurate.
2 — Fares are mostly incorrect or missing.
1 — No fares are reported or all are incorrect.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Data from all three platforms (Skyscanner, Kayak, Google Flights) is included.
4 — Data from two platforms is included, with minor omissions.
3 — Data from one platform is included, or partial data from multiple platforms.
2 — Minimal platform coverage; data from only one platform is incomplete.
1 — No platform data is included.

#### C. Detail Specificity (0.20)
Measures whether the agent provides detailed and specific information for each fare.

5 — Each fare includes specific details (e.g., price, source platform, flight type).
4 — Most fares include specific details, with minor omissions.
3 — Some fares include details, but others lack specificity.
2 — Details are mostly missing or vague.
1 — No specific details are provided.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required elements.
4 — Output is mostly well-organized, with minor formatting issues.
3 — Output is partially organized but lacks clarity or completeness.
2 — Output is poorly organized and hard to interpret.
1 — Output is unstructured or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "price_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "price_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "price_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())