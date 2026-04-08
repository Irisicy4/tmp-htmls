"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Fetch gas prices, calculate round trip cost, and compare with carpool option.
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


TASK_INSTRUCTION = """Fetch current gas prices from AAA Gas Prices and calculate the estimated cost of a round trip from Denver to Aspen (360 miles total) for a car averaging 25 miles per gallon. Recommend whether using a carpool option (costing $50 per person for 3 people) is cheaper."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to fetch current gas prices from AAA Gas Prices, calculate the cost of a round trip from Denver to Aspen (360 miles total) for a car averaging 25 miles per gallon, and compare it with a carpool option costing $50 per person for 3 people. A successful completion includes accurate calculations, correct gas price data, and a clear recommendation on which option is cheaper.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Fetch current gas prices from AAA Gas Prices and calculate the estimated cost of a round trip from Denver to Aspen (360 miles total) for a car averaging 25 miles per gallon. Recommend whether using a carpool option (costing $50 per person for 3 people) is cheaper.

## Task-Specific Constraints
- Must fetch gas prices from gasprices.aaa.com.
- Must calculate the total gas cost for the round trip based on the fetched gas price.
- Must compare the calculated gas cost with the carpool option.
- Must provide a clear recommendation on which option is cheaper.
- Output must include the fetched gas price, calculated gas cost, and comparison details.
- Must use accurate math and include all necessary details in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to gasprices.aaa.com to fetch gas prices?
- Did the agent calculate the gas cost for a 360-mile round trip using the fetched gas price and the given fuel efficiency?
- Did the agent compare the calculated gas cost with the carpool option?
- Is the output structured clearly, including all required details (gas price, gas cost, comparison, recommendation)?
- Are the calculations accurate and based on the provided constraints?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent's main output (cost comparison and recommendation) is accurate and complete.

5 — Provides an accurate cost comparison and a clear, correct recommendation.
4 — Minor errors in calculations or recommendation, but mostly accurate.
3 — Partially complete; some errors in calculations or missing details.
2 — Significant errors in calculations or recommendation.
1 — No accurate calculations or recommendation.

#### B. Source Coverage (0.30)
Measures whether the agent used the required source (gasprices.aaa.com) and included all necessary data.

5 — Fetches gas prices from gasprices.aaa.com and includes all required data.
4 — Fetches gas prices but omits some minor details.
3 — Attempts to fetch gas prices but misses key data or uses an incorrect source.
2 — Does not fetch gas prices or uses an entirely wrong source.
1 — No attempt to fetch gas prices.

#### C. Calculation Depth (0.20)
Measures the accuracy and depth of the calculations.

5 — All calculations are accurate and detailed, with clear steps shown.
4 — Minor errors in calculations, but the process is mostly correct.
3 — Some calculations are correct, but there are significant errors or omissions.
2 — Most calculations are incorrect or missing.
1 — No calculations attempted.

#### D. Output Structure and Clarity (0.15)
Measures how well the response is organized and presented.

5 — Output is well-structured, clear, and includes all required elements.
4 — Output is mostly clear but has minor formatting or structural issues.
3 — Output is somewhat clear but lacks organization or key elements.
2 — Output is poorly structured or difficult to follow.
1 — Output is completely unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "calculation_depth": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "calculation_depth": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "source_coverage": 0.30,
    "calculation_depth": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())