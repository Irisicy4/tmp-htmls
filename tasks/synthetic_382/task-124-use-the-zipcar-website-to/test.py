"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Search for car availability near Boston, MA, for this Saturday on Zipcar, filter for vehicles seating at least 5 passengers, and report the total number of available cars along with their hourly rates.
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


TASK_INSTRUCTION = """Use the Zipcar website to complete a search for car availability near Boston, MA, for this Saturday. Filter for vehicles that seat at least 5 passengers and report the total number of available cars along with their hourly rates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use the Zipcar website to search for car availability near Boston, MA, for this Saturday. The agent must filter for vehicles that seat at least 5 passengers and report the total number of available cars along with their hourly rates. A successful completion includes accurate filtering, correct reporting of the number of cars, and hourly rates.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Zipcar website to complete a search for car availability near Boston, MA, for this Saturday. Filter for vehicles that seat at least 5 passengers and report the total number of available cars along with their hourly rates.

## Task-Specific Constraints
- Must navigate to the Zipcar website.
- Must search for car availability specifically near Boston, MA.
- Must apply a filter for vehicles that seat at least 5 passengers.
- Must report the total number of available cars.
- Must include hourly rates for all reported cars.
- Output must be structured as a table or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Zipcar website?
- Did the agent search for car availability near Boston, MA, for this Saturday?
- Did the agent apply the filter for vehicles seating at least 5 passengers?
- Are the total number of available cars and their hourly rates present in the response?
- Is the output organized as a table or structured list?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly reported the total number of cars and their hourly rates.

5 — Reports the exact number of available cars and all hourly rates accurately.
4 — Reports the number of cars and hourly rates with minor inaccuracies.
3 — Reports the number of cars but misses some hourly rates or has notable inaccuracies.
2 — Reports incomplete or mostly incorrect data.
1 — Fails to report the number of cars or hourly rates.

#### B. Coverage of Requirements (0.30)
Measures whether the agent followed all task-specific constraints.

5 — Fully satisfies all constraints (e.g., navigates Zipcar, applies filters, searches Boston, etc.).
4 — Satisfies most constraints but misses minor details.
3 — Satisfies some constraints but misses key requirements.
2 — Satisfies few constraints or performs incorrectly.
1 — Fails to satisfy any constraints.

#### C. Depth and Specificity (0.20)
Measures the level of detail in the response, including hourly rates and structured output.

5 — Provides detailed hourly rates for all cars and organizes output as a table or list.
4 — Provides most hourly rates with minor omissions or formatting issues.
3 — Provides some hourly rates but lacks detail or proper structure.
2 — Provides minimal detail or unstructured output.
1 — Provides no detail or structure.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and credible.

5 — Output is fully structured, clear, and credible.
4 — Output is mostly structured and credible with minor issues.
3 — Output is partially structured but lacks clarity or credibility.
2 — Output is poorly structured or unclear.
1 — Output is unstructured or not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
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
    "coverage_of_requirements": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())