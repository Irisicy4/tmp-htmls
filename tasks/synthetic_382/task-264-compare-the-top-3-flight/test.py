"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Compare top 3 flight options from New York City (JFK) to Tokyo (Narita Airport) for a round-trip in May using Google Flights, Expedia, and SkyScanner.
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


TASK_INSTRUCTION = """Compare the top 3 flight options from New York City (JFK) to Tokyo (Narita Airport) for a hypothetical round-trip in May. Include details on price, travel time, layovers, and airline reviews. Use Google Flights, Expedia, and SkyScanner for your research."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare the top 3 flight options from New York City (JFK) to Tokyo (Narita Airport) for a round-trip in May. The agent must provide details on price, travel time, layovers, and airline reviews. The research must be conducted using Google Flights, Expedia, and SkyScanner.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Compare the top 3 flight options from New York City (JFK) to Tokyo (Narita Airport) for a hypothetical round-trip in May. Include details on price, travel time, layovers, and airline reviews. Use Google Flights, Expedia, and SkyScanner for your research.

## Task-Specific Constraints
- Must visit Google Flights, Expedia, and SkyScanner.
- Must include price data for all three flight options compared.
- Must include travel time and layover details for all options.
- Must include airline reviews or ratings for all options.
- Output must be organized as a structured table or list.
- Must provide clear comparisons between the options.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Flights, Expedia, and SkyScanner? Which ones were actually visited?
- Are price, travel time, layover details, and airline reviews present for all three options?
- Is the output organized as a structured table or list?
- Are comparisons between the options clear and supported by evidence?
- Are the airline reviews or ratings credible and sourced?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent provided correct and complete comparisons of the flight options.

5 — All required details (price, travel time, layovers, airline reviews) are correct and complete for all three options.
4 — Minor inaccuracies or missing details for one option.
3 — Partial completion: significant details missing for one or more options.
2 — Major inaccuracies or missing details for most options.
1 — No meaningful comparison provided.

#### B. Platform Coverage (0.30)
Measures whether the agent used all required platforms (Google Flights, Expedia, SkyScanner).

5 — All three platforms were used and data from each was incorporated.
4 — Two platforms were used, and data was incorporated.
3 — Only one platform was used, but data was incorporated.
2 — Only one platform was used, and data was incomplete.
1 — None of the required platforms were used.

#### C. Depth of Comparison (0.25)
Measures the depth and specificity of the comparisons provided.

5 — Comparisons include detailed analysis of price, travel time, layovers, and airline reviews for all options.
4 — Comparisons include most details but lack depth in one area.
3 — Comparisons are shallow or missing depth in multiple areas.
2 — Comparisons are vague and lack meaningful detail.
1 — No comparisons provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and uses credible sources.

5 — Output is structured as a clear table or list, with credible sources cited.
4 — Output is structured but lacks clarity or credible sourcing in one area.
3 — Output is partially structured but disorganized or missing credible sourcing.
2 — Output is poorly structured and lacks credible sourcing.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_comparison": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_comparison": "<one sentence citing specific evidence>",
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
    "depth_of_comparison": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())