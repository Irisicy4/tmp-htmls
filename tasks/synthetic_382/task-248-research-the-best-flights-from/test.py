"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Research and summarize the best flight options from New York City (JFK) to Tokyo (NRT) across three platforms based on price, layover duration, and airline ratings.
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


TASK_INSTRUCTION = """Research the best flights from New York City (JFK) to Tokyo (NRT) departing on December 10th and returning on December 20th. Compare options across Expedia, Kayak, and Skyscanner on price, layover duration, and airline ratings, then summarize the top 3 choices."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to research flight options from New York City (JFK) to Tokyo (NRT) departing on December 10th and returning on December 20th. The agent must compare options across three platforms (Expedia, Kayak, and Skyscanner) based on price, layover duration, and airline ratings, and summarize the top 3 choices in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research the best flights from New York City (JFK) to Tokyo (NRT) departing on December 10th and returning on December 20th. Compare options across Expedia, Kayak, and Skyscanner on price, layover duration, and airline ratings, then summarize the top 3 choices.

## Task-Specific Constraints
- Must visit Expedia, Kayak, and Skyscanner during the research process.
- Must include price, layover duration, and airline ratings for all compared options.
- Output must be organized as a structured list or table summarizing the top 3 choices.
- Must clearly identify the top 3 flight options based on the comparison criteria.
- Must provide specific numeric values (e.g., prices, layover durations, ratings) for each option.
- Must avoid vague or incomplete descriptions.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Expedia, Kayak, and Skyscanner? Which platforms were actually visited?
- Are price, layover duration, and airline ratings present for all compared options?
- Is the output organized as a structured list or table summarizing the top 3 choices?
- Are the numeric values (e.g., prices, layover durations, ratings) accurate and sourced from the platforms visited?
- Does the response clearly identify the top 3 flight options based on the comparison criteria?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's final output correctly and completely summarizes the top 3 flight options based on the given criteria.

5 — Summarizes 3 flight options with accurate price, layover duration, and airline ratings for each.
4 — Summarizes 3 flight options but with minor inaccuracies or omissions in one criterion.
3 — Summarizes fewer than 3 options or omits multiple criteria but provides usable information.
2 — Summarizes fewer than 3 options with significant inaccuracies or missing criteria.
1 — Does not summarize any flight options or provides unusable information.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms (Expedia, Kayak, and Skyscanner) and used them in the research.

5 — Successfully visited and used all three platforms.
4 — Visited and used two platforms but missed one.
3 — Visited only one platform but provided usable information.
2 — Attempted but failed to use any platform effectively.
1 — Did not visit any platform.

#### C. Detail and Specificity (0.25)
Measures the depth and specificity of the agent's comparisons, including numeric values and clear distinctions between options.

5 — Provides detailed numeric comparisons for all criteria across 3 options.
4 — Provides numeric comparisons for most criteria but lacks depth in one area.
3 — Provides numeric comparisons for some criteria but misses key details.
2 — Provides vague or incomplete comparisons with few numeric values.
1 — Provides no numeric comparisons or meaningful details.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and whether the data sources are credible.

5 — Output is structured as a clear table or list, and all data is sourced credibly.
4 — Output is structured but lacks clarity or has minor credibility issues.
3 — Output is partially structured but disorganized or unclear.
2 — Output is poorly structured and lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
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
    "detail_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())