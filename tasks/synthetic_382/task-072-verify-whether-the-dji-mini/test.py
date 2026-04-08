"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Verify whether the DJI Mini 3 drone is consistently priced and has matching specifications across Amazon, Best Buy, and B&H Photo Video.
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


TASK_INSTRUCTION = """Verify whether the DJI Mini 3 drone is consistently priced across Amazon, Best Buy, and B&H Photo Video. Check if the specifications (battery life and camera quality) match across all listings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare the price and specifications (battery life and camera quality) of the DJI Mini 3 drone across Amazon, Best Buy, and B&H Photo Video. A successful completion involves visiting all three platforms, extracting and comparing the required data, and presenting the findings in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the DJI Mini 3 drone is consistently priced across Amazon, Best Buy, and B&H Photo Video. Check if the specifications (battery life and camera quality) match across all listings.

## Task-Specific Constraints
- Must visit Amazon, Best Buy, and B&H Photo Video.
- Must extract price, battery life, and camera quality from each platform.
- Must include price and specification data for all three platforms in the output.
- Output must be organized as a table or structured list.
- Must explicitly state whether the prices and specifications match across platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Amazon, Best Buy, and B&H Photo Video? Which platforms were actually visited?
- Are the price, battery life, and camera quality data present for all three platforms?
- Is the output organized as a table or structured list?
- Does the response explicitly state whether prices and specifications match across platforms?
- Are the extracted data accurate and sourced correctly?

### Step 2: Dimension Scoring

#### A. Price and Specification Accuracy (0.35)
Measures whether the agent correctly extracted and compared prices and specifications.

5 — Extracted and compared prices, battery life, and camera quality for all three platforms accurately.
4 — Extracted most data correctly but missed minor details or comparisons.
3 — Extracted partial data but missed significant comparisons or details.
2 — Extracted very little data or made significant errors in comparison.
1 — Did not extract or compare any relevant data.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and included their data.

5 — Visited all three platforms and included their data in the response.
4 — Visited two platforms and included their data.
3 — Visited one platform and included its data.
2 — Attempted but failed to include data from any platform.
1 — Did not visit or include data from any platform.

#### C. Detail and Specificity (0.20)
Measures the depth and specificity of the response, including structured comparisons.

5 — Provided detailed comparisons with specific numbers and clear conclusions.
4 — Provided comparisons but lacked minor details or clarity.
3 — Provided partial comparisons but missed significant details.
2 — Provided vague or incomplete comparisons.
1 — Did not provide any comparisons.

#### D. Output Structure and Credibility (0.15)
Measures whether the output is well-organized and uses credible sources.

5 — Output is structured as a table or list and cites credible sources.
4 — Output is structured but lacks minor formatting or source credibility.
3 — Output is partially structured but lacks clarity or credibility.
2 — Output is poorly structured and lacks credibility.
1 — Output is unstructured and not credible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "price_and_specification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "price_and_specification_accuracy": "<one sentence citing specific evidence>",
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
    "price_and_specification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())