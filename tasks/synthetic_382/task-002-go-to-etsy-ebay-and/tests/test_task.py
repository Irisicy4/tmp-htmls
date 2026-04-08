"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Extract information on handmade tote bags under $40 from Etsy, eBay, and Alibaba, applying specific filters.
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


TASK_INSTRUCTION = """Go to Etsy, eBay, and Alibaba, and navigate their search filters to extract information on handmade tote bags under $40. Apply filters for material type (e.g., canvas or cotton), seller review ratings (4 stars or higher), and shipping location (USA only). Extract and report the top 5 matching options for each platform, including price and seller ratings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting information on handmade tote bags under $40 from Etsy, eBay, and Alibaba. The agent must apply filters for material type (canvas or cotton), seller review ratings (4 stars or higher), and shipping location (USA only). A successful completion includes reporting the top 5 matching options for each platform, including price and seller ratings.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Etsy, eBay, and Alibaba, and navigate their search filters to extract information on handmade tote bags under $40. Apply filters for material type (e.g., canvas or cotton), seller review ratings (4 stars or higher), and shipping location (USA only). Extract and report the top 5 matching options for each platform, including price and seller ratings.

## Task-Specific Constraints
- Must visit Etsy, eBay, and Alibaba.
- Must apply filters for material type, seller review ratings, and shipping location.
- Must extract price and seller ratings for each item.
- Must report exactly 5 matching options per platform.
- Output must be structured as a table or list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Etsy, eBay, and Alibaba? Which platforms were actually visited?
- Did the agent apply the required filters (material type, seller review ratings, shipping location)?
- Are price and seller ratings included for all reported items?
- Is the output organized as a table or structured list?
- Are there exactly 5 matching options reported per platform?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent's output correctly identifies the top 5 options per platform with required details.

5 — All platforms have 5 correct options with accurate filters applied and details included.
4 — Most platforms have 5 correct options; minor errors in filters or details.
3 — At least one platform has 5 correct options; others incomplete or partially correct.
2 — Few platforms have correct options; major errors in filters or details.
1 — No correct options identified.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and extracted data.

5 — All three platforms visited and data extracted.
4 — Two platforms visited and data extracted.
3 — At least one platform visited and data extracted.
2 — Platforms visited but no usable data extracted.
1 — No platforms visited.

#### C. Filter Application Depth (0.25)
Measures whether the agent applied all required filters (material type, seller review ratings, shipping location).

5 — All filters applied correctly across all platforms.
4 — Most filters applied correctly; minor omissions.
3 — Some filters applied correctly; significant omissions.
2 — Few filters applied; major omissions.
1 — No filters applied.

#### D. Output Structure and Clarity (0.10)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required details.
4 — Output is mostly clear; minor formatting issues.
3 — Output is usable but lacks clarity or structure.
2 — Output is poorly organized and hard to interpret.
1 — Output is unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "filter_application_depth": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "filter_application_depth": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "filter_application_depth": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())