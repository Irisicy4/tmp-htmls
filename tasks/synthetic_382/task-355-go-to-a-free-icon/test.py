"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Extract 10 high-quality icons related to 'environmental sustainability' from free icon library platforms in SVG format, ensuring proper licensing.
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


TASK_INSTRUCTION = """Go to a free icon library platform and extract 10 high-quality icons related to 'environmental sustainability' in SVG format. Ensure all icons are free to use under Creative Commons or similar licenses. Check for licensing information before downloading."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to extract 10 high-quality icons related to 'environmental sustainability' from free icon library platforms (flaticon.com, thenounproject.com, iconfinder.com). The icons must be in SVG format and free to use under Creative Commons or similar licenses. Successful completion involves verifying licensing information and ensuring all icons meet the task criteria.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to a free icon library platform and extract 10 high-quality icons related to 'environmental sustainability' in SVG format. Ensure all icons are free to use under Creative Commons or similar licenses. Check for licensing information before downloading.

## Task-Specific Constraints
- Must visit at least 2 of the specified platforms (flaticon.com, thenounproject.com, iconfinder.com).
- Must verify licensing information for all icons extracted.
- Must provide icons in SVG format only.
- Must ensure all icons are related to 'environmental sustainability'.
- Output must include a structured list of icons with their names, sources, and licensing details.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least 2 of the required platforms? Which ones were visited?
- Are 10 icons related to 'environmental sustainability' present in the response?
- Is the licensing information for all icons verified and included?
- Are all icons provided in SVG format?
- Is the output structured as a list with names, sources, and licensing details?

### Step 2: Dimension Scoring

#### A. Icon Relevance and Accuracy (0.35)
Measures whether the icons are correctly related to 'environmental sustainability' and meet the task requirements.

5 — All 10 icons are highly relevant and accurately related to 'environmental sustainability'.
4 — 8-9 icons are relevant and accurate.
3 — 6-7 icons are relevant and accurate.
2 — 3-5 icons are relevant and accurate.
1 — Fewer than 3 icons are relevant or accurate.

#### B. Platform Coverage (0.30)
Measures whether the agent visited at least 2 of the required platforms and extracted icons from them.

5 — Icons were extracted from all 3 platforms.
4 — Icons were extracted from 2 platforms.
3 — Icons were extracted from 1 platform.
2 — Attempted but failed to extract icons from any platform.
1 — Did not attempt to visit any platform.

#### C. Licensing Verification (0.25)
Measures whether licensing information was verified and correctly included for all icons.

5 — Licensing information is verified and included for all 10 icons.
4 — Licensing information is verified and included for 8-9 icons.
3 — Licensing information is verified and included for 6-7 icons.
2 — Licensing information is verified and included for 3-5 icons.
1 — Licensing information is missing or incorrect for most icons.

#### D. Output Structure and Format (0.10)
Measures whether the output is well-organized and adheres to the required structure.

5 — Output is fully structured with names, sources, and licensing details for all icons.
4 — Output is structured but missing minor details for 1-2 icons.
3 — Output is partially structured but missing details for 3-4 icons.
2 — Output is poorly structured and missing details for 5-7 icons.
1 — Output is unstructured or missing details for most icons.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "icon_relevance_and_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "licensing_verification": <1-5>,
  "output_structure_and_format": <1-5>,
  "dimension_reasoning": {{
    "icon_relevance_and_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "licensing_verification": "<one sentence citing specific evidence>",
    "output_structure_and_format": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "icon_relevance_and_accuracy": 0.35,
    "platform_coverage": 0.30,
    "licensing_verification": 0.25,
    "output_structure_and_format": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())