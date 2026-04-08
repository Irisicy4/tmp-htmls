"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a complementary color palette using Pantone's trending colors for 2023 and Coolors.co.
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


TASK_INSTRUCTION = """Visit Pantone's website to find the hex codes for their trending colors for 2023. Then, visit Coolors.co to build a complementary color palette with three additional colors. Report the full palette with hex codes and the reasoning behind the harmony of the chosen colors."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit Pantone's website to identify the hex codes for their trending colors for 2023. Then, the agent must use Coolors.co to generate a complementary color palette with three additional colors. A successful completion includes reporting the full palette (hex codes) and providing reasoning for the harmony of the chosen colors.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit Pantone's website to find the hex codes for their trending colors for 2023. Then, visit Coolors.co to build a complementary color palette with three additional colors. Report the full palette with hex codes and the reasoning behind the harmony of the chosen colors.

## Task-Specific Constraints
- Must extract hex codes for at least three trending colors from Pantone's website.
- Must use Coolors.co to generate a complementary palette with exactly three additional colors.
- Must report all hex codes in a structured format (e.g., a list or table).
- Must provide reasoning for the harmony of the chosen colors.
- Must demonstrate evidence of visiting both Pantone.com and Coolors.co.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Pantone.com and Coolors.co? Which platforms were actually visited?
- Are hex codes for at least three trending colors from Pantone present in the response?
- Are three additional complementary colors included in the palette?
- Is the output organized in a structured format (e.g., list or table)?
- Does the agent provide reasoning for the harmony of the chosen colors?

### Step 2: Dimension Scoring

#### A. Palette Completeness (0.35)
Measures whether the full palette (Pantone + complementary colors) is correctly reported.

5 — All six hex codes (three Pantone + three complementary) are present and accurate.
4 — Five hex codes are present and accurate.
3 — At least four hex codes are present and accurate.
2 — Fewer than four hex codes are present or mostly incorrect.
1 — No hex codes are present or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited both required platforms and extracted relevant data.

5 — Evidence shows the agent visited both Pantone.com and Coolors.co and extracted required data.
4 — Evidence shows the agent visited both platforms but extracted incomplete data.
3 — Evidence shows the agent visited at least one platform and extracted partial data.
2 — Evidence shows the agent visited one platform but failed to extract relevant data.
1 — No evidence of visiting either platform.

#### C. Reasoning Quality (0.20)
Measures the quality and depth of reasoning provided for the harmony of the chosen colors.

5 — Reasoning is detailed, logical, and references color theory principles.
4 — Reasoning is logical but lacks depth or references.
3 — Reasoning is present but shallow or vague.
2 — Reasoning is mostly absent or illogical.
1 — No reasoning provided.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with proper formatting.
4 — Output is structured but contains minor formatting issues.
3 — Output is partially structured but lacks clarity.
2 — Output is mostly unstructured or confusing.
1 — Output is completely unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "palette_completeness": <1-5>,
  "platform_coverage": <1-5>,
  "reasoning_quality": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "palette_completeness": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "reasoning_quality": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "palette_completeness": 0.35,
    "platform_coverage": 0.30,
    "reasoning_quality": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())