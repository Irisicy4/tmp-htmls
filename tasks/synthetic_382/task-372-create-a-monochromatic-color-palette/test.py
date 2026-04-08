"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Create a monochromatic color palette for a minimalist website design and export it as HEX codes.
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


TASK_INSTRUCTION = """Create a monochromatic color palette suitable for a minimalist website design using an online design tool. Select a primary color of your choice, generate five shades, and export the palette as a swatch or list of HEX codes."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a monochromatic color palette suitable for a minimalist website design. The agent must select a primary color, generate five shades, and export the palette as a swatch or list of HEX codes. The task is in the domain of design and requires using online tools.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a monochromatic color palette suitable for a minimalist website design using an online design tool. Select a primary color of your choice, generate five shades, and export the palette as a swatch or list of HEX codes.

## Task-Specific Constraints
- Must use at least one of the specified platforms (coolors.co, photopea.com, canva.com).
- Must generate exactly five shades of the primary color.
- Must export the palette as a swatch or list of HEX codes.
- HEX codes must be valid and formatted correctly.
- The palette must be monochromatic (all shades derived from the same primary color).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use at least one of the specified platforms? Which platform(s)?
- Did the agent generate exactly five shades of the primary color?
- Are the HEX codes valid and formatted correctly?
- Is the palette monochromatic (all shades derived from the same primary color)?
- Was the palette exported as a swatch or list of HEX codes?

### Step 2: Dimension Scoring

#### A. Palette Accuracy (0.35)
Measures whether the generated palette is monochromatic and contains exactly five valid HEX codes.

5 — Palette is monochromatic and contains five valid HEX codes.
4 — Palette is monochromatic but contains minor formatting issues or one invalid HEX code.
3 — Palette is monochromatic but contains fewer than five shades or multiple invalid HEX codes.
2 — Palette is not monochromatic or contains significant errors.
1 — No valid palette generated.

#### B. Platform Utilization (0.30)
Measures whether the agent used at least one of the specified platforms.

5 — Agent used multiple specified platforms effectively.
4 — Agent used one specified platform effectively.
3 — Agent used one specified platform but with limited effectiveness.
2 — Agent attempted but failed to use any specified platform effectively.
1 — Agent did not use any specified platform.

#### C. Output Quality (0.20)
Measures the organization and clarity of the exported palette.

5 — Palette is exported as a well-organized swatch or structured list of HEX codes.
4 — Palette is exported but lacks minor organizational clarity.
3 — Palette is exported but lacks significant clarity or structure.
2 — Palette is exported but is disorganized or incomplete.
1 — Palette was not exported.

#### D. Execution Trace Completeness (0.15)
Measures whether the tool-call trace provides sufficient evidence of task completion.

5 — Execution trace clearly shows all steps taken to complete the task.
4 — Execution trace shows most steps taken but lacks minor details.
3 — Execution trace shows partial steps taken but lacks significant details.
2 — Execution trace shows minimal steps taken or is unclear.
1 — Execution trace is absent or irrelevant.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "palette_accuracy": <1-5>,
  "platform_utilization": <1-5>,
  "output_quality": <1-5>,
  "execution_trace_completeness": <1-5>,
  "dimension_reasoning": {{
    "palette_accuracy": "<one sentence citing specific evidence>",
    "platform_utilization": "<one sentence citing specific evidence>",
    "output_quality": "<one sentence citing specific evidence>",
    "execution_trace_completeness": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "palette_accuracy": 0.35,
    "platform_utilization": 0.30,
    "output_quality": 0.20,
    "execution_trace_completeness": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())