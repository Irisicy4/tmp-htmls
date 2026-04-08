"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Set up a style guide structure in Figma for a brand refresh project.
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


TASK_INSTRUCTION = """Set up a style guide structure in a collaborative design platform for a brand refresh project. Use Figma's community library to create categories for typography, colors, and button design. Populate each category with placeholders or sample assets to organize the style guide structure."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a style guide structure for a brand refresh project using Figma. The agent must use Figma's community library to create categories for typography, colors, and button design, and populate each category with placeholders or sample assets. Successful completion requires a well-organized style guide structure with all specified categories present and populated.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a style guide structure in a collaborative design platform for a brand refresh project. Use Figma's community library to create categories for typography, colors, and button design. Populate each category with placeholders or sample assets to organize the style guide structure.

## Task-Specific Constraints
- Must use Figma's community library to source assets.
- Must create three categories: typography, colors, and button design.
- Each category must include at least three placeholders or sample assets.
- The style guide structure must be organized and visually clear.
- Must provide evidence of visiting figma.com and using its community library.
- Must complete the task within the constraints of the instruction.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to figma.com and use the community library?
- Are the required categories (typography, colors, button design) present in the response?
- Does each category include at least three placeholders or sample assets?
- Is the style guide structure organized and visually clear as described?
- Is there evidence of visiting all required platforms and adhering to the task constraints?

### Step 2: Dimension Scoring

#### A. Style Guide Completeness (0.35)
Measures whether the style guide includes all required categories and assets.

5 — All three categories (typography, colors, button design) are present and include at least three assets each.
4 — All three categories are present, but one category has fewer than three assets.
3 — At least two categories are present, but one or more are incomplete.
2 — Only one category is present or all categories are incomplete.
1 — No categories are present or completely wrong.

#### B. Platform Usage Accuracy (0.30)
Measures whether the agent correctly used Figma's community library and visited required platforms.

5 — Clear evidence of using Figma's community library and visiting all required platforms.
4 — Evidence of using Figma's community library but unclear on other platforms.
3 — Partial evidence of platform usage (e.g., visited Figma but unclear on community library usage).
2 — Minimal evidence of platform usage.
1 — No evidence of platform usage.

#### C. Asset Quality and Specificity (0.20)
Measures the quality and specificity of placeholders or sample assets.

5 — Assets are high-quality, specific, and relevant to the categories.
4 — Assets are relevant but lack specificity or quality.
3 — Assets are generic or partially relevant.
2 — Assets are mostly irrelevant or low-quality.
1 — No assets provided or completely irrelevant.

#### D. Organization and Clarity (0.15)
Measures whether the style guide structure is visually clear and well-organized.

5 — Style guide is visually clear, well-organized, and easy to navigate.
4 — Style guide is organized but lacks some clarity.
3 — Style guide is partially organized but confusing in places.
2 — Style guide is mostly disorganized or unclear.
1 — Style guide is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "style_guide_completeness": <1-5>,
  "platform_usage_accuracy": <1-5>,
  "asset_quality_and_specificity": <1-5>,
  "organization_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "style_guide_completeness": "<one sentence citing specific evidence>",
    "platform_usage_accuracy": "<one sentence citing specific evidence>",
    "asset_quality_and_specificity": "<one sentence citing specific evidence>",
    "organization_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "style_guide_completeness": 0.35,
    "platform_usage_accuracy": 0.30,
    "asset_quality_and_specificity": 0.20,
    "organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())