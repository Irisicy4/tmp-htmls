"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Research three color palette generators and recommend a palette for a travel app targeting young adults, based on the theme 'Adventure'.
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


TASK_INSTRUCTION = """Research three color palette generators and create a recommended palette for designing a travel app targeting young adults. Use a tool to generate palettes based on the theme 'Adventure'. Recommend the palette with HEX codes and explain why this palette is optimal."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research three different color palette generators, generate palettes based on the theme 'Adventure', and recommend one palette for a travel app targeting young adults. The deliverable must include HEX codes for the recommended palette and a justification for why the palette is optimal for the target audience.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research three color palette generators and create a recommended palette for designing a travel app targeting young adults. Use a tool to generate palettes based on the theme 'Adventure'. Recommend the palette with HEX codes and explain why this palette is optimal.

## Task-Specific Constraints
- Must visit at least three color palette generator platforms (e.g., coolors.co, mycolor.space, canva.com).
- Must generate palettes based on the theme 'Adventure'.
- Must recommend one palette with at least 4 HEX codes.
- Must provide a justification for why the recommended palette is optimal for a travel app targeting young adults.
- The response must clearly identify the platforms used and the generated palettes.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least three color palette generator platforms? Which ones were actually visited?
- Did the agent generate palettes based on the theme 'Adventure'?
- Does the response include a recommended palette with at least 4 HEX codes?
- Is the justification for the recommended palette clear and relevant to the target audience (young adults)?
- Is the output structured and easy to follow?

### Step 2: Dimension Scoring

#### A. Palette Recommendation Accuracy (0.35)
Measures whether the recommended palette is complete, relevant, and based on the theme 'Adventure'.

5 — Recommends a palette with 4+ HEX codes, clearly based on the theme 'Adventure', and highly relevant to young adults.
4 — Recommends a palette with 4+ HEX codes, based on the theme, but relevance to young adults is less clear.
3 — Recommends a palette with 3+ HEX codes, partially based on the theme, with limited relevance to young adults.
2 — Recommends a palette with fewer than 3 HEX codes or unrelated to the theme.
1 — No palette recommendation or completely irrelevant.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and utilized at least three color palette generator platforms.

5 — Clearly identifies and uses three or more platforms to generate palettes.
4 — Identifies and uses three platforms but provides limited detail on their use.
3 — Identifies and uses two platforms or provides incomplete evidence of three.
2 — Identifies and uses only one platform.
1 — Does not identify or use any platforms.

#### C. Justification Depth (0.20)
Measures the depth and relevance of the justification for the recommended palette.

5 — Provides a detailed, compelling justification tailored to the target audience (young adults).
4 — Provides a clear justification but lacks some detail or specificity.
3 — Provides a basic justification with limited relevance to the target audience.
2 — Provides a vague or generic justification.
1 — No justification provided.

#### D. Output Structure and Clarity (0.15)
Measures how well-organized and clear the response is.

5 — The response is well-structured, easy to follow, and free of errors.
4 — The response is mostly clear and organized, with minor issues.
3 — The response is somewhat clear but has noticeable structural or clarity issues.
2 — The response is poorly organized or difficult to follow.
1 — The response is completely disorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "palette_recommendation_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "justification_depth": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "palette_recommendation_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "justification_depth": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "palette_recommendation_accuracy": 0.35,
    "platform_coverage": 0.30,
    "justification_depth": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())