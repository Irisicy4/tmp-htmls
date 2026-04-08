"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Research and compare the top 3 free font libraries for UI design, evaluating collection size, ease of navigation, and licensing restrictions, and summarize findings in a table.
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


TASK_INSTRUCTION = """Research and compare the top 3 free font libraries for UI design. Evaluate them based on their collection size, ease of navigation, and licensing restrictions for commercial use. Provide a table summarizing your findings."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to research and compare the top 3 free font libraries for UI design. The agent must evaluate these libraries based on their collection size, ease of navigation, and licensing restrictions for commercial use. The findings must be presented in a table format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare the top 3 free font libraries for UI design. Evaluate them based on their collection size, ease of navigation, and licensing restrictions for commercial use. Provide a table summarizing your findings.

## Task-Specific Constraints
- Must visit and extract data from fonts.google.com, dafont.com, and fontlibrary.org.
- Must evaluate collection size, ease of navigation, and licensing restrictions for all three platforms.
- Output must be organized as a table with clear columns for each evaluation criterion.
- Must provide specific numbers or examples for collection size and licensing terms.
- Must ensure the table format is clear and readable.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to fonts.google.com, dafont.com, and fontlibrary.org? Which ones were actually visited?
- Does the response include evaluations for collection size, ease of navigation, and licensing restrictions for all three platforms?
- Is the output organized as a table with clear columns for each evaluation criterion?
- Are specific numbers or examples provided for collection size and licensing terms?
- Are there any factual inaccuracies or unsupported claims in the response?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (table summarizing findings) is accurate, complete, and aligned with the task requirements.

5 — Table includes all three platforms, all three evaluation criteria, and accurate data for each.
4 — Table includes all three platforms and criteria but has minor inaccuracies or omissions.
3 — Table includes at least two platforms and criteria but is incomplete or partially inaccurate.
2 — Table is mostly incomplete or inaccurate.
1 — No table or completely incorrect output.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited and extracted data from all required platforms.

5 — Data is clearly sourced from all three platforms (fonts.google.com, dafont.com, fontlibrary.org).
4 — Data is sourced from at least two platforms with minor omissions.
3 — Data is sourced from at least two platforms but lacks significant details.
2 — Data is sourced from only one platform.
1 — No data is sourced from the required platforms.

#### C. Depth of Analysis (0.20)
Measures the level of detail and specificity in the evaluations (e.g., specific numbers, examples, or licensing terms).

5 — Includes specific numbers/examples for all criteria across all platforms.
4 — Includes specific numbers/examples for most criteria across most platforms.
3 — Includes some specific numbers/examples but is incomplete.
2 — Includes vague or generic evaluations with minimal detail.
1 — No specific details or examples provided.

#### D. Output Structure and Clarity (0.15)
Measures the clarity and organization of the output, including table formatting and readability.

5 — Table is well-organized, clear, and easy to read with labeled columns and rows.
4 — Table is mostly clear but has minor formatting issues.
3 — Table is present but poorly organized or hard to read.
2 — Table is difficult to interpret or lacks clear structure.
1 — No table or completely unstructured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())