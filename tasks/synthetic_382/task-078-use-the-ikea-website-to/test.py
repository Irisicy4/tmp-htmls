"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Design and customize a modular shelving unit on ikea.com, meeting specific requirements.
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


TASK_INSTRUCTION = """Use the IKEA website to design and customize a modular shelving unit. Choose the white finish, specify dimensions to fit a 50-inch wide space, and add optional glass doors. Report the final price and configuration summary from the checkout page (do not proceed to checkout)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task requires the agent to use the IKEA website to design and customize a modular shelving unit. The agent must select a white finish, ensure the dimensions fit a 50-inch wide space, add optional glass doors, and provide the final price and configuration summary from the checkout page. Success depends on the agent's ability to meet all specified requirements and provide accurate and structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the IKEA website to design and customize a modular shelving unit. Choose the white finish, specify dimensions to fit a 50-inch wide space, and add optional glass doors. Report the final price and configuration summary from the checkout page (do not proceed to checkout).

## Task-Specific Constraints
- Must navigate the IKEA website and use the customization tool.
- Must select the white finish for the shelving unit.
- Must specify dimensions that fit a 50-inch wide space.
- Must include optional glass doors in the design.
- Must report the final price and configuration summary from the checkout page.
- Output must be structured as a clear and organized summary.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the IKEA website and use the customization tool?
- Did the agent select the white finish for the shelving unit?
- Did the agent specify dimensions that fit a 50-inch wide space?
- Did the agent include optional glass doors in the design?
- Is the final price and configuration summary accurately reported and structured?

### Step 2: Dimension Scoring

#### A. Design Accuracy (0.35)
Measures whether the shelving unit design meets all specified requirements.

5 — Design includes white finish, fits 50-inch space, and includes glass doors.
4 — Design meets most requirements but misses one minor detail.
3 — Design partially meets requirements but misses key elements.
2 — Design is mostly incorrect or incomplete.
1 — Design is absent or completely wrong.

#### B. Coverage of Requirements (0.30)
Measures whether all specified constraints are addressed.

5 — All constraints (finish, dimensions, glass doors, price) are addressed.
4 — Most constraints are addressed, with minor omissions.
3 — Some constraints are addressed, but key elements are missing.
2 — Few constraints are addressed, with major omissions.
1 — No constraints are addressed.

#### C. Price and Configuration Reporting (0.25)
Measures the accuracy and completeness of the price and configuration summary.

5 — Price and configuration summary is fully accurate and detailed.
4 — Summary is mostly accurate, with minor errors or omissions.
3 — Summary is partially accurate, with noticeable errors or missing details.
2 — Summary is mostly inaccurate or incomplete.
1 — Summary is absent or completely wrong.

#### D. Output Structure and Clarity (0.10)
Measures the organization and clarity of the final output.

5 — Output is well-organized, clear, and easy to understand.
4 — Output is mostly organized and clear, with minor issues.
3 — Output is somewhat organized but lacks clarity.
2 — Output is poorly organized and unclear.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "design_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "price_and_configuration_reporting": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "design_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "price_and_configuration_reporting": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "design_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "price_and_configuration_reporting": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())