"""
LLM-as-judge evaluator for EvolveBench task.

Category: Shopping
Task: Use the IKEA kitchen planner tool to create a basic kitchen layout for a 10x12 ft space with low-cost options.
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


TASK_INSTRUCTION = """Use the IKEA kitchen planner tool to set up a basic kitchen layout with specifications for a 10x12 ft space. Select cabinets, countertops, and appliances within the lowest price range and complete the workflow to generate a summary of the setup."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves using the IKEA kitchen planner tool to create a basic kitchen layout for a 10x12 ft space. The agent must select cabinets, countertops, and appliances within the lowest price range and complete the workflow to generate a summary of the setup. A successful completion includes a structured summary of the kitchen layout, including item details and pricing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the IKEA kitchen planner tool to set up a basic kitchen layout with specifications for a 10x12 ft space. Select cabinets, countertops, and appliances within the lowest price range and complete the workflow to generate a summary of the setup.

## Task-Specific Constraints
- Must use the IKEA kitchen planner tool to create the layout.
- Must specify dimensions of the kitchen as 10x12 ft.
- Must select cabinets, countertops, and appliances within the lowest price range.
- Must generate a summary of the setup, including item names, prices, and total cost.
- Output must be structured as a list or table.
- Must complete the workflow to the final summary stage.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use the IKEA kitchen planner tool to create the layout?
- Are the kitchen dimensions specified as 10x12 ft?
- Are cabinets, countertops, and appliances selected within the lowest price range?
- Does the response include a summary with item names, prices, and total cost?
- Is the output structured as a list or table?

### Step 2: Dimension Scoring

#### A. Layout Accuracy (0.35)
Measures whether the kitchen layout matches the specified dimensions and includes required components.

5 — Layout is 10x12 ft, includes cabinets, countertops, and appliances.
4 — Layout is correct but missing one component (e.g., appliances).
3 — Layout is partially correct but missing multiple components.
2 — Layout is incorrect or incomplete.
1 — No layout provided.

#### B. Cost Optimization (0.30)
Measures whether the selected items are within the lowest price range.

5 — All items are within the lowest price range.
4 — Most items are within the lowest price range, with minor deviations.
3 — Some items are within the lowest price range, but others are not.
2 — Few items are within the lowest price range.
1 — No consideration for price range.

#### C. Summary Completeness (0.20)
Measures whether the summary includes item names, prices, and total cost.

5 — Summary includes all required details (item names, prices, total cost).
4 — Summary includes most details but is missing minor elements.
3 — Summary is incomplete but usable.
2 — Summary is mostly missing or unclear.
1 — No summary provided.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and easy to understand.

5 — Output is structured as a clear list or table.
4 — Output is mostly structured but has minor formatting issues.
3 — Output is partially structured but lacks clarity.
2 — Output is poorly structured.
1 — No structure provided.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "layout_accuracy": <1-5>,
  "cost_optimization": <1-5>,
  "summary_completeness": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "layout_accuracy": "<one sentence citing specific evidence>",
    "cost_optimization": "<one sentence citing specific evidence>",
    "summary_completeness": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "layout_accuracy": 0.35,
    "cost_optimization": 0.30,
    "summary_completeness": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())