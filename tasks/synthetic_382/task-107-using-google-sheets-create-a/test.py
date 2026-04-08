"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Create a weekly meal plan template in Google Sheets, including sections for meals and snacks, a shopping list, and suggested Monday meals sourced from AllRecipes and BBC Good Food.
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


TASK_INSTRUCTION = """Using Google Sheets, create a weekly meal plan template. Include sections for breakfast, lunch, dinner, and snacks for each day, along with a shopping list for groceries. Add suggested meals for Monday using recipes sourced from AllRecipes and BBC Good Food."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a weekly meal plan template in Google Sheets. The template must include sections for breakfast, lunch, dinner, and snacks for each day of the week, along with a shopping list for groceries. Additionally, the agent must add suggested meals for Monday using recipes sourced from AllRecipes and BBC Good Food.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, create a weekly meal plan template. Include sections for breakfast, lunch, dinner, and snacks for each day, along with a shopping list for groceries. Add suggested meals for Monday using recipes sourced from AllRecipes and BBC Good Food.

## Task-Specific Constraints
- Must create a structured table in Google Sheets with sections for breakfast, lunch, dinner, and snacks for all seven days.
- Must include a shopping list for groceries based on the meal plan.
- Must source Monday's suggested meals from both AllRecipes and BBC Good Food.
- Must provide recipe names and URLs for Monday's meals.
- Must ensure the shopping list is organized and matches the meals planned.
- Must visit both AllRecipes and BBC Good Food during execution.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both AllRecipes and BBC Good Food to source Monday's meals?
- Is the weekly meal plan template structured correctly with sections for breakfast, lunch, dinner, and snacks for all seven days?
- Does the shopping list match the meals planned and is it organized?
- Are recipe names and URLs for Monday's meals present and sourced correctly?
- Is the output presented in Google Sheets as required?

### Step 2: Dimension Scoring

#### A. Template Structure Accuracy (0.35)
Measures whether the weekly meal plan template is correctly structured in Google Sheets.

5 — Includes all required sections (breakfast, lunch, dinner, snacks) for all seven days in a clear table format.
4 — Includes most required sections but may lack minor formatting or completeness.
3 — Includes some required sections but is incomplete or poorly formatted.
2 — Includes few required sections and lacks clarity or structure.
1 — No attempt or completely incorrect structure.

#### B. Source Utilization (0.30)
Measures whether the agent navigated to AllRecipes and BBC Good Food and sourced recipes correctly.

5 — Recipes for Monday are sourced from both platforms with names and URLs provided.
4 — Recipes are sourced from both platforms but lack minor details (e.g., one URL missing).
3 — Recipes are sourced from only one platform or are incomplete.
2 — Recipes are mostly missing or sourced incorrectly.
1 — No attempt or completely incorrect sourcing.

#### C. Shopping List Completeness (0.25)
Measures whether the shopping list is complete and matches the meals planned.

5 — Shopping list includes all items required for the meal plan and is organized.
4 — Shopping list includes most items but may lack minor details or organization.
3 — Shopping list includes some items but is incomplete or disorganized.
2 — Shopping list includes few items and is poorly matched to the meal plan.
1 — No attempt or completely incorrect shopping list.

#### D. Formatting and Presentation (0.10)
Measures the overall presentation and organization of the output.

5 — Output is well-organized, clearly presented, and easy to understand.
4 — Output is mostly well-organized but may lack minor clarity.
3 — Output is partially organized but has noticeable issues.
2 — Output is poorly organized and difficult to understand.
1 — No attempt or completely disorganized output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_structure_accuracy": <1-5>,
  "source_utilization": <1-5>,
  "shopping_list_completeness": <1-5>,
  "formatting_and_presentation": <1-5>,
  "dimension_reasoning": {{
    "template_structure_accuracy": "<one sentence citing specific evidence>",
    "source_utilization": "<one sentence citing specific evidence>",
    "shopping_list_completeness": "<one sentence citing specific evidence>",
    "formatting_and_presentation": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_structure_accuracy": 0.35,
    "source_utilization": 0.30,
    "shopping_list_completeness": 0.25,
    "formatting_and_presentation": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())