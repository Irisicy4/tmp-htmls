"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Create a weekly meal plan for a family of four using recipes from Food Network, Allrecipes, and BBC Good Food.
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


TASK_INSTRUCTION = """Use Google Sheets to create a weekly meal plan for a family of four based on recipes from Food Network, Allrecipes, and BBC Good Food. Include breakfast, lunch, and dinner for each day, along with the name of each dish and the website URL where the recipe can be found."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a weekly meal plan for a family of four using recipes sourced from Food Network, Allrecipes, and BBC Good Food. The deliverable must include breakfast, lunch, and dinner for each day, with the name of each dish and the URL of the recipe.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a weekly meal plan for a family of four based on recipes from Food Network, Allrecipes, and BBC Good Food. Include breakfast, lunch, and dinner for each day, along with the name of each dish and the website URL where the recipe can be found.

## Task-Specific Constraints
- Must visit all three specified platforms: Food Network, Allrecipes, and BBC Good Food.
- Must include breakfast, lunch, and dinner for all seven days of the week.
- Each meal entry must include the name of the dish and the recipe URL.
- Output must be organized as a table in Google Sheets.
- Recipes must be appropriate for a family of four (e.g., serving size sufficient).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all three required platforms (Food Network, Allrecipes, BBC Good Food)?
- Does the response include breakfast, lunch, and dinner for all seven days?
- Are the dish names and recipe URLs included for each meal entry?
- Is the output organized as a table in Google Sheets?
- Are the recipes appropriate for a family of four in terms of serving size?

### Step 2: Dimension Scoring

#### A. Meal Plan Completeness (0.35)
Measures whether the weekly meal plan includes all required meals and details.

5 — Includes breakfast, lunch, and dinner for all seven days, with dish names and URLs for every meal.
4 — Includes most meals, with minor omissions (e.g., missing a few URLs or meal entries).
3 — Includes at least half of the meals, but several entries are incomplete or missing.
2 — Includes fewer than half of the meals, with significant omissions.
1 — No meaningful meal plan provided.

#### B. Platform Coverage (0.30)
Measures whether the agent used all three required platforms to source recipes.

5 — Recipes are sourced from Food Network, Allrecipes, and BBC Good Food, with evidence of navigation to all three.
4 — Recipes are sourced from at least two platforms, with minor omissions.
3 — Recipes are sourced from only one platform or evidence of navigation is unclear.
2 — Recipes are not sourced from any of the specified platforms.
1 — No platform usage is evident.

#### C. Recipe Appropriateness (0.20)
Measures whether the recipes are suitable for a family of four.

5 — All recipes are appropriate, with serving sizes clearly sufficient for four people.
4 — Most recipes are appropriate, with minor issues in serving sizes.
3 — Some recipes are appropriate, but others are clearly unsuitable for four people.
2 — Few recipes are appropriate, with significant serving size issues.
1 — Recipes are entirely inappropriate or missing.

#### D. Output Organization (0.15)
Measures the structure and presentation of the meal plan.

5 — The meal plan is organized as a clear, well-formatted table in Google Sheets.
4 — The meal plan is mostly organized, with minor formatting issues.
3 — The meal plan is partially organized, but lacks clarity or structure.
2 — The meal plan is poorly organized or difficult to interpret.
1 — No meaningful organization is evident.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited all three platforms and included recipes for breakfast, lunch, and dinner for seven days. Dish names and URLs are present, and the output is structured as a table in Google Sheets. Most recipes appear suitable for a family of four.",
  "meal_plan_completeness": 5,
  "platform_coverage": 5,
  "recipe_appropriateness": 4,
  "output_organization": 5,
  "dimension_reasoning": {
    "meal_plan_completeness": "All meals and required details are included for seven days.",
    "platform_coverage": "Recipes are sourced from Food Network, Allrecipes, and BBC Good Food.",
    "recipe_appropriateness": "Most recipes are suitable for a family of four, with minor serving size issues.",
    "output_organization": "The meal plan is well-organized as a table in Google Sheets."
  },
  "overall_score": 4.75,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "meal_plan_completeness": 0.35,
    "platform_coverage": 0.30,
    "recipe_appropriateness": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())