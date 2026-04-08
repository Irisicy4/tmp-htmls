"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Create a weekly meal planner for a family of four using recipes sourced from Food Network, AllRecipes, and Epicurious, ensuring each meal takes under 30 minutes to prepare.
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


TASK_INSTRUCTION = """Use Google Sheets to create a weekly meal planner for a family of four. Include recipes sourced from Food Network, AllRecipes, and Epicurious for breakfast, lunch, and dinner. Ensure each meal is under 30 minutes to prepare."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a weekly meal planner for a family of four using Google Sheets. Recipes must be sourced from Food Network, AllRecipes, and Epicurious, and each meal must take under 30 minutes to prepare. A successful completion involves a well-structured planner with recipes for breakfast, lunch, and dinner for all seven days.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a weekly meal planner for a family of four. Include recipes sourced from Food Network, AllRecipes, and Epicurious for breakfast, lunch, and dinner. Ensure each meal is under 30 minutes to prepare.

## Task-Specific Constraints
- Must visit Food Network, AllRecipes, and Epicurious to source recipes.
- Must include recipes for breakfast, lunch, and dinner for all seven days.
- Each recipe must explicitly state the preparation time (under 30 minutes).
- The output must be organized as a structured table in Google Sheets.
- Recipes must be clearly labeled with their source platform.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Food Network, AllRecipes, and Epicurious? Which platforms were actually visited?
- Does the response include recipes for all seven days and three meals per day?
- Are preparation times explicitly stated for each recipe, and are they all under 30 minutes?
- Is the output organized as a structured table in Google Sheets?
- Are recipes clearly labeled with their source platform?

### Step 2: Dimension Scoring

#### A. Planner Completeness (0.35)
Measures whether the weekly meal planner includes recipes for breakfast, lunch, and dinner for all seven days.

5 — Includes recipes for all seven days and three meals per day.
4 — Includes recipes for at least six days and three meals per day.
3 — Includes recipes for at least five days and two meals per day.
2 — Includes recipes for fewer than five days or fewer than two meals per day.
1 — No meaningful planner created.

#### B. Platform Coverage (0.30)
Measures whether recipes were sourced from all three required platforms: Food Network, AllRecipes, and Epicurious.

5 — Recipes sourced from all three platforms.
4 — Recipes sourced from two platforms.
3 — Recipes sourced from one platform.
2 — Recipes sourced from none of the specified platforms.
1 — No evidence of platform usage.

#### C. Preparation Time Accuracy (0.25)
Measures whether all recipes explicitly state preparation times and whether they are under 30 minutes.

5 — All recipes include preparation times and are under 30 minutes.
4 — Most recipes include preparation times and are under 30 minutes.
3 — Some recipes include preparation times, but not all are under 30 minutes.
2 — Few recipes include preparation times, and many exceed 30 minutes.
1 — No preparation times provided.

#### D. Output Organization (0.10)
Measures whether the planner is well-structured and organized in Google Sheets.

5 — Planner is fully organized as a structured table in Google Sheets.
4 — Planner is mostly organized but has minor formatting issues.
3 — Planner is partially organized but lacks structure.
2 — Planner is poorly organized and difficult to interpret.
1 — No structured planner created.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent visited Food Network, AllRecipes, and Epicurious and sourced recipes for most meals. Preparation times were provided for most recipes and were under 30 minutes. The planner was organized in Google Sheets but had minor formatting issues.",
  "planner_completeness": 4,
  "platform_coverage": 5,
  "preparation_time_accuracy": 4,
  "output_organization": 4,
  "dimension_reasoning": {
    "planner_completeness": "Recipes were provided for six days with three meals per day.",
    "platform_coverage": "Recipes were sourced from all three required platforms.",
    "preparation_time_accuracy": "Most recipes included preparation times under 30 minutes.",
    "output_organization": "The planner was organized in Google Sheets but had minor formatting issues."
  },
  "overall_score": 4.25,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "planner_completeness": 0.35,
    "platform_coverage": 0.30,
    "preparation_time_accuracy": 0.25,
    "output_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())