"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Create a weekly meal plan template in Google Sheets with recipes sourced from three different websites.
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


TASK_INSTRUCTION = """Use Google Sheets to create a weekly meal plan template with breakfast, lunch, and dinner slots for each day. Populate the plan with recipes sourced from three different recipe websites. Include links to each recipe in the sheet."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a structured weekly meal plan template in Google Sheets, including slots for breakfast, lunch, and dinner for each day. The plan must be populated with recipes sourced from three different recipe websites, and each recipe must include a clickable link.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a weekly meal plan template with breakfast, lunch, and dinner slots for each day. Populate the plan with recipes sourced from three different recipe websites. Include links to each recipe in the sheet.

## Task-Specific Constraints
- Must visit at least three recipe websites (e.g., allrecipes.com, tasty.co, etc.).
- Must create a structured table in Google Sheets with slots for breakfast, lunch, and dinner for all seven days.
- Each meal slot must be populated with a recipe name and a clickable link to the recipe.
- Recipes must be sourced from three distinct websites.
- The sheet must be accessible and correctly formatted as a meal plan template.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least three recipe websites? Which ones were actually visited?
- Is the Google Sheets template structured with slots for breakfast, lunch, and dinner for all seven days?
- Are recipe names and clickable links present for each meal slot?
- Are recipes sourced from three distinct websites?
- Is the sheet correctly formatted and accessible?

### Step 2: Dimension Scoring

#### A. Template Structure Accuracy (0.35)
Measures whether the Google Sheets template is correctly structured with slots for breakfast, lunch, and dinner for all seven days.

5 — Template includes all required slots for seven days and is correctly formatted.
4 — Template includes most slots but has minor formatting issues.
3 — Template includes some slots but is incomplete or poorly formatted.
2 — Template is mostly missing or incorrectly formatted.
1 — Template is absent or completely wrong.

#### B. Source Coverage (0.30)
Measures whether recipes are sourced from three distinct websites as required.

5 — Recipes are sourced from three distinct websites and clearly identified.
4 — Recipes are sourced from two distinct websites.
3 — Recipes are sourced from one website or unclear sourcing.
2 — Recipes are mostly missing or incorrectly sourced.
1 — No recipes or sources identified.

#### C. Recipe Link Accuracy (0.20)
Measures whether each meal slot includes a clickable link to the recipe.

5 — All meal slots include clickable links to recipes.
4 — Most meal slots include clickable links to recipes.
3 — Some meal slots include clickable links to recipes.
2 — Few meal slots include clickable links to recipes.
1 — No clickable links are present.

#### D. Formatting and Accessibility (0.15)
Measures whether the sheet is well-formatted and accessible.

5 — Sheet is well-formatted and accessible without issues.
4 — Sheet is mostly well-formatted but has minor accessibility issues.
3 — Sheet is partially formatted or has some accessibility issues.
2 — Sheet is poorly formatted or inaccessible.
1 — Sheet is completely inaccessible or unformatted.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_structure_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "recipe_link_accuracy": <1-5>,
  "formatting_and_accessibility": <1-5>,
  "dimension_reasoning": {{
    "template_structure_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "recipe_link_accuracy": "<one sentence citing specific evidence>",
    "formatting_and_accessibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_structure_accuracy": 0.35,
    "source_coverage": 0.30,
    "recipe_link_accuracy": 0.20,
    "formatting_and_accessibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())