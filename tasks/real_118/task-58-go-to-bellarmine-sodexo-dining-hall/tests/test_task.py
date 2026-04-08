"""
LLM-as-judge evaluator for EvolveBench task-58.

Category: Daily Activities
Task: Check today's cafeteria menu at Bellarmine University dining hall.
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

def _call(agent_response, execution_summary, system_prompt, user_prompt_template, task_instruction):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(
                    task_instruction=task_instruction,
                    agent_response=agent_response,
                    execution_summary=execution_summary or "Not available.",
                )}
            ],
            max_tokens=1024,
        )
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes, dimensions, weights, pass_threshold):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in dimensions)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in dimensions}
    overall = sum(aggregated[d] * weights[d] for d in dimensions)
    aggregated["overall_score"] = round(overall, 2); aggregated["passed"] = overall >= pass_threshold
    median_call = sorted(valid, key=lambda v: abs(v.get("overall_score", 0) - overall))[0]
    aggregated["evidence_summary"] = median_call.get("evidence_summary", "")
    aggregated["dimension_reasoning"] = median_call.get("dimension_reasoning", {})
    aggregated["_votes_used"] = len(valid)
    return aggregated


TASK_INSTRUCTION = """Go to https://bellarmine.sodexomyway.com/en-us/locations/university-dining-hall and tell me what is on the menu today. List all available food items for breakfast, lunch, and dinner."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Assess whether an AI agent successfully retrieved and reported today's cafeteria menu from the Bellarmine University dining hall website."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: bellarmine.sodexomyway.com specifically
- Meals: breakfast, lunch, and dinner should all be covered
- Currency: today's menu — not a generic or sample menu
- Completeness: all food items listed, not just categories

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Bellarmine dining hall website?
- Was today's date menu retrieved (not a sample or generic menu)?
- Are breakfast, lunch, and dinner all covered?
- Are specific food items listed (not just categories)?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.25)
Did the agent navigate to the correct website?

5 — Agent navigated to bellarmine.sodexomyway.com and accessed today's menu.
4 — Agent reached the site but had difficulty loading the menu.
3 — Agent found Bellarmine dining info from a different source.
2 — Agent described what the menu site looks like without navigating.
1 — No navigation attempted.

#### B. Menu Currency (0.3)
Is the menu for today specifically?

5 — Menu clearly identified as today's date with specific items.
4 — Menu retrieved but date not explicitly confirmed.
3 — Menu present but may be a sample or generic menu.
2 — Generic dining hall info without today's specific items.
1 — No menu retrieved.

#### C. Meal Coverage (0.25)
Are all three meal periods covered?

5 — Breakfast, lunch, and dinner all listed with specific items.
4 — Two meal periods covered.
3 — One meal period covered.
2 — Meal periods mentioned but no specific items.
1 — No meal coverage.

#### D. Item Specificity (0.2)
Are specific food items listed?

5 — Specific named dishes listed for each meal (e.g. 'scrambled eggs, bacon, oatmeal').
4 — Most items specific but some categories without detail.
3 — Mix of specific items and categories.
2 — Only food categories listed (e.g. 'breakfast bar, grill').
1 — No specific items.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_navigation": <1-5>,
  "menu_currency": <1-5>,
  "meal_coverage": <1-5>,
  "item_specificity": <1-5>,
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "menu_currency": "<one sentence citing specific evidence>",
    "meal_coverage": "<one sentence citing specific evidence>",
    "item_specificity": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.25,
    "menu_currency": 0.3,
    "meal_coverage": 0.25,
    "item_specificity": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.",
                "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        if abs(float(overall) - PASS_THRESHOLD) <= 0.5:
            scores = _vote([first,
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION),
                            _call(agent_response, execution_summary, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, TASK_INSTRUCTION)],
                           DIMENSIONS, DIMENSION_WEIGHTS, PASS_THRESHOLD)
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