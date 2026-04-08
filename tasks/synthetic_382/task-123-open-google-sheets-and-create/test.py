"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Create a weekly meal plan template for a family of four using Google Sheets and dietary guidelines from EatRight.org.
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


TASK_INSTRUCTION = """Open Google Sheets and create a weekly meal plan template for a family of four. Include breakfast, lunch, dinner, and snacks for Monday through Sunday, with spaces for ingredient lists in each section. Use live dietary guidelines from EatRight.org to ensure balanced nutrition."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a weekly meal plan template for a family of four using Google Sheets. The template must include sections for breakfast, lunch, dinner, and snacks for each day of the week, along with spaces for ingredient lists. The agent must use live dietary guidelines from EatRight.org to ensure balanced nutrition.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and create a weekly meal plan template for a family of four. Include breakfast, lunch, dinner, and snacks for Monday through Sunday, with spaces for ingredient lists in each section. Use live dietary guidelines from EatRight.org to ensure balanced nutrition.

## Task-Specific Constraints
- Must use Google Sheets to create the template.
- Must visit EatRight.org and incorporate dietary guidelines into the meal plan.
- The template must include sections for breakfast, lunch, dinner, and snacks for each day of the week.
- Spaces for ingredient lists must be included in each section.
- The meal plan must reflect balanced nutrition based on dietary guidelines.
- The output must be organized and visually clear.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets to create the template?
- Did the agent visit EatRight.org to incorporate dietary guidelines?
- Does the template include sections for breakfast, lunch, dinner, and snacks for each day of the week?
- Are spaces for ingredient lists included in the template?
- Does the meal plan reflect balanced nutrition based on dietary guidelines?

### Step 2: Dimension Scoring

#### A. Template Completeness (0.35)
Measures whether the meal plan template includes all required sections and spaces.

5 — Includes all required sections (breakfast, lunch, dinner, snacks) and spaces for ingredient lists for all 7 days.
4 — Includes most required sections and spaces, but minor omissions.
3 — Includes some required sections and spaces, but significant omissions.
2 — Includes few required sections or spaces, mostly incomplete.
1 — Does not include any required sections or spaces.

#### B. Dietary Guideline Integration (0.30)
Measures whether the meal plan reflects balanced nutrition based on guidelines from EatRight.org.

5 — Fully integrates dietary guidelines into all meals and snacks.
4 — Mostly integrates dietary guidelines, with minor gaps.
3 — Partially integrates dietary guidelines, with noticeable gaps.
2 — Barely integrates dietary guidelines, mostly incomplete.
1 — Does not integrate dietary guidelines at all.

#### C. Platform Usage Accuracy (0.20)
Measures whether the agent correctly used Google Sheets and EatRight.org.

5 — Successfully used both platforms and incorporated relevant data.
4 — Used both platforms but with minor errors or omissions.
3 — Used one platform correctly but missed key elements from the other.
2 — Attempted to use platforms but with major errors.
1 — Did not use the required platforms.

#### D. Output Organization (0.15)
Measures whether the template is visually clear and well-organized.

5 — Template is visually clear, well-structured, and easy to read.
4 — Template is mostly clear but has minor formatting issues.
3 — Template is usable but has noticeable formatting issues.
2 — Template is poorly organized and difficult to read.
1 — Template is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_completeness": <1-5>,
  "dietary_guideline_integration": <1-5>,
  "platform_usage_accuracy": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "template_completeness": "<one sentence citing specific evidence>",
    "dietary_guideline_integration": "<one sentence citing specific evidence>",
    "platform_usage_accuracy": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_completeness": 0.35,
    "dietary_guideline_integration": 0.30,
    "platform_usage_accuracy": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())