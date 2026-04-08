"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Set up a weekly task tracker in Notion to manage household chores.
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


TASK_INSTRUCTION = """Set up a weekly task tracker in Notion to manage household chores. Create four categories: Cleaning, Cooking, Shopping, and Maintenance. Add 5 example tasks under each category and configure a filter to show only incomplete tasks. Save and report the final structure of the task tracker."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to set up a weekly task tracker in Notion for household chores. The agent must create four categories (Cleaning, Cooking, Shopping, and Maintenance), add 5 example tasks under each category, and configure a filter to show only incomplete tasks. A successful completion includes a structured report of the final tracker setup.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a weekly task tracker in Notion to manage household chores. Create four categories: Cleaning, Cooking, Shopping, and Maintenance. Add 5 example tasks under each category and configure a filter to show only incomplete tasks. Save and report the final structure of the task tracker.

## Task-Specific Constraints
- Must create the task tracker in Notion.
- Must include exactly four categories: Cleaning, Cooking, Shopping, and Maintenance.
- Each category must have 5 example tasks.
- Must configure a filter to show only incomplete tasks.
- Final response must describe the structure of the tracker clearly and completely.
- Must save the tracker in Notion.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Notion and create a task tracker?
- Are the four required categories (Cleaning, Cooking, Shopping, Maintenance) present in the response?
- Does each category include 5 example tasks?
- Is there evidence that the filter for incomplete tasks was configured correctly?
- Is the final structure of the tracker described clearly and completely?

### Step 2: Dimension Scoring

#### A. Tracker Structure Accuracy (0.35)
Measures whether the task tracker structure matches the requirements.

5 — All four categories are present, each with 5 tasks, and the filter is correctly configured.
4 — All categories are present, but one or two tasks or the filter setup are slightly incomplete.
3 — At least three categories are present, but tasks or filter setup are incomplete.
2 — Only one or two categories are present, and tasks or filter setup are mostly missing.
1 — No categories or tasks are present, or the tracker was not created.

#### B. Coverage of Requirements (0.30)
Measures whether all required elements (categories, tasks, filter) are included.

5 — All required elements are included and described in detail.
4 — Most required elements are included, but one is slightly incomplete or missing.
3 — Some required elements are included, but others are missing or incomplete.
2 — Few required elements are included, and most are missing.
1 — No required elements are included.

#### C. Detail and Specificity (0.20)
Measures the level of detail in the response (e.g., task examples, filter configuration).

5 — Response includes detailed task examples and filter settings for all categories.
4 — Response includes task examples and filter settings, but lacks some detail.
3 — Response includes basic task examples but lacks filter details or specificity.
2 — Response includes minimal task examples and no filter details.
1 — Response lacks any meaningful detail.

#### D. Output Clarity and Organization (0.15)
Measures whether the response is clear, well-organized, and easy to understand.

5 — Response is clear, well-organized, and fully describes the tracker structure.
4 — Response is mostly clear and organized, with minor issues in clarity or structure.
3 — Response is partially clear but has noticeable issues in organization.
2 — Response is unclear or poorly organized.
1 — Response is completely unclear or disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tracker_structure_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_clarity_and_organization": <1-5>,
  "dimension_reasoning": {{
    "tracker_structure_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_clarity_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tracker_structure_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "detail_and_specificity": 0.20,
    "output_clarity_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())