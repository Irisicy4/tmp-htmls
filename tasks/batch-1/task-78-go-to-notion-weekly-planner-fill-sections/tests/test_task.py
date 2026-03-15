"""
LLM-as-judge evaluator for EvolveBench task-78.

Category: Daily Activities
Task: Please fill in all the missing data on this page to make it complete. Go to https://www.notion.so/templates/weekly-plann
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


TASK_INSTRUCTION = """Please fill in all the missing data on this page to make it complete. Go to https://www.notion.so/templates/weekly-planner and fill in all empty sections of the weekly planner template."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating to a Notion weekly planner template and filling in all empty/placeholder sections with reasonable sample data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: Notion weekly planner template at the provided URL
- Action: fill ALL empty sections — not just some
- Content: reasonable, realistic sample data (tasks, priorities, goals, notes)
- Completion: agent must confirm all sections are filled

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Notion weekly planner?
- What sections were identified as empty?
- Were all sections filled with content?
- Is the filled content reasonable (realistic tasks/goals)?
- Did the agent confirm completion?

### Step 2: Dimension Scoring

#### A. Platform Access (0.2)
Did the agent access the Notion weekly planner?

5 — Agent navigated to the Notion template URL and loaded the weekly planner.
4 — Agent reached Notion but had difficulty with the template.
3 — Agent described the Notion template without navigating to it.
2 — Agent attempted Notion but could not load the template.
1 — No Notion access.

#### B. Section Coverage (0.35)
Were all empty sections identified and filled?

5 — All empty sections identified and filled: days of week, priorities, goals, notes, etc.
4 — Most sections filled; 1-2 missed.
3 — About half the sections filled.
2 — Only a few sections filled.
1 — No sections filled.

#### C. Content Quality (0.3)
Is the filled content realistic and useful?

5 — Realistic tasks, goals, and notes that a real person might have; varied and context-appropriate.
4 — Good content but slightly generic.
3 — Content is plausible but very generic (e.g. 'Task 1', 'Goal 1').
2 — Placeholder-style content.
1 — No meaningful content.

#### D. Completion Confirmation (0.15)
Did the agent confirm all sections are complete?

5 — Agent explicitly confirmed all sections filled and reviewed the result.
4 — Completion implied but not explicitly confirmed.
3 — Agent stated completion without verification.
2 — Agent stopped mid-way without confirmation.
1 — No confirmation.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_access": <1-5>,
  "section_coverage": <1-5>,
  "content_quality": <1-5>,
  "completion_confirmation": <1-5>,
  "dimension_reasoning": {{
    "platform_access": "<one sentence citing specific evidence>",
    "section_coverage": "<one sentence citing specific evidence>",
    "content_quality": "<one sentence citing specific evidence>",
    "completion_confirmation": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_access": 0.2,
    "section_coverage": 0.35,
    "content_quality": 0.3,
    "completion_confirmation": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())