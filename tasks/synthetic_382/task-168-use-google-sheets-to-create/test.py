"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Create a sprint planning tracker in Google Sheets with example tasks sourced from GitHub issues.
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


TASK_INSTRUCTION = """Use Google Sheets to create a sprint planning tracker for a software engineering team. Include three columns: task name, status (Not Started/In Progress/Completed), and assigned developer. Populate the tracker with five example tasks based on open-source project issues from GitHub repositories."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a sprint planning tracker in Google Sheets for a software engineering team. The tracker must include three specific columns (task name, status, assigned developer) and be populated with five example tasks sourced from open-source GitHub project issues. Successful completion involves correct structure, accurate sourcing, and proper organization of the tracker.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a sprint planning tracker for a software engineering team. Include three columns: task name, status (Not Started/In Progress/Completed), and assigned developer. Populate the tracker with five example tasks based on open-source project issues from GitHub repositories.

## Task-Specific Constraints
- Must create a Google Sheet with the specified column structure.
- Must populate the tracker with five tasks sourced from open-source GitHub repositories.
- Tasks must include realistic names and statuses (Not Started/In Progress/Completed).
- Must assign each task to a developer (can be fictional or real).
- Output must be organized and formatted as a table in the Google Sheet.
- Must demonstrate evidence of visiting GitHub to source tasks.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create a tracker with the specified columns?
- Did the agent visit GitHub repositories to source five example tasks?
- Are the task names, statuses, and assigned developers present and realistic?
- Is the tracker organized as a table in the Google Sheet?
- Are there any missing or incorrect elements in the tracker?

### Step 2: Dimension Scoring

#### A. Tracker Structure Accuracy (0.35)
Measures whether the Google Sheet tracker has the correct column structure and organization.

5 — Tracker has exactly three columns (task name, status, assigned developer) and is well-organized.
4 — Tracker has the correct columns but minor formatting issues.
3 — Tracker is partially complete (e.g., missing one column or poorly structured).
2 — Tracker is mostly incomplete or disorganized.
1 — Tracker is absent or completely incorrect.

#### B. Task Sourcing Completeness (0.30)
Measures whether the agent sourced five example tasks from GitHub repositories.

5 — All five tasks are sourced from GitHub, with realistic names and statuses.
4 — Four tasks are sourced correctly, or five tasks with minor inaccuracies.
3 — Three tasks are sourced correctly, or five tasks with significant inaccuracies.
2 — Two tasks are sourced correctly, or fewer than five tasks present.
1 — No tasks are sourced correctly.

#### C. Detail and Realism of Tasks (0.20)
Measures the realism and specificity of task names, statuses, and assigned developers.

5 — All tasks have realistic names, appropriate statuses, and assigned developers.
4 — Most tasks are realistic, with minor issues in names, statuses, or developers.
3 — Tasks are partially realistic, with noticeable issues in details.
2 — Tasks are mostly unrealistic or generic.
1 — Tasks are completely unrealistic or absent.

#### D. Evidence of Platform Usage (0.15)
Measures whether the agent demonstrated evidence of using Google Sheets and GitHub.

5 — Clear evidence of using both platforms (e.g., tool-call trace shows navigation).
4 — Evidence of using both platforms, but minor gaps in trace details.
3 — Partial evidence of platform usage (e.g., one platform clearly used).
2 — Minimal evidence of platform usage.
1 — No evidence of platform usage.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tracker_structure_accuracy": <1-5>,
  "task_sourcing_completeness": <1-5>,
  "detail_and_realism_of_tasks": <1-5>,
  "evidence_of_platform_usage": <1-5>,
  "dimension_reasoning": {{
    "tracker_structure_accuracy": "<one sentence citing specific evidence>",
    "task_sourcing_completeness": "<one sentence citing specific evidence>",
    "detail_and_realism_of_tasks": "<one sentence citing specific evidence>",
    "evidence_of_platform_usage": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tracker_structure_accuracy": 0.35,
    "task_sourcing_completeness": 0.30,
    "detail_and_realism_of_tasks": 0.20,
    "evidence_of_platform_usage": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())