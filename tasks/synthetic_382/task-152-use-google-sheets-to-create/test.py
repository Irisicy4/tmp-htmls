"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Create an automated issue tracking template in Google Sheets with sample data.
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


TASK_INSTRUCTION = """Use Google Sheets to create an automated issue tracking template for a software project. Include columns for issue ID, priority (with color-coded dropdown), status, description, assigned team member, and due date. Add sample data for three issues."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create an automated issue tracking template in Google Sheets for a software project. The template must include specific columns (issue ID, priority with color-coded dropdown, status, description, assigned team member, and due date) and sample data for three issues. The domain is software engineering, and successful completion requires both correct structure and sample data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create an automated issue tracking template for a software project. Include columns for issue ID, priority (with color-coded dropdown), status, description, assigned team member, and due date. Add sample data for three issues.

## Task-Specific Constraints
- Must include all specified columns in the template.
- Priority column must have a color-coded dropdown.
- Must include sample data for three issues.
- Must use Google Sheets to create the template.
- Output must be structured as a table.
- Must demonstrate correct formatting and automation features.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create the template?
- Are all required columns (issue ID, priority, status, description, assigned team member, due date) present?
- Does the priority column include a color-coded dropdown?
- Are there three sample issues with realistic data?
- Is the output structured as a table in Google Sheets?

### Step 2: Dimension Scoring

#### A. Template Completeness (0.35)
Measures whether the issue tracking template includes all required columns and features.

5 — All columns and features (e.g., color-coded dropdown for priority) are present and correct.
4 — All columns are present, but one feature (e.g., dropdown) is missing or incorrect.
3 — Most columns are present, but multiple features are missing or incorrect.
2 — Few columns are present, and features are mostly missing.
1 — No columns or features are present.

#### B. Sample Data Accuracy (0.30)
Measures whether the sample data for three issues is realistic and correctly formatted.

5 — Three issues with realistic and correctly formatted data are included.
4 — Three issues are included, but one or more have minor formatting or realism issues.
3 — Two issues are included, or three issues with significant formatting problems.
2 — One issue is included, or multiple issues with major formatting problems.
1 — No sample data is included.

#### C. Platform Usage (0.20)
Measures whether the agent correctly used Google Sheets to create the template.

5 — Google Sheets was used, and the tool trace shows correct usage of relevant features.
4 — Google Sheets was used, but the tool trace shows minor errors or inefficiencies.
3 — Google Sheets was used, but with significant errors or missing features.
2 — Google Sheets was accessed, but the template was not created correctly.
1 — Google Sheets was not used.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and formatted as a table.

5 — Output is structured as a clear, well-formatted table in Google Sheets.
4 — Output is structured as a table but has minor formatting issues.
3 — Output is structured as a table but with significant formatting problems.
2 — Output is poorly structured or not clearly a table.
1 — Output is not structured as a table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent navigated to Google Sheets and created a template with all required columns. The priority column includes a color-coded dropdown, and three sample issues with realistic data are present. The output is structured as a table.",
  "template_completeness": 5,
  "sample_data_accuracy": 5,
  "platform_usage": 5,
  "output_structure": 5,
  "dimension_reasoning": {
    "template_completeness": "All required columns and features are present in the template.",
    "sample_data_accuracy": "Three sample issues with realistic data are included.",
    "platform_usage": "Google Sheets was used correctly to create the template.",
    "output_structure": "The output is structured as a clear table in Google Sheets."
  },
  "overall_score": 5.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_completeness": 0.35,
    "sample_data_accuracy": 0.30,
    "platform_usage": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())