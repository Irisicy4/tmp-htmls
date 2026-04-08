"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Design a bug-tracking template in Google Sheets and populate it with example entries.
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


TASK_INSTRUCTION = """Use Google Sheets to design a bug-tracking template for a software project. Include columns for Bug ID, Severity, Priority, Date Reported, Assigned Developer, and Status. Populate the sheet with 5 example entries based on common bug scenarios you find online."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to design a bug-tracking template in Google Sheets with specific columns and populate it with 5 example entries based on common bug scenarios found online. The domain is software engineering, and successful completion requires both the correct structure of the template and meaningful example data entries.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to design a bug-tracking template for a software project. Include columns for Bug ID, Severity, Priority, Date Reported, Assigned Developer, and Status. Populate the sheet with 5 example entries based on common bug scenarios you find online.

## Task-Specific Constraints
- Must include all specified columns in the Google Sheets template.
- Must populate the sheet with exactly 5 example entries.
- Example entries must be based on realistic and common bug scenarios found online.
- Must visit at least one of the specified platforms (github.com/angular/angular/issues or developer.mozilla.org) to gather bug-related data.
- Output must include a clear description of the template and example entries.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to at least one of the required platforms (github.com/angular/angular/issues or developer.mozilla.org)?
- Does the Google Sheets template include all required columns (Bug ID, Severity, Priority, Date Reported, Assigned Developer, and Status)?
- Are there exactly 5 example entries in the sheet?
- Are the example entries realistic and based on common bug scenarios found online?
- Is the output organized and clearly described?

### Step 2: Dimension Scoring

#### A. Template Structure Accuracy (0.35)
Measures whether the Google Sheets template includes all required columns.

5 — All required columns are present and correctly labeled.
4 — One required column is missing or mislabeled.
3 — Two required columns are missing or mislabeled.
2 — More than two required columns are missing or mislabeled.
1 — Template structure is absent or completely wrong.

#### B. Example Entry Realism (0.30)
Measures whether the example entries are realistic and based on common bug scenarios.

5 — All 5 entries are realistic and clearly based on common bug scenarios.
4 — 4 entries are realistic and clearly based on common bug scenarios.
3 — At least 3 entries are realistic but may lack clear sourcing.
2 — Fewer than 3 entries are realistic or meaningful.
1 — Example entries are absent or completely unrealistic.

#### C. Platform Usage Coverage (0.20)
Measures whether the agent visited at least one of the specified platforms to gather bug-related data.

5 — Agent visited both specified platforms and used them effectively.
4 — Agent visited one specified platform and used it effectively.
3 — Agent visited one specified platform but usage was unclear or minimal.
2 — Agent did not visit any specified platform but attempted to gather data elsewhere.
1 — Agent did not visit any platform and no data was gathered.

#### D. Output Organization and Clarity (0.15)
Measures whether the output is well-organized and clearly described.

5 — Output is well-organized, clearly described, and easy to follow.
4 — Output is mostly well-organized but lacks minor clarity.
3 — Output is partially organized but lacks significant clarity.
2 — Output is poorly organized and difficult to follow.
1 — Output is completely disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_structure_accuracy": <1-5>,
  "example_entry_realism": <1-5>,
  "platform_usage_coverage": <1-5>,
  "output_organization_and_clarity": <1-5>,
  "dimension_reasoning": {
    "template_structure_accuracy": "<one sentence citing specific evidence>",
    "example_entry_realism": "<one sentence citing specific evidence>",
    "platform_usage_coverage": "<one sentence citing specific evidence>",
    "output_organization_and_clarity": "<one sentence citing specific evidence>"
  },
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_structure_accuracy": 0.35,
    "example_entry_realism": 0.30,
    "platform_usage_coverage": 0.20,
    "output_organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())