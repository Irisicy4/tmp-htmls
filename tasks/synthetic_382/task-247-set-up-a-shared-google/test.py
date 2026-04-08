"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Create a shared Google Sheets template to track travel expenses for a group trip to Tokyo.
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


TASK_INSTRUCTION = """Set up a shared Google Sheets template to track travel expenses for a group trip to Tokyo. Include columns for participant names, date, expense category (meals, transport, lodging), and total cost, and pre-fill sample entries for demonstration."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a shared Google Sheets template to track travel expenses for a group trip to Tokyo. The sheet must include specific columns for participant names, date, expense category (meals, transport, lodging), and total cost. Additionally, the agent must pre-fill sample entries to demonstrate functionality.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a shared Google Sheets template to track travel expenses for a group trip to Tokyo. Include columns for participant names, date, expense category (meals, transport, lodging), and total cost, and pre-fill sample entries for demonstration.

## Task-Specific Constraints
- Must create a Google Sheets document using sheets.new.
- The sheet must include columns for participant names, date, expense category, and total cost.
- The sheet must include at least three sample entries for demonstration.
- The sheet must be shared with appropriate permissions (e.g., editable by others).
- The response must include a link to the Google Sheets document.
- The response must describe the structure and contents of the sheet.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent create a Google Sheets document using sheets.new?
- Does the sheet include all required columns (participant names, date, expense category, total cost)?
- Are there at least three sample entries pre-filled in the sheet?
- Is the sheet shared with appropriate permissions (e.g., editable by others)?
- Does the response include a valid link to the Google Sheets document?

### Step 2: Dimension Scoring

#### A. Template Completeness (0.35)
Measures whether the Google Sheets template includes all required columns and sample entries.

5 — All required columns are present, and at least three sample entries are pre-filled correctly.
4 — All required columns are present, but fewer than three sample entries are pre-filled.
3 — Most required columns are present, but some are missing or incomplete.
2 — Few required columns are present, and sample entries are missing.
1 — No required columns or sample entries are present.

#### B. Sharing and Accessibility (0.30)
Measures whether the sheet is shared with appropriate permissions and includes a valid link.

5 — The sheet is shared with appropriate permissions and includes a valid link.
4 — The sheet is shared but permissions are partially incorrect or unclear.
3 — The sheet is shared but lacks a valid link or permissions are incorrect.
2 — The sheet is not shared or permissions are completely incorrect.
1 — No sharing or link provided.

#### C. Sample Data Quality (0.20)
Measures the accuracy and relevance of the pre-filled sample entries.

5 — Sample entries are accurate, relevant, and demonstrate all required categories.
4 — Sample entries are mostly accurate but lack minor details or relevance.
3 — Sample entries are partially accurate but incomplete or generic.
2 — Sample entries are mostly inaccurate or irrelevant.
1 — No sample entries are provided.

#### D. Response Clarity and Structure (0.15)
Measures whether the agent's response clearly describes the sheet and its contents.

5 — Response is clear, well-structured, and describes the sheet comprehensively.
4 — Response is mostly clear but lacks minor details or structure.
3 — Response is partially clear but incomplete or poorly structured.
2 — Response is unclear or poorly structured.
1 — Response is absent or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "template_completeness": <1-5>,
  "sharing_and_accessibility": <1-5>,
  "sample_data_quality": <1-5>,
  "response_clarity_and_structure": <1-5>,
  "dimension_reasoning": {{
    "template_completeness": "<one sentence citing specific evidence>",
    "sharing_and_accessibility": "<one sentence citing specific evidence>",
    "sample_data_quality": "<one sentence citing specific evidence>",
    "response_clarity_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "template_completeness": 0.35,
    "sharing_and_accessibility": 0.30,
    "sample_data_quality": 0.20,
    "response_clarity_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())