"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a confusion matrix calculator in Google Sheets for binary classification, including formulas for accuracy, precision, recall, and F1 score.
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


TASK_INSTRUCTION = """Using Google Sheets, create a confusion matrix calculator for binary classification. Incorporate editable fields for true positives, false positives, true negatives, and false negatives, and calculate accuracy, precision, recall, and F1 score using formulas. Use example input values from scikit-learn documentation for testing."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a confusion matrix calculator in Google Sheets for binary classification. The calculator must include editable fields for true positives, false positives, true negatives, and false negatives, and formulas to calculate accuracy, precision, recall, and F1 score. The agent must use example input values from scikit-learn documentation for testing.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, create a confusion matrix calculator for binary classification. Incorporate editable fields for true positives, false positives, true negatives, and false negatives, and calculate accuracy, precision, recall, and F1 score using formulas. Use example input values from scikit-learn documentation for testing.

## Task-Specific Constraints
- Must use Google Sheets as the platform for implementation.
- Editable fields must be provided for true positives, false positives, true negatives, and false negatives.
- Formulas for accuracy, precision, recall, and F1 score must be correctly implemented.
- Example input values from scikit-learn documentation must be used for testing.
- The final output must be structured and visually clear for users.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Google Sheets as the platform for implementation?
- Are editable fields for true positives, false positives, true negatives, and false negatives present?
- Are formulas for accuracy, precision, recall, and F1 score correctly implemented?
- Did the agent use example input values from scikit-learn documentation for testing?
- Is the final output structured and visually clear for users?

### Step 2: Dimension Scoring

#### A. Formula Accuracy (0.35)
Measures whether the formulas for accuracy, precision, recall, and F1 score are correctly implemented.

5 — All formulas are correct and calculate the expected values.
4 — Most formulas are correct, with minor errors in one calculation.
3 — Some formulas are correct, but significant errors exist in multiple calculations.
2 — Few formulas are correct, with major errors in most calculations.
1 — No formulas are correct or implemented.

#### B. Platform Usage (0.30)
Measures whether the agent correctly used Google Sheets to implement the task.

5 — Google Sheets was used correctly, with all required features implemented.
4 — Google Sheets was used, but some required features are missing.
3 — Google Sheets was used, but implementation is incomplete or unclear.
2 — Google Sheets was partially used, with major omissions.
1 — Google Sheets was not used at all.

#### C. Example Input Testing (0.25)
Measures whether example input values from scikit-learn documentation were used for testing.

5 — Example input values were used correctly, and results match expected values.
4 — Example input values were used, but results have minor discrepancies.
3 — Example input values were partially used, with significant discrepancies.
2 — Example input values were mostly missing or incorrect.
1 — Example input values were not used at all.

#### D. Output Structure (0.10)
Measures whether the final output is structured and visually clear for users.

5 — Output is well-structured, visually clear, and easy to understand.
4 — Output is mostly structured and clear, with minor visual issues.
3 — Output is partially structured, with some clarity issues.
2 — Output is poorly structured and visually unclear.
1 — Output is unstructured and confusing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "formula_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "example_input_testing": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "formula_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "example_input_testing": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "formula_accuracy": 0.35,
    "platform_usage": 0.30,
    "example_input_testing": 0.25,
    "output_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())