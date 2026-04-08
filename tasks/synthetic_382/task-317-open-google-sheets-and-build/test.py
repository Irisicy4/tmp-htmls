"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Build a confusion matrix calculator in Google Sheets for binary classification with formulas for precision, recall, and F1-score.
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


TASK_INSTRUCTION = """Open Google Sheets and build a confusion matrix calculator for binary classification. Include inputs for true positives, false positives, true negatives, and false negatives, and formulas to calculate precision, recall, and F1-score."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to build a confusion matrix calculator in Google Sheets for binary classification. The calculator must include inputs for true positives, false positives, true negatives, and false negatives, along with formulas to calculate precision, recall, and F1-score. This task falls under the domain of Data & ML Engineering, and a successful completion involves correctly setting up the sheet with all required formulas and inputs.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and build a confusion matrix calculator for binary classification. Include inputs for true positives, false positives, true negatives, and false negatives, and formulas to calculate precision, recall, and F1-score.

## Task-Specific Constraints
- Must create a Google Sheet with labeled input cells for true positives, false positives, true negatives, and false negatives.
- Must include formulas for precision, recall, and F1-score in separate cells.
- Precision formula: true positives / (true positives + false positives).
- Recall formula: true positives / (true positives + false negatives).
- F1-score formula: 2 * (precision * recall) / (precision + recall).
- Output must be organized and clearly labeled for usability.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create a new sheet?
- Are labeled input cells for true positives, false positives, true negatives, and false negatives present?
- Are formulas for precision, recall, and F1-score correctly implemented in the sheet?
- Is the sheet organized and labeled for usability?
- Did the agent follow the task-specific constraints accurately?

### Step 2: Dimension Scoring

#### A. Formula Accuracy (0.35)
Measures whether the formulas for precision, recall, and F1-score are correctly implemented.

5 — All formulas are correct and functional.
4 — One formula is slightly incorrect but mostly functional.
3 — At least one formula is partially correct but incomplete.
2 — Most formulas are incorrect or missing.
1 — No formulas are implemented.

#### B. Input Completeness (0.30)
Measures whether all required input cells (true positives, false positives, true negatives, false negatives) are present and labeled.

5 — All input cells are present and clearly labeled.
4 — All input cells are present but labeling is unclear.
3 — At least two input cells are present and labeled.
2 — Only one input cell is present or labeling is missing.
1 — No input cells are present.

#### C. Sheet Organization (0.20)
Measures whether the sheet is organized and labeled for usability.

5 — Sheet is well-organized with clear labels and logical layout.
4 — Sheet is organized but labels or layout are slightly unclear.
3 — Sheet is partially organized but lacks clarity in some areas.
2 — Sheet is poorly organized or difficult to understand.
1 — Sheet is completely disorganized or unusable.

#### D. Execution Trace Compliance (0.15)
Measures whether the agent followed the task-specific constraints and tool-call trace aligns with the task.

5 — Execution trace fully aligns with the task-specific constraints.
4 — Execution trace mostly aligns with the constraints.
3 — Execution trace partially aligns with the constraints.
2 — Execution trace minimally aligns with the constraints.
1 — Execution trace does not align with the constraints.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "formula_accuracy": <1-5>,
  "input_completeness": <1-5>,
  "sheet_organization": <1-5>,
  "execution_trace_compliance": <1-5>,
  "dimension_reasoning": {{
    "formula_accuracy": "<one sentence citing specific evidence>",
    "input_completeness": "<one sentence citing specific evidence>",
    "sheet_organization": "<one sentence citing specific evidence>",
    "execution_trace_compliance": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "formula_accuracy": 0.35,
    "input_completeness": 0.30,
    "sheet_organization": 0.20,
    "execution_trace_compliance": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())