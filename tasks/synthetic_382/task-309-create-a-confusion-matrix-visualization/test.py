"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a confusion matrix visualization for a classification task using Google Sheets, using data from the UCI Wine Quality dataset.
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


TASK_INSTRUCTION = """Create a confusion matrix visualization for a classification task using Google Sheets. Use sample predictions and labels from a CSV file available on the UCI Machine Learning Repository for the 'Wine Quality' dataset. Format the matrix with proper labels and accuracy computation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a confusion matrix visualization for a classification task using Google Sheets. The agent must use the 'Wine Quality' dataset from the UCI Machine Learning Repository, extract predictions and labels, and compute accuracy. The output must be formatted with proper row/column labels and include the accuracy metric.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Create a confusion matrix visualization for a classification task using Google Sheets. Use sample predictions and labels from a CSV file available on the UCI Machine Learning Repository for the 'Wine Quality' dataset. Format the matrix with proper labels and accuracy computation.

## Task-Specific Constraints
- Must download the 'Wine Quality' dataset from the UCI Machine Learning Repository.
- Must extract predictions and labels from the dataset.
- Must create a confusion matrix in Google Sheets with proper row and column labels.
- Must compute and include the accuracy metric in the output.
- Must provide a clear and structured response summarizing the steps taken and the final output.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the UCI Machine Learning Repository and download the correct dataset?
- Did the agent extract predictions and labels from the dataset correctly?
- Did the agent create a confusion matrix in Google Sheets with proper formatting (row/column labels)?
- Did the agent compute and include the accuracy metric in the output?
- Is the final response clear, structured, and does it summarize the steps taken?

### Step 2: Dimension Scoring

#### A. Confusion Matrix Accuracy (0.35)
Measures whether the confusion matrix is correct and matches the predictions and labels.

5 — The confusion matrix is fully correct and matches the predictions and labels.
4 — Minor errors in the confusion matrix, but mostly correct.
3 — The confusion matrix is partially correct but contains significant errors.
2 — The confusion matrix is mostly incorrect or incomplete.
1 — The confusion matrix is missing or completely wrong.

#### B. Coverage of Required Steps (0.30)
Measures whether the agent followed all required steps (e.g., downloading dataset, extracting data, creating matrix, computing accuracy).

5 — All required steps are completed correctly.
4 — One minor step is missing or incomplete.
3 — One major step is missing, but others are completed.
2 — Multiple major steps are missing or incomplete.
1 — None of the required steps are completed.

#### C. Accuracy Metric Computation (0.20)
Measures whether the accuracy metric is computed correctly and included in the output.

5 — The accuracy metric is computed correctly and included.
4 — Minor errors in the accuracy computation or presentation.
3 — The accuracy metric is partially correct or incomplete.
2 — The accuracy metric is mostly incorrect or missing.
1 — The accuracy metric is completely absent.

#### D. Output Clarity and Structure (0.15)
Measures whether the final response is clear, well-structured, and summarizes the steps taken.

5 — The response is exceptionally clear, well-structured, and comprehensive.
4 — The response is clear and structured but lacks minor details.
3 — The response is somewhat clear but lacks structure or key details.
2 — The response is unclear or poorly structured.
1 — The response is completely unclear or unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "confusion_matrix_accuracy": <1-5>,
  "coverage_of_required_steps": <1-5>,
  "accuracy_metric_computation": <1-5>,
  "output_clarity_and_structure": <1-5>,
  "dimension_reasoning": {
    "confusion_matrix_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_steps": "<one sentence citing specific evidence>",
    "accuracy_metric_computation": "<one sentence citing specific evidence>",
    "output_clarity_and_structure": "<one sentence citing specific evidence>"
  },
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "confusion_matrix_accuracy": 0.35,
    "coverage_of_required_steps": 0.30,
    "accuracy_metric_computation": 0.20,
    "output_clarity_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())