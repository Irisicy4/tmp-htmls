"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Build a confusion matrix for a mock dataset using Google Sheets based on provided JSON data.
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


TASK_INSTRUCTION = """Build a confusion matrix for a mock dataset using Google Sheets. Input the class predictions and ground truth values provided in a fictional JSON snippet. Populate the matrix showing counts for true positives, false negatives, false positives, and true negatives for each class."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a confusion matrix for a mock dataset using Google Sheets. The agent must input class predictions and ground truth values from a fictional JSON snippet and correctly populate the matrix with counts for true positives, false negatives, false positives, and true negatives for each class.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Build a confusion matrix for a mock dataset using Google Sheets. Input the class predictions and ground truth values provided in a fictional JSON snippet. Populate the matrix showing counts for true positives, false negatives, false positives, and true negatives for each class.

## Task-Specific Constraints
- Must correctly calculate and populate all counts for true positives, false negatives, false positives, and true negatives.
- Must use Google Sheets to create the confusion matrix.
- Must structure the matrix as a table with rows and columns labeled appropriately.
- Must ensure all provided classes are included in the matrix.
- Must accurately process the JSON data for predictions and ground truth values.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and use it to create the confusion matrix?
- Are all provided classes included in the matrix?
- Are the counts for true positives, false negatives, false positives, and true negatives correctly calculated?
- Is the matrix structured as a labeled table with rows and columns?
- Did the agent accurately process the JSON data for predictions and ground truth values?

### Step 2: Dimension Scoring

#### A. Matrix Accuracy (0.35)
Measures whether the confusion matrix is correctly populated with accurate counts.

5 — All counts for true positives, false negatives, false positives, and true negatives are correct for all classes.
4 — Minor errors in counts for one class, but overall matrix is mostly correct.
3 — Partial completion: matrix includes some correct counts but has major errors or omissions.
2 — Significant errors or omissions in most counts.
1 — Matrix is absent or completely incorrect.

#### B. Class Coverage (0.30)
Measures whether all provided classes are included in the confusion matrix.

5 — All classes from the JSON data are included in the matrix.
4 — One class is missing, but the majority are included.
3 — Partial completion: matrix includes at least half of the classes.
2 — Less than half of the classes are included.
1 — No classes are included.

#### C. JSON Data Processing (0.25)
Measures whether the agent accurately processed the JSON data to extract predictions and ground truth values.

5 — JSON data is fully processed, and all values are correctly extracted and used.
4 — Minor errors in processing, but most values are correctly extracted.
3 — Partial completion: some values are correctly extracted, but others are missing or incorrect.
2 — Significant errors in extracting values from the JSON data.
1 — JSON data is not processed or completely incorrect.

#### D. Matrix Structure and Formatting (0.10)
Measures whether the confusion matrix is well-organized and formatted correctly.

5 — Matrix is fully labeled with rows and columns clearly indicating classes and counts.
4 — Minor formatting issues, but labels and structure are mostly correct.
3 — Partial completion: matrix is present but lacks clear labels or structure.
2 — Significant formatting issues or lack of organization.
1 — Matrix is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated to Google Sheets and created a confusion matrix. All required classes were included, and counts were mostly accurate. Minor formatting issues were observed.",
  "matrix_accuracy": 4,
  "class_coverage": 5,
  "json_data_processing": 4,
  "matrix_structure_and_formatting": 4,
  "dimension_reasoning": {
    "matrix_accuracy": "Counts were mostly correct, with minor errors observed in one class.",
    "class_coverage": "All classes from the JSON data were included in the matrix.",
    "json_data_processing": "JSON data was processed accurately, with minor extraction errors.",
    "matrix_structure_and_formatting": "Matrix was labeled and structured correctly, with minor formatting issues."
  },
  "overall_score": 4.25,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "matrix_accuracy": 0.35,
    "class_coverage": 0.30,
    "json_data_processing": 0.25,
    "matrix_structure_and_formatting": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())