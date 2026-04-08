"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a confusion matrix calculator in Google Sheets for classification models, including precision, recall, and F1 score calculations.
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


TASK_INSTRUCTION = """Use Google Sheets to create a confusion matrix calculator for classification models. Include fields for entering predictions and labels, formulas for calculating precision, recall, and F1 score, and display a summary table with results."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a confusion matrix calculator in Google Sheets for classification models. The deliverable must include fields for entering predictions and labels, formulas for calculating precision, recall, and F1 score, and a summary table displaying the results. The task falls under the domain of data and ML engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to create a confusion matrix calculator for classification models. Include fields for entering predictions and labels, formulas for calculating precision, recall, and F1 score, and display a summary table with results.

## Task-Specific Constraints
- Must create a functional Google Sheet with formulas for precision, recall, and F1 score.
- Must include fields for entering predictions and labels.
- Must display a summary table with calculated metrics.
- Must use Google Sheets as the platform to create the deliverable.
- The formulas must be correct and applicable to classification models.
- The output must be organized and easy to interpret.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Google Sheets and create the required calculator?
- Are fields for entering predictions and labels present in the deliverable?
- Are the formulas for precision, recall, and F1 score correctly implemented?
- Is the summary table displaying the calculated metrics properly formatted?
- Does the deliverable meet the task-specific constraints?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the confusion matrix calculator is correct and functional.

5 — Includes correct and functional formulas for precision, recall, and F1 score, and all required fields.
4 — Includes mostly correct formulas and fields, with minor errors.
3 — Includes partially correct formulas or missing fields.
2 — Includes major errors in formulas or missing key components.
1 — No functional deliverable created.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints are satisfied.

5 — Fully satisfies all constraints, including fields, formulas, and summary table.
4 — Satisfies most constraints, with minor omissions.
3 — Satisfies some constraints, but key elements are missing.
2 — Satisfies few constraints, with major omissions.
1 — Does not satisfy any constraints.

#### C. Depth of Implementation (0.20)
Measures the level of detail and specificity in the deliverable.

5 — Includes detailed formulas, well-organized fields, and a clear summary table.
4 — Includes detailed formulas and fields, but lacks minor details.
3 — Includes basic formulas and fields, but lacks depth.
2 — Includes minimal effort, with significant missing details.
1 — No meaningful implementation.

#### D. Output Structure and Organization (0.15)
Measures the clarity and organization of the deliverable.

5 — Output is well-structured, easy to interpret, and visually clear.
4 — Output is mostly well-structured, with minor formatting issues.
3 — Output is usable but poorly organized.
2 — Output is disorganized and difficult to interpret.
1 — Output is completely unstructured or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_of_implementation": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_of_implementation": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "depth_of_implementation": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())