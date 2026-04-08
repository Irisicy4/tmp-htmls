"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Extract and filter datasets from Kaggle based on specific criteria, and return details of the top 5 results.
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


TASK_INSTRUCTION = """Go to Kaggle’s 'Datasets' page, apply filters for datasets in CSV format with more than 10,000 rows and updated in the last year. Extract the first 5 results, including dataset name, creator, and the number of rows."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating Kaggle's 'Datasets' page, applying specific filters (CSV format, more than 10,000 rows, updated in the last year), and extracting details of the top 5 datasets. A successful completion requires the agent to return dataset name, creator, and the number of rows for each of the 5 results in a structured format.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Kaggle’s 'Datasets' page, apply filters for datasets in CSV format with more than 10,000 rows and updated in the last year. Extract the first 5 results, including dataset name, creator, and the number of rows.

## Task-Specific Constraints
- Must navigate to Kaggle’s 'Datasets' page.
- Must apply the filters for CSV format, more than 10,000 rows, and updated in the last year.
- Must extract exactly 5 datasets.
- Must include dataset name, creator, and the number of rows for each result.
- Output must be structured as a table or JSON list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kaggle’s 'Datasets' page?
- Did the agent apply the required filters (CSV format, >10,000 rows, updated in the last year)?
- Did the agent extract exactly 5 datasets?
- Does the response include dataset name, creator, and the number of rows for each result?
- Is the output structured as a table or JSON list?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent correctly applied the required filters on Kaggle.

5 — All filters (CSV format, >10,000 rows, updated in the last year) were applied correctly.
4 — Two filters were applied correctly, but one was partially incorrect.
3 — At least one filter was applied correctly, but others were missing or incorrect.
2 — Filters were mostly incorrect or missing.
1 — No filters were applied.

#### B. Dataset Extraction Completeness (0.30)
Measures whether the agent extracted the required number of datasets and included all required fields.

5 — Extracted exactly 5 datasets with all required fields (name, creator, number of rows).
4 — Extracted 5 datasets but one field was incomplete or missing for one dataset.
3 — Extracted fewer than 5 datasets, but at least 3 with all fields present.
2 — Extracted fewer than 3 datasets or fields were mostly incomplete.
1 — No datasets extracted.

#### C. Output Structure (0.20)
Measures whether the response is structured as a table or JSON list.

5 — Output is a well-formatted table or JSON list with clear structure.
4 — Output is structured but contains minor formatting issues.
3 — Output is partially structured but difficult to interpret.
2 — Output is poorly structured or unorganized.
1 — Output is unstructured or missing.

#### D. Evidence Credibility (0.15)
Measures whether the agent’s response aligns with the tool-call trace and task requirements.

5 — Response fully aligns with the tool-call trace and task requirements.
4 — Response mostly aligns but contains minor inconsistencies.
3 — Response partially aligns but has notable inconsistencies.
2 — Response has major inconsistencies or lacks credibility.
1 — Response does not align with the tool-call trace.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "dataset_extraction_completeness": <1-5>,
  "output_structure": <1-5>,
  "evidence_credibility": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "dataset_extraction_completeness": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>",
    "evidence_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "dataset_extraction_completeness": 0.30,
    "output_structure": 0.20,
    "evidence_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())