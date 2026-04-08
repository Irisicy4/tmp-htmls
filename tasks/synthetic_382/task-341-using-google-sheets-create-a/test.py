"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a machine learning experiment tracker in Google Sheets with sample data sourced from online references.
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


TASK_INSTRUCTION = """Using Google Sheets, create a machine learning experiment tracker. Include columns for Experiment ID, Model Type, Training Dataset, Hyperparameters, Training Accuracy, Validation Accuracy, and Notes. Populate sample data using references from papers or tutorials available online."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a machine learning experiment tracker in Google Sheets. The tracker must include specific columns for Experiment ID, Model Type, Training Dataset, Hyperparameters, Training Accuracy, Validation Accuracy, and Notes. Sample data must be sourced from credible references such as research papers or tutorials available online.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Using Google Sheets, create a machine learning experiment tracker. Include columns for Experiment ID, Model Type, Training Dataset, Hyperparameters, Training Accuracy, Validation Accuracy, and Notes. Populate sample data using references from papers or tutorials available online.

## Task-Specific Constraints
- Must include all specified columns in the tracker.
- Must populate sample data for at least 5 experiments.
- Must source sample data from credible references (e.g., arxiv.org, towardsdatascience.com).
- Must use Google Sheets to create the tracker.
- Output must be structured as a table in the sheet.
- Must provide evidence of tool usage (e.g., URLs visited).

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms (arxiv.org, towardsdatascience.com)?
- Are all specified columns present in the tracker?
- Are there at least 5 experiments with sample data populated?
- Is the sample data sourced from credible references?
- Is the tracker structured correctly as a table in Google Sheets?

### Step 2: Dimension Scoring

#### A. Tracker Completeness (0.35)
Measures whether the tracker includes all required columns and sample data.

5 — All specified columns are present, and sample data for at least 5 experiments is fully populated.
4 — All columns are present, but sample data is partially populated (e.g., fewer than 5 experiments).
3 — Most columns are present, but some are missing or incomplete.
2 — Few columns are present, and sample data is mostly missing.
1 — No tracker created or completely incorrect.

#### B. Source Coverage (0.30)
Measures whether the agent used the required platforms and sourced credible references.

5 — References from both arxiv.org and towardsdatascience.com are used, with URLs provided.
4 — References from one platform are used, with URLs provided.
3 — References are used but URLs are missing or incomplete.
2 — References are mostly absent or not credible.
1 — No references used.

#### C. Data Specificity (0.20)
Measures the level of detail and specificity in the sample data.

5 — Sample data includes detailed hyperparameters, accuracy metrics, and notes for all experiments.
4 — Sample data includes most details but lacks minor elements (e.g., notes or accuracy metrics).
3 — Sample data includes basic details but lacks depth (e.g., generic hyperparameters).
2 — Sample data is vague or incomplete.
1 — No sample data provided.

#### D. Output Structure (0.15)
Measures whether the tracker is well-organized and structured correctly.

5 — Tracker is neatly organized as a table in Google Sheets, with clear formatting.
4 — Tracker is organized but formatting is inconsistent or unclear.
3 — Tracker is usable but poorly formatted.
2 — Tracker is disorganized and difficult to interpret.
1 — No tracker created or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tracker_completeness": <1-5>,
  "source_coverage": <1-5>,
  "data_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "tracker_completeness": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "data_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tracker_completeness": 0.35,
    "source_coverage": 0.30,
    "data_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())