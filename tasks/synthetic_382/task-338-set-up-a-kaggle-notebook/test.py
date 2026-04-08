"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a Kaggle notebook environment to preprocess a dataset for training a regression model.
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


TASK_INSTRUCTION = """Set up a Kaggle notebook environment to preprocess a dataset for training a regression model. Configure the notebook to load a CSV dataset from Kaggle's repository, clean missing values, and visualize correlations using a heatmap. Report the final notebook setup and steps performed."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a Kaggle notebook environment for preprocessing a dataset to train a regression model. The agent must configure the notebook to load a CSV dataset from Kaggle's repository, clean missing values, and visualize correlations using a heatmap. A successful completion includes a properly configured notebook and a clear report of the steps performed.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a Kaggle notebook environment to preprocess a dataset for training a regression model. Configure the notebook to load a CSV dataset from Kaggle's repository, clean missing values, and visualize correlations using a heatmap. Report the final notebook setup and steps performed.

## Task-Specific Constraints
- Must use Kaggle's notebook environment to perform the task.
- Must successfully load a CSV dataset from Kaggle's repository.
- Must clean missing values in the dataset.
- Must generate and display a heatmap of correlations in the dataset.
- Must provide a clear report of the notebook setup and preprocessing steps performed.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Kaggle's notebook environment to perform the task?
- Did the agent successfully load a CSV dataset from Kaggle's repository?
- Did the agent clean missing values in the dataset?
- Did the agent generate and display a heatmap of correlations in the dataset?
- Did the agent provide a clear report of the notebook setup and preprocessing steps performed?

### Step 2: Dimension Scoring

#### A. Notebook Configuration Accuracy (0.35)
Measures whether the notebook was correctly set up and configured.

5 — Notebook is fully configured with all required elements (dataset loaded, missing values cleaned, heatmap generated).
4 — Notebook is mostly configured but missing minor details.
3 — Notebook is partially configured but lacks key elements.
2 — Notebook setup is mostly incorrect or incomplete.
1 — Notebook setup is completely absent or wrong.

#### B. Dataset Preprocessing Completeness (0.30)
Measures whether the dataset preprocessing steps were completed.

5 — Missing values are fully cleaned, and preprocessing steps are clearly documented.
4 — Missing values are mostly cleaned, with minor omissions.
3 — Missing values are partially cleaned, and documentation is incomplete.
2 — Preprocessing is mostly incorrect or missing.
1 — Preprocessing is completely absent or wrong.

#### C. Visualization Quality (0.20)
Measures the quality and accuracy of the heatmap visualization.

5 — Heatmap is accurate, well-labeled, and provides clear insights into correlations.
4 — Heatmap is mostly accurate but has minor issues in labeling or clarity.
3 — Heatmap is partially accurate but lacks clarity or insights.
2 — Heatmap is mostly incorrect or unclear.
1 — Heatmap is completely absent or wrong.

#### D. Report Clarity (0.15)
Measures the clarity and completeness of the report provided.

5 — Report is clear, detailed, and covers all steps performed.
4 — Report is mostly clear but lacks minor details.
3 — Report is partially clear but omits key steps.
2 — Report is mostly unclear or incomplete.
1 — Report is completely absent or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully used Kaggle's notebook environment, loaded a CSV dataset, cleaned missing values, and generated a heatmap. The report provided is mostly clear but lacks minor details.",
  "notebook_configuration_accuracy": 4,
  "dataset_preprocessing_completeness": 4,
  "visualization_quality": 4,
  "report_clarity": 4,
  "dimension_reasoning": {
    "notebook_configuration_accuracy": "The notebook was mostly configured with all required elements except minor details.",
    "dataset_preprocessing_completeness": "Missing values were mostly cleaned, with minor omissions.",
    "visualization_quality": "The heatmap was mostly accurate but had minor issues in labeling.",
    "report_clarity": "The report was mostly clear but lacked minor details."
  },
  "overall_score": 4.0,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "notebook_configuration_accuracy": 0.35,
    "dataset_preprocessing_completeness": 0.30,
    "visualization_quality": 0.20,
    "report_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())