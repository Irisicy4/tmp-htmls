"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a hyperparameter tuning tracker in Google Sheets, populate it with data from Kaggle or Colab, and summarize the best experiment.
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


TASK_INSTRUCTION = """Use Google Sheets to build a simple hyperparameter tuning tracker. Create columns for the following: model type, learning rate, batch size, optimizer, and accuracy. Populate the tracker with information for three sample experiments using accuracy data sourced from a Kaggle notebook or Google Colab. Create a summary sheet showing the experiment with the highest accuracy."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a hyperparameter tuning tracker in Google Sheets. The tracker must include columns for model type, learning rate, batch size, optimizer, and accuracy, and must be populated with data from three experiments. The agent must source accuracy data from a Kaggle notebook or Google Colab and create a summary sheet identifying the experiment with the highest accuracy.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets to build a simple hyperparameter tuning tracker. Create columns for the following: model type, learning rate, batch size, optimizer, and accuracy. Populate the tracker with information for three sample experiments using accuracy data sourced from a Kaggle notebook or Google Colab. Create a summary sheet showing the experiment with the highest accuracy.

## Task-Specific Constraints
- Must use Google Sheets to create the tracker.
- Must include columns for model type, learning rate, batch size, optimizer, and accuracy.
- Must populate the tracker with data for three experiments.
- Accuracy data must be sourced from a Kaggle notebook or Google Colab.
- Must create a summary sheet showing the experiment with the highest accuracy.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use Google Sheets to create the tracker?
- Are the required columns (model type, learning rate, batch size, optimizer, accuracy) present?
- Did the agent populate the tracker with data for three experiments?
- Was accuracy data sourced from a Kaggle notebook or Google Colab?
- Did the agent create a summary sheet showing the experiment with the highest accuracy?

### Step 2: Dimension Scoring

#### A. Tracker Completeness (0.35)
Measures whether the tracker includes all required columns and is populated with data for three experiments.

5 — Tracker includes all required columns and is fully populated with data for three experiments.
4 — Tracker includes all required columns but is missing data for one experiment.
3 — Tracker includes most required columns but is missing one or two.
2 — Tracker is missing multiple columns or experiments.
1 — Tracker is absent or completely incorrect.

#### B. Data Sourcing Accuracy (0.30)
Measures whether the accuracy data is sourced from Kaggle or Colab as required.

5 — All accuracy data is correctly sourced from Kaggle or Colab.
4 — Most accuracy data is sourced correctly, with minor issues.
3 — Some accuracy data is sourced correctly, but others are missing or incorrect.
2 — Very little accuracy data is sourced correctly.
1 — No accuracy data is sourced correctly.

#### C. Summary Sheet Quality (0.20)
Measures whether the summary sheet correctly identifies the experiment with the highest accuracy.

5 — Summary sheet is present and correctly identifies the experiment with the highest accuracy.
4 — Summary sheet is present but has minor errors in identifying the highest accuracy.
3 — Summary sheet is present but incomplete or partially incorrect.
2 — Summary sheet is mostly incorrect or missing key information.
1 — Summary sheet is absent or completely incorrect.

#### D. Output Organization (0.15)
Measures whether the tracker and summary sheet are well-organized and easy to interpret.

5 — Tracker and summary sheet are well-organized, clear, and easy to interpret.
4 — Tracker and summary sheet are mostly well-organized, with minor issues.
3 — Tracker and summary sheet are somewhat organized but have noticeable issues.
2 — Tracker and summary sheet are poorly organized and difficult to interpret.
1 — Tracker and summary sheet are completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "tracker_completeness": <1-5>,
  "data_sourcing_accuracy": <1-5>,
  "summary_sheet_quality": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "tracker_completeness": "<one sentence citing specific evidence>",
    "data_sourcing_accuracy": "<one sentence citing specific evidence>",
    "summary_sheet_quality": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "tracker_completeness": 0.35,
    "data_sourcing_accuracy": 0.30,
    "summary_sheet_quality": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())