"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Create a confusion matrix calculator in Google Sheets with formulas for precision, recall, and F1-score using dummy data for at least three models.
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


TASK_INSTRUCTION = """Open Google Sheets and create a confusion matrix calculator for evaluating classification models. Include formula logic for precision, recall, and F1-score calculations using dummy data. Use at least three models' results for demonstration and structure the sheet for easy use."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to create a confusion matrix calculator in Google Sheets for evaluating classification models. The sheet must include formula logic for precision, recall, and F1-score calculations using dummy data. The agent must demonstrate results for at least three models and structure the sheet for easy usability.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Open Google Sheets and create a confusion matrix calculator for evaluating classification models. Include formula logic for precision, recall, and F1-score calculations using dummy data. Use at least three models' results for demonstration and structure the sheet for easy use.

## Task-Specific Constraints
- Must create a confusion matrix calculator in Google Sheets.
- Must include formulas for precision, recall, and F1-score calculations.
- Must use dummy data for at least three models.
- The sheet must be structured for easy usability.
- The agent must demonstrate the results for all three models.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent create a confusion matrix calculator in Google Sheets?
- Are formulas for precision, recall, and F1-score included and functional?
- Is dummy data for at least three models present and used correctly?
- Is the sheet structured for easy usability?
- Are the results for all three models demonstrated?

### Step 2: Dimension Scoring

#### A. Formula Accuracy (0.35)
Measures whether formulas for precision, recall, and F1-score are correctly implemented and functional.

5 — All formulas are correct and functional for all three models.
4 — Most formulas are correct, with minor errors in one model.
3 — Formulas are partially correct but incomplete or contain significant errors.
2 — Formulas are mostly incorrect or missing.
1 — No formulas are implemented.

#### B. Model Coverage (0.30)
Measures whether dummy data for at least three models is present and correctly used.

5 — Dummy data for all three models is present and correctly used.
4 — Dummy data for three models is present but partially incorrect.
3 — Dummy data for two models is present and correct.
2 — Dummy data for one model is present or mostly incorrect.
1 — No dummy data is present.

#### C. Sheet Usability (0.20)
Measures whether the sheet is structured for easy usability.

5 — Sheet is well-organized, with clear labels and intuitive layout.
4 — Sheet is mostly organized, with minor usability issues.
3 — Sheet is usable but lacks clarity or has significant usability issues.
2 — Sheet is poorly organized and difficult to use.
1 — Sheet is unusable or completely disorganized.

#### D. Results Demonstration (0.15)
Measures whether results for all three models are demonstrated.

5 — Results for all three models are demonstrated clearly.
4 — Results for all three models are demonstrated but lack clarity.
3 — Results for two models are demonstrated.
2 — Results for one model are demonstrated.
1 — No results are demonstrated.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "formula_accuracy": <1-5>,
  "model_coverage": <1-5>,
  "sheet_usability": <1-5>,
  "results_demonstration": <1-5>,
  "dimension_reasoning": {{
    "formula_accuracy": "<one sentence citing specific evidence>",
    "model_coverage": "<one sentence citing specific evidence>",
    "sheet_usability": "<one sentence citing specific evidence>",
    "results_demonstration": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "formula_accuracy": 0.35,
    "model_coverage": 0.30,
    "sheet_usability": 0.20,
    "results_demonstration": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())