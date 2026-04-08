"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Evaluate whether the agent successfully filtered sentiment analysis models on Hugging Face based on IMDb dataset, accuracy, and PyTorch compatibility, and reported the top 3 models with names and links.
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


TASK_INSTRUCTION = """Visit the Hugging Face Model Hub, search for sentiment analysis models, and complete the filter workflow to show models trained on the IMDb dataset, with at least 90% accuracy and available for PyTorch. Report the names and links of the top 3 matching models."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves searching for sentiment analysis models on Hugging Face, applying filters for IMDb dataset, 90% accuracy, and PyTorch compatibility, and reporting the top 3 models with names and links. This is a Data & ML Engineering task where success depends on correct filtering and accurate reporting of the required models.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Visit the Hugging Face Model Hub, search for sentiment analysis models, and complete the filter workflow to show models trained on the IMDb dataset, with at least 90% accuracy and available for PyTorch. Report the names and links of the top 3 matching models.

## Task-Specific Constraints
- Must navigate to the Hugging Face Model Hub.
- Must apply filters for IMDb dataset, 90% accuracy, and PyTorch compatibility.
- Must identify at least 3 models that meet the criteria.
- Must provide names and direct links to the models.
- Output must be structured as a list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Hugging Face Model Hub?
- Did the agent apply the correct filters (IMDb dataset, 90% accuracy, PyTorch)?
- Are at least 3 models reported in the response?
- Are the names and links of the models correctly provided?
- Is the output structured as a list or table?

### Step 2: Dimension Scoring

#### A. Filtering Accuracy (0.35)
Measures whether the agent applied the correct filters to identify the models.

5 — All filters (IMDb dataset, 90% accuracy, PyTorch) applied correctly.
4 — One filter slightly incorrect or missing but results mostly valid.
3 — At least two filters applied correctly; partial results valid.
2 — Only one filter applied correctly; results mostly invalid.
1 — No filters applied correctly; results invalid.

#### B. Model Identification Completeness (0.30)
Measures whether the agent identified at least 3 models meeting the criteria.

5 — Identifies 3 or more models that fully meet the criteria.
4 — Identifies 2 models that fully meet the criteria.
3 — Identifies 1 model that meets the criteria.
2 — Identifies models but none meet the criteria.
1 — No models identified.

#### C. Output Specificity (0.20)
Measures whether the agent provided names and links for the models.

5 — Provides names and direct links for all reported models.
4 — Provides names and links for most models.
3 — Provides names but links are incomplete or missing.
2 — Provides names but no links.
1 — No names or links provided.

#### D. Output Structure Quality (0.15)
Measures whether the response is organized and easy to interpret.

5 — Output is structured as a clear list or table.
4 — Output is mostly structured but slightly unclear.
3 — Output is minimally structured but interpretable.
2 — Output is poorly structured and hard to interpret.
1 — Output is unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "filtering_accuracy": <1-5>,
  "model_identification_completeness": <1-5>,
  "output_specificity": <1-5>,
  "output_structure_quality": <1-5>,
  "dimension_reasoning": {{
    "filtering_accuracy": "<one sentence citing specific evidence>",
    "model_identification_completeness": "<one sentence citing specific evidence>",
    "output_specificity": "<one sentence citing specific evidence>",
    "output_structure_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "filtering_accuracy": 0.35,
    "model_identification_completeness": 0.30,
    "output_specificity": 0.20,
    "output_structure_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())