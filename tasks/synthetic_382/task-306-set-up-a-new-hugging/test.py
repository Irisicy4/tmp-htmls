"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Set up a Hugging Face Space to benchmark sentiment analysis models on the IMDb dataset with specific configurations and report the results.
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


TASK_INSTRUCTION = """Set up a new Hugging Face Space to benchmark sentiment analysis models on the IMDb dataset. Configure the space with a CPU-only runtime environment, load three pre-trained models (BERT, DistilBERT, RoBERTa), and set up evaluation metrics for accuracy and inference speed. Report the configured environment and model list."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is to set up a Hugging Face Space to benchmark sentiment analysis models on the IMDb dataset. The agent must configure the space with a CPU-only runtime environment, load three specific pre-trained models (BERT, DistilBERT, RoBERTa), and set up evaluation metrics for accuracy and inference speed. A successful completion includes reporting the configured environment and the list of models.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a new Hugging Face Space to benchmark sentiment analysis models on the IMDb dataset. Configure the space with a CPU-only runtime environment, load three pre-trained models (BERT, DistilBERT, RoBERTa), and set up evaluation metrics for accuracy and inference speed. Report the configured environment and model list.

## Task-Specific Constraints
- The Hugging Face Space must be configured with a CPU-only runtime environment.
- The IMDb dataset must be loaded and used for benchmarking.
- The agent must load exactly three pre-trained models: BERT, DistilBERT, and RoBERTa.
- Evaluation metrics must include both accuracy and inference speed.
- The agent must report the configured environment and the list of loaded models.
- The output must be structured and clearly formatted.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent configure a Hugging Face Space with a CPU-only runtime environment?
- Did the agent load the IMDb dataset for benchmarking?
- Did the agent load all three required models (BERT, DistilBERT, RoBERTa)?
- Are both evaluation metrics (accuracy and inference speed) set up?
- Is the agent's final output structured and does it include the configured environment and model list?

### Step 2: Dimension Scoring

#### A. Configuration Accuracy (0.35)
Measures whether the Hugging Face Space was configured correctly.

5 — The Space is configured with a CPU-only runtime and all required elements.
4 — Minor issues in configuration, but the Space is mostly correct.
3 — Partial configuration; some required elements are missing.
2 — Significant issues in configuration; most elements are missing.
1 — No attempt to configure the Space.

#### B. Model and Dataset Coverage (0.30)
Measures whether all required models and the IMDb dataset were loaded.

5 — All three models and the IMDb dataset are correctly loaded.
4 — One minor omission or error in loading models or dataset.
3 — At least two models and the dataset are loaded, but with errors or omissions.
2 — Only one model or the dataset is loaded.
1 — No models or dataset are loaded.

#### C. Metric Setup Completeness (0.20)
Measures whether the required evaluation metrics are set up.

5 — Both accuracy and inference speed metrics are correctly set up.
4 — One metric is set up correctly, the other is partially correct.
3 — One metric is set up correctly, but the other is missing.
2 — Both metrics are partially set up but incomplete.
1 — No metrics are set up.

#### D. Output Structure and Clarity (0.15)
Measures the quality and clarity of the final output.

5 — The output is well-structured, clear, and includes all required information.
4 — The output is mostly clear, with minor formatting or clarity issues.
3 — The output is partially clear but lacks some required information.
2 — The output is poorly structured or unclear.
1 — The output is unstructured or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "configuration_accuracy": <1-5>,
  "model_and_dataset_coverage": <1-5>,
  "metric_setup_completeness": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "configuration_accuracy": "<one sentence citing specific evidence>",
    "model_and_dataset_coverage": "<one sentence citing specific evidence>",
    "metric_setup_completeness": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "configuration_accuracy": 0.35,
    "model_and_dataset_coverage": 0.30,
    "metric_setup_completeness": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())