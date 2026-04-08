"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Evaluate whether the agent successfully identified and reported a sentiment analysis model optimized for large datasets from the Hugging Face model hub.
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


TASK_INSTRUCTION = """Use the Hugging Face model hub to search for a sentiment analysis model optimized for performance on large datasets. Navigate the filters and complete the workflow for selecting an appropriate model. Report the chosen model's name, version, and key performance metrics."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use the Hugging Face model hub to identify a sentiment analysis model optimized for large datasets. The agent must navigate filters, evaluate models, and report the chosen model's name, version, and key performance metrics. A successful completion includes accurate identification and reporting of the model details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the Hugging Face model hub to search for a sentiment analysis model optimized for performance on large datasets. Navigate the filters and complete the workflow for selecting an appropriate model. Report the chosen model's name, version, and key performance metrics.

## Task-Specific Constraints
- Must use the Hugging Face model hub platform.
- Must apply filters to narrow down models optimized for large datasets.
- Must evaluate at least 3 models before selecting one.
- Must report the chosen model's name, version, and at least 2 key performance metrics.
- Output must be organized as a structured JSON object.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Hugging Face model hub platform?
- Did the agent apply filters to narrow down models optimized for large datasets?
- Did the agent evaluate at least 3 models before selecting one?
- Are the chosen model's name, version, and performance metrics present in the response?
- Is the output organized as a structured JSON object?

### Step 2: Dimension Scoring

#### A. Model Identification Accuracy (0.35)
Measures whether the agent correctly identified and reported a sentiment analysis model optimized for large datasets.

5 — Accurately identifies the model name, version, and at least 2 key performance metrics.
4 — Identifies the model name and version, but only 1 performance metric.
3 — Identifies the model name but lacks version or performance metrics.
2 — Incorrect or incomplete model identification.
1 — No model identified.

#### B. Filter Usage and Coverage (0.30)
Measures whether the agent applied filters and evaluated multiple models.

5 — Applies filters and evaluates at least 3 models before selecting one.
4 — Applies filters and evaluates 2 models.
3 — Applies filters but evaluates only 1 model.
2 — Applies filters incorrectly or does not evaluate models.
1 — No filters applied.

#### C. Output Specificity (0.20)
Measures the depth and specificity of the reported details.

5 — Includes detailed metrics (e.g., accuracy, dataset size) and structured comparisons.
4 — Includes metrics but lacks structured comparisons.
3 — Includes basic metrics but lacks depth.
2 — Metrics are vague or incomplete.
1 — No metrics provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the output.

5 — Output is a well-organized JSON object with credible sources cited.
4 — Output is structured but lacks citations.
3 — Output is partially structured but credible.
2 — Output is disorganized or lacks credibility.
1 — Output is absent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "model_identification_accuracy": <1-5>,
  "filter_usage_and_coverage": <1-5>,
  "output_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "model_identification_accuracy": "<one sentence citing specific evidence>",
    "filter_usage_and_coverage": "<one sentence citing specific evidence>",
    "output_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "model_identification_accuracy": 0.35,
    "filter_usage_and_coverage": 0.30,
    "output_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())