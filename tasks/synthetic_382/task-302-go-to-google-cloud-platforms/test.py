"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Compare pre-trained TensorFlow models optimized for mobile devices on text classification tasks and report the top 3 models with their specifications.
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


TASK_INSTRUCTION = """Go to Google Cloud Platform's AI model comparison page and complete the workflow for comparing pre-trained ML models on text classification tasks. Select TensorFlow models and filter by those optimized for small devices like mobile phones. Report the top 3 models displayed along with their model names and specifications."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating Google Cloud Platform's AI model comparison page to identify and compare pre-trained TensorFlow models optimized for mobile devices on text classification tasks. A successful completion requires the agent to provide the top 3 models, including their names and specifications.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Google Cloud Platform's AI model comparison page and complete the workflow for comparing pre-trained ML models on text classification tasks. Select TensorFlow models and filter by those optimized for small devices like mobile phones. Report the top 3 models displayed along with their model names and specifications.

## Task-Specific Constraints
- Must navigate to Google Cloud Platform's AI model comparison page.
- Must filter results to TensorFlow models only.
- Must apply the filter for models optimized for small devices like mobile phones.
- Must report the top 3 models displayed, including their names and specifications.
- Output must be structured as a list or table for clarity.
- Must ensure the reported specifications match the displayed data.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the Google Cloud Platform's AI model comparison page?
- Did the agent apply the required filters (TensorFlow models and optimization for small devices)?
- Are the top 3 models identified in the response?
- Are the model names and specifications correctly reported?
- Is the output structured clearly as a list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and reported the top 3 models with their specifications.

5 — All 3 models are correctly identified, and specifications are accurate.
4 — 2 models are correctly identified, and specifications are mostly accurate.
3 — At least 1 model is correctly identified with partial specifications.
2 — Models are mostly incorrect or specifications are missing.
1 — No models are correctly identified or specifications are absent.

#### B. Coverage of Filters Applied (0.30)
Measures whether the agent applied the required filters (TensorFlow and optimization for small devices).

5 — Both filters applied correctly.
4 — One filter applied correctly, and the other partially correct.
3 — At least one filter applied correctly.
2 — Filters are mostly incorrect or missing.
1 — No filters applied.

#### C. Depth of Specifications (0.20)
Measures the level of detail in the reported model specifications.

5 — Specifications include all relevant details (e.g., model size, accuracy, device optimization).
4 — Specifications include most relevant details but lack minor elements.
3 — Specifications include basic details but lack depth.
2 — Specifications are mostly incomplete or vague.
1 — Specifications are absent.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is organized and easy to interpret.

5 — Output is structured as a clear list or table with proper formatting.
4 — Output is mostly structured but lacks minor formatting clarity.
3 — Output is minimally structured but understandable.
2 — Output is poorly structured and hard to interpret.
1 — Output is unstructured or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent navigated to the correct platform and applied the required filters. The top 3 models were identified, but some specifications lacked detail. The output was structured as a table, making it easy to interpret.",
  "deliverable_accuracy": 4,
  "coverage_of_filters_applied": 5,
  "depth_of_specifications": 3,
  "output_structure_and_clarity": 5,
  "dimension_reasoning": {{
    "deliverable_accuracy": "Two models were correctly identified, and specifications were mostly accurate.",
    "coverage_of_filters_applied": "Both filters (TensorFlow and optimization for small devices) were applied correctly.",
    "depth_of_specifications": "Specifications included basic details but lacked depth.",
    "output_structure_and_clarity": "The output was structured as a clear table."
  }},
  "overall_score": 4.05,
  "passed": true
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_filters_applied": 0.30,
    "depth_of_specifications": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())