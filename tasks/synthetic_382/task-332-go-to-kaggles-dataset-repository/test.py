"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Identify and extract metadata for three Kaggle datasets suitable for training a multi-class image recognition model.
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


TASK_INSTRUCTION = """Go to Kaggle's dataset repository and use the search and filter tools to find three datasets suitable for training a multi-class image recognition model. Extract metadata for each dataset, including dataset name, source, size, number of classes, and download link."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating Kaggle's dataset repository to identify three datasets suitable for training a multi-class image recognition model. The agent must extract metadata for each dataset, including dataset name, source, size, number of classes, and download link. Successful completion requires accurate and structured output that meets the task constraints.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Kaggle's dataset repository and use the search and filter tools to find three datasets suitable for training a multi-class image recognition model. Extract metadata for each dataset, including dataset name, source, size, number of classes, and download link.

## Task-Specific Constraints
- Must identify exactly three datasets suitable for multi-class image recognition.
- Metadata must include dataset name, source, size, number of classes, and download link.
- Must use Kaggle's search and filter tools to locate datasets.
- Output must be structured as a table or JSON object.
- Metadata must be accurate and sourced directly from Kaggle.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Kaggle's dataset repository?
- Did the agent identify exactly three datasets suitable for multi-class image recognition?
- Is the metadata for each dataset complete (name, source, size, number of classes, and download link)?
- Is the output organized as a table or JSON object?
- Are the dataset details accurate and sourced from Kaggle?

### Step 2: Dimension Scoring

#### A. Metadata Accuracy (0.35)
Measures whether the extracted metadata for each dataset is correct and complete.

5 — Metadata for all three datasets is complete and accurate (name, source, size, number of classes, download link).
4 — Metadata for all three datasets is mostly complete, with minor inaccuracies.
3 — Metadata for at least two datasets is partially complete and accurate.
2 — Metadata is mostly incomplete or inaccurate for most datasets.
1 — Metadata is completely absent or incorrect.

#### B. Dataset Coverage (0.30)
Measures whether the agent identified exactly three suitable datasets.

5 — Exactly three datasets suitable for multi-class image recognition are identified.
4 — Three datasets are identified, but one or more may not be suitable.
3 — Two datasets are identified and suitable.
2 — Only one suitable dataset is identified.
1 — No suitable datasets are identified.

#### C. Detail Specificity (0.20)
Measures the level of detail and specificity in the metadata provided.

5 — Metadata includes detailed and specific values for all required fields (e.g., exact size, number of classes).
4 — Metadata includes most details but lacks minor specifics.
3 — Metadata includes basic details but lacks depth or specificity.
2 — Metadata is vague or missing key details.
1 — Metadata is absent or completely vague.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and follows the required format.

5 — Output is structured as a clear table or JSON object with all required fields.
4 — Output is mostly well-organized but has minor formatting issues.
3 — Output is organized but lacks clarity or consistency.
2 — Output is poorly organized or difficult to interpret.
1 — Output is completely unstructured or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "metadata_accuracy": <1-5>,
  "dataset_coverage": <1-5>,
  "detail_specificity": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "metadata_accuracy": "<one sentence citing specific evidence>",
    "dataset_coverage": "<one sentence citing specific evidence>",
    "detail_specificity": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "metadata_accuracy": 0.35,
    "dataset_coverage": 0.30,
    "detail_specificity": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())