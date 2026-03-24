"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Extract metadata for Transformer-based models from Hugging Face and GitHub.
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


TASK_INSTRUCTION = """Go to Hugging Face's model repository and filter for Transformer-based models trained on text-to-text tasks. Extract the first five models' names, descriptions, and the number of downloads. Make sure to cross-check with the GitHub pages for additional metadata, such as license types and last update dates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

This task requires the agent to extract metadata for Transformer-based models from Hugging Face and GitHub. Specifically, the agent must identify the first five text-to-text models, gather their names, descriptions, and download counts, and cross-check GitHub for license types and last update dates. The deliverable must be a structured output with all required fields.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Hugging Face's model repository and filter for Transformer-based models trained on text-to-text tasks. Extract the first five models' names, descriptions, and the number of downloads. Make sure to cross-check with the GitHub pages for additional metadata, such as license types and last update dates.

## Task-Specific Constraints
- Must visit both Hugging Face and GitHub platforms.
- Must extract metadata for exactly five models.
- Metadata must include: model name, description, download count, license type, and last update date.
- Output must be structured as a table or JSON object.
- Metadata must match the actual data on the platforms.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both Hugging Face and GitHub platforms?
- Did the agent extract metadata for exactly five models?
- Are all required fields (name, description, download count, license, last update) present in the response?
- Is the output structured as a table or JSON object?
- Does the metadata match the actual data on the platforms?

### Step 2: Dimension Scoring

#### A. Metadata Accuracy (0.35)
Measures whether the extracted metadata is correct and matches the source platforms.

5 — All metadata fields for all five models are accurate and match the platforms.
4 — Minor inaccuracies in one or two fields but otherwise correct.
3 — Some fields are missing or incorrect, but the majority are accurate.
2 — Most fields are missing or incorrect.
1 — No accurate metadata extracted.

#### B. Platform Coverage (0.30)
Measures whether the agent visited both required platforms and used them effectively.

5 — Both Hugging Face and GitHub were visited, and data was extracted from both.
4 — Both platforms were visited, but data from one is incomplete.
3 — Only one platform was visited, or data from one is missing.
2 — Both platforms were visited but no meaningful data was extracted.
1 — Neither platform was visited.

#### C. Metadata Completeness (0.20)
Measures whether all required fields (name, description, downloads, license, last update) are present.

5 — All five fields are present for all five models.
4 — One field is missing for one or two models.
3 — Multiple fields are missing but the majority are present.
2 — Most fields are missing for most models.
1 — No fields are present.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-structured and easy to interpret.

5 — Output is perfectly structured as a table or JSON object with clear organization.
4 — Output is structured but contains minor formatting issues.
3 — Output is partially structured but difficult to interpret.
2 — Output is poorly structured and hard to interpret.
1 — Output is unstructured or completely disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "metadata_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "metadata_completeness": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "metadata_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "metadata_completeness": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "metadata_accuracy": 0.35,
    "platform_coverage": 0.30,
    "metadata_completeness": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())