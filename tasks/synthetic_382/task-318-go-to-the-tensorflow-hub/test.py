"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Evaluate whether the agent successfully identified the top 3 object detection models compatible with TensorFlow 2.x, including their name, description, and download size.
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


TASK_INSTRUCTION = """Go to the TensorFlow Hub model repository and complete a full workflow to find the top 3 models for object detection that are compatible with TF 2.x. Report each model's name, description, and download size from the final result screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves finding the top 3 object detection models compatible with TensorFlow 2.x from the TensorFlow Hub repository. The agent must report each model's name, description, and download size. This task is in the domain of Data & ML Engineering and requires accurate data extraction and structured reporting.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to the TensorFlow Hub model repository and complete a full workflow to find the top 3 models for object detection that are compatible with TF 2.x. Report each model's name, description, and download size from the final result screen.

## Task-Specific Constraints
- Must navigate to tensorflow.org and locate the model repository.
- Must filter models to ensure compatibility with TensorFlow 2.x.
- Must identify and report exactly 3 models for object detection.
- Must include the name, description, and download size for each model.
- Output must be structured as a table or JSON list.
- Must ensure data accuracy by cross-verifying the reported details.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to tensorflow.org and locate the model repository?
- Did the agent filter models to ensure compatibility with TensorFlow 2.x?
- Are the top 3 models for object detection clearly identified and reported?
- Does the response include the name, description, and download size for each model?
- Is the output organized as a table or JSON list?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the top 3 object detection models and reported their name, description, and download size.

5 — All 3 models are correctly identified, with accurate name, description, and download size.
4 — 2 models are correctly identified, with accurate name, description, and download size.
3 — At least 1 model is correctly identified, with partial details (e.g., missing description or size).
2 — Models are identified, but details are mostly incorrect or missing.
1 — No models are correctly identified.

#### B. Coverage of Workflow (0.30)
Measures whether the agent followed the required workflow steps, including visiting tensorflow.org and filtering for TF 2.x compatibility.

5 — All required steps are completed (visited tensorflow.org, filtered for TF 2.x, identified models).
4 — Most steps are completed, but minor omissions (e.g., filtering not explicitly confirmed).
3 — Partial workflow completed; major omissions (e.g., did not filter for TF 2.x).
2 — Minimal workflow completed; only visited tensorflow.org without filtering or identifying models.
1 — Did not follow the workflow.

#### C. Depth of Reporting (0.25)
Measures the level of detail in the agent's response, including descriptions and download sizes.

5 — All models include detailed descriptions and accurate download sizes.
4 — Most models include detailed descriptions and accurate download sizes.
3 — At least one model includes partial details (e.g., description but no size).
2 — Details are mostly missing or inaccurate.
1 — No details are provided.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and data appears credible.

5 — Output is structured as a table or JSON list, with all data cross-verified and credible.
4 — Output is structured, but minor formatting issues or unclear credibility.
3 — Output is partially structured, with some data appearing credible.
2 — Output is unstructured or data credibility is questionable.
1 — Output is completely unstructured or lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent successfully navigated to tensorflow.org, filtered for TF 2.x compatibility, and reported 3 object detection models with name, description, and download size. Output was structured as a table, but one model's details were incomplete.",
  "primary_deliverable_accuracy": 4,
  "coverage_of_workflow": 5,
  "depth_of_reporting": 3,
  "output_structure_and_credibility": 4,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "Two models were fully detailed, but one had incomplete information.",
    "coverage_of_workflow": "Agent followed all required steps, including filtering for TF 2.x compatibility.",
    "depth_of_reporting": "Descriptions and sizes were provided for most models, but one lacked detail.",
    "output_structure_and_credibility": "Output was structured as a table and appeared credible."
  },
  "overall_score": 4.05,
  "passed": true
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_workflow": 0.30,
    "depth_of_reporting": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())