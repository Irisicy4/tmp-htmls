"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Identify classification datasets with more than 500 samples from the UCI Machine Learning Repository and extract specific details for the top 5.
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


TASK_INSTRUCTION = """Navigate the UCI Machine Learning Repository to find datasets labeled for 'classification' tasks with more than 500 samples. Extract the dataset names, number of features, and number of samples for the top 5 qualifying datasets."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating the UCI Machine Learning Repository to identify datasets labeled for 'classification' tasks with more than 500 samples. The agent must extract and present the dataset names, number of features, and number of samples for the top 5 qualifying datasets. This task is in the domain of Data & ML Engineering.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate the UCI Machine Learning Repository to find datasets labeled for 'classification' tasks with more than 500 samples. Extract the dataset names, number of features, and number of samples for the top 5 qualifying datasets.

## Task-Specific Constraints
- Must visit the UCI Machine Learning Repository.
- Must identify datasets labeled for 'classification' tasks.
- Must ensure datasets have more than 500 samples.
- Output must include dataset names, number of features, and number of samples.
- Output must be organized as a structured list or table.
- Must extract details for exactly the top 5 qualifying datasets.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the UCI Machine Learning Repository?
- Are the datasets labeled for 'classification' tasks?
- Do all datasets have more than 500 samples?
- Is the output organized as a structured list or table?
- Are the dataset names, number of features, and number of samples correctly extracted?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and extracted the required dataset details.

5 — Extracts dataset names, number of features, and number of samples for all 5 qualifying datasets accurately.
4 — Extracts details for 4 datasets accurately; minor errors in 1 dataset.
3 — Extracts details for at least 3 datasets; significant errors or omissions in others.
2 — Extracts details for 1-2 datasets; most information missing or incorrect.
1 — Fails to extract any meaningful dataset details.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent navigated the required platform(s) and used them appropriately.

5 — Navigates the UCI Machine Learning Repository and no less than 2 additional relevant platforms (e.g., Google, Wikipedia).
4 — Navigates the UCI Machine Learning Repository and 1 additional relevant platform.
3 — Navigates only the UCI Machine Learning Repository.
2 — Navigates the UCI Machine Learning Repository but fails to use it effectively.
1 — Fails to navigate the UCI Machine Learning Repository.

#### C. Depth and Specificity (0.25)
Measures whether the agent provides detailed and specific information for the datasets.

5 — Includes detailed dataset names, exact number of features, and exact number of samples for all datasets.
4 — Includes detailed information for 4 datasets; minor omissions in 1 dataset.
3 — Includes detailed information for at least 3 datasets; significant omissions in others.
2 — Includes partial information for 1-2 datasets; most details missing.
1 — Provides no meaningful details.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and the information is credible.

5 — Output is presented as a structured table or list; all information is sourced and credible.
4 — Output is structured; minor formatting issues or unclear sourcing.
3 — Output is partially structured; some credibility issues or missing sources.
2 — Output is poorly structured; significant credibility issues.
1 — Output is unstructured and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{
  "evidence_summary": "The agent navigated the UCI Machine Learning Repository and extracted dataset details. However, some datasets lack complete information, and the output structure is partially organized.",
  "primary_deliverable_accuracy": 3,
  "coverage_of_required_platforms": 3,
  "depth_and_specificity": 3,
  "output_structure_and_credibility": 2,
  "dimension_reasoning": {
    "primary_deliverable_accuracy": "Extracted details for 3 datasets but missed key information for others.",
    "coverage_of_required_platforms": "Navigated the UCI Machine Learning Repository but did not use additional platforms.",
    "depth_and_specificity": "Provided partial details for 3 datasets; lacked specificity for others.",
    "output_structure_and_credibility": "Output was partially structured but lacked clear sourcing."
  },
  "overall_score": 2.85,
  "passed": false
}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())