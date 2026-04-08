"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Complete the dataset license classification workflow on the OpenML website.
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


TASK_INSTRUCTION = """Complete the dataset license classification workflow on the OpenML website. Submit the metadata of any public dataset, ensuring you select appropriate data license options and complete all required fields. Report the summary of the final page confirmation screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to classify a dataset license on the OpenML website, submit metadata for a public dataset, and ensure all required fields are completed. A successful completion includes selecting an appropriate data license and providing a summary of the final confirmation screen.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Complete the dataset license classification workflow on the OpenML website. Submit the metadata of any public dataset, ensuring you select appropriate data license options and complete all required fields. Report the summary of the final page confirmation screen.

## Task-Specific Constraints
- Must navigate to openml.org and complete the license classification workflow.
- Must select an appropriate license from the options provided.
- Must ensure all required metadata fields are filled.
- Must provide a summary of the final confirmation screen.
- Must not leave any mandatory fields blank.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to openml.org and complete the workflow?
- Did the agent select an appropriate license option?
- Are all required metadata fields filled in the response?
- Is the summary of the final confirmation screen present and accurate?
- Are there any mandatory fields left blank?

### Step 2: Dimension Scoring

#### A. Workflow Completion Accuracy (0.35)
Measures whether the agent successfully completed the license classification workflow on OpenML.

5 — Workflow completed with all required steps and no errors.
4 — Workflow completed with minor omissions or errors.
3 — Workflow partially completed; some required steps missing.
2 — Workflow mostly incomplete or incorrect.
1 — Workflow not attempted or completely incorrect.

#### B. Metadata Completeness (0.30)
Measures whether all required metadata fields were filled.

5 — All required fields are filled correctly and completely.
4 — Most required fields are filled; minor omissions.
3 — Some required fields are filled; significant omissions.
2 — Few required fields are filled; mostly incomplete.
1 — No required fields are filled.

#### C. License Selection Appropriateness (0.25)
Measures whether the selected license is appropriate for the dataset.

5 — License is appropriate and justified based on the dataset.
4 — License is appropriate but justification is weak or missing.
3 — License is selected but may not be fully appropriate.
2 — License is inappropriate or poorly chosen.
1 — No license selected.

#### D. Output Structure and Summary Quality (0.10)
Measures the organization and clarity of the final output, including the summary.

5 — Output is well-organized and summary is clear and accurate.
4 — Output is organized but summary has minor inaccuracies.
3 — Output is partially organized; summary is unclear or incomplete.
2 — Output is poorly organized; summary is mostly missing or incorrect.
1 — Output is disorganized and summary is absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "workflow_completion_accuracy": <1-5>,
  "metadata_completeness": <1-5>,
  "license_selection_appropriateness": <1-5>,
  "output_structure_and_summary_quality": <1-5>,
  "dimension_reasoning": {{
    "workflow_completion_accuracy": "<one sentence citing specific evidence>",
    "metadata_completeness": "<one sentence citing specific evidence>",
    "license_selection_appropriateness": "<one sentence citing specific evidence>",
    "output_structure_and_summary_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "workflow_completion_accuracy": 0.35,
    "metadata_completeness": 0.30,
    "license_selection_appropriateness": 0.25,
    "output_structure_and_summary_quality": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())