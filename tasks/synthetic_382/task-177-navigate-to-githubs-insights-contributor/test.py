"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract contributor stats from GitHub's Insights Contributor Graph tool for Python's official repository and summarize the top 3 contributors by commit count.
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


TASK_INSTRUCTION = """Navigate to GitHub's Insights Contributor Graph tool for Python's official repository. Extract the contributor stats for the last 6 months and summarize the top 3 contributors by commit count."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires navigating to GitHub's Insights Contributor Graph tool for Python's official repository, extracting contributor statistics for the last 6 months, and summarizing the top 3 contributors by commit count. This task is in the domain of software engineering and involves data extraction and summarization.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate to GitHub's Insights Contributor Graph tool for Python's official repository. Extract the contributor stats for the last 6 months and summarize the top 3 contributors by commit count.

## Task-Specific Constraints
- Must navigate to the Insights Contributor Graph tool on GitHub.
- Must extract contributor statistics specifically for the last 6 months.
- Must identify the top 3 contributors based on commit count.
- Must provide commit counts for each of the top 3 contributors.
- Output must be organized as a structured list or table.
- Must ensure data accuracy and consistency with the tool's output.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to GitHub's Insights Contributor Graph tool?
- Did the agent extract contributor statistics for the last 6 months?
- Are the top 3 contributors identified correctly based on commit count?
- Are the commit counts accurate and consistent with the tool's output?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identified and summarized the top 3 contributors by commit count.

5 — Identifies the top 3 contributors with accurate commit counts and provides a structured summary.
4 — Identifies the top 3 contributors with minor inaccuracies in commit counts or summary.
3 — Identifies the top 3 contributors but with significant inaccuracies or incomplete summary.
2 — Identifies fewer than 3 contributors or provides mostly incorrect data.
1 — Fails to identify contributors or provides no usable data.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent navigated to the correct GitHub tool and extracted data from the required source.

5 — Successfully navigates to the Insights Contributor Graph tool and extracts data for the last 6 months.
4 — Navigates to the correct tool but extracts data for an incorrect time range or with minor omissions.
3 — Navigates to the correct tool but extracts incomplete or partially incorrect data.
2 — Navigates to an incorrect tool or extracts mostly incorrect data.
1 — Fails to navigate to the required tool or extract any usable data.

#### C. Depth and Specificity of Data (0.25)
Measures whether the agent provides detailed and specific data, including commit counts for each contributor.

5 — Provides detailed commit counts for all top 3 contributors with no missing information.
4 — Provides commit counts for all top 3 contributors with minor missing details.
3 — Provides commit counts for fewer than 3 contributors or with significant missing details.
2 — Provides mostly incomplete or incorrect commit counts.
1 — Fails to provide any commit counts or specific data.

#### D. Output Structure and Organization (0.10)
Measures whether the agent's output is well-organized and easy to interpret.

5 — Output is structured as a clear table or list with all required elements.
4 — Output is structured but with minor formatting issues or omissions.
3 — Output is partially structured but lacks clarity or completeness.
2 — Output is mostly unstructured or difficult to interpret.
1 — Output is completely unstructured or unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_and_specificity_of_data": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_and_specificity_of_data": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_and_specificity_of_data": 0.25,
    "output_structure_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())