"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Verify whether Flask, Django, and FastAPI have been updated within the past three months by checking last commit dates, latest release versions, and open issues on their GitHub repositories.
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


TASK_INSTRUCTION = """Verify whether the popular Python web frameworks Flask, Django, and FastAPI have been updated within the past three months. Check the last commit dates, latest release versions, and open issues on their GitHub repositories."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves verifying the update status of three popular Python web frameworks (Flask, Django, FastAPI) by checking their GitHub repositories. A successful completion requires accurate reporting of last commit dates, latest release versions, and open issue counts for each framework.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the popular Python web frameworks Flask, Django, and FastAPI have been updated within the past three months. Check the last commit dates, latest release versions, and open issues on their GitHub repositories.

## Task-Specific Constraints
- Must visit the GitHub repositories for Flask, Django, and FastAPI.
- Must report the last commit dates for each framework.
- Must report the latest release versions for each framework.
- Must report the number of open issues for each framework.
- Output must be organized as a structured list or table.
- Must verify that the data is sourced from the GitHub repositories.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the GitHub repositories for Flask, Django, and FastAPI?
- Are the last commit dates, latest release versions, and open issue counts present for all three frameworks?
- Is the output organized as a structured list or table?
- Are the reported dates and versions accurate based on the GitHub repositories?
- Are the open issue counts correctly sourced and reported?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent accurately reports the last commit dates, latest release versions, and open issue counts for all three frameworks.

5 — All data points (commit dates, release versions, issue counts) are correct for all three frameworks.
4 — Most data points are correct, with minor inaccuracies.
3 — Some data points are correct, but others are missing or incorrect.
2 — Few data points are correct; most are missing or incorrect.
1 — No data points are correct or present.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited and reported data from all three specified GitHub repositories.

5 — Data is reported from all three repositories (Flask, Django, FastAPI).
4 — Data is reported from two repositories, with minor omissions.
3 — Data is reported from at least one repository, but others are missing.
2 — Data is reported from none or only partially from one repository.
1 — No repositories were visited or reported.

#### C. Depth of Information (0.25)
Measures the level of detail in the agent's response, including specific dates, versions, and issue counts.

5 — Response includes detailed dates, versions, and issue counts for all three frameworks.
4 — Response includes most details, with minor omissions.
3 — Response includes some details, but lacks depth or specificity.
2 — Response includes minimal details, with significant omissions.
1 — Response lacks any meaningful details.

#### D. Output Structure and Credibility (0.10)
Measures whether the response is well-organized and sourced from credible GitHub data.

5 — Output is structured as a clear table or list and all data is sourced from GitHub.
4 — Output is mostly well-organized, with minor structural issues or unclear sourcing.
3 — Output is partially organized, with some structural issues or unclear sourcing.
2 — Output is poorly organized and lacks credible sourcing.
1 — Output is unstructured and lacks any credible sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_information": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_information": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_information": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())