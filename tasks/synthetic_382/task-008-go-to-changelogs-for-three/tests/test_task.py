"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract breaking changes from changelogs of React, Lodash, and Axios between specified versions.
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


TASK_INSTRUCTION = """Go to changelogs for three popular open-source libraries—React, Lodash, and Axios—and extract all breaking changes introduced between two specific versions: React v17 to v18, Lodash v4.17.21 to v5, and Axios v0.23 to v1. List the breaking changes along with the version they were introduced and a brief description for each."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to visit the changelogs of React, Lodash, and Axios and extract all breaking changes introduced between specific versions. The domain is software engineering, and a successful completion involves listing the breaking changes for each library, organized by version, with a brief description of each change.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to changelogs for three popular open-source libraries—React, Lodash, and Axios—and extract all breaking changes introduced between two specific versions: React v17 to v18, Lodash v4.17.21 to v5, and Axios v0.23 to v1. List the breaking changes along with the version they were introduced and a brief description for each.

## Task-Specific Constraints
- Must visit the changelog pages for React, Lodash, and Axios.
- Must extract breaking changes only between the specified versions.
- Must organize the output by library and version.
- Must include a brief description for each breaking change.
- Must ensure the extracted changes are accurate and relevant to the specified versions.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the changelog pages for React, Lodash, and Axios? Which ones were actually visited?
- Are breaking changes for all three libraries included in the response?
- Are the breaking changes organized by library and version as required?
- Are the descriptions of the breaking changes accurate and relevant to the specified versions?
- Did the agent include any extraneous or irrelevant information?

### Step 2: Dimension Scoring

#### A. Breaking Change Accuracy (0.35)
Measures whether the breaking changes listed are accurate and relevant to the specified versions.

5 — All breaking changes are accurate and relevant to the specified versions.
4 — Most breaking changes are accurate, with minor inaccuracies or omissions.
3 — Some breaking changes are accurate, but there are significant inaccuracies or omissions.
2 — Few breaking changes are accurate, with many inaccuracies or omissions.
1 — No accurate breaking changes are provided.

#### B. Coverage of Libraries (0.30)
Measures whether breaking changes are provided for all three libraries.

5 — Breaking changes are provided for all three libraries with no omissions.
4 — Breaking changes are provided for two libraries, with minor omissions for the third.
3 — Breaking changes are provided for at least two libraries, but with significant omissions.
2 — Breaking changes are provided for only one library.
1 — No breaking changes are provided for any library.

#### C. Detail and Specificity (0.20)
Measures the level of detail and specificity in the descriptions of the breaking changes.

5 — Descriptions are detailed and specific for all breaking changes.
4 — Descriptions are mostly detailed and specific, with minor gaps.
3 — Descriptions are partially detailed and specific, but lack depth in some areas.
2 — Descriptions are mostly vague or lack specificity.
1 — Descriptions are entirely vague or absent.

#### D. Output Organization (0.15)
Measures whether the output is well-organized and follows the required structure.

5 — Output is fully organized by library and version, with clear formatting.
4 — Output is mostly organized, with minor formatting issues.
3 — Output is partially organized, but lacks clarity or structure in some areas.
2 — Output is poorly organized, with significant structural issues.
1 — Output is entirely unorganized or incoherent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "breaking_change_accuracy": <1-5>,
  "coverage_of_libraries": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_organization": <1-5>,
  "dimension_reasoning": {{
    "breaking_change_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_libraries": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "breaking_change_accuracy": 0.35,
    "coverage_of_libraries": 0.30,
    "detail_and_specificity": 0.20,
    "output_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())