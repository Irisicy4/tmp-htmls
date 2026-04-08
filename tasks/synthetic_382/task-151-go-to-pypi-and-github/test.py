"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Extract breaking changes introduced in Django version 4.2, including affected APIs and migration instructions, by investigating release notes and changelogs on PyPI and GitHub.
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


TASK_INSTRUCTION = """Go to PyPI and GitHub to investigate the release notes and changelog for Django versions 4.1 and 4.2. Extract all breaking changes introduced in version 4.2, including affected APIs and migration instructions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting breaking changes introduced in Django version 4.2 by investigating release notes and changelogs on PyPI and GitHub. The agent must identify affected APIs and provide migration instructions where applicable. A successful completion includes accurate and structured output with all required details.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to PyPI and GitHub to investigate the release notes and changelog for Django versions 4.1 and 4.2. Extract all breaking changes introduced in version 4.2, including affected APIs and migration instructions.

## Task-Specific Constraints
- Must visit both PyPI and GitHub platforms.
- Must include all breaking changes introduced in Django 4.2.
- Must identify affected APIs and provide migration instructions.
- Output must be structured as a list or table.
- Must distinguish between breaking changes and other types of updates.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both PyPI and GitHub platforms?
- Are all breaking changes for Django 4.2 included in the response?
- Are affected APIs and migration instructions clearly identified?
- Is the output organized as a structured list or table?
- Are the breaking changes correctly distinguished from other updates?

### Step 2: Dimension Scoring

#### A. Breaking Changes Accuracy (0.35)
Measures whether the agent correctly identifies all breaking changes introduced in Django 4.2.

5 — Identifies all breaking changes with no errors or omissions.
4 — Identifies most breaking changes with minor omissions or inaccuracies.
3 — Identifies some breaking changes but with significant omissions or errors.
2 — Identifies very few breaking changes or includes major inaccuracies.
1 — Fails to identify any breaking changes.

#### B. Platform Coverage (0.30)
Measures whether the agent visited and utilized both PyPI and GitHub as required.

5 — Clearly uses both platforms and cites information from each.
4 — Uses both platforms but with incomplete or unclear citations.
3 — Uses only one platform or provides unclear evidence of usage.
2 — Attempts to use platforms but fails to extract relevant information.
1 — Does not use either platform.

#### C. Depth of Details (0.20)
Measures the specificity and clarity of the response, including APIs and migration instructions.

5 — Provides detailed and accurate API information and clear migration instructions.
4 — Provides mostly accurate details but with minor gaps or unclear instructions.
3 — Provides some details but lacks clarity or specificity in key areas.
2 — Provides minimal details with significant gaps or inaccuracies.
1 — Provides no meaningful details.

#### D. Output Structure and Organization (0.15)
Measures whether the output is well-structured and easy to understand.

5 — Output is well-organized, structured as a clear list or table, and easy to follow.
4 — Output is mostly well-organized but with minor formatting issues.
3 — Output is somewhat organized but lacks clarity or structure.
2 — Output is poorly organized and difficult to follow.
1 — Output is unstructured and incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "breaking_changes_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_details": <1-5>,
  "output_structure_and_organization": <1-5>,
  "dimension_reasoning": {{
    "breaking_changes_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_details": "<one sentence citing specific evidence>",
    "output_structure_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "breaking_changes_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_details": 0.20,
    "output_structure_and_organization": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())