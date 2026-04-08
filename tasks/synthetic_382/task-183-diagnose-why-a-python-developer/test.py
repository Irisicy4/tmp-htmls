"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Diagnose the cause of an 'ImportError' related to Flask extensions and recommend a fix.
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


TASK_INSTRUCTION = """Diagnose why a Python developer might encounter the error 'ImportError: cannot import name XYZ from module ABC' while working with 'Flask' version 2.2 and its extensions. Navigate community forums, GitHub issues, and the official Flask documentation to find the root cause, affected version range, and a recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose the cause of an 'ImportError' related to Flask extensions in version 2.2, identify affected version ranges, and recommend a fix. The domain is software engineering, specifically debugging and dependency management in Python. A successful completion involves identifying the root cause, citing credible sources, and providing actionable recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a Python developer might encounter the error 'ImportError: cannot import name XYZ from module ABC' while working with 'Flask' version 2.2 and its extensions. Navigate community forums, GitHub issues, and the official Flask documentation to find the root cause, affected version range, and a recommended fix.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (github.com, stackoverflow.com, flask.palletsprojects.com).
- Must identify the root cause of the error and affected version range.
- Must recommend a fix that is actionable and specific.
- Must cite credible sources for all claims made.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Is the root cause of the error clearly identified and explained?
- Is the affected version range specified correctly?
- Is the recommended fix actionable and specific?
- Are all claims backed by credible sources?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the main output (root cause, version range, and fix) is correct and complete.

5 — Root cause, affected version range, and fix are all correct and fully detailed.
4 — Root cause and fix are correct, but version range is incomplete or slightly inaccurate.
3 — Root cause is correct, but fix or version range is missing or vague.
2 — Root cause is partially correct, but fix and version range are mostly wrong or missing.
1 — None of the required elements are correct.

#### B. Platform Coverage (0.30)
Measures whether the agent visited the required platforms and used them effectively.

5 — All 3 platforms were visited, and evidence from each was used in the response.
4 — At least 2 platforms were visited, and evidence from both was used.
3 — At least 2 platforms were visited, but evidence from only one was used.
2 — Only 1 platform was visited, with minimal evidence used.
1 — No required platforms were visited.

#### C. Depth and Specificity (0.20)
Measures the level of detail in identifying the error and recommending a fix.

5 — Response includes detailed explanations, specific error conditions, and precise fixes.
4 — Response includes good detail but lacks minor specificity in explanations or fixes.
3 — Response includes basic explanations and a general fix, but lacks depth.
2 — Response is vague and lacks meaningful detail or specificity.
1 — Response is entirely superficial or absent.

#### D. Source Credibility and Structure (0.15)
Measures whether the sources cited are credible and the output is well-organized.

5 — All claims are backed by credible sources, and the output is very well-structured.
4 — Most claims are backed by credible sources, and the output is well-structured.
3 — Some claims are backed by credible sources, but the structure is basic or inconsistent.
2 — Few claims are backed by credible sources, and the output is poorly structured.
1 — No credible sources are cited, and the output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_and_specificity": 0.20,
    "source_credibility_and_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())