"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Diagnose and resolve 'ImportError' issue in Python using Flask and SQLAlchemy.
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


TASK_INSTRUCTION = """Diagnose why a user is encountering 'ImportError: cannot import name 'ABC' from 'xyz'' in Python when using Flask with SQLAlchemy. Research GitHub issues, Stack Overflow, and Flask documentation to identify the root cause, affected versions, and resolution steps."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires diagnosing an 'ImportError' in Python when using Flask with SQLAlchemy. The agent must research GitHub issues, Stack Overflow, and Flask documentation to identify the root cause, affected versions, and resolution steps. A successful completion includes a clear explanation of the error, its root cause, and actionable resolution steps.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a user is encountering 'ImportError: cannot import name 'ABC' from 'xyz'' in Python when using Flask with SQLAlchemy. Research GitHub issues, Stack Overflow, and Flask documentation to identify the root cause, affected versions, and resolution steps.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: stackoverflow.com, github.com, flask.palletsprojects.com.
- Must identify the root cause of the error and include affected versions.
- Must provide actionable resolution steps with sufficient detail.
- Must reference credible sources (e.g., GitHub issues, Flask documentation).
- Output must be organized as a structured list or table.
- Must address both Flask and SQLAlchemy compatibility issues.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response identify the root cause of the error and affected versions?
- Are actionable resolution steps included and sufficiently detailed?
- Are credible sources referenced (e.g., GitHub issues, Flask documentation)?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the error.

5 — Correctly identifies the root cause with affected versions and detailed explanation.
4 — Correctly identifies the root cause but lacks detail or affected versions.
3 — Partially identifies the root cause but misses key details.
2 — Incorrect or vague identification of the root cause.
1 — No attempt to identify the root cause.

#### B. Resolution Steps Completeness (0.30)
Measures whether the agent provides actionable and complete resolution steps.

5 — Provides clear, actionable steps with sufficient detail to resolve the issue.
4 — Provides actionable steps but lacks some detail.
3 — Provides partial steps that may resolve the issue but are incomplete.
2 — Steps are vague or unlikely to resolve the issue.
1 — No resolution steps provided.

#### C. Platform Coverage (0.20)
Measures whether the agent used all required platforms for research.

5 — Uses all three platforms (Stack Overflow, GitHub, Flask documentation) effectively.
4 — Uses two platforms effectively but misses one.
3 — Uses one platform effectively or partially uses two.
2 — Minimal platform usage or ineffective research.
1 — No platform usage.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the response.

5 — Output is well-organized, structured, and references credible sources.
4 — Output is organized but lacks some structure or credibility.
3 — Output is partially organized with minimal references.
2 — Output is disorganized or lacks credible references.
1 — Output is completely disorganized or lacks references.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "The agent visited Stack Overflow and GitHub but did not use Flask documentation. The response identifies the root cause and affected versions but lacks detail. Resolution steps are actionable but incomplete.",
  "root_cause_identification_accuracy": 3,
  "resolution_steps_completeness": 3,
  "platform_coverage": 3,
  "output_structure_and_credibility": 2,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "The response partially identifies the root cause but misses key details.",
    "resolution_steps_completeness": "Resolution steps are actionable but lack sufficient detail.",
    "platform_coverage": "Only two platforms were used effectively; Flask documentation was not visited.",
    "output_structure_and_credibility": "The output is partially organized but lacks credible references."
  }},
  "overall_score": 2.85,
  "passed": false
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "resolution_steps_completeness": 0.30,
    "platform_coverage": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())