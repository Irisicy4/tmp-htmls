"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Set up a continuous integration workflow for a JavaScript project using GitHub Actions.
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


TASK_INSTRUCTION = """Set up a new continuous integration workflow for a JavaScript project using GitHub Actions. Create a YAML file that runs tests on Node.js versions 16 and 18, installing dependencies via npm and running `npm test`. Use GitHub's workflow configuration documentation."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a continuous integration (CI) workflow for a JavaScript project using GitHub Actions. The deliverable is a YAML configuration file that runs tests on Node.js versions 16 and 18, installs dependencies via npm, and executes `npm test`. The agent must use GitHub's workflow documentation to guide their solution.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a new continuous integration workflow for a JavaScript project using GitHub Actions. Create a YAML file that runs tests on Node.js versions 16 and 18, installing dependencies via npm and running `npm test`. Use GitHub's workflow configuration documentation.

## Task-Specific Constraints
- The YAML file must include jobs for both Node.js v16 and v18.
- The YAML file must install dependencies using npm.
- The YAML file must execute `npm test` as part of the workflow.
- The agent must reference GitHub's official workflow documentation.
- The workflow must be valid and free of syntax errors.
- The agent must demonstrate the use of GitHub Actions syntax correctly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Does the YAML file include jobs for both Node.js v16 and v18?
- Does the YAML file install dependencies using npm?
- Does the YAML file execute `npm test` as part of the workflow?
- Did the agent reference GitHub's official workflow documentation?
- Is the YAML file valid and free of syntax errors?

### Step 2: Dimension Scoring

#### A. Workflow Configuration Accuracy (0.35)
Measures whether the YAML file is correctly configured to meet the task requirements.

5 — YAML file is fully correct, with jobs for Node.js v16 and v18, npm dependency installation, and `npm test` execution.
4 — YAML file is mostly correct but has minor issues (e.g., missing one required step or minor syntax issues).
3 — YAML file is partially correct but misses significant elements (e.g., only one Node.js version or missing npm installation).
2 — YAML file is mostly incorrect, with multiple missing or incorrect elements.
1 — YAML file is completely incorrect or absent.

#### B. Coverage of Requirements (0.30)
Measures whether all task-specific constraints are addressed.

5 — All constraints are fully addressed (e.g., both Node.js versions, npm installation, `npm test`, documentation reference).
4 — Most constraints are addressed, with minor omissions.
3 — Some constraints are addressed, but significant omissions exist.
2 — Few constraints are addressed, with major omissions.
1 — No constraints are addressed.

#### C. Depth of Documentation Use (0.20)
Measures whether the agent effectively used GitHub's workflow documentation.

5 — Agent clearly references and applies GitHub's workflow documentation in their solution.
4 — Agent references the documentation but applies it with minor errors.
3 — Agent references the documentation but applies it incompletely or incorrectly.
2 — Agent minimally references the documentation with significant errors.
1 — Agent does not reference the documentation.

#### D. Syntax and Formatting Quality (0.15)
Measures whether the YAML file is valid and formatted correctly.

5 — YAML file is valid, free of syntax errors, and well-formatted.
4 — YAML file is valid but has minor formatting issues.
3 — YAML file is valid but poorly formatted or has minor syntax issues.
2 — YAML file has significant syntax errors or is poorly formatted.
1 — YAML file is invalid or completely unformatted.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "workflow_configuration_accuracy": <1-5>,
  "coverage_of_requirements": <1-5>,
  "depth_of_documentation_use": <1-5>,
  "syntax_and_formatting_quality": <1-5>,
  "dimension_reasoning": {{
    "workflow_configuration_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_requirements": "<one sentence citing specific evidence>",
    "depth_of_documentation_use": "<one sentence citing specific evidence>",
    "syntax_and_formatting_quality": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "workflow_configuration_accuracy": 0.35,
    "coverage_of_requirements": 0.30,
    "depth_of_documentation_use": 0.20,
    "syntax_and_formatting_quality": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())