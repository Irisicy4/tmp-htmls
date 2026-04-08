"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Set up a GitHub Actions workflow to automatically lint Python code using 'flake8' on every pull request.
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


TASK_INSTRUCTION = """Set up a GitHub Actions workflow to automatically lint Python code using 'flake8' on every pull request. You must use GitHub's documentation to configure the YAML file, select the appropriate runner, and add a step to fail the workflow if any linting issues are detected. Provide the final YAML file content and explain its configuration."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to set up a GitHub Actions workflow to lint Python code using 'flake8' on every pull request. The agent must use GitHub's documentation to configure the YAML file, select the appropriate runner, and ensure the workflow fails if linting issues are detected. A successful completion includes a valid YAML file and an explanation of its configuration.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a GitHub Actions workflow to automatically lint Python code using 'flake8' on every pull request. You must use GitHub's documentation to configure the YAML file, select the appropriate runner, and add a step to fail the workflow if any linting issues are detected. Provide the final YAML file content and explain its configuration.

## Task-Specific Constraints
- Must visit docs.github.com to reference GitHub Actions documentation.
- Must visit github.com to verify repository settings or workflow integration.
- Must visit flake8.pycqa.org to confirm linting tool usage and configuration.
- YAML file must include a runner, a linting step using 'flake8', and a failure condition for linting issues.
- Explanation must describe the purpose of each YAML section and how it satisfies the task requirements.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to docs.github.com, github.com, and flake8.pycqa.org? Which ones were actually visited?
- Is the YAML file provided, and does it include a runner, a linting step, and a failure condition?
- Does the explanation describe the YAML configuration and its purpose accurately?
- Are all required elements (runner, linting step, failure condition) present and correctly implemented?
- Is the YAML syntax valid and consistent with GitHub Actions standards?

### Step 2: Dimension Scoring

#### A. YAML Configuration Accuracy (0.35)
Measures whether the YAML file is correctly configured to lint Python code using 'flake8' and fail on issues.

5 — YAML file includes all required elements (runner, linting step, failure condition) and is syntactically valid.
4 — YAML file includes most required elements but has minor issues or omissions.
3 — YAML file is partially complete but missing key elements or contains errors.
2 — YAML file is mostly incorrect or incomplete.
1 — YAML file is absent or completely wrong.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Agent visited docs.github.com, github.com, and flake8.pycqa.org, and used them effectively.
4 — Agent visited at least two platforms and used them effectively.
3 — Agent visited at least one platform and used it partially.
2 — Agent visited platforms but did not use them effectively.
1 — Agent did not visit any required platforms.

#### C. Explanation Depth (0.20)
Measures the quality and completeness of the YAML file explanation.

5 — Explanation covers all YAML sections, their purpose, and how they satisfy the task requirements.
4 — Explanation covers most YAML sections with minor omissions.
3 — Explanation is partially complete but lacks depth or clarity.
2 — Explanation is mostly incomplete or unclear.
1 — Explanation is absent or completely wrong.

#### D. Output Structure and Validity (0.15)
Measures whether the output is well-organized and adheres to GitHub Actions standards.

5 — Output is well-organized, syntactically valid, and adheres to standards.
4 — Output is organized and valid but has minor formatting issues.
3 — Output is partially organized or contains minor syntax errors.
2 — Output is disorganized or contains major syntax errors.
1 — Output is absent or completely invalid.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "yaml_configuration_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "explanation_depth": <1-5>,
  "output_structure_and_validity": <1-5>,
  "dimension_reasoning": {{
    "yaml_configuration_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "explanation_depth": "<one sentence citing specific evidence>",
    "output_structure_and_validity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "yaml_configuration_accuracy": 0.35,
    "platform_coverage": 0.30,
    "explanation_depth": 0.20,
    "output_structure_and_validity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())