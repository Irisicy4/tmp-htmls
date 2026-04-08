"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Create a CI/CD pipeline for a JavaScript application using GitHub Actions, including linting, testing, and deploying to Netlify.
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


TASK_INSTRUCTION = """Use the GitHub Actions workflow editor to simulate the creation of a CI/CD pipeline for a JavaScript application. Specify steps for linting, running tests, and deploying to Netlify. Report the configuration shown on the final confirmation screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves creating a CI/CD pipeline for a JavaScript application using GitHub Actions. The pipeline must include steps for linting, testing, and deploying to Netlify. A successful completion requires the agent to provide the final configuration of the workflow as displayed on the confirmation screen.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the GitHub Actions workflow editor to simulate the creation of a CI/CD pipeline for a JavaScript application. Specify steps for linting, running tests, and deploying to Netlify. Report the configuration shown on the final confirmation screen.

## Task-Specific Constraints
- Must visit github.com, docs.github.com, and netlify.com during the task.
- The workflow configuration must include separate steps for linting, testing, and deployment.
- Deployment must target Netlify with valid credentials or tokens.
- The final response must include the full YAML configuration file.
- The YAML file must be syntactically correct and adhere to GitHub Actions schema.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to github.com, docs.github.com, and netlify.com? Which ones were actually visited?
- Does the final response include the YAML configuration file?
- Are the steps for linting, testing, and deployment present in the YAML file?
- Is the YAML file syntactically correct and valid according to GitHub Actions schema?
- Was Netlify deployment configured with valid credentials or tokens?

### Step 2: Dimension Scoring

#### A. Workflow Accuracy (0.35)
Measures whether the CI/CD pipeline is correctly configured and functional.

5 — YAML file is complete, valid, and includes all required steps (linting, testing, deployment).
4 — YAML file is valid but missing minor details or has small errors.
3 — YAML file is partially complete but functional for at least one step.
2 — YAML file is mostly incorrect or incomplete.
1 — YAML file is absent or completely invalid.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them appropriately.

5 — Agent visited github.com, docs.github.com, and netlify.com, and used each platform correctly.
4 — Agent visited all platforms but usage was incomplete or slightly incorrect.
3 — Agent visited at least two platforms and used them appropriately.
2 — Agent visited only one platform or used platforms incorrectly.
1 — Agent did not visit any required platforms.

#### C. Depth of Configuration (0.20)
Measures the level of detail and specificity in the YAML file.

5 — YAML file includes detailed configurations for each step, including environment variables and error handling.
4 — YAML file includes configurations for each step but lacks minor details.
3 — YAML file includes basic configurations for each step but lacks depth.
2 — YAML file includes configurations for only one step or is overly simplistic.
1 — YAML file lacks meaningful configuration details.

#### D. Output Structure (0.15)
Measures the organization and clarity of the final response.

5 — Final response is well-organized, includes the YAML file, and is easy to understand.
4 — Final response is organized but slightly unclear or missing minor elements.
3 — Final response is partially organized but usable.
2 — Final response is poorly organized or difficult to understand.
1 — Final response is disorganized or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "workflow_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_configuration": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "workflow_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_configuration": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "workflow_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_configuration": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())