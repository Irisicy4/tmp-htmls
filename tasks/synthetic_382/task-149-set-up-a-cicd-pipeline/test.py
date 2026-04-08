"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Set up a CI/CD pipeline on CircleCI using Docker, add steps for linting with ESLint, testing, and deploying to Netlify, and report the generated configuration file.
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


TASK_INSTRUCTION = """Set up a CI/CD pipeline on CircleCI using the publicly accessible configuration generator. Use Docker as the build environment and add steps to run linting using ESLint, execute tests, and deploy to Netlify. Report the generated configuration file."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task is to set up a CI/CD pipeline on CircleCI using Docker as the build environment. The pipeline must include steps for linting with ESLint, running tests, and deploying to Netlify. A successful completion requires the agent to generate a valid CircleCI configuration file meeting these requirements and report it.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Set up a CI/CD pipeline on CircleCI using the publicly accessible configuration generator. Use Docker as the build environment and add steps to run linting using ESLint, execute tests, and deploy to Netlify. Report the generated configuration file.

## Task-Specific Constraints
- Must use CircleCI's configuration generator to create the pipeline.
- The configuration must specify Docker as the build environment.
- Must include a linting step using ESLint.
- Must include a testing step.
- Must include a deployment step to Netlify.
- The final configuration file must be valid and complete.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent use CircleCI's configuration generator?
- Does the configuration file specify Docker as the build environment?
- Are the linting, testing, and deployment steps present in the configuration?
- Is the configuration file valid and complete?
- Did the agent provide the configuration file in its response?

### Step 2: Dimension Scoring

#### A. Configuration Completeness (0.35)
Measures whether the generated configuration file includes all required steps (linting, testing, deployment).

5 — Includes all required steps (linting, testing, deployment) with correct details.
4 — Includes most required steps but with minor omissions or errors.
3 — Includes at least one required step but is incomplete.
2 — Includes minimal steps or has major errors.
1 — No relevant steps included.

#### B. Platform Usage (0.30)
Measures whether the agent correctly used CircleCI's configuration generator and other required platforms.

5 — Correctly uses CircleCI's configuration generator and mentions all required platforms.
4 — Uses CircleCI's configuration generator but omits minor platform details.
3 — Uses CircleCI but with significant omissions or errors.
2 — Minimal or incorrect use of required platforms.
1 — No evidence of platform usage.

#### C. Configuration Validity (0.20)
Measures whether the configuration file is valid and free of syntax errors.

5 — Configuration file is valid and error-free.
4 — Configuration file is valid but contains minor warnings.
3 — Configuration file is mostly valid but has significant issues.
2 — Configuration file is invalid or incomplete.
1 — No configuration file provided.

#### D. Response Clarity (0.15)
Measures how clearly the agent presents the configuration file and related details.

5 — Response is clear, well-structured, and includes all relevant details.
4 — Response is mostly clear but lacks minor details or structure.
3 — Response is somewhat clear but has significant omissions.
2 — Response is unclear or poorly structured.
1 — Response is incomprehensible or missing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "configuration_completeness": <1-5>,
  "platform_usage": <1-5>,
  "configuration_validity": <1-5>,
  "response_clarity": <1-5>,
  "dimension_reasoning": {{
    "configuration_completeness": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "configuration_validity": "<one sentence citing specific evidence>",
    "response_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "configuration_completeness": 0.35,
    "platform_usage": 0.30,
    "configuration_validity": 0.20,
    "response_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())