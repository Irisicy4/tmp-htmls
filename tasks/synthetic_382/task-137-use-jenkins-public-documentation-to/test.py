"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Configure a CI/CD pipeline using Jenkins to build and test a Python application with pytest.
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


TASK_INSTRUCTION = """Use Jenkins' public documentation to configure a sample CI/CD pipeline that builds and tests a Python application using pytest. Navigate the pipeline setup documentation and identify necessary steps for implementing build triggers, testing stages, and reporting results. Report the final pipeline configuration steps you would follow."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use Jenkins documentation to configure a CI/CD pipeline for a Python application. The pipeline must include build triggers, testing stages using pytest, and reporting results. The domain is software engineering, and a successful completion involves providing a clear and complete set of pipeline configuration steps derived from the documentation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Jenkins' public documentation to configure a sample CI/CD pipeline that builds and tests a Python application using pytest. Navigate the pipeline setup documentation and identify necessary steps for implementing build triggers, testing stages, and reporting results. Report the final pipeline configuration steps you would follow.

## Task-Specific Constraints
- Must visit www.jenkins.io and docs.pytest.org for relevant documentation.
- Must include steps for configuring build triggers in Jenkins.
- Must include steps for integrating pytest into the testing stage.
- Must describe how results are reported in the pipeline.
- Output must be organized as a clear, step-by-step configuration guide.
- Must reference specific sections or examples from the documentation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to www.jenkins.io and docs.pytest.org? Were these platforms used effectively?
- Are the steps for configuring build triggers in Jenkins present and accurate?
- Are the steps for integrating pytest into the testing stage complete and correct?
- Does the response describe how results are reported in the pipeline?
- Is the output organized as a clear, step-by-step configuration guide?

### Step 2: Dimension Scoring

#### A. Pipeline Configuration Accuracy (0.35)
Measures whether the pipeline setup steps are correct and complete.

5 — Includes all required steps (build triggers, pytest integration, result reporting) with accurate details.
4 — Includes most required steps with minor inaccuracies or omissions.
3 — Includes some required steps but lacks completeness or has notable inaccuracies.
2 — Includes few required steps and has significant inaccuracies.
1 — Does not include any correct steps.

#### B. Platform Usage and Coverage (0.30)
Measures whether the agent used the required platforms and referenced relevant documentation.

5 — Effectively uses both www.jenkins.io and docs.pytest.org, referencing specific sections/examples.
4 — Uses both platforms but references are less specific or incomplete.
3 — Uses at least one platform effectively but misses the other.
2 — Uses one platform minimally or ineffectively.
1 — Does not use either platform.

#### C. Detail and Specificity (0.20)
Measures the depth and specificity of the response.

5 — Provides detailed steps with references to specific documentation sections/examples.
4 — Provides detailed steps but lacks some references or specificity.
3 — Provides general steps but lacks detail or references.
2 — Provides vague or incomplete steps.
1 — Provides no meaningful details.

#### D. Output Organization and Clarity (0.15)
Measures how well the response is structured and organized.

5 — Output is a clear, step-by-step configuration guide with excellent formatting.
4 — Output is mostly clear and organized but has minor formatting issues.
3 — Output is usable but lacks clarity or organization.
2 — Output is poorly organized or difficult to follow.
1 — Output is completely disorganized or unclear.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "pipeline_configuration_accuracy": <1-5>,
  "platform_usage_and_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_organization_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "pipeline_configuration_accuracy": "<one sentence citing specific evidence>",
    "platform_usage_and_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_organization_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "pipeline_configuration_accuracy": 0.35,
    "platform_usage_and_coverage": 0.30,
    "detail_and_specificity": 0.20,
    "output_organization_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())