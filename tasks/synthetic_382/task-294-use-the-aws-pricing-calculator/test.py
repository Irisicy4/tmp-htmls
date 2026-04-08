"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Use the AWS Pricing Calculator to estimate the cost of training a machine learning model with specific GPU and time requirements.
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


TASK_INSTRUCTION = """Use the AWS Pricing Calculator, enter specifications for training a machine learning model with 2 NVIDIA A100 GPUs for 100 training hours, and select necessary compute and storage options. Complete the workflow and report the estimated total cost shown at the final screen."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to use the AWS Pricing Calculator to estimate the cost of training a machine learning model with 2 NVIDIA A100 GPUs for 100 training hours. The agent must select appropriate compute and storage options and report the total cost shown at the end of the workflow.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use the AWS Pricing Calculator, enter specifications for training a machine learning model with 2 NVIDIA A100 GPUs for 100 training hours, and select necessary compute and storage options. Complete the workflow and report the estimated total cost shown at the final screen.

## Task-Specific Constraints
- Must use the AWS Pricing Calculator platform.
- Must specify 2 NVIDIA A100 GPUs in the configuration.
- Must set the training duration to 100 hours.
- Must select appropriate compute and storage options.
- Must report the total cost as shown on the final screen.
- The response must include the total cost in a structured format.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the AWS Pricing Calculator platform?
- Did the agent configure 2 NVIDIA A100 GPUs and set the training duration to 100 hours?
- Did the agent select appropriate compute and storage options?
- Is the total cost reported, and is it presented in a structured format?
- Are there any errors or omissions in the agent's response?

### Step 2: Dimension Scoring

#### A. Configuration Accuracy (0.35)
Measures whether the agent correctly configured the GPU type, count, and training duration.

5 — Correctly configured 2 NVIDIA A100 GPUs and 100 training hours with no errors.
4 — Minor errors in configuration but still includes 2 NVIDIA A100 GPUs and 100 training hours.
3 — Partially correct configuration (e.g., wrong GPU type or duration).
2 — Mostly incorrect configuration with significant errors.
1 — No attempt to configure the task.

#### B. Platform Usage (0.30)
Measures whether the agent used the AWS Pricing Calculator platform as required.

5 — Successfully navigated and used the AWS Pricing Calculator platform.
4 — Used the platform but with minor navigation issues.
3 — Attempted to use the platform but with significant errors or incomplete actions.
2 — Minimal evidence of platform usage.
1 — Did not use the platform at all.

#### C. Cost Reporting (0.20)
Measures whether the agent correctly reported the total cost in a structured format.

5 — Total cost is reported accurately and in a structured format.
4 — Total cost is reported but with minor formatting issues.
3 — Total cost is reported but lacks structure or contains minor inaccuracies.
2 — Total cost is incomplete or significantly inaccurate.
1 — Total cost is not reported.

#### D. Output Structure and Clarity (0.15)
Measures the clarity and organization of the agent's final response.

5 — Response is well-organized, clear, and easy to understand.
4 — Response is mostly clear with minor organizational issues.
3 — Response is somewhat clear but lacks proper organization.
2 — Response is poorly structured or unclear.
1 — Response is completely disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "configuration_accuracy": <1-5>,
  "platform_usage": <1-5>,
  "cost_reporting": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "configuration_accuracy": "<one sentence citing specific evidence>",
    "platform_usage": "<one sentence citing specific evidence>",
    "cost_reporting": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "configuration_accuracy": 0.35,
    "platform_usage": 0.30,
    "cost_reporting": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())