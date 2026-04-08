"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Diagnose why `vite` fails to build with `vue3` and `@vue/compiler-sfc`, identify the root cause, affected version range, and recommend a fix.
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


TASK_INSTRUCTION = """Diagnose why `vite` fails to build when paired with `vue3` and `@vue/compiler-sfc` in a specific setup. Research issues on the Vite GitHub issue tracker, the Vue documentation, and Stack Overflow. Identify the root cause, affected version range, and recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing a build failure in a software engineering context. The agent must investigate the issue by consulting the Vite GitHub issue tracker, Vue documentation, and Stack Overflow. A successful completion requires identifying the root cause, the affected version range, and a recommended fix.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why `vite` fails to build when paired with `vue3` and `@vue/compiler-sfc` in a specific setup. Research issues on the Vite GitHub issue tracker, the Vue documentation, and Stack Overflow. Identify the root cause, affected version range, and recommended fix.

## Task-Specific Constraints
- Must visit at least 3 specified platforms: vite.dev, github.com, and stackoverflow.com.
- Must identify the root cause of the build failure with specific technical details.
- Must specify the affected version range for `vite`, `vue3`, and `@vue/compiler-sfc`.
- Must provide a clear, actionable recommendation for fixing the issue.
- Output must be structured as a concise, well-organized explanation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to all required platforms (vite.dev, github.com, stackoverflow.com)?
- Did the agent identify the root cause of the build failure with sufficient technical detail?
- Did the agent specify the affected version range for all relevant packages?
- Did the agent provide a clear and actionable recommendation for fixing the issue?
- Is the output well-organized and easy to follow?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent correctly identified the root cause of the build failure.

5 — Correctly identifies the root cause with detailed technical explanation.
4 — Correctly identifies the root cause but with limited technical detail.
3 — Identifies the root cause but lacks clarity or technical depth.
2 — Incorrect or vague identification of the root cause.
1 — Fails to identify the root cause.

#### B. Version Range Coverage (0.30)
Measures whether the agent specifies the affected version range for all relevant packages.

5 — Specifies the version range for all three packages with evidence.
4 — Specifies the version range for two packages with evidence.
3 — Specifies the version range for at least one package with evidence.
2 — Mentions versions but lacks evidence or specificity.
1 — Does not mention version ranges.

#### C. Recommendation Quality (0.25)
Measures the quality and clarity of the recommended fix.

5 — Provides a clear, actionable fix with supporting evidence.
4 — Provides a clear fix but lacks supporting evidence.
3 — Provides a fix but lacks clarity or evidence.
2 — Fix is vague or impractical.
1 — No fix provided.

#### D. Platform Usage and Organization (0.10)
Measures whether the agent used the required platforms and organized the output well.

5 — Uses all three platforms and organizes output clearly.
4 — Uses two platforms and organizes output clearly.
3 — Uses at least one platform and output is somewhat organized.
2 — Uses platforms but output is poorly organized.
1 — Does not use required platforms or output is disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "version_range_coverage": <1-5>,
  "recommendation_quality": <1-5>,
  "platform_usage_and_organization": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "version_range_coverage": "<one sentence citing specific evidence>",
    "recommendation_quality": "<one sentence citing specific evidence>",
    "platform_usage_and_organization": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "version_range_coverage": 0.30,
    "recommendation_quality": 0.25,
    "platform_usage_and_organization": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())