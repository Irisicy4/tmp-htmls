"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Verify whether the latest version of Material Design guidelines recommends the use of bottom navigation bars in mobile app designs.
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


TASK_INSTRUCTION = """Verify whether the latest version of Material Design guidelines recommends the use of bottom navigation bars in mobile app designs. Check the official Material Design documentation and produce a report summarizing whether the feature is recommended, deprecated, or has specific constraints."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to verify the status of bottom navigation bars in the latest Material Design guidelines. The agent must navigate the official Material Design documentation, extract relevant information, and produce a structured report indicating whether the feature is recommended, deprecated, or has specific constraints.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether the latest version of Material Design guidelines recommends the use of bottom navigation bars in mobile app designs. Check the official Material Design documentation and produce a report summarizing whether the feature is recommended, deprecated, or has specific constraints.

## Task-Specific Constraints
- Must navigate to the official Material Design documentation (material.io).
- Must explicitly state whether bottom navigation bars are recommended, deprecated, or constrained.
- Must provide direct quotes or references from the documentation.
- Must structure the output as a clear summary report.
- Must address any constraints or conditions mentioned in the guidelines.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the official Material Design documentation (material.io)?
- Does the response explicitly state whether bottom navigation bars are recommended, deprecated, or constrained?
- Are direct quotes or references from the documentation included?
- Is the output structured as a clear summary report?
- Are any constraints or conditions mentioned in the guidelines addressed?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identifies the status of bottom navigation bars according to the latest Material Design guidelines.

5 — Clearly identifies the status (recommended, deprecated, or constrained) with direct quotes and references.
4 — Identifies the status but lacks direct quotes or references.
3 — Partially identifies the status but misses key details or constraints.
2 — Incorrect or vague identification of the status.
1 — Does not identify the status at all.

#### B. Source Coverage (0.30)
Measures whether the agent navigated the required platform and included all relevant information.

5 — Navigates material.io and includes all relevant sections of the guidelines.
4 — Navigates material.io but misses minor relevant sections.
3 — Navigates material.io but misses major relevant sections.
2 — Navigates material.io but includes little relevant information.
1 — Does not navigate material.io.

#### C. Depth and Specificity (0.20)
Measures the level of detail and specificity in the response.

5 — Includes detailed constraints, conditions, and examples from the guidelines.
4 — Includes constraints and conditions but lacks examples.
3 — Includes some constraints but lacks detail.
2 — Includes vague or minimal constraints.
1 — No constraints or details provided.

#### D. Output Structure and Credibility (0.15)
Measures the organization and credibility of the response.

5 — Response is well-organized, clearly structured, and cites credible sources.
4 — Response is organized but lacks clarity or minor credibility issues.
3 — Response is partially organized but lacks clarity or has notable credibility issues.
2 — Response is poorly organized or lacks credibility.
1 — Response is disorganized and lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "source_coverage": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())