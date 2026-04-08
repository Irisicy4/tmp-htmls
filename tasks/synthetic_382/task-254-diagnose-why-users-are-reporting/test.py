"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Diagnose issues with Eurail pass purchases via PayPal and recommend solutions.
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


TASK_INSTRUCTION = """Diagnose why users are reporting errors when purchasing Eurail passes through PayPal. Search Eurail's FAQ section, PayPal's help center, and Reddit's r/travel for potential issues and resolutions. Provide the root cause, affected users, and recommended solution paths with links."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose issues with Eurail pass purchases via PayPal by researching Eurail's FAQ section, PayPal's help center, and Reddit's r/travel. The agent must provide the root cause, identify affected users, and recommend solutions with supporting links.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why users are reporting errors when purchasing Eurail passes through PayPal. Search Eurail's FAQ section, PayPal's help center, and Reddit's r/travel for potential issues and resolutions. Provide the root cause, affected users, and recommended solution paths with links.

## Task-Specific Constraints
- Must visit Eurail's FAQ section, PayPal's help center, and Reddit's r/travel.
- Must identify the root cause of the issue clearly.
- Must specify which types of users are affected by the issue.
- Must recommend actionable solution paths with supporting links.
- Output must be structured as a clear list or table.
- Must include evidence or citations for claims made.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Eurail's FAQ section, PayPal's help center, and Reddit's r/travel? Which ones were actually visited?
- Does the response clearly identify the root cause of the issue?
- Are affected user groups specified in the response?
- Are actionable solution paths provided with supporting links?
- Is the output organized as a clear list or table?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent correctly identified the root cause, affected users, and provided actionable solutions.

5 — Identifies the root cause, affected users, and provides 3+ actionable solutions with supporting links.
4 — Identifies the root cause, affected users, and provides 2 actionable solutions with supporting links.
3 — Identifies the root cause and affected users but provides only 1 actionable solution or lacks supporting links.
2 — Partially identifies the root cause or affected users; solutions are vague or unsupported.
1 — Fails to identify the root cause or affected users; no actionable solutions provided.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Effectively uses Eurail's FAQ, PayPal's help center, and Reddit's r/travel, citing evidence from all three.
4 — Uses at least two of the required platforms effectively, citing evidence.
3 — Uses at least one platform effectively, citing evidence.
2 — Visits platforms but fails to use them effectively or cite evidence.
1 — Fails to visit any required platforms.

#### C. Depth of Analysis (0.25)
Measures the level of detail and specificity in the agent's findings and recommendations.

5 — Provides detailed findings with specific examples, numbers, or comparisons.
4 — Provides moderately detailed findings with some examples or comparisons.
3 — Provides basic findings with minimal detail or examples.
2 — Findings are vague or lack specificity.
1 — Findings are absent or completely incorrect.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the agent's response.

5 — Output is well-organized as a clear list or table, with credible sources cited.
4 — Output is mostly organized, with credible sources cited.
3 — Output is somewhat organized, with limited citations.
2 — Output is disorganized or lacks credible sources.
1 — Output is completely disorganized and lacks citations.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())