"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Diagnose the root cause of recurring error messages when booking a ferry ticket from Athens to Santorini by investigating user forums, support pages, and error documentation from ferry operators.
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


TASK_INSTRUCTION = """A traveler reports difficulty booking a ferry ticket from Athens to Santorini due to recurring error messages on a Greek ferry platform. Diagnose the root cause by investigating user forums, support pages, and error documentation from the ferry operators."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose the root cause of recurring error messages when booking a ferry ticket from Athens to Santorini. The agent must investigate user forums, support pages, and error documentation from ferry operators. Successful completion requires identifying the root cause and providing actionable insights or recommendations.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A traveler reports difficulty booking a ferry ticket from Athens to Santorini due to recurring error messages on a Greek ferry platform. Diagnose the root cause by investigating user forums, support pages, and error documentation from the ferry operators.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (ferries.gr, directferries.com, reddit.com/r/travel).
- Must identify the specific error message and its context.
- Must provide actionable recommendations to resolve or circumvent the issue.
- Must cite credible sources for all claims made.
- Output must be organized as a structured list or table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify the specific error message and its context?
- Are actionable recommendations provided and do they address the root cause?
- Are all claims backed by credible sources?
- Is the output organized as a structured list or table?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identified the root cause of the error.

5 — Identifies the exact error message, its context, and underlying cause.
4 — Identifies the error message and context but provides an incomplete or partially correct cause.
3 — Identifies the error message but lacks sufficient context or cause analysis.
2 — Mentions the error vaguely without identifying the cause.
1 — Fails to identify the error message or cause.

#### B. Platform Coverage (0.30)
Measures whether the agent visited the required platforms and utilized them effectively.

5 — Uses all 3 platforms and extracts relevant information from each.
4 — Uses 2 platforms effectively and mentions the third.
3 — Uses 2 platforms but misses key information.
2 — Uses only 1 platform or extracts minimal information.
1 — Fails to use any platform effectively.

#### C. Recommendation Specificity (0.25)
Measures the quality and specificity of actionable recommendations provided.

5 — Provides 3 or more detailed, actionable recommendations addressing the root cause.
4 — Provides 2 actionable recommendations, with minor gaps in detail.
3 — Provides 1 actionable recommendation or vague suggestions.
2 — Provides recommendations that are generic or impractical.
1 — Fails to provide any recommendations.

#### D. Output Structure and Credibility (0.10)
Measures the organization of the output and credibility of sources cited.

5 — Output is well-structured, cites 3 or more credible sources.
4 — Output is structured, cites 2 credible sources.
3 — Output is minimally structured, cites 1 credible source.
2 — Output is poorly structured or lacks credible sources.
1 — Output is unstructured and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "recommendation_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "recommendation_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "platform_coverage": 0.30,
    "recommendation_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())