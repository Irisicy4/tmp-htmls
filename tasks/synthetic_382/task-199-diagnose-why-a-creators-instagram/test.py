"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Diagnose why a creator’s Instagram Reels reach has dropped by 50% in the past week and recommend fixes.
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


TASK_INSTRUCTION = """Diagnose why a creator’s Instagram Reels reach has dropped by 50% in the past week. Investigate platform policies, algorithm updates, and engagement patterns using Instagram’s blog, Social Media Today, and Reddit forums. Identify the root cause, reach trends, and recommended fixes based on publicly available information."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to diagnose the reasons behind a significant drop in Instagram Reels reach for a creator, using information from Instagram’s blog, Social Media Today, and Reddit forums. A successful completion involves identifying the root cause, analyzing reach trends, and recommending fixes based on credible, publicly available information.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Diagnose why a creator’s Instagram Reels reach has dropped by 50% in the past week. Investigate platform policies, algorithm updates, and engagement patterns using Instagram’s blog, Social Media Today, and Reddit forums. Identify the root cause, reach trends, and recommended fixes based on publicly available information.

## Task-Specific Constraints
- Must visit Instagram’s blog, Social Media Today, and Reddit forums.
- Must identify at least one platform policy or algorithm update relevant to Instagram Reels.
- Must analyze engagement metrics or patterns based on credible sources.
- Output must include actionable recommendations for improving reach.
- Response must be structured as a clear, organized list or table.
- Must cite sources explicitly for all claims.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Instagram’s blog, Social Media Today, and Reddit forums? Which ones were actually visited?
- Did the agent identify a platform policy or algorithm update relevant to Instagram Reels?
- Are engagement metrics or patterns analyzed using credible sources?
- Are actionable recommendations for improving reach included in the response?
- Is the output structured as a clear, organized list or table?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly diagnosed the root cause and provided actionable recommendations.

5 — Identifies root cause(s) accurately and provides 3+ actionable recommendations.
4 — Identifies root cause(s) accurately and provides 2 actionable recommendations.
3 — Identifies root cause(s) but recommendations are vague or incomplete.
2 — Misidentifies root cause(s) or provides only 1 vague recommendation.
1 — Fails to identify root cause(s) or provide recommendations.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent used all required platforms and sources.

5 — Uses Instagram’s blog, Social Media Today, and Reddit forums, citing all explicitly.
4 — Uses 2 of the 3 required platforms, citing most explicitly.
3 — Uses 1 required platform or cites sources vaguely.
2 — Navigates platforms but fails to extract relevant information.
1 — Does not use any required platforms.

#### C. Depth and Specificity (0.20)
Measures whether the agent provides detailed analysis and specific evidence.

5 — Includes detailed analysis with specific metrics, trends, or examples.
4 — Includes analysis with some metrics or trends but lacks depth.
3 — Provides general analysis with minimal specificity.
2 — Analysis is vague or unsupported by evidence.
1 — No meaningful analysis provided.

#### D. Source Credibility and Output Structure (0.15)
Measures whether the response is well-organized and cites credible sources.

5 — Response is well-structured and cites credible sources explicitly.
4 — Response is mostly well-structured and cites sources implicitly.
3 — Response is somewhat organized but lacks clear citations.
2 — Response is poorly organized or sources are unclear.
1 — Response is disorganized and lacks credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_and_specificity": <1-5>,
  "source_credibility_and_output_structure": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "source_credibility_and_output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_and_specificity": 0.20,
    "source_credibility_and_output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())