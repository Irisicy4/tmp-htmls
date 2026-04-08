"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether Twitter (X.com) and Instagram prioritize video content over text/image posts in their engagement algorithms.
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


TASK_INSTRUCTION = """Verify whether Twitter (X.com) and Instagram currently favor video content over text/image posts in their engagement algorithms. Search for official announcements, recent reports from analytics blogs, and public metrics about content reach trends from each platform."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to determine whether Twitter (X.com) and Instagram prioritize video content over text/image posts in their engagement algorithms. The agent must search for official announcements, recent reports from analytics blogs, and public metrics about content reach trends from each platform. A successful completion involves gathering evidence from credible sources and presenting a structured, accurate summary.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether Twitter (X.com) and Instagram currently favor video content over text/image posts in their engagement algorithms. Search for official announcements, recent reports from analytics blogs, and public metrics about content reach trends from each platform.

## Task-Specific Constraints
- Must visit at least 3 credible sources, including at least one official platform source (e.g., Twitter blog, Instagram blog).
- Must include data or metrics about content reach or engagement trends.
- Must mention whether video content is prioritized on both platforms.
- Output must be structured as a clear, concise summary.
- Must cite sources explicitly.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are engagement trends or metrics for both Twitter and Instagram included?
- Does the response explicitly state whether video content is prioritized on both platforms?
- Are sources cited explicitly and are they credible?
- Is the output organized as a clear, concise summary?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly identifies whether video content is prioritized on both platforms.

5 — Correctly identifies prioritization status for both platforms with supporting evidence.
4 — Correctly identifies prioritization status for both platforms but lacks some supporting evidence.
3 — Identifies prioritization status for one platform but not the other, or lacks sufficient evidence.
2 — Incorrect or vague conclusions about prioritization status.
1 — Does not address prioritization status at all.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent visited the required platforms and included data from credible sources.

5 — Visits at least 3 credible sources, including at least one official platform source, and cites them explicitly.
4 — Visits at least 3 sources but may lack an official platform source or explicit citations.
3 — Visits fewer than 3 sources or includes unclear citations.
2 — Visits only 1 source or includes no credible sources.
1 — Does not visit any relevant sources.

#### C. Depth of Evidence (0.20)
Measures the level of detail in the evidence provided, such as metrics or trends.

5 — Includes detailed metrics or trends for both platforms with clear comparisons.
4 — Includes some metrics or trends but lacks detail or comparison.
3 — Includes vague or incomplete evidence for one or both platforms.
2 — Minimal evidence with no metrics or trends.
1 — No evidence provided.

#### D. Output Structure and Clarity (0.15)
Measures whether the response is well-organized and easy to understand.

5 — Response is clear, concise, and well-structured with proper formatting.
4 — Response is mostly clear but may lack some structure or conciseness.
3 — Response is somewhat unclear or poorly structured but understandable.
2 — Response is difficult to follow or poorly formatted.
1 — Response is incoherent or disorganized.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "depth_of_evidence": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "depth_of_evidence": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "depth_of_evidence": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())