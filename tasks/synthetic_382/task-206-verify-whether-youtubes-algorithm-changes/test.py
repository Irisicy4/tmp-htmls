"""
LLM-as-judge evaluator for EvolveBench task.

Category: (Self) Media
Task: Verify whether YouTube’s algorithm changes in 2023 have impacted the visibility of long-form videos versus shorts.
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


TASK_INSTRUCTION = """Verify whether YouTube’s algorithm changes in 2023 have impacted the visibility of long-form videos versus shorts. Check YouTube’s official blog, recent announcements, and performance data from Social Blade."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to determine whether YouTube’s algorithm changes in 2023 have impacted the visibility of long-form videos versus shorts. The agent must gather evidence from YouTube’s official blog, recent announcements, and performance data from Social Blade. A successful completion involves synthesizing data from all required platforms and providing a structured analysis of the impact.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Verify whether YouTube’s algorithm changes in 2023 have impacted the visibility of long-form videos versus shorts. Check YouTube’s official blog, recent announcements, and performance data from Social Blade.

## Task-Specific Constraints
- Must visit YouTube’s official blog, recent announcements, and Social Blade.
- Must include specific performance metrics (e.g., views, engagement rates) for long-form videos and shorts.
- Output must compare long-form videos and shorts in terms of visibility trends.
- Must provide evidence sourced from the platforms visited.
- Output must be organized as a structured analysis (e.g., table or bullet points).
- Must address whether algorithm changes are explicitly mentioned in the sources.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to YouTube’s official blog, recent announcements, and Social Blade? Which ones were actually visited?
- Are specific performance metrics for long-form videos and shorts present in the response?
- Is the output organized as a structured analysis (e.g., table or bullet points)?
- Does the response explicitly address algorithm changes and their impact on visibility trends?
- Are sources cited and credible?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent’s output correctly and completely analyzes the impact of algorithm changes.

5 — Provides a complete and correct analysis with specific metrics and trends for both long-form videos and shorts.
4 — Mostly complete analysis with minor omissions or inaccuracies.
3 — Partial analysis with significant gaps or unclear conclusions.
2 — Poor analysis with major omissions or incorrect conclusions.
1 — No meaningful analysis provided.

#### B. Coverage of Sources (0.30)
Measures whether the agent visited and utilized all required platforms.

5 — Uses all three platforms (YouTube blog, recent announcements, Social Blade) with evidence from each.
4 — Uses two platforms with evidence from both.
3 — Uses one platform or incomplete evidence from multiple platforms.
2 — Minimal use of platforms with little evidence.
1 — No use of required platforms.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the response, including metrics, comparisons, and trends.

5 — Provides detailed metrics, comparisons, and trends for both video types.
4 — Provides metrics and comparisons but lacks some depth or trends.
3 — Provides basic metrics but lacks comparisons or trends.
2 — Minimal metrics or vague analysis.
1 — No metrics or meaningful analysis.

#### D. Source Credibility and Structure (0.10)
Measures whether sources are credible and the output is well-organized.

5 — Cites credible sources and organizes output clearly (e.g., table or bullet points).
4 — Mostly credible sources and reasonably organized output.
3 — Some credible sources but poorly organized output.
2 — Minimal credible sources and disorganized output.
1 — No credible sources and completely disorganized output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_sources": <1-5>,
  "depth_of_analysis": <1-5>,
  "source_credibility_and_structure": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_sources": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "source_credibility_and_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_sources": 0.30,
    "depth_of_analysis": 0.25,
    "source_credibility_and_structure": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())