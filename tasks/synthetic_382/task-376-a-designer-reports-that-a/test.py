"""
LLM-as-judge evaluator for EvolveBench task.

Category: Design
Task: Diagnose the cause of SVG rendering issues in Safari and recommend a fix.
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


TASK_INSTRUCTION = """A designer reports that a specific SVG graphic exported from Adobe Illustrator displays incorrectly on Safari browsers but works fine in Chrome. Diagnose the cause by researching relevant forums, Safari documentation, and community threads. Provide the root cause, affected version range, and recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing SVG rendering issues in Safari and recommending a fix. The agent must research forums, Safari documentation, and community threads to identify the root cause, affected version range, and a recommended solution. A successful completion includes accurate identification of the issue, relevant version details, and a clear fix.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A designer reports that a specific SVG graphic exported from Adobe Illustrator displays incorrectly on Safari browsers but works fine in Chrome. Diagnose the cause by researching relevant forums, Safari documentation, and community threads. Provide the root cause, affected version range, and recommended fix.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms: forums.adobe.com, developer.apple.com, stackoverflow.com.
- Must identify the root cause of the SVG rendering issue.
- Must specify the affected Safari version range.
- Must provide a clear, actionable fix or workaround.
- Output must be structured as a clear list or table.
- Must cite sources for claims made in the response.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Does the response identify the root cause of the SVG rendering issue?
- Does the response specify the affected Safari version range?
- Is the recommended fix actionable and clear?
- Are sources cited for claims made in the response?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent accurately identified the root cause of the SVG rendering issue.

5 — Accurately identifies the root cause with detailed explanation and supporting evidence.
4 — Identifies the root cause but lacks detailed explanation or evidence.
3 — Provides a plausible root cause but lacks clarity or evidence.
2 — Provides an incorrect or vague root cause.
1 — Fails to identify any root cause.

#### B. Platform Coverage (0.30)
Measures whether the agent visited the required platforms and utilized relevant information.

5 — Visits all 3 specified platforms and uses relevant information from each.
4 — Visits 2 platforms and uses relevant information from them.
3 — Visits at least 1 platform and uses some relevant information.
2 — Visits platforms but fails to use relevant information.
1 — Does not visit any specified platforms.

#### C. Version Range Specificity (0.20)
Measures whether the agent specifies the affected Safari version range.

5 — Clearly specifies the affected version range with supporting evidence.
4 — Specifies the version range but lacks supporting evidence.
3 — Provides a plausible version range but lacks clarity or evidence.
2 — Provides an incorrect or vague version range.
1 — Fails to specify any version range.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and cites credible sources.

5 — Output is structured clearly and cites credible sources for all claims.
4 — Output is mostly clear and cites credible sources for most claims.
3 — Output is somewhat clear and cites some credible sources.
2 — Output is poorly organized and lacks credible sources.
1 — Output is disorganized and lacks any credible sources.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "platform_coverage": <1-5>,
  "version_range_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "version_range_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "platform_coverage": 0.30,
    "version_range_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())