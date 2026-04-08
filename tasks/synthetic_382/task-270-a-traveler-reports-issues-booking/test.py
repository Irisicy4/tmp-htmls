"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Investigate booking issues for a France-Spain Eurail pass and suggest solutions.
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


TASK_INSTRUCTION = """A traveler reports issues booking tickets through Eurail's website for a France-Spain pass. Investigate the root cause by checking Eurail's official support page, recent community forum threads, and social media updates from Eurail. Identify the problem and suggest solutions."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating booking issues for a France-Spain Eurail pass. The agent must identify the root cause by consulting Eurail's official support page, community forums, and social media updates. A successful completion includes identifying the problem and suggesting actionable solutions.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A traveler reports issues booking tickets through Eurail's website for a France-Spain pass. Investigate the root cause by checking Eurail's official support page, recent community forum threads, and social media updates from Eurail. Identify the problem and suggest solutions.

## Task-Specific Constraints
- Must visit Eurail's official support page, Reddit's r/travel forum, and Eurail's Twitter account.
- Must identify the root cause of the booking issue.
- Must suggest at least two actionable solutions.
- Must provide evidence or citations for claims made.
- Output must be structured as a clear list or summary.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Eurail's official support page, Reddit's r/travel forum, and Eurail's Twitter account?
- Does the response identify the root cause of the booking issue?
- Are at least two actionable solutions provided?
- Are claims supported with evidence or citations?
- Is the output structured as a clear list or summary?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identifies the root cause of the booking issue.

5 — Clearly identifies the root cause with supporting evidence from at least two platforms.
4 — Identifies the root cause but lacks supporting evidence from one platform.
3 — Partially identifies the root cause; evidence is incomplete or unclear.
2 — Incorrect or vague identification of the root cause.
1 — Fails to identify the root cause entirely.

#### B. Platform Coverage (0.30)
Measures whether the agent uses all required platforms to gather information.

5 — Uses all three platforms (Eurail support, Reddit, Twitter) with detailed findings.
4 — Uses two platforms with detailed findings; third platform is partially covered.
3 — Uses two platforms but findings are incomplete or unclear.
2 — Uses only one platform or findings are very limited.
1 — Fails to use any of the required platforms.

#### C. Solution Specificity (0.25)
Measures the quality and specificity of the suggested solutions.

5 — Provides at least two actionable, detailed solutions with supporting evidence.
4 — Provides two solutions but lacks supporting evidence for one.
3 — Provides one actionable solution or two vague solutions.
2 — Provides one vague solution or incorrect suggestions.
1 — Fails to provide any solutions.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the response.

5 — Output is well-structured and cites credible sources for all claims.
4 — Output is structured but lacks citations for one claim.
3 — Output is partially structured or lacks citations for multiple claims.
2 — Output is poorly structured and lacks credibility.
1 — Output is disorganized and entirely lacks credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "solution_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "solution_specificity": "<one sentence citing specific evidence>",
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
    "solution_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())