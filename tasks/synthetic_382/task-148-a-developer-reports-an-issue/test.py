"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Diagnose and resolve an authentication error caused by using 'express-session' alongside 'passport'.
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


TASK_INSTRUCTION = """A developer reports an issue where installing 'express-session' alongside 'passport' generates an authentication error. Search Express and Passport GitHub issues, Stack Overflow discussions, and their documentation to diagnose the root cause and locate an official or community-recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves diagnosing an authentication error caused by using 'express-session' alongside 'passport'. The agent must search GitHub issues, Stack Overflow discussions, and official documentation to identify the root cause and locate a recommended fix. Success requires identifying the correct cause and providing a credible solution.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A developer reports an issue where installing 'express-session' alongside 'passport' generates an authentication error. Search Express and Passport GitHub issues, Stack Overflow discussions, and their documentation to diagnose the root cause and locate an official or community-recommended fix.

## Task-Specific Constraints
- Must visit GitHub issues for both 'express-session' and 'passport'.
- Must consult Stack Overflow discussions related to the error.
- Must reference official documentation for Express and Passport.
- Must provide a clear diagnosis of the root cause.
- Must propose a fix that is either official or widely recommended by the community.
- Output must be structured as a clear explanation with references.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to GitHub issues for both 'express-session' and 'passport'?
- Did the agent consult relevant Stack Overflow discussions?
- Did the agent reference official documentation for Express and Passport?
- Is the root cause of the error correctly identified?
- Is the proposed fix credible and sourced appropriately?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identifies the root cause of the error.

5 — Identifies the exact root cause with detailed explanation and references.
4 — Identifies the root cause but with minor gaps in explanation or references.
3 — Identifies a plausible cause but lacks sufficient detail or references.
2 — Provides a vague or incorrect cause with minimal explanation.
1 — Does not identify the root cause or provides a completely incorrect explanation.

#### B. Coverage of Required Sources (0.30)
Measures whether the agent consulted all required platforms and sources.

5 — Consults GitHub issues, Stack Overflow, and official documentation comprehensively.
4 — Consults most required sources but misses one or two minor details.
3 — Consults some required sources but misses significant platforms or details.
2 — Consults only one platform or provides minimal evidence.
1 — Does not consult any required sources.

#### C. Proposed Fix Credibility (0.25)
Measures the credibility and sourcing of the proposed fix.

5 — Provides a fix that is official or widely recommended, with clear references.
4 — Provides a credible fix but with minor gaps in references or explanation.
3 — Provides a plausible fix but lacks sufficient evidence or clarity.
2 — Provides a vague or questionable fix with minimal evidence.
1 — Does not provide a fix or provides a completely incorrect solution.

#### D. Output Structure and Clarity (0.10)
Measures the organization and clarity of the agent's response.

5 — Response is well-structured, clear, and easy to follow with proper formatting.
4 — Response is mostly clear but has minor formatting or organizational issues.
3 — Response is understandable but lacks clarity or proper formatting.
2 — Response is poorly organized and difficult to follow.
1 — Response is incoherent or completely unstructured.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "coverage_of_required_sources": <1-5>,
  "proposed_fix_credibility": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_sources": "<one sentence citing specific evidence>",
    "proposed_fix_credibility": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification_accuracy": 0.35,
    "coverage_of_required_sources": 0.30,
    "proposed_fix_credibility": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())