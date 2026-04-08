"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Research the root cause of middleware breaking in Express v5 and provide a migration strategy.
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


TASK_INSTRUCTION = """A developer reports that upgrading a Node.js project to use Express v5 leads to middleware breaking with 'unexpected arguments' errors. Research the issue across Express GitHub issues, the changelog, and Stack Overflow, and identify the root cause and recommended migration strategy."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating middleware issues caused by upgrading to Express v5. The agent must research the problem using Express GitHub issues, the changelog, and Stack Overflow, and provide a clear explanation of the root cause and a recommended migration strategy. The domain is software engineering, specifically Node.js and Express.js.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A developer reports that upgrading a Node.js project to use Express v5 leads to middleware breaking with 'unexpected arguments' errors. Research the issue across Express GitHub issues, the changelog, and Stack Overflow, and identify the root cause and recommended migration strategy.

## Task-Specific Constraints
- Must visit Express GitHub issues, the changelog, and Stack Overflow.
- Must identify the root cause of the middleware issue.
- Must provide a clear migration strategy for Express v5.
- Output must include specific references to sources (e.g., GitHub issue numbers, changelog sections, Stack Overflow links).
- Must explain any breaking changes in Express v5 middleware handling.
- Output must be structured as a clear and organized explanation.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Express GitHub issues, the changelog, and Stack Overflow? Which ones were actually visited?
- Does the response identify the root cause of the middleware issue?
- Does the response include specific references to sources (e.g., GitHub issue numbers, changelog sections, Stack Overflow links)?
- Is the migration strategy clear and actionable?
- Are breaking changes in Express v5 middleware handling explained?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent correctly identifies the root cause of the middleware issue.

5 — Clearly identifies the root cause with specific references to sources.
4 — Identifies the root cause but lacks some specificity or references.
3 — Partially identifies the root cause but is incomplete or unclear.
2 — Incorrect or vague identification of the root cause.
1 — Fails to identify the root cause.

#### B. Migration Strategy Completeness (0.30)
Measures whether the agent provides a clear and actionable migration strategy.

5 — Provides a detailed migration strategy addressing all key issues.
4 — Provides a migration strategy but lacks minor details or clarity.
3 — Provides a partial migration strategy with significant omissions.
2 — Migration strategy is vague or mostly missing.
1 — Fails to provide a migration strategy.

#### C. Source Utilization (0.20)
Measures whether the agent uses the required platforms and references sources effectively.

5 — Uses all required platforms and provides specific references to sources.
4 — Uses most platforms and provides references, but misses minor details.
3 — Uses some platforms but lacks sufficient references or specificity.
2 — Uses few platforms and provides vague or no references.
1 — Fails to use required platforms or provide references.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to understand.

5 — Output is structured, clear, and logically organized.
4 — Output is mostly clear but has minor organizational issues.
3 — Output is partially clear but lacks structure or organization.
2 — Output is unclear or poorly organized.
1 — Output is completely disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "migration_strategy_completeness": <1-5>,
  "source_utilization": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "migration_strategy_completeness": "<one sentence citing specific evidence>",
    "source_utilization": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "migration_strategy_completeness": 0.30,
    "source_utilization": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())