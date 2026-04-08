"""
LLM-as-judge evaluator for EvolveBench task.

Category: Data & ML Engineering
Task: Investigate and resolve GPU memory issues during LLM finetuning using Hugging Face Transformers.
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


TASK_INSTRUCTION = """A user reports that finetuning a large language model (LLM) using Hugging Face Transformers on GPUs results in an 'out of memory' error. Investigate GPU memory requirements and solutions by reviewing Hugging Face documentation, forums, and GitHub issues. Identify the root cause, affected configurations, and a recommended fix."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves investigating GPU memory issues during LLM finetuning using Hugging Face Transformers. The agent must identify the root cause, affected configurations, and recommend a fix. A successful completion requires consulting Hugging Face documentation, forums, and GitHub issues, and providing a clear, actionable solution.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A user reports that finetuning a large language model (LLM) using Hugging Face Transformers on GPUs results in an 'out of memory' error. Investigate GPU memory requirements and solutions by reviewing Hugging Face documentation, forums, and GitHub issues. Identify the root cause, affected configurations, and a recommended fix.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (huggingface.co, discuss.huggingface.co, github.com).
- Must identify the root cause of the 'out of memory' error.
- Must specify which configurations (e.g., model size, batch size) are affected.
- Must recommend at least one actionable fix (e.g., gradient checkpointing, reducing batch size).
- Output must be structured as a clear, actionable list or explanation.
- Must cite sources for any claims or recommendations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Did the agent identify the root cause of the 'out of memory' error?
- Did the agent specify affected configurations (e.g., model size, batch size)?
- Did the agent recommend at least one actionable fix? Was it relevant and feasible?
- Is the output structured as a clear, actionable list or explanation?

### Step 2: Dimension Scoring

#### A. Root Cause Identification (0.35)
Measures whether the agent correctly identified the root cause of the issue.

5 — Clearly identifies the root cause with detailed explanation and supporting evidence.
4 — Identifies the root cause but lacks some detail or evidence.
3 — Identifies a plausible root cause but with minimal explanation or evidence.
2 — Provides an incorrect or incomplete root cause.
1 — Fails to identify any root cause.

#### B. Coverage of Platforms and Sources (0.30)
Measures whether the agent consulted the required platforms and cited credible sources.

5 — Consults all three platforms and cites multiple credible sources.
4 — Consults at least two platforms and cites credible sources.
3 — Consults at least one platform and cites some sources.
2 — Consults platforms but fails to cite sources or uses non-credible sources.
1 — Fails to consult any platforms or cite sources.

#### C. Actionable Fixes (0.25)
Measures whether the agent provided actionable and relevant solutions.

5 — Recommends multiple actionable fixes with detailed explanations.
4 — Recommends at least one actionable fix with some explanation.
3 — Recommends at least one fix but lacks detail or relevance.
2 — Provides a vague or impractical fix.
1 — Fails to recommend any fixes.

#### D. Output Structure and Clarity (0.10)
Measures whether the response is well-structured and easy to understand.

5 — Output is highly organized, clear, and easy to follow.
4 — Output is mostly organized and clear.
3 — Output is somewhat organized but may be hard to follow in places.
2 — Output is poorly organized or unclear.
1 — Output is completely disorganized or unreadable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification": <1-5>,
  "coverage_of_platforms_and_sources": <1-5>,
  "actionable_fixes": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification": "<one sentence citing specific evidence>",
    "coverage_of_platforms_and_sources": "<one sentence citing specific evidence>",
    "actionable_fixes": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "root_cause_identification": 0.35,
    "coverage_of_platforms_and_sources": 0.30,
    "actionable_fixes": 0.25,
    "output_structure_and_clarity": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())