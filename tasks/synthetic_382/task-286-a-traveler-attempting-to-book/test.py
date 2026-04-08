"""
LLM-as-judge evaluator for EvolveBench task.

Category: Travel & Planning
Task: Diagnose the root cause of an issue preventing the addition of a second adult ticket to the Eurail cart by consulting Eurail's help center, forums, and Twitter announcements.
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


TASK_INSTRUCTION = """A traveler attempting to book tickets on the Eurail website is unable to add a second adult ticket to their cart. Find the root cause of the issue by consulting Eurail's help center, forums, and recent announcements on their Twitter page."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires diagnosing the root cause of a ticketing issue on Eurail's website. The agent must consult Eurail's help center, forums, and Twitter announcements to identify the problem and provide a clear explanation of the cause. Success is determined by the agent's ability to use the required platforms, extract relevant information, and deliver a structured and accurate explanation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
A traveler attempting to book tickets on the Eurail website is unable to add a second adult ticket to their cart. Find the root cause of the issue by consulting Eurail's help center, forums, and recent announcements on their Twitter page.

## Task-Specific Constraints
- Must visit Eurail's help center, forums, and Twitter page.
- Must identify the root cause of the issue with supporting evidence.
- Must provide a structured explanation of the findings.
- Must confirm whether the issue is technical, policy-related, or user error.
- Must reference specific announcements, FAQs, or forum posts if applicable.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Eurail's help center, forums, and Twitter page? Which ones were actually visited?
- Does the response identify the root cause of the issue clearly?
- Is the explanation structured and easy to follow?
- Are specific references to announcements, FAQs, or forum posts included?
- Does the response confirm the nature of the issue (technical, policy-related, or user error)?

### Step 2: Dimension Scoring

#### A. Root Cause Identification Accuracy (0.35)
Measures whether the agent correctly identifies the root cause of the issue.

5 — Clearly identifies the root cause with supporting evidence from multiple sources.
4 — Identifies the root cause but lacks supporting evidence from all required sources.
3 — Partially identifies the root cause but misses key details or sources.
2 — Incorrect or vague identification of the root cause.
1 — No attempt to identify the root cause.

#### B. Platform Coverage (0.30)
Measures whether the agent consulted all required platforms (help center, forums, Twitter).

5 — Consults all three platforms and uses evidence from each.
4 — Consults two platforms and uses evidence from both.
3 — Consults at least one platform and uses evidence from it.
2 — Consults one platform but fails to use evidence effectively.
1 — Does not consult any platform.

#### C. Explanation Specificity (0.20)
Measures the depth and detail of the agent's explanation.

5 — Provides detailed and specific explanations with examples and references.
4 — Provides clear explanations but lacks some specific examples or references.
3 — Provides a basic explanation with minimal detail.
2 — Provides vague or unclear explanations.
1 — Provides no explanation.

#### D. Output Structure and Credibility (0.15)
Measures whether the response is well-organized and uses credible sources.

5 — Response is well-structured, easy to follow, and cites credible sources.
4 — Response is mostly well-structured but has minor issues in organization or credibility.
3 — Response is somewhat organized but lacks credibility or clarity.
2 — Response is poorly organized or uses questionable sources.
1 — Response is disorganized and lacks credible evidence.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "root_cause_identification_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "explanation_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "root_cause_identification_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "explanation_specificity": "<one sentence citing specific evidence>",
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
    "explanation_specificity": 0.20,
    "output_structure_and_credibility": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())