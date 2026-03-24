"""
LLM-as-judge evaluator for EvolveBench task.

Category: Finance & Economics
Task: Research and compare three popular online brokers (Schwab, Fidelity, and Interactive Brokers) on trading fees, account minimums, types of investments offered, and customer support options.
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


TASK_INSTRUCTION = """Research and compare three popular online brokers (e.g., Schwab, Fidelity, and Interactive Brokers) for trading stocks in the United States. Specifically, compare them on trading fees, account minimums, types of investments offered, and customer support options. Use their official websites and at least one independent review platform to verify claims and identify differences."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves researching and comparing three popular online brokers (Schwab, Fidelity, and Interactive Brokers) on trading fees, account minimums, types of investments offered, and customer support options. The agent must use official websites and at least one independent review platform to verify claims and provide structured comparisons.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Research and compare three popular online brokers (e.g., Schwab, Fidelity, and Interactive Brokers) for trading stocks in the United States. Specifically, compare them on trading fees, account minimums, types of investments offered, and customer support options. Use their official websites and at least one independent review platform to verify claims and identify differences.

## Task-Specific Constraints
- Must visit the official websites of Schwab, Fidelity, and Interactive Brokers.
- Must use at least one independent review platform (e.g., NerdWallet).
- Must include specific data for trading fees, account minimums, types of investments, and customer support options.
- Output must be organized as a structured table or list for easy comparison.
- Must verify claims using credible sources and highlight any discrepancies.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are trading fees, account minimums, types of investments, and customer support options present in the response?
- Is the output organized as a structured table or list for comparison?
- Did the agent use at least one independent review platform to verify claims?
- Are any discrepancies or errors in the data highlighted and sourced?

### Step 2: Dimension Scoring

#### A. Comparison Accuracy (0.35)
Measures how correct and complete the comparisons are across trading fees, account minimums, types of investments, and customer support.

5 — All four comparison criteria are accurate, complete, and sourced.
4 — Three criteria are accurate and complete; one is partially complete or slightly inaccurate.
3 — At least two criteria are accurate and complete; others are missing or incomplete.
2 — Only one criterion is accurate and complete; others are mostly missing or incorrect.
1 — None of the criteria are accurate or complete.

#### B. Coverage of Required Platforms (0.30)
Measures whether the agent visited all required platforms and used at least one independent review source.

5 — All three broker platforms and one independent review platform were used.
4 — Two broker platforms and one independent review platform were used.
3 — At least two broker platforms were used; independent review platform may be missing.
2 — Only one broker platform was used; independent review platform missing.
1 — No required platforms were visited.

#### C. Detail and Specificity (0.25)
Measures the depth of information provided, including specific numbers, comparisons, and discrepancies.

5 — Provides detailed numbers for all criteria, highlights discrepancies, and includes comparisons.
4 — Provides detailed numbers for most criteria and includes comparisons.
3 — Provides some numbers and comparisons; lacks depth in one or more areas.
2 — Provides minimal numbers or comparisons; lacks depth in multiple areas.
1 — Provides no numbers or comparisons.

#### D. Output Structure and Credibility (0.10)
Measures whether the output is well-organized and sourced from credible platforms.

5 — Output is structured as a clear table or list and all sources are credible.
4 — Output is structured well but has minor issues in sourcing or organization.
3 — Output is usable but poorly organized or missing some sourcing.
2 — Output is disorganized and lacks credible sourcing.
1 — Output is unusable and lacks credible sourcing.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "comparison_accuracy": <1-5>,
  "coverage_of_required_platforms": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "comparison_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_platforms": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "comparison_accuracy": 0.35,
    "coverage_of_required_platforms": 0.30,
    "detail_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())