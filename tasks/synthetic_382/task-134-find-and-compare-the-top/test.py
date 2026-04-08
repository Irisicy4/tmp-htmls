"""
LLM-as-judge evaluator for EvolveBench task.

Category: Software Engineering
Task: Compare the top 3 JavaScript frameworks for single-page applications based on performance, community size, and documentation.
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


TASK_INSTRUCTION = """Find and compare the top 3 JavaScript frameworks for building single-page applications. Evaluate them based on performance benchmarks, community size (GitHub stars, active contributors), and available documentation. Provide a summary table with these criteria."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to identify and compare the top 3 JavaScript frameworks for single-page applications. The agent must evaluate them based on performance benchmarks, community size (GitHub stars, active contributors), and available documentation. A successful completion includes a structured summary table covering these criteria.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find and compare the top 3 JavaScript frameworks for building single-page applications. Evaluate them based on performance benchmarks, community size (GitHub stars, active contributors), and available documentation. Provide a summary table with these criteria.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (github.com, stackoverflow.com, developer.mozilla.org).
- Must include performance benchmark data for all frameworks compared.
- Must include community size metrics (GitHub stars, active contributors) for all frameworks.
- Must evaluate the quality and availability of documentation for all frameworks.
- Output must be organized as a structured summary table.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are performance benchmark data present for all frameworks?
- Are community size metrics (GitHub stars, active contributors) included for all frameworks?
- Is the quality and availability of documentation evaluated for all frameworks?
- Is the output organized as a structured summary table?

### Step 2: Dimension Scoring

#### A. Framework Comparison Accuracy (0.35)
Measures whether the agent correctly identified and compared the top 3 frameworks based on the specified criteria.

5 — Identifies 3 frameworks and compares them accurately across all criteria (performance, community size, documentation).
4 — Identifies 3 frameworks but misses minor details in comparisons.
3 — Identifies 2 frameworks or provides incomplete comparisons.
2 — Identifies 1 framework or provides mostly incorrect comparisons.
1 — Fails to identify any frameworks or provide meaningful comparisons.

#### B. Platform Coverage (0.30)
Measures whether the agent visited all required platforms and used them effectively.

5 — Visits all 3 platforms (github.com, stackoverflow.com, developer.mozilla.org) and extracts relevant data.
4 — Visits 2 platforms and extracts relevant data.
3 — Visits 1 platform and extracts some relevant data.
2 — Visits 1 platform but extracts minimal or irrelevant data.
1 — Fails to visit any platform or extract relevant data.

#### C. Detail and Specificity (0.25)
Measures the depth and specificity of the comparisons, including quantitative metrics.

5 — Includes detailed performance benchmarks, community size metrics, and documentation evaluations for all frameworks.
4 — Includes most of the required details but lacks minor specifics.
3 — Includes some details but misses key metrics or evaluations.
2 — Includes minimal details or mostly incorrect information.
1 — Provides no meaningful details.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the output.

5 — Provides a well-organized summary table with sourced and credible data.
4 — Provides a summary table with minor organizational issues or unclear sourcing.
3 — Provides a summary table but lacks clarity or credible sourcing.
2 — Provides an unstructured or poorly organized output.
1 — Provides no structured output.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "framework_comparison_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "detail_and_specificity": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "framework_comparison_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "detail_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "framework_comparison_accuracy": 0.35,
    "platform_coverage": 0.30,
    "detail_and_specificity": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())