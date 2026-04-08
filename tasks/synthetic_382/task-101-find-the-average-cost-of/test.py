"""
LLM-as-judge evaluator for EvolveBench task.

Category: Daily Activities
Task: Compare internet plan costs and speed-to-price ratios from three providers in New York City.
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


TASK_INSTRUCTION = """Find the average cost of monthly internet plans in New York City by comparing rates from three providers (e.g., Spectrum, Verizon, and Optimum). Calculate which provider offers the best value based on speed-to-price ratio for a 500 Mbps plan."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to compare internet plan costs and speed-to-price ratios from three providers in New York City: Spectrum, Verizon, and Optimum. A successful completion includes accurate pricing data, calculated averages, and a clear identification of the best value provider based on a 500 Mbps plan.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Find the average cost of monthly internet plans in New York City by comparing rates from three providers (e.g., Spectrum, Verizon, and Optimum). Calculate which provider offers the best value based on speed-to-price ratio for a 500 Mbps plan.

## Task-Specific Constraints
- Must visit at least 3 of the specified platforms (Spectrum, Verizon, Optimum).
- Must include price data for all providers compared.
- Must calculate the average monthly cost for each provider.
- Must calculate the speed-to-price ratio for a 500 Mbps plan for each provider.
- Output must clearly identify the provider offering the best value.
- Output must be organized as a table or structured list.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to the required platforms? Which ones were actually visited?
- Are price data for all three providers present in the response?
- Is the average monthly cost calculated for each provider?
- Is the speed-to-price ratio for a 500 Mbps plan calculated for each provider?
- Is the output organized as a table or structured list?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the agent accurately calculated averages and speed-to-price ratios, and identified the best value provider.

5 — All calculations are correct, and the best value provider is accurately identified.
4 — Minor errors in calculations or identification of the best value provider.
3 — Partial calculations are correct, but key elements (e.g., best value provider) are missing or incorrect.
2 — Significant errors in calculations or identification.
1 — No calculations attempted or completely incorrect.

#### B. Coverage of Platforms (0.30)
Measures whether the agent visited all required platforms and included data from each.

5 — Data from all three platforms (Spectrum, Verizon, Optimum) is present.
4 — Data from two platforms is present, with minor omissions.
3 — Data from at least one platform is present, but incomplete.
2 — Minimal or incorrect data from platforms.
1 — No platform data included.

#### C. Depth of Analysis (0.25)
Measures the level of detail in the response, including numerical comparisons and structured output.

5 — Detailed numerical comparisons and structured output (e.g., table or list) are present.
4 — Comparisons are present but lack some detail or structure.
3 — Basic comparisons are present but incomplete or unstructured.
2 — Minimal analysis or unstructured output.
1 — No analysis attempted.

#### D. Output Organization and Credibility (0.10)
Measures the clarity and credibility of the response, including proper sourcing and formatting.

5 — Output is well-organized, properly formatted, and sources are credible.
4 — Output is mostly clear and credible, with minor formatting issues.
3 — Output is usable but lacks clarity or proper sourcing.
2 — Output is disorganized or lacks credibility.
1 — Output is completely unclear or unusable.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "coverage_of_platforms": <1-5>,
  "depth_of_analysis": <1-5>,
  "output_organization_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_platforms": "<one sentence citing specific evidence>",
    "depth_of_analysis": "<one sentence citing specific evidence>",
    "output_organization_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "coverage_of_platforms": 0.30,
    "depth_of_analysis": 0.25,
    "output_organization_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())