"""
LLM-as-judge evaluator for EvolveBench task.

Category: Finance & Economics
Task: Analyze a sample portfolio using Morningstar's instant X-Ray tool to produce a breakdown of sector composition, geographic allocation, and risk level.
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


TASK_INSTRUCTION = """Navigate to the Morningstar instant X-Ray tool and analyze a sample portfolio with the following asset allocation: 40% in SPDR S&P 500 ETF (SPY), 30% in Vanguard Total Bond Market ETF (BND), and 30% in iShares MSCI Emerging Markets ETF (EEM). Complete the workflow to produce a breakdown of the portfolio's sector composition, geographic allocation, and risk level."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to navigate to Morningstar's instant X-Ray tool and analyze a sample portfolio with specified asset allocations. The agent must produce a breakdown of the portfolio's sector composition, geographic allocation, and risk level. A successful completion includes accurate data extraction, proper use of the platform, and a well-structured output.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Navigate to the Morningstar instant X-Ray tool and analyze a sample portfolio with the following asset allocation: 40% in SPDR S&P 500 ETF (SPY), 30% in Vanguard Total Bond Market ETF (BND), and 30% in iShares MSCI Emerging Markets ETF (EEM). Complete the workflow to produce a breakdown of the portfolio's sector composition, geographic allocation, and risk level.

## Task-Specific Constraints
- Must navigate to Morningstar's instant X-Ray tool.
- Must input the specified portfolio allocations (SPY 40%, BND 30%, EEM 30%).
- Must produce a breakdown of sector composition, geographic allocation, and risk level.
- Output must be structured and clearly labeled.
- Data must be accurate and match the portfolio analysis.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Morningstar's instant X-Ray tool?
- Did the agent input the correct portfolio allocations (SPY 40%, BND 30%, EEM 30%)?
- Does the output include sector composition, geographic allocation, and risk level?
- Is the output structured and clearly labeled?
- Are the data points accurate and consistent with the portfolio analysis?

### Step 2: Dimension Scoring

#### A. Primary Deliverable Accuracy (0.35)
Measures whether the agent correctly produced the required portfolio breakdown.

5 — All three breakdowns (sector, geographic, risk) are accurate and complete.
4 — Two breakdowns are accurate and complete; one is partially complete or slightly inaccurate.
3 — At least one breakdown is accurate and complete; others are partially complete.
2 — Minimal accuracy; most breakdowns are missing or incorrect.
1 — No accurate breakdowns provided.

#### B. Coverage of Required Data (0.30)
Measures whether the agent included all required data points and used the correct platform.

5 — All required data points are included, and Morningstar was used correctly.
4 — Most required data points are included; minor omissions or platform usage issues.
3 — Some required data points are included; significant omissions or platform usage issues.
2 — Few required data points are included; major omissions or incorrect platform usage.
1 — No required data points included or incorrect platform used.

#### C. Depth and Specificity (0.20)
Measures whether the agent provided detailed and specific information in the breakdown.

5 — All breakdowns include detailed, specific data points (e.g., percentages, regions, sectors).
4 — Most breakdowns include detailed data; minor omissions or lack of specificity.
3 — Some breakdowns include detailed data; significant omissions or lack of specificity.
2 — Minimal detail or specificity in the breakdowns.
1 — No detail or specificity provided.

#### D. Output Structure and Clarity (0.15)
Measures whether the output is well-organized and easy to understand.

5 — Output is well-structured, clearly labeled, and easy to interpret.
4 — Output is mostly well-structured; minor labeling or clarity issues.
3 — Output is partially structured; significant labeling or clarity issues.
2 — Output is poorly structured and difficult to interpret.
1 — Output is unstructured and incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "primary_deliverable_accuracy": <1-5>,
  "coverage_of_required_data": <1-5>,
  "depth_and_specificity": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "primary_deliverable_accuracy": "<one sentence citing specific evidence>",
    "coverage_of_required_data": "<one sentence citing specific evidence>",
    "depth_and_specificity": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "primary_deliverable_accuracy": 0.35,
    "coverage_of_required_data": 0.30,
    "depth_and_specificity": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())