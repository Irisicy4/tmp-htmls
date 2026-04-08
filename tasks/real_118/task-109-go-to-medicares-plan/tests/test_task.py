"""
LLM-as-judge evaluator for EvolveBench task-109.

Category: Insurance & Actuarial
Task: Use Medicare Plan Finder to compare Part D drug plans in Miami FL for a beneficiary
      on atorvastatin, metformin, and lisinopril; verify drug tiers on the insurer's formulary.
"""

import os, json, re

TASK_INSTRUCTION = (
    "Go to Medicare's Plan Finder and search for Part D prescription drug plans available in "
    "ZIP code 33101 (Miami, FL) for a beneficiary who takes atorvastatin 40mg, metformin 500mg, "
    "and lisinopril 10mg. Identify the three plans with the lowest estimated annual drug cost. "
    "For each plan, record: plan name, insurer, monthly premium, annual deductible, and estimated "
    "annual drug cost. Then visit the formulary lookup page for the top-ranked plan on the insurer's "
    "website and confirm that all three drugs are covered and at what tier. Produce a Medicare Part D "
    "plan comparison table with all specified columns plus a 'total cost' row."
)
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a Medicare Part D plan comparison task using the Medicare Plan Finder and insurer formulary data.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Primary source: medicare.gov/plan-compare for Part D plan search
- ZIP code: 33101 (Miami, FL) specifically
- Medications: atorvastatin 40mg, metformin 500mg, lisinopril 10mg — all three must be entered
- Selection: Three plans with LOWEST estimated annual drug cost
- Required fields: plan name, insurer, monthly premium, annual deductible, estimated annual drug cost
- Formulary check: Top-ranked plan's formulary on the insurer's own website; drug tier for each medication
- Output: Comparison table with all six columns plus a total cost row (premiums × 12 + drug costs)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate Medicare Plan Finder and enter ZIP 33101 and the three medications?
- Are three specific Part D plans retrieved with insurer names and cost details?
- Did the agent access the top plan's formulary on the insurer's website?
- Are drug tier assignments for all three medications reported?
- Is there a comparison table with a total cost row?

### Step 2: Dimension Scoring

#### A. Medicare Plan Finder Navigation
Did the agent use medicare.gov/plan-compare with correct inputs?

5 — Navigated Medicare Plan Finder, entered ZIP 33101, and added all three medications (atorvastatin, metformin, lisinopril) before retrieving plan results.
4 — Navigated Plan Finder with correct ZIP but only two of three medications entered.
3 — Navigated Plan Finder with correct ZIP but medications not entered; plans retrieved without drug-cost personalization.
2 — Mentioned Medicare Plan Finder but plan data appears generic or not from actual navigation with the specified inputs.
1 — Medicare Plan Finder not used; plans are from prior knowledge or fabricated.

#### B. Plan Cost Data Completeness
Are monthly premium, annual deductible, and estimated annual drug cost present for all three plans?

5 — All three plans have plan name, insurer, monthly premium, annual deductible, and estimated annual drug cost.
4 — Two of three plans are fully detailed; one is missing one cost field.
3 — All three plans named but cost data incomplete for at least two plans.
2 — Only one plan has complete cost data.
1 — No specific plan cost data.

#### C. Formulary Verification
Did the agent check the top plan's formulary on the insurer's own website for drug tier assignments?

5 — Agent navigated to the top-ranked insurer's formulary website and confirmed tier (1-5 or equivalent) for all three drugs.
4 — Agent checked the formulary and confirmed tiers for two of three drugs.
3 — Agent checked the formulary but reported only covered/not covered (yes/no) without tier numbers.
2 — Agent mentioned checking formulary but no specific tier information retrieved.
1 — Formulary not checked on insurer website; tier information absent or fabricated.

#### D. Output Table & Total Cost Row
Is the comparison table complete with all required columns and a total cost row?

5 — Complete table with all six columns for all three plans plus a correctly calculated total cost row (monthly premium × 12 + estimated annual drug cost).
4 — Table with five of six columns; total cost row present but calculation not shown.
3 — Table present with four or fewer columns; or total cost row missing.
2 — Narrative comparison without structured table.
1 — No table.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "plan_finder_navigation": <1-5>,
  "plan_cost_data": <1-5>,
  "formulary_verification": <1-5>,
  "output_table": <1-5>,
  "dimension_reasoning": {{
    "plan_finder_navigation": "<one sentence citing specific evidence>",
    "plan_cost_data": "<one sentence citing specific evidence>",
    "formulary_verification": "<one sentence citing specific evidence>",
    "output_table": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "plan_finder_navigation": 0.30,
    "plan_cost_data":         0.25,
    "formulary_verification": 0.25,
    "output_table":           0.20,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())


def _extract_response(result):
    task_result = result.get("task_result") or ""
    if isinstance(task_result, str) and task_result.strip():
        return task_result
    for message in reversed(result.get("conversation") or []):
        if not isinstance(message, dict): continue
        if message.get("role") == "assistant":
            content = message.get("content") or ""
            if isinstance(content, str) and len(content) > 20:
                return content
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
    except Exception as e:
        return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All judge calls failed"}
    aggregated = {dim: sorted([v[dim] for v in valid])[len(valid) // 2] for dim in DIMENSIONS}
    overall = sum(aggregated[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    aggregated["overall_score"] = round(overall, 2)
    aggregated["passed"] = overall >= PASS_THRESHOLD
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
