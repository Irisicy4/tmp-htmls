"""
LLM-as-judge evaluator for EvolveBench task.

Category: Real Estate
Task: Create a mortgage payment calculator using Google Sheets with data fetched from Bankrate and SmartAsset.
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


TASK_INSTRUCTION = """Use Google Sheets (https://sheets.google.com) to create a mortgage payment calculator for a $500,000 home purchase in San Francisco, CA, with a $100,000 down payment. Fetch the current 30-year fixed mortgage rates from Bankrate (https://www.bankrate.com/) and property tax rates for California from SmartAsset (https://smartasset.com/). Include fields for: monthly loan payment (principle + interest), property taxes, homeowner insurance estimate, and total monthly payment. Share a summary table with calculations for three different interest rates."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task asks the agent to create a mortgage payment calculator for a $500,000 home purchase in San Francisco, CA, using Google Sheets. The agent must fetch current mortgage rates from Bankrate and California property tax rates from SmartAsset. A successful completion includes a structured table with calculations for three different interest rates and all required fields.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Use Google Sheets (https://sheets.google.com) to create a mortgage payment calculator for a $500,000 home purchase in San Francisco, CA, with a $100,000 down payment. Fetch the current 30-year fixed mortgage rates from Bankrate (https://www.bankrate.com/) and property tax rates for California from SmartAsset (https://smartasset.com/). Include fields for: monthly loan payment (principle + interest), property taxes, homeowner insurance estimate, and total monthly payment. Share a summary table with calculations for three different interest rates.

## Task-Specific Constraints
- Must visit Bankrate and SmartAsset to fetch required rates.
- Must create a structured table in Google Sheets with all required fields.
- Must include calculations for three different interest rates.
- Must provide accurate mortgage rate and property tax data.
- Must calculate total monthly payment including principle, interest, taxes, and insurance.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Bankrate and SmartAsset to fetch the required data?
- Is the structured table present in the response, and does it include all required fields?
- Are calculations for three different interest rates included?
- Is the property tax rate accurate and sourced from SmartAsset?
- Are the total monthly payment calculations correct and complete?

### Step 2: Dimension Scoring

#### A. Deliverable Accuracy (0.35)
Measures whether the mortgage payment calculator is correct and complete.

5 — All required fields are present, and calculations are accurate for all three interest rates.
4 — Minor errors in calculations or missing one required field.
3 — Partial completion, with significant errors or missing multiple fields.
2 — Mostly incomplete or incorrect.
1 — No meaningful attempt.

#### B. Platform Coverage (0.30)
Measures whether the agent visited the required platforms and fetched accurate data.

5 — Both Bankrate and SmartAsset were visited, and data was correctly sourced.
4 — Both platforms were visited, but minor inaccuracies in data.
3 — Only one platform was visited, or data accuracy issues.
2 — Neither platform was visited, or data is mostly incorrect.
1 — No attempt to fetch data from required platforms.

#### C. Depth of Calculations (0.25)
Measures the specificity and detail of the calculations provided.

5 — Detailed calculations for principle, interest, taxes, and insurance are present.
4 — Minor omissions or lack of detail in one area.
3 — Partial calculations with significant omissions.
2 — Mostly incomplete or missing calculations.
1 — No meaningful calculations provided.

#### D. Output Structure and Credibility (0.10)
Measures the organization and credibility of the output.

5 — Output is well-organized, formatted correctly, and uses credible sources.
4 — Minor formatting issues or unclear organization.
3 — Partially organized, with some credibility issues.
2 — Poorly organized or mostly unclear.
1 — No meaningful structure or credibility.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "deliverable_accuracy": <1-5>,
  "platform_coverage": <1-5>,
  "depth_of_calculations": <1-5>,
  "output_structure_and_credibility": <1-5>,
  "dimension_reasoning": {{
    "deliverable_accuracy": "<one sentence citing specific evidence>",
    "platform_coverage": "<one sentence citing specific evidence>",
    "depth_of_calculations": "<one sentence citing specific evidence>",
    "output_structure_and_credibility": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "deliverable_accuracy": 0.35,
    "platform_coverage": 0.30,
    "depth_of_calculations": 0.25,
    "output_structure_and_credibility": 0.10,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())