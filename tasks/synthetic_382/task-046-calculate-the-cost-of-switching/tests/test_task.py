"""
LLM-as-judge evaluator for EvolveBench task.

Category: Insurance & Actuarial
Task: Evaluate whether the agent successfully calculated the cost savings of switching auto insurance policies by gathering quotes from three specified platforms and performing the required calculations.
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


TASK_INSTRUCTION = """Calculate the cost of switching from a current auto insurance policy to a new one. Assume the driver is a 28-year-old male in Los Angeles, CA with a clean record, driving a 2021 Honda Accord. Start by visiting Geico (https://www.geico.com/auto-insurance/), Liberty Mutual (https://www.libertymutual.com/auto-insurance), and USAA (https://www.usaa.com/inet/wc/insurance_auto_main) to get quotes for 100/300 liability and $500 deductible comprehensive/collision coverage. Then, calculate the difference in the total annual cost (premium + deductible) between the lowest quoted policy and an existing policy with an annual premium of $1,200 and a $1,000 deductible. Recommend whether switching would save money and provide the calculated savings amount."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task requires the agent to calculate the cost savings of switching auto insurance policies by gathering quotes from three specified platforms (Geico, Liberty Mutual, USAA) and performing a cost comparison against an existing policy. The agent must provide a recommendation on whether switching saves money, including the calculated savings amount.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Calculate the cost of switching from a current auto insurance policy to a new one. Assume the driver is a 28-year-old male in Los Angeles, CA with a clean record, driving a 2021 Honda Accord. Start by visiting Geico (https://www.geico.com/auto-insurance/), Liberty Mutual (https://www.libertymutual.com/auto-insurance), and USAA (https://www.usaa.com/inet/wc/insurance_auto_main) to get quotes for 100/300 liability and $500 deductible comprehensive/collision coverage. Then, calculate the difference in the total annual cost (premium + deductible) between the lowest quoted policy and an existing policy with an annual premium of $1,200 and a $1,000 deductible. Recommend whether switching would save money and provide the calculated savings amount.

## Task-Specific Constraints
- Must visit Geico, Liberty Mutual, and USAA to gather quotes.
- Must include price data for all three platforms in the response.
- Must calculate the total annual cost (premium + deductible) for each quote.
- Must compare the lowest quote to the existing policy and calculate savings.
- Must explicitly recommend whether switching saves money and provide the savings amount.
- Output must be clear, structured, and include all required calculations.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to Geico, Liberty Mutual, and USAA? Which platforms were actually visited?
- Are the premium and deductible amounts for all three quotes present in the response?
- Did the agent correctly calculate the total annual cost for each quote?
- Did the agent compare the lowest quote to the existing policy and calculate the savings correctly?
- Is the recommendation on whether to switch clear and supported by the calculations?

### Step 2: Dimension Scoring

#### A. Quote Collection Accuracy (0.35)
Measures whether the agent successfully gathered quotes from all three specified platforms.

5 — Quotes from all three platforms (Geico, Liberty Mutual, USAA) are present and accurate.
4 — Quotes from two platforms are present and accurate.
3 — Quotes from one platform are present and accurate.
2 — Attempted but quotes are incomplete or inaccurate.
1 — No quotes are present or completely incorrect.

#### B. Cost Calculation Accuracy (0.30)
Measures whether the agent correctly calculated the total annual cost (premium + deductible) for each quote.

5 — Total annual costs for all three quotes are calculated correctly.
4 — Total annual costs for two quotes are calculated correctly.
3 — Total annual cost for one quote is calculated correctly.
2 — Attempted but calculations are mostly incorrect.
1 — No calculations attempted or completely incorrect.

#### C. Savings Comparison Accuracy (0.20)
Measures whether the agent correctly compared the lowest quote to the existing policy and calculated savings.

5 — Savings calculation is accurate and clearly explained.
4 — Savings calculation is mostly accurate but lacks clarity or minor errors.
3 — Savings calculation is partially correct but incomplete.
2 — Attempted but mostly incorrect.
1 — No comparison or calculation attempted.

#### D. Output Structure and Clarity (0.15)
Measures whether the agent's response is well-structured, clear, and includes all required elements.

5 — Response is well-structured, clear, and includes all required elements.
4 — Response is mostly clear but has minor structural issues.
3 — Response is partially clear but lacks some required elements.
2 — Response is poorly structured or unclear.
1 — Response is completely disorganized or incomprehensible.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "quote_collection_accuracy": <1-5>,
  "cost_calculation_accuracy": <1-5>,
  "savings_comparison_accuracy": <1-5>,
  "output_structure_and_clarity": <1-5>,
  "dimension_reasoning": {{
    "quote_collection_accuracy": "<one sentence citing specific evidence>",
    "cost_calculation_accuracy": "<one sentence citing specific evidence>",
    "savings_comparison_accuracy": "<one sentence citing specific evidence>",
    "output_structure_and_clarity": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "quote_collection_accuracy": 0.35,
    "cost_calculation_accuracy": 0.30,
    "savings_comparison_accuracy": 0.20,
    "output_structure_and_clarity": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())