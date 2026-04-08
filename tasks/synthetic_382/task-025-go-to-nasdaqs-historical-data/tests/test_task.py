"""
LLM-as-judge evaluator for EvolveBench task.

Category: Finance & Economics
Task: Extract and cross-verify Apple Inc. (AAPL) closing prices from Nasdaq and Yahoo Finance for the past 30 days, and calculate the percentage price change.
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


TASK_INSTRUCTION = """Go to Nasdaq's historical data section and extract the closing prices of Apple Inc. (AAPL) for the past 30 days. Then visit Yahoo Finance and cross-verify the same data. Finally, extract the percentage price change over this period and determine if the data from both sites aligns."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves extracting historical closing prices for Apple Inc. (AAPL) from Nasdaq and Yahoo Finance for the past 30 days, cross-verifying the data from both sources, and calculating the percentage price change over the period. A successful completion requires accurate data extraction, verification, and calculation.

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
Go to Nasdaq's historical data section and extract the closing prices of Apple Inc. (AAPL) for the past 30 days. Then visit Yahoo Finance and cross-verify the same data. Finally, extract the percentage price change over this period and determine if the data from both sites aligns.

## Task-Specific Constraints
- Must visit both nasdaq.com and finance.yahoo.com.
- Must extract closing prices for exactly the past 30 days.
- Must include a percentage price change calculation in the output.
- Must verify that data from both sources aligns (e.g., no discrepancies).
- Output must be structured as a table or JSON with dates, prices, and percentage change.

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to both nasdaq.com and finance.yahoo.com?
- Did the agent extract closing prices for exactly 30 days from both sources?
- Did the agent calculate the percentage price change over the period?
- Did the agent verify that the data from both sources aligns?
- Is the output structured as a table or JSON with the required fields?

### Step 2: Dimension Scoring

#### A. Data Accuracy (0.35)
Measures whether the extracted data is correct and matches the actual historical prices.

5 — All data is correct and matches both sources perfectly.
4 — Minor discrepancies in data but mostly correct.
3 — Partial data extracted or significant errors in some entries.
2 — Major errors or incomplete data.
1 — No relevant data extracted.

#### B. Source Coverage (0.30)
Measures whether the agent visited both required platforms and extracted data from them.

5 — Both platforms visited, and data extracted from both.
4 — Both platforms visited, but data partially extracted from one.
3 — Only one platform visited, with partial data extracted.
2 — Only one platform visited, with minimal or no data extracted.
1 — Neither platform visited.

#### C. Calculation Accuracy (0.20)
Measures the correctness of the percentage price change calculation.

5 — Calculation is accurate and based on extracted data.
4 — Calculation is mostly accurate but with minor errors.
3 — Calculation attempted but with significant errors.
2 — Calculation attempted but mostly incorrect.
1 — No calculation attempted.

#### D. Output Structure (0.15)
Measures whether the output is well-organized and follows the required format.

5 — Output is structured as a table or JSON with all required fields.
4 — Output is structured but missing minor details.
3 — Output is partially structured but missing key fields.
2 — Output is poorly structured or hard to interpret.
1 — Output is unstructured or absent.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "data_accuracy": <1-5>,
  "source_coverage": <1-5>,
  "calculation_accuracy": <1-5>,
  "output_structure": <1-5>,
  "dimension_reasoning": {{
    "data_accuracy": "<one sentence citing specific evidence>",
    "source_coverage": "<one sentence citing specific evidence>",
    "calculation_accuracy": "<one sentence citing specific evidence>",
    "output_structure": "<one sentence citing specific evidence>"
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "data_accuracy": 0.35,
    "source_coverage": 0.30,
    "calculation_accuracy": 0.20,
    "output_structure": 0.15,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())