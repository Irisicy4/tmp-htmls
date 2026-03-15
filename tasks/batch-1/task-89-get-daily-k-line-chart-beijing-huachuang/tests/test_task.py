"""
LLM-as-judge evaluator for EvolveBench task-89.

Category: Finance & Economics
Task: Get the daily K-line chart for Beijing North Huachuang from East Money (eastmoney.com).
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


TASK_INSTRUCTION = """Get the daily K-line chart for Beijing North Huachuang from East Money (eastmoney.com)."""

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. Your job is to assess whether an AI agent successfully completed a browser-based task.

The task involves navigating eastmoney.com (a major Chinese financial platform) to find the daily candlestick (K-line) chart for Beijing North Huachuang (北方华创, stock code 002371 or 688041).

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Platform: eastmoney.com specifically
- Stock: Beijing North Huachuang (北方华创)
- Chart type: daily K-line (candlestick) chart
- Output: chart retrieved and key data points reported (current price, recent highs/lows)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent navigate to eastmoney.com?
- Was Beijing North Huachuang found by name or stock code?
- Was the daily K-line chart accessed?
- What chart data was reported (price, date range, highs/lows)?
- Was a screenshot or description of the chart provided?

### Step 2: Dimension Scoring

#### A. Platform Navigation (0.25)
Did the agent navigate to eastmoney.com?

5 — Agent navigated to eastmoney.com and accessed the stock section.
4 — Agent reached eastmoney.com but had navigation difficulty.
3 — Agent found the stock via a different Chinese financial platform.
2 — Agent described eastmoney.com without navigating.
1 — No navigation.

#### B. Stock Identification (0.2)
Was the correct stock found?

5 — Beijing North Huachuang (北方华创) found with correct stock code and exchange.
4 — Stock found but code or exchange detail unclear.
3 — Similar company found but not the exact stock.
2 — Wrong stock accessed.
1 — Stock not found.

#### C. Chart Retrieval (0.35)
Was the daily K-line chart accessed?

5 — Daily K-line chart accessed; OHLC data for recent days reported.
4 — Chart accessed but only closing price reported.
3 — Stock page accessed but chart not specifically retrieved.
2 — General stock info without chart data.
1 — No chart retrieved.

#### D. Data Reporting (0.2)
Was chart data clearly reported?

5 — Recent OHLC prices, date range, 52-week high/low, and trend description.
4 — Price and trend reported; some data missing.
3 — Only current price reported.
2 — Vague description without specific numbers.
1 — No data reported.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "platform_navigation": <1-5>,
  "stock_identification": <1-5>,
  "chart_retrieval": <1-5>,
  "data_reporting": <1-5>,
  "dimension_reasoning": {{
    "platform_navigation": "<one sentence citing specific evidence>",
    "stock_identification": "<one sentence citing specific evidence>",
    "chart_retrieval": "<one sentence citing specific evidence>",
    "data_reporting": "<one sentence citing specific evidence>",
  }},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {
    "platform_navigation": 0.25,
    "stock_identification": 0.2,
    "chart_retrieval": 0.35,
    "data_reporting": 0.2,
}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())